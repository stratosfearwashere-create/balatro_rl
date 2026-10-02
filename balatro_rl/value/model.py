"""The build-value network and its evaluator.

Input: one build record (records.py). Tokens: global, deck summary, 8 jokers (item embedding + row), consumables,
12 hand levels; a transformer over them; the pooled state (plus Phi) feeds the heads:
    ordinal   P(reach ante k) for k = 2..9, monotone in k (k = 9: the run is won); the loss only counts the
              antes beyond the record's own
    boss      P(this ante's boss is beaten), P(the next ante's boss is beaten)
    blinds    blinds beaten after this state / 24
Value for the agent (BuildValue.value): P(win) = P(reach 9), or the expected antes reached while wins are
too rare to learn from (`mode`).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..az.features import F_GLOBAL, F_JOK, F_CONS, F_LEV, N_CONS, VOCAB_SIZE
from ..sim.game import MAX_JOKERS
from ..sim.hands import N_HANDS
from .records import F_DECK, N_ORDINAL, build_record

N_TOKENS = 1 + 1 + MAX_JOKERS + N_CONS + N_HANDS
INPUT_KEYS = ("glob", "deck", "jok", "jok_id", "cons", "cons_id", "lev", "phi")


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class BuildValueNet(nn.Module):
    def __init__(self, d: int = 128, layers: int = 3, heads: int = 4, ff: int = 256, emb: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.config = {"d": d, "layers": layers, "heads": heads, "ff": ff, "emb": emb, "dropout": dropout}
        self.item = nn.Embedding(VOCAB_SIZE, emb)
        self.p_glob = nn.Linear(F_GLOBAL, d)
        self.p_deck = nn.Linear(F_DECK, d)
        self.p_jok = nn.Linear(F_JOK + emb, d)
        self.p_cons = nn.Linear(F_CONS + emb, d)
        self.p_lev = nn.Linear(F_LEV, d)
        self.group = nn.Embedding(5, d)
        gid = [0, 1] + [2] * MAX_JOKERS + [3] * N_CONS + [4] * N_HANDS
        self.register_buffer("gid", torch.tensor(gid, dtype=torch.long), persistent=False)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.state_out = mlp(2 * d + 1, d, d)
        self.ord_score = nn.Linear(d, 1)                  # ordinal: logit_k = score - threshold_k
        self.ord_steps = nn.Linear(d, N_ORDINAL)          # thresholds grow by softplus(step_k)
        self.aux = nn.Linear(d, 3)                        # boss_now, boss_next (logits), blinds_after

    def forward(self, x: dict) -> dict:
        it = self.item
        toks = [
            self.p_glob(x["glob"].float()).unsqueeze(1),
            self.p_deck(x["deck"].float()).unsqueeze(1),
            self.p_jok(torch.cat([x["jok"].float(), it(x["jok_id"].long())], -1)),
            self.p_cons(torch.cat([x["cons"].float(), it(x["cons_id"].long())], -1)),
            self.p_lev(x["lev"].float()),
        ]
        h = torch.cat(toks, 1) + self.group(self.gid).unsqueeze(0)
        mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        mask[:, 2:2 + MAX_JOKERS] = x["jok"][..., 0] > 0.5
        mask[:, 2 + MAX_JOKERS:2 + MAX_JOKERS + N_CONS] = x["cons"][..., 0] > 0.5
        h = self.norm(self.encoder(h, src_key_padding_mask=~mask))
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        phi = x["phi"].float().reshape(-1, 1)
        s = self.state_out(torch.cat([h[:, 0], pooled, phi], -1))
        score = self.ord_score(s)
        thresholds = torch.cumsum(F.softplus(self.ord_steps(s)), -1)     # increasing in k
        ord_logits = score - thresholds                                  # P(reach k) falls with k
        aux = self.aux(s)
        return {"ord": ord_logits, "boss_now": aux[:, 0], "boss_next": aux[:, 1], "blinds": aux[:, 2]}

    @staticmethod
    def reach(ord_logits: torch.Tensor) -> torch.Tensor:
        """P(reach ante k) for k = 2..9."""
        return torch.sigmoid(ord_logits)

    @staticmethod
    def expected_antes(ord_logits: torch.Tensor) -> torch.Tensor:
        """E[antes reached] (1 + sum_k P(reach k)); 9 means won."""
        return 1.0 + torch.sigmoid(ord_logits).sum(-1)


def ordinal_targets(ante: torch.Tensor, reached: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(targets [B, 8], mask [B, 8]) for k = 2..9: target 1 if reached >= k; only antes beyond the record's
    own ante count."""
    k = torch.arange(2, 2 + N_ORDINAL, device=ante.device).unsqueeze(0)
    return (reached.unsqueeze(1) >= k).float(), (k > ante.unsqueeze(1)).float()


def to_tensors(batch: dict, device) -> dict:
    out = {}
    for k in INPUT_KEYS:
        v = batch[k]
        t = torch.as_tensor(np.asarray(v))
        out[k] = t.to(device)
    return out


def save_model(net: BuildValueNet, path: str, extra: dict | None = None):
    import os
    torch.save({"config": net.config, "state_dict": net.state_dict(), "extra": extra or {}}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_model(path: str, device="cpu") -> BuildValueNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = BuildValueNet(**ck["config"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


class BuildValue:
    """Values of game states for the agent: value(worlds) -> np.ndarray in [0, 1].
    mode "win": P(win); "antes": E[antes reached] / 9 (denser while wins are rare); "mix": the mean of both."""

    def __init__(self, path: str, potential, device: str = "cpu", mode: str = "win"):
        self.net = load_model(path, device)
        self.potential = potential
        self.device = device
        self.mode = mode

    def records(self, worlds) -> list[dict]:
        return [build_record(w, self.potential) for w in worlds]

    def value(self, worlds) -> np.ndarray:
        if not worlds:
            return np.zeros(0)
        recs = self.records(worlds)
        batch = {k: np.stack([r[k] for r in recs]) if k != "phi" else np.asarray([r[k] for r in recs], np.float32)
                 for k in INPUT_KEYS}
        with torch.no_grad():
            out = self.net(to_tensors(batch, self.device))
            p = torch.sigmoid(out["ord"])
            win = p[:, -1]
            antes = (1.0 + p.sum(-1)) / 9.0
            v = win if self.mode == "win" else (antes if self.mode == "antes" else 0.5 * (win + antes))
        return v.float().cpu().numpy()
