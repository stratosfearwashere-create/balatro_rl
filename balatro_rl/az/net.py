"""C: the network. A transformer over the state's tokens (features.py), a policy head that scores each
candidate action, a value head and auxiliary heads.

Policy, residual form:  logit(a) = prior(a) + adjustment(a)
    prior(a) is the solver / rule prior (agent.py); the adjustment's last layer starts at exactly zero,
    so an untrained network plays like its prior, and training only has to learn where to differ.
A candidate's embedding combines its features (kind, exact score, P(clear), expected chips, immediate
effects, side effects), the mean of the tokens it refers to (its cards, joker, consumable, shop or pack
slot) and, for each joker, that joker's token weighted by how much this action changes its runtime state,
so "this play resets Ride the Bus" is visible to the head.

Value (the one the search backs up), residual on the potential (rewards/potential.py):
    V(s) = Phi(s) + R(s)       trained by MSE towards z (rewards/targets.py)
    V(s) = R(s)                with value_residual=False (ablation)
Phi is computed outside the network and passed in as the state's "phi"; with R starting near 0 the
untrained network already has Phi as its estimate, while the target stays z.
Auxiliary heads (predicted, never rewarded): P(clear the current / next blind); ante reached, 8 classes
(1..7, 8+); log(final chips / required) of the current blind; Phi_headroom at the next blind's start.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sim.game import MAX_JOKERS
from ..rewards.targets import N_ANTE_CLASSES
from .features import (F_GLOBAL, F_HANDCARD, F_JOK, F_CONS, F_LEV, F_ITEM, F_CAND, N_TOKENS, OFFSET, VOCAB_SIZE,
                       N_REF, GROUPS)
from .world import KINDS

STATE_KEYS = ("glob", "hand", "deck", "phand", "jok", "jok_id", "cons", "cons_id", "lev", "shop", "shop_id",
              "pack", "pack_id", "mask", "phi")
N_HEADS = 4 + N_ANTE_CLASSES          # R, clear logit, log score ratio, next headroom, ante logits
CAND_KEYS = ("c_kind", "c_f", "c_ref", "c_jd", "c_prior")


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class AZNet(nn.Module):
    def __init__(self, d: int = 128, layers: int = 3, heads: int = 4, ff: int = 256, emb: int = 32,
                 value_residual: bool = True):
        super().__init__()
        self.config = {"d": d, "layers": layers, "heads": heads, "ff": ff, "emb": emb,
                       "value_residual": value_residual}
        self.value_residual = value_residual
        self.item = nn.Embedding(VOCAB_SIZE, emb)
        self.p_glob = nn.Linear(F_GLOBAL, d)
        self.p_card = nn.Linear(F_HANDCARD, d)
        self.p_jok = nn.Linear(F_JOK + emb, d)
        self.p_cons = nn.Linear(F_CONS + emb, d)
        self.p_lev = nn.Linear(F_LEV, d)
        self.p_item = nn.Linear(F_ITEM + emb, d)
        self.group = nn.Embedding(9, d)                       # which token group (hand, deck, pack hand ...)
        gid = []
        for k, (name, n) in enumerate(GROUPS):
            gid += [k] * n
        assert len(gid) == N_TOKENS
        self.register_buffer("gid", torch.tensor(gid, dtype=torch.long), persistent=False)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        # candidates
        self.kind = nn.Embedding(len(KINDS), d)
        self.c_in = mlp(F_CAND, d, d)
        self.c_ref = nn.Linear(d, d)
        self.c_jok = nn.Linear(d, d)
        self.c_mix = mlp(3 * d, d, d)
        self.adjust = nn.Linear(d, 1)
        nn.init.zeros_(self.adjust.weight)
        nn.init.zeros_(self.adjust.bias)
        # value and auxiliary heads
        self.state_out = mlp(2 * d, d, d)
        self.heads = nn.Linear(d, N_HEADS)

    def encode(self, s: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = s["glob"].shape[0]
        it = self.item
        toks = [
            self.p_glob(s["glob"].float()).unsqueeze(1),
            self.p_card(s["hand"].float()),
            self.p_card(s["deck"].float()),
            self.p_card(s["phand"].float()),
            self.p_jok(torch.cat([s["jok"].float(), it(s["jok_id"])], -1)),
            self.p_cons(torch.cat([s["cons"].float(), it(s["cons_id"])], -1)),
            self.p_lev(s["lev"].float()),
            self.p_item(torch.cat([s["shop"].float(), it(s["shop_id"])], -1)),
            self.p_item(torch.cat([s["pack"].float(), it(s["pack_id"])], -1)),
        ]
        x = torch.cat(toks, 1) + self.group(self.gid).unsqueeze(0)
        mask = s["mask"].bool()
        h = self.norm(self.encoder(x, src_key_padding_mask=~mask))
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        state = self.state_out(torch.cat([h[:, 0], pooled], -1))
        return h, state, mask

    def forward(self, s: dict, c: dict, cmask: torch.Tensor):
        """s: state tensors [B, ...]; c: candidate tensors [B, A, ...]; cmask [B, A] (real candidates).
        Returns (logits [B, A] with padding at -1e9, adjustments [B, A], head outputs [B, N_HEADS])."""
        h, state, _ = self.encode(s)
        B, A = cmask.shape
        e = self.kind(c["c_kind"]) + self.c_in(c["c_f"].float())
        refs = c["c_ref"]
        valid = (refs >= 0).float().unsqueeze(-1)
        idx = refs.clamp(min=0).reshape(B, -1, 1).expand(-1, -1, h.shape[-1])
        picked = torch.gather(h, 1, idx).reshape(B, A, refs.shape[-1], -1) * valid
        e = e + self.c_ref(picked.sum(2) / valid.sum(2).clamp(min=1))
        o = OFFSET["jok"]
        hj = self.c_jok(h[:, o:o + MAX_JOKERS])                        # B, 8, d
        e = e + torch.einsum("baj,bjd->bad", c["c_jd"].float(), hj)
        sx = state.unsqueeze(1).expand(-1, A, -1)
        z = self.c_mix(torch.cat([e, sx, e * sx], -1))
        adj = self.adjust(F.gelu(z)).squeeze(-1)
        logits = (c["c_prior"].float() + adj).masked_fill(~cmask, -1e9)
        return logits, adj, self.heads(state)

    def value(self, out: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return out[..., 0] + (phi.float() if self.value_residual else 0.0)

    def split_heads(self, out: torch.Tensor, phi: torch.Tensor) -> dict:
        probs = torch.softmax(out[..., 4:], -1)
        classes = torch.arange(1, N_ANTE_CLASSES + 1, dtype=probs.dtype, device=probs.device)
        return {"value": self.value(out, phi), "clear": torch.sigmoid(out[..., 1]), "ratio": out[..., 2],
                "next_head": out[..., 3], "ante": (probs * classes).sum(-1), "ante_probs": probs}


# ------------------------------------------------------------------ batching
def collate(samples: list[tuple[dict, dict]], device="cpu"):
    """[(state, cands)] -> (state tensors, candidate tensors padded to the largest A, candidate mask)."""
    s = {k: torch.from_numpy(np.stack([x[0].get(k, np.float32(0.0)) for x in samples])).to(device)
         for k in STATE_KEYS}
    A = max(len(x[1]["c_kind"]) for x in samples)
    B = len(samples)
    c = {}
    for k in CAND_KEYS:
        first = samples[0][1][k]
        shape = (B, A) + first.shape[1:]
        fill = -1 if k == "c_ref" else 0
        arr = np.full(shape, fill, dtype=first.dtype if first.dtype != np.float16 else np.float32)
        for i, x in enumerate(samples):
            v = x[1][k]
            arr[i, :len(v)] = v
        c[k] = torch.from_numpy(arr).to(device)
    cmask = torch.zeros(B, A, dtype=torch.bool)
    for i, x in enumerate(samples):
        cmask[i, :len(x[1]["c_kind"])] = True
    return s, c, cmask.to(device)


def save_net(net: AZNet, path: str, extra: dict | None = None, optimizer=None):
    """Atomic: written to a temporary file, then renamed over `path`, so a run killed mid-save never leaves
    a broken checkpoint. With `optimizer`, its state is saved too (for resuming training)."""
    import os
    ck = {"config": net.config, "state_dict": net.state_dict(), "extra": extra or {}}
    if optimizer is not None:
        ck["optimizer"] = optimizer.state_dict()
    tmp = path + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, path)


def load_net(path: str, device="cpu") -> AZNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = AZNet(**ck["config"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net
