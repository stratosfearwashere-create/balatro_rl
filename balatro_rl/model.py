"""Actor-critic network.

The policy scores every action with a shared network (a "pointer" style policy):
    logit(a) = f(state_embedding, action_embedding(a))
so the same weights handle playing cards, shopping and opening packs, and the net
generalises across the 466 actions instead of memorising slot positions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import F_CARD, F_JOKER, F_ACT, VOCAB_SIZE, N_SUB, A_PLAY, A_DISC, global_size

OBS_KEYS = ("g", "cards", "jid", "jf", "cid", "aid", "af", "mask", "subs")
OPTIONAL_KEYS = ("aid2",)          # passed when present (data recorded before the replace action lacks it)


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, g_dim: int | None = None, d: int = 256, emb: int = 32):
        super().__init__()
        g_dim = g_dim or global_size()
        self.config = {"g_dim": g_dim, "d": d, "emb": emb}
        self.emb = nn.Embedding(VOCAB_SIZE, emb)
        self.card = mlp(F_CARD, 64, 64)
        self.joker = mlp(emb + F_JOKER, 64, 64)
        self.state = mlp(g_dim + 64 * 4 + emb, d, d, n=3)
        self.act = mlp(emb + F_ACT, 128, 128)
        self.sub_proj = nn.Linear(64, 128)
        self.s_proj = nn.Linear(d, 128)
        self.head = mlp(128 * 3, 128, 1)
        self.value = mlp(d, 128, 1)
        # replace actions (env.A_REPLACE) also see the joker they give up (obs "aid2"). Starts at zero, so a
        # checkpoint from before the action existed scores every old action exactly as it did
        self.replace = nn.Linear(emb, 128)
        nn.init.zeros_(self.replace.weight)
        nn.init.zeros_(self.replace.bias)

    pack_legal = True      # score only the legal actions (same logits: the others are masked to -1e9 anyway)

    def _legal_logits(self, ob: dict, ce: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """The action logits computed for the legal (state, action) pairs only, scattered into [B, A]; every
        other entry is -1e9, as the dense path's mask leaves it. Same weights, same values."""
        mask = ob["mask"]
        b, a_idx = mask.nonzero(as_tuple=True)                          # L legal pairs
        rows = self.act(torch.cat([self.emb(ob["aid"][b, a_idx]), ob["af"][b, a_idx].float()], -1))   # L,128
        if "aid2" in ob:
            a2 = ob["aid2"][b, a_idx]
            sel = (a2 > 0).nonzero(as_tuple=True)[0]
            if sel.numel():
                rows = rows.index_add(0, sel, self.replace(self.emb(a2[sel])).to(rows.dtype))
        is_sub = a_idx < 2 * N_SUB                                      # play / discard slots: + their cards
        bs, as_ = b[is_sub], a_idx[is_sub]
        subs = ob["subs"][bs, as_]                                      # Ls,5 (-1 = padding)
        valid = (subs >= 0).to(ce.dtype).unsqueeze(-1)
        picked = ce[bs.unsqueeze(1), subs.clamp(min=0)] * valid         # Ls,5,64
        sub = self.sub_proj(picked.sum(1) / valid.sum(1).clamp(min=1))
        rows = rows.index_add(0, is_sub.nonzero(as_tuple=True)[0], sub.to(rows.dtype))
        sp = self.s_proj(s)[b]
        out = self.head(torch.cat([sp, rows, sp * rows], -1)).squeeze(-1)
        logits = torch.full(mask.shape, -1e9, dtype=out.dtype, device=out.device)
        logits[b, a_idx] = out
        return logits

    def forward(self, ob: dict):
        B = ob["g"].shape[0]
        cmask = ob["cards"][..., 0:1]                                  # present flag
        ce = self.card(ob["cards"]) * cmask                            # B,8,64
        cmean = ce.sum(1) / cmask.sum(1).clamp(min=1)
        cmax = (ce - (1 - cmask) * 1e4).max(1).values.clamp(min=-10)
        jmask = ob["jf"][..., 0:1]
        je = self.joker(torch.cat([self.emb(ob["jid"]), ob["jf"]], -1)) * jmask
        jmean = je.sum(1) / jmask.sum(1).clamp(min=1)
        jmax = (je - (1 - jmask) * 1e4).max(1).values.clamp(min=-10)
        cons = self.emb(ob["cid"]).mean(1)
        g = ob["g"][:, :self.config["g_dim"]]            # older networks: without the columns added since
        s = self.state(torch.cat([g, cmean, cmax, jmean, jmax, cons], -1))
        s = F.relu(s)

        if self.pack_legal:
            return self._legal_logits(ob, ce, s), self.value(s).squeeze(-1)
        a = self.act(torch.cat([self.emb(ob["aid"]), ob["af"].float()], -1))   # B,A,128
        if "aid2" in ob:
            a2 = ob["aid2"]
            a = a + self.replace(self.emb(a2)) * (a2 > 0).unsqueeze(-1)
        # each play/discard slot also sees the mean embedding of the cards it selects
        subs = ob["subs"]                                              # B,436,5 (-1 = padding)
        valid = (subs >= 0).float().unsqueeze(-1)
        idx = subs.clamp(min=0).reshape(B, -1, 1).expand(-1, -1, ce.shape[-1])
        picked = torch.gather(ce, 1, idx).reshape(B, subs.shape[1], subs.shape[2], -1) * valid
        sub = self.sub_proj(picked.sum(2) / valid.sum(2).clamp(min=1))   # B,436,128
        a = torch.cat([a[:, :2 * N_SUB] + sub, a[:, 2 * N_SUB:]], 1)
        sp = self.s_proj(s).unsqueeze(1).expand(-1, a.shape[1], -1)
        logits = self.head(torch.cat([sp, a, sp * a], -1)).squeeze(-1)
        logits = logits.masked_fill(~ob["mask"], -1e9)
        return logits, self.value(s).squeeze(-1)


def batch_obs(obs_list: list[dict], device="cpu") -> dict:
    out = {}
    for k in OBS_KEYS + tuple(x for x in OPTIONAL_KEYS if x in obs_list[0]):
        arr = np.stack([o[k] for o in obs_list])
        t = torch.from_numpy(arr)
        if k == "af":
            t = t.float()
        out[k] = t.to(device)
    return out


def stack_obs(obs_list: list[dict], device="cpu") -> dict:
    """Like batch_obs, but keeps "af" in its stored dtype (float16 after train.compact); forward()
    converts it. For stacking a whole rollout once and slicing minibatches on the GPU."""
    keys = OBS_KEYS + tuple(x for x in OPTIONAL_KEYS if x in obs_list[0])
    return {k: torch.from_numpy(np.stack([o[k] for o in obs_list])).to(device) for k in keys}


def save_model(model: ActorCritic, path: str, extra: dict | None = None):
    torch.save({"config": model.config, "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_model(path: str, device="cpu") -> ActorCritic:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ActorCritic(**ck["config"]).to(device)
    missing = m.load_state_dict(ck["state_dict"], strict=False)
    bad = [k for k in missing.missing_keys if not k.startswith("replace.")] + list(missing.unexpected_keys)
    if bad:                                        # only the zero-initialised replace path may be absent
        raise RuntimeError(f"{path}: checkpoint does not match the model: {bad}")
    m.eval()
    return m
