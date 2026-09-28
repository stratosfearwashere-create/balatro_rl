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
        s = self.state(torch.cat([ob["g"], cmean, cmax, jmean, jmax, cons], -1))
        s = F.relu(s)

        a = self.act(torch.cat([self.emb(ob["aid"]), ob["af"].float()], -1))   # B,A,128
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
    for k in OBS_KEYS:
        arr = np.stack([o[k] for o in obs_list])
        t = torch.from_numpy(arr)
        if k == "af":
            t = t.float()
        out[k] = t.to(device)
    return out


def stack_obs(obs_list: list[dict], device="cpu") -> dict:
    """Like batch_obs, but keeps "af" in its stored dtype (float16 after train.compact); forward()
    converts it. For stacking a whole rollout once and slicing minibatches on the GPU."""
    return {k: torch.from_numpy(np.stack([o[k] for o in obs_list])).to(device) for k in OBS_KEYS}


def save_model(model: ActorCritic, path: str, extra: dict | None = None):
    torch.save({"config": model.config, "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_model(path: str, device="cpu") -> ActorCritic:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ActorCritic(**ck["config"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m
