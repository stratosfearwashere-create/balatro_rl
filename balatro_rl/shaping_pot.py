"""The trainer's per-step shaping potentials as one function of an observation, for search that compares moves by reward + V(next).

The critic is trained on shaped rewards: a potential phi adds gamma * phi(next) - phi(now) to each step, so the critic learns V_true - phi. Comparing moves by
r + V(next) leaves phi(next) out, and a move that lowers phi (spending the bank, going into debt, buying a rental, giving up a multiplier joker) then looks
better by exactly that much (the search's v51 runs went into debt and bought rentals). Score moves by r + phi(next) + V(next) instead, with phi from here.
Potentials remeasured only at blind selects (--proj8) are the same for every move of one decision and are left out.

    pot = ShapingPot(dev, build_roles="xmult,scaler", build_second=0.5, bank=0.5, build=0.3, rental=0.5, debt=0.05)
    phi = pot(obs)            # [B] on the device
"""
from __future__ import annotations

import torch

from .env import item_id
from .runeval import Cols


class ShapingPot:
    def __init__(self, dev, build_roles="xmult,mult,chips", build_second=0.0, bank=0.0, build=0.0, rental=0.0, debt=0.0):
        from .heuristic import CHIPS, XMULT
        from .az.guide import MULT, SCALERS, SLOW_SCALERS
        self.role = torch.zeros((1024, 4), device=dev)            # multiplier, +mult, chips, scaler (as train.py's build potential)
        for col, keys in enumerate((XMULT, set(MULT) - set(XMULT), CHIPS, (set(SCALERS) | set(SLOW_SCALERS)) - set(XMULT))):
            for key in keys:
                if item_id(f"j_{key}") != item_id("j_unknown"):
                    self.role[item_id(f"j_{key}"), col] = 1.0
        br = set(build_roles.split(","))
        self.role_w = torch.tensor([1.0 if "xmult" in br else 0.0, 0.5 if "mult" in br else 0.0, (1.0 / 3.0) if "chips" in br else 0.0,
                                    1.0 if "scaler" in br else 0.0, build_second], device=dev)
        self.bank, self.build, self.rental, self.debt = float(bank), float(build), float(rental), float(debt)

    def __call__(self, o, bank=None, build=None):
        g = o["g"].float()
        out = torch.zeros(g.shape[0], device=g.device)
        bk = self.bank if bank is None else bank
        bd = self.build if build is None else build
        if bk:                                                     # train.py bank(): money up to where interest stops, in units of $25
            cap = 25.0 + 25.0 * g[:, 162] + 50.0 * g[:, 177]
            out = out + bk * torch.minimum((g[:, 17] * 50.0).clamp(min=0.0), cap) / 25.0
        if bd:                                                     # train.py roles() @ role_w
            live = ((o["jid"] > 0) & (o["jf"][:, :, 9].float() < 0.5)).float().unsqueeze(-1)
            r = self.role.to(g.device)[o["jid"].long()] * live
            r = torch.cat([r.amax(1), (r[:, :, 0].sum(1, keepdim=True) >= 2).float()], 1)
            out = out + bd * (r @ self.role_w.to(g.device))
        if self.rental or self.debt:
            rent = ((o["jf"][..., 8].float() > 0.5) & (o["jid"] > 0)).sum(1).float()
            dbt = (-g[:, Cols.MONEY] * 50.0).clamp(min=0.0)
            out = out - self.rental * rent - self.debt * dbt
        return out
