"""A hand-built value function for a run: how many more blinds it will beat (plus a bonus for winning), from what the engine can compute about the build.

    V(s) = sum over the blinds b still to come of P(the run survives to b and beats it) + win_bonus * P(it beats all of them)
    P(beat b) = sigmoid(A + B * ln share_b)            share_b = the build's projected strength at b / b's target; A, B fitted on 6144 solver-played builds
    share_b   = static_b * G ** (antes from now to b)  static_b = the build as it is, its scalers grown to b (VecEnv.power_curve: the boss of each ante; the small and
                                                        big blind from the boss share x the target ratio)
    ln G      = c0 + c1 ln(1 + money) + c2 * open joker slots + c3 * ante       the strength the shops add per ante (tools/hv_fit.py fits it on the policy's games)

Every term is either the engine's exact calculation or a handful of fitted numbers: no network, no policy noise. A purchase that lifts the build at antes 6-8 (a multiplier
that fires, a scaler that grows) lifts V there; a flat joker barely moves it.

    hv = HV.load("balatro_rl/data/hv_params.json"); values = hv(venv, ids)
"""
from __future__ import annotations

import json
import os

import numpy as np

from .runeval import Cols

DEFAULT = {"A": 1.83, "B": 2.53, "c": [0.0, 0.0, 0.0, 0.0], "win_bonus": 10.0, "samples": 8, "crn": 1, "crn_seed": 12345}
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hv_params.json")


def interp_shares(shares, ante):
    """shares [m, 8] (0 where not measured) -> filled from the current ante on by log-interpolation between the measured antes."""
    out = shares.copy()
    for k in range(len(out)):
        a0 = int(ante[k]) - 1
        known = [j for j in range(a0, 8) if out[k, j] > 0]
        if len(known) >= 2:
            out[k, a0:] = np.exp(np.interp(np.arange(a0, 8), known, np.log(out[k, known])))
        elif len(known) == 1:
            out[k, a0:] = out[k, known[0]]
    return out


class HV:
    def __init__(self, params=None):
        self.p = dict(DEFAULT, **(params or {}))

    @classmethod
    def load(cls, path=PATH):
        return cls(json.load(open(path))) if os.path.exists(path) else cls()

    def features(self, venv, ids):
        """-> (shares [m, 8] filled, ante [m], blind [m], money [m], open_slots [m]) for the games ids (outside a round)."""
        ids = [int(i) for i in ids]
        g = np.asarray(venv.obs["g"])[ids]
        ante = np.clip(np.rint(g[:, Cols.ANTE] * 8), 1, 8).astype(int)
        blind = np.argmax(g[:, Cols.BLIND:Cols.BLIND + 3], 1)
        if self.p.get("crn", 1):                       # common random numbers: the same redrawn hands and a generic boss for every state measured
            venv.v.set_curve_crn(int(self.p.get("crn_seed", 12345)), True)
        raw = np.asarray(venv.v.power_curve(ids, int(self.p["samples"])), dtype=np.float64).reshape(-1, 8)
        if self.p.get("crn", 1):
            venv.v.set_curve_crn(0, False)
        shares = interp_shares(raw, ante)
        inf = venv.v.cell_info(ids)
        money = np.array([x[3] for x in inf], dtype=np.float64)
        slots = np.array([max(0, x[5] - x[7]) for x in inf], dtype=np.float64)
        return shares, ante, blind, money, slots

    def value(self, shares, ante, blind, money, slots, done=None):
        A, B, c, W = self.p["A"], self.p["B"], self.p["c"], self.p["win_bonus"]
        lnG = c[0] + c[1] * np.log1p(np.maximum(money, 0)) + c[2] * slots + c[3] * ante
        out = np.zeros(len(ante))
        for k in range(len(ante)):
            if done is not None and done[k]:
                continue
            alive, tot = 1.0, 0.0
            b0 = 3 * (int(ante[k]) - 1) + int(blind[k])
            for b in range(b0, 24):
                a_b = b // 3
                sh = shares[k, a_b] * (2.0 if b % 3 == 0 else (2.0 / 1.5 if b % 3 == 1 else 1.0)) * np.exp(lnG[k] * max(0, a_b - (int(ante[k]) - 1)))
                if sh <= 0:
                    alive = 0.0
                    break
                alive *= 1.0 / (1.0 + np.exp(-(A + B * np.log(sh))))
                tot += alive
            out[k] = tot + W * alive
        return out

    def __call__(self, venv, ids, done=None):
        sh, an, bl, mo, sl = self.features(venv, ids)
        return self.value(sh, an, bl, mo, sl, done)
