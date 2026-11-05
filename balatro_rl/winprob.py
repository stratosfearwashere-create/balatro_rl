"""An analytic win probability for a run state: the build's round score treated as a lognormal quantity that grows over the rounds left,
against the known blind targets.

    log10 S(t)  =  log10(round score now)  +  t x (g_jokers + g_shop)
    round score now  = hands x the expected best play (balatro_rl/runeval.py) / pace
    g_jokers    the held scaling jokers' growth: each joker's counter projected to the ante-8 boss by balatro_rl/growth.py, the expected best play
                recomputed with those counters, the log10 gain spread evenly over the rounds left
    g_shop      what shopping adds a round: c0 + c1 x log10(1 + income a round) + c2 x ante (income from balatro_rl/economy.py)
    P(clear blind at round t) = Phi((log10 S(t) - log10 target(t)) / sigma)
    P(win) = the product over the blinds left;  E[blinds] = blinds beaten so far + the sum of the running products

pace, sigma and c0..c2 are fitted (fit()) to the outcomes of the policy's own rollouts from many states (tools/winprob_fit.py), so the number is
"how this policy does from here", in microseconds instead of rollouts.

    from balatro_rl.winprob import WinProb
    wp = WinProb()                     # balatro_rl/data/winprob.json
    out = wp(g, jid, jf, cid)          # out["p_win"], out["e_blinds"], out["p_clear"] [n, blinds left]
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .runeval import RunEval, State, default_weights, Cols, I_BLIND_SELECT, JF_COUNTER
from .economy import future_economy
from .growth import project
from .sim.items import BLIND_BASE, ALL_BOSSES, BOSS_KEYS, VOCAB

PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "winprob.json")
DEFAULT = dict(pace=1.5, sigma=0.35, c0=0.02, c1=0.06, c2=0.0)
NO_COUNTER = {"supernova", "abstract", "bull", "bootstraps", "swashbuckler", "throwback", "steel_joker", "stone", "erosion", "hiker"}


def _phi(x):
    from math import erf
    return 0.5 * (1.0 + np.vectorize(erf)(x / math.sqrt(2.0)))


class Features:
    """The parameter-free part, computed once per state: log10 round-score now (before pace), the jokers' log10 growth a round, log10 income,
    the targets of the blinds left, the blinds beaten so far."""

    def __init__(self, g, jid, jf, cid=None, ev=None):
        ev = ev or RunEval(default_weights())
        g = np.asarray(g, dtype=np.float32)
        jid = np.asarray(jid)
        jf = np.asarray(jf, dtype=np.float32)
        s = State(g, jid, jf, cid)
        self.n = s.n
        T, _s, dg = ev.terms(g, jid, jf, cid)
        e_now = np.maximum(dg["e_best"], 1.0)
        self.log_round = np.log10(s.hands * e_now)
        # the jokers' growth: counters at the ante-8 boss -> the expected best play again
        rows = project(g, jid, jf, cid)
        jf2 = jf.copy()
        for i, rs in enumerate(rows):
            used = set()
            for r in rs:
                if r["key"] in NO_COUNTER or r["unit"] not in ("chips", "mult", "xmult"):
                    continue
                for j in range(jid.shape[1]):
                    if j not in used and jid[i, j] > 0 and VOCAB[int(jid[i, j])] == "j_" + r["key"]:
                        used.add(j)
                        jf2[i, j, JF_COUNTER] = math.log1p(abs(max(r["future"], 0.0))) / 4.0
                        break
        _T, _s2, dg2 = ev.terms(g, jid, jf2, cid)
        e_fut = np.maximum(dg2["e_best"], 1.0)
        R = np.maximum(s.rounds_left, 1)
        self.g_jokers = np.clip(np.log10(e_fut / e_now), -2.0, 6.0) / R
        econ = future_economy(g, jid, jf, cid, rounds=6)
        inc = np.where(econ["alive"].any(1), np.nanmean(np.where(econ["alive"], econ["income"], np.nan), axis=1), 3.0)
        self.log_income = np.log10(1.0 + np.maximum(np.nan_to_num(inc, nan=3.0), 0.0))
        self.ante = s.ante.astype(np.float64)
        # the blinds left, in order, with their targets (the known boss of this ante; later bosses x2)
        scaling = np.where(s.stake >= 5, 3, np.where(s.stake >= 2, 2, 1))
        nxt = np.where(s.state == I_BLIND_SELECT, s.blind, s.blind + 1)
        self.done_blinds = (s.ante - 1) * 3 + np.where(s.state == I_BLIND_SELECT, s.blind, s.blind + 1)
        K = 24
        self.targets = np.full((self.n, K), np.nan)
        for i in range(self.n):
            a, b = int(s.ante[i]), int(nxt[i])
            if b > 2:
                a, b = a + 1, 0
            k = 0
            while a <= 8 and k < K:
                base = BLIND_BASE[int(scaling[i])][a - 1]
                mult = (1.0, 1.5)[b] if b < 2 else (ALL_BOSSES[BOSS_KEYS[int(s.boss_idx[i])]][2] if (a == s.ante[i] and s.boss_known[i]) else 2.0)
                self.targets[i, k] = base * mult * (1.25 if s.stake[i] >= 8 else 1.0)
                k += 1
                b += 1
                if b > 2:
                    a, b = a + 1, 0
        self.log_t = np.log10(self.targets)


def predict(F: Features, p: dict):
    """-> p_clear [n, K] (NaN past the end), p_win [n], e_blinds [n] (expected blinds beaten in the whole game)."""
    K = F.targets.shape[1]
    t = np.arange(K)[None, :]
    g = F.g_jokers[:, None] + (p["c0"] + p["c1"] * F.log_income + p["c2"] * F.ante)[:, None]
    logS = F.log_round[:, None] - math.log10(p["pace"]) + t * g
    z = (logS - F.log_t) / p["sigma"]
    pc = np.where(np.isfinite(F.log_t), _phi(np.nan_to_num(z, nan=-50.0)), np.nan)
    run = np.cumprod(np.where(np.isfinite(pc), pc, 1.0), axis=1)
    run = np.where(np.isfinite(pc), run, np.nan)
    p_win = np.where(np.isfinite(run).any(1), np.nanmin(np.where(np.isfinite(run), run, 1.0), axis=1), 1.0)
    e_bl = F.done_blinds + np.nansum(run, axis=1)
    return pc, p_win, e_bl


class WinProb:
    def __init__(self, params=None, path=PARAMS_PATH):
        if params is None:
            params = json.load(open(path)) if os.path.exists(path) else dict(DEFAULT)
        self.p = params
        self.ev = RunEval(default_weights())

    def __call__(self, g, jid, jf, cid=None):
        F = Features(g, jid, jf, cid, self.ev)
        pc, pw, eb = predict(F, self.p)
        return dict(p_clear=pc, p_win=pw, e_blinds=eb)


def fit(F: Features, e_blinds_obs, p_win_obs=None, w_win=50.0):
    """Least squares on the expected blinds (and, weighted, the win rate) over the parameters, by a coarse-to-fine search."""
    best = None
    grid = dict(pace=[1.0, 1.5, 2.0, 3.0], sigma=[0.2, 0.35, 0.5, 0.7], c0=[-0.02, 0.0, 0.02, 0.05, 0.08], c1=[0.0, 0.04, 0.08, 0.12], c2=[-0.01, 0.0, 0.01])
    import itertools
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid.keys(), vals))
        _, pw, eb = predict(F, p)
        loss = float(np.mean((eb - e_blinds_obs) ** 2))
        if p_win_obs is not None:
            loss += w_win * float(np.mean((pw - p_win_obs) ** 2))
        if best is None or loss < best[0]:
            best = (loss, p)
    # refine each parameter around the best
    loss, p = best
    for _ in range(3):
        for k, step in (("pace", 0.15), ("sigma", 0.04), ("c0", 0.006), ("c1", 0.01), ("c2", 0.002)):
            for d in (-2, -1, 1, 2):
                q = dict(p)
                q[k] = q[k] + d * step
                if q["pace"] <= 0.3 or q["sigma"] <= 0.05:
                    continue
                _, pw, eb = predict(F, q)
                l2 = float(np.mean((eb - e_blinds_obs) ** 2)) + (w_win * float(np.mean((pw - p_win_obs) ** 2)) if p_win_obs is not None else 0.0)
                if l2 < loss:
                    loss, p = l2, q
    return p, loss
