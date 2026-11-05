"""Measure 1 on real states: play games with a random shop policy; at every blind select estimate the pass probability (K reseeded copies, exact scorer), then play the round for real
and compare (calibration: the share of rounds cleared per bracket of the estimate).

    python tools/power_demo.py --games 256 --k 64
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv          # noqa: E402
from balatro_rl.env import A_SELECT                  # noqa: E402
from balatro_rl.power import PassProbability         # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--games", type=int, default=256)
p.add_argument("--k", type=int, default=64)
p.add_argument("--steps", type=int, default=400)
a = p.parse_args()
v = CppVecEnv(a.games, "RED", "GOLD", seed0=5, threads=0)
v.set_auto_round(True, 1.5, 0)
pp = PassProbability(k=a.k)
rng = np.random.default_rng(0)
est, real, ante, blind = [], [], [], []
for step in range(a.steps):
    g = np.asarray(v.obs["g"]).copy()
    mask = np.asarray(v.obs["mask"])
    sel = np.flatnonzero(mask[:, A_SELECT])
    act = np.array([rng.choice(np.flatnonzero(r)) for r in mask])
    if len(sel):
        r = pp(v, sel)
        for e, pe in zip(sel, r["p"]):
            act[e] = A_SELECT                                           # (play the blind for real, with the scripted player's first mode)
        v.set_modes([0] * a.games)
        _, d, info = v.step(act)
        rep = v.round_report()
        for e, pe in zip(sel, r["p"]):
            est.append(float(pe)); real.append(bool(rep["cleared"][e]))
            ante.append(int(round(g[e, 0] * 8))); blind.append(int(np.argmax(g[e, 1:4])))
    else:
        _, d, info = v.step(act)
est, real, ante = np.array(est), np.array(real), np.array(ante)
print(f"{len(est)} blind selects; mean estimated pass {est.mean():.3f}; cleared for real {real.mean():.3f}")
print("estimate bracket        n   estimated   cleared for real")
for lo, hi in ((0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)):
    m = (est >= lo) & (est < hi)
    if m.sum():
        print(f"  {lo:.2f}-{min(hi, 1):.2f}       {int(m.sum()):6d}     {est[m].mean():.3f}       {real[m].mean():.3f}")
print("by ante: " + "  ".join(f"{x}: est {est[ante == x].mean():.2f} real {real[ante == x].mean():.2f} (n={int((ante == x).sum())})" for x in range(1, 9) if (ante == x).sum() > 20))
