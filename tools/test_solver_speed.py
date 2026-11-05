"""Checks the round solver's speed-ups (VecEnv.set_solver_speed) on real rounds from a beam output directory.

  lite      the same search with cheaper nodes: every plan must be identical to the full solver's, only faster
  dead_f    the hopeless-branch cut: clears lost on rounds the full solver clears, and the time saved on rounds it fails
  cache     the beam's shared round cache: one beam level on a set of roots with it off and on (end states found, time, hits)

    python tools/test_solver_speed.py traces/prune_b128_p7
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.env import A_SELECT                     # noqa: E402
from profile_beam import env                            # noqa: E402
from profile_solver import rounds_of                   # noqa: E402


def failing_rounds(files, cap=200):
    """The blind each dead line could not clear: the line replayed, then that blind selected."""
    store = env(cap, 1)
    n = 0
    for f in files:
        j = json.load(open(f))
        if j.get("solved"):
            continue
        v = env(1, j["seed"])
        for a in j.get("actions") or []:
            if a < 0 or not np.asarray(v.obs["mask"])[0][a]:
                break
            _, d, _ = v.step(np.array([a]))
            if d[0]:
                break
        g = np.asarray(v.obs["g"])[0]
        if g[4] > 0.5 and n < cap:                       # at the blind select of the blind it died on
            v.step(np.array([A_SELECT]))
            store.copy_from(v, 0, n)
            n += 1
    return store, n


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "traces/prune_b128_p7"
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    store, meta = rounds_of(files[:30])
    idx = list(range(len(meta)))
    # lite: identical plans
    store.set_solver_speed(False, False, 0.0)
    t0 = time.perf_counter(); full = store.solve_rounds(idx, 3, 1000); t_full = time.perf_counter() - t0
    store.set_solver_speed(True, False, 0.0)
    t0 = time.perf_counter(); lite = store.solve_rounds(idx, 3, 1000); t_lite = time.perf_counter() - t0
    same = sum(1 for a, b in zip(full, lite) if [list(x) for x in a] == [list(x) for x in b])
    print(f"lite: {same}/{len(idx)} rounds with identical plans; {1000 * t_full / len(idx):.1f} -> {1000 * t_lite / len(idx):.1f} ms per round ({t_full / t_lite:.1f}x faster)")
    # dead_f on cleared rounds (lite on)
    base = store.solve_best(idx, 1000)
    fstore, nf = failing_rounds(files)
    fidx = list(range(nf))
    store.set_solver_speed(True, False, 0.0)
    fstore.set_solver_speed(True, False, 0.0)
    t0 = time.perf_counter(); fb = fstore.solve_best(fidx, 1000); t_fb = time.perf_counter() - t0
    print(f"{nf} failing rounds (where dead lines died): full solver clears {int(fb[:, 1].sum())}, {fb[:, 2].mean():.0f} nodes, {1000 * t_fb / max(nf, 1):.1f} ms per round")
    for f in (1.0, 1.5, 2.0, 3.0):
        store.set_solver_speed(True, False, f)
        t0 = time.perf_counter(); r = store.solve_best(idx, 1000); t = time.perf_counter() - t0
        lost = int(((base[:, 1] > 0.5) & (r[:, 1] < 0.5)).sum())
        t1 = time.perf_counter(); rf = fstore.solve_best(fidx, 1000); tf = time.perf_counter() - t1
        print(f"dead_f {f}: cleared rounds {1000 * t / len(idx):.1f} ms (clears lost {lost} of {int(base[:, 1].sum())}), failing rounds {1000 * tf / max(nf, 1):.1f} ms "
              f"({rf[:, 2].mean():.0f} nodes; newly cleared {int(((fb[:, 1] < 0.5) & (rf[:, 1] > 0.5)).sum())}, lost {int(((fb[:, 1] > 0.5) & (rf[:, 1] < 0.5)).sum())})")
    store.set_solver_speed(False, False, 0.0)


if __name__ == "__main__":
    main()
