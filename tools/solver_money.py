"""Does stopping the round solver early (VecEnv.set_solver_extra) cost money? The same rounds as tools/profile_solver.py; for each stopping rule the best
line the solver keeps (max_plans 1, node cap 1000, as the beam's levels use) is replayed on a copy and the money after the round is read.

    python tools/solver_money.py traces/prune_b128_p7
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from profile_beam import env                                   # noqa: E402
from profile_solver import rounds_of                          # noqa: E402
import glob                                                    # noqa: E402


def money_after(store, idx, plans, seed=1):
    """Replays each plan on a copy of its round; -> money after the round (NaN: no plan), hands used."""
    n = len(idx)
    w = env(n, seed)
    w.copy_many(store, list(idx), list(range(n)))
    out = np.full(n, np.nan)
    used = np.zeros(n)
    for k in range(n):
        if not plans[k]:
            continue
        acts = plans[k][0]
        for a in acts:
            act = np.full(n, -1, dtype=np.int64)
            act[k] = a
            w.step(act)
        used[k] = sum(1 for a in acts if a < 218)
        g = np.asarray(w.obs["g"])[k]
        out[k] = g[17] * 50.0 if g[17] < 1.99 else 10.0 ** (3.0 * g[18]) - 1.0
    return out, used


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "traces/prune_b128_p7"
    files = sorted(glob.glob(os.path.join(d, "*.json")))[:30]
    store, meta = rounds_of(files)
    idx = list(range(len(meta)))
    print(f"{len(idx)} rounds")
    base = None
    for extra in (-1, 200, 50, 10, 0):
        store.set_solver_extra(extra)
        t0 = time.perf_counter()
        plans = store.solve_rounds(idx, 1, 1000)
        dt = time.perf_counter() - t0
        money, used = money_after(store, idx, plans)
        cleared = np.isfinite(money)
        if base is None:
            base = (money, used, dt)
        dm = money - base[0]
        ok = cleared & np.isfinite(base[0])
        print(f"extra {extra:>4}: {1000 * dt / len(idx):5.1f} ms/round ({dt / base[2]:.2f}x) | cleared {cleared.sum()} | money after round: mean {np.nanmean(money):.2f}, "
              f"vs full search {np.mean(dm[ok]):+.2f} per round, rounds with less money {int((dm[ok] < -0.5).sum())} (worst {np.min(dm[ok]) if ok.any() else 0:+.0f}) | plays used {used[cleared].mean():.2f}")
    store.set_solver_extra(-1)


if __name__ == "__main__":
    main()
