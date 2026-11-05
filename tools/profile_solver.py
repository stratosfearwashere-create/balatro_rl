"""Where the exact round solver's time goes. Rounds = every blind of the winning and losing lines in a beam output directory (each line replayed,
the blind selected, the solver run from its first decision) on one thread: nodes used, time per node, cleared or not, and what lower node caps
would have changed.

    python tools/profile_solver.py traces/prune_b128_p7
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv            # noqa: E402
from balatro_rl.env import A_SELECT                    # noqa: E402
from profile_beam import env                           # noqa: E402


def rounds_of(files, cap=400):
    """The first decision of every blind played along the lines: a vector env of them (one thread) and their antes / blind indices."""
    store = env(cap, 1)
    meta = []
    for f in files:
        j = json.load(open(f))
        acts = j.get("actions") or []
        v = env(1, j["seed"])
        prev_sel = False
        for a in acts:
            g = np.asarray(v.obs["g"])[0]
            in_round = g[5] > 0.5
            if in_round and prev_sel and len(meta) < cap:
                store.copy_from(v, 0, len(meta))
                meta.append((int(round(g[0] * 8)), int(np.argmax(g[1:4]))))
            prev_sel = (a == A_SELECT)
            mask = np.asarray(v.obs["mask"])[0]
            if a < 0 or not mask[a]:
                break
            _, d, _ = v.step(np.array([a]))
            if d[0]:
                break
    return store, meta


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "traces/prune_b128_p7"
    files = sorted(glob.glob(os.path.join(d, "*.json")))[:30]
    store, meta = rounds_of(files)
    idx = list(range(len(meta)))
    print(f"{len(idx)} rounds from {len(files)} lines (one thread)")
    res = {}
    for cap in (1000, 300, 100, 30):
        t0 = time.perf_counter()
        r = store.solve_best(idx, cap)
        res[cap] = (r, time.perf_counter() - t0)
    r1000, t1000 = res[1000]
    nodes = r1000[:, 2]
    cleared = r1000[:, 1] > 0.5
    print(f"cap 1000: {1000 * t1000 / len(idx):.1f} ms per round, {t1000 / nodes.sum() * 1e6:.0f} us per node, {nodes.mean():.0f} nodes per round on average")
    print(f"  cleared rounds ({cleared.sum()}): {nodes[cleared].mean():.0f} nodes on average, {nodes[cleared].sum() / nodes.sum():.0%} of all nodes")
    print(f"  failed rounds ({(~cleared).sum()}): {nodes[~cleared].mean() if (~cleared).any() else 0:.0f} nodes on average, {nodes[~cleared].sum() / nodes.sum():.0%} of all nodes")
    for cap in (300, 100, 30):
        r, t = res[cap]
        lost = (r1000[:, 1] > 0.5) & (r[:, 1] < 0.5)
        print(f"cap {cap:4d}: {1000 * t / len(idx):.1f} ms per round ({t / t1000:.2f} of cap 1000); clears lost vs cap 1000: {lost.sum()} of {cleared.sum()}")
    # how many nodes the first clearing line needs: the solver's first line is the greedy one, so cap 1-ish finds it when greedy clears
    r1, _ = store.solve_best(idx, 1), 0
    r_small = store.solve_best(idx, 8)
    print(f"greedy-first lines that already clear (cap 8): {int((r_small[:, 1] > 0.5).sum())} of {cleared.sum()} cleared rounds")
    antes = np.array([m[0] for m in meta])
    print("nodes per round by ante (cap 1000):", "  ".join(f"{a}: {nodes[antes == a].mean():.0f}" for a in sorted(set(antes))))


if __name__ == "__main__":
    main()
