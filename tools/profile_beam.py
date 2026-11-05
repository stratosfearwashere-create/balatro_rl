"""Where the seed-aware beam's time goes. Root states = the blind selects along a verified winning line (tools/verify_beam.py); one level of the
search (VecEnv.expand_level) is timed on them on one thread under variants that remove one part at a time, plus the Python side of a level.

    python tools/profile_beam.py traces/beam_retest128/GOLD_43015.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv            # noqa: E402
from balatro_rl.env import A_SELECT                    # noqa: E402
from solve_novel import pick                           # noqa: E402


def env(n, seed, threads=1):
    v = CppVecEnv(n, "RED", "GOLD", seed0=seed, threads=threads)
    v.set_exact(True)
    v.v.restart()
    v.set_weight_shape(2.5, 1.0)
    v.set_marginal(2)
    return v


def roots_along(path_file):
    j = json.load(open(path_file))
    seed, acts = j["seed"], j["actions"]
    v = env(1, seed)
    snaps = []
    store = env(64, seed)
    for a in acts:
        g = np.asarray(v.obs["g"])[0]
        if g[4] > 0.5 and len(snaps) < 64:                    # at a blind select
            store.copy_from(v, 0, len(snaps))
            snaps.append(int(round(g[0] * 8)))
        _, d, _ = v.step(np.array([a]))
        if d[0]:
            break
    return store, snaps, seed


def timed(store, idx, seed, ends=8, sw=4, sl=4, node_cap=1000, scripted=-1, prune=0, reps=1):
    dest = env(len(idx) * ends, seed)
    store.set_level_scripted(scripted)
    store.set_search_prune(prune)
    t0 = time.perf_counter()
    for _ in range(reps):
        res = store.expand_level(idx, dest, ends, sw, sl, node_cap, 0.0, 0.0, 1.0)
    dt = (time.perf_counter() - t0) / reps
    store.set_level_scripted(-1)
    store.set_search_prune(0)
    n_c = sum(len(c) for _, _, c in res)
    return dt, n_c, res, dest


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "traces/beam_retest128/GOLD_43015.json"
    store, antes, seed = roots_along(path)
    idx = list(range(len(antes)))
    print(f"{len(idx)} root states (blind selects of seed {seed}, antes {min(antes)}-{max(antes)}), one thread\n")
    base, nc, res, dest = timed(store, idx, seed)
    rows = [("full level (round solver 1000 nodes, shop width 4, depth 4, 8 ends)", base, nc)]
    for label, kw in (("round solver capped at 100 nodes", dict(node_cap=100)),
                      ("no round solving (scripted player plays the blind)", dict(scripted=0)),
                      ("no shop search (width 1, depth 1)", dict(sw=1, sl=1)),
                      ("shop width 8 (12 ends)", dict(sw=8, ends=12)),
                      ("shop depth 6", dict(sl=6)),
                      ("pruning on (rules 1 + 2)", dict(prune=3))):
        dt, n2, _, _ = timed(store, idx, seed, **kw)
        rows.append((label, dt, n2))
    print(f"{'variant':72s} {'ms / root':>10s} {'vs full':>8s} {'end states':>10s}")
    for label, dt, n2 in rows:
        print(f"{label:72s} {1000 * dt / len(idx):10.1f} {dt / base:8.2f} {n2:10d}")
    # per ante: the full level
    print("\nfull level by ante (ms / root):", end="")
    for a in sorted(set(antes)):
        ids = [i for i, x in enumerate(antes) if x == a]
        dt, _, _, _ = timed(store, ids, seed)
        print(f"  ante {a}: {1000 * dt / len(ids):.0f}", end="")
    print()
    # the Python side of one beam-128 level: pool, pick, copy
    pool = []
    for b, (win, wp, cands) in enumerate(res):
        for p_, sc, sg, slot, jk in cands:
            pool.append((sc, sg, list(p_), slot, jk))
    pool = (pool * (1024 // max(1, len(pool)) + 1))[:1024]
    pool = [(c[0] + 1e-6 * k, c[1] + k, c[2], c[3], c[4]) for k, c in enumerate(pool)]
    t0 = time.perf_counter()
    keep = pick(list(pool), 128, 32, 0.5)
    t_pick = time.perf_counter() - t0
    cur = env(128, seed, threads=0)
    big = env(1024, seed, threads=0)
    t0 = time.perf_counter()
    cur.copy_many(big, list(range(128)), list(range(128)))
    t_copy = time.perf_counter() - t0
    t0 = time.perf_counter()
    big.state_keys(list(range(1024)))
    t_keys = time.perf_counter() - t0
    print(f"\nPython side of a beam-128 level (1024 candidates): novelty pick {1000 * t_pick:.0f} ms, copying 128 states {1000 * t_copy:.0f} ms, "
          f"state keys for 1024 {1000 * t_keys:.0f} ms")
    print(f"engine side of a beam-128 level on this machine (all threads): about {128 * base / len(idx) / os.cpu_count() * 1000:.0f} ms if it scales linearly over {os.cpu_count()} threads")


if __name__ == "__main__":
    main()
