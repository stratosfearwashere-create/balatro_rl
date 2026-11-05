"""Training data for the evaluation network (balatro_rl/nnue.py): states the seed-aware beam generates, labelled by playouts.

For each seed: the beam (tools/solve_novel.py, beam --beam) runs; at every level up to --per-level of its candidate end states (blind selects)
are sampled and their features read (with the engine's future list of the seed's shops). Then each sampled state is rebuilt by replaying its
path in an exact game of that seed and played to the end by the policy (--ckpt, greedy; the future is the seed's own, so one playout is exact
for that policy). Label: the blinds the playout reaches (24 = a win). -> an npz shard per --chunk seeds.

    python tools/nnue_data.py --seeds 44001-44300 --beam 32 --per-level 48 --out data/nnue
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv                    # noqa: E402
from balatro_rl.nnue import features                             # noqa: E402
from balatro_rl.runeval import from_vecenv                       # noqa: E402
from policy_search import Pol                                    # noqa: E402
from solve_novel import solve                                    # noqa: E402


def label(pol, seed, paths, max_steps=4000):
    """Replays each path in an exact game of `seed`, then plays it out with the policy -> blinds reached (24 = won)."""
    n = len(paths)
    v = CppVecEnv(n, "RED", "GOLD", seed0=seed, threads=0)
    v.set_exact(True)
    v.set_fixed_seed(seed)
    v.v.restart()
    L = max(len(p) for p in paths)
    done = np.zeros(n, dtype=bool)
    bl = np.zeros(n)
    won = np.zeros(n, dtype=bool)
    for t in range(L):
        a = np.full(n, -1, dtype=np.int64)
        for i, p in enumerate(paths):
            if t < len(p) and not done[i]:
                a[i] = p[t]
        _, d, info = v.step(a)
        f = d & ~done
        bl[f], won[f] = info["blinds"][f], info["won"][f]
        done |= d
    for _ in range(max_steps):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        a = np.full(n, -1, dtype=np.int64)
        a[live] = pol.logits(v, live).argmax(1)
        _, d, info = v.step(a)
        f = d & ~done
        bl[f], won[f] = info["blinds"][f], info["won"][f]
        done |= d
    return np.where(won, 24.0, bl)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="44001-44300")
    p.add_argument("--beam", type=int, default=32)
    p.add_argument("--per-level", type=int, default=48)
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v7_best_greedy.pt")
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--out", default="data/nnue")
    a = p.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")]
    os.makedirs(a.out, exist_ok=True)
    pol = Pol(a.ckpt, "cuda")
    rng = np.random.default_rng(lo)
    feat_env = CppVecEnv(a.per_level, "RED", "GOLD", seed0=1, threads=0)
    feat_env.set_exact(True)
    feat_env.set_priv(True)
    buf = dict(idx=[], offs=[], y=[], seed=[], ante=[], level=[], beam_final=[])
    t0 = time.time()
    for sd in range(lo, hi + 1):
        shard = os.path.join(a.out, f"nnue_{(sd - lo) // a.chunk:03d}.npz")
        if os.path.exists(shard):
            continue
        recs = []

        def on_pool(level, dest, pool):
            k = min(a.per_level, len(pool))
            pick = rng.choice(len(pool), size=k, replace=False)
            slots = [pool[j][3] for j in pick]
            feat_env.copy_many(dest, slots, list(range(k)))
            feat_env.set_priv(True)
            o = from_vecenv(feat_env, list(range(k)))
            o["priv"] = np.asarray(feat_env.obs["priv"])[:k]
            idx, offs = features(o)
            for r, j in enumerate(pick):
                recs.append((idx[offs[r]:offs[r + 1]], list(pool[j][2]), level, int(round(o["g"][r, 0] * 8))))

        path, info = solve("RED", "GOLD", sd, a.beam, max(1, a.beam // 4), 0.5, 8, 4, 4, 1000, 0.0, 0.0, 1.0,
                           prune=7, speed=(True, True, 0.0), hand_bound=True, on_pool=on_pool)
        final = 24 if path else info["level"]
        if recs:
            y = label(pol, sd, [r[1] for r in recs])
            for (fi, _p, lev, an), yy in zip(recs, y):
                buf["idx"].append(fi)
                buf["y"].append(yy)
                buf["seed"].append(sd)
                buf["ante"].append(an)
                buf["level"].append(lev)
                buf["beam_final"].append(final)
        n_states = sum(1 for s_ in buf["seed"])
        print(f"seed {sd}: beam {'won' if path else 'died at level %d' % info['level']}, {len(recs)} states (mean label {np.mean([buf['y'][-i] for i in range(1, len(recs) + 1)]) if recs else 0:.1f} blinds); "
              f"{n_states} in the shard buffer, {time.time() - t0:.0f}s", flush=True)
        if (sd - lo + 1) % a.chunk == 0 or sd == hi:
            offs = np.cumsum([0] + [len(x) for x in buf["idx"]])
            np.savez_compressed(shard, idx=np.concatenate(buf["idx"]) if buf["idx"] else np.zeros(0, np.int64), offs=offs,
                                y=np.array(buf["y"], dtype=np.float32), seed=np.array(buf["seed"]), ante=np.array(buf["ante"]),
                                level=np.array(buf["level"]), beam_final=np.array(buf["beam_final"]))
            print(f"-> {shard} ({len(buf['y'])} states)", flush=True)
            buf = {k: [] for k in buf}


if __name__ == "__main__":
    main()
