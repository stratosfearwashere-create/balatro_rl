"""How strong is each joker? Measured in the engine, by differences, with no labels.

For a joker J and an ante A: many random contexts (0-4 other random jokers, scaling jokers at the counter they would have at ante A if held since ante 1, random hand levels in line with the ante,
the default deck). The same game state (the same deck order and draws) is built twice, once with J added and once without, and faced with the ante-A boss blind. The exact round solver gives the
best share of the boss's chip target any line reaches. J's strength at ante A is the mean of

    gain   log10(share with J) - log10(share without J), the shares floored at 1e-3
    lift   how much more often the blind is cleared (share >= 1) with J than without

over the contexts, with a standard error. The table is written to traces/joker_strength.jsonl and the top and bottom jokers are printed with the role_of label of seed_report.py alongside, which
is only used to look at the result, never to compute it.

    python tools/joker_strength.py --contexts 128 --antes 2,4,6,8
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("BALATRO_DEV_ENGINE", "")
from solve_fast import make                                                      # noqa: E402
from seed_report import role_of                                                  # noqa: E402
from balatro_rl.env import A_SELECT                                              # noqa: E402
from balatro_rl.potential import GROW                                            # noqa: E402
from balatro_rl.sim.jokers import JOKERS                                         # noqa: E402

KEYS = list(JOKERS)
FLOOR = 1e-3


def counter_at(key, ante):
    """The counter a scaling joker held since ante 1 has at the boss of `ante` (the engine's growth table)."""
    g = GROW.get(key)
    if g is None:
        return float("nan")
    start, rate, lo, hi = g
    return float(min(max(start + rate * (3 * (ante - 1) + 2), lo), hi))


def run_chunk(pairs, ante, seed0, node_cap):
    """pairs: [(joker key, other keys, levels, position)] -> [(share with, share without)]. Both builds start from the same game state."""
    n = len(pairs)
    base = make(n, "RED", "GOLD", seed0, 0)
    a = make(n, "RED", "GOLD", seed0, 0)
    b = make(n, "RED", "GOLD", seed0, 0)
    for v in (a, b):
        v.copy_many(base, list(range(n)), list(range(n)))
        v.set_auto_round(False, 1.5, 0)
    for i, (j, others, levels, pos) in enumerate(pairs):
        ks_with = others[:pos] + [j] + others[pos:]                                  # J at a random place: Blueprint and Brainstorm act on their neighbours
        a.set_build([i], ante, 2, [KEYS.index(k) for k in ks_with], [counter_at(k, ante) for k in ks_with], 0, levels, 25)
        b.set_build([i], ante, 2, [KEYS.index(k) for k in others], [counter_at(k, ante) for k in others], 0, levels, 25)
    out = []
    shares = []
    for v in (a, b):
        act = np.full(n, -1, dtype=np.int64)
        act[:n] = A_SELECT
        v.step(act)
        shares.append(v.solve_best(list(range(n)), node_cap)[:, 0])
    return list(zip(shares[0], shares[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", type=int, default=128)
    p.add_argument("--antes", default="2,4,6,8")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--node-cap", type=int, default=300)
    p.add_argument("--max-others", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="traces/joker_strength.jsonl")
    p.add_argument("--only", default="", help="comma-separated joker keys (default: all)")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    jokers = [k for k in a.only.split(",") if k] or KEYS
    antes = [int(x) for x in a.antes.split(",")]
    rows = []
    t0 = time.time()
    seed0 = 500000
    for ante in antes:
        allpairs = []
        for j in jokers:
            for _ in range(a.contexts):
                k = int(rng.integers(0, a.max_others + 1))
                pool = [x for x in KEYS if x != j]
                others = [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
                top = 1 + int(round(rng.uniform(0, 1.5) * (ante - 1)))              # the hand levels of a build in this ante
                levels = [1] * 12
                for h in rng.choice(12, size=int(rng.integers(1, 4)), replace=False):
                    levels[h] = int(rng.integers(1, top + 1))
                allpairs.append((j, others, levels, int(rng.integers(0, len(others) + 1))))
        res = []
        for c0 in range(0, len(allpairs), a.chunk):
            seed0 += 100000
            res += run_chunk(allpairs[c0:c0 + a.chunk], ante, seed0, a.node_cap)
        by = {}
        for (j, _, _, _), (sw, so) in zip(allpairs, res):
            by.setdefault(j, []).append((sw, so))
        for j, v in by.items():
            v = np.array(v)
            gain = np.log10(np.maximum(v[:, 0], FLOOR)) - np.log10(np.maximum(v[:, 1], FLOOR))
            lift = (v[:, 0] >= 1).astype(float) - (v[:, 1] >= 1).astype(float)
            rows.append({"joker": j, "ante": ante, "role": role_of(j), "n": len(v), "gain": float(gain.mean()), "gain_se": float(gain.std() / np.sqrt(len(v))),
                         "lift": float(lift.mean()), "lift_se": float(lift.std() / np.sqrt(len(v))), "clear_with": float((v[:, 0] >= 1).mean()), "clear_without": float((v[:, 1] >= 1).mean())})
        print(f"ante {ante}: {len(allpairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for ante in antes:
        rr = sorted((r for r in rows if r["ante"] == ante), key=lambda r: -r["gain"])
        print(f"\nante {ante} boss: strongest and weakest by gain (log10 share; lift = change in the share of contexts that clear)")
        for r in rr[:15] + rr[-6:]:
            print(f"  {r['joker']:18s} {r['role']:7s} gain {r['gain']:+.3f} +-{r['gain_se']:.3f}  lift {r['lift']:+.3f} +-{r['lift_se']:.3f}  clears {r['clear_without']:.2f} -> {r['clear_with']:.2f}")
        roles = {}
        for r in rr:
            roles.setdefault(r["role"], []).append(r["gain"])
        print("  mean gain by the hand-written role: " + ", ".join(f"{k} {np.mean(v):+.3f} (n {len(v)})" for k, v in sorted(roles.items())))


if __name__ == "__main__":
    main()
