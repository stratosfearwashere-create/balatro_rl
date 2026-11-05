"""Structured random builds, each with at least one +mult, one xMult and one chips joker, labelled by the fast power measure.

A build is made the way builds form in play:
  jokers       3-5: one +mult, one xMult, one chips joker (joker_desc categories), the rest from all jokers; picks weighted by the measured strength at that ante
               (balatro_rl/joker_strength.json; unmeasured jokers a base weight)
  counters     growing jokers at what they would have reached by this ante, held for part of the run (counter_at x U(0.4, 1))
  editions     at twice the shop's rates (foil 4%, holo 2.8%, polychrome 0.6%, negative 0.6%)
  hand levels  the main hand is the one the jokers favour (joker_desc hand_* columns), else a common hand; its level rises with the ante; sometimes a second hand too
  deck         suit jokers (Greedy, Lusty, Wrathful, Gluttonous, Arrowhead, Onyx Agate, Rough Gem, Bloodstone) tilt the deck toward their suit; Baron / Steel Joker bring steel cards
  money        0 .. $5 x ante
Each build faces its ante's boss (VecEnv.set_build); the label is VecEnv.fast_power (redrawn hands, best play by the exact scorer): hands x mean best play / target, and with max.

    python tools/gen_builds.py --n 1000000 --samples 4 --out checkpoints/builds_v1.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv                                      # noqa: E402
from balatro_rl.env import A_SELECT, item_id                                     # noqa: E402
from balatro_rl.joker_desc import FEATURES, item_descriptors                     # noqa: E402
from balatro_rl.sim.jokers import JOKERS                                         # noqa: E402
from joker_strength import counter_at                                            # noqa: E402

KEYS = list(JOKERS)
D = item_descriptors()
CAT = {f: [k for k in KEYS if D[item_id("j_" + k), FEATURES.index(f)] > 0.5] for f in ("mult", "xmult", "chips")}
HAND_COL = {"hand_pair": 1, "hand_two_pair": 2, "hand_three": 3, "hand_four": 7, "hand_straight": 4, "hand_flush": 5}
AFFINITY = {k: [h for f, h in HAND_COL.items() if f in FEATURES and D[item_id("j_" + k), FEATURES.index(f)] > 0.5] for k in KEYS}
COMMON = ([1, 2, 3, 4, 5, 6, 0], [0.30, 0.15, 0.10, 0.10, 0.20, 0.10, 0.05])
SUIT_OF = {"greedy_joker": 3, "lusty_joker": 1, "wrathful_joker": 0, "gluttenous_joker": 2, "arrowhead": 0, "onyx_agate": 2, "rough_gem": 3, "bloodstone": 1}
STEEL = {"baron", "steel_joker"}
ED_P = [(1, 0.04), (2, 0.028), (3, 0.006), (4, 0.006)]                       # foil, holo, polychrome, negative


def strength_weights():
    tab = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "balatro_rl", "joker_strength.json")))
    antes = tab["antes"]
    w = np.full((9, len(KEYS)), 0.3)
    for i, k in enumerate(KEYS):
        e = tab["jokers"].get(k)
        if not e or not e["measured"]:
            continue
        for a in range(1, 9):
            j = int(np.argmin([abs(a - x) for x in antes]))
            w[a, i] = 0.3 + max(0.0, e["gains"][j])
    return w


def make_build(rng, ante, W):
    n = int(rng.choice([3, 4, 5], p=[0.2, 0.3, 0.5]))
    chosen = []
    for cat in ("mult", "xmult", "chips"):
        pool = [k for k in CAT[cat] if k not in chosen]
        p = np.array([W[ante, KEYS.index(k)] for k in pool])
        chosen.append(pool[int(rng.choice(len(pool), p=p / p.sum()))])
    while len(chosen) < n:
        pool = [k for k in KEYS if k not in chosen]
        p = np.array([W[ante, KEYS.index(k)] for k in pool])
        chosen.append(pool[int(rng.choice(len(pool), p=p / p.sum()))])
    counters = []
    for k in chosen:
        c = counter_at(k, ante)
        counters.append(float("nan") if np.isnan(c) else c * rng.uniform(0.4, 1.0))
    eds = []
    for _ in chosen:
        u, e = rng.random(), 0
        for ed, p in ED_P:
            if u < p:
                e = ed
                break
            u -= p
        eds.append(e)
    aff = [h for k in chosen for h in AFFINITY[k]]
    main = int(rng.choice(aff)) if aff and rng.random() < 0.8 else int(rng.choice(COMMON[0], p=COMMON[1]))
    levels = [1] * 12
    levels[main] = 1 + int(round(rng.uniform(0, 1.5) * (ante - 1)))
    if rng.random() < 0.3:
        second = int(rng.choice(COMMON[0], p=COMMON[1]))
        levels[second] = max(levels[second], 1 + int(round(rng.uniform(0, 0.7) * (ante - 1))))
    suits = [SUIT_OF[k] for k in chosen if k in SUIT_OF]
    suit, suit_n = (int(suits[0]), int(rng.integers(0, 21))) if suits else (-1, 0)
    steel = int(rng.integers(0, 11)) if STEEL & set(chosen) else 0
    money = int(rng.integers(0, 5 * ante + 1))
    return chosen, counters, eds, levels, suit, suit_n, steel, money


def save(rec, path):
    """Writes what has been made so far (atomically) and returns it as arrays."""
    out = {k: np.array(x) for k, x in rec.items()}
    out["keys"] = out["keys"].astype(np.int16)
    out["eds"] = out["eds"].astype(np.int8)
    out["levels"] = out["levels"].astype(np.int8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--antes", default="2-8")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/builds_v1.npz")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    lo, hi = [int(x) for x in a.antes.split("-")]
    W = strength_weights()
    rec = {k: [] for k in ("ante", "keys", "eds", "counters", "levels", "suit", "suit_n", "steel", "money", "pow_mean", "pow_max", "best_mean", "best_max")}
    t0 = time.time()
    done = 0
    v = CppVecEnv(a.chunk, "RED", "GOLD", seed0=a.seed * 1_000_003 + 17, threads=0)
    v.set_auto_round(False, 1.5, 0)
    while done < a.n:
        m = min(a.chunk, a.n - done)
        v.v.restart()                                    # (fresh games: a new shuffle and boss draw for every chunk)
        antes = rng.integers(lo, hi + 1, size=m)
        for i in range(m):
            ks, cs, es, lv, su, sn, st, mo = make_build(rng, int(antes[i]), W)
            v.set_build([i], int(antes[i]), 2, [KEYS.index(k) for k in ks], cs, st, lv, mo, es, su, sn)
            rec["ante"].append(int(antes[i]))
            rec["keys"].append([KEYS.index(k) for k in ks] + [-1] * (5 - len(ks)))
            rec["eds"].append(es + [0] * (5 - len(es)))
            rec["counters"].append([0.0 if np.isnan(c) else c for c in cs] + [0.0] * (5 - len(cs)))
            rec["levels"].append(lv)
            rec["suit"].append(su)
            rec["suit_n"].append(sn)
            rec["steel"].append(st)
            rec["money"].append(mo)
        act = np.full(a.chunk, -1, dtype=np.int64)
        act[:m] = A_SELECT
        v.step(act)
        f = v.fast_power(list(range(m)), a.samples)
        for c, name in enumerate(("pow_mean", "pow_max", "best_mean", "best_max")):
            rec[name].extend(f[:, c].tolist())
        done += m
        if done % (a.chunk * 16) < a.chunk or done >= a.n:
            print(f"{done:,} builds, {done / (time.time() - t0):,.0f}/s", flush=True)
            save(rec, a.out)                              # (saved as it goes: stopping it keeps what was made)
    out = save(rec, a.out)
    pm = out["pow_mean"]
    print(f"\n{len(pm):,} builds -> {a.out} ({os.path.getsize(a.out) / 1e6:.0f} MB), {time.time() - t0:.0f}s")
    for an in range(lo, hi + 1):
        s = pm[out["ante"] == an]
        print(f"  ante {an} boss: median power {np.median(s):.3f} (share of the target), p90 {np.percentile(s, 90):.3f}; reaches the target {100 * (s >= 1).mean():.1f}%")
    top = np.argsort(-pm)[:5]
    for i in top:
        print(f"  strong: ante {out['ante'][i]} power {pm[i]:.2f}: " + ", ".join(KEYS[k] for k in out["keys"][i] if k >= 0) + f"; levels max {out['levels'][i].max()}")


if __name__ == "__main__":
    main()
