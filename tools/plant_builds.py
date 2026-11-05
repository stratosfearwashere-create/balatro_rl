"""Strong builds per ante for the reverse curriculum (train.py --plant-file): random search, then hill-climbing, scored by the fast power measure.

Step 1  random builds per ante (tools/gen_builds.make_build: realistic counters for the ante, shop-rate editions, hand levels rising with the ante,
        suit-tilted decks for suit jokers, Steel cards for Baron / Steel Joker, money 0 .. $5 x ante; no legendaries). Antes 1-3 hold only 1-2 jokers
        (the late stages of the curriculum, where the agent must complete the build itself). Each is placed at its ante's boss (VecEnv.set_build) and
        scored by VecEnv.fast_power: hands x mean best play (redrawn hands, the exact scorer) / the boss target. No solver.
Step 2  the top --keep share, spread over main hand types (no hand type above --hand-cap of an ante's set), is hill-climbed: each round every build gets
        --mutants mutations (swap a joker for another, rescale a growing joker's counter, move a hand level), a mutant that scores higher replaces it.
        Realism is kept by the generator's own bounds: the joker count, levels within 1 + 1.5 x (ante - 1), counters at most the ante's natural value.
Output: JSON {ante: [build, ...]} with each build's parameters and power, and a summary of the patterns (jokers and hand types of the strong builds).

    python tools/plant_builds.py --per-ante 20000 --keep 0.03 --rounds 4 --out data/plants/plants_v1.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gen_builds import KEYS, make_build, strength_weights          # noqa: E402
from joker_strength import counter_at                              # noqa: E402
from balatro_rl.engine_env import CppVecEnv                        # noqa: E402
from balatro_rl.env import A_SELECT                                # noqa: E402

LEGENDARY = {"caino", "canio", "triboulet", "yorick", "chicot", "perkeo"}
HANDS = ["high_card", "pair", "two_pair", "three", "straight", "flush", "full_house", "four", "straight_flush", "five", "flush_house", "flush_five"]
N_JOKERS_EARLY = {1: (1,), 2: (1, 2), 3: (2,)}


def sample(rng, ante, W):
    while True:
        ks, cs, es, lv, su, sn, st, mo = make_build(rng, ante, W)
        if not (LEGENDARY & set(ks)):
            break
    if ante in N_JOKERS_EARLY:                               # 1-2 jokers early: keep a random subset
        n = int(rng.choice(N_JOKERS_EARLY[ante]))
        keep = sorted(rng.choice(len(ks), size=n, replace=False).tolist())
        ks, cs, es = [ks[i] for i in keep], [cs[i] for i in keep], [es[i] for i in keep]
    return {"ante": ante, "keys": ks, "counters": [None if (c is None or np.isnan(c)) else float(c) for c in cs], "eds": es, "levels": lv,
            "suit": su, "suit_n": sn, "steel": st, "money": mo}


def place(v, i, b, blind=2):
    v.set_build([i], b["ante"], blind, [KEYS.index(k) for k in b["keys"]], [float("nan") if c is None else c for c in b["counters"]], b["steel"],
                b["levels"], b["money"], b["eds"], b["suit"], b["suit_n"])


def score(v, builds, samples):
    """fast power of each build at its ante's boss (hands x mean best play / target)."""
    out = np.zeros(len(builds))
    for c0 in range(0, len(builds), v.n):
        chunk = builds[c0:c0 + v.n]
        v.v.restart()
        for i, b in enumerate(chunk):
            place(v, i, b)
        act = np.full(v.n, -1, dtype=np.int64)
        act[:len(chunk)] = A_SELECT
        v.step(act)
        f = v.fast_power(list(range(len(chunk))), samples)
        out[c0:c0 + len(chunk)] = f[:, 0]
    return out


def main_hand(b):
    """the highest-levelled hand (-1: none levelled)"""
    return int(np.argmax(b["levels"])) if max(b["levels"]) > 1 else -1


def mutate(rng, b, W, pool):
    m = json.loads(json.dumps(b))
    a = m["ante"]
    op = rng.random()
    if op < 0.5 and m["keys"]:                               # swap a joker
        j = int(rng.integers(len(m["keys"])))
        cand = [k for k in pool if k not in m["keys"]]
        p = np.array([W[a, KEYS.index(k)] for k in cand])
        k = cand[int(rng.choice(len(cand), p=p / p.sum()))]
        c = counter_at(k, a)
        m["keys"][j], m["counters"][j], m["eds"][j] = k, (None if np.isnan(c) else float(c * rng.uniform(0.4, 1.0))), 0
    elif op < 0.7:                                           # rescale a growing joker's counter (at most the ante's natural value)
        js = [j for j, c in enumerate(m["counters"]) if c is not None]
        if js:
            j = int(rng.choice(js))
            nat = counter_at(m["keys"][j], a)
            if not np.isnan(nat):
                m["counters"][j] = float(np.clip(m["counters"][j] * rng.uniform(0.8, 1.3), 0.0, nat))
    else:                                                    # move a hand level (bounded by the ante)
        cap = 1 + int(round(1.5 * (a - 1)))
        h = main_hand(m) if rng.random() < 0.6 and main_hand(m) >= 0 else int(rng.integers(0, 9))
        m["levels"][h] = int(np.clip(m["levels"][h] + rng.choice([-1, 1, 1]), 1, cap))
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-ante", type=int, default=20000)
    p.add_argument("--antes", default="1-8")
    p.add_argument("--keep", type=float, default=0.03)
    p.add_argument("--hand-cap", type=float, default=0.35)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--mutants", type=int, default=6)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/plants/plants_v1.json")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    W = strength_weights()
    pool = [k for k in KEYS if k not in LEGENDARY]
    lo, hi = [int(x) for x in a.antes.split("-")]
    v = CppVecEnv(a.chunk, "RED", "GOLD", seed0=a.seed * 7919 + 3, threads=0)
    v.set_auto_round(False, 1.5, 0)
    res, summary = {}, {}
    t0 = time.time()
    for ante in range(lo, hi + 1):
        builds = [sample(rng, ante, W) for _ in range(a.per_ante)]
        pw = score(v, builds, a.samples)
        order = np.argsort(-pw)
        n_keep = max(8, int(a.keep * len(builds)))
        kept, per_hand = [], Counter()
        for i in order:                                      # the top share, no hand type above --hand-cap of it
            h = main_hand(builds[i])
            if per_hand[h] >= a.hand_cap * n_keep:
                continue
            builds[i]["power"] = float(pw[i])
            kept.append(builds[i])
            per_hand[h] += 1
            if len(kept) >= n_keep:
                break
        start = float(np.median([b["power"] for b in kept]))
        for r in range(a.rounds):                            # hill-climb
            muts = [mutate(rng, b, W, pool) for b in kept for _ in range(a.mutants)]
            mp = score(v, muts, a.samples)
            better = 0
            for i, b in enumerate(kept):
                js = range(i * a.mutants, (i + 1) * a.mutants)
                j = max(js, key=lambda x: mp[x])
                if mp[j] > b["power"]:
                    muts[j]["power"] = float(mp[j])
                    kept[i] = muts[j]
                    better += 1
            print(f"  ante {ante} round {r + 1}: {better}/{len(kept)} improved, median power {np.median([b['power'] for b in kept]):.2f}", flush=True)
        res[ante] = kept
        jk = Counter(k for b in kept for k in b["keys"])
        hd = Counter(HANDS[main_hand(b)] if main_hand(b) >= 0 else "none levelled" for b in kept)
        summary[ante] = {"random_median": float(np.median(pw)), "random_reach": float((pw >= 1).mean()), "kept": len(kept), "kept_median_start": start,
                         "kept_median": float(np.median([b["power"] for b in kept])), "jokers": jk.most_common(12), "hands": hd.most_common()}
        s = summary[ante]
        print(f"ante {ante}: random median {s['random_median']:.2f} ({100 * s['random_reach']:.1f}% reach the boss); kept {len(kept)}: median "
              f"{start:.2f} -> {s['kept_median']:.2f} after climbing ({time.time() - t0:.0f}s)\n  jokers: "
              + ", ".join(f"{k} {n}" for k, n in jk.most_common(10)) + "\n  hands: " + ", ".join(f"{h} {n}" for h, n in hd.most_common()), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"builds": {str(k): v_ for k, v_ in res.items()}, "summary": {str(k): v_ for k, v_ in summary.items()}, "args": vars(a)}, open(a.out, "w"))
    print("->", a.out)


if __name__ == "__main__":
    main()
