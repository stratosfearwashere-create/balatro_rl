"""Seed planner: which build can this seed's own shops make, and does it beat the ante-8 boss?

  1. scan     the seed's shops (exact mode, the shop-only game: every blind before the final one won without play, so all three shops of each
              ante are visited): each shop's jokers when it opens and after each of --rerolls rerolls (looked at on a copy; the main line never
              rerolls, so later shops are as the seed makes them for a player who did not reroll before), and the jokers in its Buffoon packs
              (opened on a copy). -> for each joker: the earliest shop it can be had in, at what reroll depth and price.
  2. shortlist  the jokers offered before the ante-8 boss, ranked by their measured strength at ante 8 (balatro_rl/joker_strength.json) and, for
              the scaling ones, by their growth from the shop they can be bought in to the ante-8 boss (balatro_rl/growth.py's per-round rate x
              the rounds owned); the best --keep go on.
  3. enumerate  every combination of --size of the shortlist that the run can afford in the order the shops offer them (price plus the rerolls to
              reach each one, against about $8 of income a blind), each scaling joker grown from its own shop to the ante-8 boss (investment
              jokers sharing the budget), placed at the ante-8 boss (VecEnv.set_build: the main hand at level 6, a suit-tilted deck for flush
              builds) and played by the exact round solver on --quick redrawn decks; the best --top again on --samples.
  4. rank       -> the plans, best first: the jokers, where each comes from (ante, shop, reroll depth, price), the clear rate.

    python tools/seed_plan.py --seeds 43001-43050 --out traces/seed_plans.json
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import env as E                                    # noqa: E402
from balatro_rl.engine_env import CppVecEnv                         # noqa: E402
from balatro_rl.growth import project                               # noqa: E402
from balatro_rl.runeval import from_vecenv                          # noqa: E402
from balatro_rl.sim.items import VOCAB                              # noqa: E402
from balatro_rl.sim.jokers import JOKERS                            # noqa: E402
from ante8_combos import place, main_hand, INVEST, FINALS           # noqa: E402

KEYS = list(JOKERS)
STRENGTH = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "balatro_rl", "joker_strength.json")))["jokers"]
LEGENDARY = {k for k, d in JOKERS.items() if getattr(d, "rarity", 1) >= 4}
EXCLUDE = LEGENDARY | {"madness"}                  # Madness destroys a random joker at every small / big blind: a plan cannot keep its other jokers
NO_COUNTER = {"supernova", "abstract", "bull", "bootstraps", "swashbuckler", "throwback", "steel_joker", "stone", "erosion", "hiker"}


def jname(i):
    n = VOCAB[int(i)] if 0 <= int(i) < len(VOCAB) else ""
    return n[2:] if n.startswith("j_") else None


def scan(seed, rerolls=4, stake="GOLD", antes=8):
    """-> {joker: (ante, shop, depth, price, source)} the earliest place each joker appears (depth = rerolls needed, -1 = a Buffoon pack)."""
    v = CppVecEnv(1, "RED", stake, seed0=seed, threads=1)
    v.set_exact(True)
    v.set_shop_only(True, 50, 0.5)
    v.set_start(0.0, 0.0, 1.0, 5000)
    v.v.restart()
    scr = CppVecEnv(1, "RED", stake, seed0=seed, threads=1)
    scr.set_exact(True)
    scr.set_shop_only(True, 50, 0.5)
    seen = {}
    shop_i = {}
    for _ in range(600):
        g = np.asarray(v.obs["g"])[0]
        mask = np.asarray(v.obs["mask"])[0]
        ante = int(round(g[0] * 8))
        if ante >= antes:
            break
        if g[6] > 0.5:                                                  # a shop: look at it on a copy
            s = shop_i.get(ante, 0)
            shop_i[ante] = s + 1
            for depth in range(rerolls + 1):
                if depth == 0:
                    scr.copy_from(v, 0, 0)
                sg = np.asarray(scr.obs["g"])[0]
                aid = np.asarray(scr.obs["aid"])[0]
                for slot in range(E.A_BUY, E.A_BUY_PACK):
                    k = jname(aid[slot])
                    if k and k not in seen:
                        seen[k] = (ante, s, depth, int(JOKERS[k].cost), "shop")
                if depth < rerolls and np.asarray(scr.obs["mask"])[0][E.A_REROLL]:
                    scr.step(np.array([E.A_REROLL]))
            # Buffoon packs in the opening shop: opened on a copy
            aid0 = np.asarray(v.obs["aid"])[0]
            for slot in range(E.A_BUY_PACK, E.A_VOUCHER):
                pk = VOCAB[int(aid0[slot])] if aid0[slot] > 0 else ""
                if pk.startswith("p_buffoon") and mask[slot]:
                    scr.copy_from(v, 0, 0)
                    scr.step(np.array([slot]))
                    paid = np.asarray(scr.obs["aid"])[0]
                    for q in range(E.A_PICK, E.A_PSKIP):
                        k = jname(paid[q])
                        if k and k not in seen:
                            seen[k] = (ante, s, -1, 4 if pk.endswith("normal") else (6 if pk.endswith("jumbo") else 8), "buffoon")
            v.step(np.array([E.A_LEAVE]))
        elif g[4] > 0.5:
            v.step(np.array([E.A_SELECT]))
        elif g[7] > 0.5 and mask[E.A_PSKIP]:
            v.step(np.array([E.A_PSKIP]))
        else:
            v.step(np.array([int(np.flatnonzero(mask)[0])]))
    return seen


def shortlist(seen, keep, rates):
    """The best `keep` jokers by measured ante-8 strength, plus the growth a scaler gets from its shop to the ante-8 boss."""
    out = []
    for k, (ante, s, depth, price, src) in seen.items():
        if k in EXCLUDE:
            continue
        st = STRENGTH.get(k, {})
        gain8 = st.get("gains", [0, 0, 0, 0.05])[-1] if st.get("measured") else 0.05
        rounds_owned = max(0, 23 - (3 * (ante - 1) + s + 1))
        grow = rates.get(k, 0.0) * rounds_owned
        out.append((gain8 + grow, k))
    out.sort(reverse=True)
    return [k for _, k in out[:keep]]


def growth_rates():
    """Per-round growth of each scaling joker, as a log10 score gain, from the engine's measured potential (env.hpp POT_MODE 1 table, here in Python)."""
    # counter growth per round (growth.py, on mid-game states) and what one unit is worth as a share of a typical ante-8 score
    per = {"runner": 4.08 / 300, "square": 3.0 / 300, "castle": 4.12 / 300, "wee": 5.6 / 300, "green_joker": 1.51 / 60, "trousers": 1.46 / 60,
           "red_card": 1.5 / 60, "flash": 0.8 / 60, "ride_the_bus": 0.3 / 60, "ceremonial": 1.6 / 60, "constellation": 0.045, "hologram": 0.078,
           "campfire": 0.03, "madness": 0.333 / 2, "obelisk": 0.02, "lucky_cat": 0.01, "glass": 0.01, "vampire": 0.01}
    return per


def affordable(combo, seen):
    """The purchases in the order the shops offer them: price + the rerolls to reach each (5, 6, ... a reroll) against $4 + $8 a blind."""
    items = sorted(combo, key=lambda k: seen[k][:3])
    spent = 0.0
    for k in items:
        ante, s, depth, price, src = seen[k]
        spent += price + sum(5 + r for r in range(max(depth, 0)))
        blinds = 3 * (ante - 1) + s + 1
        if spent > 4 + 8 * blinds:
            return False
    return True


def counters_for(combos, seen):
    """Each combination's projected counters at the ante-8 boss: growth.py's per-round rate (at an ante-3 state holding it) x the rounds each joker
    is owned from its own shop; investment jokers share the budget."""
    n = len(combos)
    v = CppVecEnv(n, "RED", "GOLD", seed0=31, threads=0)
    v.v.restart()
    for i, c in enumerate(combos):
        place(v, [i], c, [float("nan")] * len(c), 3, 0, 0)
    rows = project(**from_vecenv(v))
    out, steel = [], []
    for c, rs in zip(combos, rows):
        k_inv = max(1, sum(1 for x in c if x in INVEST))
        by = {r["key"]: r for r in rs}
        cs, st = [], 0
        for key in c:
            r = by.get(key)
            if r is None or r["unit"] not in ("chips", "mult", "xmult") or key in NO_COUNTER:
                cs.append(float("nan"))
                continue
            ante, s, _, _, _ = seen[key]
            owned = max(0, 23 - (3 * (ante - 1) + s + 1))
            per = r["per_round"] / (k_inv if key in INVEST else 1)
            cs.append(float(max(r["now"] + per * owned, 0.0)))
        out.append(cs)
        steel.append(st)
    return out, steel


def evaluate(combos, counters, steel, samples, seed0, nodes=300):
    n = len(combos) * samples
    v = CppVecEnv(n, "RED", "GOLD", seed0=seed0, threads=0)
    v.v.restart()
    for k, c in enumerate(combos):
        place(v, list(range(k * samples, (k + 1) * samples)), c, counters[k], 8, 2, steel[k])
    v.step(np.full(n, E.A_SELECT))
    r = v.solve_best(list(range(n)), nodes)
    return r[:, 1].reshape(len(combos), samples).mean(1), np.minimum(r[:, 0], 1.0).reshape(len(combos), samples).mean(1)


def plan_seed(seed, a, rates):
    seen = scan(seed, a.rerolls)
    short = shortlist(seen, a.keep, rates)
    combos = [list(c) for c in itertools.combinations(short, a.size) if affordable(c, seen)]
    if not combos:
        return dict(seed=seed, offered=len(seen), shortlist=short, plans=[])
    cnt, steel = counters_for(combos, seen)
    clr, sh = [], []
    for c0 in range(0, len(combos), 2048):
        x, y = evaluate(combos[c0:c0 + 2048], cnt[c0:c0 + 2048], steel[c0:c0 + 2048], a.quick, seed * 7 + c0)
        clr.append(x)
        sh.append(y)
    clr, sh = np.concatenate(clr), np.concatenate(sh)
    order = np.argsort(-(clr + 0.01 * sh))[:a.top]
    top = [combos[i] for i in order]
    c2, s2 = evaluate(top, [cnt[i] for i in order], [steel[i] for i in order], a.samples, seed * 11)
    o2 = np.argsort(-(c2 + 0.01 * s2))
    plans = []
    for j in o2:
        i = order[j]
        plans.append(dict(jokers=combos[i], clear=float(c2[j]), share=float(s2[j]), hand=["flush", "two pair", "straight"][[5, 2, 4].index(main_hand(combos[i]))]
                          if main_hand(combos[i]) in (5, 2, 4) else str(main_hand(combos[i])),
                          counters=[None if not np.isfinite(x) else round(float(x), 2) for x in cnt[i]],
                          where={k: dict(zip(("ante", "shop", "rerolls", "price", "source"), seen[k])) for k in combos[i]}))
    return dict(seed=seed, offered=len(seen), shortlist=short, combos=len(combos), plans=plans)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="43001-43050")
    p.add_argument("--rerolls", type=int, default=4)
    p.add_argument("--keep", type=int, default=16)
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--quick", type=int, default=4)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--out", default="traces/seed_plans.json")
    a = p.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")]
    rates = growth_rates()
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    t0 = time.time()
    for sd in range(lo, hi + 1):
        if str(sd) in out:
            continue
        t = time.time()
        r = plan_seed(sd, a, rates)
        out[str(sd)] = r
        best = r["plans"][0] if r["plans"] else None
        print(f"seed {sd}: {r['offered']} jokers offered, {r.get('combos', 0)} affordable combinations; best plan "
              + (f"{best['clear']:.2f} clear [{best['hand']}] " + ", ".join(f"{k}(a{best['where'][k]['ante']}s{best['where'][k]['shop']}"
                                                                            f"{'r' + str(best['where'][k]['rerolls']) if best['where'][k]['rerolls'] > 0 else ('p' if best['where'][k]['rerolls'] < 0 else '')})"
                                                                            for k in best["jokers"]) if best else "none") + f"  {time.time() - t:.0f}s", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
