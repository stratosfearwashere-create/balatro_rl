"""Which joker combinations beat the ante-8 boss at Gold? Each combination is taken as owned from ante 3 (counters at zero), its scaling jokers
projected to the ante-8 boss with balatro_rl/growth.py (the round's budget split between the investment jokers it holds), then placed at the
ante-8 boss with those counters (VecEnv.set_build: the main hand at level 6, a deck tilted to one suit for flush builds, Steel cards for Steel
Joker / Baron, $30) and the boss played by the exact round solver (it sees the draws: an upper bound on a good player) on many redrawn decks
and random final bosses.

  stage 1   --combos random combinations x --quick redraws each
  stage 2   the --top best x --samples redraws: the clear rate with its error

    python tools/ante8_combos.py --combos 3000 --quick 4 --top 40 --samples 64
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gen_builds import KEYS                                       # noqa: E402
from balatro_rl.engine_env import CppVecEnv                        # noqa: E402
from balatro_rl.env import A_SELECT                                # noqa: E402
from balatro_rl.growth import project                              # noqa: E402
from balatro_rl.runeval import from_vecenv                         # noqa: E402
from balatro_rl.sim.hands import FLUSH, STRAIGHT, TWO_PAIR, PAIR   # noqa: E402

SCALE = ["constellation", "hologram", "flash", "red_card", "green_joker", "runner", "madness", "campfire", "ride_the_bus", "trousers", "square",
         "castle", "wee", "throwback", "ceremonial", "supernova", "fortune_teller", "obelisk", "hit_the_road", "steel_joker"]      # (legendaries left out: only The Soul makes them)
SUPPORT = ["blueprint", "brainstorm", "cavendish", "duo", "trio", "family", "order", "tribe", "blackboard", "stencil", "card_sharp", "acrobat",
           "flower_pot", "seeing_double", "ancient", "bloodstone", "photograph", "baseball", "abstract", "bootstraps", "bull", "swashbuckler", "baron",
           "mime", "hanging_chad", "sock_and_buskin", "smiley", "scary_face", "jolly", "zany", "mad", "crazy", "droll", "sly", "wily", "clever",
           "devious", "crafty", "space", "four_fingers", "shortcut", "smeared", "splash", "gros_michel", "popcorn", "ice_cream", "ramen", "joker",
           "half", "mystic_summit", "raised_fist", "banner", "blue_joker", "even_steven", "odd_todd", "scholar", "walkie_talkie", "fibonacci",
           "onyx_agate", "arrowhead", "lusty_joker", "greedy_joker", "wrathful_joker", "gluttenous_joker", "spare_trousers", "dusk", "hack"]
SUPPORT = [k for k in SUPPORT if k in KEYS and k not in SCALE]
INVEST = {"constellation", "hologram", "flash", "red_card", "campfire", "ceremonial", "fortune_teller", "steel_joker"}
FINALS = 200_000 * 2


def main_hand(combo):
    if "runner" in combo or "shortcut" in combo:
        return STRAIGHT
    if "trousers" in combo or "square" in combo or "duo" in combo or "jolly" in combo or "sly" in combo or "mad" in combo:
        return TWO_PAIR
    return FLUSH


def place(v, ids, combo, counters, ante, blind, steel):
    lv = [1] * 12
    mh = main_hand(combo)
    lv[mh] = 6 if ante == 8 else 2
    if mh == TWO_PAIR:
        lv[PAIR] = 3 if ante == 8 else 1
    suit, suit_n = (0, 10) if mh == FLUSH else (-1, 0)
    if "steel_joker" in combo or "baron" in combo:
        steel = max(steel, 6)
    v.set_build(ids, ante, blind, [KEYS.index(k) for k in combo], counters, steel, lv, 30, [0] * len(combo), suit, suit_n)


def projected_counters(combos):
    """Each combination at ante 3 (blind 0), projected to the ante-8 boss; investment jokers share the budget."""
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
        cs = []
        st = 0
        for key in c:
            r = by.get(key)
            if r is None or r["unit"] not in ("chips", "mult", "xmult"):
                cs.append(float("nan"))
                continue
            per = r["per_round"] / (k_inv if key in INVEST else 1)
            R = 17
            fut = r["now"] + per * R if key in INVEST else r["future"]
            if key == "steel_joker":
                st = int(round((fut - 1.0) / 0.2))
                cs.append(float("nan"))
                continue
            if key in ("supernova", "abstract", "bull", "bootstraps", "swashbuckler", "throwback"):
                cs.append(float("nan"))                              # (computed by the engine from the state)
                continue
            cs.append(float(fut))
        out.append(cs)
        steel.append(st)
    return out, steel


def clear_rate(combos, counters, steel, samples, seed0, nodes=300):
    n = len(combos) * samples
    v = CppVecEnv(n, "RED", "GOLD", seed0=seed0, threads=0)
    v.v.restart()
    for k, c in enumerate(combos):
        ids = list(range(k * samples, (k + 1) * samples))
        place(v, ids, c, counters[k], 8, 2, steel[k])
    v.step(np.full(n, A_SELECT))
    r = v.solve_best(list(range(n)), nodes)
    cleared = r[:, 1].reshape(len(combos), samples)
    share = np.minimum(r[:, 0], 1.0).reshape(len(combos), samples)
    return cleared.mean(1), share.mean(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--combos", type=int, default=3000)
    p.add_argument("--quick", type=int, default=4)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    combos = set()
    while len(combos) < a.combos:
        k = int(rng.integers(1, 4))
        c = list(rng.choice(SCALE, size=k, replace=False)) + list(rng.choice(SUPPORT, size=5 - k, replace=False))
        combos.add(tuple(sorted(c)))
    combos = [list(c) for c in combos]
    t0 = time.time()
    cnt, steel = projected_counters(combos)
    print(f"projected {len(combos)} combinations ({time.time() - t0:.0f}s)", flush=True)
    clr, sh = [], []
    B = 1024
    for c0 in range(0, len(combos), B):
        a1, b1 = clear_rate(combos[c0:c0 + B], cnt[c0:c0 + B], steel[c0:c0 + B], a.quick, 1000 + c0)
        clr.append(a1)
        sh.append(b1)
    clr, sh = np.concatenate(clr), np.concatenate(sh)
    print(f"stage 1 ({time.time() - t0:.0f}s): combinations that cleared at least once {int((clr > 0).sum())} of {len(combos)}; mean share of the target {sh.mean():.2f}", flush=True)
    order = np.argsort(-(clr + 0.01 * sh))[:a.top]
    top = [combos[i] for i in order]
    c2, s2 = clear_rate(top, [cnt[i] for i in order], [steel[i] for i in order], a.samples, 99_000)
    o2 = np.argsort(-(c2 + 0.01 * s2))
    print(f"\nstage 2 ({time.time() - t0:.0f}s): the best {len(top)} on {a.samples} redrawn decks and final bosses each")
    print(f"{'clear rate':>10} {'share':>6}  combination (projected counters at the ante-8 boss)")
    for j in o2:
        i = order[j]
        desc = ", ".join(f"{k}" + (f" {x:.2f}" if np.isfinite(x) else "") for k, x in zip(combos[i], cnt[i]))
        err = np.sqrt(max(c2[j] * (1 - c2[j]), 1e-9) / a.samples)
        print(f"{c2[j]:10.2f} {s2[j]:6.2f}  [{('flush', 'two pair', 'straight')[(FLUSH, TWO_PAIR, STRAIGHT).index(main_hand(combos[i]))]}] {desc}  (+-{err:.2f})"
              + (f" steel cards {steel[i]}" if steel[i] else ""))


if __name__ == "__main__":
    main()
