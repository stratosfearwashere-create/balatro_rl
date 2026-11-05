"""What the shop offers, measured on the engine's own shop generation (Red deck, the stake given), per shop visit and ante: the cards in the
two card slots when the shop opens and after each reroll (planets, tarots by name, spectrals, jokers) and the two booster packs by kind and size.
The shops are rolled in the shop-only game (blinds won without play) with money enough for every reroll; nothing is bought. The table feeds
balatro_rl/growth.py (what an investment in a scaling joker can buy per round).

    python tools/shop_rates.py --games 1024 --rerolls 4 --out balatro_rl/data/shop_rates.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import env as E                                  # noqa: E402
from balatro_rl.engine_env import CppVecEnv                      # noqa: E402
from balatro_rl.sim.items import VOCAB, PLANETS, TAROTS, SPECTRALS   # noqa: E402

KIND = {}
for i, n in enumerate(VOCAB):
    if n.startswith("j_"):
        KIND[i] = ("joker", n[2:])
    elif n.startswith("c_") and n[2:] in PLANETS:
        KIND[i] = ("planet", n[2:])
    elif n.startswith("c_") and n[2:] in TAROTS:
        KIND[i] = ("tarot", n[2:])
    elif n.startswith("c_") and n[2:] in SPECTRALS:
        KIND[i] = ("spectral", n[2:])
    elif n.startswith("p_"):
        KIND[i] = ("pack", n[2:])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--rerolls", type=int, default=4, help="rerolls looked at per shop visit")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--out", default="balatro_rl/data/shop_rates.json")
    a = p.parse_args()
    n = a.games
    v = CppVecEnv(n, "RED", a.stake, seed0=9_100_000, threads=0)
    v.set_shop_only(True, 50, 0.5)
    v.set_start(0.0, 0.0, 1.0, 2000)
    v.v.restart()
    visits = defaultdict(float)                                    # ante -> shop visits
    views = defaultdict(float)                                     # ante -> reroll views
    init = defaultdict(lambda: defaultdict(float))                 # ante -> item kind / name -> count in the opening card slots
    rer = defaultdict(lambda: defaultdict(float))                  # the same per reroll view
    packs = defaultdict(lambda: defaultdict(float))
    rerolled = np.zeros(n, dtype=int)
    seen_open = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    for _ in range(4000):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        g = np.asarray(v.obs["g"])
        mask = np.asarray(v.obs["mask"])
        aid = np.asarray(v.obs["aid"])
        act = np.full(n, -1, dtype=np.int64)
        for i in live:
            ante = int(round(g[i, 0] * 8))
            if ante >= 8:
                done[i] = True
                continue
            if g[i, 6] > 0.5:                                      # in a shop
                cards = [KIND.get(int(aid[i, s])) for s in range(E.A_BUY, E.A_BUY_PACK) if aid[i, s] > 0]
                cards = [c for c in cards if c and c[0] != "pack"]
                if not seen_open[i]:
                    seen_open[i] = True
                    visits[ante] += 1
                    tgt = init[ante]
                    for s in range(E.A_BUY_PACK, E.A_VOUCHER):
                        k = KIND.get(int(aid[i, s]))
                        if k and k[0] == "pack":
                            packs[ante][k[1]] += 1
                else:
                    views[ante] += 1
                    tgt = rer[ante]
                for kind, name in cards:
                    tgt[kind] += 1
                    tgt[f"{kind}:{name}"] += 1
                if rerolled[i] < a.rerolls and mask[i, E.A_REROLL]:
                    rerolled[i] += 1
                    act[i] = E.A_REROLL
                else:
                    act[i] = E.A_LEAVE
                    rerolled[i] = 0
                    seen_open[i] = False
            elif g[i, 4] > 0.5:
                act[i] = E.A_SELECT
            elif g[i, 7] > 0.5 and mask[i, E.A_PSKIP]:
                act[i] = E.A_PSKIP
            else:
                act[i] = int(np.flatnonzero(mask[i])[0])
        _, d, _ = v.step(act)
        done |= np.asarray(d, dtype=bool)
    out = {"stake": a.stake, "games": n, "rerolls": a.rerolls, "antes": {}}
    for ante in sorted(visits):
        out["antes"][str(ante)] = {
            "visits": visits[ante], "views": views[ante],
            "open": {k: c / visits[ante] for k, c in sorted(init[ante].items())},
            "reroll": {k: c / max(views[ante], 1) for k, c in sorted(rer[ante].items())},
            "packs": {k: c / visits[ante] for k, c in sorted(packs[ante].items())}}
        o = out["antes"][str(ante)]
        print(f"ante {ante}: {visits[ante]:.0f} visits | per opening shop: planets {o['open'].get('planet', 0):.2f}, tarots {o['open'].get('tarot', 0):.2f}, "
              f"jokers {o['open'].get('joker', 0):.2f} | per reroll: planets {o['reroll'].get('planet', 0):.2f}, tarots {o['reroll'].get('tarot', 0):.2f} | packs: "
              + ", ".join(f"{k} {x:.2f}" for k, x in o["packs"].items() if x >= 0.05))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
