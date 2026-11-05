"""The money-option curve: how likely a disciplined fisher is to land a strong joker (an uncommon or rare xMult / scaling multiplier) within the next
--visits shop visits, by money held and ante, measured on the engine's own shop generation (Red deck, the stake given). The shops are rolled in
the shop-only game (blinds won without play and paid as won: the income and interest of a real run); the fisher rerolls while the money left
after the reroll stays at or above $25, leaves otherwise, and buys nothing else. The table is what train.py's --money-option-bonus and
--oracle-inputs read: P(find | money, ante).

    python tools/money_option.py --games 512 --visits 3 --out balatro_rl/data/money_option.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import env as E                      # noqa: E402
from balatro_rl.engine_env import CppVecEnv          # noqa: E402
from balatro_rl.sim.items import VOCAB               # noqa: E402
from balatro_rl.az.guide import XMULT, SLOW_SCALERS  # noqa: E402
from balatro_rl.sim.jokers import JOKERS             # noqa: E402

MONEY = [4, 8, 12, 16, 20, 25, 30, 35, 40, 50, 60]
STRONG = {k for k in (XMULT | SLOW_SCALERS | {"blueprint", "brainstorm"}) if k in JOKERS and getattr(JOKERS[k], "rarity", 1) >= 2}    # uncommon / rare multipliers
GOOD = np.array([i for i, n in enumerate(VOCAB) if n.startswith("j_") and n[2:] in STRONG], dtype=np.int64)


def measure(ante, money, games, visits, stake):
    v = CppVecEnv(games, "RED", stake, seed0=7_000_000 + 1000 * ante + money, threads=0)
    v.set_shop_only(True, 100, 0.5)
    if ante > 1:
        v.set_start_ante(ante)
    if money > 4:
        v.set_start(0.0, 0.0, 1.0, money - 4)
    v.v.restart()
    n = games
    found = np.zeros(n, dtype=bool)
    leaves = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    for _ in range(600):
        mask = np.asarray(v.obs["mask"])
        g = np.asarray(v.obs["g"])
        aid = np.asarray(v.obs["aid"])
        live = ~done & ~found & (leaves < visits)
        if not live.any():
            break
        cash = 10.0 ** (3.0 * g[:, 18]) - 1.0
        reroll_cost = np.where(g[:, 22] > 0, 0.0, g[:, 21] * 10.0)
        in_shop = g[:, 6] > 0.5
        offered = mask[:, E.A_BUY:E.A_BUY_PACK] & np.isin(aid[:, E.A_BUY:E.A_BUY_PACK], GOOD)
        hit = live & in_shop & offered.any(1)
        found |= hit
        act = np.full(n, -1, dtype=np.int64)
        sel = live & (g[:, 4] > 0.5)
        act[sel] = E.A_SELECT
        shop = live & in_shop & ~hit
        rr = shop & mask[:, E.A_REROLL] & (cash - reroll_cost >= 25.0)
        act[rr] = E.A_REROLL
        lv = shop & ~rr
        act[lv] = E.A_LEAVE
        leaves[lv] += 1
        pk = live & (g[:, 7] > 0.5)
        for i in np.flatnonzero(pk):
            act[i] = E.A_PSKIP if mask[i, E.A_PSKIP] else int(np.flatnonzero(mask[i])[0])
        rest = live & (act < 0) & ~hit
        for i in np.flatnonzero(rest):
            act[i] = int(np.flatnonzero(mask[i])[0])
        _, d, _ = v.step(act)
        done |= np.asarray(d, dtype=bool)
    return float(found.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--visits", type=int, default=3, help="shop visits the fisher gets (one ante)")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--out", default="balatro_rl/data/money_option.json")
    a = p.parse_args()
    t0 = time.time()
    table = []
    for ante in range(1, 9):
        row = [measure(ante, m, a.games, a.visits, a.stake) for m in MONEY]
        table.append(row)
        print(f"ante {ante}: " + "  ".join(f"${m}:{100 * x:.0f}%" for m, x in zip(MONEY, row)), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"stake": a.stake, "visits": a.visits, "games": a.games, "good": sorted(STRONG), "money": MONEY, "p": table},
              open(a.out, "w"), indent=1)
    print(f"-> {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
