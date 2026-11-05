"""What do the shops of a seed offer? For each ante, the jokers seen at the first shop and after R rerolls (exact-seed mode: the cards are a fixed
sequence per ante, so purchases change them little), split into multiplier jokers and growing (scaling) jokers.

    python tools/shop_scan.py --seeds 40001-40016 --rerolls 8
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.env import swap_role                                  # noqa: E402
from balatro_rl.marginal import POTENTIAL                             # noqa: E402
from balatro_rl.sim.game import Game                                  # noqa: E402
from balatro_rl.sim.jokers import JOKERS                              # noqa: E402
from balatro_rl.sim.seedrng import seed_string                        # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seeds", default="40001-40016")
p.add_argument("--rerolls", type=int, default=8)
p.add_argument("--stake", default="GOLD")
p.add_argument("--antes", type=int, default=6)
a = p.parse_args()
lo, hi = [int(x) for x in a.seeds.split("-")]
XMULT = {k for k in JOKERS if swap_role(k) == 2}
SCALERS = set(POTENTIAL)
print(f"{len(XMULT)} multiplier jokers, {len(SCALERS)} growing jokers in the pool\n")
seen_by_ante = defaultdict(lambda: [0, 0, 0])        # ante -> seeds with an xMult / a scaler / either visible
for sd in range(lo, hi + 1):
    g = Game(seed=1, deck_type="RED", stake=a.stake, exact_seed=seed_string(sd))
    g.money = 10_000
    line = []
    for ante in range(1, a.antes + 1):
        g.ante = ante
        g.shop, g.shop_packs, g.shop_vouchers, g.ante_voucher = [], [], [], None
        g.open_shop()
        cards = [(it.key[2:], it.cost) for it in g.shop if it.kind == "joker"]
        for _ in range(a.rerolls):
            g.free_rerolls = 1
            g.reroll()
            cards += [(it.key[2:], it.cost) for it in g.shop if it.kind == "joker"]
        names = {k for k, _ in cards}
        xm = sorted(k for k in names if k in XMULT)
        sc = sorted(k for k in names if k in SCALERS)
        seen_by_ante[ante][0] += bool(xm)
        seen_by_ante[ante][1] += bool(sc)
        seen_by_ante[ante][2] += bool(xm or sc)
        line.append(f"a{ante}: x{len(xm)} s{len(sc)}")
        if sd == lo:
            print(f"  seed {sd} ante {ante}: multipliers {xm}; growing {sc}")
    print(f"seed {sd}: " + "  ".join(line))
n = hi - lo + 1
print(f"\nseeds (of {n}) that show at least one ... within {a.rerolls} rerolls of the first shop of the ante:")
for ante in sorted(seen_by_ante):
    x, s, e = seen_by_ante[ante]
    print(f"  ante {ante}: multiplier {x}/{n}   growing {s}/{n}   either {e}/{n}")
