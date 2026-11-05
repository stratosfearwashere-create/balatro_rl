"""Goal-directed solving: for each seed, find which strong engine jokers the shops offer in antes 1-4, and search for a line that assembles each one
(a large ranking bonus for owning it, money weighted so it can be afforded) and plays the game out with the exact round solver.

    python tools/solve_target.py --stake GOLD --seeds 40001-40008
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import solve                                          # noqa: E402
from balatro_rl.sim.game import Game                                  # noqa: E402
from balatro_rl.sim.jokers import JOKERS                              # noqa: E402
from balatro_rl.sim.seedrng import seed_string                        # noqa: E402

ENGINES = ["hologram", "vampire", "constellation", "lucky_cat", "glass", "campfire", "obelisk", "cavendish", "madness",
           "duo", "trio", "family", "order", "tribe", "steel_joker", "baseball", "flower_pot"]
KEYS = list(JOKERS)


def offered(seed, stake, antes=4, rerolls=6):
    """The engines the shops show, with the first ante each appears in."""
    g = Game(seed=1, deck_type="RED", stake=stake, exact_seed=seed_string(seed))
    g.money = 10_000
    first = {}
    for ante in range(1, antes + 1):
        g.ante = ante
        g.shop, g.shop_packs, g.shop_vouchers, g.ante_voucher = [], [], [], None
        g.open_shop()
        names = {it.key[2:] for it in g.shop if it.kind == "joker"}
        for _ in range(rerolls):
            g.free_rerolls = 1
            g.reroll()
            names |= {it.key[2:] for it in g.shop if it.kind == "joker"}
        for k in names:
            if k in ENGINES:
                first.setdefault(k, ante)
    return sorted(first.items(), key=lambda kv: (kv[1], ENGINES.index(kv[0])))


p = argparse.ArgumentParser()
p.add_argument("--stake", default="GOLD")
p.add_argument("--seeds", default="40001-40008")
p.add_argument("--beam", type=int, default=32)
p.add_argument("--max-targets", type=int, default=5)
p.add_argument("--target-weight", type=float, default=3.0)
p.add_argument("--money-weight", type=float, default=0.02)
p.add_argument("--out-dir", default="traces/target")
a = p.parse_args()
lo, hi = [int(x) for x in a.seeds.split("-")]
os.makedirs(a.out_dir, exist_ok=True)
wins = 0
for sd in range(lo, hi + 1):
    cands = offered(sd, a.stake)[:a.max_targets]
    print(f"seed {sd}: engines offered (ante): {cands}", flush=True)
    best, won = 0, None
    for name, ante in cands:
        t = time.time()
        path, info = solve("RED", a.stake, sd, a.beam, 8, 4, 4, 1000, a.money_weight, 0.0, target=[KEYS.index(name)], target_w=a.target_weight)
        lvl = "WIN" if path else info["level"]
        print(f"    target {name:14s} -> {lvl}  ({time.time() - t:.0f}s)", flush=True)
        if path:
            won = name
            json.dump({"seed": sd, "deck": "RED", "stake": a.stake, "solved": True, "actions": path, "target": name},
                      open(os.path.join(a.out_dir, f"{a.stake}_{sd}.json"), "w"))
            break
        best = max(best, info["level"])
    wins += won is not None
    print(f"  => {'SOLVED with ' + won if won else 'best target reached blind %d' % best}", flush=True)
print(f"{wins}/{hi - lo + 1} solved", flush=True)
