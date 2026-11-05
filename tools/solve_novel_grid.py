"""Settings of the novelty beam on the same seeds: mean blind reached (a win counts as 24) and wins, per configuration.

    python tools/solve_novel_grid.py --seeds 41001-41010
    python tools/solve_novel_grid.py --seeds 42001-42010 --only "pot1"
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_novel import solve      # noqa: E402

# name: (beam, elite, floor, money weight, potential weight, scale weight)
GRID = {
    "base b32 e16":          (32, 16, 0.5, 0.0, 0.0, 0.0),
    "e8":                    (32, 8, 0.5, 0.0, 0.0, 0.0),
    "e24":                   (32, 24, 0.5, 0.0, 0.0, 0.0),
    "floor1.0":              (32, 16, 1.0, 0.0, 0.0, 0.0),
    "floor0.25":             (32, 16, 0.25, 0.0, 0.0, 0.0),
    "money.01":              (32, 16, 0.5, 0.01, 0.0, 0.0),
    "money.03":              (32, 16, 0.5, 0.03, 0.0, 0.0),
    "pot0.5":                (32, 16, 0.5, 0.0, 0.5, 0.0),
    "pot1":                  (32, 16, 0.5, 0.0, 1.0, 0.0),
    "pot1 money.01":         (32, 16, 0.5, 0.01, 1.0, 0.0),
    "scale1":                (32, 16, 0.5, 0.0, 0.0, 1.0),
    "scale1 pot1":           (32, 16, 0.5, 0.0, 1.0, 1.0),
    "pot1 e8 floor1.0":      (32, 8, 1.0, 0.0, 1.0, 0.0),
}

p = argparse.ArgumentParser()
p.add_argument("--stake", default="GOLD")
p.add_argument("--deck", default="RED")
p.add_argument("--seeds", default="41001-41010")
p.add_argument("--only", default="")
a = p.parse_args()
lo, hi = [int(x) for x in a.seeds.split("-")]
n = hi - lo + 1
for name, (beam, elite, floor, mw, pw, sw_) in GRID.items():
    if a.only and a.only not in name:
        continue
    t = time.time()
    levels, wins = [], 0
    for sd in range(lo, hi + 1):
        path, info = solve(a.deck, a.stake, sd, beam, elite, floor, 8, 4, 4, 1000, mw, pw, sw_)
        if path:
            wins += 1
            levels.append(24)
        else:
            levels.append(info["level"])
    print(f"{name:22s} mean blind {sum(levels) / n:5.2f}  wins {wins}/{n}  {time.time() - t:4.0f}s   {sorted(levels)}", flush=True)
