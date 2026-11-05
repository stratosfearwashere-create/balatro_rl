"""A grid of solver settings on the same seeds: for each, whether it wins and how far it gets (the blind where the beam dies), and the time.

    python tools/solve_grid.py --stake GOLD --seeds 100521
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import solve      # noqa: E402

# name: (ends, shop width, layers, node cap, beam, money weight, potential weight)
GRID = {
    "tiny   b32":            (8, 4, 4, 1000, 32, 0.0, 0.0),
    "tiny   b128":           (8, 4, 4, 1000, 128, 0.0, 0.0),
    "small  b32":            (8, 8, 5, 1000, 32, 0.0, 0.0),
    "medium b32":            (8, 16, 6, 1500, 32, 0.0, 0.0),
    "wide   b32":            (16, 32, 7, 1500, 32, 0.0, 0.0),
    "wide   b64 ends24":     (24, 32, 7, 1500, 64, 0.0, 0.0),
    "medium b64 money.01":   (8, 16, 6, 1500, 64, 0.01, 0.0),
    "medium b64 pot1":       (8, 16, 6, 1500, 64, 0.0, 1.0),
    "medium b64 pot3":       (8, 16, 6, 1500, 64, 0.0, 3.0),
    "tiny   b256":           (8, 4, 4, 1000, 256, 0.0, 0.0),
}

p = argparse.ArgumentParser()
p.add_argument("--stake", default="GOLD")
p.add_argument("--seeds", default="100521")
p.add_argument("--only", default="")
a = p.parse_args()
for sd in [int(x) for x in a.seeds.split(",")]:
    print(f"seed {sd} {a.stake}", flush=True)
    for name, (ends, sw, sl, cap, beam, mw, pw) in GRID.items():
        if a.only and a.only not in name:
            continue
        t = time.time()
        path, info = solve("RED", a.stake, sd, beam, ends, sw, sl, cap, mw, pw)
        print(f"  {name:22s} {'SOLVED' if path else 'died at blind %2d' % info['level']:18s} {time.time() - t:6.0f}s", flush=True)
