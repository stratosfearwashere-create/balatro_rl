"""Cheap attempts instead of one wide search: for each seed, try the small search (shop width 4, 4 layers) with different rankings (money weight, growth-potential
weight) and beam widths until one wins. Reports which variant solved each seed and the time.

    python tools/solve_variants.py --stake WHITE --seeds 31001,31002,31003,31004
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import solve_adaptive      # noqa: E402

VARIANTS = [                                   # (money weight, potential weight)
    (0.01, 0.0), (0.0, 0.0), (0.03, 0.0), (0.01, 0.5), (0.005, 1.0),
]

p = argparse.ArgumentParser()
p.add_argument("--stake", default="WHITE")
p.add_argument("--seeds", default="31001,31002,31003,31004")
p.add_argument("--widths", default="8,16,24")
p.add_argument("--out-dir", default="traces/variants")
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
widths = [int(x) for x in a.widths.split(",")]
tot_solved, tot_secs = 0, 0.0
for sd in [int(x) for x in a.seeds.split(",")]:
    t = time.time()
    won = None
    for mw, pw in VARIANTS:
        path, info = solve_adaptive("RED", a.stake, sd, widths, 8, 4, 4, 1000, mw, pw, log=lambda s: None)
        if path:
            won = (mw, pw)
            json.dump({"seed": sd, "deck": "RED", "stake": a.stake, "solved": True, "actions": path, "info": info, "variant": won},
                      open(os.path.join(a.out_dir, f"{a.stake}_{sd}.json"), "w"))
            break
    dt = time.time() - t
    tot_solved += won is not None
    tot_secs += dt
    print(f"seed {sd}: {'solved by money %.3f, potential %.1f' % won if won else 'not solved'}  {dt:.0f}s", flush=True)
print(f"{tot_solved} solved, {tot_secs:.0f}s total", flush=True)
