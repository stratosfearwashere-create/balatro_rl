"""Time a few solver settings on the same seeds (solved count and seconds), to pick the cheapest one that solves as much as the widest.

    python tools/solve_tune.py --stake WHITE --seeds 31001,31002,31003,31004
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import solve_adaptive      # noqa: E402

CONFIGS = {
    "A narrow  (ends 8,  shop 32)": dict(ends_per=8, sw=32, sl=7, node_cap=1000),
    "B medium  (ends 16, shop 32)": dict(ends_per=16, sw=32, sl=7, node_cap=1000),
    "C wide    (ends 24, shop 48)": dict(ends_per=24, sw=48, sl=7, node_cap=1500),
    "D small   (ends 8,  shop 8, layers 5)": dict(ends_per=8, sw=8, sl=5, node_cap=1000),
    "E tiny    (ends 8,  shop 4, layers 4)": dict(ends_per=8, sw=4, sl=4, node_cap=1000),
}

p = argparse.ArgumentParser()
p.add_argument("--stake", default="WHITE")
p.add_argument("--seeds", default="31001,31002,31003,31004")
p.add_argument("--widths", default="8,16,24")
p.add_argument("--configs", default="A,B,C")
a = p.parse_args()
seeds = [int(x) for x in a.seeds.split(",")]
widths = [int(x) for x in a.widths.split(",")]
for name, kw in CONFIGS.items():
    if name[0] not in a.configs.split(","):
        continue
    solved, secs = 0, 0.0
    per = []
    for sd in seeds:
        t = time.time()
        path, info = solve_adaptive("RED", a.stake, sd, widths, kw["ends_per"], kw["sw"], kw["sl"], kw["node_cap"], 0.01, 0.0, log=lambda s: None)
        dt = time.time() - t
        solved += path is not None
        secs += dt
        per.append(f"{sd}:{'ok' if path else '--'} {dt:.0f}s")
    print(f"{name}: {solved}/{len(seeds)} solved, {secs:.0f}s total   [{', '.join(per)}]", flush=True)
