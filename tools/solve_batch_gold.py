"""Beam 32 (the small search) on a batch of seeds: whether each is solved and the blind where the beam dies. Winning lines are saved.

    python tools/solve_batch_gold.py --stake GOLD --seeds 40001-40016
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import solve      # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--stake", default="GOLD")
p.add_argument("--seeds", default="40001-40016")
p.add_argument("--beam", type=int, default=32)
p.add_argument("--out-dir", default="traces/batch_gold")
a = p.parse_args()
lo, hi = [int(x) for x in a.seeds.split("-")]
os.makedirs(a.out_dir, exist_ok=True)
levels, solved, t0 = [], 0, time.time()
for sd in range(lo, hi + 1):
    t = time.time()
    path, info = solve("RED", a.stake, sd, a.beam, 8, 4, 4, 1000, 0.0, 0.0)
    if path:
        solved += 1
        json.dump({"seed": sd, "deck": "RED", "stake": a.stake, "solved": True, "actions": path, "info": info},
                  open(os.path.join(a.out_dir, f"{a.stake}_{sd}.json"), "w"))
    else:
        levels.append(info["level"])
    print(f"seed {sd}: {'SOLVED' if path else 'died at blind %2d' % info['level']}  {time.time() - t:.0f}s", flush=True)
n = hi - lo + 1
print(f"{solved}/{n} solved; mean death blind of the rest {sum(levels) / max(1, len(levels)):.1f}; {time.time() - t0:.0f}s total", flush=True)
