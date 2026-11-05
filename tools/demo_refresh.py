"""Rebuilds the demonstrations file every --every minutes from the lines the solvers keep adding (a trainer with --demo reloads it when it changes).

    python tools/demo_refresh.py --every 15 --min-level 17 --out checkpoints/demos_gold.npz
"""
import argparse
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--every", type=float, default=15)
p.add_argument("--min-level", type=int, default=17)
p.add_argument("--lines", default="traces/demo_lines")
p.add_argument("--out", default="checkpoints/demos_gold.npz")
a = p.parse_args()
while True:
    r = subprocess.run([sys.executable, "tools/make_demos.py", "--lines", a.lines, "--min-level", str(a.min_level), "--out", a.out], capture_output=True, text=True)
    print(time.strftime("%H:%M"), (r.stdout + r.stderr).strip().splitlines()[-1:], flush=True)
    time.sleep(a.every * 60)
