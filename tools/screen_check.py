"""Does the feasibility screen (seed_report.screen) predict what the search finds? Scores seeds, runs the beam on each, and compares the score with the level the beam reached and whether it won.

    python tools/screen_check.py --seeds 43001-43100
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from seed_report import scan, screen        # noqa: E402
from solve_novel import solve               # noqa: E402


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="43001-43100")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--beam", type=int, default=32)
    a = p.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")]
    rows, t0 = [], time.time()
    for sd in range(lo, hi + 1):
        sc, parts, _ = screen(scan(sd, "RED", a.stake, antes=4, pairs=6))
        path, info = solve("RED", a.stake, sd, a.beam, a.beam // 2, 0.5, 8, 4, 4, 1000, 0.0, 0.0, 1.0)
        rows.append((sd, sc, parts, 24 if path else info["level"], bool(path)))
        if (sd - lo) % 20 == 19:
            print(f"  {sd - lo + 1} seeds, {time.time() - t0:.0f}s", flush=True)
    score = np.array([r[1] for r in rows]); level = np.array([r[3] for r in rows]); won = np.array([r[4] for r in rows])
    print(f"{len(rows)} seeds; beam {a.beam} solved {int(won.sum())}; mean level {level.mean():.1f}")
    print(f"Spearman(screen score, level reached) = {spearman(score, level):+.3f}")
    for k in ("xmult", "scaler", "copy", "econ", "voucher", "early"):
        print(f"   {k:8s} {spearman([r[2][k] for r in rows], level):+.3f}")
    q = np.quantile(score, [0.25, 0.5, 0.75])
    print("score band          seeds  solved  mean level")
    for name, m in (("lowest quarter", score <= q[0]), ("second", (score > q[0]) & (score <= q[1])), ("third", (score > q[1]) & (score <= q[2])), ("top quarter", score > q[2])):
        print(f"  {name:16s} {int(m.sum()):6d} {int(won[m].sum()):7d} {level[m].mean():10.1f}")


if __name__ == "__main__":
    main()
