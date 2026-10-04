"""Progress of a PPO run from its log (train.py --engine cpp writes <out>_log.jsonl).

    python tools/ppo_stats.py checkpoints/ppo_cpp_white_log.jsonl            # one row per ~2% of the run so far
    python tools/ppo_stats.py checkpoints/ppo_cpp_white_log.jsonl --rows 40

Columns: wall time, games played so far, blinds / ante / win rate over the last games (the log's window),
how far the games of that stretch got (share lost in antes 1-2, 3-5, 6-8, and won), policy entropy, speed.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="+", help="one log, or the logs of a run and its resumptions in order")
    p.add_argument("--rows", type=int, default=25)
    a = p.parse_args()
    rows = []
    for path in a.log:                       # a resumed run starts counting again: carry the totals over
        part = [json.loads(l) for l in open(path) if l.strip()]
        if rows and part:
            base = rows[-1]
            for r in part:
                for k in ("games", "wins", "env_steps", "min"):
                    r[k] = r.get(k, 0) + base.get(k, 0)
        rows += part
    if not rows:
        raise SystemExit("empty log")
    edges = np.unique(np.linspace(0, len(rows), min(a.rows, len(rows)) + 1).astype(int))
    print(f"{'time':>7} {'games':>10} {'steps':>12} {'blinds':>7} {'ante':>5} {'win%':>6} | "
          f"{'a1-2':>5} {'a3-5':>5} {'a6-8':>5} {'won':>5} | {'entropy':>7} {'kl':>7} {'steps/s':>8}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        part = rows[lo:hi]
        r = part[-1]
        h = np.sum([x.get("antes", [0] * 9) for x in part], axis=0).astype(float)
        tot = max(h.sum(), 1.0)
        t = r["min"]
        print(f"{int(t // 60):3d}h{int(t % 60):02d}m {r.get('games', 0):>10,} {r['env_steps']:>12,} "
              f"{r.get('blinds', 0):7.2f} {r.get('ante', 0):5.2f} {r.get('win%', 0):6.2f} | "
              f"{100 * h[0:2].sum() / tot:5.1f} {100 * h[2:5].sum() / tot:5.1f} {100 * h[5:8].sum() / tot:5.1f} "
              f"{100 * h[8] / tot:5.1f} | {np.mean([x['ent'] for x in part]):7.3f} "
              f"{np.mean([x['kl'] for x in part]):7.4f} {r.get('sps', 0):>8,}")
    last = rows[-1]
    best = max(rows, key=lambda x: x.get("blinds", 0))
    print(f"\nnow: {last.get('games', 0):,} games, {last.get('wins', 0):,} wins in total; "
          f"best window {best.get('blinds', 0):.2f} blinds at {best.get('games', 0):,} games")


if __name__ == "__main__":
    main()
