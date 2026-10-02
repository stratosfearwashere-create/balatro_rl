"""CPU seconds per game: 20 no-search graded games on seeds 70001-70020.

    python bench_speed.py            # the agreed bench
    python bench_speed.py --games 5  # quicker
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--start", type=int, default=70001)
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    from balatro_rl.az.train import _make_agent, play_game
    cfg = {"shop_prior": "graded", "shop": {"rule_arcana": True}}
    agent = _make_agent("none", False, 0, cfg, None)
    prof = None
    if a.profile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    t0 = time.process_time()
    w0 = time.perf_counter()
    blinds = 0
    for seed in range(a.start, a.start + a.games):
        _, info = play_game(agent, seed, "RED", "WHITE", explore=False, record=False)
        blinds += info["blinds"]
    cpu = time.process_time() - t0
    wall = time.perf_counter() - w0
    if prof:
        prof.disable()
        import pstats
        pstats.Stats(prof).sort_stats("cumulative").print_stats(45)
    print(f"{a.games} games: {cpu / a.games:.3f} CPU s/game ({wall / a.games:.3f} wall), mean blinds {blinds / a.games:.2f}")


if __name__ == "__main__":
    main()
