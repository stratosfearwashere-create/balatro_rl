"""Solve one game, fast: the level expansion (skip or play the blind with the exact round solver, then the shop search) runs in C++ on all
cores (VecEnv.expand_level); this driver keeps the beam, removes duplicates and widens the beam only when a narrower one fails.

    python tools/solve_fast.py --seed 100521 --stake WHITE --out traces/fast_100521.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from balatro_rl.engine_env import CppVecEnv          # noqa: E402


START_MONEY = 0                                        # extra dollars every game starts with (a head start for studying what the shops can support)


def make(n, deck, stake, seed, marginal):
    v = CppVecEnv(n, deck, stake, seed0=seed, threads=0)
    v.set_exact(True)
    if START_MONEY > 0:
        v.set_start(0.0, 0.0, 1.0, int(START_MONEY))
    v.v.restart()
    v.set_weight_shape(2.5, 1.0)
    v.set_marginal(marginal)
    return v


def solve(deck, stake, seed, beam, ends_per, sw, sl, node_cap, money_w, pot_w, max_levels=40, log=None, scale_w=0.0, target=(), target_w=0.0):
    marginal = 2 if (pot_w > 0 or scale_w > 0) else 1
    cur = make(beam, deck, stake, seed, marginal)
    dest = make(beam * ends_per, deck, stake, seed, marginal)
    root = make(1, deck, stake, seed, marginal)
    cur.copy_many(root, [0], [0])
    paths, n, t0 = [[]], 1, time.time()
    for level in range(max_levels):
        res = cur.expand_level(list(range(n)), dest, ends_per, sw, sl, node_cap, money_w, pot_w, scale_w, target, target_w)
        pool = []
        for b, (win, wpath, cands) in enumerate(res):
            if win:
                return paths[b] + list(wpath), {"level": level, "secs": time.time() - t0}
            for path, score, sig, slot, _jk in cands:
                pool.append((score, sig, paths[b] + list(path), slot))
        if not pool:
            return None, {"level": level, "secs": time.time() - t0, "dead": True, "dead_paths": paths[:3]}
        pool.sort(key=lambda c: -c[0])
        seen, keep = set(), []
        for c in pool:
            if c[1] in seen:
                continue
            seen.add(c[1])
            keep.append(c)
            if len(keep) >= beam:
                break
        cur.copy_many(dest, [c[3] for c in keep], list(range(len(keep))))
        paths, n = [c[2] for c in keep], len(keep)
        if log:
            log(f"    blind {level + 1}: {len(pool)} candidates -> {n} kept, best {keep[0][0]:.2f}, {time.time() - t0:.1f}s")
    return None, {"level": max_levels, "secs": time.time() - t0}


def solve_adaptive(deck, stake, seed, widths, ends_per, sw, sl, node_cap, money_w, pot_w, log=print):
    t0 = time.time()
    for w in widths:
        log(f"  beam {w}")
        path, info = solve(deck, stake, seed, w, ends_per, sw, sl, node_cap, money_w, pot_w, log=None)
        info["beam"] = w
        log(f"    -> {'SOLVED' if path else 'no win'} at level {info['level']}, {info['secs']:.1f}s")
        if path:
            info["total_secs"] = time.time() - t0
            return path, info
    return None, {"beam": widths[-1], "total_secs": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=100521)
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="WHITE")
    p.add_argument("--widths", default="8,16,32,64")
    p.add_argument("--ends", type=int, default=8)
    p.add_argument("--shop-width", type=int, default=32)
    p.add_argument("--shop-layers", type=int, default=7)
    p.add_argument("--node-cap", type=int, default=1000)
    p.add_argument("--money-weight", type=float, default=0.01)
    p.add_argument("--potential-weight", type=float, default=0.0)
    p.add_argument("--out", default="traces/fast.json")
    a = p.parse_args()
    print(f"solving seed {a.seed}: {a.deck} deck, {a.stake} stake", flush=True)
    path, info = solve_adaptive(a.deck, a.stake, a.seed, [int(x) for x in a.widths.split(",")], a.ends, a.shop_width, a.shop_layers,
                                a.node_cap, a.money_weight, a.potential_weight, log=lambda s: print(s, flush=True))
    print("SOLVED" if path else "no win found", info, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"seed": a.seed, "deck": a.deck, "stake": a.stake, "solved": path is not None, "actions": path, "info": info},
              open(a.out, "w"))


if __name__ == "__main__":
    main()
