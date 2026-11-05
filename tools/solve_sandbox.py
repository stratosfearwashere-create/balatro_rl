"""The exact search (the novelty beam of solve_novel.py) on the late-game sandbox: how many seeds can be won at all, and with what?

The game is the sandbox of train.py (--start-ante 8 in the shop with the start money, a limit on planet hand level-ups, packs restocked, banned jokers); the search uses the
state copies and the exact round solver, so a seed is "possible" if some line of play wins it. A winning line is replayed from the seed and its end state is printed.

    python tools/solve_sandbox.py --seeds 70001-70008 --stake GOLD --beam 32
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_novel import pick                                            # noqa: E402
from balatro_rl.engine_env import CppEnv, CppVecEnv                     # noqa: E402
from balatro_rl.sim.hands import HAND_NAMES                             # noqa: E402
from balatro_rl.sim.jokers import JOKERS                                # noqa: E402

JKEYS = list(JOKERS)
BIAS = {"arcana": 3.0, "celestial": 2.0, "standard": 3.0, "buffoon": 1.0, "spectral": 1.5}
KINDS = ("arcana", "celestial", "standard", "buffoon", "spectral")


def make_sandbox(n, a, seed, marginal=1):
    v = CppVecEnv(n, a.deck, a.stake, seed0=seed, threads=0)
    v.set_start(0.0, 0.0, 1.0, a.money)
    v.set_level_cap(1.0, a.cap_n, 99)
    v.set_sandbox(a.level_cap_mode, a.pack_restock, [BIAS[k] for k in KINDS])
    if a.ban and a.ban.lower() != "none":
        keys = list(JOKERS)
        v.set_ban([keys.index(k) for k in a.ban.split(",")], 1.0)
    v.set_start_ante(a.start_ante)
    v.v.restart()
    v.set_weight_shape(2.5, 1.0)
    v.set_marginal(marginal)
    return v


def solve(a, seed, beam, elite, floor, ends_per, sw, sl, node_cap, scale_w, max_levels=12):
    marg = 2 if scale_w > 0 else 1
    cur = make_sandbox(beam, a, seed, marg)
    dest = make_sandbox(beam * ends_per, a, seed, marg)
    root = make_sandbox(1, a, seed, marg)
    cur.copy_many(root, [0], [0])
    paths, n, t0 = [[]], 1, time.time()
    last = []
    for level in range(max_levels):
        res = cur.expand_level(list(range(n)), dest, ends_per, sw, sl, node_cap, 0.0, 0.0, scale_w, [], 0.0)
        pool = []
        for b, (win, wpath, cands) in enumerate(res):
            if win:
                return paths[b] + list(wpath), {"level": level, "secs": time.time() - t0}
            for path, score, sig, slot, jk in cands:
                pool.append((score, sig, paths[b] + list(path), slot, jk))
        if not pool:
            return None, {"level": level, "secs": time.time() - t0, "dead": True, "last": last}
        keep = pick(pool, beam, elite, floor)
        last = [(round(c[0], 2), [JKEYS[j] for j in c[4]], 437 in c[2][:], len(c[2])) for c in sorted(keep, key=lambda c: -c[0])[:4]]
        cur.copy_many(dest, [c[3] for c in keep], list(range(len(keep))))
        paths, n = [c[2] for c in keep], len(keep)
    return None, {"level": max_levels, "secs": time.time() - t0}


def replay(a, seed, actions):
    e = CppEnv(a.deck, a.stake)
    e.e.start_ante, e.e.start_money_p, e.e.start_money = a.start_ante, 1.0, a.money
    e.e.level_cap_p, e.e.level_cap_n, e.e.level_cap_ante, e.e.level_cap_mode = 1.0, a.cap_n, 99, a.level_cap_mode
    e.e.pack_restock = a.pack_restock
    e.e.pack_bias = [BIAS[k] for k in KINDS]
    if a.ban and a.ban.lower() != "none":
        keys = list(JOKERS)
        e.e.banned = [keys.index(k) for k in a.ban.split(",")]
        e.e.ban_p = 1.0
    e.reset(seed)
    for x in actions:
        _, _, done, _ = e.step(int(x))
        if done:
            break
    return e.game.dump()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="70001-70008")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--start-ante", type=int, default=8)
    p.add_argument("--money", type=int, default=1000)
    p.add_argument("--cap-n", type=int, default=4)
    p.add_argument("--level-cap-mode", type=int, default=1)
    p.add_argument("--pack-restock", type=int, default=6)
    p.add_argument("--ban", default="bootstraps,bull")
    p.add_argument("--beam", type=int, default=32)
    p.add_argument("--elite", type=int, default=16)
    p.add_argument("--floor", type=float, default=0.5)
    p.add_argument("--ends", type=int, default=8)
    p.add_argument("--shop-width", type=int, default=8)
    p.add_argument("--shop-layers", type=int, default=8)
    p.add_argument("--node-cap", type=int, default=1500)
    p.add_argument("--scale-weight", type=float, default=1.0)
    p.add_argument("--out", default="traces/sandbox_beam")
    a = p.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")] if "-" in a.seeds else [int(a.seeds)] * 2
    os.makedirs(a.out, exist_ok=True)
    solved = 0
    for sd in range(lo, hi + 1):
        t = time.time()
        path, info = solve(a, sd, a.beam, a.elite, a.floor, a.ends, a.shop_width, a.shop_layers, a.node_cap, a.scale_weight)
        if path:
            solved += 1
            d = replay(a, sd, path)
            json.dump({"seed": sd, "stake": a.stake, "solved": True, "actions": path}, open(os.path.join(a.out, f"{a.stake}_{sd}.json"), "w"))
            lv = {n: l for n, l in zip(HAND_NAMES, d["hand_levels"]) if l > 1}
            print(f"seed {sd}: SOLVED ({time.time() - t:.0f}s, {len(path)} actions) -> {d['state']} | jokers {[(j['key'], j.get('edition', '')) for j in d['jokers']]} | "
                  f"levels {lv} | planets {d.get('planet_uses')} | tarots {d.get('tarots_used')} | money {d['money']} | deck {len(d['full_deck'])} | boss {d['boss']}", flush=True)
        else:
            print(f"seed {sd}: not solved (died at level {info['level']}, {time.time() - t:.0f}s); the best lines before it: " + " | ".join(
                f"score {x[0]} jokers {x[1]} skip={x[2]}" for x in info.get("last", [])[:3]), flush=True)
    print(f"{solved}/{hi - lo + 1} solved", flush=True)


if __name__ == "__main__":
    main()
