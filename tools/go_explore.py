"""Go-Explore on one exact seed.

The engine is deterministic, so "returning" to a state is a copy (VecEnv.copy_many), not a replay. The archive holds one state per cell; a cell is a level (the
number of blinds played or skipped) and the set of jokers held. Each iteration picks a batch of cells (favouring the less visited, the better scoring of their level
and the deeper), returns to them, and expands each by one level (skip or play the next blind with the exact round solver, then the shop search: VecEnv.expand_level).
Every child lands in the archive if its cell is new, or replaces the cell's state if it scores higher. The search stops at the first win.

    python tools/go_explore.py --seed 42003 --stake GOLD --minutes 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import make      # noqa: E402


def replay(deck, stake, seed, actions):
    """-> (won, blinds beaten) of the game the action list plays out (the real step function, from the seed)."""
    from balatro_rl.engine_env import CppEnv
    e = CppEnv(deck, stake)
    e.e.exact = True
    e.reset(int(seed))
    for a in actions:
        _, _, done, _ = e.step(int(a))
        if done:
            break
    d = e.game.dump()
    return bool(d.get("won", False)), int(d.get("blinds_beaten", d.get("furthest_blind", 0)))


def replay_state(deck, stake, seed, actions):
    """The game state the action list reaches: its dump (ante, blinds beaten, jokers, ...) and the number of actions that were legal."""
    from balatro_rl.engine_env import CppEnv
    e = CppEnv(deck, stake)
    e.e.exact = True
    e.reset(int(seed))
    n = 0
    for a in actions:
        _, _, done, _ = e.step(int(a))
        n += 1
        if done:
            break
    d = e.game.dump()
    keep = ("ante", "blind", "blinds_beaten", "furthest_blind", "won", "state", "money", "jokers", "game_over", "round")
    return {"actions": n, **{k: d[k] for k in keep if k in d}}


class Archive:
    def __init__(self, env, cap):
        self.env, self.cap = env, cap
        self.cell = {}                                   # key -> slot
        self.key = [None] * cap
        self.level = np.zeros(cap, dtype=np.int32)
        self.score = np.full(cap, -np.inf)
        self.visits = np.zeros(cap)
        self.kids = np.zeros(cap)                        # new cells found from this one
        self.path = [None] * cap
        self.n = 0

    def slot_for(self, key):
        if key in self.cell:
            return self.cell[key], False
        if self.n < self.cap:
            s = self.n
            self.n += 1
        else:                                            # full: replace the most visited, lowest scoring cell that is not the root
            v = np.where(np.arange(self.cap) == 0, -np.inf, self.visits - 1e-3 * self.score)
            s = int(np.argmax(v))
            del self.cell[self.key[s]]
        self.cell[key] = s
        self.key[s] = key
        self.visits[s] = 0
        self.kids[s] = 0
        return s, True

    def weights(self, power=0.5, depth=0.3):
        n = self.n
        lv, sc, vi = self.level[:n], self.score[:n], self.visits[:n]
        w = (vi + 1.0) ** -power * (1.0 + depth * lv) * (1.0 + 0.5 * self.kids[:n]) ** 0.5
        pct = np.zeros(n)
        for L in np.unique(lv):                          # a cell's score as a rank within its level
            m = np.flatnonzero(lv == L)
            pct[m] = (np.argsort(np.argsort(sc[m])) + 1.0) / len(m)
        return w * (0.25 + pct)


def go_explore(deck, stake, seed, minutes, batch, ends, sw, sl, node_cap, scale_w, cap, log, seed_rng=1, judge_path="", phi_floor=-40.0):
    root = make(1, deck, stake, seed, 2 if scale_w > 0 else 1)
    marg = 2 if scale_w > 0 else 1
    arch = Archive(make(cap, deck, stake, seed, marg), cap)
    cur = make(batch, deck, stake, seed, marg)
    dest = make(batch * ends, deck, stake, seed, marg)
    arch.env.copy_many(root, [0], [0])
    s0, _ = arch.slot_for((0, frozenset()))
    arch.path[s0], arch.score[s0], arch.level[s0] = [], 0.0, 0
    arch.n = 1
    rng = np.random.default_rng(seed_rng)
    pot = projenv = None
    if judge_path:                                       # the judge potential (potential.py, --phi-full) scores every child: that is the heuristic
        import torch
        from balatro_rl import judge as J
        from balatro_rl.potential import JudgePotential
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        projenv = make(batch * ends, deck, stake, seed, 1)
        pot = JudgePotential(None, None, J.load(judge_path, dev), None, dev, phi_floor, 0.5, True)
    t0, it, expansions = time.time(), 0, 0
    tm = {"select": 0.0, "restore": 0.0, "expand (C++)": 0.0, "archive update": 0.0}
    best_level = 0
    while time.time() - t0 < minutes * 60:
        it += 1
        t1 = time.time()
        w = arch.weights()
        k = min(batch, arch.n)
        pick = np.argsort(-(np.log(np.maximum(w, 1e-12)) + rng.gumbel(size=arch.n)))[:k]           # sample without replacement, proportional to w
        t2 = time.time()
        cur.copy_many(arch.env, pick.tolist(), list(range(k)))
        t3 = time.time()
        res = cur.expand_level(list(range(k)), dest, ends, sw, sl, node_cap, 0.0, 0.0, scale_w, [], 0.0)
        if pot is not None:                              # replace the solver's score of each child by the potential of the child's state
            slots_all = sorted({c[3] for _, _, cands in res for c in cands})
            if slots_all:
                ph = {}
                for c0 in range(0, len(slots_all), projenv.n):
                    ch = slots_all[c0:c0 + projenv.n]
                    ph.update(zip(ch, pot.phi(ch, src=dest, proj=projenv)))
                res = [(win, wp, [(c[0], float(ph[c[3]]), c[2], c[3], c[4]) for c in cands]) for win, wp, cands in res]
        t4 = time.time()
        expansions += k
        new_cells = 0
        par = [(int(q), arch.key[int(q)], arch.path[int(q)], int(arch.level[int(q)])) for q in pick]       # (an eviction below may reuse a parent's slot)
        for b, (win, wpath, cands) in enumerate(res):
            p, pkey, ppath, plevel = par[b]
            if arch.key[p] == pkey:
                arch.visits[p] += 1
            if win:
                return ppath + list(wpath), {"iter": it, "expansions": expansions, "cells": arch.n, "secs": time.time() - t0, "time_split": {k: round(v, 1) for k, v in tm.items()}}
            if not cands and arch.key[p] == pkey:
                arch.visits[p] += 50                       # a dead end: stop choosing it
            for path, score, sig, slot, jk in cands:
                key = (plevel + 1, frozenset(int(x) for x in jk))
                old = arch.cell.get(key)
                if old is not None and score <= arch.score[old]:
                    continue
                s, new = arch.slot_for(key)
                arch.env.copy_many(dest, [slot], [s])
                arch.path[s], arch.score[s], arch.level[s] = ppath + list(path), float(score), plevel + 1
                if new:
                    new_cells += 1
                    if arch.key[p] == pkey:
                        arch.kids[p] += 1
        t5 = time.time()
        tm["select"] += t2 - t1; tm["restore"] += t3 - t2; tm["expand (C++)"] += t4 - t3; tm["archive update"] += t5 - t4
        best_level = max(best_level, int(arch.level[:arch.n].max()))
        if it % 10 == 0:
            lv = arch.level[:arch.n]
            top = ", ".join(f"{L}:{int((lv == L).sum())}" for L in range(max(1, best_level - 4), best_level + 1))
            log(f"  it {it:4d} {time.time() - t0:5.0f}s  expansions {expansions}  cells {arch.n}  deepest blind {best_level}  cells at the deepest levels {top}  new this it {new_cells}")
    return None, {"iter": it, "expansions": expansions, "cells": arch.n, "deepest": best_level, "secs": time.time() - t0, "time_split": {k: round(v, 1) for k, v in tm.items()},
                  "best_path": arch.path[int(np.argmax(arch.level[:arch.n]))]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42003)
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ends", type=int, default=8)
    p.add_argument("--shop-width", type=int, default=4)
    p.add_argument("--shop-layers", type=int, default=4)
    p.add_argument("--node-cap", type=int, default=1000)
    p.add_argument("--scale-weight", type=float, default=1.0)
    p.add_argument("--cap", type=int, default=12000)
    p.add_argument("--judge", default="", help="checkpoint of the judge (judge.py): score the states with the judge potential instead of the solver's score")
    p.add_argument("--out-dir", default="traces/go_explore")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"go-explore on seed {a.seed}: {a.deck} deck, {a.stake} stake, {a.minutes} min", flush=True)
    path, info = go_explore(a.deck, a.stake, a.seed, a.minutes, a.batch, a.ends, a.shop_width, a.shop_layers, a.node_cap, a.scale_weight, a.cap,
                            lambda s: print(s, flush=True), judge_path=a.judge)
    out = {"seed": a.seed, "deck": a.deck, "stake": a.stake, "solved": path is not None, "actions": path, "info": info}
    if path:
        won, blinds = replay(a.deck, a.stake, a.seed, path)
        out["replay"] = {"won": won, "blinds": blinds}
        print("SOLVED", info, "replay:", out["replay"], flush=True)
    else:
        bp = info.pop("best_path", None)
        print("no win", info, flush=True)
        if bp is not None:
            out["actions"] = bp
            out["replay"] = replay_state(a.deck, a.stake, a.seed, bp)
            print("deepest line, replayed:", out["replay"], flush=True)
    json.dump(out, open(os.path.join(a.out_dir, f"{a.stake}_{a.seed}.json"), "w"))


if __name__ == "__main__":
    main()
