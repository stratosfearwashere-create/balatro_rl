"""Solve one game with a search that compares states blind by blind (exact-seed mode: the game is deterministic).

The search keeps a beam of states at the start of a blind (blind select). For each:
  * skip it (for its tag), or play it: the exact round solver (C++ RoundSolver: the cards to play and discard, the deck order and every
    roll are known) finds the lines that clear the blind; each clearing line leads to the shop;
  * in the shop every action is tried (buy, packs and what to pick, vouchers, rerolls, selling, using consumables), layer by layer, with
    duplicate states removed and the weakest dropped; leaving the shop gives a state at the next blind select.
The states at the next blind are ranked by the build's expected best-play score (the engine's own number, in decades) and the money
held, a few of each kind are kept, and the search goes on. No network is used. A line that clears the ante-8 boss is a win.

    python tools/solve_macro.py --seed 100521 --stake WHITE --beam 24 --out traces/macro_100521.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from balatro_rl import env as E                      # noqa: E402
from balatro_rl.engine_env import CppVecEnv          # noqa: E402

COL_BLIND_SELECT, COL_HAND, COL_SHOP, COL_PACK = 4, 5, 6, 7
COL_MONEY, COL_PHI = 17, 205
MONEY_W = 0.03
POT_W = 0.0


def make(n, deck, stake, seed, ante_weight=2.5):
    v = CppVecEnv(n, deck, stake, seed0=seed, threads=0)
    v.set_exact(True)
    v.v.restart()
    v.set_weight_shape(ante_weight, 1.0)
    v.set_marginal(2 if POT_W > 0 else 1)            # the build's expected score (column 205), and its growth potential (209), outside blinds
    return v


def score_of(g):
    """Rank states at the same point in the game: the build's expected best-play score in decades, and the money."""
    phi = np.clip(7.0 * g[:, COL_PHI], 0.0, 5.5)
    return phi + MONEY_W * (g[:, COL_MONEY] * 50.0) + POT_W * 7.0 * g[:, 209]


def signature(obs, i):
    return hash((obs["g"][i].round(3).tobytes(), obs["jid"][i].tobytes(), obs["jf"][i].round(2).tobytes(), obs["cid"][i].tobytes()))


class Macro:
    def __init__(self, deck, stake, seed, beam, ends_per, shop_width, shop_layers, node_cap, log=print):
        self.deck, self.stake, self.seed = deck, stake, seed
        self.W, self.EP, self.SW, self.SL, self.node_cap = beam, ends_per, shop_width, shop_layers, node_cap
        self.log = log
        self.work = make(max(4096, shop_width * 40), deck, stake, seed)
        self.lay_a = make(shop_width, deck, stake, seed)
        self.lay_b = make(shop_width, deck, stake, seed)
        self.cand = make(beam * (ends_per * shop_layers + 2), deck, stake, seed)
        self.cur = make(beam, deck, stake, seed)
        self.win = None
        self.nodes = 0

    # --- running actions on copies
    def child(self, src, i, j, a):
        self.work.copy_many(src, [i], [j])
        act = np.full(self.work.n, -1, dtype=np.int64)
        act[j] = a
        return act

    def expand_blind(self, b, path):
        """All the states at the next blind select reachable from beam state b: (cand slot list is filled), returns [(slot_in_work_copy..)]."""
        out = []                                         # (path, work-slot holding the end state) filled by copying into cand later
        mask = self.cur.obs["mask"][b]
        skip_state = None
        results = []                                     # (end state source env index in `store`, path)
        # skip
        if mask[E.A_SKIP]:
            self.work.copy_many(self.cur, [b], [0])
            act = np.full(self.work.n, -1, dtype=np.int64)
            act[0] = E.A_SKIP
            self.work.step(act)
            if self.work.obs["g"][0][COL_BLIND_SELECT] > 0.5:
                results.append((0, path + [E.A_SKIP]))
                self.stash(results[-1:], [0])
            else:                                        # the tag opened a pack (or a shop): it goes through the shop layers
                skip_state = (0, path + [E.A_SKIP])
        # play it
        self.work.copy_many(self.cur, [b], [1])
        act = np.full(self.work.n, -1, dtype=np.int64)
        act[1] = E.A_SELECT
        self.work.step(act)
        plans = self.work.solve_rounds([1], 3, self.node_cap)[0]
        if not plans and skip_state is None:
            return
        k = len(plans)
        slots = list(range(2, 2 + k))
        if k:
            self.work.copy_many(self.work, [1] * k, slots)
        L = max((len(p) for p in plans), default=0)
        for t in range(L):
            act = np.full(self.work.n, -1, dtype=np.int64)
            for q, p in enumerate(plans):
                if t < len(p):
                    act[slots[q]] = p[t]
            _, done, info = self.work.step(act)
            for q in range(k):
                if t < len(plans[q]) and done[slots[q]]:
                    if info["won"][slots[q]]:
                        self.win = path + [E.A_SELECT] + plans[q][:t + 1]
                        return
        self.nodes += k * L
        # the shop: every action, layer by layer
        shop_paths = [path + [E.A_SELECT] + p for p in plans]
        layer_n = 0
        src_slots = list(slots)
        if skip_state is not None:
            src_slots.append(skip_state[0])
            shop_paths = shop_paths + [skip_state[1]]
        self.lay_a.copy_many(self.work, src_slots, list(range(len(src_slots))))
        layer_n = len(src_slots)
        layer_paths = shop_paths
        ends = []                                        # (score, path) with the state parked in the cand store
        cur_lay, nxt_lay = self.lay_a, self.lay_b
        for depth in range(self.SL):
            if layer_n == 0:
                break
            mask_l = np.array(cur_lay.obs["mask"][:layer_n], copy=True)
            pairs = [(s, int(a)) for s in range(layer_n) for a in np.flatnonzero(mask_l[s])]
            if len(pairs) > self.work.n:
                pairs = pairs[:self.work.n]
            src = [s for s, _ in pairs]
            dst = list(range(len(pairs)))
            self.work.copy_many(cur_lay, src, dst)
            act = np.full(self.work.n, -1, dtype=np.int64)
            for j, (s, a) in enumerate(pairs):
                act[j] = a
            _, done, info = self.work.step(act)
            self.nodes += len(pairs)
            g = np.array(self.work.obs["g"][:len(pairs)], copy=True)
            sc = score_of(g)
            in_bs = g[:, COL_BLIND_SELECT] > 0.5
            obs_np = {k: np.array(self.work.obs[k][:len(pairs)], copy=True) for k in ("g", "jid", "jf", "cid")}
            seen = set()
            cont, fin = [], []
            for j in np.argsort(-sc):
                if done[j]:
                    continue
                sg = signature(obs_np, j)
                if sg in seen:
                    continue
                seen.add(sg)
                (fin if in_bs[j] else cont).append(j)
            fin = fin[:self.EP]
            for j in fin:
                ends.append((float(sc[j]), j, pairs[j], layer_paths))
            self.stash_from_work(fin, [layer_paths[pairs[j][0]] + [pairs[j][1]] for j in fin], sc)
            cont = cont[:self.SW]
            nxt_lay.copy_many(self.work, cont, list(range(len(cont))))
            layer_paths = [layer_paths[pairs[j][0]] + [pairs[j][1]] for j in cont]
            layer_n = len(cont)
            cur_lay, nxt_lay = nxt_lay, cur_lay

    # --- candidates for the next level
    def stash(self, results, work_slots):
        for (_, path), slot in zip(results, work_slots):
            g = np.array(self.work.obs["g"][slot:slot + 1])
            self.cands.append((float(score_of(g)[0]), path, self._park(slot)))

    def stash_from_work(self, js, paths, sc):
        for j, path in zip(js, paths):
            self.cands.append((float(sc[j]), path, self._park(j)))

    def _park(self, slot):
        k = len(self.cands)
        if k >= self.cand.n:
            return -1
        self.cand.copy_many(self.work, [slot], [k])
        return k

    def run(self, max_levels=40):
        root = make(1, self.deck, self.stake, self.seed)
        self.cur.copy_many(root, [0], [0])
        paths = [[]]
        n = 1
        t0 = time.time()
        for level in range(max_levels):
            self.cands = []
            for b in range(n):
                self.expand_blind(b, paths[b])
                if self.win:
                    return self.win, {"level": level, "nodes": self.nodes, "secs": time.time() - t0}
            if not self.cands:
                self.dead_paths = paths[:4]
                return None, {"level": level, "nodes": self.nodes, "secs": time.time() - t0, "dead_at_blind": level}
            ranked = sorted([c for c in self.cands if c[2] >= 0], key=lambda c: -c[0])
            seen, keep = set(), []
            obs_c = {k: np.array(self.cand.obs[k][:len(self.cands)], copy=True) for k in ("g", "jid", "jf", "cid")}
            for c in ranked:
                sg = signature(obs_c, c[2])
                if sg in seen:
                    continue
                seen.add(sg)
                keep.append(c)
                if len(keep) >= self.W:
                    break
            self.cur.copy_many(self.cand, [c[2] for c in keep], list(range(len(keep))))
            paths = [c[1] for c in keep]
            n = len(keep)
            self.log(f"  blind {level + 1}: {len(self.cands)} states -> {n} kept, best score {keep[0][0]:.2f}, {self.nodes} nodes, {time.time() - t0:.0f}s")
        return None, {"level": max_levels, "nodes": self.nodes, "secs": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=100521)
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="WHITE")
    p.add_argument("--beam", type=int, default=24)
    p.add_argument("--ends", type=int, default=6, help="end states kept per shop")
    p.add_argument("--shop-width", type=int, default=48)
    p.add_argument("--shop-layers", type=int, default=7)
    p.add_argument("--node-cap", type=int, default=400)
    p.add_argument("--money-weight", type=float, default=0.03, help="score = decades of expected best-play score + this * money")
    p.add_argument("--potential-weight", type=float, default=0.0, help="also rank by the growth potential of growing jokers (decades)")
    p.add_argument("--out", default="traces/macro.json")
    a = p.parse_args()
    global MONEY_W, POT_W
    MONEY_W = a.money_weight
    POT_W = a.potential_weight
    print(f"solving seed {a.seed}: {a.deck} deck, {a.stake} stake, beam {a.beam}", flush=True)
    m = Macro(a.deck, a.stake, a.seed, a.beam, a.ends, a.shop_width, a.shop_layers, a.node_cap, log=lambda s: print(s, flush=True))
    path, info = m.run()
    print("SOLVED" if path else "no win found", info, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"seed": a.seed, "deck": a.deck, "stake": a.stake, "solved": path is not None, "actions": path, "info": info,
               "dead_paths": getattr(m, "dead_paths", None)}, open(a.out, "w"))


if __name__ == "__main__":
    main()
