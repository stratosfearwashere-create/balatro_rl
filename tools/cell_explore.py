"""Cell-archive exploration of one exact seed: a Go-Explore whose steps are a shop planner and a blind solver.

    simulator   Sim: the engine's VecEnv as a state store (clone = VecEnv.copy_many, step, legal actions, phase, terminal flag, state_hash). The exact-seed engine is deterministic and the RNG
                stream positions are part of the state, so the same actions from the same state always give the same state; state_hash (every field and stream position, plus the money) tells
                exact duplicates, and is separate from the cell key.
    blind       BlindSolver: from the state at a blind select, the outcomes (action list, money, hands and discards used, resulting state key) of the exact round solver's best plans, a Pareto front
                per resulting state (more money, fewer hands and discards used) capped at K; memoised on the state hash; empty when the blind cannot be cleared (a dead end).
    shop        ShopPlanner: online policies that make shop action sequences: a greedy economy policy, biased-random legal actions (jokers first in the early antes), and targeted plans from the seed's
                oracle ("reroll until joker X shows, buy it"). Canonical order: within a stretch between rerolls buys go in increasing slot order, and once a joker reorder is made no more buying.
    cell        cell_key(info): versioned (CELL_VERSION): ante, blind, the sorted joker multiset with editions, vouchers, bucketed hand levels, bucketed money, deck statistics.
    archive     Archive of Entry(actions, snapshot, quality, visits, children_found, last_new_step, value_pred): the action log is the truth, the snapshot (a slot of a bank VecEnv) a cache that is
                rebuilt by replay when evicted. consider() inserts a new cell, or replaces a cell's entry by a better quality one.
    selector    weight = a / sqrt(visits + 1) + b * children_found / (visits + 1) + c * depth + d * value_pred; cells are sampled in a batch.
    explorer    pick a batch of cells, restore, sample shop plans, solve the blind after each, consider every resulting state; stop at a win, which is replayed through the single-game engine.

    python tools/cell_explore.py --seed 42010 --minutes 10 --batch 8 --plans 8 --k 3
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("BALATRO_DEV_ENGINE", "")
from solve_fast import make                                                      # noqa: E402
from astar_bound import ITEM2KEY, KEYS, obtainable                                      # noqa: E402
from seed_report import role_of                                                   # noqa: E402
from balatro_rl.env import swap_role                                              # noqa: E402
from balatro_rl.env import (A_BUY, A_BUY_PACK, A_LEAVE, A_PICK, A_PSKIP, A_REPLACE, A_REROLL, A_SELECT, A_SELL_C, A_SELL_J, A_SKIP, A_SWAP, A_SWAP_END, A_USE_C, A_VOUCHER,   # noqa: E402
                            MAX_SHOP, N_SUB, item_id)
from balatro_rl.sim.items import PLANETS                                          # noqa: E402

CELL_VERSION = 2
PLANET_ITEMS = {item_id("c_" + n) for n in PLANETS}
ROLE_PRI = {"copy": 4, "xmult": 3, "scaler": 2, "econ": 1, "plain": 0}


# --------------------------------------------------------------------------- 1. the simulator interface
class Sim:
    """n game states of one exact seed. A state is a slot; a game that ends is flagged in `dead` (the engine restarts that slot on another seed: the slot is not used again until cloned over)."""

    def __init__(self, n, deck, stake, seed):
        self.n, self.v = n, make(n, deck, stake, seed, 0)
        self.v.set_auto_round(False, 1.5, 0)
        self.dead = np.zeros(n, dtype=bool)
        self.won = np.zeros(n, dtype=bool)

    def clone(self, src, si, di):
        self.v.copy_many(src.v, list(si), list(di))
        for s, d in zip(si, di):
            self.dead[d], self.won[d] = src.dead[s], src.won[s]

    def step(self, actions):
        _, d, info = self.v.step(actions)
        for i in np.flatnonzero(d):
            if actions[i] >= 0:
                self.dead[i] = True
                self.won[i] = bool(info["won"][i])
        return d

    def mask(self):
        return np.asarray(self.v.obs["mask"])

    @staticmethod
    def phase_of(m):
        if m[A_SELECT]:
            return "select"
        if m[A_PSKIP]:
            return "pack"
        if m[A_LEAVE]:
            return "shop"
        return "round" if m[:A_SELECT].any() else "none"

    def phase(self, i):
        return "end" if self.dead[i] else self.phase_of(self.mask()[i])

    def legal_actions(self, i):
        return np.flatnonzero(self.mask()[i])

    def is_terminal(self, i):
        return bool(self.dead[i])

    def state_hash(self, idx):
        """Every field of the game and the RNG stream positions (Env::state_key) with the money: equal hashes = the same state."""
        k, m = self.v.state_keys(idx)
        return [int((int(a) * 1000003 + int(b)) & 0xFFFFFFFFFFFFFFFF) for a, b in zip(k, m)], [int(a) for a in k], [int(b) for b in m]

    def info(self, idx):
        return self.v.cell_info(idx)


def apply_actions(sim, slots, logs):
    """The action logs applied to the games at `slots` in lockstep. -> the set of slots that ended (lost or won; sim.won says which)."""
    pos = [0] * len(slots)
    while True:
        act = np.full(sim.n, -1, dtype=np.int64)
        go = False
        for k, s in enumerate(slots):
            if sim.dead[s] or pos[k] >= len(logs[k]):
                continue
            act[s] = logs[k][pos[k]]
            pos[k] += 1
            go = True
        if not go:
            break
        sim.step(act)
    return {s for s in slots if sim.dead[s]}


# --------------------------------------------------------------------------- 4. the cell function (versioned)
def _bucket_money(m):
    if m < 25:
        return m // 5                         # 0..4: the interest steps
    return 5 + min(6, (m - 25) // 15)         # 5..11


def _bucket_level(lv):
    return 0 if lv <= 1 else min(6, 1 + (lv - 2) // 2)


def parse_info(v):
    """The integers of Sim.info -> a dict."""
    ante, blind, state, money, vouchers, jslots, ncons, nj = v[:8]
    jokers = [(v[8 + 3 * i], v[9 + 3 * i], v[10 + 3 * i]) for i in range(nj)]
    o = 8 + 3 * nj
    levels = v[o:o + 12]
    deck = v[o + 12:o + 12 + 1 + 12 + 8 + 8]
    return dict(ante=ante, blind=blind, state=state, money=money, vouchers=vouchers, jslots=jslots, ncons=ncons, jokers=jokers, levels=levels, deck=deck)


def cell_key(inf):
    """Version 2: ante, blind index, the (joker, edition, stickers) multiset (in the real order when a joker acts on its neighbour), vouchers, the hand levels bucketed, money bucketed, the deck's size and enhancement / seal / edition counts."""
    jk = inf["jokers"]
    if any(swap_role(KEYS[j[0]]) == 3 for j in jk):                                  # a joker that acts on its neighbour (Blueprint, Brainstorm): the order is the player's, and it is part of the cell
        jk = tuple(jk)
    else:                                                                            # otherwise the game scores them in its own order (additive mult first, multipliers last): the set is the cell
        jk = tuple(sorted(jk))
    return (CELL_VERSION, inf["ante"], inf["blind"], jk, inf["vouchers"], tuple(_bucket_level(x) for x in inf["levels"]), _bucket_money(inf["money"]),
            tuple(inf["deck"]))


def depth_of(inf):
    return 3 * (inf["ante"] - 1) + inf["blind"]


def quality_of(inf):
    """Orders states within a cell: more money, then higher hand levels, then more jokers' worth of slots free."""
    return (inf["money"], sum(inf["levels"]), -len(inf["jokers"]))


# --------------------------------------------------------------------------- 5. the archive
@dataclass
class Entry:
    actions: list
    snapshot: int | None                    # a slot of the bank (a cache: rebuilt from `actions` when None)
    quality: tuple
    key: tuple = ()
    hash: int = 0
    depth: int = 0
    visits: int = 0
    children_found: int = 0
    last_new_step: int = 0
    value_pred: float = 0.0
    power: float = 0.0                      # the best share of the next boss's chip target the round solver reaches with this build (exact, not learned)


class Archive:
    def __init__(self, bank_size):
        self.cells = {}
        self.seen = {}                      # state hash -> True (exact duplicates are not looked at twice)
        self.bank_free = list(range(bank_size - 1, -1, -1))
        self.bank_owner = {}                # bank slot -> entry
        self.best_depth, self.best_entry = -1, None
        self.stats = dict(considered=0, dup_hash=0, new_cell=0, replaced=0, worse=0, evicted=0)

    def _snap_slot(self, rng):
        if self.bank_free:
            return self.bank_free.pop()
        owners = list(self.bank_owner.items())                                       # evict the snapshot of the most visited entry of a sample (it is rebuilt from its log when needed)
        pick = max((owners[i] for i in rng.integers(0, len(owners), size=min(32, len(owners)))), key=lambda t: t[1].visits)
        slot, e = pick
        e.snapshot = None
        self.stats["evicted"] += 1
        del self.bank_owner[slot]
        return slot

    FRONT = 3

    def is_dup(self, hsh):
        return hsh in self.seen

    def consider(self, sim, slot, bank, actions, parent, step, rng, hsh, inf, power=0.0):
        """The game at `slot` of `sim`, reached by `actions` from the start. A cell keeps a small front of states, none dominated in (money, power). -> True if its cell is new."""
        self.stats["considered"] += 1
        if hsh in self.seen:
            self.stats["dup_hash"] += 1
            return False
        self.seen[hsh] = True
        key, q = cell_key(inf), quality_of(inf)
        front = self.cells.get(key, [])
        if any(o.quality[0] >= q[0] and o.power >= power and o.quality[1:] >= q[1:] for o in front):
            self.stats["worse"] += 1
            return False
        s = self._snap_slot(rng)
        bank.clone(sim, [slot], [s])
        e = Entry(actions=list(actions), snapshot=s, quality=q, key=key, hash=hsh, depth=depth_of(inf), last_new_step=step, power=power)
        keep = []
        for o in front:                                                              # the entries the newcomer dominates go
            if q[0] >= o.quality[0] and power >= o.power and q[1:] >= o.quality[1:]:
                self._drop(o)
                e.visits, e.children_found = max(e.visits, o.visits), max(e.children_found, o.children_found)
                self.stats["replaced"] += 1
            else:
                keep.append(o)
        keep.append(e)
        if len(keep) > self.FRONT:                                                   # over the cap: the most money, the most power, then the best of the rest
            by_money = sorted(keep, key=lambda o: -o.quality[0])
            by_power = sorted(keep, key=lambda o: -o.power)
            chosen = [by_money[0]]
            if by_power[0] not in chosen:
                chosen.append(by_power[0])
            for o in sorted(keep, key=lambda o: by_money.index(o) + by_power.index(o)):
                if len(chosen) >= self.FRONT:
                    break
                if o not in chosen:
                    chosen.append(o)
            for o in keep:
                if o not in chosen:
                    self._drop(o)
            keep = chosen
        self.bank_owner[s] = e
        new = key not in self.cells
        self.cells[key] = keep
        self.stats["new_cell"] += int(new)
        if e.depth > self.best_depth:
            self.best_depth, self.best_entry = e.depth, e
        return new

    def _drop(self, o):
        if o.snapshot is not None:
            self.bank_free.append(o.snapshot)
            self.bank_owner.pop(o.snapshot, None)
            o.snapshot = None

    def entries(self):
        return [e for front in self.cells.values() for e in front]


# --------------------------------------------------------------------------- 6. the cell selector
class Selector:
    def __init__(self, a=1.0, b=2.0, c=0.0, d=0.0):
        self.a, self.b, self.c, self.d = a, b, c, d

    def weight(self, e):
        return self.a / math.sqrt(e.visits + 1) + self.b * e.children_found / (e.visits + 1) + self.c * e.depth + self.d * e.value_pred

    def pick(self, archive, n, rng):
        """Every ante gets an equal share of the batch (round-robin over the antes that have cells, in a random order), so the early antes' many cells cannot starve the late ones; within an ante
        the weight decides."""
        by = {}
        for e in archive.entries():
            by.setdefault(e.depth // 3, []).append(e)
        pools = {d: (es, np.array([max(self.weight(e), 1e-6) for e in es])) for d, es in by.items()}
        order = list(pools)
        rng.shuffle(order)
        out, taken = [], set()
        while len(out) < n and order:
            for d in list(order):
                es, w = pools[d]
                free = [i for i in range(len(es)) if (d, i) not in taken]
                if not free:
                    order.remove(d)
                    continue
                pw = w[free] / w[free].sum()
                i = free[int(rng.choice(len(free), p=pw))]
                taken.add((d, i))
                out.append(es[i])
                if len(out) >= n:
                    break
        return out


# --------------------------------------------------------------------------- 3. the shop planner
class ShopPlanner:
    KINDS = ("greedy", "target", "random")

    def __init__(self, oracle, rng):
        self.oracle = oracle                                                          # joker key -> the first ante the seed's shops / packs offer it
        self.rng = rng
        self.wanted = sorted(oracle, key=lambda k: (-ROLE_PRI[role_of(k)], oracle[k]))

    def init_state(self, kind, ante):
        st = dict(kind=kind, last_buy=-1, swapped=False, rerolls=0, steps=0, target=None, max_rerolls=int(self.rng.integers(2, 12)))
        if kind == "target":
            pool = [k for k in self.wanted if self.oracle[k] <= ante + 1 and role_of(k) in ("copy", "xmult", "scaler")] or self.wanted[:8]
            st["target"] = pool[int(self.rng.integers(0, min(len(pool), 12)))] if pool else None
        return st

    def choose(self, m, aid, g, st):
        """One action for a game in a shop / pack / targeting state."""
        rng = self.rng
        ante, money = int(round(g[0] * 8)), int(round(g[17] * 50))
        st["steps"] += 1
        ph = Sim.phase_of(m)
        if st["steps"] > 45:                                                         # a plan is short; force the exit
            return A_LEAVE if m[A_LEAVE] else (A_PSKIP if m[A_PSKIP] else int(rng.choice(np.flatnonzero(m))))
        if ph == "pack":
            picks = [a for a in range(A_PICK, A_PSKIP) if m[a]]
            if not picks or (st["kind"] == "random" and rng.random() < 0.3):
                return A_PSKIP
            def val(a):
                it = int(aid[a])
                return (2 if it in ITEM2KEY else 1 if it in PLANET_ITEMS else 0) + (ROLE_PRI[role_of(ITEM2KEY[it])] * 0.1 if it in ITEM2KEY else 0)
            return max(picks, key=val) if st["kind"] != "random" else int(rng.choice(picks))
        if ph == "round":                                                             # choosing the cards a consumable acts on
            return int(rng.choice(np.flatnonzero(m[:A_SELECT])))
        buys = [a for a in range(A_BUY, A_BUY + MAX_SHOP) if m[a] and not (a - A_BUY <= st["last_buy"])]
        if st["swapped"]:
            buys = []
        packs = [a for a in (A_BUY_PACK, A_BUY_PACK + 1) if m[a] and not st["swapped"]]
        vouch = [a for a in (A_VOUCHER, A_VOUCHER + 1) if m[a] and not st["swapped"]]
        use = [a for a in range(A_USE_C, A_USE_C + 8) if a < len(m) and m[a] and (int(aid[a]) in PLANET_ITEMS or st["kind"] == "random")]
        jbuys = [a for a in buys if int(aid[a]) in ITEM2KEY]
        pbuys = [a for a in buys if int(aid[a]) in PLANET_ITEMS]
        reserve = 0 if ante <= 1 else 10
        can = money >= reserve + 6
        if use:
            return use[0]
        kind = st["kind"]
        if kind == "target" and st["target"]:
            vis = [a for a in jbuys if ITEM2KEY[int(aid[a])] == st["target"]]
            if vis:
                st["last_buy"] = vis[0] - A_BUY
                return vis[0]
            if m[A_REROLL] and st["rerolls"] < st["max_rerolls"] and money >= 8:
                st["rerolls"] += 1
                st["last_buy"] = -1
                return A_REROLL
        if kind in ("greedy", "target"):
            if jbuys and can:
                a = max(jbuys, key=lambda x: ROLE_PRI[role_of(ITEM2KEY[int(aid[x])])])
                st["last_buy"] = a - A_BUY
                return a
            if pbuys and can:
                st["last_buy"] = pbuys[0] - A_BUY
                return pbuys[0]
            if vouch and money >= 14:
                return vouch[0]
            if kind == "greedy" and m[A_REROLL] and money >= 30 and st["rerolls"] < 3:
                st["rerolls"] += 1
                st["last_buy"] = -1
                return A_REROLL
            return A_LEAVE
        # biased-random: jokers first in the early antes
        opts, w = [(A_LEAVE, 0.8 + 0.12 * st["steps"])], []
        for a in jbuys:
            opts.append((a, 5.0 if ante <= 2 else 2.5))
        for a in pbuys:
            opts.append((a, 1.5))
        for a in packs:
            opts.append((a, 1.8 if ante <= 2 else 0.9))
        for a in vouch:
            opts.append((a, 0.8))
        if m[A_REROLL] and not st["swapped"] and st["rerolls"] < st["max_rerolls"]:
            opts.append((A_REROLL, 1.2))
        for a in range(A_SELL_J, A_SELL_J + 5):
            if m[a] and not st["swapped"]:
                opts.append((a, 0.25))
        for a in range(A_SELL_C, A_SELL_C + 2):
            if a < len(m) and m[a]:
                opts.append((a, 0.2))
        for a in range(A_SWAP, A_SWAP_END + 1):
            if m[a]:
                opts.append((a, 0.15))
        acts, ws = zip(*opts)
        ws = np.array(ws, dtype=float)
        a = int(acts[int(rng.choice(len(acts), p=ws / ws.sum()))])
        if A_BUY <= a < A_BUY + MAX_SHOP:
            st["last_buy"] = a - A_BUY
        elif a == A_REROLL:
            st["rerolls"] += 1
            st["last_buy"] = -1
        elif A_SWAP <= a <= A_SWAP_END:
            st["swapped"] = True
        return a

    def run(self, sim, slots, kinds, max_steps=70):
        """Plays one policy per slot until its game is back at a blind select (or ended). -> {slot: action log}. A plan may end with skipping a small / big blind."""
        n = sim.n
        logs = {s: [] for s in slots}
        ante0 = np.asarray(sim.v.obs["g"])
        sts = {s: self.init_state(k, int(round(ante0[s, 0] * 8))) for s, k in zip(slots, kinds)}
        active = set(s for s in slots if not sim.dead[s])
        for _ in range(max_steps):
            if not active:
                break
            m, aid, g = sim.mask(), np.asarray(sim.v.obs["aid"]), np.asarray(sim.v.obs["g"])
            act = np.full(n, -1, dtype=np.int64)
            for s in list(active):
                if Sim.phase_of(m[s]) == "select":
                    if m[s][A_SKIP] and sts[s]["kind"] != "greedy" and self.rng.random() < 0.08:       # skip a small / big blind for its tag
                        act[s] = A_SKIP
                        logs[s].append(A_SKIP)
                        continue
                    active.discard(s)
                    continue
                act[s] = self.choose(m[s], aid[s], g[s], sts[s])
                logs[s].append(int(act[s]))
            if not (act >= 0).any():
                break
            sim.step(act)
            for s in list(active):
                if sim.dead[s]:
                    active.discard(s)
        return logs


# --------------------------------------------------------------------------- 2. the blind solver
@dataclass
class Outcome:
    actions: list
    money: int = 0
    hands_used: int = 0
    discs_used: int = 0
    key: int = 0
    won: bool = False


class BlindSolver:
    def __init__(self, deck, stake, seed, cap, k=3, node_cap=1000):
        self.k, self.node_cap = k, node_cap
        self.sel = Sim(cap, deck, stake, seed)                                       # the states after "select", where the exact solver reads them
        self.out = Sim(cap * k, deck, stake, seed)                                   # one per plan
        self.memo = {}
        self.stats = dict(calls=0, memo_hits=0, solved=0, dead_ends=0)
        self.by_blind = {}                                                           # blind index -> [solved, dead ends]

    def solve_many(self, sim, slots, hashes):
        """For each slot (a game at a blind select) the Pareto front of outcomes, [] when the blind cannot be cleared. Memoised on the state hash."""
        res = {}
        todo = []
        first = {}
        for s, h in zip(slots, hashes):
            self.stats["calls"] += 1
            if h in self.memo:
                res[s] = self.memo[h]
                self.stats["memo_hits"] += 1
            elif h in first:
                first[h].append(s)
            else:
                first[h] = [s]
                todo.append((s, h))
        for c0 in range(0, len(todo), self.sel.n):
            ch = todo[c0:c0 + self.sel.n]
            self.sel.clone(sim, [s for s, _ in ch], list(range(len(ch))))
            act = np.full(self.sel.n, -1, dtype=np.int64)
            act[:len(ch)] = A_SELECT
            self.sel.step(act)
            plans = self.sel.v.solve_rounds(list(range(len(ch))), self.k, self.node_cap)
            jobs = []                                                                # (index in ch, plan)
            for i, pl in enumerate(plans):
                if self.sel.dead[i]:
                    pl = []
                for p in pl[:self.k]:
                    jobs.append((i, [int(x) for x in p]))
            for j0 in range(0, len(jobs), self.out.n):
                jb = jobs[j0:j0 + self.out.n]
                self.out.clone(self.sel, [i for i, _ in jb], list(range(len(jb))))
                apply_actions(self.out, list(range(len(jb))), [p for _, p in jb])
                alive = [j for j in range(len(jb)) if not self.out.dead[j]]
                keys, money = {}, {}
                if alive:
                    _, ks, ms = self.out.state_hash(alive)
                    keys, money = dict(zip(alive, ks)), dict(zip(alive, ms))
                for j, (i, p) in enumerate(jb):
                    won = bool(self.out.won[j]) if self.out.dead[j] else False
                    if self.out.dead[j] and not won:
                        continue                                                     # the plan ended the game: not an outcome
                    o = Outcome(actions=[A_SELECT] + p, money=money.get(j, 0), hands_used=sum(1 for a in p if a < N_SUB), discs_used=sum(1 for a in p if N_SUB <= a < 2 * N_SUB),
                                key=keys.get(j, 0), won=won)
                    res.setdefault(ch[i][0], []).append(o)
            infs = sim.info([s for s, _ in ch])
            for (s, h), v in zip(ch, infs):
                res[s] = self.pareto(res.get(s, []))
                bb = self.by_blind.setdefault(3 * (v[0] - 1) + v[1], [0, 0])
                bb[0 if res[s] else 1] += 1
                self.memo[h] = res[s]
                self.stats["solved" if res[s] else "dead_ends"] += 1
        for h, ss in first.items():
            for s in ss[1:]:
                res[s] = res[ss[0]]
        return res

    def pareto(self, outs):
        """The outcomes no other outcome of the same resulting state (same key) beats in money, hands used and discards used; then the K of most money."""
        keep = []
        for o in outs:
            if o.won:
                return [o]
            if any(p.key == o.key and p.money >= o.money and p.hands_used <= o.hands_used and p.discs_used <= o.discs_used and (p.money, -p.hands_used, -p.discs_used) != (o.money, -o.hands_used, -o.discs_used)
                   for p in outs):
                continue
            if any(p.key == o.key and (p.money, p.hands_used, p.discs_used) == (o.money, o.hands_used, o.discs_used) for p in keep):
                continue
            keep.append(o)
        keep.sort(key=lambda o: (-o.money, o.hands_used, o.discs_used))
        return keep[:self.k]


# --------------------------------------------------------------------------- 7. the explorer
class Explorer:
    def __init__(self, seed, stake="GOLD", deck="RED", batch=8, plans=8, k=3, snaps=20000, a=1.0, b=2.0, c=0.0, d=0.0, rng_seed=0, node_cap=1000, pairs=8):
        self.seed, self.stake, self.deck = seed, stake, deck
        self.B, self.P, self.K = batch, plans, k
        self.rng = np.random.default_rng(rng_seed)
        self.oracle = obtainable(seed, stake, deck, pairs=pairs)
        self.bank = Sim(snaps, deck, stake, seed)
        self.work = Sim(batch * plans, deck, stake, seed)
        self.stage = Sim(batch * plans * k, deck, stake, seed)
        self.root = Sim(1, deck, stake, seed)
        self.pw = Sim(batch * plans * k, deck, stake, seed)                           # for the power measure
        self.planner = ShopPlanner(self.oracle, self.rng)
        self.solver = BlindSolver(deck, stake, seed, batch * plans, k, node_cap)
        self.selector = Selector(a, b, c, d)
        self.archive = Archive(snaps)
        self.step_no, self.win, self.t_plans = 0, None, 0
        self._add_root()

    def _add_root(self):
        h, _, _ = self.root.state_hash([0])
        inf = parse_info(self.root.info([0])[0])
        self.archive.consider(self.root, 0, self.bank, [], None, 0, self.rng, h[0], inf)

    def restore(self, entries, sim, slots):
        """Each entry's game into its slot: the snapshot when there is one, else the action log replayed from the start."""
        have = [(e, s) for e, s in zip(entries, slots) if e.snapshot is not None]
        miss = [(e, s) for e, s in zip(entries, slots) if e.snapshot is None]
        if have:
            sim.clone(self.bank, [e.snapshot for e, _ in have], [s for _, s in have])
        if miss:
            sim.clone(self.root, [0] * len(miss), [s for _, s in miss])
            apply_actions(sim, [s for _, s in miss], [e.actions for e, _ in miss])

    def step(self):
        """One iteration: B cells, P shop plans each, K blind outcomes each."""
        self.step_no += 1
        ents = self.selector.pick(self.archive, self.B, self.rng)
        for e in ents:
            e.visits += 1
        owners, slots, entries = [], [], []
        for b, e in enumerate(ents):
            for p in range(self.P):
                owners.append(b)
                slots.append(b * self.P + p)
                entries.append(e)
        W = self.work
        self.restore(entries, W, slots)
        kinds = []
        for p in range(len(slots)):
            kinds.append("greedy" if p % self.P == 0 else self.rng.choice(ShopPlanner.KINDS, p=[0.2, 0.4, 0.4]))
        t0 = time.time()
        plan_logs = self.planner.run(W, slots, kinds)
        self.t_plans += time.time() - t0
        alive = [s for s in slots if not W.dead[s] and Sim.phase_of(W.mask()[s]) == "select"]
        if not alive:
            return
        hashes, _, _ = W.state_hash(alive)
        outs = self.solver.solve_many(W, alive, hashes)
        jobs = []                                                                    # (work slot, owner entry, plan log, outcome)
        for s, h in zip(alive, hashes):
            for o in outs.get(s, []):
                jobs.append((s, entries[slots.index(s)], plan_logs[s], o))
        for j0 in range(0, len(jobs), self.stage.n):
            jb = jobs[j0:j0 + self.stage.n]
            self.stage.clone(W, [s for s, _, _, _ in jb], list(range(len(jb))))
            apply_actions(self.stage, list(range(len(jb))), [o.actions for _, _, _, o in jb])
            live = [j for j in range(len(jb)) if not self.stage.dead[j]]
            hmap = {}
            if live:
                hs, _, _ = self.stage.state_hash(live)
                fresh = [(j, h) for j, h in zip(live, hs) if not self.archive.is_dup(h)]
                if fresh:
                    infs = self.stage.info([j for j, _ in fresh])
                    pw = self.power([j for j, _ in fresh], [parse_info(v) for v in infs])
                    for (j, h), v, p in zip(fresh, infs, pw):
                        hmap[j] = (h, parse_info(v), p)
            for j, (s, e, plog, o) in enumerate(jb):
                full = e.actions + plog + o.actions
                if self.stage.dead[j]:
                    if self.stage.won[j] and self.win is None:
                        self.win = full
                    continue
                if j not in hmap:
                    self.archive.stats["considered"] += 1
                    self.archive.stats["dup_hash"] += 1
                    continue
                h, inf, pwr = hmap[j]
                if self.archive.consider(self.stage, j, self.bank, full, e, self.step_no, self.rng, h, inf, pwr):
                    e.children_found += 1
                    e.last_new_step = self.step_no

    def power(self, slots, infs):
        """For each state of the stage (a shop): the best share of the next boss's chip target the exact round solver reaches with this build, deck and levels (not learned)."""
        n = len(slots)
        self.pw.clone(self.stage, slots, list(range(n)))
        for i, inf in enumerate(infs):
            self.pw.v.set_state([i], inf["ante"], 2, [float("nan")] * 8)
        act = np.full(self.pw.n, -1, dtype=np.int64)
        act[:n] = A_SELECT
        self.pw.step(act)
        r = self.pw.v.solve_best(list(range(n)), 300)
        return [0.0 if self.pw.dead[i] else float(r[i][0]) for i in range(n)]

    def summary(self):
        a = self.archive
        hist = {}
        for e in a.entries():
            hist[e.depth // 3 + 1] = hist.get(e.depth // 3 + 1, 0) + 1
        return (f"step {self.step_no}: cells {len(a.cells)}, entries {len(a.entries())} (per ante {dict(sorted(hist.items()))}), deepest blind index {a.best_depth}, {a.stats}, solver {self.solver.stats}, "
                f"plans {self.t_plans:.0f}s")

    def save(self, path):
        with open(path, "w") as f:
            for e in self.archive.entries():
                f.write(json.dumps({"v": CELL_VERSION, "actions": e.actions, "depth": e.depth, "visits": e.visits, "children": e.children_found}) + "\n")


def verify_win(deck, stake, seed, actions):
    """The win replayed through the single-game engine (CppEnv.step): the end of the path must be a won game."""
    from go_explore import replay
    return replay(deck, stake, seed, actions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42010)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--plans", type=int, default=8)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--snaps", type=int, default=20000)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--save", default="")
    p.add_argument("--rng", type=int, default=0)
    p.add_argument("--wa", type=float, default=1.0)
    p.add_argument("--wb", type=float, default=2.0)
    p.add_argument("--wc", type=float, default=0.0)
    a = p.parse_args()
    ex = Explorer(a.seed, a.stake, "RED", a.batch, a.plans, a.k, a.snaps, a.wa, a.wb, a.wc, 0.0, a.rng)
    t0 = time.time()
    while ex.win is None and ex.step_no < a.max_steps and time.time() - t0 < a.minutes * 60:
        ex.step()
        if ex.step_no % 20 == 0:
            print(ex.summary(), flush=True)
    print(ex.summary())
    print("blind index: [cleared, dead ends]", dict(sorted(ex.solver.by_blind.items())))
    if a.save:
        ex.save(a.save)
    if ex.win is not None:
        won, blinds = verify_win("RED", a.stake, a.seed, ex.win)
        print(f"WIN found: {len(ex.win)} actions; replay in the single-game engine: won={won}, blinds {blinds}")
    else:
        print("no win found")
    print("blind index: [cleared, dead ends]", dict(sorted(ex.solver.by_blind.items())))


if __name__ == "__main__":
    main()
