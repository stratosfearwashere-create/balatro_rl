"""Bounded best-first search with state merging, over the shop decisions of one seed (exact-seed game; the cards are played by the scripted player, so the search is only over what to buy, sell,
reroll, skip).

A node is the game at a blind select. Expanding it plays the blind and searches the shop that follows (engine expand_level: every purchase set, rerolls, sales, packs; the buys in one canonical
order, so sets, not orderings), giving the nodes one blind on.

  best-first   a priority queue; the node with the highest optimistic bound is expanded next (ties: the further along). On a winnable seed it dives toward a verified win.
  merging      a table of the states seen, keyed by Env::state_key (jokers, deck, hand levels, vouchers, tags, bosses and the RNG STREAM POSITIONS) and the money: each distinct state is expanded once.
  dominance    states with the same key and less money are dropped (and a stored state is evicted by a newcomer with the same key and more money). The key holds every other resource.
  stages       the queue is bounded per stage (the blind index): when a stage holds more than --cap nodes the lowest-bound ones are dropped and COUNTED. A search that ends with the queue empty and
               no drops and no pruning by an inadmissible bound has PROVED that no win exists in the action space; with drops it can only report "unresolved".
  bound        an optimistic bound on the state's best outcome (see bound.py: the module that builds it from the seed's contents); nodes whose bound is below the boss target are pruned.

    python tools/astar_shop.py --seed 42010 --beam-inner 64 --cap 2000
"""
import argparse
import heapq
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import make                                                      # noqa: E402


class Pool:
    """A store of game states in one VecEnv (slots are reused when a node is dropped)."""

    def __init__(self, env):
        self.env, self.free, self.n = env, list(range(env.n - 1, -1, -1)), env.n

    def alloc(self):
        if not self.free:
            raise MemoryError("state pool full")
        return self.free.pop()

    def release(self, slot):
        self.free.append(slot)


def search(seed, deck="RED", stake="GOLD", mode=2, inner_sw=16, inner_sl=4, ends_per=24, cap=2000, batch=16, max_nodes=200000, bound_fn=None, time_limit=600, log=print, scripted=True, audit=False):
    pool_n = cap * 24 + batch * ends_per + 64
    store = Pool(make(pool_n, deck, stake, seed, 2))                                 # the states of the open nodes
    work = make(batch, deck, stake, seed, 2)                                         # the batch being expanded
    dest = make(batch * ends_per, deck, stake, seed, 2)                              # its children
    if scripted:
        make(1, deck, stake, seed, 1).set_level_scripted(mode)
    root = make(1, deck, stake, seed, 2)
    s0 = store.alloc()
    store.env.copy_many(root, [0], [s0])
    key0, money0 = store.env.state_keys([s0])
    seen = {}                                                                         # key -> max money seen with this key (the dominance table)
    stats = dict(expanded=0, children=0, merged=0, dominated=0, pruned_bound=0, dropped_cap=0, pruned_dead=0, max_stage=0, per_stage={})
    heap, tick = [], 0
    stage_count = {}
    nodes = {}                                                                        # slot -> (key, money, level, path, bound)
    b0 = bound_fn(store.env, s0, 0) if bound_fn else 0.0
    uid = 0
    info = {0: (None, b0 if b0 is not None else 0.0)}                                   # uid -> (parent uid, bound): the lineage, for the audit of the bound along a winning line
    nodes[s0] = (int(key0[0]), int(money0[0]), 0, [], b0 if b0 is not None else 0.0, 0)
    heapq.heappush(heap, (-(b0 or 0.0), 0, tick, s0)); tick += 1
    stage_count[0] = 1
    seen[int(key0[0])] = int(money0[0])
    t0 = time.time()
    win = None
    while heap and win is None and stats["expanded"] < max_nodes and time.time() - t0 < time_limit:
        batch_slots = []
        while heap and len(batch_slots) < batch:
            _, _, _, slot = heapq.heappop(heap)
            if slot in nodes and nodes[slot] is not None:
                batch_slots.append(slot)
        if not batch_slots:
            break
        work.copy_many(store.env, batch_slots, list(range(len(batch_slots))))
        res = work.expand_level(list(range(len(batch_slots))), dest, ends_per, inner_sw, inner_sl, 1000, 0.0, 0.0, 1.0, [], 0.0)
        child_rows = []
        for b, (w, wpath, cands) in enumerate(res):
            parent = nodes[batch_slots[b]]
            stats["expanded"] += 1
            if w:
                win = parent[3] + list(wpath)
                lineage, u = [], parent[5]
                while u is not None:
                    lineage.append(info[u][1]); u = info[u][0]
                stats["lineage_bounds"] = lineage[::-1]
                break
            if not cands:
                stats["pruned_dead"] += 1
            for path, score, sig, slot, jk in cands:
                child_rows.append((parent, path, slot))
        for s in batch_slots:                                                         # the expanded nodes leave the open set
            nodes[s] = None
            store.release(s)
        if win is not None or not child_rows:
            continue
        # the children's keys, from a scratch copy in the store (a child that survives keeps its slot)
        new_slots = []
        for parent, path, dslot in child_rows:
            try:
                new_slots.append(store.alloc())
            except MemoryError:
                new_slots.append(None)
        ok = [i for i, s in enumerate(new_slots) if s is not None]
        if ok:
            store.env.copy_many(dest, [child_rows[i][2] for i in ok], [new_slots[i] for i in ok])
            keys, money = store.env.state_keys([new_slots[i] for i in ok])
        stats["children"] += len(child_rows)
        for j, i in enumerate(ok):
            parent, path, _ = child_rows[i]
            slot = new_slots[i]
            k, m = int(keys[j]), int(money[j])
            level = parent[2] + 1
            old = seen.get(k)
            if old is not None and m <= old:                                           # the same state (merged) or one with more money already seen (dominated)
                stats["merged" if m == old else "dominated"] += 1
                store.release(slot)
                continue
            seen[k] = m
            raw = bound_fn(store.env, slot, level) if bound_fn else float(level)
            bnd = raw
            if bnd is None and audit:                                                  # audit: keep the node, with the bound it would have been pruned on
                bnd = 0.0
                stats["audit_would_prune"] = stats.get("audit_would_prune", 0) + 1
            if bnd is None:                                                            # the bound says no win can come from here
                stats["pruned_bound"] += 1
                store.release(slot)
                continue
            uid += 1
            info[uid] = (parent[5], bnd)
            nodes[slot] = (k, m, level, parent[3] + list(path), bnd, uid)
            stage_count[level] = stage_count.get(level, 0) + 1
            heapq.heappush(heap, (-bnd, -level, tick, slot)); tick += 1
            stats["max_stage"] = max(stats["max_stage"], level)
        for i, s in enumerate(new_slots):
            if s is None:
                stats["dropped_cap"] += 1                                              # no room: counted
        # the per-stage cap: drop the lowest-bound nodes of an over-full stage, and count them
        for level, cnt in list(stage_count.items()):
            if cnt > cap:
                rows = sorted([(nodes[s][4], s) for s in nodes if nodes[s] is not None and nodes[s][2] == level])
                for bnd, s in rows[:cnt - cap]:
                    nodes[s] = None
                    store.release(s)
                    stats["dropped_cap"] += 1
                stage_count[level] = cap
        if stats["expanded"] % (batch * 20) == 0:
            log(f"  expanded {stats['expanded']}, open {sum(1 for v in nodes.values() if v is not None)}, furthest stage {stats['max_stage']}, merged {stats['merged']}, dominated {stats['dominated']}, "
                f"pruned by the bound {stats['pruned_bound']}, dropped {stats['dropped_cap']}, {time.time() - t0:.0f}s")
    open_left = sum(1 for v in nodes.values() if v is not None)
    if win is not None:
        verdict = "WIN"
    elif open_left == 0 and stats["dropped_cap"] == 0:
        verdict = "NO WIN IN THE ACTION SPACE (proved, up to the bound's admissibility)" if bound_fn else "NO WIN IN THE ACTION SPACE (proved: nothing was pruned by a bound or a cap)"
    elif open_left == 0:
        verdict = "UNRESOLVED (the queue emptied but states were dropped by the caps)"
    else:
        verdict = "UNRESOLVED (stopped by the node or time limit)"
    stats["seconds"] = round(time.time() - t0, 1)
    return win, verdict, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42010)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--mode", type=int, default=2, help="the scripted player's mode (-1: the exact round solver)")
    p.add_argument("--inner-sw", type=int, default=16)
    p.add_argument("--inner-sl", type=int, default=4)
    p.add_argument("--ends", type=int, default=24)
    p.add_argument("--cap", type=int, default=2000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-nodes", type=int, default=100000)
    p.add_argument("--time", type=float, default=600)
    p.add_argument("--bound", action="store_true", help="prune with the optimistic bound (astar_bound.py)")
    p.add_argument("--audit", action="store_true", help="compute the bound but never prune on it; report what it would have pruned and its values along the winning line")
    p.add_argument("--slack", type=float, default=1.25)
    p.add_argument("--per-ante", type=float, default=3)
    p.add_argument("--pairs", type=int, default=8)
    a = p.parse_args()
    bf = None
    if a.bound or a.audit:
        from astar_bound import Bound
        bf = Bound(a.seed, a.stake, "RED", pairs=a.pairs, per_ante=int(a.per_ante), slack=a.slack)
    win, verdict, st = search(a.seed, "RED", a.stake, a.mode, a.inner_sw, a.inner_sl, a.ends, a.cap, a.batch, a.max_nodes, bf, a.time, scripted=a.mode >= 0, audit=a.audit)
    if bf:
        st["bound_stats"] = bf.stats
    print(f"seed {a.seed}: {verdict}; actions {len(win) if win else '-'}; {st}")


if __name__ == "__main__":
    main()
