"""Novelty beam: the plain beam keeps the best-scoring states, which end up as variations of one build. Here half the beam is the best by score and
the rest is chosen for being different (joker sets far from everything already kept), among candidates in the better part of the pool.

    python tools/solve_novel.py --stake GOLD --seeds 40001-40008
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import make      # noqa: E402
from balatro_rl.engine_env import CppVecEnv      # noqa: E402


def pick(pool, beam, elite, floor):
    """pool: (score, sig, path, slot, jokers). Returns up to `beam` candidates: the best `elite` distinct states, then the most novel."""
    pool.sort(key=lambda c: -c[0])
    seen, keep = set(), []
    rest = []
    for c in pool:
        if c[1] in seen:
            continue
        seen.add(c[1])
        (keep if len(keep) < elite else rest).append(c)
    if rest and len(keep) < beam:
        top = keep[0][0]
        spread = max(1e-9, top - rest[-1][0])
        rest = [c for c in rest if c[0] >= top - floor * spread] or rest
        sets = [frozenset(c[4]) for c in keep]
        rsets = [frozenset(c[4]) for c in rest]
        dist = [min((1.0 - len(rs & s) / max(1, len(rs | s))) for s in sets) for rs in rsets]
        used = [False] * len(rest)
        while len(keep) < beam and not all(used):
            j = max((i for i in range(len(rest)) if not used[i]), key=lambda i: (dist[i], rest[i][0]))
            used[j] = True
            keep.append(rest[j])
            sets.append(rsets[j])
            for i in range(len(rest)):
                if not used[i]:
                    dist[i] = min(dist[i], 1.0 - len(rsets[i] & rsets[j]) / max(1, len(rsets[i] | rsets[j])))
    return keep


ARCH_KEYS = ("runner", "trousers", "square", "castle", "wee", "red_card", "green_joker", "constellation", "hologram", "campfire", "flash", "madness")


def pick_arch(pool, beam, elite, floor, per_arch):
    """Archetype slots: for every engine joker of ARCH_KEYS some candidate holds, its best `per_arch` holders are kept first (a slow starter is
    not ranked out before its joker pays off); the rest of the beam is chosen by pick()."""
    from balatro_rl.sim.jokers import JOKERS
    ids = {list(JOKERS).index(k): k for k in ARCH_KEYS}
    by = {}
    for c in sorted(pool, key=lambda c: -c[0]):
        for jk in set(c[4]):
            if jk in ids:
                by.setdefault(ids[jk], []).append(c)
    keep, seen = [], set()
    for k, cs in by.items():
        for c in cs[:per_arch]:
            if c[1] not in seen and len(keep) < beam // 2:
                seen.add(c[1])
                keep.append(c)
    rest = pick([c for c in pool if c[1] not in seen], beam - len(keep), max(1, elite - len(keep) // 2), floor)
    return keep + rest


def eval_scores(dest, slots, which):
    """The run eval (balatro_rl/runeval.py) on the end states in dest[slots]: the blended score, or one phase potential."""
    from balatro_rl.runeval import RunEval, from_vecenv
    global _EV
    try:
        ev = _EV
    except NameError:
        ev = _EV = RunEval()
    out = ev(**from_vecenv(dest, list(slots)))
    return out["score"] if which == "eval" else out[which]


_NO_COUNTER = {"supernova", "abstract", "bull", "bootstraps", "swashbuckler", "throwback", "steel_joker", "stone", "erosion", "hiker"}
_SCRATCH = {}


def proj_scores(dest, slots, deck, stake, seed):
    """Projected ante-8 strength of the end states dest[slots]: each scaling joker's counter grown to the ante-8 boss by balatro_rl/growth.py,
    the state moved there (VecEnv.set_state), the boss selected and measured by fast_power: log10(hands x mean best play / target)."""
    import numpy as np
    from balatro_rl.growth import project
    from balatro_rl.runeval import from_vecenv
    from balatro_rl.sim.items import VOCAB
    from balatro_rl.env import A_SELECT
    n = len(slots)
    if n == 0:
        return np.zeros(0)
    key = (deck, stake)
    if key not in _SCRATCH or _SCRATCH[key].n < n:
        _SCRATCH[key] = make(max(n, 1024), deck, stake, seed, 0)
    sc = _SCRATCH[key]
    inp = from_vecenv(dest, list(slots))
    rows = project(**inp)
    jid = inp["jid"]
    cnt = np.full((n, 8), np.nan)
    for i, rs in enumerate(rows):
        used = set()
        for r in rs:
            if r["key"] in _NO_COUNTER or r["unit"] not in ("chips", "mult", "xmult"):
                continue
            hit = [j for j in range(jid.shape[1]) if jid[i, j] > 0 and VOCAB[int(jid[i, j])] == "j_" + r["key"] and j not in used]
            if hit:
                used.add(hit[0])
                cnt[i, hit[0]] = r["future"]
    sc.copy_many(dest, list(slots), list(range(n)))
    sc.set_state(list(range(n)), 8, 2, cnt)
    act = np.full(sc.n, -1, dtype=np.int64)
    act[:n] = A_SELECT
    sc.step(act)
    fp = sc.fast_power(list(range(n)), 4)[:, 0]
    return np.log10(np.maximum(fp, 1e-3))


_NNUE = {}


def nnue_scores(dest, slots, path):
    """The evaluation network's predicted blinds for the end states dest[slots] (features read with the engine's future list)."""
    import numpy as np
    from balatro_rl.nnue import features, load
    from balatro_rl.runeval import from_vecenv
    if path not in _NNUE:
        _NNUE[path] = (load(path, "cuda"), {})
    net, envs = _NNUE[path]
    n = len(slots)
    if "e" not in envs or envs["e"].n < n:                          # one scratch environment, grown when needed (not one per pool size)
        envs.pop("e", None)
        e = CppVecEnv(max(n, 1024), "RED", "GOLD", seed0=1, threads=0)
        e.set_exact(True)
        e.set_priv(True)
        envs["e"] = e
    e = envs["e"]
    e.copy_many(dest, list(slots), list(range(n)))
    e.set_priv(True)
    o = from_vecenv(e, list(range(n)))
    o["priv"] = np.asarray(e.obs["priv"])[:n]
    idx, offs = features(o)
    return net.predict(idx, offs) * 24.0


def zscore(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def solve(deck, stake, seed, beam, elite, floor, ends_per, sw, sl, node_cap, money_w, pot_w=0.0, scale_w=0.0, max_levels=40, target=(), target_w=0.0,
          rank="engine", eval_w=1.0, prune=0, speed=(False, False, 0.0), order_mode=0, reroll_cap=0, shop_top=1.0, pot=(0, 6), arch=0, proj_w=0.0,
          auto_target=False, pack_macro=False, on_pool=None, hand_bound=False, nnue=None, nnue_w=0.0, round_eval=0.0, nnue_bin=None):
    """rank: "engine" (the search's own score), "eval" / "early" / "mid" / "late" (the run eval's blended score or one potential replaces it),
    "mix" (z-scored engine score + eval_w x z-scored eval)."""
    marg = 2 if (pot_w > 0 or scale_w > 0) else 1
    cur = make(beam, deck, stake, seed, marg)
    dest = make(beam * ends_per, deck, stake, seed, marg)
    cur.set_search_prune(prune & ~4)                                 # (global switches in the engine; bit 4 is the beam's own, below)
    cur.set_solver_speed(*speed)
    cur.set_order_mode(order_mode)
    cur.set_pot_mode(*pot)
    cur.set_search_pack_macro(pack_macro)
    cur.set_solver_hand_bound(hand_bound)
    if round_eval > 0 and nnue_bin:                                   # round lines: money + round_eval x the network's blinds after the round
        cur.load_nnue(nnue_bin)
    cur.set_round_eval(round_eval if nnue_bin else 0.0)
    cur.set_search_reroll_cap(reroll_cap)
    scores = [0.0]                                                   # each beam state's score when it was kept (orders them for shop_top)
    dest_b = make(beam * ends_per, deck, stake, seed, marg) if shop_top < 1.0 else None
    root = make(1, deck, stake, seed, marg)
    for e in (cur, dest, root) + ((dest_b,) if dest_b is not None else ()):
        e.set_auto_target(auto_target)
    cur.copy_many(root, [0], [0])
    paths, n, t0 = [[]], 1, time.time()
    for level in range(max_levels):
        if shop_top < 1.0 and n > 1:
            # the full shop search only for the best shop_top share of the beam (by the score they were kept with); the rest get a minimal one
            order = sorted(range(n), key=lambda b: -scores[b])
            k = max(1, int(round(shop_top * n)))
            top, rest = order[:k], order[k:]
            res = [None] * n
            ra = cur.expand_level(top, dest, ends_per, sw, sl, node_cap, money_w, pot_w, scale_w, list(target), target_w)
            rb = cur.expand_level(rest, dest_b, ends_per, 1, 1, node_cap, money_w, pot_w, scale_w, list(target), target_w) if rest else []
            moved_src, moved_dst = [], []
            off = len(top) * ends_per
            for b, r in zip(top, ra):
                res[b] = r
            for b, (win, wpath, cands) in zip(rest, rb):
                nc = []
                for path, score, sig, slot, jk in cands:
                    moved_src.append(slot); moved_dst.append(off + len(moved_dst))
                    nc.append((path, score, sig, moved_dst[-1], jk))
                res[b] = (win, wpath, nc)
            if moved_src:
                dest.copy_many(dest_b, moved_src, moved_dst)
        else:
            res = cur.expand_level(list(range(n)), dest, ends_per, sw, sl, node_cap, money_w, pot_w, scale_w, list(target), target_w)
        pool = []
        for b, (win, wpath, cands) in enumerate(res):
            if win:
                return paths[b] + list(wpath), {"level": level, "secs": time.time() - t0}
            for path, score, sig, slot, jk in cands:
                pool.append((score, sig, paths[b] + list(path), slot, jk))
        if not pool:
            return None, {"level": level, "secs": time.time() - t0, "dead": True, "best_path": paths[0]}
        if on_pool is not None:
            on_pool(level, dest, pool)
        if prune & 4:                                                 # the beam's own dominance: of the states with the same future keep the richest
            keys, money = dest.state_keys([c[3] for c in pool])
            best = {}
            for k, (c, key, m) in enumerate(zip(pool, keys.tolist(), money.tolist())):
                if key not in best or (m, c[0]) > best[key][0]:
                    best[key] = ((m, c[0]), k)
            pool = [pool[k] for _, k in best.values()]
        if nnue and nnue_w > 0:                                         # the evaluation network in the ranking (both z-scored; nnue_w >= 100: the network alone)
            nn_ = nnue_scores(dest, [c[3] for c in pool], nnue)
            mixed = nn_ if nnue_w >= 100 else zscore([c[0] for c in pool]) + nnue_w * zscore(nn_)
            pool = [(float(m), c[1], c[2], c[3], c[4]) for m, c in zip(mixed, pool)]
        if proj_w > 0 and level >= 3:                                  # the projected ante-8 strength mixed into the ranking (from the second ante)
            pr = proj_scores(dest, [c[3] for c in pool], deck, stake, seed)
            mixed = zscore([c[0] for c in pool]) + proj_w * zscore(pr)
            pool = [(float(m), c[1], c[2], c[3], c[4]) for m, c in zip(mixed, pool)]
        if rank != "engine":
            ev = eval_scores(dest, [c[3] for c in pool], "eval" if rank == "mix" else rank)
            if rank == "mix":
                mixed = zscore([c[0] for c in pool]) + eval_w * zscore(ev)
                pool = [(float(m), c[1], c[2], c[3], c[4]) for m, c in zip(mixed, pool)]
            else:
                pool = [(float(e), c[1], c[2], c[3], c[4]) for e, c in zip(ev, pool)]
        keep = pick_arch(pool, beam, elite, floor, arch) if arch > 0 else pick(pool, beam, elite, floor)
        cur.copy_many(dest, [c[3] for c in keep], list(range(len(keep))))
        paths, n = [c[2] for c in keep], len(keep)
        scores = [c[0] for c in keep]
    return None, {"level": max_levels, "secs": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--seeds", default="40001-40008")
    p.add_argument("--beam", type=int, default=32)
    p.add_argument("--elite", type=int, default=16)
    p.add_argument("--floor", type=float, default=0.5, help="novel picks come from candidates within this share of the pool's score range below the best")
    p.add_argument("--money-weight", type=float, default=0.0)
    p.add_argument("--pot-weight", type=float, default=0.0)
    p.add_argument("--scale-weight", type=float, default=0.0)
    p.add_argument("--out-dir", default="traces/novel")
    p.add_argument("--rank", default="engine", choices=["engine", "eval", "early", "mid", "late", "mix"], help="what ranks the beam's candidates (see solve)")
    p.add_argument("--eval-weight", type=float, default=1.0, help="--rank mix: the eval's weight against the engine's score, both z-scored")
    p.add_argument("--prune", type=int, default=0, help="bits: 1 = a swap / sell in the shop search must raise the score; 2 = money dominance in the shop search; "
                                                         "4 = money dominance among the beam's candidates (7 = all)")
    p.add_argument("--ends", type=int, default=8, help="end states the shop search keeps per beam state")
    p.add_argument("--solver-lite", action="store_true", help="the round solver's cheaper nodes (identical plans)")
    p.add_argument("--order-mode", type=int, default=0, help="1: joker swaps only with Blueprint / Brainstorm / Ceremonial Dagger, else chips, +mult, xMult order")
    p.add_argument("--reroll-cap", type=int, default=0, help="> 0: rerolls the shop search tries per shop visit")
    p.add_argument("--shop-top", type=float, default=1.0, help="< 1: only this share of the beam (the best) gets the full shop search")
    p.add_argument("--pot-mode", type=int, default=0, help="1: the growth potential from the growth model's rates (with --scale-weight)")
    p.add_argument("--pot-rounds", type=int, default=6, help="--pot-mode 1: rounds of growth the potential looks ahead")
    p.add_argument("--target-keys", default="", help="comma-separated jokers the shop search is rewarded for owning (with --target-w)")
    p.add_argument("--target-w", type=float, default=0.0, help="the ranking bonus per target joker owned")
    p.add_argument("--plan-file", default="", help="tools/seed_plan.py output: each seed's best plan's jokers become its targets (with --target-w)")
    p.add_argument("--plan-min-clear", type=float, default=0.05, help="--plan-file: a plan below this clear rate is not used")
    p.add_argument("--arch-slots", type=int, default=0, help="> 0: beam slots kept for the best holders of each engine joker (ARCH_KEYS)")
    p.add_argument("--nnue", default="", help="an evaluation network (balatro_rl/nnue.py) for the ranking")
    p.add_argument("--nnue-w", type=float, default=1.0, help="its weight against the search's own score (both z-scored); 100+: the network alone")
    p.add_argument("--round-eval", type=float, default=0.0, help="> 0: the round solver keeps lines by money + this x the network's expected blinds (needs --nnue-bin)")
    p.add_argument("--nnue-bin", default="", help="the network exported for the engine (balatro_rl/nnue.py export)")
    p.add_argument("--hand-bound", action="store_true", help="the round solver's hands-left bound (the richest clearing line)")
    p.add_argument("--auto-target", action="store_true", help="a tarot's target cards are chosen at once (Game::auto_targets): tarots become one action")
    p.add_argument("--pack-macro", action="store_true", help="a bought pack is opened at once in the shop search: each pick is a child of the buy")
    p.add_argument("--proj-w", type=float, default=0.0, help="> 0: the projected ante-8 strength (growth.py + fast_power at the ante-8 boss) in the ranking, both z-scored")
    p.add_argument("--solver-cache", action="store_true", help="beam states share solved rounds (each cached line replayed first)")
    p.add_argument("--dead-f", type=float, default=0.0, help="> 0: skip rounds the best play from all cards left x hands x this margin cannot clear")
    p.add_argument("--shop-width", type=int, default=4, help="partial shop states kept per depth")
    p.add_argument("--shop-depth", type=int, default=4, help="shop actions per line")
    a = p.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")]
    os.makedirs(a.out_dir, exist_ok=True)
    solved, levels, t0 = 0, [], time.time()
    for sd in range(lo, hi + 1):
        if any(os.path.exists(os.path.join(a.out_dir, f"{a.stake}_{sd}{x}.json")) for x in ("", "_dead")):
            continue                                                  # (resumable: a seed already searched is skipped)
        t = time.time()
        from balatro_rl.sim.jokers import JOKERS as _J
        tgt = [list(_J).index(k.strip()) for k in a.target_keys.split(",") if k.strip()]
        if a.plan_file:
            plans = json.load(open(a.plan_file)).get(str(sd), {}).get("plans", [])
            if plans and plans[0]["clear"] >= a.plan_min_clear:
                tgt = sorted(set(tgt) | {list(_J).index(k) for k in plans[0]["jokers"]})
        path, info = solve(a.deck, a.stake, sd, a.beam, a.elite, a.floor, a.ends, a.shop_width, a.shop_depth, 1000, a.money_weight, a.pot_weight, a.scale_weight,
                           target=tgt, target_w=a.target_w, pot=(a.pot_mode, a.pot_rounds), arch=a.arch_slots, proj_w=a.proj_w,
                           auto_target=a.auto_target, pack_macro=a.pack_macro, nnue=a.nnue or None, nnue_w=a.nnue_w, hand_bound=a.hand_bound,
                           round_eval=a.round_eval, nnue_bin=a.nnue_bin or None,
                           rank=a.rank, eval_w=a.eval_weight, prune=a.prune, speed=(a.solver_lite, a.solver_cache, a.dead_f),
                           order_mode=a.order_mode, reroll_cap=a.reroll_cap, shop_top=a.shop_top)
        if path:
            solved += 1
            json.dump({"seed": sd, "deck": a.deck, "stake": a.stake, "solved": True, "actions": path, "info": info},
                      open(os.path.join(a.out_dir, f"{a.stake}_{sd}.json"), "w"))
        else:
            levels.append(info["level"])
            json.dump({"seed": sd, "deck": a.deck, "stake": a.stake, "solved": False, "actions": info.get("best_path"), "info": {"level": info["level"]}},
                      open(os.path.join(a.out_dir, f"{a.stake}_{sd}_dead.json"), "w"))
        print(f"seed {sd}: {'SOLVED' if path else 'died at blind %2d' % info['level']}  {time.time() - t:.0f}s", flush=True)
    n = hi - lo + 1
    if a.solver_cache:
        from solve_fast import make as _mk
        h, m_, fb = _mk(1, a.deck, a.stake, 1, 1).solver_cache_stats(False)
        print(f"round cache: {h} hits, {m_} misses, {fb} fallbacks ({h / max(1, h + m_ + fb):.0%} of rounds reused)")
    print(f"{solved}/{n} solved; mean death blind of the rest {sum(levels) / max(1, len(levels)):.1f}; {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
