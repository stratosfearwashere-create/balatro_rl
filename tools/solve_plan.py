"""Backward planning over a deterministic seed: pick the build first, then check it can be bought.

    1. LIST      the jokers the shops of antes 1..A show (exact seed: the cards of an ante are a fixed sequence, pair k of it is on show after k refreshes of the ante's shop:
                 the first shop visit, then each reroll or later visit), with prices, and when each first appears.
    2. ENUMERATE candidate final builds (sets of up to five of them) and SCORE each against the ante-8 boss with the judge (judge.py, the exact expected-score features
                 included): every scaler carries the counter it would have grown to by then from the ante it is first on show.
                 A beam over sets (add a joker to the best sets so far), not billions of action sequences: a few thousand sets.
    3. CHECK     feasibility of the best plans by playing them in the exact engine, many at once (one environment per plan): every blind is played with the exact round
                 solver (VecEnv.solve_rounds), and at each shop the plan buys its jokers when they are on show and affordable (selling one that is not in the plan if the slots
                 are full), and rerolls toward the pair the next one is due in while it can still afford it. A plan that cannot pay or cannot clear a blind dies there.

    python tools/solve_plan.py --seed 42003 --stake GOLD
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEV = os.environ.get("BALATRO_DEV_ENGINE", "C:/Users/strat/AppData/Local/Temp/claude/C--balatro-rl/85bbfca6-42a2-4515-a6d7-78ba00e9b1d8/scratchpad/dev")
if DEV and os.path.isdir(DEV):                                    # the engine build with set_build (VecEnv.set_build is not in the production module yet)
    sys.path.insert(0, DEV)
    import _engine_dev
    import balatro_rl.sim as _S
    sys.modules["balatro_rl.sim._engine"] = _engine_dev
    _S._engine = _engine_dev
from solve_fast import make                                                     # noqa: E402
from balatro_rl.sim.game import Game                                              # noqa: E402
from balatro_rl.sim.jokers import JOKERS                                          # noqa: E402
from balatro_rl.sim.seedrng import seed_string                                    # noqa: E402
from balatro_rl.env import (A_BUY, A_LEAVE, A_PICK, A_PSKIP, A_REPLACE, A_REROLL, A_SELECT, A_SELL_J, F_JOKER_W, MAX_JOKERS, MAX_SHOP, item_id)   # noqa: E402

KEYS = list(JOKERS)
DECK = os.environ.get("BALATRO_DECK", "RED")                      # the deck played (BALATRO_DECK=YELLOW: $10 more to start with)
# the engine's growth table (game.hpp SCN_GROW; the same as potential.py): name -> (start, per round, floor, ceiling)
from balatro_rl.potential import GROW                                             # noqa: E402


# --------------------------------------------------------------------------- 1. what the shops show
def appearances(seed, stake, antes, pairs):
    """-> {joker key: dict(ante, pair, price, edition, eternal)}: the first appearance of each joker that is not perishable or rental (a rental costs $3 a round, a perishable
    stops working after five)."""
    g = Game(seed=1, deck_type=DECK, stake=stake, exact_seed=seed_string(seed))
    g.money = 10_000
    out, every = {}, []
    for ante in range(1, antes + 1):
        g.ante = ante
        g.shop, g.shop_packs, g.shop_vouchers, g.ante_voucher = [], [], [], None
        g.open_shop()
        for k in range(pairs):
            if k:
                g.free_rerolls = 1
                g.reroll()
            for it in g.shop:
                if it.kind != "joker" or it.joker is None:
                    continue
                j = it.joker
                key = it.key[2:]
                rec = {"key": key, "ante": ante, "pair": k, "price": int(it.cost), "edition": getattr(j, "edition", None) or "", "eternal": bool(getattr(j, "eternal", False))}
                every.append(rec)
                if getattr(j, "perishable", None) not in (None, False, 0) or getattr(j, "rental", False):
                    continue
                if key not in out:
                    out[key] = rec
    return out, every


# --------------------------------------------------------------------------- 2. scoring a build at the ante-8 boss
def counter_at_8(key, ante):
    """The counter a joker first on show in `ante` has at the ante-8 boss (bought at the first shop, then held): the engine's growth table."""
    g = GROW.get(key)
    if g is None:
        return float("nan")
    start, rate, lo, hi = g
    rounds = 3 * (8 - ante) + 2
    return float(min(max(start + rate * rounds, lo), hi))


class Scorer:
    def __init__(self, seed, stake, judge_path, dev, levels, n=256):
        from balatro_rl import judge as J
        self.v = make(n, DECK, stake, seed, 1)
        self.n, self.dev, self.levels = n, dev, levels
        self.judge = J.load(judge_path, dev)
        self.cache = {}

    def score(self, builds):
        """builds: list of lists of (key, first ante). -> list of (log10 share of the ante-8 boss's requirement reached by the best play, chance to pass it)."""
        import torch
        out = [None] * len(builds)
        todo = [i for i, b in enumerate(builds) if tuple(sorted(b)) not in self.cache]
        for c0 in range(0, len(todo), self.n):
            ch = todo[c0:c0 + self.n]
            for slot, i in enumerate(ch):
                ids = [KEYS.index(k) for k, _ in builds[i]]
                self.v.set_build([slot], 8, 2, ids, [counter_at_8(k, a) for k, a in builds[i]], 0, self.levels, 25)
            m = len(ch)
            g = torch.from_numpy(np.asarray(self.v.obs["g"])[:m]).to(self.dev)
            jid = torch.from_numpy(np.asarray(self.v.obs["jid"])[:m]).to(self.dev)
            jf = torch.from_numpy(np.asarray(self.v.obs["jf"])[:m][:, :, :F_JOKER_W]).to(self.dev)
            cid = torch.from_numpy(np.asarray(self.v.obs["cid"])[:m]).to(self.dev)
            g, jf = torch.nan_to_num(g, nan=0.0, posinf=10.0, neginf=-10.0), torch.nan_to_num(jf, nan=0.0, posinf=10.0, neginf=-10.0)
            best_r, best_p = None, None
            with torch.no_grad():
                for mode in (0, 1):
                    o = self.judge.predict(g, jid, jf, cid, torch.full((m,), mode, device=self.dev, dtype=torch.long))
                    r, p = o["log_ratio"].float(), o["p_pass"].float()
                    best_r = r if best_r is None else torch.maximum(best_r, r)
                    best_p = p if best_p is None else torch.maximum(best_p, p)
            for i, r, p in zip(ch, best_r.cpu().numpy(), best_p.cpu().numpy()):
                self.cache[tuple(sorted(builds[i]))] = (float(r), float(p))
        return [self.cache[tuple(sorted(b))] for b in builds]


def enumerate_builds(scorer, pool, size, width, expand, log):
    """Beam over sets: the best `width` sets of each size, extended with the best `expand` jokers."""
    cands = [(k, pool[k]["ante"]) for k in pool]
    single = scorer.score([[c] for c in cands])
    order = sorted(range(len(cands)), key=lambda i: -single[i][0])
    top = [cands[i] for i in order[:expand]]
    beams, results = [[c] for c in top[:width]], {}
    for s in range(1, size + 1):
        sc = scorer.score(beams)
        ranked = sorted(zip(beams, sc), key=lambda x: -x[1][0])
        results[s] = ranked
        log(f"  size {s}: {len(beams)} sets scored, best log10 share {ranked[0][1][0]:+.2f}, pass {ranked[0][1][1]:.2f}: {[k for k, _ in ranked[0][0]]}")
        if s == size:
            break
        seen, nxt = set(), []
        for b, _ in ranked[:width]:
            have = {k for k, _ in b}
            for c in top:
                if c[0] in have:
                    continue
                nb = sorted(b + [c])
                t = tuple(nb)
                if t not in seen:
                    seen.add(t)
                    nxt.append(nb)
        beams = nxt
    return results


# --------------------------------------------------------------------------- 3. playing the plans in the engine
class Runner:
    def __init__(self, seed, stake, plans, pool, max_steps=6000):
        self.K = len(plans)
        self.v = make(self.K, DECK, stake, seed, 1)
        root = make(1, DECK, stake, seed, 1)                           # (every environment of a vector has its own seed: copy the real game into all of them)
        self.v.copy_many(root, [0] * self.K, list(range(self.K)))
        self.plans, self.pool, self.max_steps = plans, pool, max_steps
        self.want = [[(item_id("j_" + k), pool[k]["ante"], pool[k]["pair"], pool[k]["price"]) for k, _ in sorted(p, key=lambda c: (pool[c[0]]["ante"], pool[c[0]]["pair"]))] for p in plans]
        self.plan_ids = [set(w[0] for w in ws) for ws in self.want]

    def run(self, log=None):
        K, v = self.K, self.v
        done = np.zeros(K, dtype=bool)
        blinds, won, ante_end = np.zeros(K, dtype=int), np.zeros(K, dtype=bool), np.zeros(K, dtype=int)
        paths = [[] for _ in range(K)]
        queue = [[] for _ in range(K)]                        # a round's plan, being played
        refresh = [dict() for _ in range(K)]
        in_shop = np.zeros(K, dtype=bool)
        last_owned = [[] for _ in range(K)]
        for step in range(self.max_steps):
            g = np.asarray(v.obs["g"]); mask = np.asarray(v.obs["mask"]); aid = np.asarray(v.obs["aid"]); aid2 = np.asarray(v.obs["aid2"]); jid = np.asarray(v.obs["jid"])
            act = np.full(K, -1, dtype=np.int64)
            need_solve = []
            for e in range(K):
                if done[e]:
                    continue
                m = mask[e]
                owned = set(int(x) for x in jid[e] if x > 0)
                last_owned[e] = sorted(owned)
                ante = int(round(g[e, 0] * 8))
                money = int(round(g[e, 17] * 50))
                if m[A_SELECT]:
                    act[e] = A_SELECT
                    in_shop[e] = False
                elif m[:A_SELECT].any():                                  # inside a round
                    in_shop[e] = False
                    if queue[e] and m[queue[e][0]]:
                        act[e] = queue[e].pop(0)
                    else:
                        queue[e] = []
                        need_solve.append(e)
                elif m[A_PSKIP] or m[A_PICK:A_PSKIP].any():
                    act[e] = A_PSKIP if m[A_PSKIP] else int(np.flatnonzero(m[A_PICK:A_PSKIP])[0]) + A_PICK
                elif m[A_LEAVE]:
                    if not in_shop[e]:
                        in_shop[e] = True
                        refresh[e][ante] = refresh[e].get(ante, 0) + 1
                    act[e] = self.shop_action(e, m, aid[e], aid2[e], owned, ante, money, refresh[e])
                    if act[e] == A_REROLL:
                        refresh[e][ante] = refresh[e].get(ante, 0) + 1
                else:
                    act[e] = int(np.flatnonzero(m)[0])
            if need_solve:
                plans = v.solve_rounds(need_solve, 1, 1000)
                for e, pl in zip(need_solve, plans):
                    if pl:
                        queue[e] = list(pl[0])
                        act[e] = queue[e].pop(0)
                    else:                                                  # the exact search found no line that clears the blind: play on and lose
                        plays = np.flatnonzero(mask[e][:A_SELECT])
                        act[e] = int(plays[0]) if len(plays) else int(np.flatnonzero(mask[e])[0])
            for e in range(K):
                if act[e] >= 0:
                    paths[e].append(int(act[e]))
            _, d, info = v.step(act)
            first = d & ~done
            blinds[first], won[first], ante_end[first] = info["blinds"][first], info["won"][first], info["ante"][first]
            done |= d
            if log and step % 400 == 0:
                log(f"    step {step}: {int(done.sum())}/{K} plans finished, best blinds so far {int(blinds.max())}")
            if done.all():
                break
        return blinds, won, ante_end, paths, last_owned

    def shop_action(self, e, m, aid, aid2, owned, ante, money, refresh):
        wants = [w for w in self.want[e] if w[0] not in owned]
        plan = self.plan_ids[e]
        # buy a planned joker that is on show and affordable
        for s in range(MAX_SHOP):
            if m[A_BUY + s] and int(aid[A_BUY + s]) in plan and int(aid[A_BUY + s]) not in owned:
                return A_BUY + s
        # the slots are full: sell something that is not in the plan for it (a replace action)
        for a in np.flatnonzero(m[A_REPLACE:A_REPLACE + MAX_SHOP * MAX_JOKERS]) + A_REPLACE:
            if int(aid[a]) in plan and int(aid[a]) not in owned and int(aid2[a]) not in plan:
                return int(a)
        # reroll toward the pair the next planned joker is due in, while its price stays in reach
        due = [w for w in wants if w[1] >= ante]
        if due:
            nxt = min(due, key=lambda w: (w[1], w[2]))
            seen = refresh.get(ante, 1) - 1                                 # the pair on show now
            if nxt[1] == ante and seen < nxt[2] and m[A_REROLL] and money - 5 >= nxt[3]:
                return A_REROLL
        return A_LEAVE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42003)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--antes", type=int, default=6, help="jokers on show in antes 1..this are candidates")
    p.add_argument("--pairs", type=int, default=8, help="refreshes of an ante's shop considered")
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--width", type=int, default=150)
    p.add_argument("--expand", type=int, default=50)
    p.add_argument("--plans", type=int, default=128, help="plans played in the engine")
    p.add_argument("--judge", default="checkpoints/judge2.pt")
    p.add_argument("--levels", default="6,6,6,6,6,6,6,6,6,6,6,6", help="hand levels the scoring assumes")
    p.add_argument("--beam-plans", type=int, default=24, help="the best plans handed to the beam search (the plan's jokers as a target); 0: skip")
    p.add_argument("--beam", type=int, default=32)
    p.add_argument("--retry-plans", type=int, default=4, help="the plans that got furthest are tried again with the wider beams")
    p.add_argument("--retry-beams", default="128,256")
    p.add_argument("--target-weight", type=float, default=3.0)
    p.add_argument("--money-weight", type=float, default=0.0)
    p.add_argument("--scale-weight", type=float, default=1.0)
    p.add_argument("--out-dir", default="traces/plan")
    a = p.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    pool, every = appearances(a.seed, a.stake, a.antes, a.pairs)
    log(f"seed {a.seed} {a.stake}: {len(pool)} different jokers on show in antes 1-{a.antes} ({len(every)} appearances)")
    scorer = Scorer(a.seed, a.stake, a.judge, dev, [int(x) for x in a.levels.split(",")])
    res = enumerate_builds(scorer, pool, a.size, a.width, a.expand, log)
    log(f"  scored {len(scorer.cache)} sets in {time.time() - t0:.0f}s")
    # the plans to play: the best of the sizes 5, 4 and 3
    chosen, seen = [], set()
    for s, take in ((a.size, a.plans // 2), (a.size - 1, a.plans // 4), (a.size - 2, a.plans // 4)):
        for b, sc in res.get(s, [])[:take]:
            t = tuple(sorted(k for k, _ in b))
            if t not in seen:
                seen.add(t)
                chosen.append((b, sc))
    plans = [b for b, _ in chosen]
    log(f"playing {len(plans)} plans in the exact engine")
    t1 = time.time()
    runner = Runner(a.seed, a.stake, plans, pool)
    blinds, won, ante_end, paths, owned = runner.run(log)
    log(f"  {time.time() - t1:.0f}s; wins {int(won.sum())}/{len(plans)}; blinds beaten: best {int(blinds.max())}, mean {blinds.mean():.1f}")
    order = np.argsort(-(won * 100 + blinds))
    names = lambda ids: [__import__("balatro_rl.sim.items", fromlist=["VOCAB"]).VOCAB[i][2:] for i in ids]
    for i in order[:8]:
        log(f"  {'WIN ' if won[i] else '    '}blinds {int(blinds[i]):2d} ante {int(ante_end[i])} score {chosen[i][1][0]:+.2f} pass {chosen[i][1][1]:.2f}  plan {[k for k, _ in plans[i]]}  held at the end {names(owned[i])}")
    os.makedirs(a.out_dir, exist_ok=True)
    best = int(order[0])
    if a.beam_plans:
        from solve_novel import solve as beam_solve
        log(f"handing the {a.beam_plans} best plans to the beam search (beam {a.beam}, target weight {a.target_weight})")
        ranked = sorted(range(len(plans)), key=lambda i: -chosen[i][1][0])[:a.beam_plans]
        results = []
        for i in ranked:
            tg = [KEYS.index(k) for k, _ in plans[i]]
            t = time.time()
            path, info = beam_solve(DECK, a.stake, a.seed, a.beam, a.beam // 2, 0.5, 8, 4, 4, 1000, a.money_weight, 0.0, a.scale_weight, target=tg, target_w=a.target_weight)
            lvl = "WIN" if path else info["level"]
            results.append((lvl, [k for k, _ in plans[i]], path))
            log(f"    plan {[k for k, _ in plans[i]]} (score {chosen[i][1][0]:+.2f}, pass {chosen[i][1][1]:.2f}) -> {lvl}  {time.time() - t:.0f}s")
            if path:
                json.dump({"seed": a.seed, "deck": DECK, "stake": a.stake, "solved": True, "actions": path, "plan": [k for k, _ in plans[i]]},
                          open(os.path.join(a.out_dir, f"{a.stake}_{a.seed}.json"), "w"))
                log(f"SOLVED seed {a.seed} with plan {[k for k, _ in plans[i]]}")
                return
        log("no plan was solved at beam %d; best level %s" % (a.beam, max(r[0] for r in results)))
        top = sorted(results, key=lambda r: -r[0])[:a.retry_plans]
        for width in [int(x) for x in a.retry_beams.split(",") if x]:
            for lvl0, keys, _ in top:
                tg = [KEYS.index(k) for k in keys]
                t = time.time()
                path, info = beam_solve(DECK, a.stake, a.seed, width, width // 4, 0.5, 8, 4, 4, 1000, a.money_weight, 0.0, a.scale_weight, target=tg, target_w=a.target_weight)
                lvl = "WIN" if path else info["level"]
                log(f"    beam {width}: plan {keys} -> {lvl}  {time.time() - t:.0f}s")
                if path:
                    json.dump({"seed": a.seed, "deck": DECK, "stake": a.stake, "solved": True, "actions": path, "plan": keys, "beam": width},
                              open(os.path.join(a.out_dir, f"{a.stake}_{a.seed}.json"), "w"))
                    log(f"SOLVED seed {a.seed} with plan {keys} at beam {width}")
                    return
    json.dump({"seed": a.seed, "deck": DECK, "stake": a.stake, "solved": bool(won[best]), "actions": paths[best], "plan": [k for k, _ in plans[best]], "blinds": int(blinds[best])},
              open(os.path.join(a.out_dir, f"{a.stake}_{a.seed}.json"), "w"))


if __name__ == "__main__":
    main()
