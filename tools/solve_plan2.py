"""Backward planning over a deterministic seed, version 2: the plan also says which hand types to level up, which blinds to skip and how much money to keep.

    1. LIST       the jokers the shops of antes 1..A show (pair k of an ante's card sequence is on show after k refreshes), with prices.
    2. ENUMERATE  candidate final builds (up to five jokers) and score each against the ante-8 boss with the judge; every scaler has grown to the counter it would have
                  by then from the ante it is first on show. A beam over sets: a few thousand scored sets.
    3. LEVELS     for the best sets, which hand types are worth levelling (the judge's score with that hand at level 10): the plan buys those planets (shop cards and
                  celestial packs) and uses them.
    4. PLAY       every plan x variant (money reserve, skipping the small / big blinds from some ante) is played in the exact engine, all at once: blinds with the exact
                  round solver, shops by the plan's rules (buy its jokers when on show and affordable, selling something outside the plan if the slots are full; reroll
                  toward the pair its next joker is due in; buy planets for its hands while the money kept stays above the reserve; leave).
    5. BEAM       the plans that got furthest are handed to the beam search with their jokers as a target (the beam handles money, packs and skips exactly).

A live page (traces/plan_live.html) shows where the solve is.

    python tools/solve_plan2.py --seed 42003 --stake GOLD
"""
import argparse
import html
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import solve_plan as P                                                           # noqa: E402  (loads the development engine)
from solve_plan import KEYS, appearances, counter_at_8, enumerate_builds         # noqa: E402
from solve_fast import make                                                      # noqa: E402
from balatro_rl.env import (A_BUY, A_BUY_PACK, A_LEAVE, A_PICK, A_PSKIP, A_REPLACE, A_REROLL, A_SELECT, A_SKIP, A_USE_C, F_JOKER_W, MAX_JOKERS, MAX_SHOP, item_id)   # noqa: E402
from balatro_rl.sim.items import PLANETS, VOCAB                                  # noqa: E402
from solve_plan import GROW, DECK                                                      # noqa: E402
GROWERS = {item_id("j_" + k) for k, g in GROW.items() if g[1] > 0}                 # the jokers that grow with play
SCRIPTED = -3                                                                    # in a path: select the blind and let the scaler-friendly scripted player play the round

HAND_NAMES = ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"]
PLANET_OF = {h: n for n, h in PLANETS.items()}
CELESTIAL = {item_id(k) for k in VOCAB if k.startswith("p_celestial")}


# --------------------------------------------------------------------------- the live page
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="4"><title>Plan search</title><style>
:root{--bg:#fafaf7;--ink:#1d1d1b;--muted:#6b6b66;--grid:#e4e3dc;--card:#fff;--c1:#2a6fb0;--c2:#d9822b;--c3:#3c9a5f;--bad:#b3402f}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--ink:#ecebe6;--muted:#9a9990;--grid:#2c2d30;--card:#202225;--c1:#6aa8e6;--c2:#f0a95a;--c3:#5cc384;--bad:#e5806e}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}main{max-width:1000px;margin:0 auto;padding:18px 16px 40px}
h1{font-size:20px;margin:0 0 2px}h2{font-size:14px;margin:0 0 8px}.sub{color:var(--muted);margin:0 0 14px;font-size:13px}
.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.steps{display:flex;gap:8px;flex-wrap:wrap}.step{border:1px solid var(--grid);border-radius:8px;padding:6px 10px;color:var(--muted)}
.step.done{color:var(--ink);border-color:var(--c3)}.step.now{color:var(--ink);border-color:var(--c2);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:3px 8px;text-align:left;border-top:1px solid var(--grid)}th{color:var(--muted);font-weight:500}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}.bar{height:10px;border-radius:3px;background:var(--c1);display:inline-block;vertical-align:middle}
.win{color:var(--c3);font-weight:600}.mut{color:var(--muted)}.big{font-size:22px;font-weight:600}</style></head><body><main>
<h1>Plan search: seed __SEED__, __STAKE__, __DECK__ deck</h1><p class="sub">__ELAPSED__ s. Updated __NOW__ (refreshes every 4 s). __RESULT__</p>
<div class="card"><div class="steps">__STEPS__</div></div>__BODY__</main></body></html>"""


class Live:
    STAGES = ["list the shops", "score builds", "choose levels", "play plans", "beam search"]

    def __init__(self, path, seed, stake):
        self.path, self.seed, self.stake, self.t0 = path, seed, stake, time.time()
        self.stage, self.done, self.result, self.sections = 0, set(), "", {}
        self.render()

    def set(self, name, body):
        self.sections[name] = body
        self.render()

    def at(self, stage):
        self.done |= set(range(stage))
        self.stage = stage
        self.render()

    def finish(self, text):
        self.done |= set(range(len(self.STAGES)))
        self.result = text
        self.render()

    def render(self):
        steps = "".join(f'<div class="step {"done" if i in self.done else "now" if i == self.stage else ""}">{i + 1}. {s}</div>' for i, s in enumerate(self.STAGES))
        body = "".join(self.sections[k] for k in sorted(self.sections))
        res = f'<span class="big win">{html.escape(self.result)}</span>' if self.result else ""
        out = (PAGE.replace("__SEED__", str(self.seed)).replace("__STAKE__", self.stake).replace("__DECK__", DECK.title()).replace("__ELAPSED__", str(int(time.time() - self.t0)))
               .replace("__NOW__", time.strftime("%H:%M:%S")).replace("__RESULT__", res).replace("__STEPS__", steps).replace("__BODY__", body))
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        open(self.path + ".tmp", "w", encoding="utf-8").write(out)
        os.replace(self.path + ".tmp", self.path)


def table(head, rows, num=()):
    h = "".join(f'<th class="{"n" if i in num else ""}">{html.escape(str(x))}</th>' for i, x in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"n" if i in num else ""}">{x}</td>' for i, x in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


# --------------------------------------------------------------------------- scoring with a levels profile
class Scorer2(P.Scorer):
    def score_levels(self, builds, levels_list):
        """The judge's (log10 share, pass chance) for each build under its own hand levels."""
        import torch
        out = []
        for c0 in range(0, len(builds), self.n):
            ch = list(range(c0, min(c0 + self.n, len(builds))))
            for slot, i in enumerate(ch):
                ids = [KEYS.index(k) for k, _ in builds[i]]
                self.v.set_build([slot], 8, 2, ids, [counter_at_8(k, a) for k, a in builds[i]], 0, levels_list[i], 25)
            m = len(ch)
            g = torch.from_numpy(np.asarray(self.v.obs["g"])[:m]).to(self.dev)
            jid = torch.from_numpy(np.asarray(self.v.obs["jid"])[:m]).to(self.dev)
            jf = torch.from_numpy(np.asarray(self.v.obs["jf"])[:m][:, :, :F_JOKER_W]).to(self.dev)
            cid = torch.from_numpy(np.asarray(self.v.obs["cid"])[:m]).to(self.dev)
            g, jf = torch.nan_to_num(g, nan=0.0, posinf=10.0, neginf=-10.0), torch.nan_to_num(jf, nan=0.0, posinf=10.0, neginf=-10.0)
            br = bp = None
            with torch.no_grad():
                for mode in (0, 1):
                    o = self.judge.predict(g, jid, jf, cid, torch.full((m,), mode, device=self.dev, dtype=torch.long))
                    r, p = o["log_ratio"].float(), o["p_pass"].float()
                    br = r if br is None else torch.maximum(br, r)
                    bp = p if bp is None else torch.maximum(bp, p)
            out += list(zip(br.cpu().numpy().tolist(), bp.cpu().numpy().tolist()))
        return out

    def best_hands(self, build, top=2, level=10):
        """The hand types that raise the build's score most when levelled."""
        lv = []
        for h in range(12):
            L = [1] * 12
            L[h] = level
            lv.append(L)
        sc = self.score_levels([build] * 12, lv)
        order = sorted(range(12), key=lambda h: -sc[h][0])
        return order[:top], sc


# --------------------------------------------------------------------------- playing the plans
class Runner2:
    def __init__(self, seed, stake, variants, pool, max_steps=8000, feed=True):
        self.K = len(variants)
        self.v = make(self.K, DECK, stake, seed, 1)
        root = make(1, DECK, stake, seed, 1)                           # (every environment of a vector has its own seed: copy the real game into all of them)
        self.v.copy_many(root, [0] * self.K, list(range(self.K)))
        self.var, self.pool, self.max_steps = variants, pool, max_steps
        self.want = [[(item_id("j_" + k), pool[k]["ante"], pool[k]["pair"], pool[k]["price"]) for k in sorted(var["keys"], key=lambda k: (pool[k]["ante"], pool[k]["pair"]))] for var in variants]
        self.plan_ids = [set(w[0] for w in ws) for ws in self.want]
        self.planets = [set(item_id("c_" + PLANET_OF[h]) for h in var["hands"]) for var in variants]
        self.feed = feed
        self.scratch = make(self.K, DECK, stake, seed, 0) if feed else None
        self.scripted_rounds = np.zeros(self.K, dtype=int)

    def run(self, log=None, live=None):
        K, v = self.K, self.v
        done = np.zeros(K, dtype=bool)
        blinds, won, ante_end = np.zeros(K, dtype=int), np.zeros(K, dtype=int) > 0, np.zeros(K, dtype=int)
        paths = [[] for _ in range(K)]
        queue = [[] for _ in range(K)]
        refresh = [dict() for _ in range(K)]
        in_shop = np.zeros(K, dtype=bool)
        last_owned = [[] for _ in range(K)]
        levels_bought = np.zeros(K, dtype=int)
        for step in range(self.max_steps):
            g = np.asarray(v.obs["g"]); mask = np.asarray(v.obs["mask"]); aid = np.asarray(v.obs["aid"]); aid2 = np.asarray(v.obs["aid2"])
            jid = np.asarray(v.obs["jid"]); cid = np.asarray(v.obs["cid"])
            act = np.full(K, -1, dtype=np.int64)
            need_solve = []
            sel = []
            for e in range(K):
                if done[e]:
                    continue
                m = mask[e]
                owned = set(int(x) for x in jid[e] if x > 0)
                last_owned[e] = sorted(owned)
                ante = int(round(g[e, 0] * 8))
                money = int(round(g[e, 17] * 50))
                var, plan, planets = self.var[e], self.plan_ids[e], self.planets[e]
                use = [A_USE_C + i for i in range(3) if m[A_USE_C + i] and int(cid[e][i]) in planets]
                if use:
                    act[e] = use[0]
                    levels_bought[e] += 1
                elif m[A_SELECT]:
                    in_shop[e] = False
                    act[e] = A_SKIP if (m[A_SKIP] and ante >= var["skip_from"]) else A_SELECT
                    if act[e] == A_SELECT and (owned & GROWERS):
                        sel.append(e)
                elif m[:A_SELECT].any():                                              # inside a round
                    in_shop[e] = False
                    if queue[e] and m[queue[e][0]]:
                        act[e] = queue[e].pop(0)
                    else:
                        queue[e] = []
                        need_solve.append(e)
                elif m[A_PSKIP] or m[A_PICK:A_PSKIP].any():                          # a pack is open: take a planet of ours or a joker of the plan, else nothing
                    pick = None
                    for i in np.flatnonzero(m[A_PICK:A_PSKIP]):
                        it = int(aid[e][A_PICK + i])
                        if it in planets or (it in plan and it not in owned):
                            pick = A_PICK + int(i)
                            break
                    act[e] = pick if pick is not None else (A_PSKIP if m[A_PSKIP] else int(np.flatnonzero(m)[0]))
                elif m[A_LEAVE]:
                    if not in_shop[e]:
                        in_shop[e] = True
                        refresh[e][ante] = refresh[e].get(ante, 0) + 1
                    act[e] = self.shop_action(e, m, aid[e], aid2[e], owned, ante, money, refresh[e], var, plan, planets)
                    if act[e] == A_REROLL:
                        refresh[e][ante] = refresh[e].get(ante, 0) + 1
                else:
                    act[e] = int(np.flatnonzero(m)[0])
            scripted = []
            if self.feed and sel:                                                     # would the scaler-friendly player clear this blind and grow a counter?
                self.scratch.copy_many(v, sel, sel)
                self.scratch.set_auto_round(True, 1.5, 1)
                self.scratch.set_modes([1] * K)
                a_ = np.full(K, -1, dtype=np.int64)
                a_[sel] = A_SELECT
                self.scratch.step_split(a_)
                rr = self.scratch.round_report()
                for e in sel:
                    grew = (rr["c1"][e] - rr["c0"][e])
                    slots = [i for i in range(MAX_JOKERS) if int(jid[e][i]) in GROWERS]
                    if rr["cleared"][e] and slots and max(grew[i] for i in slots) > 0:
                        scripted.append(e)
                        act[e] = -1
            if need_solve:
                plans = v.solve_rounds(need_solve, 1, 1000)
                for e, pl in zip(need_solve, plans):
                    if pl:
                        queue[e] = list(pl[0])
                        act[e] = queue[e].pop(0)
                    else:
                        plays = np.flatnonzero(mask[e][:A_SELECT])
                        act[e] = int(plays[0]) if len(plays) else int(np.flatnonzero(mask[e])[0])
            for e in range(K):
                if act[e] >= 0:
                    paths[e].append(int(act[e]))
            _, d, info = v.step(act)
            first = d & ~done
            blinds[first], won[first], ante_end[first] = info["blinds"][first], info["won"][first], info["ante"][first]
            done |= d
            if scripted:
                v.set_auto_round(True, 1.5, 1)
                v.set_modes([1] * K)
                a2 = np.full(K, -1, dtype=np.int64)
                a2[scripted] = A_SELECT
                _, d2, info2 = v.step(a2)
                v.set_auto_round(False, 1.5, 0)
                for e in scripted:
                    paths[e].append(SCRIPTED)
                    self.scripted_rounds[e] += 1
                first2 = d2 & ~done
                blinds[first2], won[first2], ante_end[first2] = info2["blinds"][first2], info2["won"][first2], info2["ante"][first2]
                done |= d2
            if live and step % 150 == 0:
                live(step, int(done.sum()), K, int(blinds.max()))
            if done.all():
                break
        return blinds, won, ante_end, paths, last_owned, levels_bought

    def shop_action(self, e, m, aid, aid2, owned, ante, money, refresh, var, plan, planets):
        wants = [w for w in self.want[e] if w[0] not in owned]
        reserve = var["reserve"]
        for s in range(MAX_SHOP):                                                    # a joker of the plan, on show and affordable
            if m[A_BUY + s] and int(aid[A_BUY + s]) in plan and int(aid[A_BUY + s]) not in owned:
                return A_BUY + s
        for a in np.flatnonzero(m[A_REPLACE:A_REPLACE + MAX_SHOP * MAX_JOKERS]) + A_REPLACE:   # the slots are full: swap out something outside the plan
            if int(aid[a]) in plan and int(aid[a]) not in owned and int(aid2[a]) not in plan:
                return int(a)
        due = [w for w in wants if w[1] >= ante]
        need = min(due, key=lambda w: (w[1], w[2]))[3] if due else 0                 # the price of the next joker it is waiting for
        spare = money - (need if due and due[0][1] <= ante + 1 else 0)               # money beyond what that joker will cost, when it is near
        for s in range(MAX_SHOP):                                                    # a planet for one of its hands
            if m[A_BUY + s] and int(aid[A_BUY + s]) in planets and spare - 3 >= reserve:
                return A_BUY + s
        for s in range(2):                                                           # a celestial pack
            if m[A_BUY_PACK + s] and int(aid[A_BUY_PACK + s]) in CELESTIAL and spare - 4 >= reserve:
                return A_BUY_PACK + s
        if due:                                                                      # reroll toward the pair the next joker is due in
            nxt = min(due, key=lambda w: (w[1], w[2]))
            seen = refresh.get(ante, 1) - 1
            if nxt[1] == ante and seen < nxt[2] and m[A_REROLL] and money - 5 >= nxt[3]:
                return A_REROLL
        return A_LEAVE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42003)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--antes", type=int, default=6)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--width", type=int, default=150)
    p.add_argument("--expand", type=int, default=50)
    p.add_argument("--plans", type=int, default=48, help="builds that get hands and variants and are played")
    p.add_argument("--judge", default="checkpoints/judge2.pt")
    p.add_argument("--levels", default="6,6,6,6,6,6,6,6,6,6,6,6", help="hand levels the first scoring assumes")
    p.add_argument("--reserves", default="0,8,25")
    p.add_argument("--skips", default="99,3", help="skip the small and big blinds from this ante on (99: never)")
    p.add_argument("--round-feed", type=float, default=3.0, help="dollars of preference per natural round of growth a round plan gives the build's scalers (the exact solver; 0: money only)")
    p.add_argument("--no-feed", action="store_true", help="play every round with the exact solver only (no scaler-friendly rounds)")
    p.add_argument("--beam-plans", type=int, default=16)
    p.add_argument("--beam", type=int, default=32)
    p.add_argument("--retry-plans", type=int, default=3)
    p.add_argument("--retry-beams", default="128")
    p.add_argument("--target-weight", type=float, default=3.0)
    p.add_argument("--scale-weight", type=float, default=1.0)
    p.add_argument("--out-dir", default="traces/plan2")
    p.add_argument("--page", default="traces/plan_live.html")
    a = p.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda s: print(s, flush=True)
    live = Live(a.page, a.seed, a.stake)
    scorer_env = make(1, DECK, a.stake, a.seed, 0)
    scorer_env.set_round_feed(a.round_feed)                # (a process-wide setting of the engine: the plan runner and the beam search use it too)
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    # ---- 1. the shops
    live.at(0)
    pool, every = appearances(a.seed, a.stake, a.antes, a.pairs)
    log(f"seed {a.seed} {a.stake}: {len(pool)} different jokers on show in antes 1-{a.antes} ({len(every)} appearances)")
    byante = {}
    for r in pool.values():
        byante.setdefault(r["ante"], []).append(r)
    rows = [(f"ante {an}", ", ".join(f"{r['key']} (${r['price']}, pair {r['pair']})" for r in sorted(rs, key=lambda r: r["pair"]))) for an, rs in sorted(byante.items())]
    live.set("1", f'<div class="card"><h2>Jokers on show, by the ante they first appear in</h2>{table(["", "joker (price, which refresh)"], rows)}</div>')
    # ---- 2. the builds
    live.at(1)
    scorer = Scorer2(a.seed, a.stake, a.judge, dev, [int(x) for x in a.levels.split(",")])
    res = enumerate_builds(scorer, pool, a.size, a.width, a.expand, log)
    sizes = []
    for s in sorted(res):
        b, sc = res[s][0]
        sizes.append((s, len(res[s]), f"{sc[0]:+.2f}", f"{sc[1]:.2f}", ", ".join(k for k, _ in b)))
    live.set("2", f'<div class="card"><h2>Best build of each size (judge, ante-8 boss, hand levels {a.levels.split(",")[0]})</h2>{table(["size", "sets scored", "log10 share of the target", "pass chance", "jokers"], sizes, num=(1, 2, 3))}</div>')
    log(f"  scored {len(scorer.cache)} sets in {time.time() - t0:.0f}s")
    # ---- 3. the hands
    live.at(2)
    cand = []
    for s, take in ((a.size, a.plans * 2 // 3), (a.size - 1, a.plans // 3)):
        cand += [b for b, _ in res.get(s, [])[:take]]
    plans = []
    for b in cand:
        hands, _ = scorer.best_hands(b, top=2)
        lv = [1] * 12
        lv[hands[0]], lv[hands[1]] = 10, 6
        sc = scorer.score_levels([b], [lv])[0]
        plans.append({"build": b, "keys": [k for k, _ in b], "hands": hands, "score": sc})
    plans.sort(key=lambda x: -x["score"][0])
    live.set("3", '<div class="card"><h2>Best builds with their hands levelled (the two hands that raise each build most: first at level 10, second at 6)</h2>' + table(
        ["jokers", "levelled hands", "log10 share", "pass chance"],
        [(", ".join(x["keys"]), " + ".join(HAND_NAMES[h] for h in x["hands"]), f"{x['score'][0]:+.2f}", f"{x['score'][1]:.2f}") for x in plans[:12]], num=(2, 3)) + "</div>")
    # ---- 4. play the plans
    live.at(3)
    variants = []
    for x in plans:
        for r in [int(z) for z in a.reserves.split(",")]:
            for sk in [int(z) for z in a.skips.split(",")]:
                variants.append({**x, "reserve": r, "skip_from": sk})
    log(f"playing {len(variants)} plan variants in the exact engine")
    runner = Runner2(a.seed, a.stake, variants, pool, feed=not a.no_feed)
    t1 = time.time()
    prog = lambda step, nd, K, bm: live.set("4", f'<div class="card"><h2>Playing {K} plan variants in the exact engine</h2><p>step {step}: {nd}/{K} finished, furthest {bm} blinds</p></div>')
    blinds, won, ante_end, paths, owned, lvl_used = runner.run(log, prog)
    order = np.argsort(-(won * 100 + blinds))
    names = lambda ids: [VOCAB[i][2:] for i in ids]
    rows = [(("WIN " if won[i] else "") + str(int(blinds[i])), int(ante_end[i]), ", ".join(variants[i]["keys"]), " + ".join(HAND_NAMES[h] for h in variants[i]["hands"]), variants[i]["reserve"],
             "never" if variants[i]["skip_from"] > 8 else f"ante {variants[i]['skip_from']}+", int(lvl_used[i]), int(runner.scripted_rounds[i]), ", ".join(names(owned[i]))) for i in order[:12]]
    live.set("4", f'<div class="card"><h2>Plans played in the exact engine: {len(variants)} variants, {int(won.sum())} wins, best {int(blinds.max())} blinds, mean {blinds.mean():.1f} (played in {time.time() - t1:.0f}s)</h2>' + table(
        ["blinds", "ante", "plan", "levelled hands", "money kept", "skips", "planets used", "scaler rounds", "held at the end"], rows, num=(0, 1, 4, 6, 7)) + "</div>")
    log(f"  played in {time.time() - t1:.0f}s; wins {int(won.sum())}/{len(variants)}; blinds best {int(blinds.max())}, mean {blinds.mean():.1f}")
    for i in order[:6]:
        log(f"  {'WIN ' if won[i] else '    '}blinds {int(blinds[i]):2d} ante {int(ante_end[i])} plan {variants[i]['keys']} hands {[HAND_NAMES[h] for h in variants[i]['hands']]} reserve {variants[i]['reserve']} skip {variants[i]['skip_from']} planets used {int(lvl_used[i])}")
    if won.any():
        i = int(order[0])
        json.dump({"seed": a.seed, "deck": DECK, "stake": a.stake, "solved": True, "actions": paths[i], "plan": variants[i]["keys"], "note": "-3 = select the blind with the scaler-friendly scripted player (auto-round mode 1)"}, open(os.path.join(a.out_dir, f"{a.stake}_{a.seed}.json"), "w"))
        live.finish(f"SOLVED by the plan: {', '.join(variants[i]['keys'])}")
        log(f"SOLVED seed {a.seed} by the plan runner: {variants[i]['keys']}")
        return
    # ---- 5. the beam
    live.at(4)
    from solve_novel import solve as beam_solve
    ranked = []
    seen = set()
    for x in plans:
        t = tuple(sorted(x["keys"]))
        if t not in seen:
            seen.add(t)
            ranked.append(x)
    ranked = ranked[:a.beam_plans]
    results = []

    def beam_page():
        mx = max([r[0] for r in results if r[0] != "WIN"] + [1])
        rows = [(", ".join(r[1]), f'<span class="bar" style="width:{(r[3] if r[0] == "WIN" else r[0]) / 24 * 220:.0f}px"></span> {"WIN" if r[0] == "WIN" else r[0]}', r[2], f"{r[4]:.0f}s") for r in results]
        live.set("5", '<div class="card"><h2>Beam search with the plan as a target (the number is the level reached; 24 is a win)</h2>' + table(["plan", "level", "beam", "time"], rows) + "</div>")

    for x in ranked:
        t = time.time()
        path, info = beam_solve(DECK, a.stake, a.seed, a.beam, a.beam // 2, 0.5, 8, 4, 4, 1000, 0.0, 0.0, a.scale_weight, target=[KEYS.index(k) for k in x["keys"]], target_w=a.target_weight)
        lvl = "WIN" if path else info["level"]
        results.append((lvl, x["keys"], a.beam, 24 if path else info["level"], time.time() - t))
        log(f"    beam {a.beam}: plan {x['keys']} -> {lvl}  {time.time() - t:.0f}s")
        beam_page()
        if path:
            json.dump({"seed": a.seed, "deck": DECK, "stake": a.stake, "solved": True, "actions": path, "plan": x["keys"], "beam": a.beam}, open(os.path.join(a.out_dir, f"{a.stake}_{a.seed}.json"), "w"))
            live.finish(f"SOLVED by the beam with the plan: {', '.join(x['keys'])}")
            log(f"SOLVED seed {a.seed} with plan {x['keys']}")
            return
    top = sorted(results, key=lambda r: -r[3])[:a.retry_plans]
    for width in [int(z) for z in a.retry_beams.split(",") if z]:
        for r in top:
            t = time.time()
            path, info = beam_solve(DECK, a.stake, a.seed, width, width // 4, 0.5, 8, 4, 4, 1000, 0.0, 0.0, a.scale_weight, target=[KEYS.index(k) for k in r[1]], target_w=a.target_weight)
            lvl = "WIN" if path else info["level"]
            results.append((lvl, r[1], width, 24 if path else info["level"], time.time() - t))
            log(f"    beam {width}: plan {r[1]} -> {lvl}  {time.time() - t:.0f}s")
            beam_page()
            if path:
                json.dump({"seed": a.seed, "deck": DECK, "stake": a.stake, "solved": True, "actions": path, "plan": r[1], "beam": width}, open(os.path.join(a.out_dir, f"{a.stake}_{a.seed}.json"), "w"))
                live.finish(f"SOLVED by the beam (width {width}) with the plan: {', '.join(r[1])}")
                log(f"SOLVED seed {a.seed} with plan {r[1]} at beam {width}")
                return
    live.finish(f"not solved: best level {max(r[3] for r in results)} of 24")
    log(f"not solved; best level {max(r[3] for r in results)}")


if __name__ == "__main__":
    main()
