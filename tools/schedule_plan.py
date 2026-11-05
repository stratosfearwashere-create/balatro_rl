"""Plan backward from the queue: which joker sets can clear Gold's ante 8 boss, and where in the seed's shop queue each piece can be bought.

    python tools/schedule_plan.py --seeds 42005 --show        # the builds and their schedules for a seed
    python tools/schedule_plan.py --seeds 41001-41010 --solve # then run the novelty beam towards each build, strongest first

1. queue: for each ante the jokers the shop and its rerolls show (exact-seed mode: a fixed sequence per ante), with price, edition, stickers.
2. builds: sets of up to SIZE jokers from the queue, built up greedily (a beam over additions) and ranked by the expected best play at the
   last ante (marginal.py's scorer; growing jokers at their potential; every hand type at LEVEL) against the ante-8 boss's target / HANDS.
3. schedule: each piece at its earliest (ante, reroll position) and the cheapest copy, with the running cost against a rough income.
4. search: the novelty beam with the build's jokers as ranking targets (the C++ search's `target` bonus).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.env import order_jokers                                      # noqa: E402
from balatro_rl.marginal import POTENTIAL, ROUNDS_AHEAD, _Scratch               # noqa: E402
from balatro_rl.sim.game import Game                                        # noqa: E402
from balatro_rl.sim.jokers import JOKERS, Joker                             # noqa: E402
from balatro_rl.sim.seedrng import seed_string                              # noqa: E402

KEYS = list(JOKERS)
SIZE, LEVEL, HANDS = 5, 4, 4
GROWTH = 0.6                          # the share of its listed growth rate a deliberate player gets out of a growing joker
START_MONEY, INCOME = 4, 11          # dollars at the start, and a rough net income per ante (three payouts, a little interest, minus rerolls)


def queue(seed, stake, antes=8, rolls=9):
    """Every joker the seed's shops show: ante, reroll position (0: the shop as it opens), slot, price, edition, stickers."""
    g = Game(seed=1, deck_type="RED", stake=stake, exact_seed=seed_string(seed))
    g.money = 10_000
    out = []
    for ante in range(1, antes + 1):
        g.ante = ante
        g.shop, g.shop_packs, g.shop_vouchers, g.ante_voucher = [], [], [], None
        g.open_shop()
        for r in range(rolls + 1):
            if r:
                g.free_rerolls = 1
                g.reroll()
            for slot, it in enumerate(g.shop):
                if it.kind == "joker":
                    j = it.joker
                    out.append(dict(ante=ante, r=r, slot=slot, key=it.key[2:], cost=int(it.cost), ed=j.edition, eternal=bool(j.eternal),
                                    perishable=j.perishable is not None, rental=bool(j.rental)))
    return out


class Rater:
    """Expected best play of a set of jokers at ante 8, over the sampled hands of marginal.py."""

    def __init__(self, seed, stake):
        self.g = Game(seed=1, deck_type="RED", stake=stake, exact_seed=seed_string(seed))
        self.g.ante = 8
        self.target = self.g.blind_target(2)
        self.cache = {}

    def score(self, items):
        key = tuple(sorted((i["key"], i["ed"]) for i in items))
        if key in self.cache:
            return self.cache[key]
        g = self.g.clone()
        g.ante = 8
        g.hand_levels = [LEVEL] * len(g.hand_levels)
        g.jokers = []
        for i in items:
            j = Joker(i["key"], i["ed"], False, None, False, i["cost"], 0, i["cost"])
            g.add_joker(j, force=True)
        order_jokers(g)
        ante_of = {i["key"]: i["ante"] for i in items}
        for j in g.jokers:
            if j.key in POTENTIAL:                                  # grown over the rounds between buying it and ante 8's boss
                delta, v0 = POTENTIAL[j.key]
                rounds = 3 * (8 - ante_of.get(j.key, 8))
                j.state["val"] = v0 + delta / ROUNDS_AHEAD * rounds * GROWTH
        s = _Scratch(g)
        s.prepare()
        self.cache[key] = s.base
        return s.base


def candidates(q, rater, keep=26):
    """One copy of each joker (the cheapest of the earliest ante it appears in), without rental or perishable stickers, best singles first."""
    best = {}
    for i in q:
        if i["rental"] or i["perishable"]:
            continue
        k = i["key"]
        if k not in best or (i["ante"], i["cost"], i["r"]) < (best[k]["ante"], best[k]["cost"], best[k]["r"]):
            best[k] = i
    scored = sorted(best.values(), key=lambda i: -rater.score([i]))
    return scored[:keep]


def builds(cands, rater, width=10, size=SIZE):
    """A beam over additions: the strongest sets of up to `size` jokers (score = expected best play at ante 8)."""
    beam = [((), 0.0)]
    seen, allb = set(), []
    for _ in range(size):
        nxt = []
        for combo, _s in beam:
            have = {c["key"] for c in combo}
            for c in cands:
                if c["key"] in have:
                    continue
                new = tuple(sorted(combo + (c,), key=lambda x: x["key"]))
                sig = tuple(x["key"] for x in new)
                if sig in seen:
                    continue
                seen.add(sig)
                nxt.append((new, rater.score(list(new))))
        nxt.sort(key=lambda t: -t[1])
        beam = nxt[:width]
        allb += beam
    allb.sort(key=lambda t: -t[1])
    return allb


def schedule(combo):
    """The purchases in order of availability, with the running cost against a rough income, and whether it can be afforded in time."""
    rows, spent, ok = [], 0, True
    for c in sorted(combo, key=lambda x: (x["ante"], x["r"])):
        spent += c["cost"]
        have = START_MONEY + INCOME * (c["ante"] - 1) + (INCOME if c["ante"] > 1 else 0) * 0
        rows.append(dict(ante=c["ante"], reroll=c["r"], slot=c["slot"], key=c["key"], ed=c["ed"], cost=c["cost"], spent=spent, can=spent <= have + INCOME))
        ok &= spent <= have + INCOME
    return rows, ok


def plan(seed, stake, top=6, log=print):
    t = time.time()
    q = queue(seed, stake)
    rater = Rater(seed, stake)
    cands = candidates(q, rater)
    bl = builds(cands, rater)
    out, seen = [], set()
    for combo, sc in bl:
        sig = tuple(sorted(c["key"] for c in combo))
        if sig in seen or len(combo) < 3:
            continue
        seen.add(sig)
        rows, ok = schedule(combo)
        out.append(dict(score=sc, ratio=HANDS * sc / rater.target, afford=ok, schedule=rows))
    out.sort(key=lambda b: (-b["afford"], -b["ratio"]))
    log(f"seed {seed}: {len(q)} jokers in the queue, {len(cands)} candidates, {len(bl)} builds scored, ante-8 boss {rater.target:,}, {time.time() - t:.0f}s")
    return out[:top]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--seeds", default="42005")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--show", action="store_true")
    p.add_argument("--solve", action="store_true")
    p.add_argument("--beam", type=int, default=32)
    p.add_argument("--elite", type=int, default=16)
    p.add_argument("--target-weight", type=float, default=3.0)
    p.add_argument("--out-dir", default="traces/schedule")
    a = p.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")] if "-" in a.seeds else [int(a.seeds)] * 2
    os.makedirs(a.out_dir, exist_ok=True)
    wins = 0
    for sd in range(lo, hi + 1):
        bl = plan(sd, a.stake, a.top)
        if a.show:
            for b in bl:
                print(f"  ratio {b['ratio']:6.2f} afford {b['afford']}: " + ", ".join(
                    f"{r['key']}{'(' + r['ed'].lower() + ')' if r['ed'] else ''} a{r['ante']}r{r['reroll']} ${r['cost']}" for r in b["schedule"]))
        if a.solve:
            from solve_novel import solve
            won, bestlvl = None, 0
            for b in bl:
                keys = [KEYS.index(r["key"]) for r in b["schedule"]]
                t = time.time()
                path, info = solve("RED", a.stake, sd, a.beam, a.elite, 0.5, 8, 4, 4, 1000, 0.0, 0.0, 1.0, target=keys, target_w=a.target_weight)
                lvl = 24 if path else info["level"]
                bestlvl = max(bestlvl, lvl)
                print(f"    {[r['key'] for r in b['schedule']]} -> {'WIN' if path else lvl}  ({time.time() - t:.0f}s)", flush=True)
                if path:
                    won = b
                    json.dump({"seed": sd, "deck": "RED", "stake": a.stake, "solved": True, "actions": path, "schedule": b["schedule"]},
                              open(os.path.join(a.out_dir, f"{a.stake}_{sd}.json"), "w"))
                    break
            wins += won is not None
            print(f"  => {'SOLVED' if won else 'best level %d' % bestlvl}", flush=True)
    if a.solve:
        print(f"{wins}/{hi - lo + 1} solved", flush=True)


if __name__ == "__main__":
    main()
