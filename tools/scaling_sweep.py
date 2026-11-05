"""How far does a scaling joker really scale if you play for it?

For each deterministic seed, scaling joker and starting amount of money: the game starts at ante 1 with the joker in hand (its counter at its start value) and that much money; blinds are not
required to be beaten (immortal mode: a failed blind just moves on, only the ante-8 boss ends the run), so only the joker's growth matters. A beam search over the exact-seed game
(every level = one blind: its round by the exact solver, with a preference for lines that feed the growing jokers, then the shop searched action by action: buys, rerolls, packs, sales ...)
keeps the states in which the joker's counter is highest. The counter, the ante, the money and what the counter is worth in score (the exact scorer's expected best play with the
counter as it is against the counter reset, in decades) are recorded at every blind.

    python tools/scaling_sweep.py --seeds 60001-60040 --jokers all --money 0,15,40 --out traces/scaling_sweep.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import solve_plan as P                                                           # noqa: E402  (loads the development engine: set_build, set_state)
from solve_plan import KEYS, DECK                                                # noqa: E402
from solve_fast import make                                                      # noqa: E402
from balatro_rl.env import (A_BUY, A_BUY_PACK, A_DISC, A_LEAVE, A_PICK, A_PSKIP, A_REROLL, A_SELECT, A_SELL_C, A_USE_C, global_size, item_id)   # noqa: E402
from balatro_rl.sim.items import PLANETS, VOCAB                                  # noqa: E402
from balatro_rl.marginal import MARG_G, POT_G                                    # noqa: E402
from balatro_rl.potential import GROW                                            # noqa: E402

BASE = global_size() - MARG_G - POT_G                                            # observation column: log10(expected best play) / 7
GROWERS = [k for k, g in GROW.items() if g[1] > 0]


def counter_of(jid_row, jf_row, jid):
    """The counter of the joker with item id `jid` in a state's observation (-1: not held)."""
    for j in range(len(jid_row)):
        if int(jid_row[j]) == jid:
            return float(np.expm1(4.0 * jf_row[j, 11]))
    return -1.0




# --------------------------------------------------------------------------- feeding routines for the jokers a plain search does not feed
def _ids(prefix):
    return {item_id(k) for k in VOCAB if k.startswith(prefix)}


PLANET_IDS = {item_id("c_" + n) for n in PLANETS}
PACKS = {"standard": _ids("p_standard"), "arcana": _ids("p_arcana"), "celestial": _ids("p_celestial"), "spectral": _ids("p_spectral")}
ENHANCERS = [item_id("c_" + n) for n in ("magician", "empress", "heirophant", "lovers", "chariot", "justice", "devil", "tower")]
# what each joker is fed by: the packs to buy and the cards worth picking from them (in order of preference); None: any
FEEDERS = {
    "hologram": {"packs": ["standard"], "picks": None},                                   # a playing card added to the deck
    "constellation": {"packs": ["celestial"], "picks": None, "planets": True},           # a planet used
    "vampire": {"packs": ["arcana"], "picks": ENHANCERS},                                 # enhanced cards that score
    "lucky_cat": {"packs": ["arcana"], "picks": [item_id("c_magician")]},                 # lucky cards that trigger
    "glass": {"packs": ["arcana"], "picks": [item_id("c_justice")]},                      # glass cards that shatter
    "caino": {"packs": ["arcana", "spectral"], "picks": [item_id("c_hanged_man"), item_id("c_familiar"), item_id("c_immolate"), item_id("c_grim"), item_id("c_incantation")]},   # face cards destroyed
    "campfire": {"packs": [], "picks": None, "cycle": True},                              # cards sold
}


class Feeder:
    """A scripted run through one shop visit for one joker. act(obs row, money) -> an action, or None when it is done."""

    def __init__(self, key, frac, money0):
        self.f = FEEDERS.get(key)
        self.reserve = (1.0 - frac) * money0
        self.rerolls = 0
        self.steps = 0

    def act(self, m, aid, cid, subs, money):
        self.steps += 1
        if self.f is None or self.steps > 40:
            return None
        f = self.f
        if m[:A_SELECT].any():                                                   # a tarot waits for its target cards: the largest legal set
            legal = np.flatnonzero(m[:A_SELECT])
            return int(legal[np.argmax([(subs[a] >= 0).sum() for a in legal])])
        if m[A_PSKIP] or m[A_PICK:A_PSKIP].any():                                 # a pack is open
            picks = [(a, int(aid[a])) for a in range(A_PICK, A_PSKIP) if m[a]]
            if f["picks"] is None:
                return picks[0][0] if picks else A_PSKIP
            for want in f["picks"]:
                for a, it in picks:
                    if it == want:
                        return a
            return A_PSKIP
        if f.get("planets"):
            for i in range(3):
                if m[A_USE_C + i] and int(cid[i]) in PLANET_IDS:
                    return A_USE_C + i
        if f.get("cycle"):                                                       # buy a cheap card and sell it again
            for i in range(3):
                if m[A_SELL_C + i]:
                    return A_SELL_C + i
            if money - 3 >= self.reserve:
                for s in range(4):
                    if m[A_BUY + s] and VOCAB[int(aid[A_BUY + s])].startswith("c_"):
                        return A_BUY + s
            return None
        want_packs = set().union(*[PACKS[x] for x in f["packs"]]) if f["packs"] else set()
        for s in range(2):
            if m[A_BUY_PACK + s] and int(aid[A_BUY_PACK + s]) in want_packs and money - 4 >= self.reserve:
                return A_BUY_PACK + s
        if f.get("planets"):
            for s in range(4):
                if m[A_BUY + s] and int(aid[A_BUY + s]) in PLANET_IDS and money - 3 >= self.reserve:
                    return A_BUY + s
        if self.rerolls < 3 and m[A_REROLL] and money - 5 >= self.reserve and f["packs"]:
            self.rerolls += 1
            return A_REROLL
        return None


def macro_children(cur, n, key, jid, fracs, mac, money0):
    """The states one blind on, reached by the scaler-friendly scripted round at full investment and the joker's feeding routine at each spending share in `fracs`.
    -> (the slots of `mac` that are still in the game, the post-round counters of all)."""
    K = n * len(fracs)
    src = [i for i in range(n) for _ in fracs]
    fr = [f for _ in range(n) for f in fracs]
    mac.copy_many(cur, src, list(range(K)))
    mac.set_auto_round(True, 1.5, 2)
    mac.set_margin_params(1.6, 24, 1.0)
    mac.set_modes([2] * mac.n)
    jidm = np.asarray(mac.obs["jid"])
    slot_of = [next((j for j in range(jidm.shape[1]) if int(jidm[e, j]) == jid), -1) for e in range(K)]
    act = np.full(mac.n, -1, dtype=np.int64)
    act[:K] = A_SELECT
    _, d, _, _, _ = mac.step_split(act)
    rr = mac.round_report()
    post = np.array([rr["c1"][e][slot_of[e]] if slot_of[e] >= 0 else -1.0 for e in range(K)])
    done = np.array(d[:K], dtype=bool)
    feeders = [Feeder(key, fr[e], money0) for e in range(K)]
    fin = done.copy()
    for step in range(60):
        if fin.all():
            break
        o = {k: np.asarray(mac.obs[k]) for k in ("mask", "aid", "cid", "subs", "g")}
        a = np.full(mac.n, -1, dtype=np.int64)
        for e in range(K):
            if fin[e]:
                continue
            money = int(round(o["g"][e, 17] * 50))
            x = feeders[e].act(o["mask"][e], o["aid"][e], o["cid"][e], o["subs"][e], money)
            if x is None:
                x = A_LEAVE
                fin[e] = True
            a[e] = x
        _, d2, _, _, _ = mac.step_split(a)
        done |= np.array(d2[:K], dtype=bool)
        fin |= np.array(d2[:K], dtype=bool)
    return [e for e in range(K) if not done[e]], post


def run_one(seed, key, money, stake, beam, ends, sw, sl, feed_w, max_levels=45, macros=True):
    """-> list of per-blind records [(ante, blind, counter, decades, money)] along the best line, or None."""
    jid = item_id("j_" + key)
    kj = KEYS.index(key)
    mk = lambda n, marg: make(n, DECK, stake, seed, marg)
    root = mk(1, 1)
    root.set_immortal(True, False)
    root.v.restart()                                                              # (a new game under the immortal rules, the same seed)
    root.set_marginal(1)
    root.set_build([0], 1, 0, [kj], [float("nan")], 0, [], money)
    cur, dest, scr, two = mk(beam, 1), mk(beam * ends, 1), mk(beam * ends + 2 * beam, 1), mk(2, 1)
    mac = mk(2 * beam, 1)
    for e in (cur, dest, scr, two, mac):
        e.set_immortal(True, False)
        e.set_round_feed(feed_w)
    cur.copy_many(root, [0], [0])
    n = 1
    rec = []
    init = float(GROW[key][0])
    for level in range(max_levels):
        res = cur.expand_level(list(range(n)), dest, ends, sw, sl, 1000, 0.0, 0.0, 0.0, [], 0.0)
        pool = []
        for b, (win, wpath, cands) in enumerate(res):
            for path, score, sig, slot, jk in cands:
                pool.append((score, sig, slot))
        ng = len(pool)
        if ng:
            scr.copy_many(dest, [c[2] for c in pool], list(range(ng)))
        peak_extra = {}
        if macros:                                                                  # + the children made by the feeding routines (a scripted round and shop for this joker)
            mslots, post = macro_children(cur, n, key, jid, (1.0, 0.5), mac, money)
            if mslots:
                scr.copy_many(mac, mslots, list(range(ng, ng + len(mslots))))
                for k_, e_ in enumerate(mslots):
                    pool.append((-1.0, ("m", level, e_), ng + k_))
                    peak_extra[ng + k_] = float(post[e_])
        if not pool:
            break
        jidm, jfm = np.asarray(scr.obs["jid"]), np.asarray(scr.obs["jf"])
        ctr = np.array([counter_of(jidm[i], jfm[i], jid) for i in range(len(pool))])
        order = sorted(range(len(pool)), key=lambda i: (-ctr[i], -pool[i][0]))
        keep, seen = [], set()
        for i in order:
            if pool[i][1] in seen or ctr[i] < 0:
                continue
            seen.add(pool[i][1])
            keep.append(i)
            if len(keep) >= beam:
                break
        if not keep:
            break
        g0 = np.asarray(scr.obs["g"])[keep[0]]
        rec.append((int(round(g0[0] * 8)), int(np.argmax(g0[1:4])), float(ctr[keep[0]]), float(g0[BASE]), float(g0[17] * 50.0), max(float(ctr[keep[0]]), peak_extra.get(keep[0], -1.0))))
        cur.copy_many(scr, keep, list(range(len(keep))))
        n = len(keep)
        # what the counter is worth: the same state with the counter back at its start value
        two.copy_many(cur, [0, 0], [0, 1])
        cs = np.full((2, 8), np.nan)
        jc = np.asarray(cur.obs["jid"])[0]
        for j in range(len(jc)):
            if int(jc[j]) == jid:
                cs[1, j] = init
        two.set_state([0, 1], 0, 0, cs)
        gg = np.asarray(two.obs["g"])
        rec[-1] = rec[-1][:3] + (7.0 * float(gg[0, BASE] - gg[1, BASE]),) + rec[-1][4:]
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="60001-60020")
    p.add_argument("--jokers", default="all", help="'all' (the growing jokers) or a comma list of keys")
    p.add_argument("--money", default="0,15,40")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--beam", type=int, default=12)
    p.add_argument("--ends", type=int, default=10)
    p.add_argument("--shop-width", type=int, default=8)
    p.add_argument("--shop-layers", type=int, default=10)
    p.add_argument("--feed", type=float, default=20.0, help="the round solver's preference (dollars per natural round of growth) for lines that feed the growing jokers")
    p.add_argument("--no-macros", action="store_true", help="the generic search only (no feeding routines)")
    p.add_argument("--out", default="traces/scaling_sweep.jsonl")
    a = p.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")]
    jokers = GROWERS if a.jokers == "all" else a.jokers.split(",")
    moneys = [int(x) for x in a.money.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["seed"], r["joker"], r["money"]))
            except ValueError:
                pass
    t0, n = time.time(), 0
    for seed in range(lo, hi + 1):
        for key in jokers:
            for money in moneys:
                if (seed, key, money) in done:
                    continue
                t = time.time()
                rec = run_one(seed, key, money, a.stake, a.beam, a.ends, a.shop_width, a.shop_layers, a.feed, macros=not a.no_macros)
                with open(a.out, "a") as f:
                    f.write(json.dumps({"seed": seed, "joker": key, "money": money, "levels": rec}) + "\n")
                n += 1
                last = rec[-1] if rec else None
                print(f"seed {seed} {key:14s} ${money:<3d} -> {len(rec)} blinds, final counter {last[2] if last else float('nan'):.2f} (ante {last[0] if last else 0})  {time.time() - t:.1f}s  [{n} runs, {(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
