"""The structure of recorded wins (tools/collect_wins.py output): replays each win from its seed and actions, and summarises the builds, how they formed over the antes,
the deck and hand levels, what was bought and what was done.

    python tools/analyze_wins.py traces/v7_wins100.jsonl --stake GOLD --json traces/v7_wins100_summary.json
"""
import argparse
import json
import os
import sys
from collections import Counter
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppEnv                    # noqa: E402
from balatro_rl.joker_desc import FEATURES, item_descriptors  # noqa: E402
from balatro_rl.sim.hands import HAND_NAMES                 # noqa: E402
from balatro_rl.sim.items import VOCAB                      # noqa: E402

KINDS = ("mult", "xmult", "chips", "grows", "economy")


def kinds_table():
    d = item_descriptors()
    col = {n: FEATURES.index(n) for n in KINDS}
    return {k[2:]: {n: bool(d[i, col[n]]) for n in KINDS} for i, k in enumerate(VOCAB) if k.startswith("j_")}


def action_kind(x):
    return ("play" if x < 218 else "discard" if x < 436 else "select" if x == 436 else "skip" if x == 437 else "boss_reroll" if x == 438 else
            "buy" if x < 444 else "buy_pack" if x < 446 else "voucher" if x < 448 else "reroll" if x == 448 else "leave" if x == 449 else
            "sell_joker" if x < 458 else "sell_cons" if x < 461 else "use" if x < 464 else "pick" if x < 469 else "pack_skip" if x == 469 else "swap/replace")


def replay(w, deck="RED"):
    e = CppEnv(deck, w["stake"])
    e.reset(w["seed"])
    ante_start = {}
    for x in w["actions"]:
        d = None
        if e.game.state == "BLIND_SELECT":
            d = e.game.dump()
            if d["blind_idx"] == 0 and d["ante"] not in ante_start:
                ante_start[d["ante"]] = {"jokers": [j["key"] for j in d["jokers"]], "money": d["money"], "levels": sum(l - 1 for l in d["hand_levels"][:9]),
                                         "deck": len(d["full_deck"])}
        _, _, done, _ = e.step(int(x))
        if done:
            break
    return e.game.dump(), ante_start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wins")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--json", default="")
    a = p.parse_args()
    wins = [json.loads(l) for l in open(a.wins) if l.strip()]
    kt = kinds_table()
    rows = []
    for w in wins:
        d, starts = replay(w)
        if d["state"] != "WON":
            continue
        rows.append((w, d, starts))
    n = len(rows)
    print(f"{n} of {len(wins)} wins replayed as WON ({a.stake})")
    out = {"n": n}
    # ---- the final builds
    jc = Counter(j["key"] for _, d, _ in rows for j in d["jokers"])
    print("\nJOKERS HELD AT THE WIN (share of wins holding it):")
    print("  ", [(k, round(c / n, 2)) for k, c in jc.most_common(22)])
    nj = Counter(len(d["jokers"]) for _, d, _ in rows)
    print("jokers held:", dict(sorted(nj.items())))
    per_kind = {kd: [sum(kt[j["key"]][kd] for j in d["jokers"]) for _, d, _ in rows] for kd in KINDS}
    print("per win, mean count of: " + ", ".join(f"{kd} {np.mean(v):.2f}" for kd, v in per_kind.items()))
    print("xMult jokers per win:", dict(sorted(Counter(per_kind['xmult']).items())), "| scalers per win:", dict(sorted(Counter(per_kind['grows']).items())))
    editions = Counter(j["edition"] for _, d, _ in rows for j in d["jokers"] if j["edition"])
    print("editions on held jokers:", dict(editions))
    out["joker_share"] = {k: round(c / n, 3) for k, c in jc.most_common(40)}
    # ---- combinations
    pairs = Counter()
    for _, d, _ in rows:
        ks = sorted(j["key"] for j in d["jokers"])
        pairs.update(combinations(ks, 2))
    print("\nMOST COMMON PAIRS:", [(f"{a_}+{b_}", c) for (a_, b_), c in pairs.most_common(10)])
    # ---- archetypes: the hand type played most, and a greedy grouping of builds by shared jokers
    most = Counter()
    for _, d, _ in rows:
        hp = list(d["hand_played"][:9])
        most[HAND_NAMES[int(np.argmax(hp))]] += 1
    print("\nHAND TYPE PLAYED MOST in the run:", dict(most.most_common()))
    sets = [frozenset(j["key"] for j in d["jokers"]) for _, d, _ in rows]
    groups = []
    for i, s in enumerate(sets):
        for g_ in groups:
            if len(s & sets[g_[0]]) >= 2:
                g_.append(i)
                break
        else:
            groups.append([i])
    groups.sort(key=len, reverse=True)
    print("BUILD GROUPS (builds sharing 2+ jokers with the group's first, greedy):", [len(g_) for g_ in groups[:10]], "groups of", n)
    for g_ in groups[:4]:
        c = Counter(k for i in g_ for k in sets[i])
        print(f"   group of {len(g_)}: core {[k for k, v in c.most_common(5)]}")
    # ---- how the build formed
    print("\nHOW IT FORMED (mean over wins that reached the ante): jokers / xMult / scalers / money / hand-level ups / deck size at the start of each ante")
    for ante in range(1, 9):
        ss = [s[ante] for _, _, s in rows if ante in s]
        if not ss:
            continue
        print(f"   ante {ante}: n {len(ss):3d} | jokers {np.mean([len(x['jokers']) for x in ss]):.2f} | xmult {np.mean([sum(kt[k]['xmult'] for k in x['jokers']) for x in ss]):.2f} | "
              f"scalers {np.mean([sum(kt[k]['grows'] for k in x['jokers']) for x in ss]):.2f} | money ${np.mean([x['money'] for x in ss]):.0f} | levels +{np.mean([x['levels'] for x in ss]):.1f} | deck {np.mean([x['deck'] for x in ss]):.1f}")
    # of the final jokers, when did they appear
    first = Counter()
    for _, d, st in rows:
        for j in d["jokers"]:
            when = next((an for an in sorted(st) if j["key"] in st[an]["jokers"]), 9)
            first[when] += 1
    print("final jokers by the first ante start they were already held at (9 = bought during ante 8):", dict(sorted(first.items())))
    # ---- hand levels, deck, vouchers
    lv = np.array([d["hand_levels"][:9] for _, d, _ in rows])
    print("\nHAND LEVELS at the win (mean):", {HAND_NAMES[i]: round(float(lv[:, i].mean()), 1) for i in range(9)}, "| total level-ups mean", round(float((lv - 1).sum(1).mean()), 1))
    dk = [len(d["full_deck"]) for _, d, _ in rows]
    enh = [sum(1 for c in d["full_deck"] if c[2]) for _, d, _ in rows]
    print("DECK: size mean %.1f (min %d max %d) | enhanced cards mean %.1f | with seals %.1f | with editions %.1f" % (
        np.mean(dk), min(dk), max(dk), np.mean(enh), np.mean([sum(1 for c in d["full_deck"] if c[4]) for _, d, _ in rows]), np.mean([sum(1 for c in d["full_deck"] if c[3]) for _, d, _ in rows])))
    vc = Counter(v for _, d, _ in rows for v in (d["vouchers"] if isinstance(d["vouchers"], (list, tuple, set)) else []))
    print("VOUCHERS held (share):", [(k, round(c / n, 2)) for k, c in vc.most_common(10)] if vc else d["vouchers"])
    # ---- what was done
    ac = Counter()
    for w, _, _ in rows:
        ac.update(action_kind(x) for x in w["actions"])
    print("\nACTIONS per win:", {k: round(c / n, 1) for k, c in ac.most_common()})
    print("money at the win: mean $%.0f | blinds skipped mean %.1f | planets used mean %.1f | tarots used mean %.1f | actions mean %.0f" % (
        np.mean([d["money"] for _, d, _ in rows]), np.mean([d["blinds_skipped"] for _, d, _ in rows]), np.mean([d.get("planet_uses", 0) for _, d, _ in rows]),
        np.mean([d.get("tarots_used", 0) for _, d, _ in rows]), np.mean([len(w["actions"]) for w, _, _ in rows])))
    print("final boss:", dict(Counter(d["boss"] for _, d, _ in rows).most_common()))
    if a.json:
        json.dump(out, open(a.json, "w"))


if __name__ == "__main__":
    main()
