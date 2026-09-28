"""Why runs were lost: cause-of-death tags, money over time, sticker breakdown and text replays."""
from __future__ import annotations

import collections
import statistics

import numpy as np

from .core import load_registry
from .metrics import load_evals


WEAK = 0.6          # a run that scores less than this share of the losing blind's target had a weak build
STICKERS = ("none", "eternal", "perishable", "rental")
XMULT = {"cavendish", "duo", "trio", "family", "order", "tribe", "card_sharp", "acrobat", "blackboard",
         "flower_pot", "seeing_double", "stencil", "loyalty_card", "steel_joker", "baseball", "throwback",
         "drivers_license", "constellation", "ramen", "madness", "vampire", "hologram", "campfire", "glass",
         "hit_the_road", "caino", "yorick", "obelisk", "lucky_cat", "photograph", "idol", "ancient",
         "bloodstone", "triboulet", "baron"}


def causes(game: dict) -> list[str]:
    """Tags for a lost run, most important first. Rules:
    - weak build: scored under 60% of the losing blind's target
    - boss blind (<name>): got at least 60% of the way, at a boss blind
    - scoring fell short at ante X: got at least 60% of the way, at a small or big blind
    - broke (secondary): under $5 in the median of the last two shops, with an empty joker slot"""
    d = game.get("death")
    if not d:
        return []
    if d["frac"] < WEAK:
        tags = ["weak build"]
    elif d["blind"] == 2 and d.get("boss"):
        tags = [f"boss blind ({d['boss']})"]
    else:
        tags = [f"scoring fell short at ante {d['ante']}"]
    last = game["money"][-2:]
    if last and statistics.median(m[2] for m in last) < 5 and last[-1][3] < last[-1][4]:
        tags.append("broke")
    return tags


def text(exp_id: str, n: int = 5) -> str:
    reg = load_registry()
    exp = reg["experiments"].get(exp_id)
    if exp is None:
        return f"unknown experiment {exp_id}"
    evals = load_evals(exp_id)
    games = [g for gs in evals.values() for g in gs]
    lost = [g for g in games if not g["won"]]
    if not games:
        return f"{exp_id}: no evaluation yet (status {exp['status']})"
    out = [f"{exp_id} failures: {len(lost)} of {len(games)} evaluation runs lost"]
    primary = collections.Counter(causes(g)[0].split(" (")[0] if causes(g)[0].startswith("boss") else causes(g)[0]
                                  for g in lost if causes(g))
    out.append("cause of death: " + ", ".join(f"{k} {100 * v / len(lost):.0f}%" for k, v in primary.most_common()))
    broke = sum("broke" in causes(g) for g in lost)
    out.append(f"broke at death: {100 * broke / max(1, len(lost)):.0f}% of losses")
    bosses = collections.Counter(g["death"]["boss"] for g in lost if g.get("death") and g["death"]["blind"] == 2)
    if bosses:
        out.append("bosses that ended runs: " + ", ".join(f"{k} {v}" for k, v in bosses.most_common(6)))
    where = collections.Counter((g["death"]["ante"], "SBB"[g["death"]["blind"]]) for g in lost if g.get("death"))
    out.append("where: " + ", ".join(f"a{a}{b} {v}" for (a, b), v in sorted(where.items())))
    # money over time: median money and empty-slot share when leaving the shop, by ante
    by_ante = collections.defaultdict(list)
    for g in games:
        for ante, _, money, nj, slots in g["money"]:
            by_ante[ante].append((money, nj < slots))
    out.append("money leaving the shop, median by ante (empty joker slot %): " + ", ".join(
        f"a{a} ${int(np.median([m for m, _ in v]))} ({100 * np.mean([e for _, e in v]):.0f}%)"
        for a, v in sorted(by_ante.items())))
    st = {k: np.mean([g["stickers_held"][k] for g in games]) for k in ("eternal", "perishable", "rental")}
    out.append(f"stickers (jokers held per run): eternal {st['eternal']:.2f}, perishable {st['perishable']:.2f}, "
               f"rental {st['rental']:.2f}; rental paid ${np.mean([g['rental_paid'] for g in games]):.1f} per run")
    at_death = collections.Counter()
    for g in lost:
        for j in g["jokers"]:
            for k in ("eternal", "perishable", "rental"):
                at_death[k] += j[k]
    out.append(f"stickered jokers held at death (per loss): " + ", ".join(
        f"{k} {at_death[k] / max(1, len(lost)):.2f}" for k in ("eternal", "perishable", "rental")))
    # does it avoid jokers in general, or Gold's stickered ones? (older evaluations lack these fields)
    offered = [g["joker_offers"] for g in games if "joker_offers" in g]
    if offered:
        tot = {k: [sum(o[k][0] for o in offered), sum(o[k][1] for o in offered)] for k in STICKERS}
        out.append("jokers offered -> bought, by sticker: " + ", ".join(
            f"{k} {o}->{b} ({100 * b / max(1, o):.0f}%)" for k, (o, b) in tot.items()))
        took = [g for g in games if "joker_offers" in g]
        bought_stickered = [g["furthest"] for g in took if any(g["joker_offers"][k][1] for k in STICKERS[1:])]
        plain_only = [g["furthest"] for g in took if not any(g["joker_offers"][k][1] for k in STICKERS[1:])]
        if bought_stickered and plain_only:
            out.append(f"blinds cleared: runs that bought a stickered joker {np.mean(bought_stickered):.2f} "
                       f"({len(bought_stickered)} runs), runs that bought only plain ones {np.mean(plain_only):.2f} "
                       f"({len(plain_only)} runs)")
    no_x = sum(1 for g in lost if not any(j["key"] in XMULT or j.get("edition") == "POLYCHROME" for j in g["jokers"]))
    out.append(f"losses with no x-mult joker at death: {no_x} of {len(lost)}")
    # replays: spread over causes
    traced = [g for g in lost if g["trace"]]
    picked, seen = [], set()
    for g in traced:
        c = causes(g)[0]
        if c not in seen:
            picked.append(g)
            seen.add(c)
    picked += [g for g in traced if g not in picked]
    for g in picked[:n]:
        d = g["death"]
        jok = ", ".join(j["key"] + "".join(f"[{k}]" for k in ("eternal", "perishable", "rental") if j[k])
                        for j in g["jokers"]) or "none"
        out.append(f"\n--- seed {g['seed']}: {' + '.join(causes(g))}; died a{d['ante']} {'SBB'[d['blind']]} "
                   f"with {d['frac']:.0%} of {d['target']}, best hand {d['ratio']:.0%} x {d['hands']} hands; "
                   f"${d['money']}; jokers: {jok}")
        lines = g["trace"]
        if len(lines) > 30:
            lines = lines[:6] + [f"... ({len(lines) - 30} lines)"] + lines[-24:]
        out += ["  " + ln for ln in lines]
    return "\n".join(out)
