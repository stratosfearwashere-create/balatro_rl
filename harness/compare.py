"""Paired comparison of two experiments on the same held-out seeds."""
from __future__ import annotations

import numpy as np

from .core import load_registry
from .metrics import load_evals, summary
from .stats import exact_binomial_two_sided, fmt_ci, paired_diff_hierarchical


def result(a: str, b: str) -> dict:
    ea, eb = load_evals(a), load_evals(b)
    ga = {}
    for seed, games in ea.items():
        for g in games:
            ga.setdefault(g["seed"], []).append(g)
    gb = {}
    for seed, games in eb.items():
        for g in games:
            gb.setdefault(g["seed"], []).append(g)
    common = sorted(set(ga) & set(gb))
    if not common:
        return {"error": "no held-out seeds in common (different tiers or no evaluation yet)"}
    # blinds: a [training seeds x shared games] table per experiment, compared with a bootstrap that
    # resamples training seeds as well as games (so a lucky seed can't carry the result)
    def table(ev):
        return np.array([[{g["seed"]: g["furthest"] for g in games}[s] for s in common] for games in ev.values()
                         if all(s in {g["seed"] for g in games} for s in common)])
    ta, tb = table(ea), table(eb)
    # wins: pair run i of A with run i of B on the same held-out seed
    a_only = b_only = 0
    for s in common:
        for x, y in zip(ga[s], gb[s]):
            a_only += x["won"] and not y["won"]
            b_only += y["won"] and not x["won"]
    sa, sb = summary(a), summary(b)
    rule = sa["blinds"][1] > sb["blinds"][2]
    d, lo, hi, p = paired_diff_hierarchical(ta, tb)
    return {"n": len(common), "seeds": (len(ta), len(tb)), "diff": (d, lo, hi), "p_blinds": p,
            "a_only_wins": a_only, "b_only_wins": b_only,
            "p_wins": exact_binomial_two_sided(min(a_only, b_only), a_only + b_only),
            "a": sa, "b": sb, "non_overlapping": rule}


def text(a: str, b: str) -> str:
    reg = load_registry()
    for e in (a, b):
        if e not in reg["experiments"]:
            return f"unknown experiment {e}"
    ta, tb = reg["experiments"][a]["tier"], reg["experiments"][b]["tier"]
    if ta != tb:
        return f"different tiers ({a}: {ta}, {b}: {tb}): not comparable"
    r = result(a, b)
    if "error" in r:
        return r["error"]
    return "\n".join([
        f"{a} vs {b} on {r['n']} shared held-out seeds, {r['seeds'][0]} vs {r['seeds'][1]} training seeds [{ta}]",
        f"blinds: {a} {fmt_ci(*r['a']['blinds'])} | {b} {fmt_ci(*r['b']['blinds'])}",
        f"difference ({a} - {b}): {fmt_ci(*r['diff'])}, p = {r['p_blinds']:.4f} "
        f"(bootstrap over training seeds and games)",
        f"wins: only {a} won {r['a_only_wins']} pairs, only {b} won {r['b_only_wins']}; exact McNemar p = {r['p_wins']:.4f}",
        f"promotion rule ({a}'s 95% CI entirely above {b}'s): {'PASS' if r['non_overlapping'] else 'fail'}",
    ])
