"""Paired comparison of two experiments on the same held-out seeds."""
from __future__ import annotations

import numpy as np

from .core import load_registry
from .metrics import load_evals, summary
from .stats import exact_binomial_two_sided, fmt_ci, mean_ci, paired_permutation


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
    # blinds: average over training seeds per held-out game, then pair games
    diffs = [np.mean([g["furthest"] for g in ga[s]]) - np.mean([g["furthest"] for g in gb[s]]) for s in common]
    # wins: pair run i of A with run i of B on the same held-out seed
    a_only = b_only = 0
    for s in common:
        for x, y in zip(ga[s], gb[s]):
            a_only += x["won"] and not y["won"]
            b_only += y["won"] and not x["won"]
    sa, sb = summary(a), summary(b)
    rule = sa["blinds"][1] > sb["blinds"][2]
    return {"n": len(common), "diff": mean_ci(diffs), "p_blinds": paired_permutation(diffs),
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
        f"{a} vs {b} on {r['n']} shared held-out seeds [{ta}]",
        f"blinds: {a} {fmt_ci(*r['a']['blinds'])} | {b} {fmt_ci(*r['b']['blinds'])}",
        f"paired difference ({a} - {b}): {fmt_ci(*r['diff'])}, permutation p = {r['p_blinds']:.4f}",
        f"wins: only {a} won {r['a_only_wins']} pairs, only {b} won {r['b_only_wins']}; exact McNemar p = {r['p_wins']:.4f}",
        f"promotion rule ({a}'s 95% CI entirely above {b}'s): {'PASS' if r['non_overlapping'] else 'fail'}",
    ])
