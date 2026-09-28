"""Statistics without scipy: bootstrap intervals, Wilson intervals, paired permutation and exact tests."""
from __future__ import annotations

import math

import numpy as np


def mean_ci_clustered(groups: list[list[float]], reps: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% CI of values grouped by training seed: resample seeds, then values within each
    (so seed-to-seed variation widens the interval, as it should)."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    allv = np.concatenate(groups)
    if len(groups) == 1:
        return mean_ci(allv, reps, seed)
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        stats[i] = np.mean(np.concatenate([groups[j][rng.integers(0, len(groups[j]), len(groups[j]))] for j in pick]))
    return float(allv.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mean_ci(values, reps: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), (reps, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def paired_permutation(diffs, reps: int = 20000, seed: int = 0) -> float:
    """Two-sided p-value for mean(diffs) != 0 by random sign flips."""
    d = np.asarray(diffs, float)
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(reps, len(d)))
    return float((np.abs((flips * d).mean(1)) >= obs - 1e-12).mean())


def exact_binomial_two_sided(k: int, n: int) -> float:
    """McNemar's exact test: k of n discordant pairs went one way, p = 0.5 under no difference."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(min(1.0, sum(pr for pr in probs if pr <= probs[k] + 1e-15)))


def fmt_ci(m, lo, hi, digits=2, pct=False) -> str:
    if pct:
        return f"{100 * m:.{digits - 1}f}% [{100 * lo:.{digits - 1}f}, {100 * hi:.{digits - 1}f}]"
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
