"""Diagnostics. Held-out win rate is the measure of success; everything here only helps read it."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def calibration(values, wins, games=None, buckets: int = 8, min_wins: int = 5) -> dict:
    """Bucket states by `values` (quantiles) and report the actual win rate per bucket.
    status: "rises" when the win rate goes up with the value (top bucket above bottom and a non-negative
    rank correlation over buckets), "FLAT_OR_FALLING" otherwise, "insufficient wins" when fewer than
    `min_wins` games were won (`games`: the game id of each state; states of one game are not independent)."""
    v = np.asarray(values, float)
    y = np.asarray(wins, float)
    n_won = len({gm for gm, w in zip(games, y) if w}) if games is not None else int(y.sum())
    out = {"n": int(len(v)), "won_games": n_won, "buckets": []}
    if len(v) == 0:
        out["status"] = "no data"
        return out
    edges = np.unique(np.quantile(v, np.linspace(0, 1, buckets + 1)))
    idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, max(0, len(edges) - 2))
    rates, means = [], []
    for b in range(max(1, len(edges) - 1)):
        m = idx == b
        if not m.any():
            continue
        rates.append(float(y[m].mean()))
        means.append(float(v[m].mean()))
        out["buckets"].append({"lo": float(edges[b]), "hi": float(edges[min(b + 1, len(edges) - 1)]),
                               "n": int(m.sum()), "value": round(means[-1], 4), "win_rate": round(rates[-1], 4)})
    if n_won < min_wins:
        out["status"] = "insufficient wins"
    elif len(rates) < 2:
        out["status"] = "one bucket"
    else:
        corr = _spearman(means, rates)
        out["spearman"] = round(corr, 3)
        out["status"] = "rises" if (rates[-1] > rates[0] and corr >= 0) else "FLAT_OR_FALLING"
    return out


def value_calibration(values, targets, in_round=None, buckets: int = 8) -> dict:
    """How well V predicts the value target z it is trained towards: rmse, and ece = the mean over
    quantile buckets of V (weighted by size) of |mean V - mean z|. With `in_round` (one flag per state),
    the rmse of in-round states and of the others (shop, packs, blind select) separately."""
    v, z = np.asarray(values, float), np.asarray(targets, float)
    if len(v) == 0:
        return {"n": 0}
    out = {"n": int(len(v)), "rmse": round(float(np.sqrt(np.mean((v - z) ** 2))), 4),
           "bias": round(float(np.mean(v - z)), 4)}
    edges = np.unique(np.quantile(v, np.linspace(0, 1, buckets + 1)))
    idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, max(0, len(edges) - 2))
    ece = 0.0
    for b in range(max(1, len(edges) - 1)):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(float(v[m].mean()) - float(z[m].mean()))
    out["ece"] = round(float(ece), 4)
    if in_round is not None:
        r = np.asarray(in_round, bool)
        for name, m in (("rmse_round", r), ("rmse_build", ~r)):
            if m.any():
                out[name] = round(float(np.sqrt(np.mean((v[m] - z[m]) ** 2))), 4)
    return out


def _spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def breakdown(infos) -> dict:
    """Held-out results: win rate overall, games (and wins) by ante reached, and per boss type the share
    of that boss's blinds cleared and the win rate of runs that met it."""
    n = len(infos)
    by_ante = defaultdict(lambda: [0, 0])
    for i in infos:
        a = "won" if i["won"] else f"ante {min(i['ante'], 8)}"
        by_ante[a][0] += 1
        by_ante[a][1] += int(i["won"])
    boss = defaultdict(lambda: [0, 0, 0, 0])            # met, cleared, runs, run wins
    for i in infos:
        seen = set()
        for key, cleared in i.get("bosses", []):
            boss[key][0] += 1
            boss[key][1] += int(cleared)
            if key not in seen:
                seen.add(key)
                boss[key][2] += 1
                boss[key][3] += int(i["won"])
    return {"win_rate": sum(i["won"] for i in infos) / max(1, n),
            "by_ante": {k: v[0] for k, v in sorted(by_ante.items())},
            "by_boss": {k: {"met": v[0], "cleared%": round(100 * v[1] / v[0], 1),
                            "run_win%": round(100 * v[3] / max(1, v[2]), 1)}
                        for k, v in sorted(boss.items(), key=lambda kv: -kv[1][0])}}


class HackAlarm:
    """Warns when the shaped return has risen for `n` evaluations in a row while the held-out win rate is
    flat or falling over the same evaluations (possible reward hacking)."""

    def __init__(self, n: int = 3):
        self.n = n
        self.history: list[tuple[float, float]] = []

    def update(self, shaped_return: float, win_rate: float) -> str | None:
        self.history.append((shaped_return, win_rate))
        h = self.history[-(self.n + 1):]
        if len(h) < self.n + 1:
            return None
        rising = all(b[0] > a[0] for a, b in zip(h, h[1:]))
        if rising and h[-1][1] <= h[0][1]:
            return (f"shaped return rose for {self.n} evaluations ({h[0][0]:.4f} -> {h[-1][0]:.4f}) while held-out "
                    f"win rate went {h[0][1]:.3f} -> {h[-1][1]:.3f}: possible reward hacking")
        return None


def mean_or_nan(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else math.nan
