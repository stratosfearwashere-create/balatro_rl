"""Compact text summaries of an experiment (never raw logs)."""
from __future__ import annotations

import collections
import json

import numpy as np

from .core import config, exp_dir, load_registry
from .stats import fmt_ci, mean_ci_clustered, wilson


def load_evals(exp_id: str) -> dict[int, list[dict]]:
    out = {}
    for d in sorted(exp_dir(exp_id).glob("seed*")):
        f = d / "eval.json"
        if f.exists():
            out[int(d.name[4:])] = json.loads(f.read_text())["games"]
    return out


def load_logs(exp_id: str) -> dict[int, list[dict]]:
    out = {}
    for d in sorted(exp_dir(exp_id).glob("seed*")):
        f = d / "model_log.jsonl"
        if f.exists():
            out[int(d.name[4:])] = [json.loads(l) for l in f.read_text().splitlines() if l.startswith("{")]
    return out


def summary(exp_id: str) -> dict:
    """The numbers behind `metrics` (also used by compare, promote and report)."""
    evals = load_evals(exp_id)
    groups = [[g["furthest"] for g in games] for games in evals.values()]
    games = [g for gs in evals.values() for g in gs]
    if not games:
        return {}
    m, lo, hi = mean_ci_clustered(groups)
    wins = sum(g["won"] for g in games)
    return {"blinds": (m, lo, hi), "per_seed": {s: float(np.mean([g["furthest"] for g in gs])) for s, gs in evals.items()},
            "wins": (wins, len(games)), "win_ci": wilson(wins, len(games)), "games": len(games)}


def _curve(rows: list[dict], key: str, points: int = 8) -> str:
    rows = [r for r in rows if key in r]
    if not rows:
        return "-"
    idx = np.linspace(0, len(rows) - 1, min(points, len(rows))).astype(int)
    return " ".join(f"{rows[i][key]:g}" for i in idx)


def text(exp_id: str) -> str:
    reg = load_registry()
    exp = reg["experiments"].get(exp_id)
    if exp is None:
        return f"unknown experiment {exp_id}"
    tier = config()["tiers"][exp["tier"]]
    max_blinds = 3 * tier["win_ante"]
    used = sum(r.get("minutes", 0) for r in exp.get("runs", []))
    out = [f"{exp_id} '{exp['spec'].get('name', '')}' [{exp['tier']}] {exp['status']}; branch {exp['branch']} "
           f"@ {exp['commit'][:8]}; {used:.0f}/{exp['budget_minutes']} min"]
    for r in exp.get("runs", []):
        out.append(f"  seed {r['seed']}: {r['status']}, {r.get('minutes', 0):.0f} min; eval {r.get('eval', '-')}")
    evals = load_evals(exp_id)
    games = [g for gs in evals.values() for g in gs]
    if games:
        s = summary(exp_id)
        out.append(f"blinds cleared (of {max_blinds}): {fmt_ci(*s['blinds'])}  per seed: "
                   + ", ".join(f"{k}: {v:.2f}" for k, v in s["per_seed"].items()) + f"  ({len(games)} runs)")
        p, wlo, whi = s["win_ci"]
        out.append(f"win rate: {s['wins'][0]}/{s['wins'][1]} = {fmt_ci(p, wlo, whi, pct=True)}")
        antes = collections.Counter("won" if g["won"] else f"a{g['ante']}" for g in games)
        order = sorted((k for k in antes if k != "won"), key=lambda k: int(k[1:])) + (["won"] if "won" in antes else [])
        out.append("run ended at: " + ", ".join(f"{k} {100 * antes[k] / len(games):.0f}%" for k in order))
        for a in ("4", "6"):
            v = [g["ratio_at_ante"][a] for g in games if a in g["ratio_at_ante"]]
            if v:
                out.append(f"best hand / target at the start of ante {a}: median {np.median(v):.2f} "
                           f"({len(v)} runs reached it)")
    logs = load_logs(exp_id)
    for seed, rows in logs.items():
        if not rows:
            continue
        out.append(f"training seed {seed} ({len(rows)} iterations, env steps {rows[-1].get('env_steps', '?')}):")
        for key, label in (("ret", "return"), ("blinds", "blinds (train)"), ("ent", "entropy"), ("kl", "kl")):
            out.append(f"  {label:15s} {_curve(rows, key)}")
        by_phase = collections.defaultdict(list)
        for r in rows:
            if "v" in r:
                by_phase[r.get("phase", "ppo")].append(r["v"])
        out.append("  value loss by phase: " + ", ".join(f"{k} {np.mean(v):.3f}" for k, v in by_phase.items()))
    return "\n".join(out)
