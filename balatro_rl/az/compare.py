"""Compare variants of the agent on the same held-out seeds (the evaluation standard for behaviour changes).

A grid file (JSON) names the variants; each has AgentConfig overrides, an optional checkpoint, optional
reward-config overrides (e.g. the bounded value) and whether it searches:

    {"base":      {"cfg": {}},
     "no_search": {"cfg": {}, "search": false},
     "tau0.2":    {"cfg": {"tau": 0.2, "c_scale": 0.5}, "rewards": {"potential": {"value_bound": "floor_sigmoid"}}}}

    python -m balatro_rl.az.compare grid --grid grid.json --out checkpoints/eval/stage1 --games 300
    python -m balatro_rl.az.compare table --out checkpoints/eval/stage1 --base base
    python -m balatro_rl.az.compare changed --grid grid.json --out checkpoints/eval/stage1 --base base --games 40

grid   plays every variant greedily on seeds EVAL_SEED0 .. (the same for all) and writes <out>/<name>.json
       (train.evaluate's result); variants already written are skipped.
changed  how often a variant decides differently from the base variant *in the same position*: the base
       plays the game, and at each of its decisions the variant is asked what it would do there, both from
       the same random seed for that decision. (Whole games can't be compared move by move: they part ways
       at the first different move.) Written to <out>/<name>.changed.json.
table  variant | games/hour | blinds +- SE | paired difference to the base +- SE | wins | % decisions
       changed | override rate. games/hour is 3600 x workers / CPU seconds per game, so it doesn't depend
       on what else the machine is doing; the override rate is the share of the decisions the network saw
       whose final choice is not one of the prior's top choices, and in brackets the share where the prior
       had the final choice at least 1 logit below its top (by phase with --detail).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

from ..rewards.config import RewardConfig
from .train import EVAL_SEED0, STAKE, _checkpoint_meta, _make_agent, _split, evaluate


def _variant(v: dict):
    """(model, AgentConfig overrides, search, RewardConfig or None) of one grid entry."""
    rcfg = RewardConfig.from_dict(v["rewards"]) if v.get("rewards") else None
    return v.get("model", "none"), dict(v.get("cfg") or {}), bool(v.get("search", True)), rcfg


def run_grid(grid: dict, out: str, games: int, workers: int, seed0: int = EVAL_SEED0, only=None):
    os.makedirs(out, exist_ok=True)
    for name, v in grid.items():
        path = os.path.join(out, f"{name}.json")
        if (only and name not in only) or os.path.exists(path):
            continue
        model, cfg, search, rcfg = _variant(v)
        t = time.time()
        res = evaluate(model, games, workers, search, stake=STAKE, seed0=seed0, cfg_over=cfg, rcfg=rcfg)
        res.update(name=name, variant=v, workers=workers, wall_min=round((time.time() - t) / 60, 2))
        with open(path, "w") as f:
            json.dump(res, f, default=float)
        print(f"{name}: {res['blinds']:.2f} blinds, {res['win%']:.1f}% wins, "
              f"override {res['override'].get('all', {}).get('final%', 0.0):.1f}%, {res['wall_min']} min", flush=True)


# ------------------------------------------------------------------ decisions changed
def _agent_for(v: dict, seed: int):
    model, cfg, search, rcfg = _variant(v)
    meta = _checkpoint_meta(model)
    rcfg = rcfg or RewardConfig.from_dict(meta.get("rewards"))
    cfg = {"lam": rcfg.schedule(meta.get("step", 0), meta.get("lambda_clock")).lam, **cfg}
    return _make_agent(model, search, seed, cfg, dict(rcfg.potential.__dict__))


def _changed_worker(args):
    base, other, seeds = args
    from ..sim.game import Game
    from .world import World
    a, b = _agent_for(base, seeds[0]), _agent_for(other, seeds[0])
    n = diff = n_net = diff_net = 0
    for s in seeds:
        w = World(Game(seed=s, deck_type="RED", stake=STAKE))
        while not w.done:
            h = s * 1_000_003 + w.steps                 # both decide this position from the same seed
            a.reseed(h)
            da = a.decide(w, explore=False)
            if da.action is None:
                break
            b.reseed(h)
            db = b.decide(w, explore=False)
            n += 1
            diff += int(da.action != db.action)
            if da.enc is not None:                      # not auto-played
                n_net += 1
                diff_net += int(da.action != db.action)
            w.step(da.action)
    return n, diff, n_net, diff_net


def changed(grid: dict, out: str, base: str, games: int, workers: int, seed0: int = EVAL_SEED0, only=None):
    os.makedirs(out, exist_ok=True)
    for name, v in grid.items():
        path = os.path.join(out, f"{name}.changed.json")
        if name == base or (only and name not in only) or os.path.exists(path):
            continue
        seeds = list(range(seed0, seed0 + games))
        with mp.get_context("spawn").Pool(min(workers, games)) as pool:
            parts = pool.map(_changed_worker, [(grid[base], v, c) for c in _split(seeds, workers)])
        n, diff, n_net, diff_net = (sum(p[i] for p in parts) for i in range(4))
        res = {"base": base, "games": games, "decisions": n, "changed%": 100.0 * diff / max(1, n),
               "network_decisions": n_net, "changed_of_network%": 100.0 * diff_net / max(1, n_net)}
        with open(path, "w") as f:
            json.dump(res, f)
        print(f"{name} vs {base}: {res['changed%']:.1f}% of {n} decisions differ", flush=True)


# ------------------------------------------------------------------ table
def _se(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def table(out: str, base: str | None, detail: bool = False, order=None) -> list[dict]:
    res = {}
    for f in sorted(os.listdir(out)):
        if f.endswith(".json") and not f.endswith(".changed.json"):
            with open(os.path.join(out, f)) as fh:
                res[f[:-5]] = json.load(fh)
    names = [n for n in (order or [])] + [n for n in res if n not in (order or [])]
    names = [n for n in names if n in res]
    b = {s: bl for s, bl, _, _ in res[base]["per_game"]} if base in res else None
    rows = []
    for n in names:
        r = res[n]
        blinds = [g[1] for g in r["per_game"]]
        row = {"variant": n, "games": len(blinds), "games/hour": 3600.0 * r["workers"] / r["cpu_sec/game"],
               "blinds": float(np.mean(blinds)), "se": _se(blinds), "wins": int(sum(g[2] for g in r["per_game"])),
               "override": r["override"]}
        if b is not None and n != base:
            d = [bl - b[s] for s, bl, _, _ in r["per_game"] if s in b]
            row["diff"], row["diff_se"] = float(np.mean(d)), _se(d)
        cp = os.path.join(out, f"{n}.changed.json")
        if os.path.exists(cp):
            with open(cp) as fh:
                row["changed%"] = json.load(fh)["changed%"]
        rows.append(row)
    print(f"{'variant':<28} {'games/h':>8} {'blinds +- SE':>14} {'vs ' + (base or '-'):>16} {'wins':>9} "
          f"{'changed%':>9} {'override%':>12}")
    for r in rows:
        d = f"{r['diff']:+.2f} +- {r['diff_se']:.2f}" if "diff" in r else ""
        ch = f"{r['changed%']:.1f}" if "changed%" in r else ""
        al = r["override"].get("all", {})
        ov = "" if "final%" not in al else f"{al['final%']:.1f}" + (f" ({al['strong%']:.1f})" if "strong%" in al else "")
        print(f"{r['variant']:<28} {r['games/hour']:>8.0f} {r['blinds']:>7.2f} +- {r['se']:.2f} {d:>16} "
              f"{r['wins']:>3}/{r['games']:<5} {ch:>9} {ov:>12}")
        if detail:
            print("    " + "  ".join(f"{ph}: {v['final%']:.1f}%" + (f" ({v['strong%']:.1f})" if "strong%" in v else "")
                                     + f" of {v['n']}" for ph, v in r["override"].items() if ph != "all"))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["grid", "changed", "table"])
    p.add_argument("--grid", help="JSON file: {variant name: {cfg, search, model, rewards}}")
    p.add_argument("--out", default="checkpoints/eval")
    p.add_argument("--base", default=None, help="variant the others are compared with")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--only", default="", help="comma-separated variant names")
    p.add_argument("--detail", action="store_true")
    a = p.parse_args()
    grid = {}
    if a.grid:
        with open(a.grid) as f:
            grid = json.load(f)
    only = [x for x in a.only.split(",") if x] or None
    if a.cmd == "grid":
        run_grid(grid, a.out, a.games, a.workers, only=only)
    elif a.cmd == "changed":
        changed(grid, a.out, a.base, a.games, a.workers, only=only)
    else:
        table(a.out, a.base, a.detail, order=list(grid))


if __name__ == "__main__":
    main()
