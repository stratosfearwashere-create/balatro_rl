"""The evaluation every harness decision is based on. Protected: agents can read it, not change it.

Runs a checkpoint with an experiment's code (--code, a worktree) but always with the main branch's
simulator, on a fixed list of held-out seeds, greedily, and writes one record per game.

    python eval/run_eval.py --code <worktree> --checkpoint <ckpt> --seeds eval/seeds_cheap.json --games 200
        --stake GOLD --win-ante 4 --joker-pool eval/joker_pool_cheap.json --tactical <ckpt> --out eval.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from itertools import combinations
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]


def load_code(code: Path):
    """Import the experiment's balatro_rl package, with balatro_rl.sim taken from the main branch."""
    sys.path.insert(0, str(code))
    import balatro_rl
    sim_dir = MAIN / "balatro_rl" / "sim"
    spec = importlib.util.spec_from_file_location("balatro_rl.sim", sim_dir / "__init__.py",
                                                  submodule_search_locations=[str(sim_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["balatro_rl.sim"] = mod
    spec.loader.exec_module(mod)
    balatro_rl.sim = mod
    return balatro_rl


def best_ratio(g, Plan) -> float:
    """Best single play available right now, as a share of what the blind still needs."""
    need = max(g.target - g.chips, 1)
    k = min(len(g.hand), 8)
    subs = [c for r in range(1, 6) for c in combinations(range(k), r)]
    return max(p[0] for p in g.predict_many(subs, Plan(g))) / need


def stickers(j) -> dict:
    return {"eternal": bool(j.eternal), "perishable": j.perishable is not None, "rental": bool(j.rental)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", default=str(MAIN))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--win-ante", type=int, default=8)
    p.add_argument("--joker-pool", default=None)
    p.add_argument("--tactical", default=None, help="frozen card player: evaluate in the strategic environment")
    p.add_argument("--max-minutes", type=float, default=30)
    p.add_argument("--traces", type=int, default=60, help="text traces kept for this many lost games")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    load_code(Path(a.code))
    import numpy as np
    import torch
    from balatro_rl.env import BalatroEnv, A_LEAVE, describe_action
    from balatro_rl.model import load_model, batch_obs
    from balatro_rl.sim.scoring import Plan

    torch.set_num_threads(1)
    model = load_model(a.checkpoint, "cpu")
    pool = json.load(open(a.joker_pool)) if a.joker_pool else None
    seeds = json.load(open(a.seeds))[:a.games]
    if a.tactical:
        from balatro_rl.strategic import StrategicEnv
        env = StrategicEnv(a.deck, a.stake, tactical=a.tactical, win_ante=a.win_ante, joker_pool=pool)
    else:
        env = BalatroEnv(a.deck, a.stake, win_ante=a.win_ante, joker_pool=pool)

    t0 = time.time()
    games, kept = [], 0
    for seed in seeds:
        if (time.time() - t0) / 60 > a.max_minutes:
            break
        obs = env.reset(seed)
        g = env.g
        rec = {"seed": seed, "blinds": [], "money": [], "trace": [], "ratio_at_ante": {}}
        held, rental_paid, blind_key, done = {}, 0, None, False
        while not done:
            for j in g.jokers:
                held.setdefault(j.uid, stickers(j))
            if g.state == "SELECTING_HAND":
                key = (g.ante, g.blind_idx, g.blinds_beaten)
                if key != blind_key:                         # a blind starts
                    blind_key = key
                    r = best_ratio(g, Plan)
                    rec["blinds"].append({"ante": g.ante, "blind": g.blind_idx, "boss": g.boss if g.blind_idx == 2 else "",
                                          "target": g.target, "hands": g.hands_left, "ratio": round(r, 3)})
                    rec["ratio_at_ante"].setdefault(str(g.ante), round(r, 3))
            with torch.no_grad():
                logits, _ = model(batch_obs([obs], "cpu"))
            act = int(logits.argmax(-1))
            if act == A_LEAVE:
                rec["money"].append([g.ante, g.blind_idx, g.money, len(g.jokers), g.joker_slots])
            if g.state in ("SHOP", "BLIND_SELECT", "PACK"):
                rec["trace"].append(f"a{g.ante} {g.state.lower():12s} ${g.money:<4} {describe_action(g, act, obs)}")
            beaten = g.blinds_beaten
            obs, _, done, info = env.step(act)
            if rec["blinds"] and g.blinds_beaten > beaten:
                b = rec["blinds"][-1]
                b.update(cleared=True, chips=g.chips, frac=round(g.chips / b["target"], 3))
                rental_paid += 3 * sum(1 for j in g.jokers if j.rental)
                rec["trace"].append(f"a{b['ante']} blind {'SBB'[b['blind']]}{' ' + b['boss'] if b['boss'] else ''}: "
                                    f"target {b['target']}, best hand {b['ratio']:.0%} of it -> cleared {g.chips}")
        rec.update(furthest=g.furthest_blind, ante=g.ante, won=bool(info["won"]), rental_paid=rental_paid,
                   jokers=[{"key": j.key, "edition": j.edition, **stickers(j)} for j in g.jokers],
                   slots=g.joker_slots, stickers_held={k: sum(v[k] for v in held.values())
                                                       for k in ("eternal", "perishable", "rental")})
        if not rec["won"] and rec["blinds"]:
            b = rec["blinds"][-1]
            b.update(cleared=False, chips=g.chips, frac=round(g.chips / b["target"], 3) if b["target"] else 0.0)
            rec["death"] = {**b, "discards_left": g.discards_left, "money": g.money}
            rec["trace"].append(f"a{b['ante']} blind {'SBB'[b['blind']]}{' ' + b['boss'] if b['boss'] else ''}: "
                                f"target {b['target']}, best hand {b['ratio']:.0%} of it -> LOST with {g.chips}")
        if rec["won"] or kept >= a.traces:
            rec["trace"] = []
        else:
            kept += 1
        games.append(rec)
    out = {"meta": {k: v for k, v in vars(a).items()}, "minutes": round((time.time() - t0) / 60, 2),
           "complete": len(games) == len(seeds), "games": games}
    Path(a.out).write_text(json.dumps(out))
    wins = sum(g["won"] for g in games)
    print(f"{len(games)} games, mean blinds {np.mean([g['furthest'] for g in games]):.2f}, {wins} won -> {a.out}")


if __name__ == "__main__":
    main()
