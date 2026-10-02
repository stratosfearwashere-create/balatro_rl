"""Head-to-head of the two pipelines on the same held-out seeds at equal training time.

    # train the PPO pipeline for a fixed wall-clock budget: behaviour cloning, then PPO in chunks
    python -m balatro_rl.headtohead ppo-train --minutes 90 --envs 7 --out checkpoints/runs/ppo_h2h
    # evaluate a PPO-pipeline policy (a checkpoint, or "heuristic") on the az evaluation seeds
    python -m balatro_rl.headtohead ppo-eval --policy checkpoints/runs/ppo_h2h/ppo_best.pt --name ppo \\
        --out checkpoints/eval/h2h --games 300

ppo-eval writes <out>/<name>.json in the format of az.compare, so `python -m balatro_rl.az.compare table
--out <out> --base <name>` puts both pipelines in one table (blinds +- SE and the paired difference; the
seeds are az.train.EVAL_SEED0 .., the same games for both). Equal time means equal wall-clock on the same
machine with the same number of worker processes; PPO's time includes its behaviour-cloning warm start.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

EVAL_SEED0 = 10_000
STAKE = "WHITE"


# ------------------------------------------------------------------ PPO training for a time budget
def _run(args: list[str], log: str) -> float:
    t = time.time()
    with open(log, "a") as f:
        subprocess.run([sys.executable, "-m", "balatro_rl.train"] + args, check=True, stdout=f, stderr=subprocess.STDOUT)
    return (time.time() - t) / 60


def ppo_train(minutes: float, envs: int, out: str, bc_iters: int, chunk: int, steps: int, stake: str, deck: str):
    """Behaviour cloning for bc_iters, then PPO in chunks of `chunk` iterations (each continuing from the
    last) until the next chunk would run past the budget. Progress is in <out>/h2h.json."""
    os.makedirs(out, exist_ok=True)
    state_path = os.path.join(out, "h2h.json")
    state = {"budget_min": minutes, "bc_min": None, "chunks": []}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    common = ["--envs", str(envs), "--stake", stake, "--deck", deck]
    bc, ppo = os.path.join(out, "bc.pt"), os.path.join(out, "ppo.pt")
    log = os.path.join(out, "train_stdout.txt")

    def save():
        with open(state_path, "w") as f:
            json.dump(state, f, indent=1)
    if state["bc_min"] is None:
        state["bc_min"] = _run(["bc", "--iters", str(bc_iters), "--out", bc] + common, log)
        save()
    while True:
        used = state["bc_min"] + sum(state["chunks"])
        per = max(state["chunks"][-1] if state["chunks"] else 0.0, 0.01)
        if state["chunks"] and used + per > minutes:
            break
        if not state["chunks"] and used >= minutes:
            break
        init = ppo if state["chunks"] else bc
        # ppo_best.pt is kept across chunks only by the training code's own rule (best training blinds of
        # that chunk); evaluate ppo.pt (the last) and ppo_best.pt both
        state["chunks"].append(_run(["ppo", "--init", init, "--iters", str(chunk), "--steps", str(steps),
                                     "--out", ppo, "--save-every", "10"] + common, log))
        save()
    state["total_min"] = state["bc_min"] + sum(state["chunks"])
    save()
    print(json.dumps(state))


# ------------------------------------------------------------------ evaluation on the az seeds
def _eval_worker(args):
    policy, seeds, deck, stake = args
    import torch
    torch.set_num_threads(1)
    from .env import BalatroEnv
    from .evaluate import make_policy
    pol = make_policy(policy)
    env = BalatroEnv(deck, stake)
    out = []
    for s in seeds:
        t = time.process_time()
        obs = env.reset(s)
        done = False
        info = {}
        while not done:
            obs, _, done, info = env.step(pol(env, obs))
        out.append([s, int(env.g.furthest_blind), int(bool(info["won"])), min(int(info["ante"]), 8),
                    time.process_time() - t])
    return out


def ppo_eval(policy: str, name: str, out: str, games: int, workers: int, deck: str, stake: str):
    os.makedirs(out, exist_ok=True)
    seeds = list(range(EVAL_SEED0, EVAL_SEED0 + games))
    chunks = [c for c in (seeds[i::workers] for i in range(workers)) if c]
    t = time.time()
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        rows = sorted(sum(pool.map(_eval_worker, [(policy, c, deck, stake) for c in chunks]), []))
    res = {"name": name, "variant": {"policy": policy}, "games": games, "workers": workers,
           "blinds": float(np.mean([r[1] for r in rows])), "win%": 100.0 * float(np.mean([r[2] for r in rows])),
           "ante": float(np.mean([r[3] for r in rows])), "cpu_sec/game": float(np.mean([r[4] for r in rows])),
           "per_game": [r[:4] for r in rows], "override": {}, "wall_min": round((time.time() - t) / 60, 2)}
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(res, f)
    print(f"{name}: {res['blinds']:.2f} blinds, {res['win%']:.1f}% wins, {res['wall_min']} min")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("ppo-train")
    t.add_argument("--minutes", type=float, required=True, help="wall-clock budget, behaviour cloning included")
    t.add_argument("--envs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    t.add_argument("--out", default="checkpoints/runs/ppo_h2h")
    t.add_argument("--bc-iters", type=int, default=150)
    t.add_argument("--chunk", type=int, default=100, help="PPO iterations per chunk")
    t.add_argument("--steps", type=int, default=128)
    t.add_argument("--stake", default=STAKE)
    t.add_argument("--deck", default="RED")
    e = sub.add_parser("ppo-eval")
    e.add_argument("--policy", required=True, help="heuristic | random | checkpoint of the PPO pipeline")
    e.add_argument("--name", required=True)
    e.add_argument("--out", default="checkpoints/eval/h2h")
    e.add_argument("--games", type=int, default=300)
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    e.add_argument("--stake", default=STAKE)
    e.add_argument("--deck", default="RED")
    a = p.parse_args()
    if a.cmd == "ppo-train":
        ppo_train(a.minutes, a.envs, a.out, a.bc_iters, a.chunk, a.steps, a.stake, a.deck)
    else:
        ppo_eval(a.policy, a.name, a.out, a.games, a.workers, a.deck, a.stake)


if __name__ == "__main__":
    main()
