"""Plays a network greedily on unmodified games until it has N wins, and writes each win (seed, stake, actions) to a JSON-lines file.

    python tools/collect_wins.py --ckpt checkpoints/ppo_gold_v7_best_greedy.pt --stake GOLD --wins 100 --out traces/v7_wins100.jsonl

CPU only by default (so it can run beside a trainer). Games are the engine's, seeds seed0 + i, played 512 at a time; a game that is not won is dropped.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS      # noqa: E402
from balatro_rl.model import load_model                    # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--wins", type=int, default=100)
    p.add_argument("--max-games", type=int, default=200_000)
    p.add_argument("--seed0", type=int, default=30_000_000)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--threads", type=int, default=4, help="torch threads (the engine uses its own)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sample", action="store_true", help="sample actions from the policy instead of playing greedily")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    model = load_model(a.ckpt, a.device)
    N = a.envs
    v = CppVecEnv(N, a.deck, a.stake, seed0=a.seed0, threads=0)
    if getattr(model, "marg", 0):
        v.set_marginal(int(model.marg))
    hist = [[] for _ in range(N)]
    ep = np.zeros(N, dtype=np.int64)
    games = wins = 0
    t0 = time.time()
    gen = torch.Generator().manual_seed(a.seed0)
    with open(a.out, "a") as f:
        while wins < a.wins and games < a.max_games:
            ob = {k: torch.from_numpy(np.array(v.obs[k])).to(a.device) for k in OBS_KEYS if k in v.obs}
            with torch.no_grad():
                logits, _ = model(ob)
            logits = logits.float()
            act = (torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).squeeze(1) if a.sample else logits.argmax(1)).cpu().numpy()
            for e in range(N):
                hist[e].append(int(act[e]))
            _, d, info = v.step(act)
            for e in np.flatnonzero(d):
                games += 1
                if info["won"][e]:
                    wins += 1
                    f.write(json.dumps({"seed": int(v.seed0 + e + N * ep[e]), "stake": a.stake, "actions": hist[e]}) + "\n")
                    f.flush()
                hist[e] = []
                ep[e] += 1
            if games and games % 2000 < int(d.sum()):
                print(json.dumps({"games": games, "wins": wins, "win%": round(100 * wins / games, 3), "min": round((time.time() - t0) / 60, 1)}), flush=True)
    print(json.dumps({"done": True, "games": games, "wins": wins, "win%": round(100 * wins / max(1, games), 3), "min": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
