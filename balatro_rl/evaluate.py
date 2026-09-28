"""Evaluate a policy (heuristic, random or a trained checkpoint) on the simulator.

    python -m balatro_rl.evaluate --policy heuristic --games 200
    python -m balatro_rl.evaluate --policy checkpoints/ppo.pt --games 200
"""
from __future__ import annotations

import argparse
import collections
import time

import numpy as np

from .env import BalatroEnv, describe_action
from .heuristic import HeuristicPolicy


def make_policy(spec: str, device: str = "cpu", greedy: bool = True):
    if spec == "heuristic":
        h = HeuristicPolicy()
        return lambda env, obs: h.act(env.g, obs)
    if spec == "random":
        rng = np.random.default_rng(0)
        return lambda env, obs: int(rng.choice(np.flatnonzero(obs["mask"])))
    import torch
    from .model import load_model, batch_obs
    model = load_model(spec, device)

    def act(env, obs):
        with torch.no_grad():
            logits, _ = model(batch_obs([obs], device))
            if greedy:
                return int(logits.argmax(-1).item())
            return int(torch.distributions.Categorical(logits=logits).sample().item())
    return act


def run(policy, games: int, deck: str, stake: str, seed0: int = 10_000, verbose: bool = False):
    env = BalatroEnv(deck, stake)
    antes, wins, blinds = [], 0, []
    t = time.time()
    for k in range(games):
        obs = env.reset(seed0 + k)
        done = False
        while not done:
            a = policy(env, obs)
            if verbose:
                print(f"  ante {env.g.ante} {env.g.state:15s} ${env.g.money:<4} {describe_action(env.g, a, obs)}")
            obs, r, done, info = env.step(a)
        antes.append(info["ante"] if not info["won"] else 9)
        blinds.append(info["blinds"])
        wins += info["won"]
    dist = collections.Counter(antes)
    return {
        "games": games, "win_rate": wins / games, "mean_blinds": float(np.mean(blinds)),
        "mean_ante": float(np.mean([min(a, 8) for a in antes])),
        "ante_reached": {("won" if a == 9 else f"ante {a}"): dist[a] for a in sorted(dist)},
        "seconds": round(time.time() - t, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="heuristic", help="heuristic | random | path to .pt checkpoint")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--sample", action="store_true", help="sample actions instead of greedy argmax")
    p.add_argument("--verbose", action="store_true", help="print every action (use with --games 1)")
    a = p.parse_args()
    pol = make_policy(a.policy, greedy=not a.sample)
    print(run(pol, a.games, a.deck, a.stake, verbose=a.verbose))


if __name__ == "__main__":
    main()
