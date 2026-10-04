"""Play policy checkpoints on the same seeds, on the C++ engine, and compare them.

    python tools/eval_policy.py checkpoints/a.pt checkpoints/b.pt --games 300 --seed0 10000 --stake WHITE

Each checkpoint plays seeds seed0 .. seed0 + games - 1 twice: greedy (the most likely action) and sampled
(as in training). Reported: blinds beaten, wins, ante reached, and the paired difference to the first
checkpoint on the same seeds.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from balatro_rl.engine_env import CppVecEnv, OBS_KEYS      # noqa: E402
from balatro_rl.model import load_model                      # noqa: E402


def play(path: str, games: int, seed0: int, deck: str, stake: str, greedy: bool, device: str, seed: int = 0):
    """The first episode of each of `games` environments: seeds seed0 + i. -> (blinds [games], won [games])."""
    model = load_model(path, device)
    venv = CppVecEnv(games, deck, stake, seed0=seed0, threads=0)
    blinds = np.full(games, -1, dtype=np.int64)
    won = np.zeros(games, dtype=bool)
    gen = torch.Generator(device=device).manual_seed(seed)
    while (blinds < 0).any():
        ob = {k: torch.from_numpy(venv.obs[k]).to(device) for k in OBS_KEYS}
        with torch.no_grad():
            logits, _ = model(ob)
        if greedy:
            act = logits.argmax(1)
        else:
            act = torch.multinomial(torch.softmax(logits.float(), 1), 1, generator=gen).squeeze(1)
        _, done, info = venv.step(act.cpu().numpy())
        for e in np.flatnonzero(done):
            if blinds[e] < 0:
                blinds[e] = info["blinds"][e]
                won[e] = info["won"][e]
    return blinds, won


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="WHITE")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    print(f"{a.games} games, seeds {a.seed0}..{a.seed0 + a.games - 1}, {a.deck} deck, {a.stake} stake")
    base = {}
    for path in a.checkpoints:
        for mode in ("greedy", "sampled"):
            b, w = play(path, a.games, a.seed0, a.deck, a.stake, mode == "greedy", a.device)
            line = (f"{os.path.basename(path):28s} {mode:8s} {b.mean():6.2f} blinds  {int(w.sum()):4d} wins "
                    f"({100 * w.mean():4.1f}%)  ante {1 + (np.minimum(b, 23) // 3).mean():.2f}")
            if mode in base:
                d = b - base[mode]
                line += f"   paired {d.mean():+.2f} +- {d.std() / np.sqrt(len(d)):.2f}"
            else:
                base[mode] = b
            print(line, flush=True)


if __name__ == "__main__":
    main()
