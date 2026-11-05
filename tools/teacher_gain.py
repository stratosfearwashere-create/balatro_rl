"""How much better is a SEED-FREE teacher than the policy it is built on?

The shop-only game (the scripted mode-2 player plays the rounds). The base policy is a trained shop network. The teacher plays the same games, but at every shop decision it tries each legal action
(and "what the base would do") on copies of the game whose future is REDRAWN (VecEnv.reseed_rng: new draws; nothing of the real seed's future is used), plays each out with the base policy
to the end of the run on the same draws for every candidate (paired), and takes the action with the best average result when it beats the base's own choice by --thr blinds.
Both play the same seeds. If the teacher is not clearly better, looping teacher and learner has nothing to feed on.

    python tools/teacher_gain.py checkpoints/ppo_gold_v30_watch.pt --games 64 --seeds 16
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import shopvalue as SV                      # noqa: E402
from balatro_rl.env import A_LEAVE                          # noqa: E402
from balatro_rl.model import load_model                     # noqa: E402
from balatro_rl.shopq import make_env                       # noqa: E402


class Teacher:
    def __init__(self, base, pr, n_seeds, thr, max_states=12):
        self.base, self.pr, self.n_seeds, self.thr, self.max_states = base, pr, n_seeds, thr, max_states
        self.salt, self.overrides, self.decisions = 5_000_000, 0, 0

    def __call__(self, env, ids):
        a = self.base(env, ids)
        mask = np.asarray(env.obs["mask"])
        shop = [int(e) for e in ids if mask[e, A_LEAVE]]
        for c0 in range(0, len(shop), self.max_states):
            chunk = shop[c0:c0 + self.max_states]
            states = [(e, [-1] + SV.shop_candidates(mask[e])) for e in chunk]
            while sum(len(c) for _, c in states) * self.n_seeds > self.pr.cap:
                states.pop()
            res = self.pr.values_many(env, states, self.n_seeds, salt0=self.salt)
            self.salt += 1000
            pos = {int(e): i for i, e in enumerate(ids)}
            for (e, cands), r in zip(states, res):
                self.decisions += 1
                gain = r["v"] - r["v"][0]
                j = int(np.argmax(gain))
                if j > 0 and gain[j] > self.thr:
                    a[pos[e]] = cands[j]
                    self.overrides += 1
        return a


def play(policy, n, seed0, max_steps=900):
    v = make_env(n, "RED", "GOLD", seed0, exact=False)
    done = np.zeros(n, dtype=bool)
    blinds, won = np.zeros(n), np.zeros(n, dtype=bool)
    for _ in range(max_steps):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        act = np.full(n, -1, dtype=np.int64)
        act[live] = policy(v, live)
        _, d, info = v.step(act)
        first = d & ~done
        blinds[first], won[first] = info["blinds"][first], info["won"][first]
        done |= d
    return blinds, won


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--seeds", type=int, default=16, help="rollouts per candidate")
    p.add_argument("--thr", type=float, default=0.3, help="blinds a candidate must beat the base's choice by")
    p.add_argument("--seed0", type=int, default=9_500_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    model = load_model(a.checkpoint, a.device)
    base = SV.ManagerPolicy(model, a.device)
    t0 = time.time()
    b0, w0 = play(base, a.games, a.seed0)
    print(f"base policy:  {b0.mean():.2f} blinds, {int(w0.sum())} wins on {a.games} Gold games  ({time.time() - t0:.0f}s)", flush=True)
    pr = SV.PairedRollouts(base, cap=4096)
    teacher = Teacher(base, pr, a.seeds, a.thr)
    t0 = time.time()
    b1, w1 = play(teacher, a.games, a.seed0)
    d = b1 - b0
    print(f"seed-free teacher: {b1.mean():.2f} blinds, {int(w1.sum())} wins  ({time.time() - t0:.0f}s); it overrode the base at {teacher.overrides} of {teacher.decisions} shop decisions", flush=True)
    print(f"gain {d.mean():+.2f} +- {d.std() / np.sqrt(len(d)):.2f} blinds per game (the games are not step-for-step paired: the draws diverge once the choices differ)", flush=True)


if __name__ == "__main__":
    main()
