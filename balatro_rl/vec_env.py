"""Parallel environments in worker processes (auto-reset on episode end)."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from .env import BalatroEnv
from .heuristic import HeuristicPolicy


def _worker(remote, idx, n, deck, stake, seed0, want_expert, reward_kw, strategic_kw):
    if strategic_kw:                       # a tactical player handles the cards inside step()
        import torch
        from .strategic import StrategicEnv
        torch.set_num_threads(1)
        env = StrategicEnv(deck, stake, **strategic_kw, **reward_kw)
    else:
        env = BalatroEnv(deck, stake, **reward_kw)
    expert = HeuristicPolicy(rng=np.random.default_rng(seed0 + idx), lookahead=False)
    episode = 0

    def new_obs():
        nonlocal episode
        o = env.reset(seed0 + idx + n * episode)
        episode += 1
        return o

    def label(o):
        return expert.act(env.g, o) if want_expert else -1

    obs = new_obs()
    while True:
        cmd, data = remote.recv()
        if cmd == "get":
            remote.send((obs, label(obs)))
        elif cmd == "step":
            obs2, r, done, info = env.step(int(data))
            if done:
                info = dict(info, episode_end=True)
                obs2 = new_obs()
            obs = obs2
            remote.send((obs, r, done, info, label(obs)))
        elif cmd == "close":
            remote.close()
            break


class VecEnv:
    def __init__(self, n: int, deck="RED", stake="GOLD", seed0: int = 0, expert: bool = False,
                 reward_kw: dict | None = None, strategic_kw: dict | None = None):
        # "fork" is fastest but only exists on Linux/macOS; Windows needs "spawn"
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(method)
        self.n = n
        self.remotes, work = zip(*[ctx.Pipe() for _ in range(n)])
        self.procs = [ctx.Process(target=_worker, args=(w, i, n, deck, stake, seed0, expert, reward_kw or {}, strategic_kw),
                                  daemon=True)
                      for i, w in enumerate(work)]
        for p in self.procs:
            p.start()

    def current(self):
        for r in self.remotes:
            r.send(("get", None))
        res = [r.recv() for r in self.remotes]
        return [x[0] for x in res], np.array([x[1] for x in res])

    def step(self, actions):
        idx = range(self.n)
        self.step_async(actions, idx)
        return self.step_wait(idx)

    # split-phase stepping of a subset of envs, so the caller can work while they run
    def step_async(self, actions, idx):
        for i, a in zip(idx, actions):
            self.remotes[i].send(("step", int(a)))

    def step_wait(self, idx):
        res = [self.remotes[i].recv() for i in idx]
        obs, rew, done, info, lab = zip(*res)
        return list(obs), np.array(rew, dtype=np.float32), np.array(done), list(info), np.array(lab)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=1)
