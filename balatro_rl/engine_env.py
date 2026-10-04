"""Environments on the C++ engine (sim/_engine): the same observations, actions and rewards as env.BalatroEnv
(tests/test_engine.py compares them value for value), many times faster.

    env = CppEnv("RED", "WHITE"); obs = env.reset(seed); obs, r, done, info = env.step(a)     # like BalatroEnv
    venv = CppVecEnv(256, "RED", "WHITE", seed0=7, threads=8)                                  # like VecEnv
    obs = venv.obs            # dict of [n, ...] arrays: live views, overwritten by the next step
    rew, done, infos = venv.step(actions)

Not supported (use the Python environment): shape_phi (the build potential is Python code), the strategic
environment and a restricted joker pool. Build with `python setup_cython.py build`.
"""
from __future__ import annotations

import numpy as np

try:
    from .sim import _engine
    AVAILABLE = True
except ImportError:                                    # not built
    _engine = None
    AVAILABLE = False

OBS_KEYS = ("g", "cards", "jid", "jf", "cid", "aid", "af", "mask", "subs")


def _check(reward_kw: dict):
    if not AVAILABLE:
        raise RuntimeError("the C++ engine is not built: run `python setup_cython.py build`")
    kw = dict(reward_kw or {})
    if kw.pop("shape_phi", 0.0):
        raise ValueError("shape_phi needs the Python environment (the build potential is Python code)")
    if kw.pop("joker_pool", None):
        raise ValueError("joker_pool needs the Python environment")
    kw.pop("reward_config", None)
    return {"ante_weight": kw.pop("ante_weight", 1.0), "win_bonus": kw.pop("win_bonus", 10.0),
            "win_ante": kw.pop("win_ante", 8), "shape_chips": kw.pop("shape_chips", 0.0),
            "shape_gamma": kw.pop("shape_gamma", 0.995)}


class CppEnv:
    """One environment (BalatroEnv's interface). `game` is the C++ game (dump(), state, money ...)."""

    def __init__(self, deck: str = "RED", stake: str = "GOLD", **reward_kw):
        self.e = _engine.Env(deck, stake, **_check(reward_kw))
        self.obs = None

    @property
    def game(self):
        return self.e.game

    def reset(self, seed: int = 0) -> dict:
        self.obs = self.e.reset(int(seed))
        return self.obs

    def step(self, a: int):
        self.obs, r, done, info = self.e.step(int(a))
        return self.obs, r, done, info


class CppVecEnv:
    """n environments stepped together on `threads` threads (0: one per core), auto-reset like vec_env.VecEnv:
    environment i plays seeds seed0 + i + n * episode. `obs` holds live views of the engine's buffers: copy
    (or move to the GPU) what must outlive the next step."""

    def __init__(self, n: int, deck: str = "RED", stake: str = "GOLD", seed0: int = 0, threads: int = 0,
                 reward_kw: dict | None = None):
        kw = _check(reward_kw or {})
        self.n = n
        self.v = _engine.VecEnv(n, deck, stake, int(seed0), int(threads), kw["ante_weight"], kw["win_bonus"],
                                kw["win_ante"], kw["shape_chips"], kw["shape_gamma"])
        self.obs = self.v.obs()

    def step(self, actions):
        """-> (rewards [n] float32, dones [n] bool, infos). infos: ante / won / blinds arrays describing the
        step just taken (for a finished environment: the outcome of the episode that ended)."""
        rew, done, ante, won, blinds = self.v.step(np.ascontiguousarray(actions, dtype=np.int64))
        return rew, done, {"ante": ante, "won": won, "blinds": blinds}

    def close(self):
        self.v = None
        self.obs = None
