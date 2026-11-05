"""Power measure 1: the chance of passing the blind the game is at, from the jokers, deck, hand levels and money it has now.

The state is copied K times; each copy is given its own draws (VecEnv.reseed_rng: the cards come off the deck in another order, boss and joker randomness redrawn) and the
round is played to its end with the engine's scripted player, which scores every play with the exact scorer (the engine's own scoring: jokers, editions, enhancements, levels,
retriggers). The share of the copies that clear the blind is the pass probability; the mean log10 share of the target reached says by how much. The player's best of its two modes
(0: score most, 1: keep the scalers growing) is used, as the manager can choose either.

    from balatro_rl.power import PassProbability
    pp = PassProbability(k=64)
    r = pp(venv, [0, 5, 9])        # envs at a blind select -> {"p": [3], "p_mode": [3, 2], "log_share": [3]}
"""
from __future__ import annotations

import numpy as np

from .env import A_SELECT


class PassProbability:
    def __init__(self, k=64, pace=1.5, modes=(0, 1), deck="RED", stake="GOLD", cap=4096, threads=0, salt0=1_000_003):
        from .engine_env import CppVecEnv
        self.k, self.pace, self.modes, self.salt = k, pace, tuple(modes), salt0
        self.scratch = CppVecEnv(cap, deck, stake, seed0=1, threads=threads)
        self.cap = cap
        self.calls = 0

    def __call__(self, venv, ids):
        """ids: environments of venv that are at a blind select. Not for exact-seed games (their draws are fixed by the seed: reseed_rng only works in the default mode)."""
        ids = [int(i) for i in ids]
        per = max(1, self.cap // self.k)
        out_p = np.zeros((len(ids), len(self.modes)))
        out_r = np.zeros((len(ids), len(self.modes)))
        for c0 in range(0, len(ids), per):
            ch = ids[c0:c0 + per]
            src = [e for e in ch for _ in range(self.k)]
            dst = list(range(len(src)))
            for mi, mode in enumerate(self.modes):
                self.scratch.copy_many(venv, src, dst)
                self.calls += 1
                self.scratch.reseed_rng(dst, [self.salt * (self.calls + 1) + 7919 * i + mode for i in range(len(src))])
                self.scratch.set_auto_round(True, self.pace, mode)
                self.scratch.step_split(np.concatenate([np.full(len(src), A_SELECT), np.full(self.scratch.n - len(src), -1)]))
                rep = self.scratch.round_report()
                cleared = rep["cleared"][:len(src)].reshape(len(ch), self.k)
                ratio = np.log10(np.maximum(rep["ratio"][:len(src)], 1e-3)).reshape(len(ch), self.k)
                out_p[c0:c0 + len(ch), mi] = cleared.mean(1)
                out_r[c0:c0 + len(ch), mi] = ratio.mean(1)
        return {"p": out_p.max(1), "p_mode": out_p, "log_share": out_r.max(1)}
