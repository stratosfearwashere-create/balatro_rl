"""One-step look-ahead over the shop with the judge potential.

For a shop decision, every action that changes the build (buy, replace, sell a joker) is played on a copy of the game (VecEnv.copy_many + step) and the copy is
scored with the potential (potential.py: the ante-8 boss with every scaler at its natural value there, plus the money term). The difference to the potential of the
state as it is says what the action is worth in the judge's terms; 'leave' is worth 0. Two uses:

  input    the difference goes into the network as a feature of each such action (ActorCritic phi=1: obs "pa" [B, A, 2] = (difference / PA_SCALE, flag));
  target   a policy loss pulls the manager toward softmax(difference / tau) over the flagged actions (hier.py --search-coef): search distilled into the policy.
"""
from __future__ import annotations

import numpy as np

from .env import A_BUY, A_LEAVE, A_REPLACE, A_SELL_J, MAX_JOKERS, MAX_SHOP

PA_SCALE = 5.0            # nats per unit of the feature


def candidate_actions(n_actions: int) -> np.ndarray:
    """[A] bool: the actions that are played on copies (a buy, a replace, a joker sale)."""
    c = np.zeros(n_actions, dtype=bool)
    c[A_BUY:A_BUY + MAX_SHOP] = True
    c[A_SELL_J:A_SELL_J + MAX_JOKERS] = True
    c[A_REPLACE:] = True
    return c


class ShopSearch:
    def __init__(self, pot, pool, proj, n_actions):
        self.pot, self.pool, self.proj = pot, pool, proj
        self.cand = candidate_actions(n_actions)
        self.A = n_actions
        self.sims = 0

    def run(self, venv, mask: np.ndarray, env_ids: np.ndarray, phi_now: np.ndarray) -> np.ndarray:
        """mask [N, A] bool; env_ids: the environments at a decision outside a round; phi_now: their potential. -> pa [N, A, 2] float32 (zeros where nothing was simulated)."""
        N = mask.shape[0]
        pa = np.zeros((N, self.A, 2), dtype=np.float32)
        if len(env_ids) == 0:
            return pa
        sub = mask[env_ids]
        leave = sub[:, A_LEAVE]
        pa[env_ids[leave], A_LEAVE, 1] = 1.0                                   # (worth 0 by definition)
        rows, cols = np.nonzero(sub & self.cand[None, :])
        if len(rows) == 0:
            pa[env_ids[leave], A_LEAVE, 1] = 0.0                               # nothing to compare it with
            return pa
        src = env_ids[rows]
        base = np.full(N, np.nan)
        base[env_ids] = phi_now
        M = self.pool.n
        for c0 in range(0, len(src), M):
            s, a = src[c0:c0 + M], cols[c0:c0 + M]
            m = len(s)
            slots = list(range(m))
            self.pool.copy_many(venv, s.tolist(), slots)
            act = np.full(M, -1, dtype=np.int64)
            act[:m] = a
            self.pool.step_split(act)
            ph = self.pot.phi(slots, src=self.pool, proj=self.proj)
            d = (ph - base[s]) / PA_SCALE
            pa[s, a, 0] = np.clip(np.nan_to_num(d, nan=0.0), -4.0, 4.0)
            pa[s, a, 1] = 1.0
            self.sims += m
        has = pa[env_ids][:, :, 1].sum(1) > 1.5
        pa[env_ids[~has], A_LEAVE, 1] = 0.0                                    # a lone option is no comparison
        return pa
