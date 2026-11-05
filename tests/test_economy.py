"""The future-economy projection (balatro_rl/economy.py): the payout rules on hand-made states."""
import numpy as np
import pytest

from balatro_rl.engine_env import CppVecEnv, _engine
from balatro_rl.economy import future_economy
from balatro_rl.runeval import Cols, from_vecenv

pytestmark = pytest.mark.skipif(_engine is None, reason="engine not built")


def fresh(n=4, money=None):
    v = CppVecEnv(n, "RED", "GOLD", seed0=21, threads=1)
    v.v.restart()
    inp = from_vecenv(v)
    if money is not None:
        inp["g"][:, Cols.MONEY] = min(money, 100) / 50
        inp["g"][:, Cols.LOG_MONEY] = np.log10(money + 1) / 3
    return inp


def test_a_fresh_run_earns_blind_rewards_hands_and_interest():
    out = future_economy(**fresh(), rounds=3, e_best=np.full(4, 1e6))       # an overwhelming build: one play a blind, three hands left
    p = out["parts"]
    assert np.allclose(p["blind"][0], [0, 4, 5])                            # Gold: no small blind reward, big $4, boss $5
    assert np.allclose(p["hands"][0], [3, 3, 3])
    assert np.allclose(p["interest"][0, 0], 0)                              # $4 earns no interest
    assert np.allclose(out["money"][0], [4, 7, 15, 26])                     # 4 -> +3 (hands) -> +4 +3 +$1 interest -> +5 +3 +$3 interest
    assert out["alive"].all()


def test_interest_is_capped_and_a_weak_build_stops():
    out = future_economy(**fresh(money=90), rounds=2, e_best=np.full(4, 1e6))
    assert np.allclose(out["parts"]["interest"][:, 0], 5)                   # $90 earns the $5 cap
    weak = future_economy(**fresh(), rounds=3, e_best=np.full(4, 1.0))
    assert not weak["alive"].any() and np.isnan(weak["money"][:, 1:]).all()
