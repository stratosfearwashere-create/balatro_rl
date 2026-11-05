"""The run eval (balatro_rl/runeval.py): columns, ranges, speed, and the direction of its terms."""
import time

import numpy as np
import pytest

from balatro_rl.engine_env import CppVecEnv, _engine
from balatro_rl.runeval import Cols, RunEval, TERMS, State, default_weights, from_vecenv, hand_reach, joker_rules, HAND_NAMES
from balatro_rl.sim.hands import PAIR, FLUSH, TRIPS, HC
from balatro_rl.sim.jokers import JOKERS

pytestmark = pytest.mark.skipif(_engine is None, reason="engine not built")
KEYS = list(JOKERS)


def fresh(n=64, seed0=11):
    v = CppVecEnv(n, "RED", "GOLD", seed0=seed0, threads=1)
    v.v.restart()
    return v


def test_columns_match_the_observation():
    v = fresh(4)
    g = np.asarray(v.obs["g"])
    assert g.shape[1] >= Cols.END
    assert np.allclose(g[:, Cols.ANTE], 1 / 8)                           # ante 1
    assert (g[:, Cols.STATE + 0] > 0.5).all()                            # at the blind select
    s = State(**from_vecenv(v))
    assert (s.money == 4).all() and (s.hands == 4).all() and (s.discards == 3).all() and (s.hand_size == 8).all()
    assert (s.ranks.sum(1) == 52).all() and (s.suits == 13).all()


def test_scores_are_probabilities_and_terms_finite():
    v = fresh(64)
    out = RunEval(default_weights())(**from_vecenv(v))
    for k in ("score", "win", "early", "mid", "late"):
        assert np.isfinite(out[k]).all() and (out[k] >= 0).all() and (out[k] <= 1).all()
    assert np.isfinite(out["terms"]).all() and out["terms"].shape == (64, len(TERMS))


def test_hand_reach_orders_hands_sensibly():
    v = fresh(8)
    s = State(**from_vecenv(v))
    r0 = hand_reach(s, joker_rules(s), np.zeros(8))                     # the deal alone
    r3 = hand_reach(s, joker_rules(s), np.full(8, 3))                   # three redraws
    assert (r0[:, HC] == 1).all()
    assert (r0[:, PAIR] > r0[:, TRIPS]).all() and (r0[:, PAIR] > r0[:, FLUSH]).all()
    assert (r3 >= r0 - 1e-9).all()                                       # redraws never hurt
    assert 0.5 < r0[0, PAIR] < 0.95 and r0[0, FLUSH] < 0.2 and 0.5 < r3[0, FLUSH] < 0.97


def test_money_raises_economy_and_a_multiplier_joker_raises_scaling():
    v = fresh(16)
    inp = from_vecenv(v)
    ev = RunEval(default_weights())
    base = ev(**inp)["terms"]
    rich = {k: x.copy() for k, x in inp.items()}
    rich["g"][:, Cols.MONEY] = 40 / 50                                  # $40
    rich["g"][:, Cols.LOG_MONEY] = np.log10(41) / 3
    t_rich = ev(**rich)["terms"]
    assert (t_rich[:, TERMS.index("economy")] > base[:, TERMS.index("economy")]).all()
    # the same games with a joker set: a plain +mult joker vs an xMult joker (The Duo) vs a scaler (Ride the Bus)
    scaling = TERMS.index("scaling")
    power = TERMS.index("power")
    w = CppVecEnv(3, "RED", "GOLD", seed0=11, threads=1)
    w.v.restart()
    for i, k in enumerate(("joker", "duo", "ride_the_bus")):
        w.set_build([i], 2, 0, [KEYS.index(k)], None, 0, None, 10)
    t = ev(**from_vecenv(w))["terms"]
    assert t[1, scaling] > t[0, scaling] and t[2, scaling] > t[0, scaling]
    assert t[0, power] > ev(**from_vecenv(fresh(1)))["terms"][0, power] - 1.0   # a joker never makes the build weaker by much


def test_speed_under_a_millisecond_per_state_in_a_batch():
    v = fresh(512)
    ev = RunEval(default_weights())
    inp = from_vecenv(v)
    ev(**inp)
    t0 = time.perf_counter()
    ev(**inp)
    dt = time.perf_counter() - t0
    assert dt / 512 < 1e-3, f"{1e6 * dt / 512:.0f} us per state"
