"""Scripted rounds (Env::scripted_action / auto_round), the scenario builds of the judge's data, set_state, and the judge potential's contract."""
import numpy as np
import pytest

from balatro_rl.engine_env import CppVecEnv, _engine
from balatro_rl.env import A_SELECT, A_LEAVE

NEED = ("set_auto_round", "round_report", "set_build_seeds", "set_power", "set_state", "set_modes", "set_marginal_blind_only")
pytestmark = pytest.mark.skipif(_engine is None or not all(hasattr(_engine.VecEnv, m) for m in NEED), reason="engine not rebuilt with scripted rounds")


def scenario(n=64, ante=4, seeds=None, power=(1.0, 8.0), blind=-1):
    v = CppVecEnv(n, "RED", "GOLD", seed0=11, threads=1)
    v.set_scenario(1.0, 1.0, 0.0, 0.0, ante, ante, 3, 0.5)
    v.set_power(*power)
    v.set_build_seeds(seeds if seeds is not None else [1000 + i for i in range(n)], blind, 0.7)
    v.v.restart()
    return v


def test_a_scripted_round_ends_the_scenario_and_reports_it():
    v = scenario(128, ante=2)
    v.set_auto_round(True, 1.5, 0)
    _, d, info, _, trunc = v.step_split(np.full(128, A_SELECT))
    rep = v.round_report()
    assert d.all()                                                         # a one-blind scenario ends with its round: cleared (cut off) or lost
    assert np.array_equal(rep["cleared"], trunc | info["won"])
    assert (rep["hands"] >= 1).all() and (rep["hands"] <= 6).all()
    assert rep["cleared"].mean() > 0.5                                     # ante 2 builds of this power mostly pass


def test_the_same_build_meets_new_draws_and_builds_are_reproducible():
    seeds = [777] * 16
    v = scenario(16, seeds=seeds)
    jid0 = np.array(v.obs["jid"]).copy()
    g0 = np.array(v.obs["g"]).copy()
    assert (jid0 == jid0[0]).all() and np.allclose(g0[:, :8], g0[0, :8])      # one build in every env
    v2 = scenario(16, seeds=seeds)
    assert np.array_equal(np.array(v2.obs["jid"]), jid0)                   # and the same again from the seed
    v.set_auto_round(True, 1.5, 0)
    outcomes = []
    for _ in range(6):
        v.step_split(np.full(16, A_SELECT))
        outcomes.append(v.round_report()["ratio"].copy())
    assert np.std(np.concatenate(outcomes)) > 0                            # different draws gave different rounds


def test_build_power_makes_stronger_builds():
    from balatro_rl.env import global_size
    from balatro_rl.marginal import MARG_G, POT_G
    base = global_size() - MARG_G - POT_G

    def strength(power):
        v = scenario(512, ante=6, power=power, seeds=[5000 + i for i in range(512)])
        v.set_marginal(1)
        g = np.asarray(v.obs["g"])
        return float(np.median(7.0 * g[:, base] - 6.0 * g[:, 8]))
    assert strength((12.0, 12.0)) > strength((1.0, 1.0)) + 0.8              # decades of expected best play against the target


def test_scaler_friendly_mode_keeps_ride_the_bus_alive():
    from balatro_rl.env import item_id
    bus = item_id("j_ride_the_bus")
    res = {}
    for mode in (0, 1):
        v = scenario(2048, ante=5, power=(1.0, 10.0), seeds=[3000000 + i for i in range(2048)], blind=1)
        has = (np.array(v.obs["jid"]) == bus)
        if has.sum() < 20:
            pytest.skip("too few Ride the Bus builds in the sample")
        v.set_auto_round(True, 1.5, mode)
        dc = np.zeros((2048, 8))
        for _ in range(3):
            v.step_split(np.full(2048, A_SELECT))
            r = v.round_report()
            dc += r["c1"] - r["c0"]
        res[mode] = dc[has].mean()
    assert res[1] > res[0] + 0.5                                           # the scaler-friendly player loses less of its counter


def test_set_state_moves_a_copy_to_another_ante_with_other_counters():
    v = scenario(8, ante=3, seeds=[42] * 8)
    la = CppVecEnv(8, "RED", "GOLD", seed0=1, threads=1)
    la.set_marginal(1)
    la.copy_many(v, list(range(8)), list(range(8)))
    ante0 = (np.asarray(la.obs["g"])[:, 0] * 8).round()
    la.set_state(list(range(8)), 8, 2, np.full((8, 8), 10.0))
    g = np.asarray(la.obs["g"])
    assert ((g[:, 0] * 8).round() == 8).all() and (ante0 == 3).all()
    assert (np.asarray(v.obs["g"])[:, 0] * 8).round().max() == 3             # the original is untouched
    assert g[:, 4].min() > 0.5                                             # a blind select


@pytest.mark.skipif(_engine is None or not hasattr(_engine.VecEnv, "margin_report"), reason="engine not rebuilt with scripted mode 2")
def test_margin_mode_keeps_clearing_and_protects_the_scalers():
    from balatro_rl.env import item_id
    bus = item_id("j_ride_the_bus")
    res = {}
    for mode in (0, 2):
        v = scenario(2048, ante=3, power=(1.0, 6.0), seeds=[3000000 + i for i in range(2048)], blind=1)
        has = (np.array(v.obs["jid"]) == bus)
        v.set_auto_round(True, 1.5, mode)
        v.step_split(np.full(2048, A_SELECT))
        r = v.round_report()
        res[mode] = (r["cleared"].mean(), (r["c1"] - r["c0"])[has].mean() if has.any() else 0.0)
        if mode == 2:
            mr = v.margin_report()
            assert np.isfinite(mr).all() and (mr[:, 1] >= 0).all() and (mr[:, 1] <= 1).all()      # lambda is a share
    assert abs(res[2][0] - res[0][0]) < 0.02                                    # investing never costs the blind
    assert res[2][1] >= res[0][1] - 1e-9                                        # and keeps the counter at least as high
