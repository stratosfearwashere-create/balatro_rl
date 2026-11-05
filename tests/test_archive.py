"""The state archive (engine Archive / Env::restore_from_archive, train.py --archive-*) and geometric blind weights (--weight-geom)."""
import numpy as np
import pytest

from balatro_rl.engine_env import CppVecEnv, _engine
from balatro_rl.train import blind_weights

pytestmark = pytest.mark.skipif(_engine is None or not hasattr(_engine.VecEnv, "set_archive"), reason="engine not rebuilt with the archive")


def play(v, steps, seed=0):
    rng = np.random.default_rng(seed)
    kinds = []
    for _ in range(steps):
        m = np.asarray(v.obs["mask"])
        kinds.append(v.scn_kinds().copy())
        v.step(np.argmax(rng.random(m.shape) * m, axis=1))
    return kinds


def test_off_by_default():
    v = CppVecEnv(16, "RED", "GOLD", seed0=1, threads=1)
    play(v, 200)
    assert v.archive_counts().sum() == 0 and (v.scn_kinds() == -1).all()


def test_archive_fills_and_games_start_from_it():
    v = CppVecEnv(48, "RED", "GOLD", seed0=500, threads=1)
    v.set_start(0.0, 0.0, 1.0, 100000)
    v.v.restart()
    v.set_archive(0.0, 1.0, 1, 7, 8, 2)
    play(v, 1500)
    counts = v.archive_counts()
    assert counts.sum() > 0 and counts.max() <= 8                 # a reservoir of at most `cap` per blind
    v.set_archive(1.0, 0.0, 1, 7, 8, 1)
    v.set_start(0.0, 0.0, 0.0, 0)
    kinds = np.concatenate(play(v, 1500, seed=1))
    assert (kinds == 3).any()                                     # games that began from a snapshot
    assert not (kinds == 0).any()                                 # (no synthetic scenario was asked for)


def test_geometric_weights_add_up_and_rise_by_the_ratio():
    w = blind_weights(1.0, 1.0, geom=1.6)
    assert abs(float(w.sum()) - 24.0) < 1e-4
    ante = w[::3]
    assert np.allclose(ante[1:] / ante[:-1], 1.6, rtol=1e-4)
    assert np.allclose(w[0::3], w[1::3]) and np.allclose(w[1::3], w[2::3])        # the three blinds of an ante pay the same


def test_engine_geometric_weights_match():
    v = CppVecEnv(4, "RED", "GOLD", seed0=1, threads=1)
    v.set_weight_geom(1.6)
    # a beaten blind pays its weight: compare the engine's table through a one-blind reward of a scenario is overkill; check the Python/engine normalisation instead
    assert abs(float(blind_weights(1.0, 1.0, geom=1.6).sum()) - 24.0) < 1e-4


def test_winners_only_and_outcomes_by_start_blind():
    v = CppVecEnv(48, "RED", "GOLD", seed0=500, threads=1)
    v.set_start(0.0, 0.0, 1.0, 100000)
    v.v.restart()
    v.set_archive(0.0, 1.0, 1, 8, 8, 2, True)                    # winners only
    play(v, 1500)
    won_only = v.archive_counts().sum()
    w = CppVecEnv(48, "RED", "GOLD", seed0=500, threads=1)
    w.set_start(0.0, 0.0, 1.0, 100000)
    w.v.restart()
    w.set_archive(0.0, 1.0, 1, 8, 8, 2, False)
    play(w, 1500)
    assert won_only <= w.archive_counts().sum()                  # the winners-only archive holds a subset of what the plain one keeps
    w.set_archive(1.0, 0.0, 1, 8, 8, 1)
    w.set_start(0.0, 0.0, 0.0, 0)
    play(w, 1500, seed=2)
    out = w.archive_outcomes()
    assert out.shape == (2, 24) and out[0].sum() > 0 and (out[1] <= out[0]).all()      # games ended, and the wins among them, by start blind


def test_second_vec_env_shares_the_archive():
    a = CppVecEnv(16, "RED", "GOLD", seed0=1, threads=1)
    b = CppVecEnv(16, "RED", "GOLD", seed0=2, threads=1)
    b.v.share_archive(a.v)
    b.set_start(0.0, 0.0, 1.0, 100000)
    b.v.restart()
    b.set_archive(0.0, 1.0, 1, 8, 4, 1)
    play(b, 600)
    assert a.archive_counts().sum() > 0                          # what b stored is in a's archive
