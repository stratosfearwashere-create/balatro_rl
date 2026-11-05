"""--autoplay: at a blind select the action 'leave shop' means 'select it and let the exact round solver play it'."""
import numpy as np
import pytest

from balatro_rl import env as E
from balatro_rl.engine_env import CppEnv, CppVecEnv, _engine

pytestmark = pytest.mark.skipif(_engine is None or not hasattr(_engine.VecEnv, "set_autoplay"), reason="engine not rebuilt with autoplay")


def test_off_by_default():
    e = CppEnv("RED", "GOLD")
    e.reset(3)
    assert not e.obs["mask"][E.A_LEAVE]


def test_the_solver_plays_a_blind():
    cleared = 0
    for seed in range(1, 41):
        e = CppEnv("RED", "GOLD")
        e.e.autoplay = True
        e.reset(seed)
        assert e.obs["mask"][E.A_LEAVE]
        e.step(E.A_LEAVE)
        cleared += int(e.game.furthest_blind >= 1)
    assert cleared >= 36                                       # the first blind of the run, almost always cleared


def test_an_autoplayed_game_runs_and_pays_like_a_played_one():
    v = CppVecEnv(16, "RED", "GOLD", seed0=5, threads=1)
    v.set_autoplay(True)
    rng = np.random.default_rng(0)
    total, done_games = 0.0, 0
    for _ in range(600):
        m = np.asarray(v.obs["mask"])
        g = np.asarray(v.obs["g"])
        a = np.argmax(rng.random(m.shape) * m, axis=1)
        a = np.where((g[:, 4] > 0.5) & m[:, E.A_LEAVE], E.A_LEAVE, a)
        r, d, info = v.step(a)
        total += float(r.sum())
        done_games += int(d.sum())
    assert done_games > 0 and total > 0
