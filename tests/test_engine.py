"""The C++ engine (balatro_rl.sim._engine) against the Python game and environment.

Each test plays seeded games on both with the same actions (tests/engine_diff.py) and requires, after every
step, the same game state (every field, every card of every pile, jokers and their state, shop, pack), the
same state of the random generator, and with obs=True the same observation arrays and rewards. A longer run:

    python -m tests.engine_diff --games 2000 --spice --obs --deep 12 --p-spice 0.3 --coverage
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

_engine = pytest.importorskip("balatro_rl.sim._engine", reason="the C++ engine is not built")

from balatro_rl import env as E                                  # noqa: E402
from balatro_rl.sim.items import DECKS, STAKES                    # noqa: E402
from tests import engine_diff as D                               # noqa: E402


def _play(seeds, **kw):
    steps = 0
    for s in seeds:
        deck, stake = kw.pop("deck", None) or DECKS[s % len(DECKS)], kw.pop("stake", None) or STAKES[(s // 3) % len(STAKES)]
        bad, info = D.play_one(s, deck, stake, kw.get("spice", False), kw.get("obs", False),
                               deep=kw.get("deep", 0), p_spice=kw.get("p_spice", 0.15), win_ante=kw.get("win_ante", 8))
        assert not bad, f"seed {s} {deck} {stake}:\n  " + "\n  ".join(bad)
        steps += info["steps"]
    return steps


def test_tables_are_generated_from_the_python_definitions():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_engine_tables.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_same_states_on_plain_games():
    assert _play(range(0, 48)) > 1000


def test_same_states_with_injected_content_in_long_games():
    """Random jokers, consumables, vouchers, bosses, tags and card changes; hands start at level 12 so the
    games reach the late antes and the finisher bosses."""
    assert _play(range(100, 124), spice=True, deep=12, p_spice=0.3) > 2000


def test_same_observations_and_rewards():
    assert _play(range(200, 224), spice=True, obs=True, deep=12, p_spice=0.3) > 2000
    assert _play(range(230, 250), obs=True) > 300


def test_shortened_runs():
    _play(range(300, 306), obs=True, deep=12, win_ante=2)


def _policy(obs, rng):
    """A deterministic choice from a batch of observations: the best play in a round, else a random legal action."""
    m = obs["mask"]
    in_round = m[:, :E.N_SUB].any(1) & (obs["g"][:, 5] > 0)
    best = np.argmax(np.where(m[:, :E.N_SUB], obs["af"][:, :E.N_SUB, E.RATIO_COL], -1), 1)
    rand = np.argmax(rng.random(m.shape) * m, 1)
    return np.where(in_round, best, rand)


def test_vec_env_is_the_single_env_batched():
    """Any thread count gives the same batch, and each row is the single environment on the seeds
    seed0 + i + n * episode."""
    n, seed0 = 6, 11
    v1 = _engine.VecEnv(n, "RED", "WHITE", seed0, 1)
    v3 = _engine.VecEnv(n, "RED", "WHITE", seed0, 3)
    o1, o3 = v1.obs(), v3.obs()
    singles = [_engine.Env("RED", "WHITE") for _ in range(n)]
    sobs = [e.reset(seed0 + i) for i, e in enumerate(singles)]
    episode = [1] * n
    rng = np.random.default_rng(0)
    ended = 0
    for step in range(400):
        for k in D.OBS_KEYS:
            assert np.array_equal(o1[k], o3[k]), f"step {step}: threads change {k}"
            for i in range(n):
                assert np.array_equal(o1[k][i], sobs[i][k]), f"step {step} env {i}: {k}"
        a = _policy(o1, rng)
        r1, d1, ante1, won1, bl1 = v1.step(a)
        r3, d3, ante3, won3, bl3 = v3.step(a)
        assert np.array_equal(r1, r3) and np.array_equal(d1, d3) and np.array_equal(bl1, bl3)
        for i in range(n):
            ob, r, done, info = singles[i].step(int(a[i]))
            assert np.float32(r) == r1[i] and done == d1[i] and info["blinds"] == bl1[i]
            if done:
                ended += 1
                ob = singles[i].reset(seed0 + i + n * episode[i])
                episode[i] += 1
            sobs[i] = ob
    assert ended >= 10


def test_wrapper_matches_the_python_environment():
    from balatro_rl.engine_env import CppEnv
    py, cp = E.BalatroEnv("RED", "WHITE"), CppEnv("RED", "WHITE")
    po, co = py.reset(3), cp.reset(3)
    rng = np.random.default_rng(1)
    for _ in range(60):
        assert not D.obs_diff(po, co)
        a = int(rng.choice(np.flatnonzero(po["mask"])))
        po, pr, pd, _ = py.step(a)
        co, cr, cd, _ = cp.step(a)
        assert pr == cr and pd == cd
        if pd:
            break
