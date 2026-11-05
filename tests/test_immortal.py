"""--immortal (Game::immortal, Env reward by cleared blinds): a failed blind does not end the run; the reward is each blind actually beaten, once."""
import numpy as np
import pytest

from balatro_rl.engine_env import CppVecEnv, _engine
from balatro_rl.train import blind_weights

pytestmark = pytest.mark.skipif(_engine is None or not hasattr(_engine.VecEnv, "set_immortal"), reason="engine not rebuilt with immortal mode")


def run(immortal, steps=3000, n=48):
    v = CppVecEnv(n, "RED", "GOLD", seed0=11, threads=1)
    v.set_weight_geom(1.6)
    if immortal:
        v.set_immortal(True)
        v.v.restart()
    rng = np.random.default_rng(0)
    rew = np.zeros(n)
    games = []
    for _ in range(steps):
        m = np.asarray(v.obs["mask"])
        r, d, info = v.step(np.argmax(rng.random(m.shape) * m, axis=1))
        rew += r
        for e in np.flatnonzero(d):
            games.append((int(info["ante"][e]), int(info["blinds"][e]), int(v.last_bits()[e]), bool(info["won"][e]), float(rew[e])))
            rew[e] = 0
    return games


def test_normal_games_end_at_the_first_failure():
    g = run(False, 600)
    assert max(a for a, *_ in g) <= 3                      # (a random policy dies in the early antes)


def test_immortal_games_run_to_the_ante_8_boss():
    g = run(True)
    assert len(g) > 10
    assert all(a == 8 for a, *_ in g)                      # only failing (or winning) the ante-8 boss ends a game


def test_reward_is_the_blinds_actually_beaten():
    w = blind_weights(1.0, 1.0, geom=1.6)
    for ante, blinds, bits, won, total in run(True):
        assert blinds == bin(bits).count("1")
        expected = sum(w[b] for b in range(24) if (bits >> b) & 1) + (10.0 if won else 0.0)          # (the default win bonus is 10)
        assert abs(total - expected) < 1e-3


def test_blind_multipliers_scale_the_reward():
    w = blind_weights(1.0, 1.0, geom=1.6)
    n = 48
    v = CppVecEnv(n, "RED", "GOLD", seed0=11, threads=1)
    v.set_weight_geom(1.6)
    v.set_immortal(True)
    mult = np.ones(24)
    mult[:3] = 0.0                                              # ante 1 pays nothing
    mult[3:6] = 0.5
    v.set_blind_mult(mult)
    v.v.restart()
    rng = np.random.default_rng(0)
    rew = np.zeros(n)
    checked = 0
    for _ in range(3000):
        m = np.asarray(v.obs["mask"])
        r, d, info = v.step(np.argmax(rng.random(m.shape) * m, axis=1))
        rew += r
        for e in np.flatnonzero(d):
            bits = int(v.last_bits()[e])
            expected = sum(w[b] * mult[b] for b in range(24) if (bits >> b) & 1) + (10.0 if info["won"][e] else 0.0)
            assert abs(rew[e] - expected) < 1e-3
            rew[e] = 0
            checked += 1
    assert checked > 5


def test_pay_on_fail_keeps_the_economy_without_paying_the_reward():
    def money_by_ante(pay):
        v = CppVecEnv(64, "RED", "GOLD", seed0=11, threads=1)
        v.set_weight_geom(1.6)
        v.set_immortal(True, pay)
        v.v.restart()
        rng = np.random.default_rng(0)
        seen, rewards = [], 0.0
        for _ in range(2500):
            m = np.asarray(v.obs["mask"])
            g = np.asarray(v.obs["g"])
            seen += [g[e, 17] for e in range(64) if round(g[e, 0] * 8) >= 4 and g[e, 4 + 2] > 0.5]          # money (g 17) in a shop (g 6) from ante 4 on
            r, d, info = v.step(np.argmax(rng.random(m.shape) * m, axis=1))
            rewards += r.sum()
        return float(np.mean(seen)) if seen else 0.0, rewards
    poor, r0 = money_by_ante(False)
    paid, r1 = money_by_ante(True)
    assert paid > poor                                          # late shops hold more money when a failed blind is still paid
    assert r0 > 0 and r1 > 0
