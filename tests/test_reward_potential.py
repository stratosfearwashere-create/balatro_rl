"""With part credit off the blind rewards are a pure potential: the return of any game is W(furthest blind beaten) + the win bonus, whatever the route (the sum of the per-blind weights
of the blinds beaten, each once)."""
import numpy as np
import pytest

from balatro_rl.engine_env import CppVecEnv, _engine
from balatro_rl.train import blind_weights

pytestmark = pytest.mark.skipif(_engine is None or not hasattr(_engine.VecEnv, "set_part_credit"), reason="engine not rebuilt with set_part_credit")


def play(part_credit, n=128, steps=1500, seed0=500):
    v = CppVecEnv(n, "RED", "GOLD", seed0=seed0, threads=1, reward_kw={"ante_weight": 2.5, "win_bonus": 10.0, "win_ante": 8, "shape_gamma": 0.999})
    v.set_weight_shape(2.5, 1.0)
    if not part_credit:
        v.set_part_credit(False)
    rng = np.random.default_rng(1)
    ret = np.zeros(n)
    out = []
    for _ in range(steps):
        mask = np.asarray(v.obs["mask"])
        act = np.array([rng.choice(np.flatnonzero(r)) for r in mask])
        r, d, info = v.step(act)
        ret += r
        for e in np.flatnonzero(d):
            out.append((float(ret[e]), int(info["blinds"][e]), bool(info["won"][e])))
            ret[e] = 0.0
    v.close()
    return out


def test_the_return_is_the_weight_of_the_blinds_beaten():
    w = np.asarray(blind_weights(2.5, 1.0), dtype=np.float64)
    games = play(False)
    assert len(games) > 100
    for ret, blinds, won in games:
        assert abs(ret - (w[:blinds].sum() + (10.0 if won else 0.0))) < 1e-3          # exactly the potential of where the game ended


def test_part_credit_adds_to_it_when_on():
    w = np.asarray(blind_weights(2.5, 1.0), dtype=np.float64)
    games = play(True)
    extra = [ret - (w[:b].sum() + (10.0 if won else 0.0)) for ret, b, won in games]
    assert max(extra) > 1e-3 and min(extra) > -1e-6                                     # a loss earns something for the chips scored
