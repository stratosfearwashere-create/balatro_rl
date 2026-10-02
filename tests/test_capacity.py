"""Smoke test for capacity: build features, random builds, measuring a build, the model, and the
capacity-guided shop rule. Runs with untrained networks, so it needs no checkpoint files; the one
test that uses a trained capacity model is skipped when it is absent."""
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from balatro_rl import capacity as C  # noqa: E402
from balatro_rl.env import (A_BUY, A_BUY_PACK, A_LEAVE, A_REROLL, A_SELL_C, A_USE_C, A_VOUCHER,  # noqa: E402
                            BalatroEnv, F_JOKER, N_ACTIONS, encode)
from balatro_rl.model import ActorCritic  # noqa: E402
from balatro_rl.sim.game import Game, MAX_JOKERS, MAX_VOUCHERS  # noqa: E402
from balatro_rl.tactical import NetTactics  # noqa: E402


def test_build_features():
    g = Game(seed=0)
    gv, jid, jf = C.build_features(g)
    assert gv.shape == (C.g_size(),) and gv.dtype == np.float32
    assert jid.shape == (MAX_JOKERS,) and jf.shape == (MAX_JOKERS, F_JOKER + C.F_POS)
    assert not jid.any() and not jf.any()                  # no jokers yet
    j = g.random_joker()
    g.add_joker(j)
    gv2, jid2, jf2 = C.build_features(g)
    assert jid2[0] > 0 and jf2[0, 0] == 1.0 and jf2[0, F_JOKER] == 1.0 and not jf2[1:].any()
    assert jf2[0, F_JOKER - 3] == pytest.approx(j.sell_value() / 10)      # from the price paid (Joker.cost)
    g.state = "SELECTING_HAND"                             # the features don't depend on the screen
    assert np.array_equal(C.build_features(g)[0], gv2) and g.state == "SELECTING_HAND"
    g.hand_levels[C.FLUSH] += 2
    assert not np.array_equal(C.build_features(g)[0], gv2)


def test_build_actions_fit_the_action_space():
    assert len(set(C.BUILD_ACTIONS)) == len(C.BUILD_ACTIONS)
    assert min(C.BUILD_ACTIONS) == A_BUY and max(C.BUILD_ACTIONS) == N_ACTIONS - 1
    not_build = set(range(A_BUY_PACK, A_LEAVE + 1)) | set(range(A_SELL_C, A_USE_C))
    assert {A_VOUCHER + i for i in range(MAX_VOUCHERS)} | {A_REROLL, A_LEAVE} <= not_build
    assert not not_build & set(C.BUILD_ACTIONS)            # packs, both vouchers, reroll, leave, selling consumables


def test_random_build_and_measure():
    torch.manual_seed(0)
    tactics = NetTactics(ActorCritic().eval())             # an untrained network plays the cards
    for seed in range(3):
        rng = random.Random(seed)
        g = C.random_build(rng, seed)
        assert 1 <= g.ante <= 8 and g.state == "BLIND_SELECT" and len(g.jokers) <= g.joker_slots
        assert all(j.cost is not None and j.sell_value() >= 1 for j in g.jokers)
        before = C.build_features(g)
        y, cleared = C.measure(g, tactics, 2, rng)
        ceiling = np.log10(1 + C.TARGET_X * C.BLIND_BASE[g.scaling()][g.ante - 1])
        assert 0.0 <= y <= ceiling + 1e-9 and 0.0 <= cleared <= 1.0
        assert g.state == "BLIND_SELECT"                   # measured on copies
        assert all(np.array_equal(a, b) for a, b in zip(before, C.build_features(g)))


def test_auc():
    assert C._auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert C._auc([4, 3, 2, 1], [False, False, True, True]) == 0.0
    assert np.isnan(C._auc([1, 2], [True, True]))


def test_model_trains_saves_and_loads(tmp_path):
    rng = np.random.default_rng(0)
    n, gd = 1200, C.g_size()
    jf = np.zeros((n, MAX_JOKERS, F_JOKER + C.F_POS), np.float32)
    jf[:, 0, 0] = 1.0
    np.savez(tmp_path / "shard0000.npz", g=rng.random((n, gd), dtype=np.float32),
             jid=rng.integers(1, 50, (n, MAX_JOKERS)), jf=jf, y=rng.random(n, dtype=np.float32),
             cleared=np.zeros(n, np.float32), ante=np.ones(n, np.int64))
    out = str(tmp_path / "capacity.pt")
    C.train(str(tmp_path), out, epochs=1, device="cpu")
    cap = C.Capacity(out)
    games = [Game(seed=1), C.random_build(random.Random(2), 2)]
    y = cap(games)
    assert y.shape == (2,) and np.isfinite(y).all()
    assert np.allclose(y, cap(games))                      # deterministic


class _Shop:
    """Just what capacity_choice needs of a strategic environment."""

    def __init__(self, env):
        self.env = env

    @property
    def g(self):
        return self.env.g


def _first_shop(seed):
    env = BalatroEnv("RED", "WHITE")
    env.reset(seed)
    g = env.g
    g.select_blind()
    g.chips = g.target
    g.win_round()
    assert g.state == "SHOP"
    g.money = 50
    env.obs = encode(g, env.cnt)
    return env


def test_capacity_choice_overrules_only_above_the_threshold():
    def n_jokers(games):                                   # stand-in model: every joker is worth 1
        return np.array([len(x.jokers) for x in games], float)
    for seed in range(40):
        env = _first_shop(seed)
        buys = [i for i, it in enumerate(env.g.shop) if it.kind == "joker" and env.obs["mask"][A_BUY + i]]
        if buys:
            break
    else:
        pytest.fail("no shop with an affordable joker in 40 seeds")
    senv = _Shop(env)
    shop_before = [it.key for it in env.g.shop]
    a, over = C.capacity_choice(senv, env.obs, A_LEAVE, n_jokers, 0.5)
    assert over and a - A_BUY in buys
    assert C.capacity_choice(senv, env.obs, A_LEAVE, n_jokers, 1.5) == (A_LEAVE, False)
    assert C.capacity_choice(senv, env.obs, A_BUY + buys[0], n_jokers, 0.5) == (A_BUY + buys[0], False)
    assert env.g.state == "SHOP" and not env.g.jokers and [it.key for it in env.g.shop] == shop_before
    env.g.state = "BLIND_SELECT"                           # outside shops and packs it never interferes
    assert C.capacity_choice(senv, env.obs, A_LEAVE, n_jokers, 0.0) == (A_LEAVE, False)


def test_with_trained_model():
    model = "checkpoints/capacity.pt"
    if not os.path.exists(model):
        pytest.skip("needs a trained " + model)
    y = C.Capacity(model)([Game(seed=0)])
    assert y.shape == (1,) and np.isfinite(y).all()
