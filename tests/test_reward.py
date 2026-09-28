"""Blinds replayed after a -1 Ante voucher (Hieroglyph/Petroglyph) don't count as progress."""
from balatro_rl.sim.game import Game


def test_furthest_blind_ignores_replays():
    g = Game(seed=1, stake="WHITE")
    g.ante, g.blind_idx = 2, 1
    g.win_round()                       # ante 2 big blind = blind 5
    assert (g.blinds_beaten, g.furthest_blind) == (1, 5)
    g.ante, g.blind_idx = 1, 2          # set back an ante, then beat the ante 1 boss
    g.win_round()
    assert (g.blinds_beaten, g.furthest_blind) == (2, 5)
    g.ante, g.blind_idx = 8, 2
    g.win_round()                       # ante 8 boss = blind 24 = a win
    assert g.furthest_blind == 24


def test_blind_weights():
    from balatro_rl.env import blind_weights
    assert blind_weights(1.0) == [1.0] * 24               # default: every blind worth 1, as before
    w = blind_weights(3.0)
    assert abs(sum(w) - 24) < 1e-9
    assert abs(w[23] / w[0] - 3) < 1e-9                   # ante 8 worth 3x ante 1
    assert w[0] == w[1] == w[2] and w[3] > w[2]           # same within an ante, rising across antes


def test_weighted_rewards_in_play():
    import numpy as np
    from balatro_rl.env import BalatroEnv, blind_weights
    from balatro_rl.heuristic import HeuristicPolicy
    w = blind_weights(3.0)
    env = BalatroEnv("RED", "WHITE", ante_weight=3.0, win_bonus=50.0)
    pol = HeuristicPolicy(rng=np.random.default_rng(0), lookahead=False)
    for seed in range(20):
        obs, done, total = env.reset(seed), False, 0.0
        while not done:
            obs, r, done, info = env.step(pol.act(env.g, obs))
            total += r
        earned = sum(w[:env.g.furthest_blind])
        extra = total - earned
        if info["won"]:
            assert abs(extra - 50.0) < 1e-9
        else:
            assert -1e-9 <= extra <= 0.5 * w[min(env.g.furthest_blind, 23)] + 1e-9
