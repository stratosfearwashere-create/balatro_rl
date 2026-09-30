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


def _run(env, seed):
    import numpy as np
    from balatro_rl.heuristic import HeuristicPolicy
    pol = HeuristicPolicy(rng=np.random.default_rng(0), lookahead=False)
    obs, done, rews = env.reset(seed), False, []
    while not done:
        obs, r, done, info = env.step(pol.act(env.g, obs))
        rews.append(r)
    return rews


def test_shaping_telescopes():
    """Potential-based shaping with gamma=1 adds exactly -Phi(start) over a run: no free reward."""
    from balatro_rl.env import BalatroEnv
    for seed in range(8):
        plain = _run(BalatroEnv("RED", "WHITE"), seed)
        env = BalatroEnv("RED", "WHITE", shape_chips=0.5, shape_phi=1.0, shape_gamma=1.0)
        phi0 = env.build(__import__("balatro_rl.sim.game", fromlist=["Game"]).Game(seed=seed, stake="WHITE"))
        shaped = _run(env, seed)
        assert len(plain) == len(shaped)                         # same run: shaping doesn't touch the game
        assert abs(sum(shaped) - sum(plain) + phi0) < 1e-9       # Phi(start): the build potential only


def test_shaping_rewards_blind_progress():
    from balatro_rl.env import BalatroEnv
    import numpy as np
    from balatro_rl.heuristic import HeuristicPolicy
    env = BalatroEnv("RED", "WHITE", shape_chips=0.5, shape_gamma=1.0)
    pol = HeuristicPolicy(rng=np.random.default_rng(0), lookahead=False)
    obs, seen = env.reset(0), 0
    while obs is not None and seen < 5:
        g = env.g
        was_in_blind, before = g.state == "SELECTING_HAND", g.chips
        obs, r, done, _ = env.step(pol.act(g, obs))
        if was_in_blind and g.state == "SELECTING_HAND" and g.chips > before and g.furthest_blind == 0:
            assert abs(r - 0.5 * (g.chips - before) / g.target) < 1e-9   # ante 1 blinds weigh 1
            seen += 1
        if done:
            break
    assert seen > 0
