"""The strategic environment: PPO only sees strategic decisions, a tactical network plays the cards."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from balatro_rl.env import A_SELECT, blind_weights  # noqa: E402
from balatro_rl.model import ActorCritic, save_model  # noqa: E402
from balatro_rl.strategic import StrategicEnv  # noqa: E402


def test_strategic_env(tmp_path):
    ck = str(tmp_path / "net.pt")
    save_model(ActorCritic(), ck)                          # an untrained network is enough here
    env = StrategicEnv("RED", "WHITE", tactical=ck, margin=0.2, ante_weight=3.0, win_bonus=50.0)
    w = blind_weights(3.0)
    rng = np.random.default_rng(0)
    for seed in range(4):
        obs, done, total, starts = env.reset(seed), False, 0.0, 0
        while not done:
            if env.g.state == "SELECTING_HAND":
                starts += 1
                assert obs["mask"][:A_SELECT].sum() == 1   # only the tactical suggestion starts the blind
            assert env.g.state != "SELECTING_HAND" or not env.handed_over
            obs, r, done, info = env.step(int(rng.choice(np.flatnonzero(obs["mask"]))))
            total += r
        played = env.g.blinds_beaten + (0 if info["won"] else 1)
        assert starts >= played                            # a decision at every blind start
        earned = sum(w[:env.g.furthest_blind])
        margin_cap = 0.2 * env.g.blinds_beaten
        assert earned - 1e-9 <= total <= earned + margin_cap + 50 + 1.5 + 1e-9
