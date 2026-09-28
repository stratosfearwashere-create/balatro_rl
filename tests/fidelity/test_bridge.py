"""The bridge must rebuild, from API JSON alone, a state the model sees the same way as the simulator."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from balatro_rl.bridge import game_from_state, to_rpc  # noqa: E402
from balatro_rl.env import encode, N_SUB, A_PLAY, RATIO_COL, A_REROLL_BOSS  # noqa: E402
from balatro_rl.heuristic import HeuristicPolicy  # noqa: E402
from mock_balatrobot import Mock  # noqa: E402



def test_bridge_matches_simulator():
    checked = mismatched_masks = 0
    for seed in range(15):
        mock = Mock()
        mock.handle("start", {"deck": "RED", "stake": "GOLD", "seed": str(seed)})
        h = HeuristicPolicy(lookahead=False)
        memory = {}
        for _ in range(400):
            gs = mock.state()
            if gs["state"] == "GAME_OVER":
                break
            if gs["state"] == "ROUND_EVAL":
                mock.handle("cash_out", None)
                continue
            sim_obs = encode(mock.g)
            sim_obs["mask"][A_REROLL_BOSS] = False
            g2, hmap = game_from_state(gs, memory=memory)
            br_obs = encode(g2)
            br_obs["mask"][A_REROLL_BOSS] = False
            checked += 1
            if not np.array_equal(sim_obs["mask"], br_obs["mask"]):
                mismatched_masks += 1
            if gs["state"] == "SELECTING_HAND":
                a = sim_obs["af"][A_PLAY:A_PLAY + N_SUB, RATIO_COL]
                b = br_obs["af"][A_PLAY:A_PLAY + N_SUB, RATIO_COL]
                # predicted scores agree unless a joker's hidden internal state differs
                assert np.mean(np.isclose(a, b, rtol=1e-3, atol=1e-3)) > 0.95
            # act through the API using the bridge's own translation
            act = h.act(g2, br_obs)
            method, params = to_rpc(g2, act, hmap, br_obs)
            mock.handle(method, params)
    assert checked > 200
    assert mismatched_masks / checked < 0.05, (mismatched_masks, checked)
