"""Replay a solved line (seed + actions) from scratch in a fresh exact-seed game and report the outcome.

    python tools/replay_line.py traces/macro_100521.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv      # noqa: E402

d = json.load(open(sys.argv[1]))
v = CppVecEnv(1, d["deck"], d["stake"], seed0=d["seed"], threads=1)
v.set_exact(True)
v.v.restart()
v.set_weight_shape(2.5, 1.0)
v.set_marginal(1)
res = None
for i, a in enumerate(d["actions"]):
    assert v.obs["mask"][0][a], f"illegal action {a} at step {i}"
    _, done, info = v.step(np.array([a]))
    if done[0]:
        res = (i + 1, int(info["blinds"][0]), bool(info["won"][0]), int(info["ante"][0]))
        break
print(f"seed {d['seed']} {d['deck']} {d['stake']}: {len(d['actions'])} actions;",
      "ended after %d steps, blinds %d, won %s, ante %d" % res if res else "did not end")
