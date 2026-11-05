"""Replays every line the seed-aware beam (tools/solve_novel.py) wrote in a directory, in a fresh exact-seed game, and checks the claimed result:
a SOLVED line must end with the game won, every action must be legal. Prints per seed: verified or not, blinds, skips, the final jokers.

    python tools/verify_beam.py traces/beam_retest
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import make                                  # noqa: E402
from balatro_rl.env import A_SKIP                            # noqa: E402
from balatro_rl.sim.items import VOCAB                       # noqa: E402


def replay(seed, actions, deck="RED", stake="GOLD", order_mode=0):
    v = make(1, deck, stake, seed, 1)
    v.set_order_mode(order_mode)
    v.set_auto_target(os.environ.get("AUTO_TARGET", "0") == "1")
    skips, last_jokers = 0, []
    for k, a in enumerate(actions):
        if a < 0:                                            # -3: select and let the scripted player play (shop-only search); not used here
            return dict(ok=False, why=f"special action {a} at step {k}")
        mask = np.asarray(v.obs["mask"])[0]
        if not mask[a]:
            return dict(ok=False, why=f"illegal action {a} at step {k}")
        skips += int(a == A_SKIP)
        last_jokers = [VOCAB[x][2:] for x in np.asarray(v.obs["jid"])[0] if x > 0]
        _, d, info = v.step(np.array([a]))
        if d[0]:
            return dict(ok=True, won=bool(np.asarray(info["won"])[0] > 0.5), blinds=int(np.asarray(info["blinds"])[0]), steps=k + 1, skips=skips, jokers=last_jokers)
    return dict(ok=True, won=False, blinds=None, steps=len(actions), skips=skips, jokers=last_jokers, unfinished=True)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "traces/beam_retest"
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    claimed = verified = 0
    for f in files:
        j = json.load(open(f))
        if not j.get("solved"):
            continue
        claimed += 1
        r = replay(j["seed"], j["actions"], j.get("deck", "RED"), j.get("stake", "GOLD"), int(os.environ.get("ORDER_MODE", "0")))
        good = r["ok"] and r.get("won")
        verified += int(good)
        print(f"seed {j['seed']}: {'VERIFIED WIN' if good else 'NOT A WIN'}  {r.get('why', '')} blinds {r.get('blinds')} skips {r.get('skips')} decisions {r.get('steps')} jokers {r.get('jokers')}")
    print(f"{verified}/{claimed} claimed wins verified by replay ({len(files)} seeds searched)")


if __name__ == "__main__":
    main()
