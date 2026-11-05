"""Late-game skill of a policy: how it does from the late positions its own games reach. Greedy, every round by the exact solver (as the run trains).

    1. COLLECT  --collect normal Gold games are played; the start of every blind at antes 5-7 they reach is kept (the engine's state archive, a reservoir per blind)
    2. REPLAY   for each start ante, --games games begin from those positions (a random kept one each, its future redrawn) and play to the end:
                win rate, blinds beaten after the start, how far they get

    python tools/late_eval.py --ckpt checkpoints/ppo_gold_v51.pt --config checkpoints/ppo_gold_v51.json --collect 2000 --games 1000
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS              # noqa: E402
from balatro_rl.model import load_model                            # noqa: E402


def arg_of(args, k, d):
    return type(d)(args[args.index(k) + 1]) if k in args else d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v51.pt")
    p.add_argument("--config", default="checkpoints/ppo_gold_v51.json")
    p.add_argument("--collect", type=int, default=2000)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--antes", default="5,6,7")
    p.add_argument("--seed0", type=int, default=9_400_000)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    args = json.load(open(a.config))["args"]
    torch.set_num_threads(a.threads)
    net = load_model(a.ckpt, "cpu").eval()

    def make(n, seed):
        v = CppVecEnv(n, "RED", "GOLD", seed0=seed, threads=a.threads)
        if getattr(net, "marg", 0):
            v.set_marginal(int(net.marg))
        v.set_autoplay(True, arg_of(args, "--autoplay-nodes", 400), True)
        if "--solver-tarots" in args:
            v.v.set_solver_tarots(True, 3)
        if "--solver-smart" in args:
            v.v.set_solver_order(True)
        if "--solver-lite" in args:
            v.v.set_solver_speed(True, False, 0.0)
        if "--solver-tt" in args:
            v.v.set_solver_tt(True)
        return v

    @torch.no_grad()
    def play(v, n, max_steps=4000):
        done = np.zeros(n, bool)
        bl, won, ante = np.zeros(n), np.zeros(n, bool), np.zeros(n)
        for _ in range(max_steps):
            if done.all():
                break
            o = {k: torch.from_numpy(np.asarray(v.obs[k])) for k in OBS_KEYS}
            lg, _ = net(o)
            _, d, info = v.step(np.where(done, -1, lg.argmax(1).numpy()))
            f = d & ~done
            bl[f], won[f], ante[f] = info["blinds"][f], info["won"][f], info["ante"][f]
            done |= d
        return bl, won, ante

    antes = [int(x) for x in a.antes.split(",")]
    col = make(a.collect, a.seed0)
    col.v.set_archive(0.0, 1.0, min(antes), max(antes), 400, 1, False)        # keep every blind start (a reservoir of 400 per blind)
    col.v.restart()
    bl, won, ante = play(col, a.collect)
    fill = col.archive_counts(-1)
    print(json.dumps({"collect": a.collect, "blinds": round(float(bl.mean()), 3), "wins": int(won.sum()),
                      "kept_by_ante": {an: int(sum(fill[3 * (an - 1):3 * an])) for an in antes}}), flush=True)
    for an in antes:
        if sum(fill[3 * (an - 1):3 * an]) < 3:
            print(json.dumps({"start_ante": an, "skipped": "too few positions reached"}), flush=True)
            continue
        rep = make(a.games, a.seed0 + 1000 * an)
        rep.v.share_archive(col.v)
        rep.v.set_archive(1.0, 0.0, an, an, 400, 1, False)                     # every game starts from a kept blind start of this ante
        rep.v.restart()
        b2, w2, a2 = play(rep, a.games)
        start = 3 * (an - 1)
        furthest = np.where(w2, 9, a2)
        print(json.dumps({"start_ante": an, "games": a.games, "win_rate": round(float(w2.mean()), 4), "blinds_after_start": round(float((b2 - start).mean()), 2),
                          "reach": {f"ante {k}+": round(float((furthest >= k).mean()), 3) for k in range(an + 1, 10) if k <= 8} | {"won": round(float(w2.mean()), 4)}}),
              flush=True)
        rep.close()


if __name__ == "__main__":
    main()
