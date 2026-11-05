"""Greedy vs one-step lookahead (balatro_rl/eval_lookahead.py) on the same Gold seeds, with every round played by the exact solver (as v46 trains).

    python tools/eval_la.py --ckpt checkpoints/ppo_gold_v46.pt --games 50 --k 8 --device cpu --threads 4
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS          # noqa: E402
from balatro_rl.eval_lookahead import Lookahead                # noqa: E402
from balatro_rl.model import load_model                        # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v46.pt")
    p.add_argument("--config", default="checkpoints/ppo_gold_v46.json", help="the run's args: reward shape, solver settings")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed0", type=int, default=9_000_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--modes", default="greedy,lookahead")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--scorer", default="critic", choices=["critic", "hv"], help="lookahead: score moves by the critic (+ potentials) or the hand-built value hv.HV")
    a = p.parse_args()
    args = json.load(open(a.config))["args"]

    def arg(k, d=None, flag=False):
        if flag:
            return k in args
        return type(d)(args[args.index(k) + 1]) if k in args else d
    kw = {"ante_weight": arg("--ante-weight", 1.0), "win_bonus": arg("--win-bonus", 10.0), "win_ante": 8, "joker_pool": None, "shape_chips": 0.0,
          "shape_phi": 0.0, "reward_config": None, "shape_gamma": arg("--gamma", 0.995)}
    torch.set_num_threads(a.threads)
    net = load_model(a.ckpt, a.device).eval()
    nodes = arg("--autoplay-nodes", 400)
    tar = arg("--solver-tarots", flag=True)

    def mk(n, seed):
        v = CppVecEnv(n, "RED", a.stake, seed0=seed, threads=a.threads, reward_kw=kw, verdant_bonus=arg("--verdant-bonus", 0.0))
        if arg("--weight-power", 1.0) != 1.0:
            v.set_weight_shape(arg("--ante-weight", 1.0), arg("--weight-power", 1.0))
        if getattr(net, "marg", 0):
            v.set_marginal(int(net.marg))
        v.set_autoplay(True, nodes, True)
        if tar:
            v.v.set_solver_tarots(True, arg("--solver-targets", 3))
        if arg("--solver-smart", flag=True):
            v.v.set_solver_order(True)
        if arg("--solver-lite", flag=True):
            v.v.set_solver_speed(True, False, 0.0)
        if arg("--solver-tt", flag=True):
            v.v.set_solver_tt(True)
        v.v.restart()
        return v

    for mode in a.modes.split(","):
        t0 = time.time()
        ev = mk(a.games, a.seed0)
        la = None
        if mode == "lookahead":
            from balatro_rl.shaping_pot import ShapingPot                       # (the trainer's per-step potentials: the critic learned V - phi)
            pot = ShapingPot(a.device, build_roles=arg("--build-roles", "xmult,mult,chips"), build_second=arg("--build-second", 0.0), bank=arg("--bank-bonus", 0.0),
                             build=arg("--build-bonus", 0.0), rental=arg("--rental-penalty", 0.0), debt=arg("--debt-penalty", 0.0))
            scorer = None
            if a.scorer == "hv":
                from balatro_rl.hv import HV
                scorer = HV.load()
            la = Lookahead(lambda s: mk(s, a.seed0 + 77), net, a.device, k=a.k, slots=256, pot=None if scorer else pot, scorer=scorer)
        n = a.games
        fin = np.zeros(n, bool)
        bl, won, ante = np.zeros(n), np.zeros(n, bool), np.zeros(n)
        with torch.no_grad():
            for _ in range(6000):
                o = {k: torch.from_numpy(np.asarray(ev.obs[k])).to(a.device) for k in OBS_KEYS}
                lg, _ = net(o)
                act = la.act(ev, lg, fin) if la else np.where(fin, -1, lg.argmax(1).cpu().numpy())
                _, d, info = ev.step(act)
                f = d & ~fin
                bl[f], won[f], ante[f] = info["blinds"][f], info["won"][f], info["ante"][f]
                fin |= d
                if fin.all():
                    break
        hist = [int(((np.where(won, 9, ante)) == k).sum()) for k in range(1, 10)]
        extra = f", lookahead overruled the policy on {la.n_changed}/{la.n_decisions} decisions" if la else ""
        print(f"{mode:9s}: {n} {a.stake.title()} games, {bl.mean():.2f} blinds, {int(won.sum())} wins; furthest ante 1..8, won: {hist}{extra} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
