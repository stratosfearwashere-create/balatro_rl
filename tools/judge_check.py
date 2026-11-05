"""Does the judge's score predict how far a real game goes, better than trivial features?

A policy plays full games (real rounds); at the first blind select of each of antes 1-4 the state is scored five ways:
    phi        the judge potential (potential.py, full: scalers grown to ante 8, the ante-8 boss's pass chance, money term)
    p_now      the judge's pass chance for the blind about to be played
    ratio      the exact scorer's number alone: log10 of the best play's score over the blind's target
    money, n_jokers
Each is compared with how many blinds the game went on to beat (Spearman rank correlation, per ante, over games), and with whether it reached ante 6.

    python tools/judge_check.py checkpoints/ppo_gold_v30_best.pt --games 256
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import solve_plan                                                                # noqa: E402,F401  (loads the development engine: set_state)
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS                            # noqa: E402
from balatro_rl.env import global_size                                           # noqa: E402
from balatro_rl.marginal import MARG_G, POT_G                                    # noqa: E402
from balatro_rl.model import load_model                                          # noqa: E402
from balatro_rl import judge as J                                                # noqa: E402
from balatro_rl.potential import JudgePotential                                  # noqa: E402
from balatro_rl.power import PassProbability                                      # noqa: E402

BASE = global_size() - MARG_G - POT_G


def spearman(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--judge", default="checkpoints/judge2.pt")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    dev = a.device
    model = load_model(a.checkpoint, "cpu")
    judge = J.load(a.judge, dev)
    N = a.games
    v = CppVecEnv(N, "RED", a.stake, seed0=a.seed0, threads=0)
    la = CppVecEnv(N, "RED", a.stake, seed0=1, threads=0)
    for e in (v, la):
        e.set_marginal(1)
    pot = JudgePotential(v, la, judge, None, dev, -40.0, 0.5, True)
    pp = PassProbability(k=32, cap=4096)
    seen = [dict() for _ in range(N)]          # per game: ante -> the features at its first blind select
    final = np.full(N, -1.0)
    won = np.zeros(N, dtype=bool)
    done = np.zeros(N, dtype=bool)
    for step in range(5000):
        g = np.asarray(v.obs["g"])
        ante = np.rint(g[:, 0] * 8).astype(int)
        sel = np.flatnonzero((g[:, 4] > 0.5) & (ante <= 4) & ~done)
        todo = [e for e in sel if ante[e] not in seen[e]]
        with torch.no_grad():
            o = {k: torch.from_numpy(np.array(v.obs[k])) for k in OBS_KEYS if k in v.obs}
            lg, val = model(o)
        val = val.float().numpy()
        if todo:
            pmc = pp(v, todo)["p"]
            ph = pot.phi(todo)
            gg, jid, jf, cid = pot._obs(v, todo)
            pn = pot._p_pass(gg, jid, jf, cid).cpu().numpy()
            for k, e in enumerate(todo):
                seen[e][int(ante[e])] = {"phi": float(ph[k]), "p_now": float(pn[k]), "ratio": float(7.0 * g[e, BASE] - 6.0 * g[e, 8]), "money": float(g[e, 17] * 50.0),
                                         "n_jokers": float((np.asarray(v.obs["jid"])[e] > 0).sum()), "critic": float(val[e]), "p_mc": float(pmc[k])}
        act = lg.float().argmax(1).numpy()
        act = np.where(done, -1, act)
        _, d, info = v.step(act)
        first = d & ~done
        final[first], won[first] = info["blinds"][first], info["won"][first]
        done |= d
        if done.all():
            break
    print(f"{a.games} {a.stake} games with {os.path.basename(a.checkpoint)}: mean blinds {final.mean():.2f}, wins {int(won.sum())}")
    print("Spearman correlation of each score, taken at the first blind select of the ante, with the blinds the game went on to beat:")
    print(f"{'ante':>5s} {'n':>5s} {'critic V':>9s} {'sampled pass':>13s} {'exact ratio':>12s} {'judge pass':>11s} {'judge phi':>10s} {'money':>7s} {'n_jokers':>9s}")
    for ante in (1, 2, 3, 4):
        rows = [(seen[e][ante], final[e]) for e in range(N) if ante in seen[e] and final[e] >= 0]
        if not rows:
            continue
        y = [r[1] for r in rows]
        print(f"{ante:5d} {len(rows):5d} " + " ".join(f"{spearman([r[0][k] for r in rows], y):{w}.3f}" for k, w in (("critic", 9), ("p_mc", 13), ("ratio", 12), ("p_now", 11), ("phi", 10), ("money", 7), ("n_jokers", 9))))


if __name__ == "__main__":
    main()
