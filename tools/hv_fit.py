"""Fit and check the hand-built value function (balatro_rl/hv.py) on a policy's own games.

    1. PLAY     --games greedy Gold games (every round by the exact solver); at each blind select: the build's curve (VecEnv.power_curve), money, open slots
    2. FIT G    from the first blind select of ante a to that of ante a+1 in the same game: ln(share at the ante-8 boss) grows by ln G; least squares on
                [1, ln(1 + money), open slots, ante]
    3. CHECK    V's predicted blinds still to come against the blinds each game actually went on to beat (correlation, mean, by ante), held-out games

    python tools/hv_fit.py --ckpt checkpoints/ppo_gold_v51.pt --config checkpoints/ppo_gold_v51.json --games 1000
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS              # noqa: E402
from balatro_rl.hv import HV, PATH                                  # noqa: E402
from balatro_rl.model import load_model                            # noqa: E402
from balatro_rl.runeval import Cols                                # noqa: E402


def arg_of(args, k, d):
    return type(d)(args[args.index(k) + 1]) if k in args else d


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v51.pt")
    p.add_argument("--config", default="checkpoints/ppo_gold_v51.json")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed0", type=int, default=9_600_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default=PATH)
    a = p.parse_args()
    args = json.load(open(a.config))["args"]
    net = load_model(a.ckpt, a.device).eval()
    n = a.games
    v = CppVecEnv(n, "RED", "GOLD", seed0=a.seed0, threads=a.threads)
    if getattr(net, "marg", 0):
        v.set_marginal(int(net.marg))
    v.set_autoplay(True, arg_of(args, "--autoplay-nodes", 400), True)
    v.v.set_solver_tarots(True, 3); v.v.set_solver_order(True); v.v.set_solver_speed(True, False, 0.0); v.v.set_solver_tt(True)
    v.v.restart()
    hv = HV()
    recs = defaultdict(list)                      # game -> [(position, shares, ante, blind, money, slots)]
    fin = np.zeros(n, bool)
    final = np.zeros(n)
    won = np.zeros(n, bool)
    last_pos = np.full(n, -1)
    with torch.no_grad():
        for _ in range(6000):
            o = {k: torch.from_numpy(np.asarray(v.obs[k])).to(a.device) for k in OBS_KEYS if k in v.obs}
            g = np.asarray(v.obs["g"])
            st = np.argmax(g[:, Cols.STATE:Cols.STATE + 4], 1)
            ante = np.clip(np.rint(g[:, Cols.ANTE] * 8), 1, 8).astype(int)
            blind = np.argmax(g[:, Cols.BLIND:Cols.BLIND + 3], 1)
            pos = 3 * (ante - 1) + blind
            sel = [int(i) for i in np.flatnonzero((st == 0) & ~fin & (pos != last_pos))]       # (each blind select once)
            if sel:
                sh, an, bl, mo, sl = hv.features(v, sel)
                for k, i in enumerate(sel):
                    recs[i].append((int(pos[i]), sh[k], int(an[k]), int(bl[k]), float(mo[k]), float(sl[k])))
                    last_pos[i] = pos[i]
            lg, _ = net(o)
            _, d, info = v.step(np.where(fin, -1, lg.argmax(1).cpu().numpy()))
            f = d & ~fin
            final[f], won[f] = info["blinds"][f], info["won"][f]
            fin |= d
            if fin.all():
                break
    print(json.dumps({"games": n, "blinds": round(float(final.mean()), 3), "wins": int(won.sum()), "states": sum(len(x) for x in recs.values())}), flush=True)
    games = sorted(recs)
    rng = np.random.default_rng(0)
    test = set(rng.choice(games, size=len(games) // 5, replace=False).tolist())
    # 2. G: consecutive antes' first blind selects
    X, y = [], []
    for gi in games:
        if gi in test:
            continue
        firsts = {}
        for r in recs[gi]:
            if r[3] == 0 and r[2] not in firsts:
                firsts[r[2]] = r
        for an_, r in firsts.items():
            nx = firsts.get(an_ + 1)
            if nx is None or r[1][7] <= 0 or nx[1][7] <= 0:
                continue
            X.append([1.0, np.log1p(max(r[4], 0)), r[5], float(an_)])
            y.append(np.log(nx[1][7]) - np.log(r[1][7]))
    X, y = np.asarray(X), np.asarray(y)
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    print(json.dumps({"fit_G": {"pairs": len(y), "c": [round(float(x), 4) for x in c], "mean_lnG": round(float(y.mean()), 3),
                                "G_at_mean": round(float(np.exp((X @ c).mean())), 3), "r2": round(float(1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)}}), flush=True)
    params = dict(hv.p, c=[float(x) for x in c])
    # calibration: one shrink k on the growth (ln G -> k ln G) chosen on the training games so the predicted blinds still to come match on average
    tr_states = [(gi, r) for gi in games if gi not in test for r in recs[gi][::3]]
    act_tr = np.array([final[gi] - r[0] + (hv.p["win_bonus"] if won[gi] else 0.0) for gi, r in tr_states])
    best = None
    for k in np.linspace(0.0, 1.2, 25):
        hk = HV(dict(params, c=[float(x) * k for x in c]))
        pr = np.array([hk.value(r[1][None], np.array([r[2]]), np.array([r[3]]), np.array([r[4]]), np.array([r[5]]))[0] for gi, r in tr_states])
        err = abs(pr.mean() - act_tr.mean())
        if best is None or err < best[0]:
            best = (err, k)
    params["c"] = [float(x) * best[1] for x in c]
    params["g_shrink"] = float(best[1])
    print(json.dumps({"calibration": {"shrink": round(float(best[1]), 3), "train_states": len(tr_states)}}), flush=True)
    hv2 = HV(params)
    # 3. check on held-out games: predicted blinds still to come vs actual
    for name, model in (("no shop growth (G = 1)", hv), ("fitted G", hv2)):
        pred, act, ants = [], [], []
        for gi in test:
            for r in recs[gi]:
                pr = model.value(r[1][None], np.array([r[2]]), np.array([r[3]]), np.array([r[4]]), np.array([r[5]]))[0] - model.p["win_bonus"] * 0
                pred.append(pr); act.append(final[gi] - r[0] + (model.p["win_bonus"] if won[gi] else 0.0)); ants.append(r[2])
        pred, act, ants = np.asarray(pred), np.asarray(act), np.asarray(ants)
        by = {int(k): round(spearman(pred[ants == k], act[ants == k]), 3) for k in range(1, 9) if (ants == k).sum() > 30}
        print(json.dumps({"check": name, "states": len(pred), "spearman": round(spearman(pred, act), 3), "pred_mean": round(float(pred.mean()), 2),
                          "actual_mean": round(float(act.mean()), 2), "spearman_within_ante": by}), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(params, open(a.out, "w"), indent=2)
    print("->", a.out)


if __name__ == "__main__":
    main()
