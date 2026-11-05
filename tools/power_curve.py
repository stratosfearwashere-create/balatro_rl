"""Build power against the blinds, and build failures against play failures.

Greedy games of a checkpoint. At the start of every blind the game is copied and the exact round solver (it sees the draws: an upper bound on round play) finds the best share of the blind's
chip target any line reaches with the build as it is. That is the build's POWER at that blind; it does not depend on how the policy plays the round.

Reported:
  * power by ante and blind (median and quartiles of log10 share; 0 = exactly the target): does the build keep pace as the targets grow?
  * for every lost game, the blind it died on: a BUILD loss if even the solver could not clear it (share < 1), a PLAY loss if the solver could (share >= 1) but the policy did not;
  * the margin the game had one blind before it died (how close to the edge it was walking).

    python tools/power_curve.py checkpoints/ppo_gold_v30_best.pt --games 200
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS       # noqa: E402
from balatro_rl.env import A_SELECT                           # noqa: E402
from balatro_rl.model import load_model                      # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--node-cap", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    model = load_model(a.checkpoint, a.device)
    N = a.games
    venv = CppVecEnv(N, "RED", a.stake, seed0=a.seed0, threads=0)
    scratch = CppVecEnv(N, "RED", a.stake, seed0=1, threads=0)
    if getattr(model, "marg", 0):
        venv.set_marginal(int(model.marg))
        scratch.set_marginal(int(model.marg))
    done_ep = np.zeros(N, dtype=bool)
    hist = [[] for _ in range(N)]                         # per game: (ante, blind index within the ante, solver share)
    blinds_beaten = np.zeros(N, dtype=int)
    won = np.zeros(N, dtype=bool)
    for _ in range(10000):
        if done_ep.all():
            break
        ob = {k: torch.from_numpy(np.asarray(venv.obs[k])).to(a.device) for k in OBS_KEYS}
        g = np.asarray(venv.obs["g"])
        with torch.no_grad():
            logits, _ = model(ob)
        act = logits.argmax(1).cpu().numpy()
        sel = [e for e in range(N) if not done_ep[e] and act[e] == A_SELECT]
        ante_now = np.rint(g[:, 0] * 8).astype(int)
        _, done, info = venv.step(act)
        live = [e for e in sel if not done[e]]
        if live:                                          # the blinds just begun: their power, measured on a copy
            scratch.copy_many(venv, live, live)
            r = scratch.solve_best(live, a.node_cap)
            for k, e in enumerate(live):
                bidx = len([h for h in hist[e] if h[0] == ante_now[e]])
                hist[e].append((int(ante_now[e]), min(bidx, 2), float(r[k, 0])))
        for e in np.flatnonzero(done):
            if not done_ep[e]:
                done_ep[e] = True
                blinds_beaten[e] = info["blinds"][e]
                won[e] = info["won"][e]
    # power by ante / blind
    by = defaultdict(list)
    for e in range(N):
        for ante, b, s in hist[e]:
            by[(ante, b)].append(np.log10(max(s, 1e-3)))
    print(f"{os.path.basename(a.checkpoint)}: {N} greedy Gold games, {blinds_beaten.mean():.2f} blinds, {int(won.sum())} wins")
    print("\nBUILD POWER at the start of each blind: log10(best share of the target the exact solver reaches), 0 = exactly the target")
    print(f"{'ante':>4s} {'blind':>6s} {'n':>5s} {'p25':>7s} {'median':>7s} {'p75':>7s} {'share >= 1':>11s}")
    for ante in range(1, 9):
        for b, name in enumerate(("small", "big", "boss")):
            v = by.get((ante, b))
            if not v or len(v) < 5:
                continue
            v = np.array(v)
            print(f"{ante:4d} {name:>6s} {len(v):5d} {np.percentile(v, 25):+7.2f} {np.median(v):+7.2f} {np.percentile(v, 75):+7.2f} {100 * (v >= 0).mean():10.0f}%")
    # loss attribution
    build_l, play_l = defaultdict(int), defaultdict(int)
    edge = []
    for e in range(N):
        if won[e] or not hist[e]:
            continue
        ante, b, s = hist[e][-1]
        (play_l if s >= 1.0 else build_l)[ante] += 1
        if len(hist[e]) >= 2:
            edge.append(np.log10(max(hist[e][-2][2], 1e-3)))
    nb, npl = sum(build_l.values()), sum(play_l.values())
    print(f"\nLOSSES: {nb + npl}; BUILD losses (even the solver could not clear the blind) {nb} ({100 * nb / max(nb + npl, 1):.0f}%), "
          f"PLAY losses (the solver could) {npl} ({100 * npl / max(nb + npl, 1):.0f}%)")
    print("  by ante, build / play: " + ", ".join(f"{k}: {build_l[k]}/{play_l[k]}" for k in sorted(set(build_l) | set(play_l))))
    if edge:
        edge = np.array(edge)
        print(f"  power one blind before the death: median {np.median(edge):+.2f} (log10), {100 * (edge < np.log10(1.5)).mean():.0f}% under 1.5x the target")


if __name__ == "__main__":
    main()
