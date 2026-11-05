"""How good is a PPO checkpoint's critic? Explained variance of its value against the returns the games then give.

The policy (sampled, as in training) plays the first full game of each environment with the training reward (blind potential, ante weight, win bonus, no part credit for a lost blind). At every step
the critic's value is stored; after the game the discounted Monte-Carlo return and the GAE lambda-return of every step are computed with the training gamma and lambda. Explained variance is
1 - Var(return - value) / Var(return): 1 is a perfect critic, 0 no better than a constant, negative worse. Reported over all steps, by ante, and for the steps where the policy is at a
blind select / shop / in a round.

    python tools/critic_ev.py checkpoints/ppo_gold_v30.pt --games 256 --gamma 0.999 --lam 0.98
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS       # noqa: E402
from balatro_rl.model import load_model                      # noqa: E402


def ev(pred, target):
    v = float(np.var(target))
    return float("nan") if v < 1e-12 or len(target) < 20 else 1.0 - float(np.var(target - pred)) / v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--seed0", type=int, default=910000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--ante-weight", type=float, default=2.5)
    p.add_argument("--win-bonus", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lam", type=float, default=0.98)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    model = load_model(a.checkpoint, a.device)
    N = a.games
    venv = CppVecEnv(N, "RED", a.stake, seed0=a.seed0, threads=0, reward_kw={"ante_weight": a.ante_weight, "win_bonus": a.win_bonus})
    venv.set_part_credit(False)
    if getattr(model, "marg", 0):
        venv.set_marginal(int(model.marg))
    vals = [[] for _ in range(N)]
    rews = [[] for _ in range(N)]
    ante = [[] for _ in range(N)]
    kind = [[] for _ in range(N)]
    done_ep = np.zeros(N, dtype=bool)
    gen = torch.Generator(device=a.device).manual_seed(0)
    for _ in range(a.max_steps):
        if done_ep.all():
            break
        ob = {k: torch.from_numpy(np.asarray(venv.obs[k])).to(a.device) for k in OBS_KEYS}
        with torch.no_grad():
            logits, v = model(ob)
        act = torch.multinomial(torch.softmax(logits.float(), 1), 1, generator=gen).squeeze(1).cpu().numpy()
        g = np.asarray(venv.obs["g"])
        mask = np.asarray(venv.obs["mask"])
        v = v.float().cpu().numpy().reshape(-1)
        rew, done, _ = venv.step(act)
        for e in range(N):
            if done_ep[e]:
                continue
            vals[e].append(float(v[e]))
            rews[e].append(float(rew[e]))
            ante[e].append(int(round(g[e, 0] * 8)))
            kind[e].append(0 if mask[e, 436] else 1 if mask[e, 468] else 2)           # 0 blind select, 1 shop, 2 anything else (a round, a pack)
            if done[e]:
                done_ep[e] = True
    P, G, L, A, K = [], [], [], [], []
    for e in range(N):
        if not done_ep[e]:
            continue
        r, vv = np.array(rews[e]), np.array(vals[e])
        T = len(r)
        mc = np.zeros(T)
        lr = np.zeros(T)
        run = 0.0
        run_l = 0.0
        for t in range(T - 1, -1, -1):
            run = r[t] + a.gamma * run
            nv = vv[t + 1] if t + 1 < T else 0.0
            run_l = r[t] + a.gamma * ((1 - a.lam) * nv + a.lam * run_l) if t + 1 < T else r[t]
            mc[t], lr[t] = run, run_l
        P += list(vv)
        G += list(mc)
        L += list(lr)
        A += ante[e]
        K += kind[e]
    P, G, L, A, K = map(np.array, (P, G, L, A, K))
    print(f"{a.checkpoint}: {int(done_ep.sum())} finished games, {len(P)} steps, return mean {G.mean():.2f} std {G.std():.2f}, value mean {P.mean():.2f} std {P.std():.2f}")
    print(f"explained variance vs the Monte-Carlo return (gamma {a.gamma}): {ev(P, G):+.3f}   vs the lambda-return (lambda {a.lam}): {ev(P, L):+.3f}")
    print("by ante (MC):   " + "  ".join(f"{k}: {ev(P[A == k], G[A == k]):+.2f} (n {int((A == k).sum())})" for k in range(1, 9) if (A == k).sum() > 20))
    print("by decision (MC): " + "  ".join(f"{n}: {ev(P[K == k], G[K == k]):+.2f}" for k, n in ((0, 'blind select'), (1, 'shop'), (2, 'round/pack'))))
    sel = K == 0
    print(f"at blind selects: value vs return correlation {np.corrcoef(P[sel], G[sel])[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
