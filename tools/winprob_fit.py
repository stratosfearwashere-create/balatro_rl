"""Fits balatro_rl/winprob.py to a policy's own outcomes: states along the policy's Gold games (blind selects and shops), each played to the end
--rollouts times by the policy on redrawn futures (VecEnv.reseed_rng); the model's expected blinds and win rate fitted to theirs. Held-out
check: Spearman / mean error on states of other games. -> balatro_rl/data/winprob.json

    python tools/winprob_fit.py --ckpt checkpoints/ppo_gold_v7_best_greedy.pt --games 400 --rollouts 16
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import env as E                                          # noqa: E402
from balatro_rl.engine_env import CppVecEnv                               # noqa: E402
from balatro_rl.runeval import from_vecenv, Cols                          # noqa: E402
from balatro_rl.winprob import Features, fit, predict, PARAMS_PATH        # noqa: E402
from policy_search import Pol                                             # noqa: E402
from runeval_fit import spearman                                          # noqa: E402


def collect(pol, games, seed0, p_snap, rng):
    v = CppVecEnv(games, "RED", "GOLD", seed0=seed0, threads=0)
    v.v.restart()
    bank = CppVecEnv(4096, "RED", "GOLD", seed0=1, threads=0)
    owner = []
    done = np.zeros(games, dtype=bool)
    for _ in range(4000):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        mask = np.asarray(v.obs["mask"])
        outr = ~mask[:, :E.A_SELECT].any(1)
        for gi in live:
            if outr[gi] and len(owner) < bank.n and rng.random() < p_snap:
                bank.copy_from(v, int(gi), len(owner))
                owner.append(int(gi))
        act = np.full(games, -1, dtype=np.int64)
        act[live] = pol.logits(v, live).argmax(1)
        _, d, _ = v.step(act)
        done |= d
    return bank, np.array(owner)


def rollouts(pol, bank, m, K, salt0):
    N = m * K
    S = CppVecEnv(N, "RED", "GOLD", seed0=2, threads=0)
    S.copy_many(bank, [j // K for j in range(N)], list(range(N)))
    S.reseed_rng(list(range(N)), [salt0 + j * 7919 for j in range(N)])
    done = np.zeros(N, dtype=bool)
    bl = np.zeros(N)
    won = np.zeros(N, dtype=bool)
    for _ in range(4000):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        a = np.full(N, -1, dtype=np.int64)
        a[live] = pol.logits(S, live).argmax(1)
        _, d, info = S.step(a)
        f = d & ~done
        bl[f], won[f] = info["blinds"][f], info["won"][f]
        done |= d
    return bl.reshape(m, K).mean(1), won.reshape(m, K).mean(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v7_best_greedy.pt")
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--snap", type=float, default=0.15)
    p.add_argument("--seed0", type=int, default=800_000)
    p.add_argument("--out", default=PARAMS_PATH)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    pol = Pol(a.ckpt, "cuda")
    t0 = time.time()
    bank, owner = collect(pol, a.games, a.seed0, a.snap, rng)
    m = len(owner)
    print(f"{m} states from {a.games} games ({time.time() - t0:.0f}s)", flush=True)
    eb, pw = [], []
    for c0 in range(0, m, 512):
        sub = CppVecEnv(min(512, m - c0), "RED", "GOLD", seed0=3, threads=0)
        sub.copy_many(bank, list(range(c0, c0 + sub.n)), list(range(sub.n)))
        x, y = rollouts(pol, sub, sub.n, a.rollouts, 1000 + c0)
        eb.append(x)
        pw.append(y)
    eb, pw = np.concatenate(eb), np.concatenate(pw)
    print(f"rollouts done ({time.time() - t0:.0f}s): mean expected blinds {eb.mean():.2f}, mean win rate {pw.mean():.4f}", flush=True)
    F = Features(**from_vecenv(bank, list(range(m))))
    test = np.isin(owner, np.unique(owner)[::5])                 # every 5th game held out
    tr = ~test

    class Sub:
        pass
    Ftr = Sub()
    for k, val in F.__dict__.items():
        setattr(Ftr, k, val[tr] if isinstance(val, np.ndarray) and len(val) == m else val)
    params, loss = fit(Ftr, eb[tr], pw[tr])
    _, pw_hat, eb_hat = predict(F, params)
    print("fitted:", {k: round(v, 4) for k, v in params.items()}, f"train loss {loss:.3f}")
    print(f"held-out ({test.sum()} states): expected blinds Spearman {spearman(eb_hat[test], eb[test]):.3f}, mean |error| {np.abs(eb_hat[test] - eb[test]).mean():.2f} blinds, "
          f"win rate predicted {pw_hat[test].mean():.4f} vs {pw[test].mean():.4f}")
    by = {}
    ante = F.ante.astype(int)
    for an in range(1, 9):
        mm = test & (ante == an)
        if mm.sum() >= 15:
            by[an] = (int(mm.sum()), round(spearman(eb_hat[mm], eb[mm]), 3), round(float(np.abs(eb_hat[mm] - eb[mm]).mean()), 2))
    print("by ante (n, Spearman, mean |error|):", by)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(params, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
