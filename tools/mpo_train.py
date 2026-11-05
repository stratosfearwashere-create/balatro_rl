"""Discrete MPO (maximum a posteriori policy optimisation) on the Gold game: python tools/mpo_train.py --init checkpoints/ppo_gold_imm2.pt --out checkpoints/mpo1.pt

The game is normal Gold rules, immortal with pay-on-fail (every game runs to the ante-8 boss; the reward is each blind beaten, geometric x1.6 per ante, and the win), no shaping.
The network has the policy head and a Q head over the same (state, action) rows (model.add_qhead): Q(s, a) for every legal action.
Each iteration:
  1. DATA. T steps of N environments with the current policy (sampled); the policy's log-probabilities over all actions are kept (pi_old).
  2. CRITIC TARGETS. Rewards are divided by the running std of the discounted return. V(s) = sum_a pi(a|s) Q(s, a); the lambda-returns G_t = r_t + gamma [(1 - lambda) V(s_t+1) + lambda G_t+1]
     (bootstrapped from V at the end of the rollout, cut at a game's end). Q(s_t, a_t) is regressed to G_t.
  3. E-STEP. For each state the target over the legal actions is q(a|s) proportional to pi_old(a|s) exp(Q(s, a) / eta): the old policy tilted toward the actions with higher Q. eta is set by its
     dual, which keeps KL(q || pi_old) near epsilon.
  4. M-STEP. The policy is fitted to q (cross entropy) under a trust region: beta * KL(pi_old || pi), beta by its dual, which keeps that KL near epsilon_m.
Evaluation (every --eval-every iterations): greedy and sampled blinds on normal (mortal) Gold, 100 fixed games.
"""
import argparse
import copy
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS      # noqa: E402
from balatro_rl.model import load_model                     # noqa: E402
from balatro_rl.streams import ReturnNorm                   # noqa: E402
from branch_train import evaluate                           # noqa: E402

KEYS = [k for k in OBS_KEYS if k != "priv"]


def dev_obs(env, dev):
    return {k: torch.as_tensor(np.array(env.obs[k], copy=True)).to(dev) for k in KEYS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="", help="start from this checkpoint (default: a fresh, randomly initialised network: attention over jokers and hand, item descriptors, Q head)")
    p.add_argument("--out", default="checkpoints/mpo1.pt")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=64, help="steps per iteration per environment")
    p.add_argument("--epochs", type=int, default=4, help="passes over the iteration's states")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--q-coef", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=0.1, help="E-step KL bound (eta's dual)")
    p.add_argument("--eps-m", type=float, default=0.01, help="M-step KL bound (beta's dual)")
    p.add_argument("--explore", type=float, default=0.02, help="share of random legal actions in the data")
    p.add_argument("--weight-geom", type=float, default=1.6)
    p.add_argument("--win-bonus", type=float, default=30.0)
    p.add_argument("--q-warmup", type=int, default=10, help="iterations the policy is not updated (the Q head starts at zero)")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--iters", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev = torch.device(a.device)
    amp = str(dev).startswith("cuda")
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    if a.init:
        model = load_model(a.init, dev)
    else:
        from balatro_rl.model import ActorCritic
        model = ActorCritic(attn=2, desc=1, qhead=1).to(dev)
    if not hasattr(model, "qhead"):
        model.add_qhead()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    log_eta = torch.tensor(np.log(0.5), device=dev, requires_grad=True)
    log_beta = torch.tensor(np.log(1.0), device=dev, requires_grad=True)
    dual_opt = torch.optim.Adam([log_eta, log_beta], lr=1e-2)

    env = CppVecEnv(a.envs, a.deck, a.stake, seed0=a.seed * 1_000_000 + 7, threads=0, reward_kw={"win_bonus": a.win_bonus})
    env.set_weight_geom(a.weight_geom)
    env.set_immortal(True, True)
    env.v.restart()
    norm = ReturnNorm(a.envs, a.gamma, 0.5)
    N, T = a.envs, a.steps
    log = a.out.replace(".pt", "_log.jsonl")
    base_g = evaluate(model.eval(), a, dev, n=a.eval_games)
    base_s = evaluate(model.eval(), a, dev, n=a.eval_games, sampled=True)
    print(json.dumps({"initial_eval": base_g, "initial_eval_sampled": base_s}), flush=True)
    model.train()
    games = wins = 0
    win_blinds = deque(maxlen=2000)
    t_start = time.time()
    good = copy.deepcopy(model.state_dict())

    for it in range(1, a.iters + 1):
        t0 = time.time()
        # ---------------------------------------------------------------- 1. data
        model.eval()
        store = {k: [] for k in KEYS}
        acts, lps, rews, dones = [], [], [], []
        clear_acc = np.zeros(24)
        clear_n = 0
        for t in range(T):
            ob = dev_obs(env, dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                lg, _ = model(ob)
            lg = lg.float()
            mask = ob["mask"]
            act = torch.multinomial(torch.softmax(lg, 1), 1, generator=gen).squeeze(1)
            if a.explore > 0:
                rnd = torch.multinomial(mask.float() + 1e-9, 1, generator=gen).squeeze(1)
                act = torch.where(torch.rand(N, generator=gen, device=dev) < a.explore, rnd, act)
            for k in KEYS:
                store[k].append(ob[k].half() if ob[k].dtype == torch.float32 else ob[k])
            acts.append(act)
            lps.append(F.log_softmax(lg, 1).clamp(min=-1e4).half())
            r, d, info = env.step(act.cpu().numpy())
            rews.append(np.asarray(r, dtype=np.float64))
            dones.append(np.asarray(d, dtype=bool))
            if dones[-1].any():
                bits = env.last_bits()
                for e in np.flatnonzero(dones[-1]):
                    clear_acc += (int(bits[e]) >> np.arange(24)) & 1
                    clear_n += 1
                    games += 1
                    wins += int(info["won"][e])
                    win_blinds.append(int(info["blinds"][e]))
        final = dev_obs(env, dev)
        S = T * N
        flat = {k: torch.stack(v).reshape((S,) + tuple(v[0].shape[1:])) for k, v in store.items()}
        acts_f = torch.stack(acts).reshape(S)
        lp_old = torch.stack(lps).reshape(S, -1)
        t1 = time.time()

        # ---------------------------------------------------------------- 2. critic targets
        def v_of(ob_c):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                lg, _ = model(ob_c)
                q = model.q_values()
            lg, q = lg.float(), q.float()
            return (torch.softmax(lg, 1) * q).masked_fill(~ob_c["mask"], 0.0).sum(1)

        def chunked_v(ob_all, n):
            out = []
            for s in range(0, n, 1024):
                out.append(v_of({k: (x[s:s + 1024].float() if x.dtype == torch.float16 else x[s:s + 1024]) for k, x in ob_all.items()}))
            return torch.cat(out).cpu().numpy().astype(np.float64)

        Vall = chunked_v(flat, S).reshape(T, N)
        Vfin = v_of(final).cpu().numpy().astype(np.float64)
        rew_np, done_np = np.stack(rews), np.stack(dones)
        norm.update(rew_np, done_np)
        rn = rew_np / norm.std
        G = np.zeros((T, N))
        for t in reversed(range(T)):
            nv = Vfin if t == T - 1 else Vall[t + 1]
            ng = Vfin if t == T - 1 else G[t + 1]
            G[t] = rn[t] + a.gamma * (~done_np[t]) * ((1 - a.lam) * nv + a.lam * ng)
        G_f = torch.as_tensor(G.reshape(S), dtype=torch.float32, device=dev)
        t2 = time.time()

        # ---------------------------------------------------------------- 3-4. E-step and M-step
        model.train()
        stats = {"q_loss": [], "kl": [], "eta": [], "beta": [], "ent": [], "q_mean": [], "w_ent": []}
        for ep in range(a.epochs):
            perm = torch.randperm(S, device=dev)
            for s in range(0, S, a.batch):
                idx = perm[s:s + a.batch]
                ob = {k: (x[idx].float() if x.dtype == torch.float16 else x[idx]) for k, x in flat.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    lg, _ = model(ob)
                    Qall = model.q_values()
                lg, Qall = lg.float(), Qall.float()
                mask = ob["mask"]
                lp = F.log_softmax(lg, 1)
                loss_q = ((Qall.gather(1, acts_f[idx, None]).squeeze(1) - G_f[idx]) ** 2).mean()
                lpo = lp_old[idx].float()
                Qd = Qall.detach().masked_fill(~mask, 0.0)
                eta = log_eta.exp().clamp(1e-3, 1e3)
                dual_eta = eta * a.eps + eta * torch.logsumexp(lpo + Qd / eta, 1).mean()
                with torch.no_grad():
                    w = torch.softmax(lpo + Qd / eta.detach(), 1)
                kl_old = (lpo.exp() * (lpo - lp)).masked_fill(~mask, 0.0).sum(1).mean()
                beta = log_beta.exp().clamp(1e-3, 1e3)
                loss_pi = -(w * lp).masked_fill(~mask, 0.0).sum(1).mean() + beta.detach() * kl_old
                dual_beta = beta * (a.eps_m - kl_old.detach())
                if it <= a.q_warmup:                         # the Q head learns first; the policy is not moved
                    loss = a.q_coef * loss_q
                else:
                    loss = loss_pi + a.q_coef * loss_q
                opt.zero_grad()
                dual_opt.zero_grad()
                loss.backward()
                (dual_eta + dual_beta).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                dual_opt.step()
                stats["q_loss"].append(float(loss_q)); stats["kl"].append(float(kl_old)); stats["eta"].append(float(eta)); stats["beta"].append(float(beta))
                stats["ent"].append(float(-(lp.exp() * lp).masked_fill(~mask, 0.0).sum(1).mean()))
                stats["w_ent"].append(float(-(w * torch.log(w + 1e-12)).masked_fill(~mask, 0.0).sum(1).mean()))
                stats["q_mean"].append(float(Qall.gather(1, acts_f[idx, None]).mean()))
        t3 = time.time()
        row = {"iter": it, "games": games, "blinds": round(float(np.mean(win_blinds)), 3) if win_blinds else 0.0, "wins": wins,
               "win%": round(100.0 * wins / max(1, games), 3), "sps": int(S / (t1 - t0)), "ret": round(float(rew_np.sum() / max(1, clear_n)), 3), "ret_std": round(float(norm.std), 3),
               "clear_rate": [round(float(x) / max(1, clear_n), 4) for x in clear_acc], "clear_games": clear_n,
               **{k: round(float(np.mean(v)), 5) for k, v in stats.items()},
               "secs": {"data": round(t1 - t0, 1), "targets": round(t2 - t1, 1), "update": round(t3 - t2, 1)}, "min": round((time.time() - t_start) / 60, 1)}
        if it % a.eval_every == 0:
            model.eval()
            g, sm = evaluate(model, a, dev, n=a.eval_games), evaluate(model, a, dev, n=a.eval_games, sampled=True)
            row["eval"], row["eval_sampled"] = g, sm
            if g["blinds"] < base_g["blinds"] - 1.5 or sm["blinds"] < base_s["blinds"] - 1.5:      # the policy got clearly worse: back to the last good one
                model.load_state_dict(good)
                row["reset"] = True
            else:
                good = copy.deepcopy(model.state_dict())
                torch.save({"config": model.config, "state_dict": model.state_dict()}, a.out)
            model.train()
        print(json.dumps(row), flush=True)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
