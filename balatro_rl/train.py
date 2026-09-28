"""Training: behaviour cloning (DAgger) from the heuristic, then PPO.

    # 1) imitate the rule-based player (fast warm start)
    python -m balatro_rl.train bc  --iters 150 --envs 8  --out checkpoints/bc.pt
    # 2) reinforcement learning on top
    python -m balatro_rl.train ppo --init checkpoints/bc.pt --iters 3000 --envs 16 --out checkpoints/ppo.pt
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import ActorCritic, batch_obs, stack_obs, save_model, load_model, OBS_KEYS
from .vec_env import VecEnv


def compact(o: dict) -> dict:
    """Shrink an observation for storage (action features to float16)."""
    o = dict(o)
    o["af"] = o["af"].astype(np.float16)
    return o


class EpisodeStats:
    def __init__(self, n=200):
        self.blinds = collections.deque(maxlen=n)
        self.antes = collections.deque(maxlen=n)
        self.wins = collections.deque(maxlen=n)
        self.returns = collections.deque(maxlen=n)

    def add(self, info):
        if info.get("episode_end"):
            self.blinds.append(info["blinds"])
            self.antes.append(info["ante"])
            self.wins.append(float(info["won"]))

    def summary(self):
        if not self.blinds:
            return {}
        out = {"episodes": len(self.blinds), "blinds": round(float(np.mean(self.blinds)), 2),
               "ante": round(float(np.mean(self.antes)), 2), "win%": round(100 * float(np.mean(self.wins)), 2)}
        if self.returns:
            out["ret"] = round(float(np.mean(self.returns)), 3)
        return out


def game_kw(a) -> dict:
    """Environment options shared by every mode: reward shape and the cheap-experiment game options."""
    pool = None
    if a.joker_pool_file:
        with open(a.joker_pool_file) as f:
            pool = json.load(f)
    return {"ante_weight": a.ante_weight, "win_bonus": a.win_bonus, "win_ante": a.win_ante, "joker_pool": pool}


def log(path, row):
    print(json.dumps(row), flush=True)
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- behaviour cloning
def train_bc(a):
    torch.manual_seed(a.seed)
    dev = a.device
    model = load_model(a.init, dev) if a.init else ActorCritic().to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    venv = VecEnv(a.envs, a.deck, a.stake, seed0=a.seed * 1_000_000, expert=True, reward_kw=game_kw(a))
    obs, labels = venv.current()
    buf = collections.deque(maxlen=a.buffer)
    stats = EpisodeStats()
    rng = random.Random(a.seed)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        beta = max(0.0, 1.0 - it / max(1, a.iters * a.beta_frac))   # prob. of following the expert
        model.eval()
        for _ in range(a.steps):
            with torch.no_grad():
                logits, _ = model(batch_obs(obs, dev))
                student = logits.argmax(-1).cpu().numpy()
            for o, l in zip(obs, labels):
                buf.append((compact(o), int(l)))
            acts = [l if rng.random() < beta else s for l, s in zip(labels, student)]
            obs, _, _, infos, labels = venv.step(acts)
            for inf in infos:
                stats.add(inf)
        model.train()
        losses, accs = [], []
        data = list(buf)
        n_up = a.updates or max(1, a.epochs * len(data) // a.batch)
        for _ in range(n_up):
            if True:
                mb = rng.sample(data, min(a.batch, len(data)))
                ob = batch_obs([x[0] for x in mb], dev)
                y = torch.tensor([x[1] for x in mb], device=dev)
                logits, _ = model(ob)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
                accs.append((logits.argmax(-1) == y).float().mean().item())
        row = {"iter": it, "beta": round(beta, 2), "loss": round(float(np.mean(losses)), 4),
               "acc": round(float(np.mean(accs)), 3), "min": round((time.time() - t0) / 60, 1),
               "phase": a.phase, "env_steps": it * a.steps * a.envs}
        row.update(stats.summary())
        log(a.log, row)
        if it % a.save_every == 0 or it == a.iters:
            save_model(model, a.out, {"mode": "bc", "iter": it})
    venv.close()


# --------------------------------------------------------------------------- PPO
def train_ppo(a):
    torch.manual_seed(a.seed)
    dev = a.device
    model = load_model(a.init, dev) if a.init else ActorCritic().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    use_amp = a.amp and str(dev).startswith("cuda")

    def fwd(ob):
        # --amp: the network runs in bfloat16; probabilities and losses stay in float32
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, v = model(ob)
        return logits.float(), v.float()
    venv = VecEnv(a.envs, a.deck, a.stake, seed0=a.seed * 1_000_000 + 7,
                  reward_kw=game_kw(a),
                  strategic_kw={"tactical": a.tactical or a.init, "margin": a.margin} if a.strategic else None)
    obs, _ = venv.current()
    stats = EpisodeStats()
    ep_ret = np.zeros(a.envs)
    t0 = time.time()
    best = -1.0
    for it in range(1, a.iters + 1):
        model.eval()
        T, N = a.steps, a.envs
        # --pipeline: two groups of envs take turns, so one group simulates while the network
        # picks actions for the other. Each env still sees the same policy; only timing changes.
        groups = [list(range(N // 2)), list(range(N // 2, N))] if a.pipeline else [list(range(N))]
        gobs = [[obs[i] for i in g] for g in groups]
        gbuf = [{"obs": [], "act": [], "lp": [], "val": [], "rew": [], "done": []} for _ in groups]

        def act_and_send(k):
            with torch.no_grad():
                logits, v = fwd(batch_obs(gobs[k], dev))
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                lp = dist.log_prob(act)
            act = act.cpu().numpy()
            venv.step_async(act, groups[k])
            b = gbuf[k]
            b["obs"].append([compact(o) for o in gobs[k]])
            b["act"].append(act)
            b["lp"].append(lp.cpu().numpy())
            b["val"].append(v.cpu().numpy())

        def wait(k):
            gobs[k], r, d, infos, _ = venv.step_wait(groups[k])
            for e, re_, de in zip(groups[k], r, d):
                ep_ret[e] += re_
                if de:
                    stats.returns.append(float(ep_ret[e]))
                    ep_ret[e] = 0.0
            gbuf[k]["rew"].append(r)
            gbuf[k]["done"].append(d.astype(np.float32))
            for inf in infos:
                stats.add(inf)

        act_and_send(0)
        for t in range(T):
            if len(groups) == 2:
                act_and_send(1)
            wait(0)
            if t + 1 < T:
                act_and_send(0)
            if len(groups) == 2:
                wait(1)
        # stitch the groups back together in env order: lists of T rows, each covering all N envs
        store_obs = [sum((b["obs"][t] for b in gbuf), []) for t in range(T)]
        acts, logps, vals, rews, dones = (
            [np.concatenate([b[key][t] for b in gbuf]) for t in range(T)]
            for key in ("act", "lp", "val", "rew", "done"))
        obs = sum(gobs, [])
        with torch.no_grad():
            _, last_v = fwd(batch_obs(obs, dev))
            last_v = last_v.cpu().numpy()
        vals_a = np.array(vals)
        adv = np.zeros((T, N), dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else vals_a[t + 1]
            nonterm = 1.0 - dones[t]
            delta = rews[t] + a.gamma * nv * nonterm - vals_a[t]
            gae = delta + a.gamma * a.lam * nonterm * gae
            adv[t] = gae
        ret = adv + vals_a

        # the whole rollout goes to the GPU once; minibatches are slices of it
        big = stack_obs([o for row in store_obs for o in row], dev)
        flat_act = torch.tensor(np.concatenate(acts), device=dev)
        flat_lp = torch.tensor(np.concatenate(logps), device=dev)
        flat_adv = torch.tensor(adv.reshape(-1), device=dev)
        flat_ret = torch.tensor(ret.reshape(-1), device=dev)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        if use_amp:
            # bfloat16 rounds differently at play-time batch sizes than in update minibatches, which
            # would add noise to the PPO ratio; recompute the old log-probs the way the update sees them
            with torch.no_grad():
                chunks = []
                for k in range(0, T * N, a.batch):
                    sl = slice(k, k + a.batch)
                    lg, _ = fwd({key: t[sl] for key, t in big.items()})
                    chunks.append(torch.distributions.Categorical(logits=lg).log_prob(flat_act[sl]))
                flat_lp = torch.cat(chunks)

        model.train()
        idx = np.arange(T * N)
        pl, vl, ents, kls = [], [], [], []
        for _ in range(a.epochs):
            np.random.shuffle(idx)
            for k in range(0, len(idx), a.batch):
                mb = torch.from_numpy(idx[k:k + a.batch]).to(dev)
                logits, v = fwd({key: t[mb] for key, t in big.items()})
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(flat_act[mb])
                ratio = torch.exp(lp - flat_lp[mb])
                s1 = ratio * flat_adv[mb]
                s2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * flat_adv[mb]
                ploss = -torch.min(s1, s2).mean()
                vloss = F.mse_loss(v, flat_ret[mb])
                ent = dist.entropy().mean()
                loss = ploss + a.vf * vloss - a.ent * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                # kept on the GPU and fetched once per iteration (each .item() waits for the GPU)
                pl.append(ploss.detach()); vl.append(vloss.detach()); ents.append(ent.detach())
                kls.append((flat_lp[mb] - lp).mean().detach())
        pl, vl, ents, kls = (torch.stack(x).double().cpu().numpy() for x in (pl, vl, ents, kls))
        row = {"iter": it, "pi": round(float(np.mean(pl)), 4), "v": round(float(np.mean(vl)), 3),
               "ent": round(float(np.mean(ents)), 3), "kl": round(float(np.mean(kls)), 4),
               "min": round((time.time() - t0) / 60, 1), "phase": a.phase, "env_steps": it * T * N}
        row.update(stats.summary())
        log(a.log, row)
        if it % a.save_every == 0 or it == a.iters:
            save_model(model, a.out, {"mode": "ppo", "iter": it})
            s = stats.summary().get("blinds", -1)
            if s > best:
                best = s
                save_model(model, a.out.replace(".pt", "_best.pt"), {"mode": "ppo", "iter": it, "blinds": s})
    venv.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["bc", "ppo"])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=64, help="env steps per env per iteration")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--buffer", type=int, default=6000, help="BC replay size")
    p.add_argument("--updates", type=int, default=6, help="BC gradient steps per iteration (0 = full epochs)")
    p.add_argument("--beta-frac", type=float, default=0.5, help="BC: fraction of iters to anneal expert mixing")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.002)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--init", default=None, help="checkpoint to start from")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--log", default=None)
    p.add_argument("--pipeline", action="store_true",
                   help="PPO: overlap simulation and GPU work with two groups of envs "
                        "(use about twice your core count for --envs)")
    p.add_argument("--ante-weight", type=float, default=1.0,
                   help="PPO reward: an ante-8 blind is worth this many times an ante-1 blind, rising linearly "
                        "(the 24 blinds still add up to 24)")
    p.add_argument("--win-bonus", type=float, default=10.0, help="PPO reward for winning the run")
    p.add_argument("--strategic", action="store_true",
                   help="PPO only makes strategic decisions; a tactical network plays the cards (see strategic.py)")
    p.add_argument("--tactical", default=None, help="--strategic: checkpoint that plays the cards (default: --init)")
    p.add_argument("--margin", type=float, default=0.2,
                   help="--strategic: bonus x how comfortably a blind was cleared (score/target - 1, capped at 1)")
    p.add_argument("--win-ante", type=int, default=8,
                   help="end runs as won after this ante's boss (cheap experiments); 8 is the real game")
    p.add_argument("--joker-pool-file", default=None,
                   help="JSON list of joker keys that shops and packs may offer (cheap experiments)")
    p.add_argument("--phase", default=None, help="label written to every log row (e.g. a curriculum stage)")
    p.add_argument("--amp", action="store_true",
                   help="PPO, experimental: run the network in bfloat16 mixed precision (~12%% faster on an RTX 4080, "
                        "but in a short test updates moved the policy ~2x further and entropy rose faster)")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.epochs is None:
        a.epochs = 2 if a.mode == "bc" else 4
    if a.lr is None:
        a.lr = 1e-3 if a.mode == "bc" else (1e-4 if a.init else 3e-4)
    if a.phase is None:
        a.phase = a.mode
    if a.pipeline and a.envs < 2:
        p.error("--pipeline needs --envs 2 or more")
    if a.strategic and not (a.tactical or a.init):
        p.error("--strategic needs --tactical or --init")
    if a.log is None:
        a.log = a.out.replace(".pt", "_log.jsonl")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    (train_bc if a.mode == "bc" else train_ppo)(a)


if __name__ == "__main__":
    main()
