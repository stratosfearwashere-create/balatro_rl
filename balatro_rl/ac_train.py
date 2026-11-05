"""Self-play training WITHOUT look-ahead search: the network makes every decision outside a round (shop, packs, blind select / skip), PPO v30's
network (frozen) plays the rounds, and the network learns from the outcomes - PPO-style actor-critic on those decisions only:
  the outcome of a game z = blinds beaten / 24 + 1 for a win (the same target azc used); advantage = z - V(s) at the decision (Monte Carlo,
  normalised per batch); clipped policy ratio, value regression to z, an entropy bonus.
Starts from a checkpoint (e.g. the distilled big network: balatro_rl/distill.py).

    python -m balatro_rl.ac_train --init checkpoints/big_distill.pt --out checkpoints/big_ac.pt
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from . import azc
from . import env as E
from .azc import Buffer, KEYS, obs_batch
from .distill import evaluate
from .engine_env import CppVecEnv
from .model import load_model, save_model


def template_of(v):
    return {k: (tuple(np.asarray(v.obs[k]).shape[1:]), np.asarray(v.obs[k]).dtype) for k in KEYS}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", default="checkpoints/big_distill.pt")
    p.add_argument("--round-net", default="checkpoints/ppo_gold_v30.pt")
    p.add_argument("--out", default="checkpoints/big_ac.pt")
    p.add_argument("--games", type=int, default=1024, help="games in parallel")
    p.add_argument("--rows", type=int, default=32768, help="finished decisions per update")
    p.add_argument("--iters", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf", type=float, default=1.0)
    p.add_argument("--ent", type=float, default=0.003)
    p.add_argument("--win", type=float, default=1.0, help="the outcome's bonus for a win (z = blinds / 24 + this)")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-games", type=int, default=400)
    p.add_argument("--eval-seed0", type=int, default=9_000_000)
    a = p.parse_args()
    dev = "cuda"
    net = load_model(a.init, dev)
    rnet = load_model(a.round_net, dev).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    log = open(a.out.replace(".pt", "_log.jsonl"), "a")
    net.eval()
    b0, w0 = evaluate(net, rnet, a.eval_games, a.eval_seed0)
    print(json.dumps({"iter": 0, "eval": {"blinds": b0, "wins": w0}, "params": sum(t.numel() for t in net.parameters())}), flush=True)
    log.write(json.dumps({"iter": 0, "eval": {"blinds": b0, "wins": w0}}) + "\n"); log.flush()

    v = CppVecEnv(a.games, "RED", "GOLD", seed0=321_000_000, threads=0)
    v.set_exact(True)
    v.v.restart()
    v.set_auto_round(False, 1.5, 0)
    tmpl = template_of(v)
    buf = Buffer(cap=10 ** 9)
    episode = np.zeros(a.games, dtype=np.int64)
    t0 = time.time()
    results = []
    for it in range(1, a.iters + 1):
        ti = time.time()
        # ---- self-play until --rows decisions have outcomes
        while len(buf.ready()) < a.rows:
            mask = np.asarray(v.obs["mask"])
            act = np.full(a.games, -1, dtype=np.int64)
            inr = mask[:, :E.A_SELECT].any(1)
            r, s = np.flatnonzero(inr), np.flatnonzero(~inr)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if len(r):
                    lg, _ = rnet(obs_batch(v, list(r), dev))
                    act[r] = np.where(mask[r], lg.float().cpu().numpy(), -np.inf).argmax(1)
                if len(s):
                    lg, val = net(obs_batch(v, list(s), dev))
            if len(s):
                lg = lg.float().masked_fill(~torch.from_numpy(mask[s]).to(dev), -1e9)
                logp = torch.log_softmax(lg, -1)
                pick = torch.multinomial(logp.exp(), 1).squeeze(-1)
                lp_pick = logp.gather(1, pick[:, None]).squeeze(1).cpu().numpy()
                pick = pick.cpu().numpy()
                val = val.float().cpu().numpy()
                for j, i in enumerate(s):
                    if mask[i].sum() <= 1:
                        act[i] = int(np.flatnonzero(mask[i])[0])
                        continue
                    row = buf.add(v, i, None, (int(i), int(episode[i])))
                    row["a"] = int(np.searchsorted(row["idx"], pick[j]))
                    row["logp"] = float(lp_pick[j])
                    row["v"] = float(val[j])
                    act[i] = int(pick[j])
            _, d, info = v.step(act)
            for i in np.flatnonzero(d):
                z = info["blinds"][i] / 24.0 + (a.win if info["won"][i] else 0.0)
                buf.finish((int(i), int(episode[i])), z)
                results.append((int(info["blinds"][i]), bool(info["won"][i])))
                episode[i] += 1
        rows = buf.ready()
        buf.rows = [x for x in buf.rows if x["z"] is None]                  # keep the unfinished games' decisions for later
        t_play = time.time() - ti
        # ---- PPO update on the finished decisions
        z = np.array([x["z"] for x in rows])
        adv = z - np.array([x["v"] for x in rows])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        net.train()
        stats = []
        for _ in range(a.epochs):
            order = np.random.permutation(len(rows))
            for c0 in range(0, len(rows), a.batch):
                ix = order[c0:c0 + a.batch]
                mb = [rows[k] for k in ix]
                ob = Buffer.batch(mb, tmpl, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, val = net(ob)
                lg = lg.float().masked_fill(~ob["mask"], -1e9)
                logp_all = torch.log_softmax(lg, -1)
                acts = torch.tensor([int(x["idx"][x["a"]]) for x in mb], device=dev)
                lp = logp_all.gather(1, acts[:, None]).squeeze(1)
                old = torch.tensor([x["logp"] for x in mb], device=dev)
                A = torch.tensor(adv[ix], device=dev, dtype=torch.float32)
                ratio = torch.exp(lp - old)
                pl = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean()
                zt = torch.tensor(z[ix], device=dev, dtype=torch.float32)
                vl = ((val.float() - zt) ** 2).mean()
                pr = logp_all.exp()
                ent = -(pr * logp_all.masked_fill(~ob["mask"], 0)).sum(-1).mean()
                loss = pl + a.vf * vl - a.ent * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats.append((float(pl), float(vl), float(ent), float(((ratio - 1).abs() > a.clip).float().mean())))
        net.eval()
        st = np.mean(stats, 0)
        recent = results[-2000:]
        row = {"iter": it, "games": len(results), "train_blinds": round(float(np.mean([b for b, _ in recent])), 3),
               "train_wins_per_1000": round(1000 * float(np.mean([w for _, w in recent])), 2), "decisions": len(rows),
               "policy_loss": round(st[0], 4), "value_loss": round(st[1], 5), "entropy": round(st[2], 3), "clipped": round(st[3], 3),
               "play_s": round(t_play, 1), "min": round((time.time() - t0) / 60, 1)}
        if a.eval_every and it % a.eval_every == 0:
            b_, w_ = evaluate(net, rnet, a.eval_games, a.eval_seed0)
            row["eval"] = {"blinds": b_, "wins": w_}
            save_model(net, a.out, {"mode": "ac", "iter": it})
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n"); log.flush()


if __name__ == "__main__":
    main()
