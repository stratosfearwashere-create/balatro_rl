"""Warm-start a bigger network by distillation: a student learns a teacher network's move probabilities and value on the states the teacher
actually meets, so it starts as good as the teacher (then search / RL training can use its larger capacity).

Games run in parallel on the exact engine (Gold); PPO v30's network plays the rounds; the teacher (azc's network) chooses every move outside a
round, sampled from its own probabilities (so the states are varied, not one greedy line). At each such state the student is trained on:
cross-entropy to the teacher's legal-move distribution + --vf x squared error to the teacher's value. Evaluation: the student alone (greedy
shops, v30's rounds) on held-out Gold seeds, against the teacher on the same seeds.

    python -m balatro_rl.distill --out checkpoints/big_distill.pt --steps 20000
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import azc
from . import env as E
from .engine_env import CppVecEnv
from .model import ActorCritic, load_model, save_model


@torch.no_grad()
def evaluate(shop_net, round_net, n, seed0):
    v = CppVecEnv(n, "RED", "GOLD", seed0=seed0, threads=0)
    v.set_exact(True)
    v.v.restart()
    v.set_auto_round(False, 1.5, 0)
    done = np.zeros(n, bool)
    bl = np.zeros(n)
    wn = np.zeros(n, bool)
    for _ in range(9000):
        mask = np.asarray(v.obs["mask"])
        live = np.flatnonzero(~done)
        act = np.full(n, -1, dtype=np.int64)
        inr = mask[:, :E.A_SELECT].any(1)
        r, s = live[inr[live]], live[~inr[live]]
        for net, idx in ((round_net, r), (shop_net, s)):
            if len(idx):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = net(azc.obs_batch(v, list(idx), "cuda"))
                act[idx] = np.where(mask[idx], lg.float().cpu().numpy(), -np.inf).argmax(1)
        _, d, info = v.step(act)
        f = d & ~done
        bl[f], wn[f] = info["blinds"][f], info["won"][f]
        done |= d
        if done.all():
            break
    return float(bl.mean()), int(wn.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default="checkpoints/azc_big.pt")
    p.add_argument("--round-net", default="checkpoints/ppo_gold_v30.pt")
    p.add_argument("--out", default="checkpoints/big_distill.pt")
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--tok", type=int, default=192)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--attn", type=int, default=6)
    p.add_argument("--aw", type=int, default=256)
    p.add_argument("--vw", type=int, default=256)
    p.add_argument("--res", type=int, default=3)
    p.add_argument("--games", type=int, default=1024, help="games running in parallel")
    p.add_argument("--steps", type=int, default=20000, help="gradient steps")
    p.add_argument("--batch", type=int, default=512, help="states per gradient step")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=1.0, help="the teacher's sampling temperature when stepping the games")
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--eval-games", type=int, default=400)
    p.add_argument("--eval-seed0", type=int, default=9_000_000)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    dev = "cuda"
    teacher = load_model(a.teacher, dev).eval()
    rnet = load_model(a.round_net, dev).eval()
    cfg = dict(teacher.config)
    cfg.update(d=a.d, tok=a.tok, heads=a.heads, attn=a.attn, aw=a.aw, vw=a.vw, res=a.res, res_expand=2)
    student = load_model(a.out, dev) if a.resume else ActorCritic(**cfg).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    log = open(a.out.replace(".pt", "_log.jsonl"), "a")
    n_params = sum(t.numel() for t in student.parameters())
    tb, tw = evaluate(teacher, rnet, a.eval_games, a.eval_seed0)
    print(json.dumps({"student_params": n_params, "teacher_params": sum(t.numel() for t in teacher.parameters()), "teacher_eval": {"blinds": tb, "wins": tw}}), flush=True)
    log.write(json.dumps({"step": 0, "teacher_eval": {"blinds": tb, "wins": tw}, "student_params": n_params}) + "\n"); log.flush()

    v = CppVecEnv(a.games, "RED", "GOLD", seed0=123_000_000, threads=0)
    v.set_exact(True)
    v.v.restart()
    v.set_auto_round(False, 1.5, 0)
    rng = np.random.default_rng(0)
    pool = []                                                       # states waiting for a gradient step: (obs batch, teacher log-probs, teacher value, mask)
    t0, step, seen = time.time(), 0, 0
    pl_hist, vl_hist, agree_hist = [], [], []
    student.train()
    while step < a.steps:
        mask = np.asarray(v.obs["mask"])
        act = np.full(a.games, -1, dtype=np.int64)
        inr = mask[:, :E.A_SELECT].any(1)
        r, s = np.flatnonzero(inr), np.flatnonzero(~inr)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if len(r):
                lg, _ = rnet(azc.obs_batch(v, list(r), dev))
                act[r] = np.where(mask[r], lg.float().cpu().numpy(), -np.inf).argmax(1)
            if len(s):
                ob = azc.obs_batch(v, list(s), dev)
                tl, tv = teacher(ob)
        if len(s):
            tl = tl.float()
            m = torch.from_numpy(mask[s]).to(dev)
            logp_t = torch.log_softmax(tl.masked_fill(~m, -1e9) / a.temp, -1)
            pick = torch.multinomial(logp_t.exp(), 1).squeeze(-1).cpu().numpy()
            act[s] = pick
            # train the student on these states (one gradient step per --batch states gathered)
            pool.append((ob, torch.log_softmax(tl.masked_fill(~m, -1e9), -1), tv.float(), m, tl.masked_fill(~m, -1e9).argmax(-1)))
            seen += len(s)
            if sum(len(x[2]) for x in pool) >= a.batch:
                obs = {k: torch.cat([x[0][k] for x in pool]) for k in pool[0][0]}
                lp_t = torch.cat([x[1] for x in pool]); v_t = torch.cat([x[2] for x in pool]); mm = torch.cat([x[3] for x in pool])
                top_t = torch.cat([x[4] for x in pool])
                pool = []
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sl, sv = student(obs)
                sl = sl.float().masked_fill(~mm, -1e9)
                lp_s = torch.log_softmax(sl, -1)
                pt = lp_t.exp()
                ploss = -(pt * lp_s.masked_fill(~mm, 0)).sum(-1).mean()
                vloss = F.mse_loss(sv.float(), v_t)
                loss = ploss + a.vf * vloss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                pl_hist.append(float(ploss)); vl_hist.append(float(vloss)); agree_hist.append(float((sl.argmax(-1) == top_t).float().mean()))
                if step % 200 == 0:
                    row = {"step": step, "policy_ce": round(np.mean(pl_hist[-200:]), 4), "value_mse": round(np.mean(vl_hist[-200:]), 5),
                           "top_move_agreement": round(np.mean(agree_hist[-200:]), 4), "states": seen, "lr": round(sched.get_last_lr()[0], 6),
                           "min": round((time.time() - t0) / 60, 1)}
                    if step % a.eval_every == 0 or step == a.steps:
                        student.eval()
                        b_, w_ = evaluate(student, rnet, a.eval_games, a.eval_seed0)
                        student.train()
                        row["student_eval"] = {"blinds": b_, "wins": w_, "teacher": tb}
                        save_model(student, a.out, {"mode": "distill", "step": step, "teacher": a.teacher})
                    print(json.dumps(row), flush=True)
                    log.write(json.dumps(row) + "\n"); log.flush()
        v.step(act)
    save_model(student, a.out, {"mode": "distill", "step": step, "teacher": a.teacher})


if __name__ == "__main__":
    main()
