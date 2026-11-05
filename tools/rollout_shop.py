"""Rollout shop player: v51's decisions outside rounds improved by playing each of its top choices out.

At every blind select, shop and pack decision with more than one legal choice: the policy's top --k actions are each tried in the same
--futures redrawn futures (Env::hide_future before the action: packs, shops, draws and bosses still to come are new draws, shared by every
candidate), the game played on by the policy (rounds by the engine's solver, which sees the redrawn draws) to the end; the action with the
most blinds beaten (+ --win-bonus for a win) is played. The real game's rounds are played honestly (gap_study.Honest) or with foresight.

    python tools/rollout_shop.py --games 200 --k 6 --futures 6 --out traces/gap/rollout_k6f6.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS                  # noqa: E402
from balatro_rl.model import load_model                                # noqa: E402
from gap_study import Honest, arg_of, describe                         # noqa: E402


class Rollouts:
    def __init__(self, net, dev, size, args, stake, threads, win_bonus, max_steps):
        self.net, self.dev, self.win_bonus, self.max_steps = net, dev, win_bonus, max_steps
        self.R = CppVecEnv(size, "RED", stake, seed0=3, threads=threads)
        if getattr(net, "marg", 0):
            self.R.set_marginal(int(net.marg))
        self.R.set_autoplay(True, arg_of(args, "--autoplay-nodes", 400), True)
        self.R.v.restart()
        self.size = size
        self.n_games = 0

    @torch.no_grad()
    def values(self, V, jobs, salts):
        """jobs [(game, action)], salts [salt per job] -> value per job: the blinds beaten at the end (+ win bonus)."""
        out = np.zeros(len(jobs))
        R = self.R
        for c0 in range(0, len(jobs), self.size):
            ch = jobs[c0:c0 + self.size]
            m = len(ch)
            dst = list(range(m))
            R.copy_many(V, [g for g, _ in ch], dst)
            R.v.hide_future(dst, salts[c0:c0 + m])
            fin = np.ones(self.size, bool)
            fin[:m] = False
            act = np.full(self.size, -1, dtype=np.int64)
            act[:m] = [a for _, a in ch]
            val = np.zeros(self.size)
            _, d, info = R.step(act)
            for _ in range(self.max_steps):
                f = d & ~fin
                val[f] = info["blinds"][f] + self.win_bonus * info["won"][f]
                fin |= d
                if fin.all():
                    break
                live = np.flatnonzero(~fin)                              # (the network only for the games still being played)
                o = {k: torch.from_numpy(np.asarray(R.obs[k])[live]).to(self.dev) for k in OBS_KEYS if k in R.obs}
                lg, _ = self.net(o)
                act = np.full(self.size, -1, dtype=np.int64)
                act[live] = lg.argmax(1).cpu().numpy()
                _, d, info = R.step(act)
            out[c0:c0 + m] = val[:m]
            self.n_games += m
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v51.pt")
    p.add_argument("--config", default="checkpoints/ppo_gold_v51.json")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed0", type=int, default=9_500_000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--k", type=int, default=6, help="the policy's top-k legal actions tried")
    p.add_argument("--futures", type=int, default=6)
    p.add_argument("--win-bonus", type=float, default=10.0)
    p.add_argument("--z", type=float, default=1.5, help="override the policy only when the best candidate beats its choice by z paired standard errors")
    p.add_argument("--rollout-envs", type=int, default=4096)
    p.add_argument("--mode", choices=["honest", "foresight"], default="honest")
    p.add_argument("--honest", type=int, default=5)
    p.add_argument("--honest-scratch", type=int, default=1024)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="")
    a = p.parse_args()
    args = json.load(open(a.config))["args"]
    dev = a.device
    net = load_model(a.ckpt, dev).eval()
    n = a.games
    V = CppVecEnv(n, "RED", a.stake, seed0=a.seed0, threads=a.threads)
    if getattr(net, "marg", 0):
        V.set_marginal(int(net.marg))
    V.v.set_solver_tarots(True, arg_of(args, "--solver-targets", 3))
    V.v.set_solver_order(True)
    V.v.set_solver_speed(True, False, 0.0)
    V.v.set_solver_tt(True)
    honest = None
    if a.mode == "foresight":
        V.set_autoplay(True, arg_of(args, "--autoplay-nodes", 400), True)
    else:
        honest = Honest(V, a.honest, 400, a.honest_scratch, "RED", a.stake, a.threads)
    V.v.restart()
    ro = Rollouts(net, dev, a.rollout_envs, args, a.stake, a.threads, a.win_bonus, 6000)
    fin = np.zeros(n, bool)
    res = [None] * n
    log = [[] for _ in range(n)]
    changed = decided = 0
    salt = 0
    t0 = time.time()
    step = 0
    with torch.no_grad():
        while not fin.all() and step < 20000:
            step += 1
            obs = V.obs
            g = np.asarray(obs["g"])
            mask = np.asarray(obs["mask"])
            inround = (g[:, 5] > 0.5) & ~fin
            outside = np.flatnonzero(~fin & ~inround)
            act = np.full(n, -1, dtype=np.int64)
            if len(outside):
                o = {k: torch.from_numpy(np.asarray(obs[k])[outside]).to(dev) for k in OBS_KEYS if k in obs}
                lg, _ = net(o)
                lg = lg.float().cpu().numpy()
                jobs, owners = [], []
                for r, i in enumerate(outside):
                    legal = np.flatnonzero(mask[i])
                    best = int(np.argmax(np.where(mask[i], lg[r], -1e30)))
                    act[i] = best
                    if len(legal) < 2:
                        continue
                    order = legal[np.argsort(-lg[r][legal])][:a.k]
                    owners.append((i, [int(x) for x in order], best))
                    for c in order:
                        for f in range(a.futures):
                            jobs.append((int(i), int(c)))
                if jobs:
                    salts = []
                    for i, cands, _ in owners:
                        base = (salt + i) * 1_000_003
                        for _c in cands:
                            salts.extend(base + 7919 * f + 13 for f in range(a.futures))
                    salt += n
                    vals = ro.values(V, jobs, salts)
                    pos = 0
                    for i, cands, best in owners:
                        rows = []
                        for c in cands:
                            rows.append(vals[pos:pos + a.futures])
                            pos += a.futures
                        rows = np.array(rows)                                   # [k, futures]: the same futures in every row
                        means = rows.mean(1)
                        top = int(np.argmax(means))
                        bi = cands.index(best)
                        diff = rows[top] - rows[bi]
                        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 1e9
                        pick = cands[top] if top != bi and diff.mean() > a.z * max(se, 1e-6) else best
                        decided += 1
                        if pick != best:
                            changed += 1
                            d = V.v.dump(int(i))
                            log[i].append({"a": d["ante"], "b": d["blind_idx"], "s": d["state"][0], "m": d["money"], "policy": describe(best, d),
                                           "rollout": describe(pick, d), "gain": round(float(max(means) - means[cands.index(best)]), 2)})
                        act[i] = pick
            ids = np.flatnonzero(inround)
            if len(ids):
                act[ids] = honest.moves(V, ids, mask, np.asarray(obs["af"]))
            _, done, info = V.step(act)
            for i in np.flatnonzero(done & ~fin):
                res[i] = {"blinds": float(info["blinds"][i]), "won": bool(info["won"][i]), "ante": int(info["ante"][i])}
            fin |= done
            if step % 20 == 0:
                print(json.dumps({"step": step, "done": int(fin.sum()), "decisions": decided, "changed": changed, "rollout_games": ro.n_games,
                                  "s": round(time.time() - t0)}), flush=True)
    bl = np.array([r["blinds"] for r in res if r])
    summary = {"games": n, "mode": a.mode, "k": a.k, "futures": a.futures, "blinds": round(float(bl.mean()), 3), "wins": sum(1 for r in res if r and r["won"]),
               "decisions": decided, "changed": changed, "rollout_games": ro.n_games, "s": round(time.time() - t0)}
    print(json.dumps(summary), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(json.dumps(summary) + "\n")
            for i in range(n):
                f.write(json.dumps({"game": i, "result": res[i], "changes": log[i]}) + "\n")


if __name__ == "__main__":
    main()
