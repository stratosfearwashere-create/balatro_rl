"""Shop policy iteration for the two-layer agent (the exact solver plays every round; the PPO network decides outside rounds): shopq.py's loop, with solver rounds.

    1. DATA     greedy games with the network; at some shop states every shop candidate (each purchase, sale, reroll, use, leave) and "-1" (the network's own choice) is played on
                copies with the same redrawn futures (common random numbers), then the run goes on with the network + solver to its end. Label: Gain(a) = V(a) - V(own choice),
                V = blinds beaten + win bonus.
    2. FIT      the network's Q head (ActorCritic.add_qhead: a head on the (state, action) rows) regresses every candidate's Gain (shopq.train_q); the policy is untouched.
    3. CHECK    the same Gold seeds played greedy and with QPolicy (in a shop, the best predicted Gain when it clears --thr, else the network's choice).
    Step 3 of the plan (distilling Q's choices into the policy) follows only if 3 shows a gain.

    python -m balatro_rl.qiter --ckpt checkpoints/ppo_gold_v51.pt --config checkpoints/ppo_gold_v51.json --states 600 --seeds 8 --out checkpoints/qiter1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import shopq as SQ
from . import shopvalue as SV


def arg_of(args, k, d):
    return type(d)(args[args.index(k) + 1]) if k in args else d


class Setup:
    """The run's game settings (from its args): reward shape, solver rounds, marginal inputs."""

    def __init__(self, args, marg, threads):
        self.args, self.marg, self.threads = args, marg, threads
        self.kw = {"ante_weight": arg_of(args, "--ante-weight", 1.0), "win_bonus": arg_of(args, "--win-bonus", 10.0), "win_ante": 8, "joker_pool": None,
                   "shape_chips": 0.0, "shape_phi": 0.0, "reward_config": None, "shape_gamma": arg_of(args, "--gamma", 0.995)}
        self.nodes = arg_of(args, "--autoplay-nodes", 400)

    def env(self, n, seed0):
        from .engine_env import CppVecEnv
        v = CppVecEnv(n, "RED", "GOLD", seed0=seed0, threads=self.threads, reward_kw=self.kw)
        if self.marg:
            v.set_marginal(int(self.marg))
        v.set_autoplay(True, self.nodes, True)
        a = self.args
        if "--solver-tarots" in a:
            v.v.set_solver_tarots(True, 3)                  # (engine globals)
        if "--solver-smart" in a:
            v.v.set_solver_order(True)
        if "--solver-lite" in a:
            v.v.set_solver_speed(True, False, 0.0)
        if "--solver-tt" in a:
            v.v.set_solver_tt(True)
        v.v.restart()
        return v


class SolverRollouts(SV.PairedRollouts):
    """PairedRollouts with the rounds played by the exact solver (the environment's autoplay) instead of the scripted player."""

    def __init__(self, policy, setup, cap=4096, max_steps=500, win_bonus=10.0):
        self.policy, self.cap, self.max_steps, self.win_bonus = policy, cap, max_steps, win_bonus
        self.mode, self.pace = 2, 1.5                       # (unused: the round player stays the solver)
        self.ro = setup.env(cap, 1)
        self.ro.set_auto_round = lambda *x, **y: None       # (the base class switches on the scripted player: keep the solver)
        self.ro.set_modes = lambda *x, **y: None
        self.horizon = 0

    def values_many(self, venv, states, n_seeds, salt0=1_000_000):
        """As PairedRollouts.values_many; with self.horizon > 0 each rollout stops once it has beaten that many more blinds (or the game ends), and its value is
        the blinds beaten in that window (+ the win bonus): the moves are compared over the next two antes, without the noise of the rest of the run."""
        if self.horizon <= 0:
            return super().values_many(venv, states, n_seeds, salt0)
        rows = [(si, c) for si, (e, cands) in enumerate(states) for c in cands]
        R = len(rows) * n_seeds
        ro = self.ro
        src = [states[si][0] for si, c in rows for _ in range(n_seeds)]
        ro.copy_many(venv, src, list(range(R)))
        ro.reseed_rng(list(range(R)), [salt0 + r for _ in rows for r in range(n_seeds)])
        start = np.zeros(ro.n)
        g = np.asarray(ro.obs["g"])
        inf = ro.v.cell_info(list(range(R)))
        start[:R] = [3 * (x[0] - 1) + x[1] for x in inf]                 # (the blind index the game is at: its blinds so far, skips counted as passed)
        act = np.full(ro.n, -1, dtype=np.int64)
        act[:R] = np.repeat(np.asarray([c for _, c in rows], dtype=np.int64), n_seeds)
        done = np.zeros(ro.n, dtype=bool)
        done[R:] = True
        gained, won = np.zeros(ro.n), np.zeros(ro.n, dtype=bool)
        for step in range(self.max_steps):
            if done.all():
                break
            if step > 0:
                live = np.flatnonzero(~done)
                act = np.full(ro.n, -1, dtype=np.int64)
                act[live] = self.policy(ro, live)
            _, d, info = ro.step(act)
            fin = d & ~done
            gained[fin] = info["blinds"][fin] - start[fin]
            won[fin] = info["won"][fin]
            done |= d
            live = np.flatnonzero(~done)
            if len(live):
                pos = np.array([3 * (x[0] - 1) + x[1] for x in ro.v.cell_info([int(i) for i in live])], dtype=np.float64)
                hit = live[pos - start[live] >= self.horizon]
                gained[hit] = self.horizon
                done[hit] = True
        out, k = [], 0
        for si, (e, cands) in enumerate(states):
            C = len(cands)
            b = gained[k * n_seeds:(k + C) * n_seeds].reshape(C, n_seeds)
            w = won[k * n_seeds:(k + C) * n_seeds].reshape(C, n_seeds)
            k += C
            val = b + self.win_bonus * w
            out.append({"v": val.mean(1), "se": val.std(1, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros(C), "blinds": b, "won": w,
                        "reach": np.zeros((C, 9))})
        return out


@torch.no_grad()
def gen(setup, policy, n_states, n_seeds, out_path, envs=96, p_label=0.1, per_call=24, seed=1, log=print, horizon=0):
    rng = np.random.default_rng(seed)
    v = setup.env(envs, seed * 1_000_000 + 17)
    pr = SolverRollouts(policy, setup, cap=4096)
    pr.horizon = horizon
    store = {k: [] for k in SQ.SAVE_KEYS}
    cands_all, v_all, se_all = [], [], []
    from .env import A_LEAVE
    t0, salt = time.time(), 1_000_000 * seed
    while len(cands_all) < n_states:
        mask = np.asarray(v.obs["mask"])
        shop = np.flatnonzero(mask[:, A_LEAVE])
        pick = shop[rng.random(len(shop)) < p_label][:per_call]
        if len(pick):
            states = [(int(e), ([-1] + SV.shop_candidates(mask[e]))[:SQ.CMAX]) for e in pick]
            while states and sum(len(c) for _, c in states) * n_seeds > pr.cap:
                states.pop()
            if states:
                res = pr.values_many(v, states, n_seeds, salt0=salt)
                salt += 1000
                for (e, cands), r in zip(states, res):
                    for k in SQ.SAVE_KEYS:
                        x = np.array(v.obs[k][e])
                        store[k].append(x.astype(np.float16) if k == "af" else x.astype(np.int8) if k == "subs" else x)
                    c = np.full(SQ.CMAX, -2, dtype=np.int16); c[:len(cands)] = cands
                    vv = np.zeros(SQ.CMAX, dtype=np.float32); vv[:len(cands)] = r["v"]
                    ss = np.zeros(SQ.CMAX, dtype=np.float32); ss[:len(cands)] = r["se"]
                    cands_all.append(c); v_all.append(vv); se_all.append(ss)
                log(f"  {len(cands_all)}/{n_states} labelled shop states ({time.time() - t0:.0f}s)")
        v.step(policy(v, np.arange(envs)))
    np.savez_compressed(out_path, **{k: np.stack(x) for k, x in store.items()}, cands=np.stack(cands_all), v=np.stack(v_all), se=np.stack(se_all))
    v.close()
    return len(cands_all)


@torch.no_grad()
def play(setup, policy, n, seed0, max_steps=900):
    v = setup.env(n, seed0)
    done = np.zeros(n, dtype=bool)
    blinds, won = np.zeros(n), np.zeros(n, dtype=bool)
    for _ in range(max_steps):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        a = np.full(n, -1, dtype=np.int64)
        a[live] = policy(v, live)
        _, d, info = v.step(a)
        first = d & ~done
        blinds[first], won[first] = info["blinds"][first], info["won"][first]
        done |= d
    v.close()
    return blinds, won


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v51.pt")
    p.add_argument("--config", default="checkpoints/ppo_gold_v51.json")
    p.add_argument("--out", default="checkpoints/qiter1")
    p.add_argument("--states", type=int, default=600)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--thr", default="0.3,1.0", help="QPolicy thresholds to check (comma list)")
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--eval-seed0", type=int, default=9_200_000)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-gen", action="store_true", help="reuse <out>_data.npz")
    p.add_argument("--horizon", type=int, default=0, help="> 0: a rollout stops after this many more blinds; the label is the blinds gained in that window (0: to the end of the run)")
    p.add_argument("--all-params", action="store_true", help="the Q fit also adjusts the network's shared layers (default: the Q head only)")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    from .model import load_model
    args = json.load(open(a.config))["args"]
    model = load_model(a.ckpt, a.device).eval()
    setup = Setup(args, getattr(model, "marg", 0), a.threads)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    data = a.out + "_data.npz"
    log = lambda s: print(s, flush=True)
    if not a.skip_gen:
        t = time.time()
        n = gen(setup, SV.ManagerPolicy(model, a.device), a.states, a.seeds, data, log=log, horizon=a.horizon)
        log(json.dumps({"data": data, "states": n, "secs": round(time.time() - t)}))
    d = SQ.load_shards([data])
    gains = []
    for vv, cc in zip(d["v"], d["cands"]):
        k = int((cc >= -1).sum())
        own = vv[0]
        gains.extend((vv[1:k] - own).tolist())
    g = np.asarray(gains)
    log(json.dumps({"label_gains": {"n": len(g), "mean": round(float(g.mean()), 3), "share_above_0.5": round(float((g > 0.5).mean()), 3),
                                    "share_below_-0.5": round(float((g < -0.5).mean()), 3), "best_per_state_mean": None}}))
    if not hasattr(model, "qhead"):
        model.add_qhead()
    SQ.train_q(model, d, a.device, epochs=a.epochs, log=log, all_params=a.all_params)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, a.out + ".pt")
    b0, w0 = play(setup, SV.ManagerPolicy(model, a.device), a.eval_games, a.eval_seed0)
    log(json.dumps({"check": "greedy", "games": a.eval_games, "blinds": round(float(b0.mean()), 3), "se": round(float(b0.std() / np.sqrt(len(b0))), 3), "wins": int(w0.sum())}))
    for thr in [float(x) for x in a.thr.split(",")]:
        qp = SV.QPolicy(model, a.device, thr=thr)
        b1, w1 = play(setup, qp, a.eval_games, a.eval_seed0)
        log(json.dumps({"check": f"QPolicy thr {thr}", "games": a.eval_games, "blinds": round(float(b1.mean()), 3), "wins": int(w1.sum()),
                        "paired_diff": round(float((b1 - b0).mean()), 3), "paired_se": round(float((b1 - b0).std() / np.sqrt(len(b0))), 3),
                        "q_overrides": f"{qp.used}/{qp.calls}"}))


if __name__ == "__main__":
    main()
