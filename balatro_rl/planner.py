"""A build planner that plans on what can actually show up: at every decision outside a round, each candidate move is tried in K sampled futures
(the unseen future redrawn, the same K redraws for every candidate) and each future is played to the end of the run; the move with the best average
outcome is taken, then the planner looks again at the next real shop.

Inside a sampled future (the engine's shop-only game): blinds are won without being played (paid as won with one hand left, scaling jokers advanced a
round), but only if the build could beat them: at every blind select the build's power (env.fast_power: hands x the mean best play / the target) must
reach --survive, else the future ends there; the ante-8 boss is played by the exact round solver. Shops in a future are played by azc's network
(greedy). Score: a win 2, else the blinds reached / 24.

In the real game, rounds are played by PPO v30's network (it plays rounds better than the solver) and the planner makes every other decision.

    python -m balatro_rl.planner --games 64 --k 16 --cands 6
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from . import azc
from . import env as E
from .engine_env import CppVecEnv
from .model import load_model


# P(the real players clear a blind) by the build's fast_power share at its select (measured: 10,027 Gold blinds, azc's shops + v30's rounds;
# checkpoints/survival_calib.npy). The estimate is very pessimistic: a share of 0.05 is still cleared 61% of the time.
SURV_X = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.35, 1.75, 2.5, 4.0, 6.0])
SURV_P = np.array([0.614, 0.762, 0.820, 0.901, 0.942, 0.958, 0.957, 0.970, 0.970, 0.968, 0.986, 0.987, 0.992, 0.995, 0.995, 0.995])


def survival(share):
    return np.interp(share, SURV_X, SURV_P)


def _logits(net, v, idx):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg, _ = net(azc.obs_batch(v, list(idx), "cuda"))
    return lg.float().cpu().numpy()


class Planner:
    def __init__(self, a):
        self.a = a
        self.shop_net = load_model(a.shop_net, "cuda")
        self.round_net = load_model(a.round_net, "cuda")
        self.scr = CppVecEnv(a.scratch, "RED", a.stake, seed0=1, threads=0)
        self.scr.set_exact(True)
        self.scr.v.restart()
        self.scr.set_auto_round(False, 1.5, 0)
        self.scr.set_shop_only(True, a.final_nodes, 0.5)
        self.salt = 1

    def candidates(self, v, i, mask, lg):
        legal = np.flatnonzero(mask[i, E.A_SELECT:]) + E.A_SELECT
        order = legal[np.argsort(-lg[legal])]
        c = list(order[:self.a.cands])
        for extra in (E.A_LEAVE, E.A_SELECT, E.A_PSKIP):
            if mask[i, extra] and extra not in c:
                c.append(extra)
        return c

    def evaluate(self, real, jobs):
        """jobs: (game, action, future k). -> scores [len(jobs)] (each job's future played to the end)."""
        a = self.a
        out = np.zeros(len(jobs))
        for c0 in range(0, len(jobs), self.scr.n):
            ch = jobs[c0:c0 + self.scr.n]
            n = len(ch)
            slots = list(range(n))
            self.checked = {}
            self.scr.copy_many(real, [gi for gi, _, _ in ch], slots)
            self.scr.v.reseed(slots, [int((self.salt * 1_000_003 + gi * 7919 + k * 104729) % (2 ** 40)) + 1 for gi, _, k in ch])
            act = np.full(self.scr.n, -1, dtype=np.int64)
            for s, (_, c, _) in enumerate(ch):
                act[s] = c
            score = np.full(n, np.nan)
            alive_p = np.ones(n)                               # the chance the future is still alive (the blinds passed so far survived)
            acc = np.zeros(n)                                  # the expected score of the deaths so far
            _, d, info = self.scr.step(act)
            for s in np.flatnonzero(d[:n]):
                score[s] = info["blinds"][s] / 24.0 + (1.0 if info["won"][s] else 0.0)
            for _ in range(a.max_steps):
                alive = np.flatnonzero(np.isnan(score))
                if not len(alive):
                    break
                mask = np.asarray(self.scr.obs["mask"])
                g = np.asarray(self.scr.obs["g"])
                at_sel = [s for s in alive if mask[s, E.A_SELECT]]
                if at_sel:                                     # each blind: survived with the measured chance for its power share
                    fp = self.scr.fast_power(at_sel, a.fp_samples)[:, 0]
                    for s, p_ in zip(at_sel, fp):
                        final = np.rint(g[s, 0] * 8) >= 8 and g[s, 3] > 0.5                  # the ante-8 boss: the solver plays it (shop-only final)
                        if final or self.checked.get(s) == (round(float(g[s, 0]), 3), int(np.argmax(g[s, 1:4]))):
                            continue
                        self.checked[s] = (round(float(g[s, 0]), 3), int(np.argmax(g[s, 1:4])))
                        pos = 3 * (np.rint(g[s, 0] * 8) - 1) + np.argmax(g[s, 1:4])
                        q = survival(p_)
                        acc[s] += alive_p[s] * (1 - q) * pos / 24.0
                        alive_p[s] *= q
                        if alive_p[s] < 0.02:
                            score[s] = acc[s] / (1 - alive_p[s]) if alive_p[s] < 1 else pos / 24.0
                    alive = np.flatnonzero(np.isnan(score))
                    if not len(alive):
                        break
                act = np.full(self.scr.n, -1, dtype=np.int64)
                lg = _logits(self.shop_net, self.scr, alive)
                act[alive] = np.where(mask[alive], lg, -np.inf).argmax(1)
                _, d, info = self.scr.step(act)
                for s in alive:
                    if d[s]:
                        score[s] = acc[s] + alive_p[s] * (info["blinds"][s] / 24.0 + (1.0 if info["won"][s] else 0.0))
            left = np.isnan(score)
            if left.any():                                     # (ran past the step cap: where it got to)
                g = np.asarray(self.scr.obs["g"])
                for s in np.flatnonzero(left):
                    score[s] = (3 * (np.rint(g[s, 0] * 8) - 1) + np.argmax(g[s, 1:4])) / 24.0
            out[c0:c0 + n] = score
        self.salt += 1
        return out

    def play(self, n_games, seed0, log=None):
        a = self.a
        v = CppVecEnv(n_games, "RED", a.stake, seed0=seed0, threads=0)
        v.set_exact(True)
        v.v.restart()
        v.set_auto_round(False, 1.5, 0)
        done = np.zeros(n_games, bool)
        bl = np.zeros(n_games)
        wn = np.zeros(n_games, bool)
        changed = 0
        decisions = 0
        t0 = time.time()
        for step in range(9000):
            mask = np.asarray(v.obs["mask"])
            live = np.flatnonzero(~done)
            act = np.full(n_games, -1, dtype=np.int64)
            inr = mask[:, :E.A_SELECT].any(1)
            r = [i for i in live if inr[i]]
            s = [i for i in live if not inr[i]]
            if r:
                act[r] = np.where(mask[r], _logits(self.round_net, v, r), -np.inf).argmax(1)
            if s:
                lg = _logits(self.shop_net, v, s)
                greedy = np.where(mask[s], lg, -np.inf).argmax(1)
                jobs, cands = [], {}
                for j, i in enumerate(s):
                    c = self.candidates(v, i, mask, lg[j])
                    if len(c) <= 1 or a.no_plan:
                        act[i] = greedy[j]
                        continue
                    cands[i] = c
                    for x in c:
                        for k in range(a.k):
                            jobs.append((i, x, k))
                    act[i] = greedy[j]
                if jobs:
                    sc = self.evaluate(v, jobs)
                    pos = 0
                    for i, c in cands.items():
                        m = sc[pos:pos + len(c) * a.k].reshape(len(c), a.k).mean(1)
                        pos += len(c) * a.k
                        best = c[int(np.argmax(m))]
                        decisions += 1
                        changed += int(best != act[i])
                        act[i] = best
            _, d, info = v.step(act)
            f = d & ~done
            bl[f], wn[f] = info["blinds"][f], info["won"][f]
            done |= d
            if log and step % 50 == 0:
                print(json.dumps({"step": step, "done": int(done.sum()), "decisions": decisions, "changed_from_network": changed, "min": round((time.time() - t0) / 60, 1)}), flush=True)
            if done.all():
                break
        return bl, wn, decisions, changed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--seed0", type=int, default=9_000_000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--k", type=int, default=16, help="sampled futures per candidate")
    p.add_argument("--cands", type=int, default=6, help="candidate moves per decision (the network's top ones, plus leave / select / skip pack)")
    p.add_argument("--survive", type=float, default=1.0, help="a sampled future dies at a blind its build's fast_power share is below this")
    p.add_argument("--fp-samples", type=int, default=4)
    p.add_argument("--final-nodes", type=int, default=600)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--scratch", type=int, default=8192)
    p.add_argument("--no-plan", action="store_true", help="the baseline: the networks alone (azc's shops, v30's rounds), same seeds")
    p.add_argument("--shop-net", default="checkpoints/azc_big.pt")
    p.add_argument("--round-net", default="checkpoints/ppo_gold_v30.pt")
    a = p.parse_args()
    pl = Planner(a)
    t = time.time()
    bl, wn, dec, ch = pl.play(a.games, a.seed0, log=True)
    print(json.dumps({"games": a.games, "blinds": round(float(bl.mean()), 3), "wins": int(wn.sum()), "decisions": dec, "changed_from_network": ch,
                      "k": a.k, "cands": a.cands, "survive": a.survive, "minutes": round((time.time() - t) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
