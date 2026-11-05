"""Afterstate learning with the evaluation network (balatro_rl/nnue.py), TD-Gammon style, on ordinary Gold games (the seed unknown).

Rounds are played by the policy network (--ckpt, greedy). At every decision outside a round (shop, pack, blind select), each candidate action
(the policy's --k most likely, plus leaving / skipping when legal) is applied on a copy (a reroll or a pack bought on a copy whose future is
redrawn, so it does not see the real next shop); the copy's position is read as features (VecEnv.nnue_feature_lists) and scored by the network.
The best is played (with probability --eps a random candidate). When a game ends, every position it chose gets the game's outcome as its target
(blinds reached / 24; a win 1.0, Monte Carlo); the network trains on a replay buffer of recent targets. Every --eval-every games: the greedy agent
on --eval-games fixed seeds against the policy alone.

    python tools/nnue_td.py --init checkpoints/nnue1.pt --games 20000 --out checkpoints/nnue_td1.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import env as E                                    # noqa: E402
from balatro_rl.engine_env import CppVecEnv                         # noqa: E402
from balatro_rl.nnue import NNUE, N_FEATURES, load                  # noqa: E402
from policy_search import Pol                                       # noqa: E402

CHANCE = set(range(E.A_BUY_PACK, E.A_VOUCHER)) | {E.A_REROLL}


def to_batch(lists, dev):
    offs = np.cumsum([0] + [len(x) for x in lists])
    idx = np.fromiter((f for x in lists for f in x), dtype=np.int64, count=int(offs[-1]))
    return torch.as_tensor(idx, device=dev), torch.as_tensor(offs, device=dev)


class Agent:
    def __init__(self, net, pol, k, dev, nnue_bin):
        self.net, self.pol, self.k, self.dev = net, pol, k, dev
        self.nnue_bin = nnue_bin
        self.salt = 0

    def run(self, seeds, eps=0.0, record=True, max_steps=4000):
        """Plays the games `seeds`; -> blinds [G], won [G], and per game the chosen positions' feature lists."""
        G = len(seeds)
        v = CppVecEnv(G, "RED", "GOLD", seed0=seeds[0], threads=0)
        v.v.restart()
        S = CppVecEnv(G * (self.k + 2), "RED", "GOLD", seed0=1, threads=0)
        S.load_nnue(self.nnue_bin)                                    # (the feature layout for nnue_feature_lists)
        rng = np.random.default_rng(seeds[0])
        done = np.zeros(G, dtype=bool)
        bl = np.zeros(G)
        won = np.zeros(G, dtype=bool)
        chosen = [[] for _ in range(G)]
        for _ in range(max_steps):
            live = np.flatnonzero(~done)
            if not len(live):
                break
            mask = np.asarray(v.obs["mask"])
            lg = self.pol.logits(v, live)
            act = np.full(G, -1, dtype=np.int64)
            act[live] = lg.argmax(1)
            outr = ~mask[live, :E.A_SELECT].any(1) & (mask[live].sum(1) >= 2)
            sg = live[outr]
            if len(sg):
                lgo = lg[outr]
                src, acts, own = [], [], []
                for r, gi in enumerate(sg):
                    cands = [int(c) for c in np.argsort(-lgo[r])[:self.k] if mask[gi, c]]
                    for extra in (E.A_LEAVE, E.A_SKIP, E.A_SELECT, E.A_PSKIP):
                        if mask[gi, extra] and extra not in cands:
                            cands.append(int(extra))
                    for c in cands[:self.k + 2]:
                        src.append(int(gi))
                        acts.append(c)
                        own.append(r)
                n = len(src)
                S.copy_many(v, src, list(range(n)))
                self.salt += 1
                ch = [j for j, c in enumerate(acts) if c in CHANCE]
                if ch:
                    S.reseed_rng(ch, [self.salt * 1_000_003 + j * 7919 + 3 for j in ch])
                a0 = np.full(S.n, -1, dtype=np.int64)
                a0[:n] = [(-1 if c == E.A_SELECT else c) for c in acts]          # selecting: the blind select itself is the position
                _, d, info = S.step(a0)
                feats = S.v.nnue_feature_lists(list(range(n)))
                with torch.no_grad():
                    val = self.net(*to_batch(feats, self.dev)).cpu().numpy()
                fin = d[:n]
                if fin.any():
                    bits = S.last_bits()
                    beaten = np.array([bin(int(x)).count("1") for x in bits[:n]], dtype=np.float64)
                    val = np.where(fin, np.where(info["won"][:n], 1.0, beaten / 24.0), val)
                for r, gi in enumerate(sg):
                    js = [j for j in range(n) if own[j] == r]
                    if not js:
                        continue
                    j = js[int(rng.integers(len(js)))] if rng.random() < eps else js[int(np.argmax(val[js]))]
                    act[gi] = acts[j]
                    if record and not fin[j]:
                        chosen[gi].append(feats[j])
            _, d, info = v.step(act)
            f = d & ~done
            if f.any():                                              # blinds actually beaten (a skipped blind is not progress)
                bits = v.last_bits()
                beaten = np.array([bin(int(x)).count("1") for x in bits])
                bl[f], won[f] = beaten[f], info["won"][f]
            done |= d
        return bl, won, chosen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v7_best_greedy.pt")
    p.add_argument("--init", default="checkpoints/nnue1.pt")
    p.add_argument("--nnue-bin", default="checkpoints/nnue1.bin", help="any exported network (only its feature layout is used here)")
    p.add_argument("--games", type=int, default=20000)
    p.add_argument("--batch-games", type=int, default=128)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--buffer", type=int, default=200_000)
    p.add_argument("--steps-per-batch", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=2048)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--seed0", type=int, default=2_000_000)
    p.add_argument("--out", default="checkpoints/nnue_td1.pt")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = load(a.init, dev).train() if a.init else NNUE(N_FEATURES).to(dev)
    pol = Pol(a.ckpt, dev)
    agent = Agent(net, pol, a.k, dev, a.nnue_bin)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    buf_f, buf_y = [], []
    eval_seeds = list(range(700_000, 700_000 + a.eval_games))
    net.eval()
    b0, w0, _ = agent.run(eval_seeds, eps=0.0, record=False)
    v7b = Agent(net, pol, 0, dev, a.nnue_bin)
    print(f"start: the network as the shop agent on {a.eval_games} eval games: {b0.mean():.2f} blinds beaten, {int(w0.sum())} wins", flush=True)
    t0 = time.time()
    played = 0
    next_eval = a.eval_every
    seed = a.seed0
    rng = np.random.default_rng(0)
    while played < a.games:
        net.eval()
        bl, won, chosen = agent.run(list(range(seed, seed + a.batch_games)), eps=a.eps)
        seed += a.batch_games
        played += a.batch_games
        for gi in range(len(bl)):
            z = 1.0 if won[gi] else bl[gi] / 24.0
            for f in chosen[gi]:
                buf_f.append(f)
                buf_y.append(z)
        if len(buf_y) > a.buffer:
            buf_f, buf_y = buf_f[-a.buffer:], buf_y[-a.buffer:]
        net.train()
        ys = np.asarray(buf_y, dtype=np.float32)
        tot = 0.0
        for _ in range(a.steps_per_batch):
            rows = rng.integers(0, len(ys), size=512)
            pred = net(*to_batch([buf_f[r] for r in rows], dev))
            loss = torch.mean((pred - torch.as_tensor(ys[rows], device=dev)) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"{played} games ({time.time() - t0:.0f}s): batch {bl.mean():.2f} blinds, {int(won.sum())} wins; buffer {len(ys)}, "
              f"rmse {np.sqrt(tot / a.steps_per_batch) * 24:.2f} blinds", flush=True)
        if played >= next_eval:
            next_eval += a.eval_every
            net.eval()
            be, we, _ = agent.run(eval_seeds, eps=0.0, record=False)
            print(f"  EVAL after {played} games: {be.mean():.2f} blinds, {int(we.sum())} wins on {a.eval_games} games", flush=True)
            torch.save({"config": net.config, "state": net.state_dict()}, a.out)
    torch.save({"config": net.config, "state": net.state_dict()}, a.out)


if __name__ == "__main__":
    main()
