"""Neuroevolution of the shop policy with CMA-ES (cma.py), on the shop-only game (the scripted player plays the rounds; the policy acts outside them).

One weight vector is one policy. A generation draws `pop` weight vectors from the search distribution and plays each of them in `games` games; every candidate meets the same
games (common random numbers: the games are copies of `games` freshly seeded exact-seed games), so the ranking is not luck of the draw. Fitness = the mean blinds beaten (+ a bonus
per win). All pop x games games run at once in one vector environment and every candidate's network is evaluated in one batched matrix product on the GPU.

The network (ShopNet): 72 actions outside rounds, each scored from its own feature row, its item's embedding and a context encoded from the state:
    state  = [symlog(global features), symlog(8 jokers' features), sum of the held jokers' embeddings, sum of the consumables' embeddings]
    context = relu(W2 relu(W1 state))
    logit(a) = W4 relu(W3 relu(Wa [symlog(action row), embedding(item)] + Wc context))
hidden=256, emb=16 is 283k weights; hidden=512 is 0.89M (the default); hidden=1024, emb=32 is 3.1M (about the most a 16 GB GPU takes with pop 128: 10 GB). All of it is searched, embeddings included.

    python -m balatro_rl.evo --algo lm --hidden 256 --pop 128 --games 8 --out checkpoints/cma1.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import cma
from .env import A_SELECT, F_ACT_W, F_JOKER_W, MAX_JOKERS, N_ACTIONS, VOCAB_SIZE, global_size

A0 = A_SELECT
NA = N_ACTIONS - A0                                        # the 72 actions outside rounds
N_CONS = 3


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


class ShopNet:
    """The layout of a policy's weights in one flat vector, and its batched forward pass (the weights of P candidates at once: [P, n])."""

    def __init__(self, hidden=256, emb=16):
        self.H, self.E = hidden, emb
        self.G = global_size()
        self.S = self.G + MAX_JOKERS * F_JOKER_W + 2 * emb
        self.A = F_ACT_W + emb
        H, E, S, A = hidden, emb, self.S, self.A
        self.shapes = [("emb", (VOCAB_SIZE, E)), ("W1", (S, H)), ("b1", (H,)), ("W2", (H, H)), ("b2", (H,)), ("Wa", (A, H)), ("Wc", (H, H)), ("ba", (H,)),
                       ("W3", (H, H // 2)), ("b3", (H // 2,)), ("W4", (H // 2, 1)), ("b4", (1,))]
        self.offs, o = {}, 0
        for name, sh in self.shapes:
            self.offs[name] = (o, sh)
            o += int(np.prod(sh))
        self.n = o

    def init(self, seed=0, scale=1.0):
        """He-style random weights as one vector (biases zero, the last layer small so the first policy is near-uniform)."""
        g = torch.Generator().manual_seed(seed)
        v = torch.zeros(self.n)
        for name, sh in self.shapes:
            o, _ = self.offs[name]
            sz = int(np.prod(sh))
            if name.startswith("W"):
                std = scale * (2.0 / sh[0]) ** 0.5 * (0.1 if name == "W4" else 1.0)
                v[o:o + sz] = torch.randn(sz, generator=g) * std
            elif name == "emb":
                v[o:o + sz] = torch.randn(sz, generator=g) * 0.3
        return v

    def parts(self, W):
        """W [P, n] -> dict of views [P, *shape]."""
        P = W.shape[0]
        return {name: W[:, o:o + int(np.prod(sh))].reshape((P,) + sh) for name, (o, sh) in self.offs.items()}

    def forward(self, W, g, jf, jid, cid, af, aid, mask):
        """W [P, n]; the observation of P x M games, batched as [P, M, ...]: g [P,M,G], jf [P,M,8,F], jid [P,M,8], cid [P,M,3], af [P,M,72,F_ACT], aid [P,M,72], mask [P,M,72]
        -> logits [P, M, 72] (-1e9 where the action is illegal)."""
        p = self.parts(W)
        P, M = g.shape[:2]
        ar = torch.arange(P, device=W.device)[:, None]

        def lookup(ids):                                                    # the embedding table of candidate p at ids [P, M, k]
            k = ids.shape[2]
            e = p["emb"][ar, ids.reshape(P, -1).long()]
            return e.reshape(P, M, k, -1)
        ej = (lookup(jid) * (jid > 0).unsqueeze(-1)).sum(2)
        ec = (lookup(cid) * (cid > 0).unsqueeze(-1)).sum(2)
        x = torch.cat([symlog(g), symlog(jf).reshape(P, M, -1), ej, ec], -1)             # [P, M, S]
        h = torch.relu(torch.baddbmm(p["b1"][:, None], x, p["W1"]))
        c = torch.relu(torch.baddbmm(p["b2"][:, None], h, p["W2"]))
        arow = torch.cat([symlog(af), lookup(aid)], -1).reshape(P, M * NA, self.A)        # [P, M*72, A]
        ah = torch.bmm(arow, p["Wa"]).reshape(P, M, NA, -1)
        ctx = torch.bmm(c, p["Wc"]).unsqueeze(2)
        z = torch.relu(ah + ctx + p["ba"][:, None, None])
        z = torch.relu(torch.bmm(z.reshape(P, M * NA, -1), p["W3"]) + p["b3"][:, None]).reshape(P, M, NA, -1)
        logit = (z @ p["W4"][:, None]).squeeze(-1) + p["b4"][:, None]                    # [P, M, 72]
        return logit.masked_fill(~mask, -1e9)


class Evaluator:
    """Plays P candidates x M games in one vector environment; every candidate gets copies of the same M games."""

    def __init__(self, net: ShopNet, P, M, deck="RED", stake="GOLD", pace=1.5, threads=0, dev="cuda", win_bonus=10.0, max_steps=500):
        from .engine_env import CppVecEnv
        self.net, self.P, self.M, self.dev, self.win_bonus, self.max_steps = net, P, M, dev, win_bonus, max_steps
        self.deck, self.stake, self.pace, self.threads = deck, stake, pace, threads
        self.N = P * M
        self.v = CppVecEnv(self.N, deck, stake, seed0=1, threads=threads, reward_kw={"ante_weight": 8.0, "win_bonus": 30.0, "win_ante": 8, "shape_gamma": 0.999})
        self.v.set_exact(True)
        self.v.set_auto_round(True, pace, 0)
        self.src = [i % M for i in range(self.N)]
        self.dst = list(range(self.N))

    def _roots(self, seed0):
        from .engine_env import CppVecEnv
        r = CppVecEnv(self.M, self.deck, self.stake, seed0=seed0, threads=1)
        r.set_exact(True)
        r.v.restart()
        return r

    @torch.no_grad()
    def run(self, W, seed0):
        """W [P, n] (on the device) -> (fitness [P] mean blinds + bonus, blinds [P, M], won [P, M])."""
        v, N, P, M, dev = self.v, self.N, self.P, self.M, self.dev
        v.copy_many(self._roots(seed0), self.src, self.dst)
        v.set_auto_round(True, self.pace, 0)
        done = np.zeros(N, dtype=bool)
        blinds = np.zeros(N)
        won = np.zeros(N, dtype=bool)
        for step in range(self.max_steps):
            o = v.obs
            mask_full = np.asarray(o["mask"])
            t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
            logits = self.net.forward(W, t(o["g"]).float().reshape(P, M, -1), t(np.asarray(o["jf"])[:, :, :F_JOKER_W]).float().reshape(P, M, MAX_JOKERS, F_JOKER_W),
                                      t(o["jid"]).reshape(P, M, -1), t(o["cid"]).reshape(P, M, -1), t(np.asarray(o["af"])[:, A0:, :F_ACT_W]).float().reshape(P, M, NA, F_ACT_W),
                                      t(np.asarray(o["aid"])[:, A0:]).reshape(P, M, NA), t(mask_full[:, A0:]).reshape(P, M, NA))
            act = (logits.reshape(N, NA).argmax(1).cpu().numpy() + A0).astype(np.int64)
            inround = mask_full[:, :A0].any(1)                                      # (never in this game; the first legal action if it happens)
            if inround.any():
                act = np.where(inround, mask_full.argmax(1), act)
            act = np.where(done, -1, act)
            _, d, info = v.step(act)
            first = d & ~done
            blinds[first], won[first] = info["blinds"][first], info["won"][first]
            done |= d
            if done.all():
                break
        if not done.all():                                                           # (a game that ran past the step cap: its progress so far)
            blinds[~done] = info["blinds"][~done]
        fit = (blinds + self.win_bonus * won).reshape(P, M).mean(1)
        return torch.from_numpy(fit), blinds.reshape(P, M), won.reshape(P, M)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default="lm", choices=["lm", "sep", "full"], help="lm: low-rank CMA-ES (hundreds of thousands of weights); sep: diagonal; full: the whole covariance (n <= ~3000)")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--emb", type=int, default=16)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--games", type=int, default=8, help="games per candidate per generation")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--m", type=int, default=0, help="directions of the low-rank covariance (0: 4 + 3 ln n)")
    p.add_argument("--gens", type=int, default=100000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--win-bonus", type=float, default=10.0)
    p.add_argument("--pace", type=float, default=1.5)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-games", type=int, default=128)
    p.add_argument("--eval-seed0", type=int, default=9_000_000)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--out", default="checkpoints/cma1.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu-fraction", type=float, default=0.0)
    return p.parse_args(argv)


def train(a):
    dev = a.device
    if str(dev).startswith("cuda") and a.gpu_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)
    net = ShopNet(a.hidden, a.emb)
    log_path = a.out.replace(".pt", "_log.jsonl")
    kw = {"m": a.m or None} if a.algo == "lm" else {}
    g0 = 0
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location="cpu", weights_only=False)
        assert ck["hidden"] == a.hidden and ck["emb"] == a.emb, "the checkpoint's network is another size"
        es = cma.make(a.algo, ck["mean"], a.sigma, a.pop, device=dev, seed=a.seed + ck["gen"], **kw)
        es.load_state_dict(ck["es"])
        g0 = ck["gen"]
        print(json.dumps({"resumed": a.out, "gen": g0}), flush=True)
    else:
        es = cma.make(a.algo, net.init(a.seed), a.sigma, a.pop, device=dev, seed=a.seed, **kw)
    print(json.dumps({"weights": net.n, "algo": a.algo, "pop": es.lam, "games": a.games, "games_per_generation": es.lam * a.games, "device": str(dev)}), flush=True)
    train_ev = Evaluator(net, es.lam, a.games, a.deck, a.stake, a.pace, a.threads, dev, a.win_bonus)
    eval_ev = Evaluator(net, 1, a.eval_games, a.deck, a.stake, a.pace, a.threads, dev, a.win_bonus)
    best_eval, t0 = -1.0, time.time()
    for gen in range(g0 + 1, a.gens + 1):
        tg = time.time()
        X = es.ask()
        fit, blinds, won = train_ev.run(X, seed0=a.seed * 10_000_000 + gen * 1000)
        es.tell((-fit).double().cpu())
        row = {"gen": gen, "best": round(float(fit.max()), 3), "mean": round(float(fit.mean()), 3), "median": round(float(fit.median()), 3), "blinds_mean": round(float(blinds.mean()), 3),
               "wins": int(won.sum()), "sigma": round(es.sigma, 5), "secs": round(time.time() - tg, 1), "min": round((time.time() - t0) / 60, 1)}
        if a.eval_every and gen % a.eval_every == 0:
            f_, b_, w_ = eval_ev.run(es.mean[None].float(), seed0=a.eval_seed0)
            row["eval"] = {"games": a.eval_games, "blinds": round(float(b_.mean()), 3), "win_rate": round(float(w_.mean()), 4)}
            if row["eval"]["blinds"] > best_eval:
                best_eval = row["eval"]["blinds"]
                torch.save({"mean": es.mean.cpu(), "hidden": a.hidden, "emb": a.emb, "gen": gen, "eval": row["eval"]}, a.out.replace(".pt", "_best.pt"))
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if gen % a.save_every == 0:
            torch.save({"mean": es.mean.cpu(), "es": es.state_dict(), "hidden": a.hidden, "emb": a.emb, "gen": gen}, a.out)
    torch.save({"mean": es.mean.cpu(), "es": es.state_dict(), "hidden": a.hidden, "emb": a.emb, "gen": a.gens}, a.out)


if __name__ == "__main__":
    train(parse())
