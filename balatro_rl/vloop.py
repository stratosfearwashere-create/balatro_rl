"""The V(s) loop (train.py --vloop-every): the strategic layer's value V(s) (the PPO critic) steers the tactical layer (the exact round solver).

The solver keeps the clearing line of a round with the highest money + W x V(state after the round) (engine ROUND_EVAL_W, NNUE_NET). The engine cannot run the
PPO network, so its value network (balatro_rl/nnue.py layout: sparse features -> 256 accumulator -> 32 -> 32 -> 1) is fitted to the critic:
  collect  at each rollout step, a sample of the games outside a round: the engine's features of the state (VecEnv.nnue_feature_lists) and the critic's V(s)
  fit      every K updates, a few hundred steps on the recent samples; the target is V scaled into (0, 1) by the 5th..95th percentile of the recent samples (re-set every fit; a fit tracking the critic below corr 0.3 is not loaded) (was: the 1st..99th percentile of the
           first samples), so the engine's output x 24 is "V in 0..24 units" and W converts it to dollars
  load     the network is exported (nnue.export) and loaded into the engine (VecEnv.load_nnue, one network for the process: training and evaluation)
Until the first fit the solver keeps lines by money alone (W = 0).
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch

from .nnue import NNUE, N_FEATURES, export


class VLoop:
    def __init__(self, venv, dev, bin_path, w=10.0, buffer=200_000, per_step=96, steps=300, lr=1e-3, init="", min_corr=0.3):
        self.venv, self.dev, self.bin_path, self.w = venv, dev, bin_path, float(w)
        self.lr, self.min_corr, self.good = lr, float(min_corr), 0
        self.per_step, self.steps = int(per_step), int(steps)
        self.feats = deque(maxlen=buffer)
        self.vals = deque(maxlen=buffer)
        if init:
            from .nnue import load
            self.net = load(init, dev).train()
        else:
            self.net = NNUE(N_FEATURES).to(dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.lo = self.hi = None
        self.fits = 0
        self.rng = np.random.default_rng(12345)
        os.makedirs(os.path.dirname(bin_path) or ".", exist_ok=True)
        export(self.net.eval(), bin_path)                    # (the engine needs a network loaded for the feature layout; it is not used until the first fit)
        venv.load_nnue(bin_path)
        venv.set_round_eval(0.0)

    def collect(self, mask_np, values):
        """mask_np: the legal-action mask [N, A] of the current step; values: the critic's V(s) [N] (a tensor). Samples games outside a round."""
        from .env import A_SELECT
        out = np.flatnonzero(~mask_np[:, :A_SELECT].any(1))
        if not len(out):
            return
        if len(out) > self.per_step:
            out = self.rng.choice(out, size=self.per_step, replace=False)
        fl = self.venv.v.nnue_feature_lists([int(i) for i in out])
        v = values.detach().float().reshape(-1)[torch.as_tensor(out, device=values.device)].cpu().numpy()
        for f, x in zip(fl, v):
            if len(f):
                self.feats.append(np.asarray(f, dtype=np.int64))
                self.vals.append(float(x))

    def _batch(self, rows):
        parts = [self.feats[r] for r in rows]
        offs = np.cumsum([0] + [len(p) for p in parts])
        return torch.as_tensor(np.concatenate(parts), device=self.dev), torch.as_tensor(offs, device=self.dev)

    def fit(self):
        """-> a log dict. Fits the engine's network to the critic on the recent samples and loads it into the engine."""
        n = len(self.vals)
        if n < 5000:
            return {"vloop_samples": n}
        vals = np.asarray(self.vals, dtype=np.float64)
        # the scale is re-set at every fit (the solver only compares lines of one round under one network, so a moving scale is harmless); the 5th..95th
        # percentile, because a few huge values (wins) on a 1st..99th scale squeezed nearly every target next to 0 and the network fitted a constant (v47 update 300)
        self.lo, self.hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
        if self.hi - self.lo < 1e-3:
            self.hi = self.lo + 1.0
        y = np.clip((vals - self.lo) / (self.hi - self.lo), 0.005, 0.995)
        self.net.train()
        tot = 0.0
        for _ in range(self.steps):
            rows = self.rng.integers(0, n, size=1024)
            bi, bo = self._batch(rows)
            pred = self.net(bi, bo)
            loss = torch.mean((pred - torch.as_tensor(y[rows], dtype=torch.float32, device=self.dev)) ** 2)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            tot += float(loss)
        self.net.eval()
        with torch.no_grad():                                # fit quality on the newest samples
            rows = np.arange(max(0, n - 4096), n)
            bi, bo = self._batch(rows)
            p = self.net(bi, bo).cpu().numpy()
        corr = float(np.corrcoef(p, y[rows])[0, 1]) if np.std(p) > 1e-6 and np.std(y[rows]) > 0 else 0.0
        self.fits += 1
        loaded = corr >= self.min_corr
        if loaded:                                           # (a poor fit is not given to the solver: it keeps the last good network, or money alone)
            export(self.net, self.bin_path)
            self.venv.load_nnue(self.bin_path)
            self.venv.set_round_eval(self.w)
            self.good += 1
        elif np.std(p) <= 1e-6:                              # a collapsed network (constant output): start it again
            self.net = NNUE(N_FEATURES).to(self.dev)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return {"vloop_fits": self.fits, "vloop_loaded": int(loaded), "vloop_good": self.good, "vloop_samples": n,
                "vloop_rmse": round(float(np.sqrt(tot / self.steps)), 4), "vloop_corr": round(corr, 3),
                "vloop_range": [round(self.lo, 3), round(self.hi, 3)], "vloop_w": self.w if self.good else 0.0}
