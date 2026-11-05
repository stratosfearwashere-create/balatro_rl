"""CMA-ES in three sizes, on torch tensors (the GPU for the big ones). All minimise; ask() returns a population [lam, n], tell(f) takes its fitness values.

    CMAES     the full covariance matrix (Hansen's purecma): O(n^2) memory and an eigendecomposition every few generations: for n up to a few thousand.
    SepCMAES  a diagonal covariance (Ros & Hansen 2008): O(n): any n, but it cannot learn correlations between weights.
    LowRankCMAES  the covariance is the identity outside a learned subspace of m ~ 4 + 3 ln n directions and a full CMA covariance inside it (see the class):
              O(n m) memory, O(n (m + lam)^2) time per generation: for hundreds of thousands to millions of weights.

    es = LowRankCMAES(x0, sigma0, lam=128)
    X = es.ask()                  # [128, n]
    es.tell(fitness)              # [128], smaller is better
    es.mean                       # the current estimate of the optimum

`state_dict()` / `load_state_dict()` resume a run.
"""
from __future__ import annotations

import math

import torch


def default_lam(n: int) -> int:
    return 4 + int(3 * math.log(n))


def log_weights(lam: int, mu: int | None = None):
    mu = mu or lam // 2
    w = math.log(lam / 2 + 0.5) - torch.log(torch.arange(1, mu + 1, dtype=torch.float64))
    w = w / w.sum()
    return w, float(1.0 / (w ** 2).sum())


class _Base:
    def __init__(self, x0, sigma0: float, lam: int | None, device=None, dtype=torch.float32, seed: int | None = None):
        x0 = torch.as_tensor(x0, dtype=dtype)
        self.n = x0.numel()
        self.device = torch.device(device) if device is not None else x0.device
        self.dtype = dtype
        self.mean = x0.reshape(-1).clone().to(self.device)
        self.sigma = float(sigma0)
        self.lam = lam or default_lam(self.n)
        self.mu = self.lam // 2
        w, self.mu_eff = log_weights(self.lam, self.mu)
        self.w = w.to(self.device, dtype)
        self.gen = 0
        self.gen_ = torch.Generator(device=self.device)
        self.gen_.manual_seed(seed if seed is not None else 12345)
        self.chi_n = math.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, dtype=self.dtype, generator=self.gen_)

    def _select(self, f):
        f = torch.as_tensor(f, dtype=torch.float64).reshape(-1)
        assert f.numel() == self.lam
        f = torch.nan_to_num(f, nan=float("inf"))
        return torch.argsort(f)[:self.mu].to(self.device)

    def state_dict(self):
        return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in self.__dict__.items() if k not in ("gen_", "device", "w")}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            setattr(self, k, v.to(self.device) if torch.is_tensor(v) else v)


class SepCMAES(_Base):
    def __init__(self, x0, sigma0, lam=None, **kw):
        super().__init__(x0, sigma0, lam, **kw)
        n, me = self.n, self.mu_eff
        self.cs = max((me + 2) / (n + me + 5), min(0.3, 2 * self.lam / n))      # (the textbook rate is ~1/n: too slow to adapt in a few thousand generations when n is large)
        self.ds = 1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.cc = max((4 + me / n) / (n + 4 + 2 * me / n), min(0.3, 2 * self.lam / n))
        k = (n + 2) / 3.0                                              # the separable learning-rate boost
        self.c1 = min(1.0, k * 2 / ((n + 1.3) ** 2 + me))
        self.cmu = min(1 - self.c1, k * 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.C = torch.ones(n, device=self.device, dtype=self.dtype)
        self.ps = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.pc = torch.zeros(n, device=self.device, dtype=self.dtype)

    def ask(self):
        self.Z = self._randn(self.lam, self.n)
        self.Y = self.Z * self.C.sqrt()
        return self.mean + self.sigma * self.Y

    def tell(self, f):
        idx = self._select(f)
        Ys = self.Y[idx]
        yw = (self.w[:, None] * Ys).sum(0)
        self.mean = self.mean + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * yw / self.C.sqrt()
        nps = float(self.ps.norm())
        hs = float(nps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (1.4 + 2 / (self.n + 1)) * self.chi_n)
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc ** 2 + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (self.w[:, None] * Ys ** 2).sum(0)).clamp(min=1e-20)
        self.sigma *= math.exp(min(1.0, self.cs / self.ds * (nps / self.chi_n - 1)))
        self.gen += 1


class LowRankCMAES(_Base):
    """CMA-ES whose covariance is the identity outside a learned subspace of dimension <= m and a full covariance inside it: C = I + U (A - I) U^T, U [n, k] orthonormal,
    A = diag(eigenvalues) with k <= m. Each generation the subspace is widened by the directions of the evolution path and of the selected steps that it does not yet
    contain, the standard CMA update (rank-one + rank-mu) is applied to the covariance in that wider basis, and the m directions that differ most from isotropic are kept
    (the rest go back to the identity). Memory O(n m), time O(n (m + lam)^2) per generation, so hundreds of thousands to millions of weights are fine; what it cannot learn
    are correlations spread over more than m directions."""

    def __init__(self, x0, sigma0, lam=None, m=None, **kw):
        super().__init__(x0, sigma0, lam, **kw)
        n = self.n
        self.m = min(m or (4 + int(3 * math.log(n))), n // 2)
        me = self.mu_eff
        d = self.m + self.mu                                          # the dimension the covariance is adapted in: sets the learning rates
        self.cs = max((me + 2) / (n + me + 5), min(0.3, 2 * self.lam / n))
        self.ds = 1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.cc = max((4 + me / n) / (n + 4 + 2 * me / n), min(0.3, 2 * self.lam / n))
        self.c1 = 2 / ((d + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((d + 2) ** 2 + me))
        self.U = torch.zeros(n, 0, device=self.device, dtype=self.dtype)
        self.ev = torch.zeros(0, device=self.device, dtype=self.dtype)          # the eigenvalues of C along the columns of U
        self.ps = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.pc = torch.zeros(n, device=self.device, dtype=self.dtype)

    def ask(self):
        self.Z = self._randn(self.lam, self.n)
        Y = self.Z
        if self.U.shape[1]:
            Y = self.Z + ((self.Z @ self.U) * (self.ev.sqrt() - 1)) @ self.U.T
        self.Y = Y
        return self.mean + self.sigma * Y

    def _inv_sqrt(self, y):
        if not self.U.shape[1]:
            return y
        return y + ((y @ self.U) * (self.ev.rsqrt() - 1)) @ self.U.T

    def tell(self, f):
        idx = self._select(f)
        Ys = self.Y[idx]
        yw = (self.w[:, None] * Ys).sum(0)
        self.mean = self.mean + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * self._inv_sqrt(yw)
        nps = float(self.ps.norm())
        hs = float(nps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (1.4 + 2 / (self.n + 1)) * self.chi_n)
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * yw
        # widen the subspace by what the path and the selected steps contain outside it
        k = self.U.shape[1]
        R = torch.cat([self.pc[None], Ys], 0)
        if k:
            R = R - (R @ self.U) @ self.U.T
        r = min(R.shape[0], self.n - k)
        Q, _ = torch.linalg.qr(R.T)                                # [n, R.shape[0]] orthonormal, orthogonal to U
        Q = Q[:, :r]
        if k:
            Q = Q - self.U @ (self.U.T @ Q)
            Q, _ = torch.linalg.qr(Q)
        W = torch.cat([self.U, Q], 1)                              # [n, k + r]
        dw = W.shape[1]
        Chat = torch.diag(torch.cat([self.ev, torch.ones(dw - k, device=self.device, dtype=self.dtype)]))
        yh = Ys @ W
        ph = self.pc @ W
        Chat = ((1 - self.c1 - self.cmu) * Chat + self.c1 * (torch.outer(ph, ph) + (1 - hs) * self.cc * (2 - self.cc) * Chat)
                + self.cmu * (yh.T * self.w) @ yh)
        Chat = ((Chat + Chat.T) / 2).double()
        ev, V = torch.linalg.eigh(Chat)
        ev = ev.clamp(min=1e-12)
        keep = torch.argsort(-ev.log().abs())[:self.m]
        keep = keep[ev[keep].log().abs() > 1e-9]
        self.U = W @ V[:, keep].to(self.dtype)
        self.ev = ev[keep].to(self.dtype)
        self.sigma *= math.exp(min(1.0, self.cs / self.ds * (nps / self.chi_n - 1)))
        self.gen += 1


class CMAES(_Base):
    def __init__(self, x0, sigma0, lam=None, **kw):
        kw.setdefault("dtype", torch.float64)
        super().__init__(x0, sigma0, lam, **kw)
        n, me = self.n, self.mu_eff
        assert n <= 6000, "CMAES keeps an n x n matrix and eigendecomposes it: use SepCMAES or LMMAES"
        self.cs = (me + 2) / (n + me + 5)
        self.ds = 1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.C = torch.eye(n, device=self.device, dtype=self.dtype)
        self.B = torch.eye(n, device=self.device, dtype=self.dtype)
        self.Dd = torch.ones(n, device=self.device, dtype=self.dtype)
        self.ps = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.pc = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.eig_gen = 0

    def ask(self):
        self.Z = self._randn(self.lam, self.n)
        self.Y = (self.Z * self.Dd) @ self.B.T
        return self.mean + self.sigma * self.Y

    def tell(self, f):
        idx = self._select(f)
        Ys = self.Y[idx]
        yw = (self.w[:, None] * Ys).sum(0)
        self.mean = self.mean + self.sigma * yw
        Cinv_yw = self.B @ ((self.B.T @ yw) / self.Dd)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * Cinv_yw
        nps = float(self.ps.norm())
        hs = float(nps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (1.4 + 2 / (self.n + 1)) * self.chi_n)
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * yw
        rank_mu = (Ys.T * self.w) @ Ys
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (torch.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C) + self.cmu * rank_mu)
        self.sigma *= math.exp(min(1.0, self.cs / self.ds * (nps / self.chi_n - 1)))
        self.gen += 1
        if self.gen - self.eig_gen > 1 / (10 * self.n * (self.c1 + self.cmu)):               # (the eigendecomposition is the expensive part)
            self.eig_gen = self.gen
            self.C = (self.C + self.C.T) / 2
            ev, B = torch.linalg.eigh(self.C)
            ev = ev.clamp(min=1e-30)
            self.Dd, self.B = ev.sqrt(), B


def make(name: str, x0, sigma0, lam=None, **kw):
    return {"full": CMAES, "sep": SepCMAES, "lm": LowRankCMAES}[name](x0, sigma0, lam, **kw)
