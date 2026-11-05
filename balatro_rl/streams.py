"""Two reward streams with annealed shaping (train.py --streams).

The reward of a run is split in two:
  shaped  the per-blind rewards (and every shaping bonus the trainer adds), attributed to the ante the decision was taken in;
  sparse  the win bonus, the ante-8 bonus and the blind rewards from --sparse-from-ante on.
The critic has a value output per shaped ante (8) and one for the sparse stream (9 in all), each trained on its own, stationary return (the
shaped stream and the sparse stream each have a return normaliser). The policy's advantage is

      A = sum over antes k of w[k] * A_shaped[k]  +  A_sparse                      (in reward units: each stream's advantage is scaled back
                                                                                    by its normaliser's std before they are added)

so only the weights w move while the critic's targets stay put. A weight w[k] fades once the agent has mastered ante k (a set share of the
recent episodes reaches ante k + 1), linearly over a number of updates, and is pushed back up if the reach rate collapses after the fade.
Truncation by a step limit bootstraps from the value of the state it ended in; a loss or a win does not.
"""
from __future__ import annotations

from collections import deque

import numpy as np

N_ANTES = 8                        # shaped value outputs: one per ante
N_OUT = N_ANTES + 1                # ... and the sparse stream's


class RunningStat:
    """Running mean and variance of a stream of values (batches)."""

    def __init__(self):
        self.n, self.mean, self.m2 = 1e-4, 0.0, 0.0

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        k, mu = x.size, float(x.mean())
        m2 = float(((x - mu) ** 2).sum())
        tot = self.n + k
        d = mu - self.mean
        self.m2 += m2 + d * d * self.n * k / tot
        self.mean += d * k / tot
        self.n = tot

    @property
    def var(self) -> float:
        return self.m2 / self.n if self.n > 1.0 else 1.0


class ReturnNorm:
    """The std of a stream's discounted return, tracked over the rollouts; rewards are divided by max(std, floor) so the value targets stay in a
    stable range whatever the reward's units (a floor keeps a stream that is almost always zero from being blown up by one big reward)."""

    def __init__(self, n_envs: int, gamma: float, floor: float = 0.5):
        self.G = np.zeros(n_envs)
        self.prev_done = np.zeros(n_envs, dtype=bool)
        self.gamma, self.floor = gamma, floor
        self.stat = RunningStat()

    def update(self, rew: np.ndarray, done: np.ndarray):
        """rew, done: [T, N] (raw rewards of the stream; episode ends)."""
        gs = np.empty_like(rew, dtype=np.float64)
        for t in range(rew.shape[0]):
            self.G = np.where(self.prev_done, 0.0, self.G) * self.gamma + rew[t]
            gs[t] = self.G
            self.prev_done = done[t].astype(bool)
        self.stat.update(gs)

    @property
    def std(self) -> float:
        return max(float(np.sqrt(self.stat.var)), self.floor)


def gae(rew: np.ndarray, val: np.ndarray, last_val: np.ndarray, done: np.ndarray, trunc: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """Generalised advantage estimation for [T, N, K] rewards and values (K streams or components), with the two kinds of episode end told
    apart. done[t] ends the episode after step t (the trace is cut); trunc[t] says it was cut by a limit, not by a loss or a win: the step then
    bootstraps from the value of the state it was taken in (the next observation is another game's), while a loss or a win bootstraps from 0.
    The last step of the rollout bootstraps from last_val."""
    T = rew.shape[0]
    adv = np.zeros_like(rew, dtype=np.float32)
    g = np.zeros(rew.shape[1:], dtype=np.float32)
    done = done.astype(bool)
    trunc = trunc.astype(bool) & done
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nv = np.where(trunc[t][:, None], val[t], nv)
        live = (~(done[t] & ~trunc[t])).astype(np.float32)[:, None]        # a loss or a win: no value beyond it
        delta = rew[t] + gamma * nv * live - val[t]
        g = delta + gamma * lam * (~done[t]).astype(np.float32)[:, None] * g
        adv[t] = g
    return adv


def by_ante(shaped: np.ndarray, ante: np.ndarray) -> np.ndarray:
    """[T, N] shaped rewards -> [T, N, N_ANTES]: each in the column of the ante its decision was taken in (antes 1..8)."""
    k = np.clip(ante.astype(np.int64) - 1, 0, N_ANTES - 1)
    out = np.zeros(shaped.shape + (N_ANTES,), dtype=np.float32)
    np.put_along_axis(out, k[..., None], shaped[..., None].astype(np.float32), axis=-1)
    return out


class ReachTracker:
    """The furthest ante of the last `window` finished games (9: won): the share that got past each ante. A share of episodes, not a mean:
    a mean hides a policy that either dies early or goes far."""

    def __init__(self, window: int = 500):
        self.window = window
        self.furthest = deque(maxlen=window)

    def add(self, ante: int, won: bool):
        self.furthest.append(9 if won else int(ante))

    def ready(self) -> bool:
        return len(self.furthest) >= self.window // 2

    def rates(self) -> np.ndarray:
        """[N_ANTES]: the share of the recent games that passed ante k+1 (reached ante k+2, or won at k = 8)."""
        f = np.asarray(self.furthest) if self.furthest else np.zeros(1)
        return np.array([float((f >= k + 2).mean()) for k in range(N_ANTES)])


class MilestoneFade:
    """One shaping weight per ante. Weight k starts fading (linearly, over `fade_iters` updates, never a step) once the share of recent
    episodes past ante k reaches `trigger`; if that share falls under `trigger * reverse_ratio` after the fade has begun, the weight is raised
    again at the same speed (a share in between holds it). `begin` lists the weights whose fade started for the first time at this update."""

    def __init__(self, trigger: float = 0.25, fade_iters: int = 200, reverse_ratio: float = 0.5, floor: float = 0.0):
        self.w = np.ones(N_ANTES)
        self.fading = np.zeros(N_ANTES, dtype=bool)
        self.started = np.zeros(N_ANTES, dtype=bool)
        self.trigger, self.step, self.reverse_ratio, self.floor = trigger, 1.0 / max(1, fade_iters), reverse_ratio, floor

    def update(self, reach: np.ndarray) -> list:
        begin = []
        for k in range(N_ANTES):
            if reach[k] >= self.trigger:
                if not self.started[k]:                  # (only the first start: a share that hovers round the trigger pauses and resumes the fade
                    begin.append(k)                      #  without starting it again, so noise does not set off an entropy bump every time)
                self.fading[k] = self.started[k] = True
                self.w[k] = max(self.floor, self.w[k] - self.step)
            else:
                self.fading[k] = False
                if self.started[k] and reach[k] < self.trigger * self.reverse_ratio:
                    self.w[k] = min(1.0, self.w[k] + self.step)
        return begin


class EntropyBump:
    """A temporary rise of the entropy coefficient when a fade begins, decaying linearly back over `decay_iters` updates."""

    def __init__(self, peak: float = 0.02, decay_iters: int = 150):
        self.peak, self.decay_iters, self.left = peak, max(1, decay_iters), 0

    def trigger(self):
        self.left = self.decay_iters

    def value(self) -> float:
        return self.peak * self.left / self.decay_iters

    def step(self):
        self.left = max(0, self.left - 1)


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    v = float(np.var(target))
    return float("nan") if v < 1e-12 else float(1.0 - np.var(target - pred) / v)


class StallDetector:
    """Notices that a win rate has stopped improving and asks for an entropy bump (on decisions outside rounds, in train.py).

    update(win_rate_in_percent) is called once per training update. Once two full windows of `window` updates exist, the mean of the last window
    is compared with the mean of the one before: a gain under `delta` points is a stall, and the bump level to use is returned (it doubles for every
    stall in a row, `base` first, up to `cap`; a window that did improve starts it over). After a bump the detector waits for a whole new window
    before it can fire again. A guard watches the `guard_len` updates after a bump and says `cut` if the win rate falls more than `guard` points
    under what it was when the bump fired (the bump is then ended early)."""

    def __init__(self, window: int = 150, delta: float = 1.0, base: float = 0.003, cap: float = 0.012, guard: float = 3.0, guard_len: int = 20):
        self.window, self.delta, self.base, self.cap, self.guard, self.guard_len = window, delta, base, cap, guard, guard_len
        self.hist: list = []
        self.k = 0
        self.pre = None
        self.guard_left = 0

    def reset(self):
        """A new stake (or any change that moves the win rate by itself): start the windows again."""
        self.hist, self.k, self.pre, self.guard_left = [], 0, None, 0

    def update(self, wr: float, bump_active: bool = False):
        """-> (level, cut): level is the bump to start now (or None), cut says to end the running bump."""
        W = self.window
        self.hist.append(float(wr))
        cut = False
        if self.guard_left > 0:
            self.guard_left -= 1
            if self.pre is not None and wr < self.pre - self.guard:
                cut, self.guard_left = True, 0
        level = None
        if len(self.hist) >= 2 * W and not bump_active and not cut:
            recent = sum(self.hist[-W:]) / W
            prev = sum(self.hist[-2 * W:-W]) / W
            if recent - prev < self.delta:
                self.k += 1
                level = min(self.cap, self.base * 2 ** (self.k - 1))
                self.pre, self.guard_left = recent, self.guard_len
                self.hist = self.hist[-W:]                       # (a whole new window before it can fire again)
            else:
                self.k = 0
                self.hist = self.hist[-2 * W:]
        return level, cut
