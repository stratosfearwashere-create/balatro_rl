"""Shop decisions valued by what they do to the rest of the run.

    Gain(a) = V(state after a) - V(state now)          (a = buy / sell / reroll / reorder / skip / leave; leave is the "do nothing" candidate, so Gain(leave) = 0)

PairedRollouts   V from simulation: the state is copied once per (candidate, seed); the candidate is played; then the run goes on with the rollout policy (the shop network for shops and
                 packs, the scripted mode-2 player for rounds) to its end. Every candidate meets the same seeds, so the differences between candidates are the action's, not luck. Money,
                 interest and the slot a purchase takes are inside V because the rollouts live them (no separate Cost term is needed). V = mean blinds beaten (+ a bonus per win);
                 P(reach ante k) comes with it.
GapReport        the requirement curve: for the build as it is, the pass margin at every blind still to come, with each scaler's counter grown to that round by its natural rate and
                 the judge's pass chance (margin = 10 ** (log10 share of the requirement the best play reaches + 0.09): 1 is a coin flip): where the first round below 1 is (the gap
                 the shop has to bridge with power now) and whether the build clears every round until a scaler matures.
VJudge           a cheap V without rollouts from the same curve: the expected number of blinds survived, sum over the future rounds t of prod_{s<=t} p(s); Gain from it for every
                 candidate in one batch.
"""
from __future__ import annotations

import numpy as np
import torch

from .env import (A_BUY, A_BUY_PACK, A_LEAVE, A_PICK, A_PSKIP, A_REPLACE, A_REROLL, A_SELECT, A_SELL_C, A_SELL_J, A_SKIP, A_SWAP, A_SWAP_END, A_USE_C, A_VOUCHER, F_JOKER_W, N_ACTIONS)

KEYS_NO_PRIV = None


def obs_keys():
    global KEYS_NO_PRIV
    if KEYS_NO_PRIV is None:
        from .engine_env import OBS_KEYS
        KEYS_NO_PRIV = [k for k in OBS_KEYS if k != "priv"]
    return KEYS_NO_PRIV


def shop_candidates(mask_row):
    """The legal actions of a shop state worth valuing: every purchase, sale, reroll, reorder, use of a consumable, and leaving. (Pack picks are the policy's.)"""
    m = np.asarray(mask_row, dtype=bool)
    out = [a for a in range(A_BUY, N_ACTIONS) if m[a] and not (A_PICK <= a <= A_PSKIP)]
    return out


class ManagerPolicy:
    """The shop network, greedy (or sampled at a temperature), as a rollout / data policy: called with the environment and the indices of the games to act in."""

    def __init__(self, manager, dev, temperature=0.0, explore=0.0, seed=0):
        self.m, self.dev, self.temperature, self.explore = manager, dev, temperature, explore
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def logits(self, env, ids):
        o = {k: torch.from_numpy(np.ascontiguousarray(np.asarray(env.obs[k])[ids])).to(self.dev) for k in obs_keys()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.dev).startswith("cuda")):
            lg, _ = self.m(o)
        return lg.float()

    def __call__(self, env, ids):
        lg = self.logits(env, ids)
        a = (torch.distributions.Categorical(logits=lg / self.temperature).sample() if self.temperature > 0 else lg.argmax(1)).cpu().numpy()
        if self.explore > 0:
            mask = np.asarray(env.obs["mask"])[ids]
            for i in np.flatnonzero(self.rng.random(len(ids)) < self.explore):
                a[i] = self.rng.choice(np.flatnonzero(mask[i]))
        return a


class QPolicy(ManagerPolicy):
    """The shop network improved by its Q head: in a shop, take the legal shop action with the highest predicted Gain over the network's own choice when that is above `thr`;
    otherwise (and everywhere else) do what the network does."""

    def __init__(self, model, dev, thr=0.0, **kw):
        super().__init__(model, dev, **kw)
        self.thr = thr
        self.used = 0
        self.calls = 0

    @torch.no_grad()
    def __call__(self, env, ids):
        o = {k: torch.from_numpy(np.ascontiguousarray(np.asarray(env.obs[k])[ids])).to(self.dev) for k in obs_keys()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.dev).startswith("cuda")):
            lg, _ = self.m(o)
        lg = lg.float()
        q = self.m.q_values().float()
        mask = o["mask"]
        a = lg.argmax(1)
        shop = mask[:, A_LEAVE]
        sa = torch.zeros_like(mask)
        sa[:, A_BUY:] = mask[:, A_BUY:]
        sa[:, A_PICK:A_PSKIP + 1] = False
        qm = q.masked_fill(~sa, -1e9)
        best = qm.argmax(1)
        take = shop & (qm.gather(1, best[:, None]).squeeze(1) > self.thr)
        a = torch.where(take, best, a)
        self.used += int(take.sum()); self.calls += int(shop.sum())
        return a.cpu().numpy()


class SoftQPolicy(ManagerPolicy):
    """The policy IS the Q head (learning from scratch): the action is sampled from softmax(Q / tau) over the legal actions (greedy with tau = 0), in every state."""

    def __init__(self, model, dev, tau=0.3, **kw):
        super().__init__(model, dev, **kw)
        self.tau = tau

    @torch.no_grad()
    def __call__(self, env, ids):
        o = {k: torch.from_numpy(np.ascontiguousarray(np.asarray(env.obs[k])[ids])).to(self.dev) for k in obs_keys()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.dev).startswith("cuda")):
            self.m(o)
        q = self.m.q_values().float().masked_fill(~o["mask"], -1e9)
        if self.tau > 0:
            a = torch.distributions.Categorical(logits=q / self.tau).sample()
        else:
            a = q.argmax(1)
        a = a.cpu().numpy()
        if self.explore > 0:
            mask = np.asarray(env.obs["mask"])[ids]
            for i in np.flatnonzero(self.rng.random(len(ids)) < self.explore):
                a[i] = self.rng.choice(np.flatnonzero(mask[i]))
        return a


def all_candidates(mask_row):
    """Every legal action outside rounds (blind select / skip, the shop, packs): the candidates of a from-scratch policy."""
    return [int(a) for a in np.flatnonzero(np.asarray(mask_row, dtype=bool)) if a >= A_SELECT]


class PairedRollouts:
    def __init__(self, policy, cap=4096, deck="RED", stake="GOLD", mode=2, pace=1.5, max_steps=500, win_bonus=10.0, threads=0):
        from .engine_env import CppVecEnv
        self.policy, self.cap, self.mode, self.pace = policy, cap, mode, pace
        self.max_steps, self.win_bonus = max_steps, win_bonus
        self.ro = CppVecEnv(cap, deck, stake, seed0=1, threads=threads)
        self.ro.set_marginal(0)

    def values_many(self, venv, states, n_seeds, salt0=1_000_000):
        """states: list of (env index, candidate actions); a candidate -1 is "no forced action": the rollout policy decides what to do first. Every candidate of every state meets the same
        n_seeds draws. -> list of dicts: v [C] mean outcome (blinds beaten + win bonus), se [C], blinds [C, n_seeds], won, reach [C, 9]."""
        rows = [(si, c) for si, (e, cands) in enumerate(states) for c in cands]
        R = len(rows) * n_seeds
        assert R <= self.cap, f"{R} rollouts do not fit the scratch environment ({self.cap})"
        ro = self.ro
        src = [states[si][0] for si, c in rows for _ in range(n_seeds)]
        ro.copy_many(venv, src, list(range(R)))
        ro.reseed_rng(list(range(R)), [salt0 + r for _ in rows for r in range(n_seeds)])
        ro.set_auto_round(True, self.pace, self.mode)
        ro.set_modes([self.mode] * ro.n)
        act = np.full(ro.n, -1, dtype=np.int64)
        act[:R] = np.repeat(np.asarray([c for _, c in rows], dtype=np.int64), n_seeds)
        done = np.zeros(ro.n, dtype=bool)
        done[R:] = True
        blinds, won, ante = np.zeros(ro.n), np.zeros(ro.n, dtype=bool), np.ones(ro.n)
        _, d, info, _, _ = ro.step_split(act)
        first = d & ~done
        blinds[first], won[first], ante[first] = info["blinds"][first], info["won"][first], info["ante"][first]
        done |= d
        for step in range(self.max_steps):
            if done.all():
                break
            live = np.flatnonzero(~done)
            a = np.full(ro.n, -1, dtype=np.int64)
            a[live] = self.policy(ro, live)
            _, d, info, _, _ = ro.step_split(a)
            first = d & ~done
            blinds[first], won[first], ante[first] = info["blinds"][first], info["won"][first], info["ante"][first]
            done |= d
        if not done.all():
            rest = ~done
            blinds[rest], ante[rest] = info["blinds"][rest], info["ante"][rest]
        out, k = [], 0
        for si, (e, cands) in enumerate(states):
            C = len(cands)
            b = blinds[k * n_seeds:(k + C) * n_seeds].reshape(C, n_seeds)
            w = won[k * n_seeds:(k + C) * n_seeds].reshape(C, n_seeds)
            k += C
            val = b + self.win_bonus * w
            furthest = np.where(w, 9, ante[(k - C) * n_seeds:k * n_seeds].reshape(C, n_seeds))
            out.append({"v": val.mean(1), "se": val.std(1, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros(C), "blinds": b, "won": w,
                        "reach": np.stack([(furthest >= j + 1).mean(1) for j in range(9)], 1)})
        return out

    def values(self, venv, e, cands, n_seeds, salt0=1_000_000):
        return self.values_many(venv, [(e, cands)], n_seeds, salt0)[0]


# ------------------------------------------------------------------ the requirement curve
GROW_NATURAL = None


def _grow():
    global GROW_NATURAL
    if GROW_NATURAL is None:
        from .potential import GROW
        from .env import item_id
        GROW_NATURAL = {item_id("j_" + k): g for k, g in GROW.items()}
    return GROW_NATURAL


class GapReport:
    """The pass margin at each future blind for the build as it is. Needs the development engine's set_state (hypothetical ante / blind / counters, exact features remade)."""

    def __init__(self, judge, dev, deck="RED", stake="GOLD", cap=512, threads=0, c8=0.09):
        from .engine_env import CppVecEnv
        self.judge, self.dev, self.c8 = judge, dev, c8
        self.la = CppVecEnv(cap, deck, stake, seed0=1, threads=threads)
        self.la.set_marginal(1)
        self.cap = cap

    @torch.no_grad()
    def curves(self, venv, ids):
        """-> (log_ratio [len(ids), 24], p_pass [len(ids), 24]) at every blind index 0..23 (NaN for the ones already behind). Each scaler grows by its natural rate per round from now."""
        ids = [int(i) for i in ids]
        g = np.asarray(venv.obs["g"])
        jid = np.asarray(venv.obs["jid"])
        jf = np.asarray(venv.obs["jf"])
        out_r = np.full((len(ids), 24), np.nan)
        out_p = np.full((len(ids), 24), np.nan)
        G = _grow()
        for n, e in enumerate(ids):
            ante = int(round(g[e, 0] * 8))
            cur_blind = int(np.argmax(g[e, 1:4]))
            t0 = 3 * (ante - 1) + cur_blind
            ts = list(range(t0, 24))
            slots = list(range(len(ts)))
            self.la.copy_many(venv, [e] * len(ts), slots)
            counters = np.full((len(ts), 8), np.nan)
            cur = np.expm1(4.0 * jf[e, :, 11])                                            # the observation holds log1p(|counter|) / 4
            for k, t in enumerate(ts):
                for j in range(min(8, jid.shape[1])):
                    ent = G.get(int(jid[e, j]))
                    if ent is not None and jid[e, j] > 0:
                        start, rate, lo, hi = ent
                        counters[k, j] = min(max(cur[j] + rate * (t - t0), lo), hi)
            for k, t in enumerate(ts):
                self.la.set_state([k], t // 3 + 1, t % 3, counters[k:k + 1])
            tg = lambda a: torch.from_numpy(np.nan_to_num(np.asarray(a), nan=0.0, posinf=10.0, neginf=-10.0)).to(self.dev)
            gg, jj = tg(self.la.obs["g"])[:len(ts)].float(), tg(np.asarray(self.la.obs["jf"])[:len(ts), :, :F_JOKER_W]).float()
            ji, ci = torch.from_numpy(np.asarray(self.la.obs["jid"])[:len(ts)]).to(self.dev), torch.from_numpy(np.asarray(self.la.obs["cid"])[:len(ts)]).to(self.dev)
            lr = pp = None
            for mode in (0, 1):
                o = self.judge.predict(gg, ji, jj, ci, torch.full((len(ts),), mode, device=self.dev, dtype=torch.long))
                r_, p_ = o["log_ratio"].float(), o["p_pass"].float()
                lr = r_ if lr is None else torch.maximum(lr, r_)
                pp = p_ if pp is None else torch.maximum(pp, p_)
            out_r[n, t0:] = lr.cpu().numpy()
            out_p[n, t0:] = pp.cpu().numpy()
        return out_r, out_p

    def report(self, venv, e):
        """-> dict(margin [24], gap_round (first blind with margin < 1, or None), min_margin, survive (expected blinds survived from the product of the pass chances))."""
        lr, pp = self.curves(venv, [e])
        margin = 10.0 ** (lr[0] + self.c8)
        valid = ~np.isnan(margin)
        below = np.flatnonzero(valid & (margin < 1.0))
        p = np.where(valid, pp[0], 1.0)
        surv = float(np.cumprod(p[valid]).sum()) if valid.any() else 0.0
        return {"margin": margin, "gap_round": int(below[0]) if len(below) else None, "min_margin": float(np.nanmin(margin)) if valid.any() else float("nan"), "survive": surv}


def expected_survival(p_pass):
    """V from a pass-chance curve: the expected number of blinds survived = sum over the future rounds t of the product of the pass chances up to t. p_pass [T] (NaN: already behind us)."""
    p = np.asarray(p_pass, dtype=np.float64)
    p = p[~np.isnan(p)]
    return float(np.cumprod(p).sum()) if len(p) else 0.0
