"""AlphaZero-style training on the C++ engine: search improves the network, the network guides the search.

Strategic layer (decisions outside a round: shop, packs, blind select / skip), searched WITHOUT foresight:
    candidates  the network's top --m legal actions by prior, plus leave / select / skip when legal
    futures     each candidate is tried on --k copies whose unseen future is redrawn (VecEnv.reseed: shops, packs, bosses, tags, shuffles, rolls still to come); the same
                k redraws for every candidate (common random numbers), so candidates are compared on the same futures
    rollout     the copy plays on with the network (greedy) until the next blind select or the start of a round (at most --rollout steps), the leaf
    leaf value  V = (1 - beta) x V_net + beta x V_exact; V_exact = a fitted map of (blinds so far, log10 projected power at the ante-8 boss) (VecEnv.projected_power:
                scaling counters moved over the rounds left). beta goes from 1 to 0 over --beta-iters iterations, so the first games are guided by the exact measure
    target      improved policy over the candidates: softmax(prior logit + c x normalised Q); the action played is its argmax (sampled in the first --explore moves of a game)
Tactical layer (rounds), judged by the strategic value:
    options     play the round as it is, or first use a held consumable on one of its --t best target choices (by the network's prior)
    lines       for each option the exact round solver's best --p lines that clear the blind (it sees the draws: round play has foresight, a known limitation)
    choice      the option and line whose end state has the highest V (+ the win when the line wins the run); no clearing line: the run is lost
Training: the policy is fitted to the search's improved policy (cross entropy), the value to the game's outcome z = blinds beaten / 24 + 1 if won.

    python -m balatro_rl.azc --out checkpoints/azc_v1.pt --iters 40 --games 64
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import env as E
from .engine_env import CppVecEnv, OBS_KEYS
from .model import ActorCritic, load_model, save_model

KEYS = [k for k in OBS_KEYS if k != "priv"]
SKIP_COL = 80
THREADS = 0                                                     # engine threads per process (0: all cores); a worker uses its share
from .sim.jokers import JOKERS as _JK
JOKER_IDS = {E.item_id("j_" + k) for k in _JK}
from .sim.items import PLANETS as _PL, VOCAB as _VOC
PLANET_IDS = {E.item_id("c_" + k) for k in _PL}
CELESTIAL_IDS = {E.item_id(k) for k in _VOC if k.startswith("p_celestial")}                                                   # obs g column: blinds skipped / 5 (env.encode_global)
DENSE = ("g", "cards", "jid", "jf", "cid", "subs")


# ------------------------------------------------------------------------------------------------ observations
NARROW = {}           # e.g. {"af": np.float16, "subs": np.int8, "aid": np.int16, "aid2": np.int16}: no clear gain measured (the conversion costs CPU), so off


def obs_batch(v: CppVecEnv, idx, dev):
    out = {}
    for k in KEYS:
        x = np.asarray(v.obs[k])[idx]
        if k in NARROW:
            x = x.astype(NARROW[k])
        out[k] = torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True)
    return out


class Buffer:
    """Training samples: the observation (per-action arrays kept for legal actions only), a policy target over actions (or none), and the outcome z (filled at game end)."""

    def __init__(self, cap=200_000):
        self.cap, self.rows = cap, []

    def add(self, v: CppVecEnv, i: int, pi: dict | None, game_key):
        mask = np.asarray(v.obs["mask"])[i]
        idx = np.flatnonzero(mask)
        row = {k: np.asarray(v.obs[k])[i].copy() for k in DENSE}
        row["cards"] = row["cards"].astype(np.float16)
        row["subs"] = row["subs"].astype(np.int8)
        row.update(idx=idx.astype(np.int16), af=np.asarray(v.obs["af"])[i][idx].astype(np.float16),
                   aid=np.asarray(v.obs["aid"])[i][idx].astype(np.int32), aid2=np.asarray(v.obs["aid2"])[i][idx].astype(np.int32),
                   pi=pi, game=game_key, z=None)
        self.rows.append(row)
        return row

    def finish(self, game_key, z):
        for r in reversed(self.rows):
            if r["game"] == game_key and r["z"] is None:
                r["z"] = z

    def ready(self):
        return [r for r in self.rows if r["z"] is not None]

    def trim(self):
        if len(self.rows) > self.cap:
            self.rows = self.rows[-self.cap:]

    @staticmethod
    def batch(rows, template, dev):
        b = len(rows)
        out = {}
        for k, (shape, dt) in template.items():
            if k in DENSE:
                out[k] = torch.from_numpy(np.stack([r[k] for r in rows]).astype(dt)).to(dev)
            else:
                out[k] = torch.zeros((b,) + shape, dtype=getattr(torch, str(np.dtype(dt))), device=dev)
        for j, r in enumerate(rows):
            ii = torch.from_numpy(r["idx"].astype(np.int64)).to(dev)
            out["mask"][j, ii] = True
            w = min(r["af"].shape[1], out["af"].shape[2])
            out["af"][j, ii, :w] = torch.from_numpy(r["af"][:, :w].astype(np.float32)).to(dev).to(out["af"].dtype)
            out["aid"][j, ii] = torch.from_numpy(r["aid"].astype(np.int64)).to(dev)
            out["aid2"][j, ii] = torch.from_numpy(r["aid2"].astype(np.int64)).to(dev)
        return out


# ------------------------------------------------------------------------------------------------ the value used by the search
class Valuer:
    """V(s) = (1 - beta) V_net + beta V_exact; V_exact = coef . [1, blinds/24, log10 projected power], refitted to outcomes each iteration."""

    def __init__(self, model, dev):
        self.model, self.dev = model, dev
        self.coef = np.array([0.0, 1.0, 0.12, 0.1])     # [1, blinds beaten / 24, log10 projected power, min(money, 50) / 50]: fixed unless --refit
        self.refit_on = False
        self.pp_samples = 8
        self.beta = 1.0

    def exact_feats(self, v: CppVecEnv, idx):
        g = np.asarray(v.obs["g"])[idx]
        pos = 3 * (np.rint(g[:, 0] * 8) - 1) + np.argmax(g[:, 1:4], 1)
        skipped = np.rint(g[:, SKIP_COL] * 5)                       # (a skipped blind is not a beaten one)
        blinds = (pos - skipped).clip(0, 24) / 24.0
        pp = v.projected_power(list(idx), self.pp_samples)
        money = np.clip(np.rint(g[:, 17] * 50), 0, 50) / 50.0
        lp = np.clip(np.log10(np.maximum(pp, 1e-4)), -4.0, 0.5) + 4.0            # >= 0: a live state is never worth less than the same blinds with the run lost
        return np.stack([np.ones(len(idx)), blinds, lp, money], 1)

    @torch.no_grad()
    def __call__(self, v: CppVecEnv, idx):
        if len(idx) == 0:
            return np.zeros(0)
        fe = self.exact_feats(v, idx)
        ve = fe @ self.coef
        if self.beta >= 1.0:
            return ve
        vn = []
        for c0 in range(0, len(idx), 1024):
            sub = list(idx[c0:c0 + 1024])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.dev).startswith("cuda")):
                _, val = self.model(obs_batch(v, sub, self.dev))
            vn.append(val.float().cpu().numpy())
        vn = np.concatenate(vn)
        return (1 - self.beta) * vn + self.beta * ve

    def refit(self, feats, z):
        if self.refit_on and len(z) > 50:
            self.coef, *_ = np.linalg.lstsq(feats, z, rcond=None)


# ------------------------------------------------------------------------------------------------ the agent
class AZ:
    def __init__(self, a, model, dev):
        self.a, self.model, self.dev = a, model, dev
        self.valuer = Valuer(model, dev)
        self.cap = a.games * (a.m + 3) * a.k
        self.scr = self._env(self.cap)
        self.rs = self._env(max(64, a.games * (1 + 3 * a.t) * a.p))       # round layer copies
        self.rng = np.random.default_rng(a.seed)
        self.salt = 1

    def _env(self, n):
        v = CppVecEnv(n, "RED", self.a.stake, seed0=1, threads=THREADS)
        v.set_exact(True)
        v.v.restart()
        v.set_auto_round(False, 1.5, 0)
        return v

    @torch.no_grad()
    def logits(self, v, idx):
        out = []
        for c0 in range(0, len(idx), 1024):
            sub = list(idx[c0:c0 + 1024])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.dev).startswith("cuda")):
                lg, _ = self.model(obs_batch(v, sub, self.dev))
            out.append(lg.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, E.N_ACTIONS))

    # -------------------------------------------------------------------------------- strategic search
    def _evaluate(self, real: CppVecEnv, jobs, q):
        """jobs: (game, candidate action, future index); each is played on a copy whose unseen future is redrawn (the same redraw for a given game and future index),
        then rolled on to its leaf; the leaf values are appended to q[(game, candidate)]."""
        a = self.a
        for c0 in range(0, len(jobs), self.cap):
            ch = jobs[c0:c0 + self.cap]
            slots = list(range(len(ch)))
            self.scr.copy_many(real, [gi for gi, _, _ in ch], slots)
            salts = [int((self.salt * 1_000_003 + gi * 7919 + k * 104729) % (2**40)) + 1 for gi, _, k in ch]
            if not getattr(a, "foresight", False):              # foresight (a known seed): the copies keep the true future
                self.scr.v.reseed(slots, salts)
            act = np.full(self.scr.n, -1, dtype=np.int64)
            for s, (_, c, _) in zip(slots, ch):
                act[s] = c
            _, d, info = self.scr.step(act)
            alive = [s for s in slots if not d[s]]
            ended = {s: info["blinds"][s] / 24.0 + (1.0 if info["won"][s] else 0.0) for s in slots if d[s]}
            queue = {}
            passes = {s: 0 for s in slots}
            # roll on through --horizon blind selects: shops and packs by the network (greedy), a round by the exact solver's best line (no line: the run is lost);
            # the leaf is the --horizon-th blind select reached
            for _ in range(a.rollout * a.horizon):
                m2 = np.asarray(self.scr.obs["mask"])
                g2 = np.asarray(self.scr.obs["g"])
                at_sel, force_sel = [], []
                for s in alive:
                    if m2[s, E.A_SELECT] and s not in queue:
                        (at_sel if passes[s] >= a.horizon - 1 else force_sel).append(s)
                go = [s for s in alive if s not in at_sel]
                if not go:
                    break
                fresh = [s for s in go if s not in queue and g2[s, 5] > 0.5 and g2[s, 9] == 0 and abs(g2[s, 11] - g2[s, 13]) < 1e-6]
                if fresh:
                    for s, pl in zip(fresh, self.scr.solve_rounds(fresh, 1, getattr(a, "rollout_node_cap", a.node_cap))):
                        queue[s] = list(pl[0]) if pl else None
                act = np.full(self.scr.n, -1, dtype=np.int64)
                net = []
                for s in force_sel:                                    # a blind select before the horizon: play it (the round follows)
                    act[s] = E.A_SELECT
                    passes[s] += 1
                for s in go:
                    if s in force_sel:
                        continue
                    if s in queue:
                        pl = queue[s]
                        if pl is None or (g2[s, 5] < 0.5) or not pl:
                            if pl is None:                             # no clearing line: the copy loses here
                                ended[s] = None
                            del queue[s]
                            if pl is None:
                                continue
                            if g2[s, 5] < 0.5 or not pl:
                                net.append(s)
                                continue
                        else:
                            act[s] = pl.pop(0)
                            if not m2[s, act[s]]:
                                act[s] = -1
                                del queue[s]
                                net.append(s)
                            continue
                    else:
                        net.append(s)
                net = [s for s in net if ended.get(s, 0) is not None or s not in ended]
                if net:
                    l2 = self.logits(self.scr, net)
                    act[net] = np.where(m2[net], l2, -np.inf).argmax(1)          # (masked argmax, all at once)
                lost_now = [s for s in go if s in ended and ended[s] is None]
                for s in lost_now:
                    ended[s] = (3 * (np.rint(g2[s, 0] * 8) - 1) + np.argmax(g2[s, 1:4]) - np.rint(g2[s, SKIP_COL] * 5)) / 24.0
                    alive.remove(s)
                _, d, info = self.scr.step(act)
                for s in go:
                    if s in alive and d[s]:
                        alive.remove(s)
                        ended[s] = info["blinds"][s] / 24.0 + (1.0 if info["won"][s] else 0.0)
            vals = np.zeros(len(slots))
            live = [s for s in slots if s not in ended]
            if live:
                vals[live] = self.valuer(self.scr, live)
            for s, z in ended.items():
                vals[s] = z
            for s, (gi, c, k) in zip(slots, ch):
                q.setdefault((gi, c), []).append(vals[s])

    def search(self, real: CppVecEnv, games, explore):
        """games: indices of real games at a non-round decision. -> {game: (action, improved policy dict)}."""
        a = self.a
        lg = self.logits(real, games)
        mask = np.asarray(real.obs["mask"])
        cands = {}
        for j, gi in enumerate(games):
            legal = np.flatnonzero(mask[gi])
            top = list(legal[np.argsort(-lg[j][legal])][:a.m])
            for forced in (E.A_LEAVE, E.A_SELECT, E.A_SKIP):
                if mask[gi, forced] and forced not in top:
                    top.append(forced)
            # every joker on offer is tried: each buy and each pack pick of a joker, and for each offered joker the swap the network rates best
            aid = np.asarray(real.obs["aid"])[gi]
            best_rep = {}
            for x in legal:
                if int(aid[x]) not in JOKER_IDS:
                    continue
                if E.A_BUY <= x < E.A_BUY + E.MAX_SHOP or E.A_PICK <= x < E.A_PSKIP:
                    if x not in top:
                        top.append(int(x))
                elif x >= E.A_REPLACE:
                    s_ = (x - E.A_REPLACE) // E.MAX_JOKERS
                    if s_ not in best_rep or lg[j][x] > lg[j][best_rep[s_]]:
                        best_rep[s_] = int(x)
            for x in best_rep.values():
                if x not in top:
                    top.append(x)
            # and every planet: buying one, a celestial pack, using a held one (hand levels against scalers)
            cid = np.asarray(real.obs["cid"])[gi]
            for x in legal:
                it_ = int(aid[x])
                if (E.A_BUY <= x < E.A_BUY + E.MAX_SHOP and it_ in PLANET_IDS) or (E.A_BUY_PACK <= x < E.A_VOUCHER and it_ in CELESTIAL_IDS)                         or (E.A_PICK <= x < E.A_PSKIP and it_ in PLANET_IDS):
                    if x not in top:
                        top.append(int(x))
            for ci in range(E.MAX_CONSUMABLES):
                u = E.A_USE_C + ci
                if mask[gi, u] and int(cid[ci]) in PLANET_IDS and u not in top:
                    top.append(u)
            cands[gi] = top
        # sequential halving: every candidate on the first --k-first futures, then only the better half (at least 2) on the rest, up to --k in all
        q = {}
        k1 = min(a.k, max(1, a.k_first))
        self._evaluate(real, [(gi, c, k) for gi in games for c in cands[gi] for k in range(k1)], q)
        if a.k > k1:
            jobs2 = []
            for gi in games:
                cs = cands[gi]
                if len(cs) <= 1:
                    continue
                mq = np.array([np.mean(q[(gi, c)]) for c in cs])
                keep = max(2, (len(cs) + 1) // 2)
                for c in [cs[i] for i in np.argsort(-mq)[:keep]]:
                    jobs2 += [(gi, c, k) for k in range(k1, a.k)]
            if jobs2:
                self._evaluate(real, jobs2, q)
        self.salt += 1
        out = {}
        for j, gi in enumerate(games):
            cs = cands[gi]
            qv = np.array([np.mean(q[(gi, c)]) for c in cs])
            qn = (qv - qv.mean()) / (qv.std() + 1e-6)
            pl = np.array([lg[j][c] for c in cs])
            pl = pl - pl.max()
            imp = np.exp(pl + a.c_q * qn)
            imp = imp / imp.sum()
            if explore:
                pick = cs[int(self.rng.choice(len(cs), p=imp))]
            else:
                pick = cs[int(np.argmax(imp))]
            out[gi] = (int(pick), {int(c): float(p) for c, p in zip(cs, imp)})
        return out

    # -------------------------------------------------------------------------------- tactical layer
    def round_plan(self, real: CppVecEnv, games):
        """games at the first decision of a round. -> {game: list of actions to play (option actions, then the line)}; [] when nothing clears (the run is lost)."""
        a = self.a
        mask = np.asarray(real.obs["mask"])
        opts = []                                                 # (game, prefix actions)
        for gi in games:
            opts.append((gi, []))
            for ci in range(E.MAX_CONSUMABLES):
                u = E.A_USE_C + ci
                if mask[gi, u]:
                    opts.append((gi, [u]))                         # the targets, if it needs any, are chosen below
        # apply the option prefixes on copies; consumables that need targets: the network's best --t target sets
        expanded = []
        for c0 in range(0, len(opts), self.rs.n):
            ch = opts[c0:c0 + self.rs.n]
            slots = list(range(len(ch)))
            self.rs.copy_many(real, [gi for gi, _ in ch], slots)
            act = np.full(self.rs.n, -1, dtype=np.int64)
            for s, (_, pre) in zip(slots, ch):
                if pre:
                    act[s] = pre[0]
            self.rs.step(act)
            m2 = np.asarray(self.rs.obs["mask"])
            tf = self.rs.v.targeting_flags(slots)
            needs = [s for s, (_, pre) in zip(slots, ch) if pre and tf[s]]
            l2 = self.logits(self.rs, needs) if needs else None
            for s, (gi, pre) in zip(slots, ch):
                if pre and s in needs:
                    legal = np.flatnonzero(m2[s, :E.A_SELECT])
                    best = legal[np.argsort(-l2[needs.index(s)][legal])][:a.t]
                    for tgt in best:
                        expanded.append((gi, pre + [int(tgt)]))
                else:
                    expanded.append((gi, pre))
        # for each option: the solver's lines, each line's end state valued
        best = {gi: (-1e9, []) for gi in games}
        for c0 in range(0, len(expanded), max(1, self.rs.n // a.p)):
            ch = expanded[c0:c0 + max(1, self.rs.n // a.p)]
            slots = list(range(len(ch)))
            self.rs.copy_many(real, [gi for gi, _ in ch], slots)
            for step_i in range(2):                                 # replay the option's actions (use, target)
                act = np.full(self.rs.n, -1, dtype=np.int64)
                for s, (_, pre) in zip(slots, ch):
                    if step_i < len(pre):
                        act[s] = pre[step_i]
                if (act >= 0).any():
                    self.rs.step(act)
            plans = self.rs.solve_rounds(slots, a.p, a.node_cap)
            jobs = [(j, pl) for j, pls in enumerate(plans) for pl in pls]
            if not jobs:
                continue
            # play each line on a fresh copy of the option's state
            src = CppVecEnv(len(ch), "RED", self.a.stake, seed0=1, threads=THREADS)
            src.set_exact(True)
            src.v.restart()
            src.copy_many(self.rs, slots, slots)
            n_jobs = len(jobs)
            line_env = self.scr if n_jobs <= self.scr.n else CppVecEnv(n_jobs, "RED", self.a.stake, seed0=1, threads=THREADS)
            line_env.copy_many(src, [j for j, _ in jobs], list(range(n_jobs)))
            pos = [0] * n_jobs
            done_at = {}
            won = {}
            while True:
                act = np.full(line_env.n, -1, dtype=np.int64)
                go = False
                for q_, (j, pl) in enumerate(jobs):
                    if q_ in done_at or pos[q_] >= len(pl):
                        continue
                    act[q_] = pl[pos[q_]]
                    pos[q_] += 1
                    go = True
                if not go:
                    break
                _, d, info = line_env.step(act)
                for q_ in range(n_jobs):
                    if act[q_] >= 0 and d[q_] and q_ not in done_at:
                        done_at[q_] = True
                        won[q_] = bool(info["won"][q_])
            live = [q_ for q_ in range(n_jobs) if q_ not in done_at]
            vals = np.zeros(n_jobs)
            if live:
                vals[live] = self.valuer(line_env, live)
            for q_ in done_at:
                vals[q_] = 2.0 if won.get(q_) else 0.0
            for q_, (j, pl) in enumerate(jobs):
                gi, pre = ch[j]
                if vals[q_] > best[gi][0]:
                    best[gi] = (vals[q_], pre + list(pl))
        return {gi: best[gi][1] for gi in games}


# ------------------------------------------------------------------------------------------------ self-play and training
class Snapshots:
    """Mid-run starts: game states saved at the first blind select of the late antes (one pool per ante, the last pool for that ante and later),
    kept by reservoir sampling, so that some self-play games can start from them (with their unseen future redrawn) and the late game gets practised."""

    def __init__(self, az: "AZ", antes, cap: int, seed: int):
        self.antes = sorted(antes)
        self.cap = cap
        self.pools = {an: az._env(cap) for an in self.antes}
        self.count = {an: 0 for an in self.antes}
        self.seen = {an: 0 for an in self.antes}
        self.rng = np.random.default_rng(seed)

    def bucket(self, ante: int):
        b = [an for an in self.antes if an <= ante]
        return b[-1] if b else None

    def offer(self, real: CppVecEnv, i: int, ante: int):
        b = self.bucket(ante)
        if b is None:
            return
        self.seen[b] += 1
        if self.count[b] < self.cap:
            slot = self.count[b]
            self.count[b] += 1
        else:
            slot = int(self.rng.integers(self.seen[b]))
            if slot >= self.cap:
                return
        self.pools[b].copy_many(real, [i], [slot])

    def ready(self) -> bool:
        return any(self.count.values())

    def sample(self, n: int):
        """n (pool, slot) picks: a pool uniformly among the filled ones (the deeper antes are rarer, so they get more than their share), then a slot."""
        full = [an for an in self.antes if self.count[an]]
        out = []
        for _ in range(n):
            an = full[int(self.rng.integers(len(full)))]
            out.append((self.pools[an], int(self.rng.integers(self.count[an]))))
        return out


def play(az: AZ, n_games, seed0, train: bool, buf: Buffer | None, feats_log, policy_only=False, recorder=None, snap: Snapshots | None = None, snap_frac=0.0):
    """n_games games in parallel; -> (blinds [n], won [n]). With snap: the late-ante states reached are offered to it, and a snap_frac share of the games
    start from its saved states (az.from_snap marks them)."""
    a = az.a
    real = CppVecEnv(n_games, "RED", a.stake, seed0=seed0, threads=THREADS)
    real.set_exact(True)
    real.v.restart()
    real.set_auto_round(False, 1.5, 0)
    from_snap = np.zeros(n_games, dtype=bool)
    if snap is not None and snap_frac > 0 and snap.ready():
        n_s = int(round(snap_frac * n_games))
        for j, (pool, slot) in enumerate(snap.sample(n_s)):
            real.copy_many(pool, [slot], [j])
        real.v.reseed(list(range(n_s)), [int((seed0 * 1_000_003 + j * 7919) % (2**40)) + 1 for j in range(n_s)])
        from_snap[:n_s] = True
    az.from_snap = from_snap
    last_ante = np.zeros(n_games, dtype=int)
    done = np.zeros(n_games, dtype=bool)
    blinds = np.zeros(n_games, dtype=int)
    won = np.zeros(n_games, dtype=bool)
    moves = np.zeros(n_games, dtype=int)
    queue = {i: [] for i in range(n_games)}                       # actions of a round plan still to play
    keys = {i: f"{seed0}:{i}" for i in range(n_games)}
    for _ in range(4000):
        if done.all():
            break
        mask, g = np.asarray(real.obs["mask"]), np.asarray(real.obs["g"])
        if recorder is not None:
            recorder(real, done)
        if snap is not None:
            at_sel = [i for i in range(n_games) if not done[i] and mask[i, E.A_SELECT]]
            if at_sel:
                for i, inf in zip(at_sel, real.cell_info(at_sel)):
                    if inf[0] > last_ante[i]:                      # the first select of a new ante
                        last_ante[i] = inf[0]
                        snap.offer(real, i, int(inf[0]))
        act = np.full(n_games, -1, dtype=np.int64)
        decide, rounds, other = [], [], []
        for i in range(n_games):
            if done[i]:
                continue
            if queue[i] and not (g[i, 5] > 0.5):                       # the round is over (the line goes on past the blind): drop the rest
                queue[i] = []
            if queue[i]:
                nxt = queue[i].pop(0)
                act[i] = nxt if mask[i, nxt] else -1
                if act[i] < 0:
                    queue[i] = []
                continue
            in_round = mask[i, :E.A_SELECT].any() and not mask[i, E.A_LEAVE] and not mask[i, E.A_PSKIP]
            fresh_round = in_round and g[i, 5] > 0.5 and g[i, 9] == 0 and abs(g[i, 11] - g[i, 13]) < 1e-6
            if fresh_round:
                rounds.append(i)
            elif in_round:
                other.append(i)                                    # mid-round or targeting: the network
            else:
                decide.append(i)
        forced = [i for i in decide if mask[i].sum() == 1]               # one legal action: nothing to search
        for i in forced:
            act[i] = int(np.flatnonzero(mask[i])[0])
        decide = [i for i in decide if i not in forced]
        if decide:
            if policy_only:
                lg = az.logits(real, decide)
                for j, i in enumerate(decide):
                    legal = np.flatnonzero(mask[i])
                    act[i] = legal[np.argmax(lg[j][legal])]
            else:
                res = az.search(real, decide, explore=train)
                fe = az.valuer.exact_feats(real, decide) if buf is not None else None
                for j, i in enumerate(decide):
                    act[i], pi = res[i]
                    if buf is not None:
                        row = buf.add(real, i, pi, keys[i])
                        row["feats"] = fe[j]
                    moves[i] += 1
        if rounds:
            if buf is not None:
                for i in rounds:
                    buf.add(real, i, None, keys[i])
            plans = az.round_plan(real, rounds)
            for i in rounds:
                pl = plans[i]
                if not pl:                                         # nothing clears: play the network's choice (the run will end)
                    other.append(i)
                    continue
                act[i] = pl[0]
                queue[i] = list(pl[1:])
        if other:
            lg = az.logits(real, other)
            act[other] = np.where(mask[other], lg, -np.inf).argmax(1)
        _, d, info = real.step(act)
        for i in np.flatnonzero(d):
            if not done[i]:
                done[i] = True
                blinds[i], won[i] = info["blinds"][i], bool(info["won"][i])
                if buf is not None:
                    buf.finish(keys[i], blinds[i] / 24.0 + (1.0 if won[i] else 0.0))
    return blinds, won


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _worker(wid, args_dict, task_q, result_q):
    """A self-play worker (its own process: its own network copy, engine pools and Python loop). Each task: the current weights (a file), the search's settings
    (beta, the exact value's weights), how many games and their seeds; it plays them with the search and sends back the samples and the results."""
    global THREADS
    a = argparse.Namespace(**args_dict)
    THREADS = a.worker_threads
    dev = resolve_device(a.worker_device or a.device)
    torch.set_num_threads(a.worker_torch_threads if dev == "cpu" else 1)      # (no GPU: the network runs on this worker's own cores)
    model, az = None, None
    while True:
        task = task_q.get()
        if task is None:
            break
        ck = torch.load(task["weights"], map_location=dev, weights_only=False)
        if model is None:
            model = ActorCritic(**ck["config"]).to(dev)
        model.load_state_dict(ck["state_dict"], strict=False)
        model.eval()
        if az is None:
            az = AZ(a, model, dev)
            snap = Snapshots(az, [int(x) for x in a.snap_antes.split(",")], a.snap_cap, 77 + wid) if a.snap_frac > 0 else None
        az.model = az.valuer.model = model
        az.valuer.beta = task["beta"]
        az.valuer.coef = np.array(task["coef"])
        az.valuer.pp_samples = a.pp_samples
        buf = Buffer(cap=10**9)
        ts = time.time()
        bl, wn = play(az, task["games"], task["seed0"], True, buf, None, snap=snap, snap_frac=a.snap_frac)
        result_q.put({"wid": wid, "rows": buf.ready(), "blinds": bl, "won": wn, "from_snap": az.from_snap, "play_s": time.time() - ts})


class Workers:
    """The learner's side: --workers processes kept for the whole run."""

    def __init__(self, a):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        n = a.workers
        args = dict(vars(a))
        args["games"] = -(-a.games // n)                      # each worker's own share sizes its pools (not the whole iteration's games)
        cores = a.cores or os.cpu_count() or 8
        dev = resolve_device(a.worker_device or a.device)
        args["worker_torch_threads"] = a.worker_torch_threads if a.worker_torch_threads > 0 else (max(1, cores // n // 4) if dev == "cpu" else 1)
        args["worker_threads"] = max(1, cores // n - (args["worker_torch_threads"] if dev == "cpu" else 0))
        self.task_qs = [ctx.Queue() for _ in range(n)]
        self.result_q = ctx.Queue()
        self.procs = [ctx.Process(target=_worker, args=(w, args, self.task_qs[w], self.result_q), daemon=True) for w in range(n)]
        for pr in self.procs:
            pr.start()
        self.weights = a.out.replace(".pt", "_workers.pt")
        self.a = a

    def play(self, model, valuer, it):
        a = self.a
        save_model(model, self.weights + ".tmp", {"mode": "azc-workers"})
        os.replace(self.weights + ".tmp", self.weights)
        n = len(self.procs)
        per = [a.games // n + (1 if w < a.games % n else 0) for w in range(n)]
        for w in range(n):
            if per[w]:
                self.task_qs[w].put({"weights": self.weights, "beta": valuer.beta, "coef": [float(c) for c in valuer.coef], "games": per[w],
                                     "seed0": 5_000_000 + it * 10_007 + w * 1_000})
        rows, bls, wns, fss, ps = [], [], [], [], []
        for _ in range(sum(1 for x in per if x)):
            r = self.result_q.get()
            rows += r["rows"]
            bls.append(r["blinds"])
            wns.append(r["won"])
            fss.append(r["from_snap"])
            ps.append(r["play_s"])
        return rows, np.concatenate(bls), np.concatenate(wns), np.concatenate(fss), max(ps)

    def close(self):
        for q in self.task_qs:
            q.put(None)


def train_net(model, opt, buf: Buffer, template, dev, epochs, batch, vf, n_batches=0):
    rows = buf.ready()
    if not rows:
        return {}
    rng = np.random.default_rng()
    pl_, vl_ = [], []
    model.train()
    if n_batches > 0:                                       # a fixed amount of training per iteration: n_batches random minibatches from the window
        plan = [rng.choice(len(rows), size=min(batch, len(rows)), replace=False) for _ in range(n_batches)]
    else:                                                   # --epochs passes over everything in the window
        plan = []
        for _ in range(epochs):
            order = rng.permutation(len(rows))
            plan += [order[c0:c0 + batch] for c0 in range(0, len(order), batch)]
    for idx_ in plan:
        if True:
            mb = [rows[i] for i in idx_]
            ob = Buffer.batch(mb, template, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
                lg, v = model(ob)
            lg, v = lg.float(), v.float()
            z = torch.tensor([r["z"] for r in mb], dtype=torch.float32, device=dev)
            vloss = F.mse_loss(v, z)
            has = [j for j, r in enumerate(mb) if r["pi"]]
            ploss = torch.zeros((), device=dev)
            if has:
                tgt = torch.zeros((len(has), lg.shape[1]), device=dev)
                for t_, j in enumerate(has):
                    for c, p in mb[j]["pi"].items():
                        tgt[t_, c] = p
                logp = torch.log_softmax(lg[has].masked_fill(~ob["mask"][has], -1e9), 1)
                ploss = -(tgt * logp).sum(1).mean()
            loss = ploss + vf * vloss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pl_.append(float(ploss))
            vl_.append(float(vloss))
    model.eval()
    return {"policy_loss": round(float(np.mean(pl_)), 4), "value_loss": round(float(np.mean(vl_)), 4), "samples": len(rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="checkpoints/azc_v1.pt")
    p.add_argument("--init", default="")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--m", type=int, default=4, help="candidate actions searched per decision (plus leave / select / skip)")
    p.add_argument("--k", type=int, default=6, help="redrawn futures per candidate (finalists)")
    p.add_argument("--k-first", type=int, default=6, help="futures every candidate gets before the worse half is dropped")
    p.add_argument("--rollout", type=int, default=25, help="network steps per blind in a rollout")
    p.add_argument("--horizon", type=int, default=2, help="blind selects a rollout plays through before its leaf")
    p.add_argument("--pp-samples", type=int, default=8, help="hands sampled by the projected power measure")
    p.add_argument("--p", type=int, default=3, help="solver lines per round option")
    p.add_argument("--t", type=int, default=3, help="target choices per consumable in a round")
    p.add_argument("--node-cap", type=int, default=150)
    p.add_argument("--rollout-node-cap", type=int, default=150, help="the solver's node cap for rounds inside the search's rollouts (--node-cap: the rounds actually played)")
    p.add_argument("--c-q", type=float, default=2.0, help="weight of the normalised search value in the improved policy")
    p.add_argument("--beta-iters", type=int, default=10, help="iterations over which the exact value gives way to the network's")
    p.add_argument("--epochs", type=int, default=3, help="passes over the window per iteration (when --train-batches is 0)")
    p.add_argument("--buffer-cap", type=int, default=100_000, help="the sliding window: the most recent samples kept for training")
    p.add_argument("--train-batches", type=int, default=400, help="minibatches per iteration (0: --epochs passes over the window)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-iter", type=int, default=0, help="continue a run (with --init): the iteration count (beta schedule, log) starts after this")
    p.add_argument("--refit", action="store_true", help="refit the exact value's weights to outcomes each iteration (default: fixed)")
    p.add_argument("--device", default="auto", help="auto (the GPU if there is one) | cpu | cuda")
    p.add_argument("--worker-device", default="", help="the self-play workers' device (default: --device); cpu keeps many workers off one GPU while the learner trains on it")
    p.add_argument("--cores", type=int, default=0, help="CPU cores to use (0: all the machine reports; set it in a container that sees more cores than it was given)")
    p.add_argument("--worker-torch-threads", type=int, default=0, help="PyTorch threads per worker on CPU (0: a quarter of its cores)")
    p.add_argument("--workers", type=int, default=0, help="self-play worker processes (0: self-play in the learner)")
    p.add_argument("--beta-floor", type=float, default=0.0, help="the exact value's share in the search never falls below this")
    p.add_argument("--snap-frac", type=float, default=0.0, help="share of the self-play games that start from a saved late-ante state of earlier games (0: off)")
    p.add_argument("--snap-antes", default="5,6,7", help="the antes whose first blind select is saved (one pool each; the last also takes later antes)")
    p.add_argument("--snap-cap", type=int, default=48, help="states kept per ante pool (per worker)")
    p.add_argument("--pp-weight", type=float, default=0.12, help="the exact value's weight per decade of projected ante-8 power")
    a = p.parse_args()
    dev = resolve_device(a.device)
    if dev == "cpu":
        torch.set_num_threads(a.cores or os.cpu_count() or 8)   # the learner trains between self-play phases, when the workers are idle
    model = load_model(a.init, dev) if a.init else ActorCritic(attn=2, desc=1).to(dev)
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    # with workers the learner only plays the evaluations: size its pools for those
    az = AZ(argparse.Namespace(**{**vars(a), "games": a.eval_games}) if a.workers > 0 else a, model, dev)
    az.valuer.refit_on = a.refit
    az.valuer.coef[2] = a.pp_weight
    az.valuer.pp_samples = a.pp_samples
    buf = Buffer(cap=a.buffer_cap)
    workers = Workers(a) if a.workers > 0 else None
    snap = Snapshots(az, [int(x) for x in a.snap_antes.split(",")], a.snap_cap, 77) if a.snap_frac > 0 and workers is None else None
    t0 = time.time()
    log = open(a.out.replace(".pt", "_log.jsonl"), "a")
    tmpl_env = CppVecEnv(1, "RED", a.stake, seed0=1, threads=0)
    template = {k: (tuple(np.asarray(tmpl_env.obs[k]).shape[1:]), np.asarray(tmpl_env.obs[k]).dtype) for k in KEYS}
    for it in range(a.start_iter + 1, a.iters + 1):
        az.valuer.beta = max(a.beta_floor, 1.0 - (it - 1) / max(1, a.beta_iters))
        ts = time.time()
        if workers is not None:
            new_rows, bl, wn, fs, _ = workers.play(model, az.valuer, it)
            buf.rows += new_rows
        else:
            bl, wn = play(az, a.games, 5_000_000 + it * 10_007, True, buf, None, snap=snap, snap_frac=a.snap_frac)
            fs = az.from_snap
        t_play = time.time() - ts
        rows = [r for r in buf.ready() if r.get("feats") is not None][-20000:]
        if rows:                                              # refit the exact value's map to the outcomes seen
            az.valuer.refit(np.stack([r["feats"] for r in rows]), np.array([r["z"] for r in rows]))
        stats = train_net(model, opt, buf, template, dev, a.epochs, a.batch, a.vf, a.train_batches)
        buf.trim()
        fresh = ~fs                                           # the games played from the start (the mid-run starts are logged apart)
        row = {"iter": it, "beta": round(az.valuer.beta, 3), "exact_coef": [round(float(c), 3) for c in az.valuer.coef], "train_blinds": round(float(bl[fresh].mean()), 2), "train_wins": int(wn[fresh].sum()),
               "games": int(fresh.sum()), "play_s": round(t_play, 1), "min": round((time.time() - t0) / 60, 1), **stats}
        if fs.any():
            row.update(snap_games=int(fs.sum()), snap_blinds=round(float(bl[fs].mean()), 2), snap_wins=int(wn[fs].sum()))
        if a.eval_every and (it % a.eval_every == 0 or it == 1):
            eb, ew = play(az, a.eval_games, 10_000, False, None, None)
            pb, pw = play(az, a.eval_games, 10_000, False, None, None, policy_only=True)
            row.update(eval_search_blinds=round(float(eb.mean()), 2), eval_search_wins=int(ew.sum()), eval_net_blinds=round(float(pb.mean()), 2), eval_net_wins=int(pw.sum()))
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
        save_model(model, a.out, {"mode": "azc", "iter": it})


if __name__ == "__main__":
    main()
