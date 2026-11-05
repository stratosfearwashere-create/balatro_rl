"""One-step lookahead for the greedy evaluation (train.py --eval-lookahead K): decisions outside a round compare the outcomes of the moves available.

At each decision outside a round (shop, pack, blind select / skip), the policy's K most likely legal moves, plus leave / skip / select / pack-skip when legal, are
each applied to a copy of the game (VecEnv.copy_many); a move whose result is random (a reroll, a pack) is applied on a copy whose unseen future is redrawn
(VecEnv.reseed_rng), so the lookahead never sees the real draw. Each move is scored r + V(s'): the step's reward (a select with autoplay plays the whole round
with the solver) plus the critic's value of the position it leads to (0 when the game ends). The best-scoring move is played. Training never uses this.
"""
from __future__ import annotations

import numpy as np
import torch

from . import env as E
from .engine_env import OBS_KEYS

CHANCE = set(range(E.A_BUY_PACK, E.A_VOUCHER)) | {E.A_REROLL}
EXTRAS = (E.A_LEAVE, E.A_SKIP, E.A_SELECT, E.A_PSKIP)


class Lookahead:
    def __init__(self, make_env, fwd, dev, k=8, slots=2048, pot=None, no_overrule=(), scorer=None):
        # scorer(env, ids) -> values: a hand-built value (hv.HV) instead of the critic; a move is then scored blinds beaten by it + scorer(next position),
        # and moves into a pack (where the curve cannot be measured) are left to the policy
        self.scorer = scorer
        self.shop_only = True
        # action kinds the search may not switch to (it keeps them only when the policy itself picked one): "skip", "sell"
        blocked = set()
        if "skip" in no_overrule:
            blocked.add(E.A_SKIP)
        if "sell" in no_overrule:
            blocked.update(range(E.A_SELL_J, E.A_SELL_J + E.MAX_JOKERS))
        self.blocked = blocked
        self.pot = pot                                    # the trainer's per-step shaping potentials (shaping_pot.ShapingPot): the critic learned V - phi
        self.S = make_env(slots)
        self.fwd, self.dev, self.k = fwd, dev, int(k)
        self.salt = 0
        self.n_decisions = 0
        self.n_changed = 0

    def close(self):
        self.S.close()

    def act(self, ev, logits, fin):
        """-> actions [n] for the evaluation environments `ev`: greedy inside a round, the lookahead's choice outside one."""
        mask = np.asarray(ev.obs["mask"])
        lg = logits.float().cpu().numpy()
        greedy = lg.argmax(1)
        act = np.where(fin, -1, greedy)
        outr = np.flatnonzero(~mask[:, :E.A_SELECT].any(1) & (mask.sum(1) >= 2) & ~fin)
        if not len(outr):
            return act
        src, acts, own = [], [], []
        for i in outr:
            row = np.where(mask[i], lg[i], -np.inf)
            cands = [int(c) for c in np.argsort(-row)[:self.k] if mask[i, c]]
            for x in EXTRAS:
                if mask[i, x] and x not in cands:
                    cands.append(int(x))
            if self.blocked:
                cands = [c for c in cands if c not in self.blocked or c == int(greedy[i])]
            if self.scorer is not None and not (E.A_BUY_PACK <= int(greedy[i]) < E.A_VOUCHER or E.A_PICK <= int(greedy[i]) <= E.A_PSKIP):
                cands = [c for c in cands if not (E.A_BUY_PACK <= c < E.A_VOUCHER or E.A_PICK <= c <= E.A_PSKIP)]
            elif self.scorer is not None:
                cands = [int(greedy[i])]                                       # (a pack decision: the policy's)
            if self.scorer is not None and self.shop_only and mask[i, E.A_SELECT]:
                cands = [int(greedy[i])]                                       # (blind select / skip: the policy's; the hand-built value has no price for tags)
            for c in cands:
                src.append(int(i))
                acts.append(c)
                own.append(int(i))
        score = np.zeros(len(src))
        S = self.S
        for c0 in range(0, len(src), S.n):
            sl = slice(c0, min(len(src), c0 + S.n))
            m = sl.stop - sl.start
            S.copy_many(ev, src[sl], list(range(m)))
            ch = [j for j in range(m) if acts[c0 + j] in CHANCE]
            if ch:
                self.salt += 1
                S.reseed_rng(ch, [self.salt * 1_000_003 + j * 7919 + 11 for j in ch])
            a0 = np.full(S.n, -1, dtype=np.int64)
            a0[:m] = acts[sl]
            if self.scorer is not None:                                      # a blind is beaten by playing it: select (or the autoplay leave) at a blind select
                at_sel = np.array([mask[src[c0 + j], E.A_SELECT] for j in range(m)], dtype=bool)
                plays = at_sel & np.isin(np.asarray(acts[sl]), [E.A_SELECT, E.A_LEAVE])
            r, d, _ = S.step(a0)
            if self.scorer is not None:
                dn = np.asarray(d[:m], dtype=bool)
                b1, b0 = plays.astype(float), 0.0
                live = [j for j in range(m) if not dn[j]]
                val = np.zeros(m)
                if live:
                    val[live] = self.scorer(S, live)
                score[sl] = np.where(dn, 0.0, (b1 - b0) + val)
                continue
            o2 = {k: torch.from_numpy(np.asarray(S.obs[k])[:m]).to(self.dev) for k in OBS_KEYS if k in S.obs}
            with torch.no_grad():
                _, v2 = self.fwd(o2)
            v2 = v2.float()
            if v2.dim() > 1:
                v2 = v2.sum(-1)
            v2 = v2.cpu().numpy()
            ph = self.pot(o2).float().cpu().numpy() if self.pot is not None else 0.0
            score[sl] = np.asarray(r[:m], dtype=np.float64) + np.where(d[:m], 0.0, v2 + ph)
        own = np.asarray(own)
        acts = np.asarray(acts)
        for i in outr:
            js = np.flatnonzero(own == i)
            best = int(acts[js[int(np.argmax(score[js]))]])
            self.n_decisions += 1
            self.n_changed += int(best != greedy[i])
            act[i] = best
        return act
