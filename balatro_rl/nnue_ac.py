"""A Stockfish-shaped actor-critic for PPO (train.py --arch nnue): the state is a sparse set of active features summed into a 256-wide
accumulator (an embedding bag, like NNUE's first layer), clipped ReLU, then small dense layers; no attention, no transformer.

State features (built on the GPU from the ordinary observation, so the trainer and the environments are unchanged): the jokers held (which,
edition, counter bucket, rental, perishable), hand levels and how often each hand was played, the deck's make-up, vouchers, money, ante x blind x
phase, the boss, consumables, hands / discards / hand size / joker slots (as balatro_rl/nnue.py, without the seed's future list), plus what a
round needs: each card in hand (rank x suit, enhancement, edition, seal, debuffed), the hands and discards left, how much of the blind is done and
its size.

Policy: each legal action's row (its item's embedding, its action features, and for a play / discard the mean of its cards' small embeddings)
-> 64, combined with a 64-wide projection of the state -> a logit. Value: the state -> 32 -> 32 -> 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import F_CARD, F_ACT, VOCAB_SIZE, N_SUB, RATIO_COL
OFF_DISC = RATIO_COL + 5                                        # the discard rows' kept-card features (env.action_features)
from .nnue import L as NL, J0, C0, NJ, NC, MONEY_EDGES
from .runeval import Cols
from .sim.items import BOSS_KEYS

NBOSS = len(BOSS_KEYS)
# the state features: nnue.py's layout without the future list, then the round's
_BASE = NL.off["fut0"]
OFF = dict(NL.off)
OFF["card"] = _BASE
OFF["cenh"] = OFF["card"] + 52
OFF["ced"] = OFF["cenh"] + 9
OFF["cseal"] = OFF["ced"] + 5
OFF["cdeb"] = OFF["cseal"] + 5
OFF["hleft"] = OFF["cdeb"] + 1
OFF["dleft"] = OFF["hleft"] + 7
OFF["ratio"] = OFF["dleft"] + 8
OFF["target"] = OFF["ratio"] + 11
N_STATE = OFF["target"] + 16
PAD = N_STATE                                                     # the padding index (a zero row)


def _f(x):
    return x.float()


def topk_mask(ob, topk):
    """topk > 0: only the topk plays (by the engine's predicted score) and the topk discards (by what the kept cards build toward: the best
    suit count, rank count, straight window, pairs kept) stay legal; every other action is unchanged. A pure function of the observation, so
    acting, the update and any evaluation agree."""
    mask = ob["mask"]
    if not topk:
        return mask
    k = min(int(topk), N_SUB)
    af = ob["af"]
    mp = mask[:, :N_SUB]
    ps = af[:, :N_SUB, RATIO_COL + 1].float().masked_fill(~mp, -1e9)
    keep_p = mp & (ps >= ps.topk(k, dim=1).values[:, -1:])
    md = mask[:, N_SUB:2 * N_SUB]
    kf = af[:, N_SUB:2 * N_SUB, OFF_DISC:OFF_DISC + 6].float()
    pot = torch.maximum(torch.maximum(kf[..., 0], kf[..., 1] * 1.25), kf[..., 2]) + 0.25 * kf[..., 3] + 0.05 * (1.0 - kf[..., 5])
    pd = pot.masked_fill(~md, -1e9)
    keep_d = md & (pd >= pd.topk(k, dim=1).values[:, -1:])
    out = mask.clone()
    out[:, :N_SUB] = keep_p
    out[:, N_SUB:2 * N_SUB] = keep_d
    return out


class NnueActorCritic(nn.Module):
    pack_legal = True

    def __init__(self, width: int = 256, emb: int = 16, aw: int = 64, topk: int = 0, **_ignored):
        super().__init__()
        self.config = {"arch": "nnue", "width": width, "emb": emb, "aw": aw, "topk": int(topk)}
        self.topk = int(topk)
        self.acc = nn.Embedding(N_STATE + 1, width, padding_idx=PAD)
        nn.init.normal_(self.acc.weight, std=0.02)
        with torch.no_grad():
            self.acc.weight[PAD].zero_()
        self.acc_bias = nn.Parameter(torch.zeros(width))
        self.l1 = nn.Linear(width, 32)
        self.l2 = nn.Linear(32, 32)
        self.value_out = nn.Linear(32, 1)
        self.s_proj = nn.Linear(width, aw)
        self.item = nn.Embedding(VOCAB_SIZE, emb)
        self.card = nn.Linear(F_CARD, emb)
        self.act = nn.Sequential(nn.Linear(emb + F_ACT + emb, aw), nn.ReLU(), nn.Linear(aw, aw))
        self.head = nn.Sequential(nn.Linear(3 * aw, aw), nn.ReLU(), nn.Linear(aw, 1))
        self.value_detach = False
        self.streams = 0
        self.f_act = F_ACT
        self.register_buffer("money_edges", torch.tensor(MONEY_EDGES, dtype=torch.float32), persistent=False)
        self.register_buffer("pb", torch.tensor([0, 2, 5, 10, 20, 1e9], dtype=torch.float32), persistent=False)
        self.register_buffer("eb", torch.tensor([0, 1, 2, 4, 8, 1e9], dtype=torch.float32), persistent=False)

    # ------------------------------------------------------------------ the sparse state features, [B, K] indices (PAD = none)
    def features(self, ob):
        g = _f(ob["g"])
        B = g.shape[0]
        dev = g.device
        out = []
        pad = torch.full((B,), PAD, dtype=torch.long, device=dev)

        def cond(mask, idx):
            return torch.where(mask, idx.long(), pad if idx.dim() == 1 else torch.full_like(idx.long(), PAD))

        C = Cols
        ante = torch.clamp(torch.round(g[:, C.ANTE] * 8), 1, 8).long()
        blind = g[:, C.BLIND:C.BLIND + 3].argmax(1)
        state = g[:, C.STATE:C.STATE + 4].argmax(1)
        money = torch.where(g[:, C.MONEY] < 1.99, g[:, C.MONEY] * 50.0, torch.pow(10.0, 3.0 * g[:, C.LOG_MONEY]) - 1.0)
        money = torch.where(g[:, C.NEG] > 0.5, torch.full_like(money, -1.0), money)
        out.append(OFF["money"] + torch.clamp(torch.searchsorted(self.money_edges, money.contiguous()), max=15))
        out.append(OFF["phase"] + ((ante - 1) * 3 + blind) * 4 + state)
        lv = torch.clamp(torch.round(g[:, C.LEVELS:C.LEVELS + 12] * 10), 1, 10).long()
        h = torch.arange(12, device=dev)
        out.append(cond(lv > 1, OFF["lvl"] + h * 10 + lv - 1))
        played = g[:, C.PLAYED:C.PLAYED + 12] * 30.0
        out.append(cond(played > 0.5, OFF["played"] + h * 6 + torch.searchsorted(self.pb, played.contiguous())))
        ranks = torch.clamp(torch.round(g[:, C.RANKS:C.RANKS + 13] * 4), 0, 8).long()
        out.append(OFF["rank"] + torch.arange(13, device=dev) * 9 + ranks)
        suits = torch.clamp(torch.round(g[:, C.SUITS:C.SUITS + 4] * 13 / 3), 0, 11).long()
        out.append(OFF["suit"] + torch.arange(4, device=dev) * 12 + suits)
        enh = torch.round(g[:, C.ENH:C.ENH + 8] * 10)
        out.append(cond(enh > 0, OFF["enh"] + torch.arange(8, device=dev) * 6 + torch.clamp(torch.searchsorted(self.eb, enh.contiguous()), max=5)))
        seal = torch.round(g[:, C.SEAL:C.SEAL + 4] * 10)
        out.append(cond(seal > 0, OFF["seal"] + torch.arange(4, device=dev) * 6 + torch.clamp(torch.searchsorted(self.eb, seal.contiguous()), max=5)))
        full = torch.round(g[:, C.FULL] * 52)
        out.append((OFF["deck"] + torch.clamp(torch.floor((full - 28) / 4), 0, 11)).long().unsqueeze(1))
        vou = g[:, C.VOUCH:C.VOUCH + 32] > 0.5
        out.append(cond(vou, OFF["vouch"] + torch.arange(32, device=dev).expand(B, -1)))
        boss = g[:, C.BOSS:C.BOSS + NBOSS]
        out.append(cond(boss.max(1).values > 0.5, OFF["boss"] + boss.argmax(1)).unsqueeze(1))
        out.append((OFF["hands"] + torch.clamp(torch.round(g[:, C.ROUND_HANDS] * 5), 0, 6)).long().unsqueeze(1))
        out.append((OFF["disc"] + torch.clamp(torch.round(g[:, C.ROUND_DISC] * 5), 0, 7)).long().unsqueeze(1))
        out.append((OFF["hsize"] + torch.clamp(torch.round(g[:, C.HAND_SIZE] * 8) - 4, 0, 9)).long().unsqueeze(1))
        out.append((OFF["jslots"] + torch.clamp(torch.round(g[:, C.JSLOTS] * 7), 0, 7)).long().unsqueeze(1))
        # the round
        out.append((OFF["hleft"] + torch.clamp(torch.round(g[:, C.HANDS_LEFT] * 5), 0, 6)).long().unsqueeze(1))
        out.append((OFF["dleft"] + torch.clamp(torch.round(g[:, C.DISC_LEFT] * 5), 0, 7)).long().unsqueeze(1))
        out.append((OFF["ratio"] + torch.clamp(torch.floor(g[:, C.CHIPS_RATIO] * 5), 0, 10)).long().unsqueeze(1))
        out.append((OFF["target"] + torch.clamp(torch.floor(g[:, C.LOG_TARGET] * 6 * 2), 0, 15)).long().unsqueeze(1))
        # jokers
        jid = ob["jid"].long()
        jf = _f(ob["jf"])
        j = jid - J0
        ok = (j >= 0) & (j < NJ)
        jc = torch.clamp(j, 0, NJ - 1)
        out.append(cond(ok, OFF["j_id"] + jc))
        for e in range(4):
            out.append(cond(ok & (jf[..., 2 + e] > 0.5), OFF["j_ed"] + jc * 4 + e))
        cnt = jf[..., 11]
        out.append(cond(ok & (cnt > 0), OFF["j_cnt"] + jc * 8 + torch.clamp((cnt * 8).long(), 0, 7)))
        out.append(cond(ok & (jf[..., 8] > 0.5), OFF["j_rent"] + jc))
        out.append(cond(ok & (jf[..., 7] > 0.05), OFF["j_perish"] + jc))
        cid = ob["cid"].long() - C0
        out.append(cond((cid >= 0) & (cid < NC), OFF["cons"] + torch.clamp(cid, 0, NC - 1)))
        # cards in hand (F_CARD: present, rank 13, suit 4, enhancement 9, edition 5, seal 5, debuffed, forced, hidden)
        cards = _f(ob["cards"])
        present = cards[..., 0] > 0.5
        hidden = cards[..., 39] > 0.5
        r = cards[..., 1:14].argmax(-1)
        s_ = cards[..., 14:18].argmax(-1)
        has_rs = present & ~hidden & (cards[..., 1:14].sum(-1) > 0.5)
        out.append(cond(has_rs, OFF["card"] + r * 4 + s_))
        e9 = cards[..., 18:27].argmax(-1)
        out.append(cond(present & (e9 > 0), OFF["cenh"] + e9))
        ed = cards[..., 27:32].argmax(-1)
        out.append(cond(present & (ed > 0), OFF["ced"] + ed))
        se = cards[..., 32:37].argmax(-1)
        out.append(cond(present & (se > 0), OFF["cseal"] + se))
        out.append(cond(present & (cards[..., 37] > 0.5), torch.full_like(r, OFF["cdeb"])))
        return torch.cat([x if x.dim() == 2 else x.unsqueeze(1) for x in out], 1)

    def state(self, ob):
        idx = self.features(ob)
        a = F.embedding_bag(idx, self.acc.weight, mode="sum", padding_idx=PAD) + self.acc_bias      # (the sum without the [B, K, width] gather)
        return torch.clamp(a, 0.0, 1.0)                               # clipped ReLU

    def _value(self, s, ob=None):
        x = s.detach() if self.value_detach else s
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        return self.value_out(x).squeeze(-1)

    def _rows(self, ob, ce, b, a_idx):
        """The action rows for the (state, action) pairs b, a_idx: item embedding, action features, the mean of a play / discard's cards."""
        it = self.item(ob["aid"][b, a_idx].long())
        af = ob["af"][b, a_idx][:, :self.f_act].float()
        is_sub = a_idx < 2 * N_SUB
        srow = ob["subs"][b, torch.clamp(a_idx, max=2 * N_SUB - 1)].long()       # L, 5: the action's own row only
        valid = ((srow >= 0) & is_sub.unsqueeze(1)).float().unsqueeze(-1)
        picked = ce[b.unsqueeze(1).expand(-1, 5), srow.clamp(min=0)]   # L, 5, emb
        sub = (picked * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.act(torch.cat([it, af, sub], -1))

    def restrict(self, ob):
        return topk_mask(ob, self.topk)

    def forward(self, ob, aux: bool = False):
        s = self.state(ob)
        ce = self.card(_f(ob["cards"]))                              # B, cards, emb
        mask = self.restrict(ob)
        b, a_idx = mask.nonzero(as_tuple=True)
        rows = self._rows(ob, ce, b, a_idx)
        sp = self.s_proj(s)[b]
        lg = self.head(torch.cat([sp, rows, sp * rows], -1)).squeeze(-1)
        logits = torch.full(mask.shape, -1e9, device=lg.device, dtype=lg.dtype)
        logits[b, a_idx] = lg
        self._s, self._ce = s, ce
        v = self._value(s, ob)
        if aux:
            return logits, v, None
        return logits, v
