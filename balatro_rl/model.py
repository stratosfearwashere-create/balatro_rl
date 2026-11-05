"""Actor-critic network.

The policy scores every action with a shared network (a "pointer" style policy):
    logit(a) = f(state_embedding, action_embedding(a))
so the same weights handle playing cards, shopping and opening packs, and the net
generalises across the 466 actions instead of memorising slot positions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import (F_CARD, F_JOKER, F_ACT, F_JOKER_W, F_ACT_W, F_JOKER_P, F_ACT_P, VOCAB_SIZE, N_SUB, A_PLAY, A_DISC,
                  MAX_JOKERS, global_size)
from .marginal import MARG_G, MARG_J, POT_G

OBS_KEYS = ("g", "cards", "jid", "jf", "cid", "aid", "af", "mask", "subs")
OPTIONAL_KEYS = ("aid2",)          # passed when present (data recorded before the replace action lacks it)


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """A pre-norm residual MLP block (as in the RL network-scaling work, e.g. SimBa): x + W2 relu(W1 LayerNorm(x)). The second layer starts
    at zero, so a block added to a trained trunk leaves it unchanged."""

    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expand)
        self.fc2 = nn.Linear(dim * expand, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class AttnBlock(nn.Module):
    """One pre-norm self-attention layer. keep [B, L] marks the tokens that exist; the others are never
    attended to (their own outputs are ignored by the caller)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        x = x + self.out(a.transpose(1, 2).reshape(B, L, D))
        return x + self.ff(self.n2(x))


AUX_BLINDS = 6.0          # the unit of the blinds-this-game prediction (ActorCritic aux head 1)


class ActorCritic(nn.Module):
    value_detach = False      # train.py --value-warmup: the value head reads the state without a gradient into the encoder
    def __init__(self, g_dim: int | None = None, d: int = 256, emb: int = 32, attn: int = 0, desc: int = 0,
                 tok: int = 128, heads: int = 4, aux: int = 0, marg: int = 0, priv: int = 0, priv_actor: int = 0, streams: int = 0, spec: int = 0,
                 aux_play: int = 0, aux_loo: int = 0, qhead: int = 0, goal_in: int = 0, goal_out: int = 0, phi: int = 0,
                 aw: int = 128, vw: int = 128, res: int = 0, res_expand: int = 2, extra: int = 0, topk: int = 0, reach: int = 0):
        """attn: layers of self-attention over the jokers, the hand, the consumables and a global token (0: the
        jokers and the hand are each summarised by their mean and maximum, as every network before had it).
        desc: an item's embedding also gets a projection of what the item does (joker_desc.item_descriptors;
        any non-zero value: the table's width is stored). Both default to off, which is the older network.
        spec: with desc, the table also carries joker_spec's exact numeric columns (magnitudes, growth, conditions); 2: and the measured strength of each joker (joker_strength.json)."""
        super().__init__()
        # marg: the marginal-value columns (marginal.py, filled by encode(marginal=True) / the engine's set_marginal) are
        # inputs: the global vector, each joker and each action row carry a few more columns
        # (marg 1: the marginal-value columns; 2: also the potential columns)
        self.marg = int(marg)
        g_dim = g_dim or (global_size() if marg >= 2 else global_size() - POT_G if marg == 1
                          else global_size() - MARG_G - POT_G)
        self.f_joker = F_JOKER_P if marg >= 2 else F_JOKER_W if marg == 1 else F_JOKER
        self.f_act = F_ACT_P if marg >= 2 else F_ACT_W if marg == 1 else F_ACT
        self.config = {"g_dim": g_dim, "d": d, "emb": emb}
        # widths of the action path (aw) and the value head (vw), and residual blocks on the state (res): the defaults are the older network
        self.aw, self.vw, self.res = int(aw), int(vw), int(res)
        if aw != 128:
            self.config["aw"] = int(aw)
        if vw != 128:
            self.config["vw"] = int(vw)
        if res:
            self.config.update(res=int(res), res_expand=int(res_expand))
        # topk: only the topk plays and the topk discards (nnue_ac.topk_mask) are scored; stored, so evaluation applies the same limit
        self.topk = int(topk)
        if self.topk:
            self.config["topk"] = self.topk
        if marg:
            self.config["marg"] = int(marg)
        self.attn = int(attn)
        self.emb = nn.Embedding(VOCAB_SIZE, emb)
        self.has_desc = bool(desc)
        if self.has_desc:
            from .joker_desc import item_descriptors
            table = torch.from_numpy(item_descriptors(spec=bool(spec), strength=int(spec) >= 2))
            self.register_buffer("desc", table)                 # saved with the weights
            self.desc_proj = nn.Linear(table.shape[1], emb)
            self.config["desc"] = int(table.shape[1])
            if spec:
                self.config["spec"] = int(spec)                  # 1: the joker_spec columns, 2: also the measured strength columns
        if self.attn:
            self.config.update(attn=self.attn, tok=tok, heads=heads)
            self.card = mlp(F_CARD, tok, tok)
            self.joker = mlp(emb + self.f_joker, tok, tok)
            self.cons_tok = nn.Linear(emb, tok)
            self.g_tok = mlp(g_dim, tok, tok)
            self.slot = nn.Parameter(torch.randn(MAX_JOKERS, tok) * 0.02)      # a joker's position (they score
            self.kind = nn.Parameter(torch.randn(4, tok) * 0.02)               # left to right); the token type
            self.mix = nn.ModuleList(AttnBlock(tok, heads) for _ in range(self.attn))
            self.mix_norm = nn.LayerNorm(tok)
            self.state = mlp(g_dim + tok * 5, d, d, n=3)
            self.card_out = nn.Linear(tok, 64)     # a play / discard slot gathers its cards' encodings: for
            ce_dim = 64                            # hundreds of slots per state, so they are kept narrow
        else:
            self.card = mlp(F_CARD, 64, 64)
            self.joker = mlp(emb + self.f_joker, 64, 64)
            self.state = mlp(g_dim + 64 * 4 + emb, d, d, n=3)
            ce_dim = 64
        self.act = mlp(emb + self.f_act, aw, aw)
        self.sub_proj = nn.Linear(ce_dim, aw)
        self.s_proj = nn.Linear(d, aw)
        self.head = mlp(aw * 3, aw, 1)
        self.value = mlp(d, vw, 1)
        self.trunk = nn.ModuleList(ResBlock(d, res_expand) for _ in range(self.res))
        self.trunk_norm = nn.LayerNorm(d) if self.res else None
        self.extra_dim = 0
        if extra:
            self.add_extra(int(extra))
        # aux: prediction heads on the state, trained beside the policy and the value (train.py --aux-coef):
        # [0] the logit that the blind now faced is beaten, [1] the blinds this game will beat / AUX_BLINDS
        if aux:
            self.config["aux"] = int(aux)
            self.aux = mlp(d, 128, int(aux))
        # replace actions (env.A_REPLACE) also see the joker they give up (obs "aid2"). Starts at zero, so a
        # checkpoint from before the action existed scores every old action exactly as it did
        self.replace = nn.Linear(emb, aw)
        nn.init.zeros_(self.replace.weight)
        nn.init.zeros_(self.replace.bias)
        self.priv = 0
        self.priv_actor = 0
        if priv:
            self.add_priv(int(priv))
            if priv_actor:
                self.add_priv_actor()
        self.streams = 0
        if streams:
            self.add_streams(int(streams))
        self.aux_play = 0
        self.aux_loo = 0
        self._qcache = None
        if qhead:
            self.add_qhead()
        if reach:
            self.add_reach_head()
        if aux_play:
            self.add_aux_play()
        if aux_loo:
            self.add_aux_loo()
        self.phi = 0
        if phi:
            self.add_phi()
        self.goal_in = 0
        self.goal_out = 0
        if goal_in:
            self.add_goal_input(int(goal_in))
        if goal_out:
            self.add_goal_head(int(goal_out))

    def add_phi(self):
        """The judge potential (lookahead.py): each action row also reads obs "pa" [B, A, 2] = (what the action changes in the potential, flag), through a projection that
        starts at zero; and a head predicts the potential of the state from the state (phi_pred(), trained as an auxiliary)."""
        self.phi = 1
        self.config["phi"] = 1
        self.phi_act = nn.Linear(2, self.aw)
        nn.init.zeros_(self.phi_act.weight)
        nn.init.zeros_(self.phi_act.bias)
        self.phi_head = mlp(self.config["d"], 128, 1)
        dev = next(self.parameters()).device
        self.phi_act.to(dev)
        self.phi_head.to(dev)

    def phi_pred(self) -> torch.Tensor:
        """[B] the potential of the states of the last forward(), in units of 10 nats."""
        return self.phi_head(self._s.to(self.phi_head[0].weight.dtype)).squeeze(-1)

    def add_goal_input(self, k: int):
        """Hierarchy, the worker's side: the network is told which of k goals (postures) it is playing for. ob["goal"] [B] selects an embedding that is
        added to the state both heads read (zero-initialised, so a network without it is unchanged)."""
        self.goal_in = int(k)
        self.config["goal_in"] = int(k)
        self.goal_emb = nn.Embedding(int(k), self.config["d"])
        nn.init.zeros_(self.goal_emb.weight)
        self.goal_emb.to(next(self.parameters()).device)

    def add_goal_head(self, k: int):
        """Hierarchy, the manager's side: a head over k goals, read from the state of the last forward() (goal_logits)."""
        self.goal_out = int(k)
        self.config["goal_out"] = int(k)
        self.goal_head = mlp(self.config["d"], 128, int(k))
        self.goal_head.to(next(self.parameters()).device)

    def goal_logits(self) -> torch.Tensor:
        return self.goal_head(self._s.to(self.goal_head[0].weight.dtype))

    def values(self, ob: dict) -> torch.Tensor:
        """The value only (no action rows, no logits): the state is encoded as in forward() (with the privileged future list added when the network reads it) and the value head read.
        For a search that evaluates many states and needs nothing else (tools/az_mcts.py)."""
        assert self.attn, "values() needs the attention encoder"
        ce, s = self._encode_attn(ob)
        if self.priv_actor and "priv" in ob:
            s = s + self.priv_to_state(self.priv_enc(ob["priv"].to(s.dtype)))
        return self._value(s, ob)

    def add_reach_head(self):
        """Outcome heads on the state vector (train.py --reach-coef): logits of P(the game reaches ante 5), 6, 7, 8 and P(it is won), well-scaled 0..1 targets that
        a search can rank moves by. The last layer starts at zero (p = 0.5): the policy and value are untouched."""
        self.config["reach"] = 1
        self.reach = mlp(self.config["d"], 128, 5)
        nn.init.zeros_(self.reach[-1].weight)
        nn.init.zeros_(self.reach[-1].bias)
        self.reach.to(next(self.parameters()).device)

    def reach_probs(self) -> torch.Tensor:
        """[B, 5] the outcome probabilities for the batch the last forward() saw."""
        return torch.sigmoid(self.reach(self._s.float()).float())

    def add_qhead(self):
        """A second head over the same (state, action) rows as the policy's: Q(s, a) for every legal action (tools/mpo_train.py). The last layer starts at zero."""
        self.config["qhead"] = 1
        self.qhead = mlp(self.aw * 3, self.aw, 1)
        nn.init.zeros_(self.qhead[-1].weight)
        nn.init.zeros_(self.qhead[-1].bias)
        self.qhead.to(next(self.parameters()).device)

    def q_values(self) -> torch.Tensor:
        """[B, A] Q of the legal actions of the batch the last forward() saw (0 for the others; mask them)."""
        b, a_idx, feat, shape = self._qcache
        out = self.qhead(feat.to(self.qhead[0].weight.dtype)).squeeze(-1)
        q = torch.zeros(shape, dtype=out.dtype, device=out.device)
        q[b, a_idx] = out
        return q

    def add_aux_play(self):
        """An auxiliary head: the (log) score of the play that was chosen, from the state and the chosen action's row (train.py --aux-play-coef).
        Trained on what the engine scored; it teaches the encoders what a build and a hand are worth together."""
        self.aux_play = 1
        self.config["aux_play"] = 1
        self.play_head = mlp(256, 128, 1)
        self.play_head.to(next(self.parameters()).device)

    def add_aux_loo(self):
        """An auxiliary head: each held joker's leave-one-out contribution (the engine's marginal columns of that joker, marginal.py), from the
        joker's token and the state (train.py --aux-loo-coef). Meaningful only when the network does not read those columns itself (marg 0)."""
        assert not self.marg, "--aux-loo: the network reads the marginal columns as inputs, the head would only copy them"
        self.aux_loo = 1
        self.config["aux_loo"] = 1
        jd = self.config.get("tok", 64) if self.attn else 64
        self.loo_head = mlp(jd + self.config["d"], 64, MARG_J)
        self.loo_head.to(next(self.parameters()).device)

    def aux_heads(self, ob: dict, chosen: torch.Tensor) -> dict:
        """The heads above, for the batch the last forward() saw. chosen: the action taken in each state."""
        out = {}
        s, ce = self._s, self._ce
        if self.aux_play:
            b = torch.arange(chosen.shape[0], device=chosen.device)
            rows = self._rows(ob, ce, b, chosen)
            out["play"] = self.play_head(torch.cat([self.s_proj(s), rows.to(s.dtype)], -1)).squeeze(-1)
        if self.aux_loo:
            jt = self._jt
            out["loo"] = self.loo_head(torch.cat([jt, s.unsqueeze(1).expand(-1, jt.shape[1], -1)], -1))
        return out

    def add_streams(self, k: int):
        """Two reward streams (streams.py): the value is a vector of k outputs, one per shaped ante and the sparse stream's last; the policy and
        the single value head of an older network are untouched (the single head is simply not read)."""
        self.streams = int(k)
        self.config["streams"] = int(k)
        d = self.config["d"]
        self.value_streams = mlp(d, 128, int(k))
        nn.init.zeros_(self.value_streams[-1].weight)
        nn.init.zeros_(self.value_streams[-1].bias)
        self.value_streams.to(next(self.parameters()).device)

    def add_priv_actor(self):
        """The policy also reads the privileged observation: its encoding is added to the state both heads use (zero-initialised, so
        the network starts as it was). Such a network can only play where the future is known (exact-seed games with the engine's priv on)."""
        d = self.config["d"]
        self.priv_actor = 1
        self.config["priv_actor"] = 1
        self.priv_to_state = nn.Linear(64, d)
        nn.init.zeros_(self.priv_to_state.weight)
        nn.init.zeros_(self.priv_to_state.bias)
        self.priv_to_state.to(next(self.parameters()).device)

    def add_extra(self, k: int):
        """Extra per-state inputs ob["ex"] [B, k] (train.py --oracle-inputs: the ante-8 oracle and the money-option curve), added to the state both
        heads read through a projection that starts at zero, so a network that never had them plays exactly as before."""
        self.extra_dim = int(k)
        self.config["extra"] = int(k)
        self.extra_in = nn.Linear(int(k), self.config["d"])
        nn.init.zeros_(self.extra_in.weight)
        nn.init.zeros_(self.extra_in.bias)
        self.extra_in.to(next(self.parameters()).device)

    def add_priv(self, dim: int):
        """A critic-only branch: the value also reads the privileged observation (the next shop, the next boss, the top of the deck: what
        the policy cannot see). It starts at zero, so the value is what it was; the state it adds to is detached, so no gradient from
        the privileged value reaches the encoder the policy uses, and the policy's inputs and outputs are untouched."""
        d = self.config["d"]
        self.priv = int(dim)
        self.config["priv"] = int(dim)
        self.priv_enc = mlp(dim, 128, 64)
        self.value_priv = mlp(d + 64, 128, 1)
        nn.init.zeros_(self.value_priv[-1].weight)
        nn.init.zeros_(self.value_priv[-1].bias)
        dev = next(self.parameters()).device
        self.priv_enc.to(dev)
        self.value_priv.to(dev)

    def _value(self, s: torch.Tensor, ob: dict) -> torch.Tensor:
        if self.streams:
            return self.value_streams(s.detach() if self.value_detach else s)          # [B, streams]
        v = self.value(s.detach() if self.value_detach else s).squeeze(-1)
        if self.priv and "priv" in ob:
            pe = self.priv_enc(ob["priv"].to(s.dtype))
            v = v + self.value_priv(torch.cat([s.detach(), pe], -1)).squeeze(-1)
        return v

    def item(self, ids: torch.Tensor) -> torch.Tensor:
        """An item's embedding: learned per id, plus (desc) a projection of what the item does."""
        e = self.emb(ids)
        if self.has_desc:
            e = e + self.desc_proj(self.desc[ids])
        return e

    def _encode_attn(self, ob: dict):
        """The jokers, the cards in hand, the consumables and a global token attend to each other. -> the
        cards' encodings [B, H, 64] (what a play / discard slot is built from) and the state [B, d]."""
        g = ob["g"][:, :self.config["g_dim"]]
        cards = ob["cards"].float()
        jf = ob["jf"][..., :self.f_joker].float()
        cm, jm, km = cards[..., 0] > 0.5, jf[..., 0] > 0.5, ob["cid"] > 0
        x = torch.cat([(self.g_tok(g) + self.kind[3]).unsqueeze(1),
                       self.joker(torch.cat([self.item(ob["jid"]), jf], -1)) + self.slot + self.kind[1],
                       self.card(cards) + self.kind[0],
                       self.cons_tok(self.item(ob["cid"])) + self.kind[2]], 1)
        keep = torch.cat([torch.ones_like(cm[:, :1]), jm, cm, km], 1)          # the global token is always there
        for block in self.mix:
            x = block(x, keep)
        h = self.mix_norm(x)
        J, H = jm.shape[1], cm.shape[1]
        hg, hj, hc, hk = h[:, 0], h[:, 1:1 + J], h[:, 1 + J:1 + J + H], h[:, 1 + J + H:]
        self._jt = hj
        jw, cw, kw = jm.unsqueeze(-1).to(h.dtype), cm.unsqueeze(-1).to(h.dtype), km.unsqueeze(-1).to(h.dtype)
        jmean = (hj * jw).sum(1) / jw.sum(1).clamp(min=1)
        jmax = (hj - (1 - jw) * 1e4).max(1).values.clamp(min=-10)
        cmean = (hc * cw).sum(1) / cw.sum(1).clamp(min=1)
        kmean = (hk * kw).sum(1) / kw.sum(1).clamp(min=1)
        s = F.relu(self.state(torch.cat([g, hg, jmean, jmax, cmean, kmean], -1)))
        return self.card_out(hc) * cw, s

    pack_legal = True      # score only the legal actions (same logits: the others are masked to -1e9 anyway)

    def _legal_logits(self, ob: dict, ce: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """The action logits computed for the legal (state, action) pairs only, scattered into [B, A]; every
        other entry is -1e9, as the dense path's mask leaves it. Same weights, same values."""
        mask = ob["mask"]
        b, a_idx = mask.nonzero(as_tuple=True)                          # L legal pairs
        rows = self._rows(ob, ce, b, a_idx)
        sp = self.s_proj(s)[b]
        feat = torch.cat([sp, rows, sp * rows], -1)
        if hasattr(self, "qhead"):
            self._qcache = (b, a_idx, feat, mask.shape)
        out = self.head(feat).squeeze(-1)
        logits = torch.full(mask.shape, -1e9, dtype=out.dtype, device=out.device)
        logits[b, a_idx] = out
        return logits

    def _rows(self, ob: dict, ce: torch.Tensor, b: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
        """The action rows [L, 128] of the (state b, action a_idx) pairs: the item, its features, and (a play / discard) the cards it selects."""
        rows = self.act(torch.cat([self.item(ob["aid"][b, a_idx].long()), ob["af"][b, a_idx][:, :self.f_act].float()], -1))  # L,128
        if self.phi and "pa" in ob:
            rows = rows + self.phi_act(ob["pa"][b, a_idx].to(rows.dtype)).to(rows.dtype)
        if "aid2" in ob:
            a2 = ob["aid2"][b, a_idx].long()                            # (a rollout may store the ids narrow)
            sel = (a2 > 0).nonzero(as_tuple=True)[0]
            if sel.numel():
                rows = rows.index_add(0, sel, self.replace(self.item(a2[sel])).to(rows.dtype))
        is_sub = a_idx < 2 * N_SUB                                      # play / discard slots: + their cards
        bs, as_ = b[is_sub], a_idx[is_sub]
        subs = ob["subs"][bs, as_].long()                               # Ls,5 (-1 = padding)
        valid = (subs >= 0).to(ce.dtype).unsqueeze(-1)
        picked = ce[bs.unsqueeze(1), subs.clamp(min=0)] * valid         # Ls,5,64
        sub = self.sub_proj(picked.sum(1) / valid.sum(1).clamp(min=1))
        rows = rows.index_add(0, is_sub.nonzero(as_tuple=True)[0], sub.to(rows.dtype))
        return rows

    def forward(self, ob: dict, aux: bool = False):
        """-> (logits, value), and with aux=True the prediction heads' outputs as a third."""
        if getattr(self, "topk", 0):
            from .nnue_ac import topk_mask
            ob = {**ob, "mask": topk_mask(ob, self.topk)}
        B = ob["g"].shape[0]
        if self.attn:
            ce, s = self._encode_attn(ob)
        else:
            cmask = ob["cards"][..., 0:1]                                  # present flag
            ce = self.card(ob["cards"]) * cmask                            # B,8,64
            cmean = ce.sum(1) / cmask.sum(1).clamp(min=1)
            cmax = (ce - (1 - cmask) * 1e4).max(1).values.clamp(min=-10)
            jf = ob["jf"][..., :self.f_joker]
            jmask = jf[..., 0:1]
            je = self.joker(torch.cat([self.item(ob["jid"]), jf], -1)) * jmask
            jmean = je.sum(1) / jmask.sum(1).clamp(min=1)
            jmax = (je - (1 - jmask) * 1e4).max(1).values.clamp(min=-10)
            self._jt = je
            cons = self.item(ob["cid"]).mean(1)
            g = ob["g"][:, :self.config["g_dim"]]            # older networks: without the columns added since
            s = self.state(torch.cat([g, cmean, cmax, jmean, jmax, cons], -1))
            s = F.relu(s)
        if self.priv_actor and "priv" in ob:
            s = s + self.priv_to_state(self.priv_enc(ob["priv"].to(s.dtype)))
        if self.goal_in and "goal" in ob:
            s = s + self.goal_emb(ob["goal"].long()).to(s.dtype)
        if self.res:
            for blk in self.trunk:
                s = blk(s)
            s = self.trunk_norm(s)
        if getattr(self, "extra_dim", 0) and "ex" in ob:
            s = s + self.extra_in(ob["ex"].to(s.dtype)).to(s.dtype)
        self._s, self._ce = s, ce

        if self.pack_legal:
            if aux:
                return self._legal_logits(ob, ce, s), self._value(s, ob), self.aux(s)
            return self._legal_logits(ob, ce, s), self._value(s, ob)
        a = self.act(torch.cat([self.item(ob["aid"]), ob["af"][..., :self.f_act].float()], -1))  # B,A,128
        if "aid2" in ob:
            a2 = ob["aid2"]
            a = a + self.replace(self.item(a2)) * (a2 > 0).unsqueeze(-1)
        # each play/discard slot also sees the mean embedding of the cards it selects
        subs = ob["subs"]                                              # B,436,5 (-1 = padding)
        valid = (subs >= 0).float().unsqueeze(-1)
        idx = subs.clamp(min=0).reshape(B, -1, 1).expand(-1, -1, ce.shape[-1])
        picked = torch.gather(ce, 1, idx).reshape(B, subs.shape[1], subs.shape[2], -1) * valid
        sub = self.sub_proj(picked.sum(2) / valid.sum(2).clamp(min=1))   # B,436,128
        a = torch.cat([a[:, :2 * N_SUB] + sub, a[:, 2 * N_SUB:]], 1)
        sp = self.s_proj(s).unsqueeze(1).expand(-1, a.shape[1], -1)
        logits = self.head(torch.cat([sp, a, sp * a], -1)).squeeze(-1)
        logits = logits.masked_fill(~ob["mask"], -1e9)
        if aux:
            return logits, self._value(s, ob), self.aux(s)
        return logits, self._value(s, ob)


def batch_obs(obs_list: list[dict], device="cpu") -> dict:
    out = {}
    for k in OBS_KEYS + tuple(x for x in OPTIONAL_KEYS if x in obs_list[0]):
        arr = np.stack([o[k] for o in obs_list])
        t = torch.from_numpy(arr)
        if k == "af":
            t = t.float()
        out[k] = t.to(device)
    return out


def stack_obs(obs_list: list[dict], device="cpu") -> dict:
    """Like batch_obs, but keeps "af" in its stored dtype (float16 after train.compact); forward()
    converts it. For stacking a whole rollout once and slicing minibatches on the GPU."""
    keys = OBS_KEYS + tuple(x for x in OPTIONAL_KEYS if x in obs_list[0])
    return {k: torch.from_numpy(np.stack([o[k] for o in obs_list])).to(device) for k in keys}


def save_model(model: ActorCritic, path: str, extra: dict | None = None):
    torch.save({"config": model.config, "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_model(path: str, device="cpu") -> ActorCritic:
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["config"].get("arch") == "nnue":                     # the Stockfish-shaped actor-critic (nnue_ac.py)
        from .nnue_ac import NnueActorCritic
        m = NnueActorCritic(**ck["config"]).to(device)
        m.load_state_dict(ck["state_dict"])
        return m.eval()
    m = ActorCritic(**ck["config"]).to(device)
    missing = m.load_state_dict(ck["state_dict"], strict=False)
    bad = [k for k in missing.missing_keys if not k.startswith("replace.")] + list(missing.unexpected_keys)
    if bad:                                        # only the zero-initialised replace path may be absent
        raise RuntimeError(f"{path}: checkpoint does not match the model: {bad}")
    m.eval()
    return m
