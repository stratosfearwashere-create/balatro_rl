"""DreamerV3 on the shop-only game (the scripted player plays the rounds; the agent acts outside them: blind select / skip, shop, packs, vouchers, selling).

A world model is learned from the engine's own observations and rewards, and the actor and the critic are trained on rollouts imagined in its latent space.

  world model   RSSM (a layer-normalised GRU for the deterministic state h, a stochastic state z of 32 categoricals x 32 classes with 1% unimix and straight-through
                gradients); encoder / decoder over the observation vector; reward head and critic with two-hot targets on symlog bins; continue head.
                Loss = prediction (observation, reward, continue) + 1.0 dyn KL + 0.1 rep KL, both with free bits (1 nat).
  actor/critic  trained on imagined rollouts (horizon 15, gamma 0.997, lambda 0.95) from every state the world model inferred in the replayed sequences.
                Actor: REINFORCE on advantages scaled by max(1, EMA of the 5-95 percentile range of the returns) + a small entropy bonus;
                critic: two-hot cross-entropy to the lambda returns plus a regulariser toward a slow EMA copy of itself.
  actions       the 72 actions outside rounds (engine actions 436..507) as a fixed categorical. Legality: the real mask when acting in the game; in imagination a mask
                head (trained with BCE on the real masks) stands in for it.
  observation   vector = symlog of [global features, the 8 jokers' features, the 72 actions' features] + the ids (jokers, consumables, the item behind every action) embedded
                + the legality mask. The decoder rebuilds the continuous part (symlog MSE), every id (cross-entropy) and the mask (BCE).

    python -m balatro_rl.dreamer --out checkpoints/dream1.pt --envs 64
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import A_SELECT, F_ACT_W, F_JOKER_W, N_ACTIONS, VOCAB_SIZE, MAX_JOKERS

A0 = A_SELECT                                       # the first action outside rounds
NA = N_ACTIONS - A0                                 # 72
N_CONS = 3
N_IDS = MAX_JOKERS + N_CONS + NA                    # ids in an observation: jokers, consumables, the item behind each action


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class TwoHot:
    """Two-hot targets / decoding over 255 bins evenly spaced in symlog space (-20..20)."""

    def __init__(self, dev, n=255, lim=20.0):
        self.bins = torch.linspace(-lim, lim, n, device=dev)

    def target(self, x):                            # x raw [...] -> [..., n]
        y = symlog(x).clamp(self.bins[0], self.bins[-1])
        hi = torch.searchsorted(self.bins, y.contiguous()).clamp(1, len(self.bins) - 1)
        lo = hi - 1
        w_hi = ((y - self.bins[lo]) / (self.bins[hi] - self.bins[lo])).clamp(0, 1)
        out = torch.zeros(y.shape + (len(self.bins),), device=y.device)
        out.scatter_(-1, lo.unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
        out.scatter_add_(-1, hi.unsqueeze(-1), w_hi.unsqueeze(-1))
        return out

    def loss(self, logits, x):
        return -(self.target(x) * torch.log_softmax(logits.float(), -1)).sum(-1)

    def mean(self, logits):
        return symexp((torch.softmax(logits.float(), -1) * self.bins).sum(-1))


def mlp(i, h, o, layers=2, last_zero=False):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    out = nn.Linear(d, o)
    if last_zero:
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
    mods.append(out)
    return nn.Sequential(*mods)


class RSSM(nn.Module):
    def __init__(self, act_dim, embed, deter=512, stoch=32, classes=32, hidden=512):
        super().__init__()
        self.deter, self.stoch, self.classes = deter, stoch, classes
        self.zdim = stoch * classes
        self.feat_dim = deter + self.zdim
        self.img_in = nn.Sequential(nn.Linear(self.zdim + act_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru_in = nn.Linear(hidden + deter, 3 * deter, bias=False)
        self.gru_norm = nn.LayerNorm(3 * deter)
        self.prior_net = nn.Sequential(nn.Linear(deter, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, self.zdim))
        self.post_net = nn.Sequential(nn.Linear(deter + embed, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, self.zdim))
        self.h0 = nn.Parameter(torch.zeros(deter))

    def _gru(self, x, h):
        r, c, u = self.gru_norm(self.gru_in(torch.cat([x, h], -1))).chunk(3, -1)
        r, u = torch.sigmoid(r), torch.sigmoid(u - 1.0)
        return u * torch.tanh(r * c) + (1 - u) * h

    def _dist_logits(self, raw):
        lg = raw.reshape(raw.shape[:-1] + (self.stoch, self.classes)).float()
        p = 0.99 * torch.softmax(lg, -1) + 0.01 / self.classes          # unimix
        return torch.log(p)

    def _sample(self, lg):
        p = torch.softmax(lg, -1)
        idx = torch.distributions.Categorical(probs=p, validate_args=False).sample()
        oh = F.one_hot(idx, self.classes).float()
        return (oh + p - p.detach()).reshape(lg.shape[:-2] + (self.zdim,))

    def initial(self, B, dev):
        h = torch.tanh(self.h0).expand(B, -1)
        lg = self._dist_logits(self.prior_net(h))
        z = F.one_hot(lg.argmax(-1), self.classes).float().reshape(B, self.zdim)
        return h, z

    def obs_step(self, state, act, embed, first):
        """state (h, z); act [B, A] (the action that led here, zero at a first step); embed [B, E]; first [B] bool. -> new state, post logits, prior logits."""
        h, z = state
        f = first.unsqueeze(-1)
        h0, z0 = self.initial(h.shape[0], h.device)
        h, z, act = torch.where(f, h0, h), torch.where(f, z0, z), act * (~f)
        h = self._gru(self.img_in(torch.cat([z, act], -1)), h)
        prior = self._dist_logits(self.prior_net(h))
        post = self._dist_logits(self.post_net(torch.cat([h, embed], -1)))
        return (h, self._sample(post)), post, prior

    def img_step(self, state, act):
        h, z = state
        h = self._gru(self.img_in(torch.cat([z, act], -1)), h)
        prior = self._dist_logits(self.prior_net(h))
        return (h, self._sample(prior)), prior

    def feat(self, state):
        return torch.cat([state[0], state[1]], -1)


def kl_cat(a, b):
    """KL(a || b) of two [.., stoch, classes] log-prob tensors, summed over the stochastic dims."""
    return (a.exp() * (a - b)).sum(-1).sum(-1)


class Dreamer(nn.Module):
    def __init__(self, dc, deter=512, hidden=512, emb=16, layers=2, dev="cpu"):
        super().__init__()
        self.dc = dc
        self.id_emb = nn.Embedding(VOCAB_SIZE, emb)
        in_dim = dc + N_IDS * emb + NA
        self.encoder = mlp(in_dim, hidden, hidden, layers)
        self.rssm = RSSM(NA + 1, hidden, deter=deter, hidden=hidden)           # (the action input has one extra slot: "an action the policy did not choose")
        fd = self.rssm.feat_dim
        self.decoder = mlp(fd, hidden, hidden, layers)
        self.dec_cont = nn.Linear(hidden, dc)
        self.dec_ids = nn.Linear(hidden, N_IDS * VOCAB_SIZE)
        self.mask_head = mlp(fd, hidden // 2, NA, 1)
        self.reward = mlp(fd, hidden, 255, layers, last_zero=True)
        self.cont = mlp(fd, hidden, 1, layers)
        self.actor = mlp(fd, hidden, NA, layers)
        self.critic = mlp(fd, hidden, 255, layers, last_zero=True)
        self.slow = mlp(fd, hidden, 255, layers, last_zero=True)
        self.slow.load_state_dict(self.critic.state_dict())
        for p in self.slow.parameters():
            p.requires_grad_(False)
        self.th = TwoHot(dev)
        self.ret_scale = 1.0
        self.config = {"dc": dc, "deter": deter, "hidden": hidden, "emb": emb, "layers": layers}

    def embed(self, cont, ids, mask):
        e = self.id_emb(ids.long()).flatten(-2)
        return self.encoder(torch.cat([cont.float(), e, mask.float()], -1))

    def observe(self, cont, ids, mask, act, first, state=None):
        """A sequence [B, L, ...] -> (feats [B, L, F], post logits, prior logits, the final state)."""
        B, L = cont.shape[:2]
        emb = self.embed(cont, ids, mask)
        if state is None:
            state = self.rssm.initial(B, cont.device)
        feats, posts, priors = [], [], []
        for t in range(L):
            state, po, pr = self.rssm.obs_step(state, act[:, t], emb[:, t], first[:, t])
            feats.append(self.rssm.feat(state)); posts.append(po); priors.append(pr)
        return torch.stack(feats, 1), torch.stack(posts, 1), torch.stack(priors, 1), state

    def decode(self, feat):
        h = self.decoder(feat)
        return self.dec_cont(h), self.dec_ids(h).reshape(feat.shape[:-1] + (N_IDS, VOCAB_SIZE)), self.mask_head(feat)

    def policy_logits(self, feat, mask):
        lg = self.actor(feat).float()
        m = mask.bool()
        m = torch.where(m.any(-1, keepdim=True), m, torch.ones_like(m))
        return lg.masked_fill(~m, -1e9)


# --------------------------------------------------------------------------- the environment side
def featurize(obs):
    """The engine's observation dict (N envs) -> (continuous [N, dc] float16 symlog, ids [N, N_IDS] int16, mask [N, NA] bool)."""
    g = np.asarray(obs["g"], dtype=np.float32)
    jf = np.asarray(obs["jf"], dtype=np.float32)[:, :, :F_JOKER_W].reshape(len(g), -1)
    af = np.asarray(obs["af"], dtype=np.float32)[:, A0:, :F_ACT_W].reshape(len(g), -1)
    c = np.concatenate([g, jf, af], 1)
    c = np.sign(c) * np.log1p(np.abs(np.nan_to_num(c, nan=0.0, posinf=1e4, neginf=-1e4)))
    ids = np.concatenate([np.asarray(obs["jid"]), np.asarray(obs["cid"]), np.asarray(obs["aid"])[:, A0:]], 1)
    return c.astype(np.float16), ids.astype(np.int16), np.asarray(obs["mask"])[:, A0:].copy()


class Replay:
    """One ring buffer per environment (a stream of records; a record is a state with the action and reward that led to it)."""

    def __init__(self, n_env, cap, dc):
        self.n, self.cap = n_env, cap
        self.cont = np.zeros((n_env, cap, dc), dtype=np.float16)
        self.ids = np.zeros((n_env, cap, N_IDS), dtype=np.int16)
        self.mask = np.zeros((n_env, cap, NA), dtype=bool)
        self.act = np.zeros((n_env, cap), dtype=np.int16)           # the action that led here (NA: none or not the policy's)
        self.rew = np.zeros((n_env, cap), dtype=np.float32)
        self.first = np.zeros((n_env, cap), dtype=bool)
        self.term = np.zeros((n_env, cap), dtype=bool)
        self.ptr = np.zeros(n_env, dtype=np.int64)
        self.count = np.zeros(n_env, dtype=np.int64)

    def add(self, e, cont, ids, mask, act, rew, first, term):
        i = self.ptr[e] % self.cap
        self.cont[e, i], self.ids[e, i], self.mask[e, i] = cont, ids, mask
        self.act[e, i], self.rew[e, i], self.first[e, i], self.term[e, i] = act, rew, first, term
        self.ptr[e] += 1
        self.count[e] = min(self.count[e] + 1, self.cap)

    def add_many(self, es, cont, ids, mask, act, rew, first, term):
        i = self.ptr[es] % self.cap
        self.cont[es, i], self.ids[es, i], self.mask[es, i] = cont, ids, mask
        self.act[es, i], self.rew[es, i], self.first[es, i], self.term[es, i] = act, rew, first, term
        self.ptr[es] += 1
        self.count[es] = np.minimum(self.count[es] + 1, self.cap)

    def size(self):
        return int(self.count.sum())

    def sample(self, B, L, rng):
        ok = np.flatnonzero(self.count >= L)
        es = rng.choice(ok, B)
        starts = (rng.random(B) * (self.count[es] - L + 1)).astype(np.int64)
        base = self.ptr[es] - self.count[es] + starts
        idx = (base[:, None] + np.arange(L)[None]) % self.cap
        e2 = es[:, None]
        return {"cont": self.cont[e2, idx], "ids": self.ids[e2, idx], "mask": self.mask[e2, idx], "act": self.act[e2, idx].astype(np.int64),
                "rew": self.rew[e2, idx], "first": self.first[e2, idx], "term": self.term[e2, idx]}


# --------------------------------------------------------------------------- training pieces
def lambda_returns(rew, val, cont, gamma, lam):
    """rew, val, cont [B, H(+1)]: rew[:, t] is received on arriving at step t, val[:, t] is the value of step t. Returns for steps 0..H-1."""
    H = rew.shape[1] - 1
    out = [None] * H
    nxt = val[:, -1]
    for t in reversed(range(H)):
        nxt = rew[:, t + 1] + gamma * cont[:, t + 1] * ((1 - lam) * val[:, t + 1] + lam * nxt)
        out[t] = nxt
    return torch.stack(out, 1)


def train_step(m, opts, batch, a, dev, stats):
    cont, ids, mask = [torch.from_numpy(batch[k]).to(dev) for k in ("cont", "ids", "mask")]
    act = F.one_hot(torch.from_numpy(batch["act"]).to(dev), NA + 1).float()
    rew = torch.from_numpy(batch["rew"]).to(dev)
    first = torch.from_numpy(batch["first"]).to(dev)
    term = torch.from_numpy(batch["term"]).to(dev)
    # ---- world model
    feats, post, prior, _ = m.observe(cont, ids, mask, act, first)
    rc, ri, rm = m.decode(feats)
    l_cont = 0.5 * ((rc.float() - cont.float()) ** 2).sum(-1)
    l_ids = F.cross_entropy(ri.float().reshape(-1, VOCAB_SIZE), ids.long().reshape(-1), reduction="none").reshape(ids.shape).sum(-1)
    l_mask = F.binary_cross_entropy_with_logits(rm.float(), mask.float(), reduction="none").sum(-1)
    l_rew = m.th.loss(m.reward(feats), rew)
    l_con = F.binary_cross_entropy_with_logits(m.cont(feats).squeeze(-1).float(), (~term).float(), reduction="none")
    dyn = kl_cat(post.detach(), prior).clamp(min=1.0)
    rep = kl_cat(post, prior.detach()).clamp(min=1.0)
    wm = (l_cont + l_ids + l_mask + l_rew + l_con + 1.0 * dyn + 0.1 * rep).mean()
    opts["wm"].zero_grad(set_to_none=True)
    wm.backward()
    nn.utils.clip_grad_norm_([p for g in opts["wm"].param_groups for p in g["params"]], a.clip)
    opts["wm"].step()
    # ---- imagination from every inferred state
    with torch.no_grad():
        B, L = feats.shape[:2]
        h0 = feats.reshape(B * L, -1).detach()
        if a.imag_starts and a.imag_starts < len(h0):
            h0 = h0[torch.randperm(len(h0), device=dev)[:a.imag_starts]]
        state = (h0[:, :m.rssm.deter], h0[:, m.rssm.deter:])
    fs, acts_i, lps, ents = [h0], [], [], []
    for _ in range(a.horizon):
        f = fs[-1].detach()
        lg = m.policy_logits(f, torch.sigmoid(m.mask_head(f).float()) > 0.5)
        dist = torch.distributions.Categorical(logits=lg, validate_args=False)
        ai = dist.sample()
        lps.append(dist.log_prob(ai)); ents.append(dist.entropy()); acts_i.append(ai)
        oh = torch.cat([F.one_hot(ai, NA).float(), torch.zeros(len(ai), 1, device=dev)], -1)
        with torch.no_grad():
            state, _ = m.rssm.img_step(state, oh)
        fs.append(m.rssm.feat(state))
    F_ = torch.stack(fs, 1)                                                     # [BL, H+1, F]
    with torch.no_grad():
        r_i = m.th.mean(m.reward(F_))
        c_i = torch.sigmoid(m.cont(F_).squeeze(-1).float())
        v_i = m.th.mean(m.critic(F_))
        ret = lambda_returns(r_i, v_i, c_i, a.gamma, a.lam)                      # [BL, H]
        weight = torch.cumprod(torch.cat([torch.ones_like(c_i[:, :1]), c_i[:, :-1] * a.gamma], 1), 1)[:, :a.horizon].detach()
        lo, hi = torch.quantile(ret.flatten(), 0.05), torch.quantile(ret.flatten(), 0.95)
        m.ret_scale = 0.99 * m.ret_scale + 0.01 * float(hi - lo)
        scale = max(1.0, m.ret_scale)
        adv = ((ret - v_i[:, :-1]) / scale)
    lp, ent = torch.stack(lps, 1), torch.stack(ents, 1)
    actor_loss = -(weight * (adv * lp + a.ent * ent)).mean()
    opts["actor"].zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(m.actor.parameters(), a.clip)
    opts["actor"].step()
    fc = F_[:, :-1].detach()
    lgc = m.critic(fc)
    with torch.no_grad():
        slow_t = torch.softmax(m.slow(fc).float(), -1)
    critic_loss = (weight * (m.th.loss(lgc, ret) - (slow_t * torch.log_softmax(lgc.float(), -1)).sum(-1))).mean()
    opts["critic"].zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(m.critic.parameters(), a.clip)
    opts["critic"].step()
    with torch.no_grad():
        for ps, pc in zip(m.slow.parameters(), m.critic.parameters()):
            ps.mul_(0.98).add_(pc.detach(), alpha=0.02)
    for k, v in (("wm", wm), ("cont", l_cont.mean()), ("ids", l_ids.mean()), ("mask", l_mask.mean()), ("rew", l_rew.mean()), ("dyn", dyn.mean()), ("rep", rep.mean()),
                 ("actor", actor_loss), ("critic", critic_loss), ("ent", ent.mean()), ("ret", ret.mean()), ("scale", torch.tensor(scale))):
        stats.setdefault(k, []).append(float(v))


class Acting:
    """The per-environment latent state while acting in the game."""

    def __init__(self, m, n, dev):
        self.m, self.n, self.dev = m, n, dev
        self.state = m.rssm.initial(n, dev)
        self.state = (self.state[0].clone(), self.state[1].clone())
        self.prev = torch.zeros(n, NA + 1, device=dev)
        self.first = torch.ones(n, dtype=torch.bool, device=dev)

    @torch.no_grad()
    def act(self, cont, ids, mask, greedy=False):
        m = self.m
        c, i, k = (torch.from_numpy(x).to(self.dev) for x in (cont, ids, mask))
        emb = m.embed(c, i, k)
        self.state, _, _ = m.rssm.obs_step(self.state, self.prev, emb, self.first)
        lg = m.policy_logits(m.rssm.feat(self.state), k)
        a = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg, validate_args=False).sample()
        return a.cpu().numpy()

    def after(self, a, done):
        oh = torch.zeros(self.n, NA + 1, device=self.dev)
        oh[torch.arange(self.n), torch.from_numpy(a).to(self.dev)] = 1.0
        d = torch.from_numpy(done).to(self.dev)
        self.prev = oh * (~d).unsqueeze(-1)
        self.first = d.clone()


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--out", default="checkpoints/dream1.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--collect", type=int, default=16, help="steps per environment between rounds of training")
    p.add_argument("--updates", type=int, default=16, help="training steps per round")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--imag-starts", type=int, default=512, help="imagined rollouts per training step (0: one from every inferred state)")
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent", type=float, default=3e-4)
    p.add_argument("--lr-wm", type=float, default=1e-4)
    p.add_argument("--lr-ac", type=float, default=3e-5)
    p.add_argument("--clip", type=float, default=100.0)
    p.add_argument("--cap", type=int, default=2000, help="records kept per environment")
    p.add_argument("--warmup", type=int, default=20000, help="records before training starts")
    p.add_argument("--deter", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--rounds", type=int, default=1_000_000)
    p.add_argument("--pace", type=float, default=1.5)
    p.add_argument("--ante-weight", type=float, default=8.0)
    p.add_argument("--weight-power", type=float, default=2.0)
    p.add_argument("--win-bonus", type=float, default=60.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--eval-seed0", type=int, default=9_000_000)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu-fraction", type=float, default=0.0)
    p.add_argument("--dev-engine", default="")
    return p.parse_args(argv)


def make_env(n, a, seed0):
    from .engine_env import CppVecEnv
    v = CppVecEnv(n, a.deck, a.stake, seed0=seed0, threads=a.threads,
                  reward_kw={"ante_weight": a.ante_weight, "win_bonus": a.win_bonus, "win_ante": 8, "shape_gamma": a.gamma})
    v.set_weight_shape(a.ante_weight, a.weight_power)
    v.set_auto_round(True, a.pace, 0)
    v.set_marginal(1)
    v.set_marginal_blind_only(True)
    return v


def evaluate(m, a, dev, n):
    ev = make_env(n, a, a.eval_seed0)
    act = Acting(m, n, dev)
    fin = np.zeros(n, dtype=bool)
    blinds = np.zeros(n); won = np.zeros(n, dtype=bool); ante = np.ones(n)
    m.eval()
    for _ in range(3000):
        c, i, k = featurize(ev.obs)
        full = np.asarray(ev.obs["mask"])
        pa = act.act(c, i, k, greedy=True)
        env_a = A0 + pa
        bad = ~full[np.arange(n), env_a]
        env_a = np.where(bad, [int(np.flatnonzero(r)[0]) for r in full], env_a)
        _, d, info = ev.step(np.where(fin, -1, env_a))
        first = d & ~fin
        blinds[first], won[first], ante[first] = info["blinds"][first], info["won"][first], info["ante"][first]
        fin |= d
        act.after(np.where(bad, NA, pa), d)
        if fin.all():
            break
    ev.close()
    m.train()
    return {"games": n, "win_rate": round(float(won.mean()), 4), "blinds": round(float(blinds.mean()), 3), "ante": round(float(ante.mean()), 2)}


def train(a):
    if a.dev_engine:
        import sys
        sys.path.insert(0, a.dev_engine)
        import _engine_dev
        import balatro_rl.sim as S
        sys.modules["balatro_rl.sim._engine"] = _engine_dev
        S._engine = _engine_dev
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = a.device
    if str(dev).startswith("cuda") and a.gpu_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)
    N = a.envs
    venv = make_env(N, a, a.seed * 1_000_000 + 7)
    c0, i0, k0 = featurize(venv.obs)
    dc = c0.shape[1]
    m = Dreamer(dc, deter=a.deter, hidden=a.hidden, dev=dev).to(dev)
    m.th = TwoHot(dev)
    log_path = a.out.replace(".pt", "_log.jsonl")
    it0, games, wins, best = 0, 0, 0, -1.0
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location=dev, weights_only=False)
        m.load_state_dict(ck["model"])
        m.ret_scale = ck.get("ret_scale", 1.0)
        it0 = ck["iter"]
        last = [json.loads(l) for l in open(log_path) if l.strip().startswith("{")][-1]
        games, wins = last["games"], last.get("wins", 0)
        print(json.dumps({"resumed": a.out, "iter": it0}), flush=True)
    wm_params = [p for n_, p in m.named_parameters() if n_.split(".")[0] in ("id_emb", "encoder", "rssm", "decoder", "dec_cont", "dec_ids", "mask_head", "reward", "cont")]
    opts = {"wm": torch.optim.Adam(wm_params, lr=a.lr_wm, eps=1e-8),
            "actor": torch.optim.Adam(m.actor.parameters(), lr=a.lr_ac, eps=1e-5),
            "critic": torch.optim.Adam(m.critic.parameters(), lr=a.lr_ac, eps=1e-5)}
    rb = Replay(N, a.cap, dc)
    acting = Acting(m, N, dev)
    ar = np.arange(N)
    for e in range(N):                                            # the first record of every stream
        rb.add(e, c0[e], i0[e], k0[e], NA, 0.0, True, False)
    ep_blinds = []
    t0 = time.time()
    sps = 0
    for it in range(it0 + 1, a.rounds + 1):
        tc = time.time()
        m.eval()
        for _ in range(a.collect):
            c, i, k = featurize(venv.obs)
            full = np.asarray(venv.obs["mask"])
            pa = acting.act(c, i, k)
            env_a = A0 + pa
            bad = ~full[ar, env_a]                                  # (an action the policy cannot take: a state that is not outside a round, in practice never)
            if bad.any():
                env_a = np.where(bad, [int(np.flatnonzero(r)[0]) for r in full], env_a)
                pa = np.where(bad, NA, pa)
            r, d, info, _, tr = venv.step_split(env_a)
            c2, i2, k2 = featurize(venv.obs)
            r = np.nan_to_num(np.asarray(r, dtype=np.float32))
            live = np.flatnonzero(~d)
            if len(live):
                rb.add_many(live, c2[live], i2[live], k2[live], pa[live], r[live], np.zeros(len(live), bool), np.zeros(len(live), bool))
            for e in np.flatnonzero(d):                             # the game ended: a closing record (the state it ended in), then the first record of the next game
                rb.add(e, c[e], i[e], k[e], pa[e], r[e], False, not tr[e])
                rb.add(e, c2[e], i2[e], k2[e], NA, 0.0, True, False)
                games += 1
                wins += int(info["won"][e])
                ep_blinds.append(int(info["blinds"][e]))
            acting.after(pa, np.asarray(d))
        collect_s = time.time() - tc
        stats = {}
        m.train()
        tt = time.time()
        if rb.size() >= a.warmup:
            for _ in range(a.updates):
                train_step(m, opts, rb.sample(a.batch, a.seq, rng), a, dev, stats)
        train_s = time.time() - tt
        sps = int(N * a.collect / max(1e-9, collect_s + train_s))
        row = {"iter": it, "games": games, "wins": wins, "blinds": round(float(np.mean(ep_blinds[-1000:])), 3) if ep_blinds else None, "sps": sps,
               "min": round((time.time() - t0) / 60, 1), "records": rb.size(), "collect_s": round(collect_s, 2), "train_s": round(train_s, 2),
               "model": {k: round(float(np.mean(v)), 3) for k, v in stats.items()}}
        if a.eval_every and it % a.eval_every == 0:
            ev = evaluate(m, a, dev, a.eval_games)
            row["eval"], row["eval_stake"] = ev, a.stake
            if ev["blinds"] > best:
                best = ev["blinds"]
                torch.save({"model": m.state_dict(), "config": m.config, "iter": it, "ret_scale": m.ret_scale, "eval": ev}, a.out.replace(".pt", "_best.pt"))
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if it % a.save_every == 0:
            torch.save({"model": m.state_dict(), "config": m.config, "iter": it, "ret_scale": m.ret_scale}, a.out)
    torch.save({"model": m.state_dict(), "config": m.config, "iter": a.rounds, "ret_scale": m.ret_scale}, a.out)


if __name__ == "__main__":
    train(parse())
