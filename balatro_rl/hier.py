"""Hierarchical PPO for Balatro (the C++ engine, Gold stake).

Two policies, two time scales, one game.

  worker   plays the rounds: every decision while a blind is being played (plays, discards, consumables and sales in the round). Its problem is short and
           dense: clear this blind. Its reward is the progress toward the blind's requirement (capped at 1: no pay for overkill), a bonus for clearing it, and a
           small bonus for the posture the manager chose. Its episode is one round (gamma 0.99), so its credit never has to cross a shop.
  manager  makes every decision outside a round: blind select / skip, the shop, packs, selling, vouchers. It never sees a round's plays; it sees the outcome of
           the round the worker played with the build it made. Its reward is the engine's own (blind rewards that rise with the ante, the win bonus), summed over
           everything that happens until its next decision (a semi-Markov step), and its horizon is the ~150 decisions of a run that are not plays.
  posture  when the manager selects a blind it also picks one of K postures (--goals): 0 neutral, 1 bank (leftover hands pay: $1 each and the interest they
           keep), 2 grow (more hands played feed the counters that grow per hand). The worker is told the posture (a learned embedding added to its state) and
           gets a small bonus on a clear for living up to it. The postures are tactics, not valuations of builds: the manager learns when each pays, from the
           extrinsic reward alone.

Why: in one policy over ~2,500 steps 90% of the steps are plays, so the 10% that decide the run (the build) get little of the entropy, the advantage and the
credit. Here each level has its own network, discount, entropy and exploration, and the worker can learn from rounds of scenario builds
(--scenario-phases) without the manager having to.

    python -m balatro_rl.hier --out checkpoints/hier1.pt --attn 2 --joker-desc --spec
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .env import A_SELECT, KIND_OF, N_KINDS, N_SUB
from .model import ActorCritic

NARROW = {"pa": torch.float16, "af": torch.float16, "priv": torch.float16, "subs": torch.int8, "aid": torch.int16, "aid2": torch.int16}


# --------------------------------------------------------------------------- the returns (numpy, tested in tests/test_hier.py)
def worker_gae(rew, val, done, is_w, last_val, gamma, lam):
    """GAE over the worker's steps. rew / val / done / is_w: [T, N]; `done` marks the step that ends a round (a clear, a loss, or the end of a game): nothing is
    carried across it, and the value after it is 0. last_val [N]: the worker's value after the last step, for the envs still in a round. -> advantages [T, N]
    (0 where the manager acted)."""
    T, N = rew.shape
    adv = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros(N)
    nv = np.asarray(last_val, dtype=np.float64).copy()
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        g = delta + gamma * lam * nonterm * gae
        gae = np.where(is_w[t], g, 0.0)
        adv[t] = gae
        nv = np.where(is_w[t], val[t], 0.0)
    return adv


def manager_gae(rew, val, done, trunc, is_m, last_val, gamma, lam):
    """GAE over the manager's steps, a semi-Markov step each: the reward of a manager step is everything the engine paid from it up to the next manager step
    (the rounds the worker played in between included). `done`: the game ended at this step; `trunc`: an episode cut off (bootstrapped from the step's own value).
    last_val [N]: the manager's value after the rollout (nan where the env is in a round: the last manager step then bootstraps from its own value).
    -> advantages, window rewards [T, N] (0 where the worker acted)."""
    T, N = rew.shape
    adv = np.zeros((T, N), dtype=np.float32)
    win = np.zeros((T, N), dtype=np.float32)
    acc = np.zeros(N)
    dfl = np.zeros(N, dtype=bool)
    tfl = np.zeros(N, dtype=bool)
    nv = np.asarray(last_val, dtype=np.float64).copy()
    gnext = np.zeros(N)
    for t in reversed(range(T)):
        acc = acc + rew[t]
        dfl = dfl | (done[t] & ~trunc[t])
        tfl = tfl | (done[t] & trunc[t])
        m = is_m[t]
        if not m.any():
            continue
        own = val[t]
        nv_eff = np.where(np.isnan(nv), own, nv)
        cut = dfl | tfl
        boot = np.where(dfl, 0.0, np.where(tfl, own, nv_eff))
        delta = acc + gamma * boot - own
        g = delta + gamma * lam * (1.0 - cut) * gnext
        adv[t] = np.where(m, g, 0.0)
        win[t] = np.where(m, acc, 0.0)
        nv = np.where(m, own, nv)
        gnext = np.where(m, g, gnext)
        acc = np.where(m, 0.0, acc)
        dfl = np.where(m, False, dfl)
        tfl = np.where(m, False, tfl)
    return adv, win


def worker_reward(prog_before, prog_after_obs, cleared, done, engine_r, w_blind, posture, hands_left, round_hands, clear_bonus, bank_w, grow_w):
    """The worker's reward for each step (arrays of one shape). prog_*: chips / target so far (capped at 1); a step that clears the blind ends at 1; a step
    that ends the game by a loss reads its progress from the engine's part credit (0.5 x the blind's weight x the share scored). A clear pays clear_bonus and,
    by posture, bank_w x the share of the round's hands left over (posture 1) or grow_w x the share played (posture 2)."""
    pb = np.clip(prog_before, 0.0, 1.0)
    share_lost = np.minimum(1.0, engine_r / np.maximum(0.5 * w_blind, 1e-9))
    pa = np.where(cleared, 1.0, np.where(done, share_lost, np.clip(prog_after_obs, 0.0, 1.0)))
    left = np.maximum(hands_left - 1.0, 0.0) / np.maximum(round_hands, 1.0)
    bonus = np.where(posture == 1, bank_w * left, np.where(posture == 2, grow_w * (1.0 - left), 0.0))
    return (pa - pb) + cleared * (clear_bonus + bonus)


# --------------------------------------------------------------------------- the networks and the policy
def build_nets(a, dev):
    kw = dict(attn=a.attn, desc=int(a.joker_desc), spec=int(a.spec))
    worker = ActorCritic(goal_in=a.goals, **kw).to(dev)
    manager = ActorCritic(goal_out=a.goals, phi=int(getattr(a, "search", False)), **kw).to(dev)
    return worker, manager


def save_hier(path, worker, manager, extra=None):
    torch.save({"worker": worker.state_dict(), "manager": manager.state_dict(), "config": {"worker": worker.config, "manager": manager.config}, **(extra or {})}, path)


def load_hier(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    w = ActorCritic(**ck["config"]["worker"]).to(device)
    m = ActorCritic(**ck["config"]["manager"]).to(device)
    w.load_state_dict(ck["worker"], strict=False)
    m.load_state_dict(ck["manager"], strict=False)
    w.eval(); m.eval()
    return w, m


class Floor:
    """The exploration floor of train.py's policy(): a share eps of the behaviour spread over the legal kinds of action, evenly within each kind."""

    def __init__(self, dev):
        self.type_of = torch.as_tensor(KIND_OF, device=dev)
        oh = torch.zeros((len(KIND_OF), N_KINDS), device=dev)
        oh[torch.arange(len(KIND_OF)), torch.as_tensor(KIND_OF)] = 1.0
        self.onehot = oh

    def probs(self, logits, mask, eps):
        p = torch.softmax(logits, 1)
        if eps <= 0:
            return p
        m = mask.float()
        per_type = m @ self.onehot
        share = (per_type > 0).float()
        share = share / share.sum(1, keepdim=True).clamp(min=1e-9)
        floor = m * (share / per_type.clamp(min=1))[:, self.type_of]
        return (1 - eps) * p + eps * floor


# --------------------------------------------------------------------------- training
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--out", default="checkpoints/hier1.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu-fraction", type=float, default=0.0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--eval-seed0", type=int, default=9_000_000)
    p.add_argument("--record-wins", action="store_true")
    # network
    p.add_argument("--attn", type=int, default=2)
    p.add_argument("--joker-desc", action="store_true")
    p.add_argument("--spec", action="store_true")
    p.add_argument("--goals", type=int, default=3, help="postures the manager sets for the worker (0: none: the plain two-level split)")
    # optimisation
    p.add_argument("--lr-w", type=float, default=2e-4)
    p.add_argument("--lr-m", type=float, default=1.5e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--ent-w", type=float, default=0.003)
    p.add_argument("--ent-m", type=float, default=0.02)
    p.add_argument("--explore-w", type=float, default=0.0, help="the floor of the worker's behaviour (a share spread over the legal kinds)")
    p.add_argument("--explore-m", type=float, default=0.02, help="... and of the manager's")
    p.add_argument("--gamma-w", type=float, default=0.99)
    p.add_argument("--lam-w", type=float, default=0.95)
    p.add_argument("--gamma-m", type=float, default=0.999)
    p.add_argument("--lam-m", type=float, default=0.95)
    # rewards
    p.add_argument("--ante-weight", type=float, default=8.0)
    p.add_argument("--weight-power", type=float, default=2.0)
    p.add_argument("--win-bonus", type=float, default=30.0)
    p.add_argument("--m-scale", type=float, default=0.1, help="the manager's rewards are multiplied by this (the engine's run to ~60)")
    p.add_argument("--clear-bonus", type=float, default=0.5)
    p.add_argument("--bank-w", type=float, default=0.3)
    p.add_argument("--grow-w", type=float, default=0.3)
    # the practice worlds (as train.py)
    p.add_argument("--scenario-phases", default="", help="P1,P2,P3 updates: see train.scenario_schedule")
    p.add_argument("--scenario-p", type=float, default=0.0)
    p.add_argument("--scenario-mix", default="1,0,0")
    p.add_argument("--scenario-real-start", type=float, default=0.2)
    p.add_argument("--scenario-real-end", type=float, default=0.9)
    p.add_argument("--scenario-min-ante", type=int, default=1)
    p.add_argument("--scenario-max-ante", type=int, default=7)
    p.add_argument("--scenario-reroll-cap", type=int, default=3)
    p.add_argument("--scenario-couple", type=float, default=0.5)
    p.add_argument("--ban-jokers", default="")
    p.add_argument("--reroll-cap", type=int, default=0)
    p.add_argument("--hand-level-cap", type=int, default=0)
    p.add_argument("--flash-cap", type=int, default=0)
    p.add_argument("--start-money", type=int, default=0)
    p.add_argument("--start-money-p", type=float, default=0.0)
    # the shop-only game and the judge potential
    p.add_argument("--auto-round", action="store_true", help="the select action plays the whole round with the scripted player (engine Env::scripted_action); the manager decides outside rounds only; "
                                                              "with --goals 2 its posture picks the player's mode (0 score most, 1 scaler-friendly)")
    p.add_argument("--pace", type=float, default=1.5)
    p.add_argument("--judge", default="", help="checkpoint of the build judge (judge.py)")
    p.add_argument("--scaling", default="", help="checkpoint of the scaling model (judge.py)")
    p.add_argument("--phi-weight", type=float, default=0.0, help="the judge potential's weight (0: none): Phi is in [floor, 0]")
    p.add_argument("--phi-floor", type=float, default=-20.0)
    p.add_argument("--phi-full", action="store_true", help="the potential = the ante-8 boss with every scaler at its natural ante-8 value (engine growth table), no next-blind term")
    p.add_argument("--search", action="store_true", help="one-step look-ahead with the potential (lookahead.py): each buy / replace / sale is scored on a copy and goes in as an action feature; "
                                                         "needs --phi-weight / --judge; adds the potential head and the search loss to the manager")
    p.add_argument("--search-coef", type=float, default=0.5, help="weight of the manager's search loss: cross-entropy to softmax(potential change / tau) over the scored actions")
    p.add_argument("--search-tau", type=float, default=1.0, help="... tau, in nats of potential")
    p.add_argument("--phi-aux", type=float, default=0.2, help="weight of the manager's auxiliary head that predicts the potential of the state (units of 10 nats)")
    p.add_argument("--search-pool", type=int, default=1024, help="copies of the game made at once for the look-ahead")
    p.add_argument("--rnd-coef", type=float, default=0.0, help="random network distillation (0: off): the manager is paid coef x the (normalised) prediction error of a network that tries to match a fixed random one, "
                                                               "on the state it reaches: a bonus for builds and states it has not seen")
    p.add_argument("--rnd-lr", type=float, default=1e-3)
    p.add_argument("--phi-money", type=float, default=0.5, help="the potential's money term: this many nats for a full bank ($25)")
    p.add_argument("--blind-scale", type=float, default=1.0, help="the blind rewards (and the part credit of a lost blind) times this; the win bonus is not")
    p.add_argument("--dev-engine", default="", help="a directory with _engine_dev")
    return p.parse_args(argv)


def train(a):
    if a.dev_engine:
        import sys
        sys.path.insert(0, a.dev_engine)
        import _engine_dev
        import balatro_rl.sim as S
        sys.modules["balatro_rl.sim._engine"] = _engine_dev
        S._engine = _engine_dev
    from .engine_env import CppVecEnv, OBS_KEYS
    from .train import blind_weights, scenario_schedule, EpisodeStats
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = a.device
    cuda = str(dev).startswith("cuda")
    if cuda and a.gpu_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)
    use_amp = cuda and not a.no_amp
    N, T = a.envs, a.steps
    log_path = a.out.replace(".pt", "_log.jsonl")
    wins_path = a.out.replace(".pt", "_wins.jsonl")
    it0, games0, wins0 = 0, 0, 0
    worker, manager = build_nets(a, dev)
    if a.resume and os.path.exists(a.out) and os.path.exists(log_path):
        worker, manager = load_hier(a.out, dev)
        last = [json.loads(l) for l in open(log_path) if l.strip().startswith("{")][-1]
        it0, games0, wins0 = last["iter"], last["games"], last.get("wins", 0)
        print(json.dumps({"resumed": a.out, "iter": it0}), flush=True)
    opt_w = torch.optim.Adam(worker.parameters(), lr=a.lr_w, eps=1e-5)
    opt_m = torch.optim.Adam(manager.parameters(), lr=a.lr_m, eps=1e-5)
    venv = CppVecEnv(N, a.deck, a.stake, seed0=a.seed * 1_000_000 + 7 + it0 * 1_000_003, threads=a.threads,
                     reward_kw={"ante_weight": a.ante_weight, "win_bonus": a.win_bonus, "win_ante": 8, "shape_gamma": a.gamma_m})
    venv.set_weight_shape(a.ante_weight, a.weight_power)
    from .sim.jokers import JOKERS
    jkeys = list(JOKERS)

    def sandbox(v):
        if a.ban_jokers:
            v.set_ban([jkeys.index(k) for k in a.ban_jokers.split(",")], 1.0)
        if a.reroll_cap > 0 or a.hand_level_cap > 0 or a.flash_cap > 0:
            v.set_caps(a.reroll_cap, a.hand_level_cap, a.flash_cap)
        if a.start_money_p > 0 and a.start_money > 0:
            v.set_start(0.0, 0.0, a.start_money_p, a.start_money, 0.0, 0, 0.0)
    sandbox(venv)
    pot = None
    if a.auto_round:
        venv.set_auto_round(True, a.pace, 0)
    if a.blind_scale != 1.0:
        venv.set_reward(a.blind_scale, 0.0)
    if a.phi_weight > 0:
        from . import judge as J
        from .potential import JudgePotential
        venv.set_marginal(1)
        venv.set_marginal_blind_only(True)
        la = CppVecEnv(N, a.deck, a.stake, seed0=1, threads=a.threads)
        la.set_marginal(1)
        pot = JudgePotential(venv, la, J.load(a.judge, dev), J.load(a.scaling, dev), dev, a.phi_floor, a.phi_money, a.phi_full)
    search = None
    if a.search:
        assert pot is not None and a.phi_full, "--search needs --phi-weight, --judge, --scaling and --phi-full"
        assert a.eval_games <= N, "--search: the evaluation's potential runs in the training pools (--eval-games <= --envs)"
        from .lookahead import ShopSearch, PA_SCALE
        from .env import N_ACTIONS
        pool = CppVecEnv(a.search_pool, a.deck, a.stake, seed0=2, threads=a.threads)
        proj = CppVecEnv(a.search_pool, a.deck, a.stake, seed0=3, threads=a.threads)
        proj.set_marginal(1)
        search = ShopSearch(pot, pool, proj, N_ACTIONS)
    w_tab = torch.tensor(blind_weights(a.ante_weight, a.weight_power), device=dev)
    scen_on = a.scenario_p > 0 or bool(a.scenario_phases)
    floor = Floor(dev)
    rnd = None
    if a.rnd_coef > 0:
        from .env import VOCAB_SIZE, F_JOKER_W as FJW
        dim = ob_dim = None

        def rnd_feat(g, jid, jf, cid):
            B = g.shape[0]
            mh = torch.zeros(B, VOCAB_SIZE, device=g.device).scatter_add_(1, jid.long(), torch.ones_like(jid, dtype=torch.float32))
            mc = torch.zeros(B, VOCAB_SIZE, device=g.device).scatter_add_(1, cid.long(), torch.ones_like(cid, dtype=torch.float32))
            mh[:, 0] = 0; mc[:, 0] = 0
            x = torch.cat([g.float(), jf[..., :FJW].float().flatten(1), mh, mc], 1)
            return torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
        rd = 210 + 8 * FJW + 2 * VOCAB_SIZE
        r_tgt = torch.nn.Sequential(torch.nn.Linear(rd, 256), torch.nn.ReLU(), torch.nn.Linear(256, 64)).to(dev)
        for q in r_tgt.parameters():
            q.requires_grad_(False)
        r_pred = torch.nn.Sequential(torch.nn.Linear(rd, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 64)).to(dev)
        r_opt = torch.optim.Adam(r_pred.parameters(), lr=a.rnd_lr)
        rnd = {"mean": 0.0, "var": 1.0, "n": 1e-4}

        @torch.no_grad()
        def rnd_err(g, jid, jf, cid):
            x = rnd_feat(g, jid, jf, cid)
            return ((r_pred(x) - r_tgt(x)) ** 2).mean(1)
    from .joker_desc import FEATURES as DF, item_descriptors
    from .env import item_id
    jcat = torch.tensor(np.stack([item_descriptors()[:, DF.index(n)] for n in ("mult", "xmult", "grows", "economy", "chips", "is_joker")], 1),
                        dtype=torch.float32, device=dev)
    buy_kinds = torch.tensor([i for i, n in enumerate(["play", "discard", "select", "skip", "reroll_boss", "buy", "buy_pack", "voucher", "reroll", "leave",
                                                       "sell_joker", "sell_cons", "use_cons", "pick", "pack_skip", "swap_joker", "replace"]) if n in ("buy", "replace", "pick")], device=dev)

    def current():
        out = {}
        for k in OBS_KEYS:
            t = torch.from_numpy(venv.obs[k])
            if k in ("af", "priv") and cuda:
                t = t.half()
            out[k] = t.to(dev) if cuda else t.clone()
        return out

    def attach(o, v):
        """Fills o["pa"] (the look-ahead's per-action feature) for the environments outside a round; -> the potential of those environments' states, [N] (nan elsewhere)."""
        n_ = o["g"].shape[0]
        phi_now = np.full(n_, np.nan, dtype=np.float32)
        if search is None:
            return phi_now
        m_ids = torch.nonzero(o["g"][:, 5] <= 0.5).squeeze(1).cpu().numpy()
        ph = pot.phi(m_ids, src=v)
        phi_now[m_ids] = ph
        o["pa"] = torch.from_numpy(search.run(v, np.asarray(v.obs["mask"]), m_ids, ph)).to(torch.float16).to(dev)
        return phi_now

    def forward(net, o):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            lg, v = net(o)
        return lg.float(), v.float()

    def sub(o, idx):
        return {k: v[idx] for k, v in o.items()}

    # ------------------------------------------------------------------ the fixed evaluation (greedy, unmodified games, the same seeds every time)
    def fixed_eval(n):
        ev = CppVecEnv(n, a.deck, a.stake, seed0=a.eval_seed0, threads=a.threads)
        sandbox(ev)
        if a.auto_round:
            ev.set_auto_round(True, a.pace, 0)
            ev.set_weight_shape(a.ante_weight, a.weight_power)
        fin = np.zeros(n, dtype=bool)
        ante = np.ones(n, dtype=np.int64); won = np.zeros(n, dtype=bool); blinds = np.zeros(n, dtype=np.int64); steps = np.zeros(n, dtype=np.int64)
        post = torch.zeros(n, dtype=torch.long, device=dev)
        worker.eval(); manager.eval()
        with torch.no_grad():
            for _ in range(6000):
                o = {k: torch.from_numpy(ev.obs[k]).to(dev) for k in OBS_KEYS}
                o["goal"] = post
                if search is not None:
                    o["pa"] = torch.zeros((n, o["mask"].shape[1], 2), dtype=torch.float16, device=dev)
                    attach(o, ev)
                inr = o["g"][:, 5] > 0.5
                act = torch.zeros(n, dtype=torch.long, device=dev)
                iw = torch.nonzero(inr).squeeze(1); im = torch.nonzero(~inr).squeeze(1)
                if iw.numel():
                    act[iw] = forward(worker, sub(o, iw))[0].argmax(1)
                if im.numel():
                    lg, _ = forward(manager, sub(o, im))
                    act[im] = lg.argmax(1)
                    if a.goals:
                        gsel = manager.goal_logits().float().argmax(1)
                        sel = act[im] == A_SELECT
                        post[im[sel]] = gsel[sel]
                an = np.where(fin, -1, act.cpu().numpy())
                if a.auto_round:
                    ev.set_modes(post.cpu().numpy())
                _, d, info = ev.step(an)
                first = d & ~fin
                ante[first], won[first], blinds[first] = info["ante"][first], info["won"][first], info["blinds"][first]
                steps += ~fin
                fin |= d
                post[torch.from_numpy(d).to(dev)] = 0
                if fin.all():
                    break
        ev.close()
        furthest = np.where(won, 9, ante)
        return {"games": n, "win_rate": round(float(won.mean()), 4), "blinds": round(float(blinds.mean()), 3), "mean_len": round(float(steps.mean()), 1),
                "ante_hist": [int((furthest == k).sum()) for k in range(1, 10)]}

    # ------------------------------------------------------------------ buffers
    ob = current()
    ob["goal"] = torch.zeros(N, dtype=torch.long, device=dev)
    if search is not None:
        ob["pa"] = torch.zeros((N, ob["mask"].shape[1], 2), dtype=torch.float16, device=dev)
    phi_cur = attach(ob, venv)
    phi_buf = np.full((T, N), np.nan, dtype=np.float32)
    buf = {k: torch.empty((T, N) + tuple(v.shape[1:]), dtype=NARROW.get(k, v.dtype), device=dev) for k, v in ob.items()}
    acts = torch.zeros((T, N), dtype=torch.long, device=dev)
    lps = torch.zeros((T, N), device=dev)                      # log-prob of the action (the manager's: + the posture's where one was set)
    goals_t = torch.zeros((T, N), dtype=torch.long, device=dev)
    sets_goal = torch.zeros((T, N), dtype=torch.bool, device=dev)
    vals = np.zeros((T, N), dtype=np.float32)
    is_w = np.zeros((T, N), dtype=bool)
    rews = np.zeros((T, N), dtype=np.float32)
    dones = np.zeros((T, N), dtype=bool)
    truncs = np.zeros((T, N), dtype=bool)
    cleared = np.zeros((T, N), dtype=bool)
    kbuf = np.full((T, N), -1, dtype=np.int8)
    gcols = np.zeros((T, N, 8), dtype=np.float32)               # g columns read before the step: ante, blind one-hot (3), progress, hands left, round hands
    prog_after = np.zeros((T, N), dtype=np.float32)
    in_round_next = np.zeros((T, N), dtype=bool)
    post = torch.zeros(N, dtype=torch.long, device=dev)
    prev_blinds = np.zeros(N, dtype=np.int64)
    phi_prev = np.full(N, np.nan)
    pot_sum = 0.0
    if pot is not None:
        ids0 = np.flatnonzero(np.asarray(venv.obs["g"])[:, 4] > 0.5)
        phi_prev[ids0] = pot.phi(ids0)
    stats = EpisodeStats(n=2000)
    rnd_raw = np.zeros((T, N), dtype=np.float32)
    rnd_in = np.zeros((T, N), dtype=bool)
    games, wins = games0, wins0
    ep_count = np.zeros(N, dtype=np.int64)
    hist = [[] for _ in range(N)]
    best_eval = -1.0
    t0 = time.time()

    for it in range(it0 + 1, a.iters + 1):
        if scen_on:
            p_now, mix_now = scenario_schedule(a, it)
            venv.set_scenario(p_now, *mix_now, a.scenario_min_ante, a.scenario_max_ante, a.scenario_reroll_cap, a.scenario_couple)
        scn_stat = np.zeros((3, 4), dtype=np.int64)
        buy_acc = torch.zeros(8, 6, device=dev)
        rounds = np.zeros((8, 2), dtype=np.int64)               # worker rounds by ante: [cleared, lost]
        post_count = np.zeros(max(a.goals, 1), dtype=np.int64)
        kinds = torch.zeros(N_KINDS, device=dev)
        sim_s = 0.0
        it_t0 = time.time()
        worker.eval(); manager.eval()
        for t in range(T):
            ob["goal"] = post
            for k in buf:
                buf[k][t] = ob[k]
            phi_buf[t] = phi_cur / 10.0
            inr = ob["g"][:, 5] > 0.5
            kb_t = venv.scn_kinds() if scen_on else np.full(N, -1)
            iw = torch.nonzero(inr).squeeze(1); im = torch.nonzero(~inr).squeeze(1)
            act = torch.zeros(N, dtype=torch.long, device=dev); lp = torch.zeros(N, device=dev); val = torch.zeros(N, device=dev)
            goal = torch.zeros(N, dtype=torch.long, device=dev); setg = torch.zeros(N, dtype=torch.bool, device=dev)
            with torch.no_grad():
                if iw.numel():
                    o = sub(ob, iw)
                    lg, v = forward(worker, o)
                    pr = floor.probs(lg, o["mask"], a.explore_w)
                    dist = torch.distributions.Categorical(probs=pr, validate_args=False)
                    aw = dist.sample()
                    act[iw], lp[iw], val[iw] = aw, dist.log_prob(aw), v
                if im.numel():
                    o = sub(ob, im)
                    lg, v = forward(manager, o)
                    pr = floor.probs(lg, o["mask"], a.explore_m)
                    dist = torch.distributions.Categorical(probs=pr, validate_args=False)
                    am = dist.sample()
                    lpm = dist.log_prob(am)
                    if a.goals:
                        gd = torch.distributions.Categorical(logits=manager.goal_logits().float(), validate_args=False)
                        gs = gd.sample()
                        sel = am == A_SELECT
                        lpm = lpm + gd.log_prob(gs) * sel
                        goal[im], setg[im] = gs, sel
                    act[im], lp[im], val[im] = am, lpm, v
                    kinds += torch.bincount(floor.type_of[am], minlength=N_KINDS)
                    bought = torch.isin(floor.type_of[am], buy_kinds)
                    if bool(bought.any()):
                        ids = o["aid"].gather(1, am[:, None]).squeeze(1)[bought].long()
                        ai = (o["g"][:, 0] * 8.0).round().clamp(1, 8).long()[bought] - 1
                        keep = torch.from_numpy(kb_t[im.cpu().numpy()] < 0).to(dev)[bought]
                        if bool(keep.any()):
                            buy_acc.index_add_(0, ai[keep], jcat[ids[keep]])
            acts[t], lps[t], goals_t[t], sets_goal[t] = act, lp, goal, setg
            vals[t] = val.cpu().numpy()
            is_w[t] = inr.cpu().numpy()
            g0 = ob["g"].float()
            gcols[t, :, :7] = torch.stack([(g0[:, 0] * 8.0).round().clamp(1, 8), g0[:, 1], g0[:, 2], g0[:, 3], g0[:, 9], g0[:, 11] * 5.0, g0[:, 13] * 5.0], 1).cpu().numpy()
            kbuf[t] = kb_t
            act_np = act.cpu().numpy()
            if a.auto_round and a.goals:
                venv.set_modes(torch.where(setg, goal, post).cpu().numpy())      # the posture chosen at this select is the player's mode for the round
            ts = time.time()
            r, d, info, _, tr = venv.step_split(act_np)
            sim_s += time.time() - ts
            if pot is not None:
                g_new = np.asarray(venv.obs["g"])
                ids = np.flatnonzero(g_new[:, 4] > 0.5)               # at a blind select now (the first of a new game included)
                phi_new = pot.phi(ids)
                r = np.array(r, dtype=np.float64)
                for e in np.flatnonzero(d):                            # a game that ended: a win is 0, a loss the floor
                    if not np.isnan(phi_prev[e]):
                        r[e] += a.phi_weight * ((0.0 if info["won"][e] else a.phi_floor) - phi_prev[e])
                    phi_prev[e] = np.nan
                for e, pn in zip(ids, phi_new):
                    if not d[e] and not np.isnan(phi_prev[e]):
                        r[e] += a.phi_weight * (pn - phi_prev[e])
                    phi_prev[e] = pn
            if rnd is not None:                                    # the novelty of the state the step reached (a manager state only; a finished game is paid nothing)
                on = current()
                e_ = rnd_err(on["g"].to(dev), on["jid"].to(dev), on["jf"].to(dev), on["cid"].to(dev)).cpu().numpy()
                outside = (on["g"][:, 5].cpu().numpy() <= 0.5) & ~np.asarray(d)
                rnd_raw[t] = np.where(outside, e_, 0.0)
                rnd_in[t] = outside
            rews[t], dones[t], truncs[t] = r, d, tr
            cleared[t] = info["blinds"] > prev_blinds
            prev_blinds = np.where(d, 0, info["blinds"])
            if a.record_wins:
                for e in range(N):
                    hist[e].append(int(act_np[e]))
            # the posture the manager chose is held until the game ends
            if a.goals:
                post = torch.where(setg, goal, post)
            for e in np.flatnonzero(d):
                real = kbuf[t, e] < 0
                if not real:
                    k_ = int(kbuf[t, e])
                    scn_stat[k_] += [1, int(tr[e]), int(not tr[e] and not info["won"][e]), int(info["won"][e])]
                else:
                    stats.add({"episode_end": True, "blinds": int(info["blinds"][e]), "ante": int(info["ante"][e]), "won": bool(info["won"][e])})
                    if info["won"][e]:
                        wins += 1
                        with open(wins_path, "a") as f:
                            f.write(json.dumps({"iter": it, "games": games, "seed": int(venv.seed0 + e + N * ep_count[e]), "stake": a.stake,
                                                **({"actions": hist[e]} if a.record_wins else {})}) + "\n")
                games += 1
                ep_count[e] += 1
                hist[e] = []
            if d.any():
                post[torch.from_numpy(d).to(dev)] = 0
            ob = current()
            ob["goal"] = post
            if search is not None:
                ob["pa"] = torch.zeros((N, ob["mask"].shape[1], 2), dtype=torch.float16, device=dev)
            phi_cur = attach(ob, venv)
            gn = ob["g"].float()
            both = torch.stack([gn[:, 9], (gn[:, 5] > 0.5).float()], 1).cpu().numpy()
            prog_after[t], in_round_next[t] = both[:, 0], both[:, 1] > 0.5
        if a.goals:                                              # the postures chosen, for the log
            post_count = np.bincount(goals_t[sets_goal].cpu().numpy(), minlength=a.goals)
        # ------------------------------------------------------------------ the returns
        with torch.no_grad():
            ob["goal"] = post
            inr = ob["g"][:, 5] > 0.5
            last_w = np.zeros(N); last_m = np.full(N, np.nan)
            iw = torch.nonzero(inr).squeeze(1); im = torch.nonzero(~inr).squeeze(1)
            if iw.numel():
                last_w[iw.cpu().numpy()] = forward(worker, sub(ob, iw))[1].cpu().numpy()
            if im.numel():
                last_m[im.cpu().numpy()] = forward(manager, sub(ob, im))[1].cpu().numpy()
        for name_, arr_ in (("rews", rews), ("vals", vals), ("prog_after", prog_after)):
            if not np.isfinite(arr_).all():
                bad_ = np.argwhere(~np.isfinite(arr_))[:3].tolist()
                raise RuntimeError(f"non-finite {name_} at (t, env) {bad_}")
        w_blind = np.asarray(w_tab.cpu().numpy())[(3 * (gcols[..., 0].astype(np.int64) - 1) + gcols[..., 1:4].argmax(-1)).clip(0, 23)]
        posture_np = buf["goal"].cpu().numpy()
        rew_w = worker_reward(gcols[..., 4], prog_after, cleared, dones, rews, w_blind, posture_np, gcols[..., 5], gcols[..., 6], a.clear_bonus, a.bank_w, a.grow_w)
        done_w = dones | ~in_round_next
        adv_w = worker_gae(rew_w, vals, done_w.astype(np.float64), is_w, last_w, a.gamma_w, a.lam_w)
        is_m = ~is_w
        train_m = is_m & (kbuf != 0)                          # (a one-blind scenario has nothing to decide outside its round)
        rnd_row = {}
        if rnd is not None:
            vals_ = rnd_raw[rnd_in]
            if vals_.size:
                bm, bv, bn = vals_.mean(), vals_.var(), vals_.size
                d_ = bm - rnd["mean"]; tot = rnd["n"] + bn
                rnd["mean"] += d_ * bn / tot
                rnd["var"] = (rnd["var"] * rnd["n"] + bv * bn + d_ ** 2 * rnd["n"] * bn / tot) / tot
                rnd["n"] = min(tot, 1e6)
            bonus = np.where(rnd_in, np.clip(rnd_raw / np.sqrt(rnd["var"] + 1e-8), 0.0, 5.0), 0.0) * a.rnd_coef
            rnd_row = {"err": round(float(vals_.mean()) if vals_.size else 0.0, 5), "bonus": round(float(bonus[rnd_in].mean()) if rnd_in.any() else 0.0, 4)}
        else:
            bonus = 0.0
        adv_m, win_r = manager_gae(rews * a.m_scale + bonus, vals, dones, truncs, is_m, last_m, a.gamma_m, a.lam_m)
        if not np.isfinite(adv_m).all() or not np.isfinite(last_m[~np.isnan(last_m)]).all():
            raise RuntimeError("non-finite manager advantages: last_m %s" % np.asarray(last_m)[~np.isfinite(last_m)][:5])
        ret_w = adv_w + vals
        ret_m = adv_m + vals
        for t in range(T):                                    # worker rounds by ante (cleared / lost)
            sel = is_w[t] & done_w[t]
            if sel.any():
                an = (gcols[t, sel, 0].astype(int) - 1)
                clr = cleared[t, sel]
                for x, c in zip(an, clr):
                    rounds[x, 0 if c else 1] += 1
        # ------------------------------------------------------------------ the updates
        big = {k: v.reshape((T * N,) + tuple(v.shape[2:])) for k, v in buf.items()}
        phi_flat = torch.from_numpy(phi_buf.reshape(-1)).to(dev)
        flat_act, flat_lp, flat_goal, flat_sg = acts.reshape(-1), lps.reshape(-1), goals_t.reshape(-1), sets_goal.reshape(-1)

        def update(net, opt, mask_np, adv_np, ret_np, ent_coef, is_manager):
            idx_all = np.flatnonzero(mask_np.reshape(-1))
            if len(idx_all) < 16:
                return {}
            adv_t = torch.tensor(adv_np.reshape(-1)[idx_all], device=dev)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            ret_t = torch.tensor(ret_np.reshape(-1)[idx_all], device=dev)
            idx_t = torch.from_numpy(idx_all).to(dev)
            net.train()
            pls, vls, ents, kls, sls, hit, pls_ = [], [], [], [], [], [], []
            for _ in range(a.epochs):
                perm = torch.randperm(len(idx_all), device=dev)
                for s0 in range(0, len(idx_all), a.batch):
                    mb = perm[s0:s0 + a.batch]
                    ii = idx_t[mb]
                    o = {k: v[ii] for k, v in big.items()}
                    lg, v = forward(net, o)
                    pr = floor.probs(lg, o["mask"], a.explore_m if is_manager else a.explore_w)
                    dist = torch.distributions.Categorical(probs=pr, validate_args=False)
                    lp = dist.log_prob(flat_act[ii])
                    ent = dist.entropy()
                    if is_manager and a.goals:
                        gd = torch.distributions.Categorical(logits=net.goal_logits().float(), validate_args=False)
                        sg = flat_sg[ii]
                        lp = lp + gd.log_prob(flat_goal[ii]) * sg
                        ent = ent + gd.entropy() * sg
                    ratio = torch.exp(lp - flat_lp[ii])
                    s1 = ratio * adv_t[mb]
                    s2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv_t[mb]
                    ploss = -torch.min(s1, s2).mean()
                    vloss = F.mse_loss(v, ret_t[mb])
                    loss = ploss + a.vf * vloss - ent_coef * ent.mean()
                    if is_manager and search is not None:
                        fl = o["pa"][..., 1] > 0.5
                        dA = o["pa"][..., 0].float() * PA_SCALE
                        has = fl.sum(1) >= 2
                        if bool(has.any()):
                            q = torch.softmax((dA / a.search_tau).masked_fill(~fl, -1e9), 1)
                            logp = torch.log_softmax(lg.masked_fill(~fl, -1e9), 1)
                            sl = -(q * logp * fl).sum(1)[has].mean()
                            loss = loss + a.search_coef * sl
                            sls.append(sl.detach()); hit.append((lg.masked_fill(~fl, -1e9).argmax(1) == dA.masked_fill(~fl, -1e9).argmax(1))[has].float().mean().detach())
                        tgt = phi_flat[ii]
                        okp = torch.isfinite(tgt)
                        if bool(okp.any()):
                            pl = F.mse_loss(net.phi_pred().float()[okp], tgt[okp])
                            loss = loss + a.phi_aux * pl
                            pls_.append(pl.detach())
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
                    pls.append(ploss.detach()); vls.append(vloss.detach()); ents.append(ent.mean().detach()); kls.append((flat_lp[ii] - lp).mean().detach())
            net.eval()
            f = lambda xs: round(float(torch.stack(xs).mean()), 4)
            out = {"n": int(len(idx_all)), "pi": f(pls), "v": f(vls), "ent": f(ents), "kl": f(kls)}
            if sls:
                out.update({"search": f(sls), "agree": f(hit)})
            if pls_:
                out["phi_mse"] = f(pls_)
            return out
        uw = update(worker, opt_w, is_w, adv_w, ret_w, a.ent_w, False)
        um = update(manager, opt_m, train_m, adv_m, ret_m, a.ent_m, True)
        if rnd is not None:
            sel = torch.from_numpy(np.flatnonzero(rnd_in.reshape(-1))).to(dev)
            if len(sel) > 16:
                sel = sel[torch.randperm(len(sel), device=dev)[:max(16, len(sel) // 2)]]
                for s0 in range(0, len(sel), a.batch):
                    ii = sel[s0:s0 + a.batch]
                    x = rnd_feat(big["g"][ii], big["jid"][ii], big["jf"][ii], big["cid"][ii])
                    with torch.no_grad():
                        y = r_tgt(x)
                    lo = ((r_pred(x) - y) ** 2).mean()
                    r_opt.zero_grad(); lo.backward(); r_opt.step()
        if cuda:
            torch.cuda.empty_cache()
        # ------------------------------------------------------------------ log, save, evaluate
        row = {"iter": it, "games": games, "wins": wins, **stats.summary(), "ent": um.get("ent"), "v": um.get("v"), "pi": um.get("pi"), "kl": um.get("kl"),
               "sps": int(T * N / max(1e-9, time.time() - it_t0)), "min": round((time.time() - t0) / 60, 1), "phase": "hier",
               "worker": uw, "manager": um, **({"rnd": rnd_row} if rnd is not None else {}),
               **({"phi_mean": round(float(np.nanmean(phi_prev)), 3), "judge_calls": pot.calls} if pot is not None else {}),
               "round_clear": [round(float(c / max(1, c + l)), 3) for c, l in rounds], "rounds_by_ante": [int(c + l) for c, l in rounds],
               "posture": [int(x) for x in post_count] if a.goals else [],
               "kinds": {n: int(x) for n, x in zip(["play", "discard", "select", "skip", "reroll_boss", "buy", "buy_pack", "voucher", "reroll", "leave", "sell_joker", "sell_cons", "use_cons",
                                                    "pick", "pack_skip", "swap_joker", "replace"], kinds.cpu().numpy()) if x},
               "buy_share": [[round(float(x), 3) for x in row_[:5] / row_[5].clamp(min=1)] + [int(row_[5])] for row_ in buy_acc.cpu()],
               **({"scn_p": round(p_now, 3), "scn": {n: dict(zip(("games", "cut", "lost", "won"), [int(x) for x in scn_stat[k]]))
                   for k, n in enumerate(("one_blind", "few_blinds", "full_run"))}} if scen_on else {})}
        if a.eval_every and it % a.eval_every == 0:
            ev = fixed_eval(a.eval_games)
            row["eval"], row["eval_stake"] = ev, a.stake
            if ev["blinds"] > best_eval:
                best_eval = ev["blinds"]
                save_hier(a.out.replace(".pt", "_best.pt"), worker, manager, {"iter": it, "eval": ev})
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if it % a.save_every == 0:
            save_hier(a.out, worker, manager, {"iter": it})
    save_hier(a.out, worker, manager, {"iter": a.iters})


if __name__ == "__main__":
    train(parse())
