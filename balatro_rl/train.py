"""Training: behaviour cloning (DAgger) from the heuristic, then PPO.

    # 1) imitate the rule-based player (fast warm start)
    python -m balatro_rl.train bc  --iters 150 --envs 8  --out checkpoints/bc.pt
    # 2) reinforcement learning on top
    python -m balatro_rl.train ppo --init checkpoints/bc.pt --iters 3000 --envs 16 --out checkpoints/ppo.pt
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import ActorCritic, batch_obs, stack_obs, save_model, load_model, OBS_KEYS
from .vec_env import VecEnv
from .streams import (N_ANTES, N_OUT, EntropyBump, MilestoneFade, ReachTracker, ReturnNorm, StallDetector, by_ante, explained_variance,
                      gae as gae_streams)


def compact(o: dict) -> dict:
    """Shrink an observation for storage (action features to float16)."""
    o = dict(o)
    o["af"] = o["af"].astype(np.float16)
    if "priv" in o:
        o["priv"] = o["priv"].astype(np.float16)              # (2000 columns with the future list: the rollout holds them all)
    return o


class EpisodeStats:
    def __init__(self, n=200):
        self.blinds = collections.deque(maxlen=n)
        self.antes = collections.deque(maxlen=n)
        self.wins = collections.deque(maxlen=n)
        self.returns = collections.deque(maxlen=n)

    def add(self, info):
        if info.get("episode_end"):
            self.blinds.append(info["blinds"])
            self.antes.append(info["ante"])
            self.wins.append(float(info["won"]))

    def summary(self):
        if not self.blinds:
            return {}
        out = {"episodes": len(self.blinds), "blinds": round(float(np.mean(self.blinds)), 2),
               "ante": round(float(np.mean(self.antes)), 2), "win%": round(100 * float(np.mean(self.wins)), 2)}
        if self.returns:
            out["ret"] = round(float(np.mean(self.returns)), 3)
        return out


PACK_KINDS = ("arcana", "celestial", "standard", "buffoon", "spectral")


def pack_bias_list(spec: str) -> list:
    """"arcana:3,standard:3,celestial:2" -> the five weights multipliers in the engine's order (the rest 1)."""
    w = {k: 1.0 for k in PACK_KINDS}
    for part in [x for x in spec.split(",") if x.strip()]:
        k, v = part.split(":")
        assert k.strip() in w, f"--pack-bias: unknown pack kind {k!r} (of {PACK_KINDS})"
        w[k.strip()] = float(v)
    return [w[k] for k in PACK_KINDS]


def blind_weights(ante_weight: float, power: float, geom: float = 0.0):
    """The 24 blind rewards as the engine makes them (Env::set_weights): 1 at ante 1 to ante_weight at ante 8 along (ante / 7) ** power, summing to 24."""
    if geom > 0:                                        # --weight-geom: 1, r, r^2 ... by ante
        raw = np.array([geom ** (b // 3) for b in range(24)])
    else:
        raw = np.array([1 + (ante_weight - 1) * ((b // 3) / 7) ** power if power != 1.0 else 1 + (ante_weight - 1) * (b // 3) / 7 for b in range(24)])
    return (raw * 24 / raw.sum()).astype(np.float32)


SHAPING = ("strength_bonus", "income_bonus", "power_bonus", "bank_bonus", "build_bonus", "grow_bonus")


def shaping_budget(a) -> dict:
    """An upper bound, per term, on the shaping reward one episode can pay (a run of 24 blinds): what a policy could collect by farming the shaping alone.
    Per-round terms are paid at most once a round and the weights of the 24 blinds sum to 24; the bank and build potentials net to zero over a game
    (their transient peak is counted); the growth potential is kept at the end of the game: 5 jokers whose counters reach 1e4 (log scale)."""
    import math
    W = blind_weights(a.ante_weight, a.weight_power, getattr(a, "weight_geom", 0.0))
    rel = a.shape_relative
    total, peak = (float(W.sum()), float(W.max())) if rel else (24.0, 1.0)

    def top(x, xf):
        return max(x, xf) if xf is not None else x
    g = lambda n: top(getattr(a, n), getattr(a, n + "_final", None))
    return {"strength": g("strength_bonus") * total, "income": g("income_bonus") * a.income_cap * total, "power": g("power_bonus") * total,
            "bank": g("bank_bonus") * peak, "build": g("build_bonus") * peak * (1.0 + a.build_second + 0.84),
            "grow": g("grow_bonus") * peak * 5 * math.log1p(1e4) / 4.0, "verdant": a.verdant_bonus * peak}


def enforce_win_ratio(a):
    """--win-ratio R: the win bonus must exceed R times the most shaping an episode can pay (shaping_budget), or every shaping coefficient is scaled down
    until it does, so that no amount of farming the shaping is worth more than a tenth of a win."""
    if a.win_ratio <= 0:
        return
    win = a.stream_win_bonus if a.streams else a.win_bonus
    budget = shaping_budget(a)
    total = sum(budget.values())
    need = a.win_ratio * total
    f = 1.0 if win >= need else win / need
    if f < 1.0:
        for n in SHAPING:
            setattr(a, n, getattr(a, n) * f)
            if getattr(a, n + "_final", None) is not None:
                setattr(a, n + "_final", getattr(a, n + "_final") * f)
        a.verdant_bonus *= f
    print(json.dumps({"win_ratio": a.win_ratio, "win_bonus": win, "shaping_budget": {k: round(v, 3) for k, v in budget.items()}, "budget_total": round(total, 3),
                      "win_needed": round(need, 1), "shaping_scaled_by": round(f, 4)}), flush=True)


def blind_log_table():
    """[9, 3] log10(blind target + 1) of each ante (1-8, row 0 unused) and blind (small, big, boss): the reference the strength bonus looks ahead to."""
    from .sim.game import Game
    g = Game(seed=0)
    out = np.zeros((9, 3), dtype=np.float32)
    for ante in range(1, 9):
        g.ante = ante
        for b in range(3):
            out[ante, b] = np.log10(g.blind_target(b) + 1.0)
    return out


def seed_archive(a, venv, dev, jlog=print):
    """--archive-seed-ckpt: fill the archive before training with the blind starts of games that were WON by another network (played greedily from a rich start,
    --archive-seed-money), so the late game has states a win came from; the games share the training archive and keep snapshots only from games that win."""
    from .engine_env import CppVecEnv
    from .engine_env import OBS_KEYS as _OK
    from .model import load_model
    net = load_model(a.archive_seed_ckpt, dev)
    net.eval()
    for money in [int(x) for x in a.archive_seed_money.split(",")]:
        ve = CppVecEnv(128, a.deck, a.stake, seed0=7_700_000 + 1000 * money, threads=a.threads)
        ve.v.share_archive(venv.v)
        ve.set_start(0.0, 0.0, 1.0, money)
        ve.set_archive(0.0, 1.0, a.archive_seed_ante, 8, a.archive_cap, a.archive_min, True)
        ve.v.restart()
        done_games, t0 = 0, time.time()
        while done_games < a.archive_seed_games:
            o = {k: torch.as_tensor(np.asarray(ve.obs[k])).to(dev) for k in _OK if k in ve.obs}
            with torch.no_grad():
                logits, _ = net(o)
            _, d, _ = ve.step(logits.argmax(1).cpu().numpy())
            done_games += int(d.sum())
        ve.close()
        print(json.dumps({"archive_seed": {"money": money, "games": done_games, "secs": round(time.time() - t0, 1), "fill": [int(x) for x in venv.archive_counts(-1)]}}), flush=True)


def plant_archive(a, venv, jlog=print):
    """--plant-file: put the planted builds (tools/plant_builds.py: strong builds per ante, found by random search and hill-climbing) into the archive, each
    at the start of every blind of its ante, for the reverse curriculum (--archive-ante reverse: games start at ante 8 and the start moves back as they are won)."""
    import sys as _sys
    from .engine_env import CppVecEnv
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    from plant_builds import place
    d = json.load(open(a.plant_file))
    items = [(b, blind) for ante, bs in d["builds"].items() for b in bs[:a.plant_per_ante] for blind in range(3)]
    ve = CppVecEnv(512, a.deck, a.stake, seed0=8_800_000, threads=a.threads)
    ve.v.share_archive(venv.v)
    put = 0
    for c0 in range(0, len(items), ve.n):
        chunk = items[c0:c0 + ve.n]
        ve.v.restart()
        for i, (b, blind) in enumerate(chunk):
            place(ve, i, b, blind)
        put += int(ve.v.archive_put(list(range(len(chunk)))))
    ve.close()
    jlog(json.dumps({"planted": {"builds": put, "file": a.plant_file, "fill": [int(x) for x in venv.archive_counts(-1)]}}))


def scenario_schedule(a, it: int):
    """-> (share of games that are scenarios, (w_one, w_few, w_full)) at iteration `it` (engine Env::apply_scenario).
    --scenario-phases P1,P2,P3: phase 1 (P1 iterations) mostly one-blind scenarios, to teach play and the critic what builds are worth; phase 2 (P2)
    mostly several blinds with shops and random money; phase 3 (P3, then on) real runs from ante 1 rise from --scenario-real-start to
    --scenario-real-end of the games, the scenarios that remain being mostly longer ones. Without phases: --scenario-p of the games, --scenario-mix."""
    if not a.scenario_phases:
        w = [float(x) for x in a.scenario_mix.split(",")]
        return a.scenario_p, tuple(w)
    p1, p2, p3 = [int(x) for x in a.scenario_phases.split(",")]
    if it <= p1:
        return 1.0 - a.scenario_real_start, (0.85, 0.10, 0.05)
    if it <= p1 + p2:
        return 1.0 - a.scenario_real_start, (0.15, 0.70, 0.15)
    f = min(1.0, (it - p1 - p2) / max(1, p3))
    real = a.scenario_real_start + (a.scenario_real_end - a.scenario_real_start) * f
    return 1.0 - real, (0.10, 0.45, 0.45)


FLAT_MULT = ("abstract,bootstraps,crazy,droll,erosion,even_steven,gluttenous_joker,greedy_joker,gros_michel,half,joker,jolly,lusty_joker,mad,misprint,"
             "mystic_summit,onyx_agate,raised_fist,scholar,shoot_the_moon,smiley,swashbuckler,walkie_talkie,wrathful_joker,zany,fibonacci")   # +mult, not a multiplier, not growing
FLAT_CHIPS = "sly,wily,clever,devious,crafty,banner,blue_joker,odd_todd,scary_face,arrowhead,stone,stuntman,ice_cream,bull"   # +chips that do not grow
GROW_MULT = "green_joker,ride_the_bus,red_card,supernova,trousers,fortune_teller,flash,ceremonial,popcorn"   # +mult that grows (or decays) over the run
CAP_ALIASES = {"flat_mult": FLAT_MULT, "flat_chips": FLAT_CHIPS, "grow_mult": GROW_MULT}


def apply_joker_cap(env, a):
    """--cap-jokers / --cap-max: groups of jokers ('+' between groups; each a comma list or flat_mult / flat_chips), at most the group's
    maximum (--cap-max: one number for all, or a comma list per group) held at once (buying, swapping in or picking one more is not legal)."""
    if not a.cap_jokers:
        return
    from .sim.jokers import JOKERS as _JC
    keys = list(_JC)
    groups = a.cap_jokers.split("+")
    maxes = [int(x) for x in str(a.cap_max).split(",")]
    maxes = maxes * len(groups) if len(maxes) == 1 else maxes
    env.v.set_joker_cap([], -1)
    for grp, mx in zip(groups, maxes):
        env.v.add_joker_cap([keys.index(k) for k in CAP_ALIASES.get(grp, grp).split(",")], mx)


def game_kw(a) -> dict:
    """Environment options shared by every mode: reward shape and the cheap-experiment game options."""
    pool = None
    if a.joker_pool_file:
        with open(a.joker_pool_file) as f:
            pool = json.load(f)
    return {"ante_weight": a.ante_weight, "win_bonus": a.win_bonus, "win_ante": a.win_ante, "joker_pool": pool,
            "shape_chips": a.shape_chips, "shape_phi": a.shape_phi, "reward_config": a.reward_config,
            "shape_gamma": a.gamma}


def log(path, row):
    print(json.dumps(row), flush=True)
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- behaviour cloning
def train_bc(a):
    torch.manual_seed(a.seed)
    dev = a.device
    model = load_model(a.init, dev) if a.init else ActorCritic().to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    venv = VecEnv(a.envs, a.deck, a.stake, seed0=a.seed * 1_000_000, expert=True, reward_kw=game_kw(a))
    obs, labels = venv.current()
    buf = collections.deque(maxlen=a.buffer)
    stats = EpisodeStats()
    rng = random.Random(a.seed)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        beta = max(0.0, 1.0 - it / max(1, a.iters * a.beta_frac))   # prob. of following the expert
        model.eval()
        for _ in range(a.steps):
            with torch.no_grad():
                logits, _ = model(batch_obs(obs, dev))
                student = logits.argmax(-1).cpu().numpy()
            for o, l in zip(obs, labels):
                buf.append((compact(o), int(l)))
            acts = [l if rng.random() < beta else s for l, s in zip(labels, student)]
            obs, _, _, infos, labels = venv.step(acts)
            for inf in infos:
                stats.add(inf)
        model.train()
        losses, accs = [], []
        data = list(buf)
        n_up = a.updates or max(1, a.epochs * len(data) // a.batch)
        for _ in range(n_up):
            if True:
                mb = rng.sample(data, min(a.batch, len(data)))
                ob = batch_obs([x[0] for x in mb], dev)
                y = torch.tensor([x[1] for x in mb], device=dev)
                logits, _ = model(ob)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
                accs.append((logits.argmax(-1) == y).float().mean().item())
        row = {"iter": it, "beta": round(beta, 2), "loss": round(float(np.mean(losses)), 4),
               "acc": round(float(np.mean(accs)), 3), "min": round((time.time() - t0) / 60, 1),
               "phase": a.phase, "env_steps": it * a.steps * a.envs}
        row.update(stats.summary())
        log(a.log, row)
        if it % a.save_every == 0 or it == a.iters:
            save_model(model, a.out, {"mode": "bc", "iter": it})
    venv.close()


# --------------------------------------------------------------------------- PPO
def train_ppo(a):
    torch.manual_seed(a.seed)
    dev = a.device
    model = load_model(a.init, dev) if a.init else ActorCritic().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    use_amp = a.amp and str(dev).startswith("cuda")

    def fwd(ob):
        # --amp: the network runs in bfloat16; probabilities and losses stay in float32
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, v = model(ob)
        return logits.float(), v.float()
    venv = VecEnv(a.envs, a.deck, a.stake, seed0=a.seed * 1_000_000 + 7,
                  reward_kw=game_kw(a),
                  strategic_kw={"tactical": a.tactical or a.init, "margin": a.margin} if a.strategic else None)
    obs, _ = venv.current()
    stats = EpisodeStats()
    ep_ret = np.zeros(a.envs)
    t0 = time.time()
    best = -1.0
    for it in range(1, a.iters + 1):
        model.eval()
        T, N = a.steps, a.envs
        # --pipeline: two groups of envs take turns, so one group simulates while the network
        # picks actions for the other. Each env still sees the same policy; only timing changes.
        groups = [list(range(N // 2)), list(range(N // 2, N))] if a.pipeline else [list(range(N))]
        gobs = [[obs[i] for i in g] for g in groups]
        gbuf = [{"obs": [], "act": [], "lp": [], "val": [], "rew": [], "done": []} for _ in groups]

        def act_and_send(k):
            with torch.no_grad():
                logits, v = fwd(batch_obs(gobs[k], dev))
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                lp = dist.log_prob(act)
            act = act.cpu().numpy()
            venv.step_async(act, groups[k])
            b = gbuf[k]
            b["obs"].append([compact(o) for o in gobs[k]])
            b["act"].append(act)
            b["lp"].append(lp.cpu().numpy())
            b["val"].append(v.cpu().numpy())

        def wait(k):
            gobs[k], r, d, infos, _ = venv.step_wait(groups[k])
            for e, re_, de in zip(groups[k], r, d):
                ep_ret[e] += re_
                if de:
                    stats.returns.append(float(ep_ret[e]))
                    ep_ret[e] = 0.0
            gbuf[k]["rew"].append(r)
            gbuf[k]["done"].append(d.astype(np.float32))
            for inf in infos:
                stats.add(inf)

        act_and_send(0)
        for t in range(T):
            if len(groups) == 2:
                act_and_send(1)
            wait(0)
            if t + 1 < T:
                act_and_send(0)
            if len(groups) == 2:
                wait(1)
        # stitch the groups back together in env order: lists of T rows, each covering all N envs
        store_obs = [sum((b["obs"][t] for b in gbuf), []) for t in range(T)]
        acts, logps, vals, rews, dones = (
            [np.concatenate([b[key][t] for b in gbuf]) for t in range(T)]
            for key in ("act", "lp", "val", "rew", "done"))
        obs = sum(gobs, [])
        with torch.no_grad():
            _, last_v = fwd(batch_obs(obs, dev))
            last_v = last_v.cpu().numpy()
        vals_a = np.array(vals)
        adv = np.zeros((T, N), dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else vals_a[t + 1]
            nonterm = 1.0 - dones[t]
            delta = rews[t] + a.gamma * nv * nonterm - vals_a[t]
            gae = delta + a.gamma * a.lam * nonterm * gae
            adv[t] = gae
        ret = adv + vals_a

        # the whole rollout goes to the GPU once; minibatches are slices of it
        big = stack_obs([o for row in store_obs for o in row], dev)
        flat_act = torch.tensor(np.concatenate(acts), device=dev)
        flat_lp = torch.tensor(np.concatenate(logps), device=dev)
        flat_adv = torch.tensor(adv.reshape(-1), device=dev)
        flat_ret = torch.tensor(ret.reshape(-1), device=dev)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        if use_amp:
            # bfloat16 rounds differently at play-time batch sizes than in update minibatches, which
            # would add noise to the PPO ratio; recompute the old log-probs the way the update sees them
            with torch.no_grad():
                chunks = []
                for k in range(0, T * N, a.batch):
                    sl = slice(k, k + a.batch)
                    lg, _ = fwd({key: t[sl] for key, t in big.items()})
                    chunks.append(torch.distributions.Categorical(logits=lg).log_prob(flat_act[sl]))
                flat_lp = torch.cat(chunks)

        model.train()
        idx = np.arange(T * N)
        pl, vl, ents, kls = [], [], [], []
        for _ in range(a.epochs):
            np.random.shuffle(idx)
            for k in range(0, len(idx), a.batch):
                mb = torch.from_numpy(idx[k:k + a.batch]).to(dev)
                logits, v = fwd({key: t[mb] for key, t in big.items()})
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(flat_act[mb])
                ratio = torch.exp(lp - flat_lp[mb])
                s1 = ratio * flat_adv[mb]
                s2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * flat_adv[mb]
                ploss = -torch.min(s1, s2).mean()
                vloss = F.mse_loss(v, flat_ret[mb])
                ent = dist.entropy().mean()
                loss = ploss + a.vf * vloss - a.ent * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                # kept on the GPU and fetched once per iteration (each .item() waits for the GPU)
                pl.append(ploss.detach()); vl.append(vloss.detach()); ents.append(ent.detach())
                kls.append((flat_lp[mb] - lp).mean().detach())
        pl, vl, ents, kls = (torch.stack(x).double().cpu().numpy() for x in (pl, vl, ents, kls))
        row = {"iter": it, "pi": round(float(np.mean(pl)), 4), "v": round(float(np.mean(vl)), 3),
               "ent": round(float(np.mean(ents)), 3), "kl": round(float(np.mean(kls)), 4),
               "min": round((time.time() - t0) / 60, 1), "phase": a.phase, "env_steps": it * T * N}
        row.update(stats.summary())
        log(a.log, row)
        if it % a.save_every == 0 or it == a.iters:
            save_model(model, a.out, {"mode": "ppo", "iter": it})
            s = stats.summary().get("blinds", -1)
            if s > best:
                best = s
                save_model(model, a.out.replace(".pt", "_best.pt"), {"mode": "ppo", "iter": it, "blinds": s})
    venv.close()


def stake_mix(spec: str, n: int, target: str) -> list | None:
    """The stake of each of n environments for --stake-mix (None: all on the target stake)."""
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        name, _, w = part.partition(":")
        out += [name.strip().upper()] * int(round(float(w) * n))
    if len(out) >= n:
        raise SystemExit("--stake-mix leaves no environment on the target stake")
    return out + [target] * (n - len(out))


def train_ppo_cpp(a):
    """PPO on the C++ engine (engine_env.CppVecEnv): every environment is stepped in one call on a thread
    pool and the observations arrive as batched arrays, which go straight to the device. The rollout is kept
    there (action features as float16) and minibatches are slices of it. Same algorithm, rewards and log rows
    as train_ppo; no --pipeline / --strategic / --shape-phi / joker pool (those need the Python environment)."""
    from .engine_env import CppVecEnv, OBS_KEYS
    torch.manual_seed(a.seed)
    dev = a.device
    cuda = str(dev).startswith("cuda")
    if cuda and a.gpu_fraction > 0:
        # a ceiling on this process's GPU memory: past it the allocator frees its cached blocks instead of
        # spilling into the slow shared memory (Windows), where a long rollout's update slows to a crawl
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)
    if a.win_only:                                       # the win bonus is the only reward: the shaped stream is off for good, and no blind pays
        a.streams = True                                 # into the sparse stream (blind 24 on: none)
        a.sparse_from_ante = 9
    # --resume: carry on a run from its own checkpoint and log (iteration count, schedules, totals); the
    # optimiser's moments start again
    it0, games0, wins0, min0 = 0, 0, 0, 0.0
    resumed = {}
    if a.resume and os.path.exists(a.out) and os.path.exists(a.log):
        with open(a.log) as f:
            last = [l for l in f if l.strip()]
        if last:
            last = json.loads(last[-1])
            it0, games0, wins0, min0 = last["iter"], last.get("games", 0), last.get("wins", 0), last.get("min", 0.0)
            resumed = last
        model = load_model(a.out, dev)
        print(json.dumps({"resumed": a.out, "iter": it0, "games": games0, "lr_scale": a.lr_scale}), flush=True)
    else:
        if a.arch == "nnue" and not a.init:                  # the Stockfish-shaped actor-critic (nnue_ac.py)
            from .nnue_ac import NnueActorCritic
            model = NnueActorCritic(width=a.nnue_width, topk=a.topk).to(dev)
        else:
          model = load_model(a.init, dev) if a.init else ActorCritic(attn=a.attn, d=a.d, tok=a.tok, emb=a.emb, heads=a.heads, desc=int(a.joker_desc), spec=(2 if a.strength else int(a.spec)),
                                                                         aw=a.aw, vw=a.vw, res=a.res, topk=a.topk,
                                                                         aux=2 if a.aux_coef > 0 else 0,
                                                                         marg=2 if a.potential else int(a.marginal),
                                                                         streams=N_OUT if a.streams else 0).to(dev)
    if a.streams and not getattr(model, "streams", 0):
        model.add_streams(N_OUT)
    if a.oracle_inputs and not getattr(model, "extra_dim", 0):
        model.add_extra(3)
    assert not (a.streams and a.priv), "--streams and --priv are not combined"
    if a.streams:                                       # the terminal reward and the per-blind rewards' rise over the antes
        a.win_bonus = a.stream_win_bonus
    enforce_win_ratio(a)
    if a.aux_play_coef > 0 and not getattr(model, "aux_play", 0):      # the auxiliary heads of an older network start here
        model.add_aux_play()
    if a.aux_loo_coef > 0 and not getattr(model, "aux_loo", 0):
        model.add_aux_loo()
    if a.reach_coef > 0 and not hasattr(model, "reach"):      # the outcome heads: P(reach ante 5..8), P(win) (a network made before them starts them at 0.5)
        model.add_reach_head()
    # fused: the same Adam step in one kernel (measured 1.11x on the update)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5, fused=cuda)
    use_amp = a.amp and cuda

    def fwd(ob):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, v = model(ob)
        return logits.float(), v.float()

    # --aux-coef: the prediction heads (model.ActorCritic aux)
    use_aux = a.aux_coef > 0 and hasattr(model, "aux")
    from .joker_desc import FEATURES as _DF, item_descriptors as _item_desc
    _D = _item_desc()                                  # [vocabulary, features]: which jokers are +mult, xMult, growing, economy, chips
    joker_cat = torch.tensor(np.stack([_D[:, _DF.index(n)] for n in ("mult", "xmult", "grows", "economy", "chips", "is_joker")], 1),
                             dtype=torch.float32, device=dev)
    use_play = a.aux_play_coef > 0 and bool(getattr(model, "aux_play", 0))
    use_loo = a.aux_loo_coef > 0 and bool(getattr(model, "aux_loo", 0))
    scen_on = a.scenario_p > 0 or bool(a.scenario_phases) or a.archive_p > 0
    split_step = a.streams or scen_on                    # step_split: the sparse part of the reward and which episode ends were cut off
    from .model import AUX_BLINDS

    def fwd_aux(ob):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, v, x = model(ob, aux=True)
        return logits.float(), v.float(), x.float()
    venv = CppVecEnv(a.envs, a.deck, a.stake, seed0=a.seed * 1_000_000 + 7 + it0 * 1_000_003, threads=a.threads,
                     reward_kw=game_kw(a), start_xmult=a.start_xmult, start_any=a.start_any,
                     verdant_bonus=a.verdant_bonus, stakes=stake_mix(a.stake_mix, a.envs, a.stake))
    env_stake = np.array(venv.stakes)
    if getattr(model, "marg", 0):                      # the marginal-value inputs (marginal.py) the network was made for
        venv.set_marginal(int(model.marg))
    elif use_loo:                                      # the leave-one-out head's targets: the engine fills the marginal columns, the network does not read them
        venv.set_marginal(1)
    if a.exact:                                        # the game's own named random streams (a future that depends on the actions only)
        venv.set_exact(True)
        venv.v.restart()
    if a.priv:                                         # the critic also sees the next shop, the next boss and the top of the deck
        from .engine_env import F_PRIV
        if not getattr(model, "priv", 0):
            model.add_priv(F_PRIV)
            opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5, fused=cuda)
        if a.priv_actor and not getattr(model, "priv_actor", 0):
            model.add_priv_actor()
            opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5, fused=cuda)
        venv.set_priv(True)
    # the joker school (--school-share): the student jokers' engine ids; the engine gives one of them, eternal and with
    # a kit, to a share of the games, and the trainer pays for the growth of its counter (log of the multiplier)
    from .sim.jokers import JOKERS as _SJ
    _skeys = list(_SJ)
    school_students = [_skeys.index(k) for k in a.school_students.split(",") if k] if a.school_share > 0 else []
    if a.death_money_penalty > 0:
        venv.set_death_penalty(a.death_money_penalty, a.death_money_cap)
    if a.blind_scale != 1.0 or a.ante8_bonus != 0.0:
        venv.set_reward(a.blind_scale, a.ante8_bonus)
    if a.streams:
        venv.set_sparse(3 * (a.sparse_from_ante - 1))
        venv.set_weight_shape(a.stream_ante_weight, a.weight_power)
        if a.weight_geom > 0:
            venv.set_weight_geom(a.weight_geom)
    elif a.weight_power != 1.0:
        venv.set_weight_shape(a.ante_weight, a.weight_power)
        if a.weight_geom > 0:
            venv.set_weight_geom(a.weight_geom)
    # --streams: the shaped stream's per-ante weights fade as each ante is mastered; the return of each stream is normalised on its own
    fade = MilestoneFade(a.stream_trigger, a.stream_fade_iters, a.stream_reverse, a.stream_floor)
    ent_bump = EntropyBump(a.fade_ent_bump, a.fade_ent_decay)
    reach = ReachTracker(a.stream_window)
    norm_ret = ReturnNorm(a.envs, a.gamma, a.ret_norm_floor)     # --ret-norm: the single-stream path's rewards, divided by the std of the discounted return
    norm_c = [ReturnNorm(a.envs, a.gamma, a.ret_norm_floor) for _ in range(N_ANTES)]       # --stream-equal: one per ante
    norm_sh = ReturnNorm(a.envs, a.gamma, a.ret_norm_floor)
    norm_sp = ReturnNorm(a.envs, a.gamma, a.ret_norm_floor)
    if a.streams and resumed.get("fade_w"):
        fade.w = np.array(resumed["fade_w"], dtype=np.float64)
        fade.started = fade.w < 1.0
        ent_bump.left = int(resumed.get("ent_bump_left", 0))
    if a.ban_jokers:
        from .sim.jokers import JOKERS as _J
        _keys = list(_J)
        _ban = [_keys.index(k) for k in a.ban_jokers.split(",")]           # (a name that is no joker: an error)
        if a.ban_top <= 0:
            venv.set_ban(_ban, 1.0)
    apply_joker_cap(venv, a)
    if a.reroll_cap > 0 or a.hand_level_cap > 0 or a.flash_cap > 0:         # the rules of a capped-resource sandbox (rerolls per shop, hand level a planet can reach, Flash Card's rerolls)
        venv.set_caps(a.reroll_cap, a.hand_level_cap, a.flash_cap)
    ante_ema = np.array(resumed.get("ante_ema", [0.0] * 8), dtype=np.float64)      # --immortal-fade: each ante's smoothed clear rate (mean of its three blinds)
    ante_mult = np.ones(8)
    if a.autoplay:
        venv.set_autoplay(True, a.autoplay_nodes, a.autoplay_force)
    if a.round_nnue:                                    # the tactical layer's long-term term: the solver keeps a clearing line by money + w x V(state after the round)
        venv.load_nnue(a.round_nnue)                    # (one network and weight for the whole process: the evaluation's solver uses them too)
        venv.set_round_eval(a.round_eval_w)
    if a.catalogue > 0:                                 # the catalogue shop in this share of training games (every non-legendary joker, 24 to a page, free page turns)
        venv.v.set_catalogue(a.catalogue)
    if a.blind_bases:                                   # a longer game with slower scaling: each ante's base chips (training games only; the evaluation plays the stake's)
        venv.v.set_blind_bases([int(float(x)) for x in a.blind_bases.split(",")])
    if a.start_any_eternal:                             # the random starting joker(s) cannot be sold
        venv.v.set_start_eternal(True)
    if a.start_any2 > 0:                                # of the games given a random joker (--start-any), this share get a second one
        venv.v.set_start_any2(a.start_any2)
    if a.solver_smart:                                  # the round solver's speed-ups (engine globals: training and evaluation): set-up discards first where the
        venv.v.set_solver_order(True)                   # greedy line cannot clear, cheap steps, and no position searched twice
    if a.solver_lite:
        venv.v.set_solver_speed(True, False, 0.0)
    if a.solver_tt:
        venv.v.set_solver_tt(True)
    if a.solver_tarots:                                 # the solver also uses held consumables that need target cards, mid-round (the rest are the network's, outside rounds)
        venv.v.set_solver_tarots(True, a.solver_targets)
    vloop = None
    if a.vloop_every > 0:                               # the V(s) loop: the engine's value network is fitted to the critic every --vloop-every updates (vloop.py)
        from .vloop import VLoop
        vloop = VLoop(venv, dev, a.out.replace(".pt", "_v.bin"), w=a.round_eval_w, buffer=a.vloop_buffer, per_step=a.vloop_per_step,
                      steps=a.vloop_steps, init=a.vloop_init)
    if a.immortal:
        venv.set_immortal(True, a.immortal_pay)
        venv.v.restart()
    if a.archive_p > 0 or a.archive_store > 0:          # the state archive: starts from the policy's own earlier games
        lo_a, hi_a = (1, 8) if a.archive_ante == "auto" else (int(resumed.get("archive_lo", 8)), 8) if a.archive_ante == "reverse" else [int(x) for x in a.archive_ante.split("-")]
        venv.set_archive(a.archive_p, a.archive_store, lo_a, hi_a, a.archive_cap, a.archive_min, a.archive_winners)
        band_lo = lo_a if a.archive_ante == "reverse" else 0
        band_acc = np.zeros((2, 24), dtype=np.int64)
        band_wr = None
        last_out = [[0] * 24, [0] * 24]
        if a.archive_seed_ckpt:
            seed_archive(a, venv, dev)
        if a.plant_file:
            plant_archive(a, venv)
        band_since = it0                                     # (the update the current reverse stage began)
        fade_from = None                                     # (the update the band reached ante 1: the planted starts fade from there)
    if a.shape_relative and a.verdant_bonus > 0:        # the Verdant Leaf bonus too is in units of the blind's weight
        venv.set_verdant_relative(True)
    if a.start_ante > 1:                                # the late-game sandbox: the first games too start here, with the money and the planet cap
        venv.set_start(a.start_xmult, a.start_any, a.start_money_p, a.start_money, a.start_money2_p, a.start_money2, a.start_grow)
        if a.level_cap_share > 0:
            venv.set_level_cap(a.level_cap_share, a.level_cap_n, a.level_cap_ante)
        venv.set_sandbox(a.level_cap_mode, a.pack_restock, pack_bias_list(a.pack_bias))
        if a.scaled_start > 0 or a.scaled_shop:
            venv.set_scaled(a.scaled_start, a.scaled_shop, a.scaled_rounds)
        venv.set_start_ante(a.start_ante)
        venv.v.restart()
    if a.fixed_seed >= 0:                              # PPO on one seed
        venv.set_fixed_seed(a.fixed_seed)
        venv.v.restart()
    if a.log_partial:                                  # (was applied only with --start-ante > 1)
        venv.set_log_partial(True)
    if a.shop_only:
        venv.set_shop_only(True, a.shop_only_nodes, a.partial_weight)
        venv.v.restart()
    if a.no_part_credit:                               # (was applied only with --start-ante > 1: every run from ante 1 before 2026-10-19 trained WITH part credit)
        venv.set_part_credit(False)
    if a.no_stickers or a.small_blind_pay:             # rule changes (games restarted so the first ones follow them too)
        venv.set_rules(a.no_stickers, a.small_blind_pay)
        venv.v.restart()
    target = env_stake == a.stake                    # the logged blinds / antes / wins are the target stake's
    ladder = [x.strip().upper() for x in a.ladder.split(",") if x.strip()]
    ladder_since, ladder_hits, ladder_preview = int(resumed.get("ladder_since", 0)), 0, None
    ladder_bump = EntropyBump(a.ladder_ent_bump, a.ladder_ent_decay)
    ladder_bump.left = int(resumed.get("ladder_bump_left", 0))
    stall = StallDetector(a.stall_iters, a.stall_delta, a.stall_ent_bump, a.stall_ent_cap, a.stall_guard)
    stall_bump = EntropyBump(a.stall_ent_bump, a.stall_ent_decay)
    stall_bump.left = int(resumed.get("stall_bump_left", 0))
    stall.k = int(resumed.get("stall_k", 0))
    if a.stall_ent and a.stall_now:
        stall.k = max(stall.k, 1)
        stall_bump.peak = a.stall_ent_bump
        stall_bump.trigger()
        print(json.dumps({"stall": "fired by hand at the start", "level": a.stall_ent_bump}), flush=True)

    def apply_ladder(i):
        """Train on ladder[i]: all environments from their next game on, except --ladder-keep of them on the stake below."""
        nonlocal env_stake, target
        a.stake = ladder[i]
        keep = int(round(a.ladder_keep * a.envs)) if i > 0 else 0
        ahead = int(round(a.ladder_ahead * a.envs)) if i + 1 < len(ladder) else 0
        new = [ladder[i]] * (a.envs - keep - ahead) + ([ladder[i - 1]] * keep) + ([ladder[i + 1]] * ahead)
        venv.set_stakes(new)
        env_stake = np.array(venv.stakes)
        target = env_stake == a.stake
    if ladder:
        assert a.stake in ladder or resumed.get("ladder_stake") in ladder, "--stake is not on --ladder"
        ladder_i = ladder.index(resumed.get("ladder_stake", a.stake))
        if a.ladder_set:
            assert a.ladder_set.upper() in ladder, "--ladder-set is not on --ladder"
            ladder_i = ladder.index(a.ladder_set.upper())
            ladder_since = 0
            ladder_bump.trigger()
            print(json.dumps({"ladder": "set", "to": ladder[ladder_i]}), flush=True)
        if ladder_i > 0 or a.stake != ladder[ladder_i]:
            apply_ladder(ladder_i)
    # --stake-curriculum: the target stake's environments move to --stake-easy when its blinds stop improving
    cur_prev, cur_blinds, cur_games = None, 0.0, 0
    if a.stake_curriculum:
        n_hard = int(resumed.get("n_hard", int(target.sum())))
        if n_hard < int(target.sum()):
            _new = list(env_stake)
            for _i in np.flatnonzero(target)[n_hard:]:
                _new[_i] = a.stake_easy
            venv.set_stakes(_new)
            env_stake = np.array(venv.stakes)
            target = env_stake == a.stake
    wins_other: dict = {}
    games_by: dict = {}                              # games finished on each stake, in total (for win rates)
    T, N = a.steps, a.envs
    from .env import KIND_OF, N_KINDS, ACTION_TYPES
    kind_names = list(ACTION_TYPES) + ["replace"]
    # --ban-top: the shifting world. Joker purchases (shop, replace, pack picks) are counted by item; every
    # --ban-every iterations the --ban-top most bought become the ban list, out of the pool in --ban-share of games
    from .sim.jokers import JOKERS as _JOKERS
    from .sim.items import VOCAB as _VOCAB
    joker_keys = list(_JOKERS)
    from .env import item_id as _item_id
    jvocab = torch.tensor([_item_id(f"j_{k}") for k in joker_keys], device=dev)       # engine joker id -> vocab id
    buy_kinds = torch.tensor([kind_names.index(k) for k in ("buy", "replace", "pick")], device=dev)
    buy_counts = torch.zeros(len(_VOCAB), device=dev)    # purchases by vocab id since the last recompute
    ban_ids: list = list(resumed.get("ban_ids", []))
    ban_games = ban_blinds = norm_games = norm_blinds = 0
    off_list = 0.0
    type_onehot = torch.zeros((len(KIND_OF), N_KINDS), device=dev)
    type_onehot[torch.arange(len(KIND_OF)), torch.as_tensor(KIND_OF)] = 1.0
    type_of = torch.as_tensor(KIND_OF, device=dev)

    # --build-bonus: which of the three scoring roles the jokers held cover (multiplier, +mult, chips); a
    # debuffed joker (perished, or a boss's) covers none
    from .env import item_id
    from .heuristic import CHIPS, XMULT
    from .az.guide import MULT
    from .az.guide import SCALERS as _SCL, SLOW_SCALERS as _SSCL
    role = torch.zeros((1024, 4), device=dev)                    # columns: multiplier, +mult, chips, scaler (a joker that grows; the multiplier ones count as multipliers)
    for col, keys in enumerate((XMULT, set(MULT) - set(XMULT), CHIPS, (set(_SCL) | set(_SSCL)) - set(XMULT))):
        for key in keys:
            if item_id(f"j_{key}") != item_id("j_unknown"):      # a name that is no joker (and a face-down
                role[item_id(f"j_{key}"), col] = 1.0             # joker) covers nothing
    _br = set(a.build_roles.split(","))                          # --build-roles: which roles the build potential pays for
    role_w = torch.tensor([1.0 if "xmult" in _br else 0.0, 0.5 if "mult" in _br else 0.0, (1.0 / 3.0) if "chips" in _br else 0.0,
                           1.0 if "scaler" in _br else 0.0, a.build_second], device=dev)   # the last: a second multiplier

    from .env import A_BUY, A_BUY_PACK, A_PICK, A_PSKIP, A_REPLACE
    take = torch.zeros(len(KIND_OF), device=dev)          # rows that acquire an item: buy, pack pick, replace
    take[A_BUY:A_BUY_PACK] = 1.0
    take[A_PICK:A_PSKIP] = 1.0
    take[A_REPLACE:] = 1.0

    def xmult_takes(o):
        """[B, A]: legal actions that take a multiplier joker on offer (shop, pack, or a replace that gives up
        a joker that is not one itself)."""
        return role[o["aid"].long(), 0] * take * o["mask"].float() * (1.0 - role[o["aid2"].long(), 0])
    # --explore-items: the things the policy never tries by itself: economy jokers (a value generator), the
    # boss tools, and the tarots that thin, copy and concentrate the deck; and selling Luchador on a live boss
    from .az.guide import ECON
    from .env import A_SELL_J, MAX_JOKERS
    expose = torch.zeros(1024, device=dev)
    # jokers that grow by their own counter (the engine's START_GROW): the "grow" exposure group, --grow-bonus
    GROWING = ("runner", "square", "castle", "wee", "green_joker", "ride_the_bus", "red_card", "trousers", "flash",
               "constellation", "hologram", "vampire", "obelisk", "lucky_cat", "glass", "campfire", "hit_the_road")
    growing = torch.zeros(1024, device=dev)
    for key in GROWING:
        growing[item_id(f"j_{key}")] = 1.0
    groups = {x.strip() for x in a.explore_groups.split(",") if x.strip()}      # of: econ, boss, deck, grow
    assert groups <= {"econ", "boss", "luchador", "deck", "grow"}, f"--explore-groups: unknown group in {sorted(groups)}"
    # (Credit Card is on the guide's list but earns nothing: it lets the agent spend into debt, and forced
    # exposure to it taught exactly that)
    for key in (([f"j_{k}" for k in ECON if k != "credit_card"] if "econ" in groups else [])
                + (["j_luchador", "j_mr_bones"] if "luchador" in groups else [])
                + (["c_hanged_man", "c_death", "c_strength"] if "deck" in groups else [])
                + ([f"j_{k}" for k in GROWING] if "grow" in groups else [])):
        if item_id(key) not in (item_id("j_unknown"), item_id("c_unknown")):
            expose[item_id(key)] = 1.0
    # the "boss" group: buying the vouchers that allow a boss reroll (Director's Cut, then Retcon), and the
    # reroll itself when it is on offer
    from .env import A_VOUCHER, A_REROLL, A_REROLL_BOSS
    boss_rows = torch.zeros(len(KIND_OF), device=dev)
    boss_voucher = torch.zeros(1024, device=dev)
    if "boss" in groups:
        boss_rows[A_VOUCHER:A_REROLL] = 1.0
        boss_voucher[item_id("v_directors_cut")] = 1.0
        boss_voucher[item_id("v_retcon")] = 1.0
    luchador = item_id("j_luchador") if "luchador" in groups else -1
    sell_rows = torch.zeros(len(KIND_OF), device=dev)
    sell_rows[A_SELL_J:A_SELL_J + MAX_JOKERS] = 1.0
    reroll_boss = torch.zeros(len(KIND_OF), device=dev)
    reroll_boss[A_REROLL_BOSS] = 1.0 if "boss" in groups else 0.0

    def item_takes(o):
        """[B, A]: legal actions that take an exposed item on offer (shop, pack, or a replace that gives up a
        joker that is neither a multiplier nor exposed itself), buy a boss-reroll voucher, or reroll the boss."""
        m = o["mask"].float()
        aid, aid2 = o["aid"].long(), o["aid2"].long()
        t = expose[aid] * take * m * (1.0 - role[aid2, 0]) * (1.0 - expose[aid2])
        t = t + boss_voucher[aid] * boss_rows * m + reroll_boss * m
        if luchador >= 0:                 # the "luchador" group: selling it while a boss blind is being played
            live_boss = ((o["g"][:, 5] > 0.5) & (o["g"][:, 53] > 0.5)).float().unsqueeze(1)     # in a round; boss active
            t = t + (aid == luchador).float() * sell_rows * m * live_boss
        return t
    from .env import A_SELECT as _A_SEL, N_ACTIONS as _N_ACT
    nonround = torch.zeros(_N_ACT, device=dev)
    nonround[_A_SEL:] = 1.0
    act_state = {"bal": torch.ones(_N_ACT, device=dev), "next": torch.ones(_N_ACT, device=dev), "freq": torch.full((_N_ACT,), 1.0 / max(1, _N_ACT - _A_SEL), device=dev)}
    kind_w = torch.ones(N_KINDS, device=dev)
    kind_w[kind_names.index("use_cons")] = a.explore_use

    def policy(logits, mask, o=None, xt=None, et=None):
        """The behaviour policy: the network's, mixed with a floor that spreads explore_types evenly over the
        kinds of action that are legal (play, discard, buy, sell a joker, reroll, voucher, skip ...) and evenly
        within a kind, so no kind of action can fall to probability zero and stop being tried. The update uses
        the same mixture, so PPO's ratio is exact."""
        if a.explore_types <= 0 and a.explore_xmult <= 0 and a.explore_items <= 0 and a.explore_actions <= 0:
            return torch.distributions.Categorical(logits=logits, validate_args=False)
        m = mask.float()
        per_type = m @ type_onehot                                   # legal actions of each kind  [B, 16]
        # every legal kind gets one share of the floor, "use a consumable" explore_use shares (a held tarot is
        # in hand at few decisions, so an even split tries it too seldom to learn what it does to a hand)
        share = (per_type > 0).float() * kind_w
        share = share / share.sum(1, keepdim=True).clamp(min=1e-9)
        floor = m * (share / per_type.clamp(min=1))[:, type_of]      # [B, A], sums to 1 over the legal actions
        probs = (1 - a.explore_types) * torch.softmax(logits, 1) + a.explore_types * floor
        if a.explore_actions > 0 and o is not None:
            # outside a round every individual legal action gets an equal slice of explore_actions of the probability (weighted toward the rarely taken ones with --explore-balance)
            w_a = m * nonround * (o["g"][:, 5] <= 0.5).float().unsqueeze(1)
            if a.explore_balance:
                w_a = w_a * act_state["bal"]
            cnt_a = w_a.sum(1, keepdim=True)
            z_a = a.explore_actions * (cnt_a > 0).float()
            probs = (1 - z_a) * probs + z_a * w_a / cnt_a.clamp(min=1e-9)
        if a.explore_xmult > 0 and o is not None:
            # explore_xmult of the probability, when a multiplier joker can be taken, is spread over taking one:
            # the policy passes on most of them, so it never learns what they do; this makes it find out
            if xt is None:
                xt = xmult_takes(o)
            cnt = xt.sum(1, keepdim=True)
            x = a.explore_xmult * (cnt > 0).float()
            probs = (1 - x) * probs + x * xt / cnt.clamp(min=1)
        if a.explore_items > 0 and o is not None:
            if et is None:
                et = item_takes(o)
            cnt = et.sum(1, keepdim=True)
            y = a.explore_items * (cnt > 0).float()
            probs = (1 - y) * probs + y * et / cnt.clamp(min=1)
        return torch.distributions.Categorical(probs=probs, validate_args=False)

    def current():
        # the engine's buffers are overwritten by the next step: copy them (to the GPU, or on the CPU)
        out = {}
        for k in OBS_KEYS:
            t = torch.from_numpy(venv.obs[k])
            if k in ("af", "priv") and cuda:
                t = t.half()         # the rollout stores them as float16 anyway: a smaller copy, and the policy
            out[k] = t.to(dev) if cuda else t.clone()      # acting sees exactly what the update will see
        if getattr(model, "topk", 0):
            from .nnue_ac import topk_mask
            out["mask"] = topk_mask(out, model.topk)
        return out
    orac_on = a.oracle_bonus > 0 or a.money_option_bonus > 0 or a.oracle_inputs
    from .env import A_SELECT as A_SEL_ID
    money_tab = None
    if orac_on:
        import json as _json
        _mt = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "money_option.json")))
        money_tab = {"money": np.array(_mt["money"], dtype=np.float64), "p": np.array(_mt["p"], dtype=np.float64)}
    orac_acc = np.zeros(3)
    orac_n = [0]
    orac_calls = [0]
    orac_cache = {}                                   # build signature -> (clear share, scaled log best share): the oracle is remeasured only when the build changes
    from .heur import G_LEVELS as _GL, G_FULL as _GF, G_ENH as _GE

    def build_sig(venv_, idx):
        """Per game in idx: what the oracle depends on (jokers and their debuff flags, hand levels, deck size and enhancements). The ante, the scaling
        counters and the blinds skipped are left out: the projected end state of a build hardly moves from round to round, so one measurement per build."""
        g_ = np.asarray(venv_.obs["g"])[idx]
        jid_ = np.asarray(venv_.obs["jid"])[idx]
        jf_ = np.asarray(venv_.obs["jf"])[idx]
        parts = np.concatenate([jid_, np.rint(jf_[:, :, 9]), np.rint(g_[:, _GL:_GL + 12] * 10), np.rint(g_[:, _GF:_GF + 1] * 52),
                                np.rint(g_[:, _GE:_GE + 8] * 10)], 1).astype(np.int32)
        return [row.tobytes() for row in parts]

    def attach_ex(venv_, ex_arr, o):
        """--oracle-*: for the environments outside a round, remeasure [the ante-8 clear share, the scaled log of the best share (1e-4 -> 0, 1 -> 1),
        P(find a strong joker fishing with this money)] (Env::final_boss_clear, the money-option table) and attach them as ob["ex"]; a round's
        steps keep the values of its last measurement."""
        if not orac_on:
            return
        mask_np = np.asarray(venv_.obs["mask"])
        outr = np.flatnonzero(~mask_np[:, :A_SEL_ID].any(1))
        if len(outr):
            if len(orac_cache) > 400_000:                        # (bounded: cleared before this step's entries go in, so they are found below)
                orac_cache.clear()
            sigs = build_sig(venv_, outr)
            need = [k for k, sg in enumerate(sigs) if sg not in orac_cache]
            if need:
                fb = venv_.final_boss_clear([int(outr[k]) for k in need], a.oracle_samples, a.oracle_nodes)
                for k, row in zip(need, fb):
                    orac_cache[sigs[k]] = (float(row[0]), float(np.clip((np.log10(max(row[1], 1e-4)) + 4.0) / 4.0, 0.0, 1.0)))
                orac_calls[0] += len(need)
            vals_ = np.array([orac_cache[sg] for sg in sigs], dtype=np.float64)
            ex_arr[outr, 0] = vals_[:, 0]
            ex_arr[outr, 1] = vals_[:, 1]
            g_ = np.asarray(venv_.obs["g"])[outr]
            money_ = 10.0 ** (3.0 * g_[:, 18]) - 1.0
            ante_ = np.clip(np.rint(g_[:, 0] * 8.0), 1, 8).astype(int)
            pf = np.zeros(len(outr))
            for an in range(1, 9):
                sel_ = ante_ == an
                if sel_.any():
                    pf[sel_] = np.interp(money_[sel_], money_tab["money"], money_tab["p"][an - 1])
            ex_arr[outr, 2] = pf
            orac_acc[:] += ex_arr[outr].sum(0)
            orac_n[0] += len(outr)
        o["ex"] = torch.from_numpy(ex_arr).to(dev)
    ex_state = np.zeros((venv.n, 3), dtype=np.float32)
    ob = current()
    attach_ex(venv, ex_state, ob)

    la_coef = [a.bank_bonus, a.build_bonus]              # (the bank / build potentials' current weights, for the evaluation's lookahead)

    def fixed_eval(stake=None, n=None, la_k=0):
        """A fixed evaluation that never changes: greedy games on a fixed set of seeds (never trained on), the same every time, whatever the reward
        phase. -> win rate, the furthest ante reached (lost in 1..8, won), blinds beaten, and the length of a game in environment steps.
        (stake: another stake than the one in training: the ladder's preview.)"""
        n = n or a.eval_games
        ev = CppVecEnv(n, a.deck, stake or a.stake, seed0=a.eval_seed0, threads=a.threads)
        if a.autoplay and a.autoplay_force:               # every round solver-played: the evaluation too (the network never learned to play rounds)
            ev.set_autoplay(True, a.autoplay_nodes, True)
        if getattr(model, "marg", 0):
            ev.set_marginal(int(model.marg))
        if a.exact:
            ev.set_exact(True)
            ev.v.restart()
        if a.priv:
            ev.set_priv(True)
        if a.ban_jokers and a.ban_top <= 0:               # (the evaluation plays the same joker pool as the training: the bans too)
            from .sim.jokers import JOKERS as _JE
            _ke = list(_JE)
            ev.set_ban([_ke.index(k) for k in a.ban_jokers.split(",")], 1.0)
        apply_joker_cap(ev, a)
        if a.fixed_seed >= 0:
            ev.set_fixed_seed(a.fixed_seed)
            ev.v.restart()
        if a.shop_only:
            ev.set_shop_only(True, a.shop_only_nodes, a.partial_weight)
            ev.v.restart()
        if a.no_stickers or a.small_blind_pay:          # (the evaluation plays the same rules)
            ev.set_rules(a.no_stickers, a.small_blind_pay)
            ev.v.restart()
        if a.reroll_cap > 0 or a.hand_level_cap > 0 or a.flash_cap > 0:
            ev.set_caps(a.reroll_cap, a.hand_level_cap, a.flash_cap)
        if a.start_ante <= 1 and a.start_money_p >= 1.0 and a.start_money > 0:     # (an unlimited-money sandbox: the evaluation has the money too)
            ev.set_start(0.0, 0.0, a.start_money_p, a.start_money, 0.0, 0, 0.0)
            ev.v.restart()
        if a.start_ante > 1:                              # the late-game sandbox: the evaluation games start there too
            ev.set_start(0.0, 0.0, a.start_money_p, a.start_money, 0.0, 0, 0.0)
            if a.level_cap_share > 0:
                ev.set_level_cap(a.level_cap_share, a.level_cap_n, a.level_cap_ante)
            ev.set_sandbox(a.level_cap_mode, a.pack_restock, pack_bias_list(a.pack_bias))
            if a.scaled_start > 0 or a.scaled_shop:
                ev.set_scaled(a.scaled_start, a.scaled_shop, a.scaled_rounds)
            ev.set_start_ante(a.start_ante)
            ev.v.restart()
        fin_ = np.zeros(n, dtype=bool)
        steps_ = np.zeros(n, dtype=np.int64)
        ante_ = np.ones(n, dtype=np.int64)
        won_ = np.zeros(n, dtype=bool)
        blinds_ = np.zeros(n, dtype=np.int64)
        model.eval()
        ex_ev = np.zeros((n, 3), dtype=np.float32)
        la = None
        if la_k > 0:                                      # one-step lookahead outside rounds (eval_lookahead.py): copies built like the training games (same rewards)
            from .eval_lookahead import Lookahead

            def make_env(slots):
                s_ = CppVecEnv(slots, a.deck, stake or a.stake, seed0=a.eval_seed0 + 77, threads=a.threads, reward_kw=game_kw(a), verdant_bonus=a.verdant_bonus)
                apply_joker_cap(s_, a)
                if a.weight_power != 1.0:
                    s_.set_weight_shape(a.ante_weight, a.weight_power)
                if getattr(model, "marg", 0):
                    s_.set_marginal(int(model.marg))
                if a.autoplay:
                    s_.set_autoplay(True, a.autoplay_nodes, a.autoplay_force)
                s_.v.restart()
                return s_
            from .shaping_pot import ShapingPot
            _pot = ShapingPot(dev, build_roles=a.build_roles, build_second=a.build_second, rental=a.rental_penalty, debt=a.debt_penalty)
            la = Lookahead(make_env, fwd, dev, k=la_k, pot=lambda o_: _pot(o_, bank=la_coef[0], build=la_coef[1]))
        with torch.no_grad():
            for _ in range(6000):
                o = {k: torch.from_numpy(ev.obs[k]).to(dev) for k in OBS_KEYS}
                attach_ex(ev, ex_ev, o)
                logits, _ = fwd(o)
                if la is not None:
                    act = la.act(ev, logits, fin_)
                else:
                    act = np.where(fin_, -1, logits.argmax(1).cpu().numpy())          # (a finished game has been reset: leave it alone)
                _, d, info = ev.step(act)
                first = d & ~fin_
                ante_[first], won_[first], blinds_[first] = info["ante"][first], info["won"][first], info["blinds"][first]
                steps_ += ~fin_
                fin_ |= d
                if fin_.all():
                    break
        ev.close()
        furthest = np.where(won_, 9, ante_)
        out_ = {"games": n, "win_rate": round(float(won_.mean()), 4), "blinds": round(float(blinds_.mean()), 3),
                "mean_len": round(float(steps_.mean()), 1), "ante_hist": [int((furthest == k).sum()) for k in range(1, 10)]}
        if la is not None:
            out_["la_changed"] = round(la.n_changed / max(1, la.n_decisions), 3)     # share of decisions where the lookahead overruled the policy
            la.close()
        return out_

    from .env import VOUCHER_KEYS
    v0 = 162 - VOUCHER_KEYS.index("seed_money")          # g: one column per voucher owned, from v0
    assert v0 + VOUCHER_KEYS.index("money_tree") == 177, "the voucher columns of the observation moved"
    col_seed, col_tree = 162, 177

    def bank(o):
        """Money held that earns interest, in units of $25: interest stops at $25, at $50 with Seed Money, at
        $100 with Money Tree (g column 17 is min(money, 100) / 50)."""
        g = o["g"]
        cap = 25.0 + 25.0 * g[:, col_seed].float() + 50.0 * g[:, col_tree].float()
        return torch.minimum((g[:, 17].float() * 50.0).clamp(min=0.0), cap) / 25.0

    def roles(o):
        """[N, 5]: holds a working multiplier / +mult / chips / scaling joker, and holds two or more working
        multiplier jokers (jf column 9 is the debuff flag)."""
        live = ((o["jid"] > 0) & (o["jf"][:, :, 9] < 0.5)).float().unsqueeze(-1)
        r = role[o["jid"]] * live
        return torch.cat([r.amax(1), (r[:, :, 0].sum(1, keepdim=True) >= 2).float()], 1)

    from .env import RATIO_COL, N_SUB, A_LEAVE as A_LEAVE_ID

    def power(o, act):
        """How strongly the play `act` beats the blind, 0..1 (only read where a blind is beaten by it): half
        for the score reached as a multiple of the target (log scale, full at 5x), half for the hands left
        unused. g columns: 9 chips / target so far, 11 hands left / 5, 13 the round's hands / 5; an action's
        RATIO_COL is its predicted score over the chips still needed (capped at 5)."""
        g = o["g"]
        rows = torch.arange(act.shape[0], device=act.device)
        ratio = o["af"][rows, act.clamp(max=N_SUB - 1), RATIO_COL].float()
        done_share = g[:, 9].float().clamp(0.0, 1.0)
        over = (done_share + ratio * (1.0 - done_share)).clamp(1.0, 5.0)
        hands = (g[:, 13].float() * 5.0 - 1.0).clamp(min=1.0)
        spare = ((g[:, 11].float() * 5.0 - 1.0) / hands).clamp(0.0, 1.0)
        return 0.5 * torch.log(over) / float(np.log(5.0)) + 0.5 * spare

    ref_tab = torch.tensor(blind_log_table(), device=dev)

    def strength(o, act):
        """[N] build strength at a round's first play or discard, 0..1 (0 elsewhere): the best single play's
        predicted score (the engine's full scoring: hand level, cards, every joker's chips, +mult and xMult, in
        their order) as a multiple of the blind's target, on a log scale from 0.1x (0) to about 3x (1).
        g columns: 5 in a round, 8 log target, 9 chips / target, 11 / 13 hands left / the round's, 12 / 14
        discards likewise; an action's RATIO_COL + 1 is log10(its score + 1) / 7."""
        g = o["g"].float()
        fresh = ((g[:, 5] > 0.5) & (g[:, 9] == 0) & ((g[:, 11] - g[:, 13]).abs() < 1e-6)
                 & ((g[:, 12] - g[:, 14]).abs() < 1e-6) & (act < 2 * N_SUB))
        sc = o["af"][:, :N_SUB, RATIO_COL + 1].float() * o["mask"][:, :N_SUB]
        if a.strength_ahead > 0:
            # --strength-ahead K: the build is measured against the targets K antes later (the same blind, up to ante 8), on a window from 1/30 of
            # that target (0) to the target itself (1): an early build only scores as it grows toward what the later antes ask, not for clearing
            # the blinds of its own ante easily (which any build does at the start, and which is all a run that dies early ever sees)
            ante_i = (g[:, 0] * 8.0).round().clamp(1, 8).long()
            blind_i = g[:, 1:4].argmax(1)
            ref = ref_tab[(ante_i + a.strength_ahead).clamp(max=8), blind_i]
            log_ratio = 7.0 * sc.max(1).values - ref
            return ((log_ratio + 1.5) / 1.5).clamp(0.0, 1.0) * fresh, fresh
        log_ratio = 7.0 * sc.max(1).values - 6.0 * g[:, 8]                  # log10(best play / target)
        return ((log_ratio + 1.0) / 1.5).clamp(0.0, 1.0) * fresh, fresh

    def phi8_val(o):
        """[N] --phi8's potential before the weight: the best single play's predicted score against the ante-8 boss target, on a window of --phi8-span decades."""
        sc = o["af"][:, :N_SUB, RATIO_COL + 1].float() * o["mask"][:, :N_SUB]
        log_ratio = 7.0 * sc.max(1).values - ref_tab[8, 2]
        return ((log_ratio + a.phi8_span) / a.phi8_span).clamp(0.0, 1.0)

    def growth(o):
        """[N] how far the growing jokers held have grown: the sum of their counters on the observation's log
        scale (jf column 11 is log1p(|counter|) / 4; column 9 the debuff flag)."""
        live = ((o["jid"] > 0) & (o["jf"][:, :, 9] < 0.5)).float()
        return (growing[o["jid"]] * o["jf"][:, :, 11].float() * live).sum(1)

    # what outlasts a round, in the observation: g columns 81-92 hand levels, 135 the deck's size, 136-147 its
    # enhancements and seals, 148 joker slots, 150 consumable slots, 152-183 vouchers; a joker's edition
    # (jf columns 1-5) and counter (11); which jokers are held, in which order (jid); the consumables (cid)
    lasting_g = torch.tensor(list(range(81, 93)) + list(range(135, 149)) + [150] + list(range(152, 184)), device=dev)
    lasting_j = torch.tensor([1, 2, 3, 4, 5, 11], device=dev)

    def changed_lasting(before, after):
        """[N] did this step change anything that outlasts the round?"""
        return ((before["jid"] != after["jid"]).any(1) | (before["cid"] != after["cid"]).any(1)
                | ((before["jf"][:, :, lasting_j].float() - after["jf"][:, :, lasting_j].float()).abs() > 1e-6).flatten(1).any(1)
                | ((before["g"][:, lasting_g].float() - after["g"][:, lasting_g].float()).abs() > 1e-6).any(1))

    def money_of(o):
        """Dollars held (g column 18 is log10(money + 1) / 3; a debt reads as 0)."""
        return torch.pow(10.0, 3.0 * o["g"][:, 18].float()) - 1.0

    def potentials(o):
        """On the device: the bank potential [N], the joker-role potential [N], the share holding each role."""
        r = roles(o)
        return bank(o), r @ role_w, r.mean(0)
    w_tab = torch.tensor(blind_weights(a.ante_weight, a.weight_power, a.weight_geom), device=dev)

    def w_of(o):
        """[N] the reward of the blind being faced (the engine's weights, 24 summing to 24): what --shape-relative scales the shaping by."""
        g = o["g"].float()
        ante_i = (g[:, 0] * 8.0).round().clamp(1, 8).long()
        return w_tab[3 * (ante_i - 1) + g[:, 1:4].argmax(1)]
    phi_prev, psi_prev, _ = potentials(ob)
    w_prev = w_of(ob)
    w_dec = torch.zeros((T, N), device=dev)         # --shape-relative: the blind's weight at each decision
    shaped = torch.zeros((T, N), device=dev)        # the two bonuses' reward terms: moved off the device once
    orac = torch.zeros((T, N), device=dev)          # the oracle potentials' reward terms
    orac_w1, orac_w3 = a.oracle_bonus, a.money_option_bonus

    def ophi(ex_arr):
        return torch.from_numpy(ex_arr[:, 1] * orac_w1 + ex_arr[:, 2] * orac_w3).to(dev).float()
    ophi_prev = ophi(ex_state)
    powers = torch.zeros((T, N), device=dev)        # power(ob, act) of every step; counted where a blind falls
    strengths = torch.zeros((T, N), device=dev)
    phi8_last = np.zeros(N, dtype=np.float32)        # --phi8: the potential each game holds (since its last measurement)
    proj8_last = np.zeros(N, dtype=np.float32)       # --proj8: the potential each game holds (measured at its blind selects)
    proj8_delta = np.zeros((T, N), dtype=np.float32)
    proj8_sum = [0.0, 0]
    # --rental-penalty / --debt-penalty: potentials (charged as the state arises, given back as it ends: they move the credit, not the optimum) on the rental
    # jokers held (each costs $3 a round: the price the shop's $1 hides) and on the dollars of debt
    money_pot_on = a.rental_penalty > 0 or a.debt_penalty > 0
    # --surv-pot / --grow-pen: potentials from the build's clear-chance curve (VecEnv.power_curve: the build projected to each ante's boss), at blind selects:
    #   survival  = the expected reward of every blind still to come if the build stopped changing (sum of blind weight x P(clear it and every blind before))
    #   growth    = minus how many decades the build falls behind the targets over the next two antes (flat builds fall about one a two antes; scalers hold)
    curve_on = a.surv_pot > 0 or a.grow_pen > 0 or a.grow_reward > 0
    grow_rew = np.zeros((T, N), dtype=np.float32)         # --grow-reward: the (non-potential) growth payment of each blind select
    curve_last = np.zeros(N, dtype=np.float32)
    curve_delta = np.zeros((T, N), dtype=np.float32)
    curve_stat = [0.0, 0.0, 0]
    curve_gsum = [0.0]
    _bw = np.asarray(blind_weights(a.ante_weight, a.weight_power, a.weight_geom), dtype=np.float64)       # the 24 blinds' rewards

    def curve_potential(shares, ante, blind):
        """shares [m, 8] the projected share at each ante's boss; ante, blind [m] where each game stands -> (survival, behind) [m]."""
        m_ = len(ante)
        surv = np.zeros(m_); behind = np.zeros(m_); growth = np.zeros(m_)
        shares = shares.copy()
        for k in range(m_):                                  # (antes the engine skipped: log-interpolated between the ones it measured)
            known = [j for j in range(int(ante[k]) - 1, 8) if shares[k, j] > 0]
            if len(known) >= 2:
                shares[k, int(ante[k]) - 1:] = np.exp(np.interp(np.arange(int(ante[k]) - 1, 8), known, np.log(shares[k, known])))
        for k in range(m_):
            alive, tot = 1.0, 0.0
            for b in range(3 * (int(ante[k]) - 1) + int(blind[k]), 24):
                sh = shares[k, b // 3] * (2.0 if b % 3 == 0 else (2.0 / 1.5 if b % 3 == 1 else 1.0))      # (the small / big blind: the boss share x its target ratio)
                p = 1.0 / (1.0 + np.exp(-(1.83 + 2.53 * np.log(max(sh, 1e-4)))))                     # (fitted on solver clears: p = 0.5 at a share of 0.49)
                alive *= p
                tot += _bw[b] * alive
            surv[k] = tot
            a0 = int(ante[k]) - 1
            a2 = min(7, a0 + 2)
            if a2 > a0 and shares[k, a0] > 0 and shares[k, a2] > 0:
                lr_ = float(np.log10(shares[k, a2] / shares[k, a0]))
                behind[k] = min(2.0, max(0.0, -lr_))
                growth[k] = min(2.0, max(-2.0, lr_))
            elif a2 == a0:                                   # (the last ante: nothing ahead to grow toward; counted as keeping pace)
                growth[k] = 0.0
        return surv, behind, growth
    mpot_last = np.zeros(N, dtype=np.float32)
    mpot_delta = np.zeros((T, N), dtype=np.float32)
    from .runeval import Cols as _MC
    from .runeval import Cols as _PCV
    if a.proj8 > 0:
        from .runeval import Cols as _PC
    phi8_delta = np.zeros((T, N), dtype=np.float32)     # strength(ob, act): non-zero at a round's first play / discard
    incomes = torch.zeros((T, N), device=dev)       # money after the step minus money before it
    grows = torch.zeros((T, N), device=dev)         # growth(ob) after the step minus before it
    lasting = torch.zeros((T, N), dtype=torch.bool, device=dev)   # the step changed something lasting (not at a game's end)
    growth_prev = growth(ob)

    def school_growth(o, sv):
        """[N] log of the student's counter (the multiplier it has grown to) in school games, else 0. sv: the
        student's vocabulary id per environment (0: none). jf column 11 is log1p(|counter|) / 4."""
        slot = (o["jid"] == sv[:, None]) & (sv[:, None] > 0)
        val = torch.expm1(4.0 * o["jf"][:, :, 11].float())
        return (torch.log(val.clamp(min=1.0)) * slot.float()).sum(1)

    def student_vocab():
        ids = venv.school_student()
        return torch.from_numpy(ids).to(dev), jvocab[torch.from_numpy(np.maximum(ids, 0)).to(dev)] * torch.from_numpy(ids >= 0).to(dev)
    _, sv_now = student_vocab()
    sg_prev = school_growth(ob, sv_now)
    school_rews = torch.zeros((T, N), device=dev)

    def build_score(o):
        """[N] decades of the build's expected best-play score (g column 205 = log10(1 + score) / 7, filled outside
        rounds by the marginal inputs), capped at --build-score-cap; -1 inside a round (no reading)."""
        v = (7.0 * o["g"][:, 205].float()).clamp(0.0, a.build_score_cap)
        return torch.where(o["g"][:, 5] > 0.5, torch.full_like(v, -1.0), v)
    bs_last = build_score(ob).clamp(min=0.0)
    bs_rews = torch.zeros((T, N), device=dev)
    money_prev = money_of(ob)
    beaten = np.zeros((T, N), dtype=np.float32)     # a play beat a blind at this step
    cleared = np.zeros((T, N), dtype=bool)          # a blind was beaten at this step (the prediction heads)
    finals = np.zeros((T, N), dtype=np.float32)     # where a game ended: the blinds it beat
    fin_ante = np.zeros((T, N), dtype=np.float32)   # ... the ante it ended in, and whether it was won (the outcome heads' targets)
    fin_won = np.zeros((T, N), dtype=np.float32)
    use_reach = a.reach_coef > 0 and hasattr(model, "reach")
    prev_blinds = np.zeros(N, dtype=np.int64)
    # the rollout on the device, stored narrow: action features as float16, card subsets as int8, item ids of
    # the actions as int16 (the model widens the rows it reads)
    narrow = {"af": torch.float16, "priv": torch.float16, "subs": torch.int8, "aid": torch.int16, "aid2": torch.int16}
    buf = {k: torch.empty((T, N) + tuple(v.shape[1:]), dtype=narrow.get(k, v.dtype), device=dev)
           for k, v in ob.items()}
    demos = None
    if a.demo:                                         # --demo: strong lines' build decisions, imitated alongside PPO
        from .demo_data import DemoSet
        demos = DemoSet(a.demo, {k: v[:1].to(narrow.get(k, v.dtype)) for k, v in ob.items()}, dev, seed=a.seed, won_weight=a.demo_won_weight)
        print(f"demonstrations: {demos.n} decisions from {a.demo} (won lines {100 * demos.won_share:.0f}%)", flush=True)
        if a.demo_pretrain > 0 and it0 == 0:
            model.train()
            for k_ in range(a.demo_pretrain):
                dob, dact = demos.sample(a.demo_batch)
                dlogits, _ = fwd(dob)
                l_demo = F.cross_entropy(dlogits.float(), dact)
                opt.zero_grad()
                l_demo.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                if (k_ + 1) % 200 == 0:
                    print(f"demo pretrain {k_ + 1}/{a.demo_pretrain}: loss {l_demo.item():.3f}, accuracy {(dlogits.argmax(1) == dact).float().mean().item():.3f}", flush=True)
            model.eval()
    acts = torch.empty((T, N), dtype=torch.long, device=dev)
    lps = torch.empty((T, N), dtype=torch.float32, device=dev)
    vals = np.zeros((T, N, N_OUT) if a.streams else (T, N), dtype=np.float32)
    rews = np.zeros((T, N), dtype=np.float32)
    dones = np.zeros((T, N), dtype=np.float32)
    sparse_r = np.zeros((T, N), dtype=np.float32)       # --streams: the sparse part of the reward
    truncs = np.zeros((T, N), dtype=bool)               # --streams: the episode was ended by the step limit (bootstrapped, unlike a loss or a win)
    stats = EpisodeStats(n=a.stats_window)          # outcomes of the last stats_window finished games
    ep_ret = np.zeros(N)
    partner_paths = [x.strip() for x in a.partners.split(",") if x.strip()]
    partner_models = []
    pop_rng = np.random.default_rng(a.seed + 12345)
    ctrl = np.full(N, -1, dtype=np.int64)                 # who plays each environment now: -1 this policy, k a partner
    just_done = np.ones(N, dtype=bool)

    def reload_partners():
        import shutil
        ms = []
        for pth in partner_paths:
            if os.path.exists(pth):
                try:
                    tmp = pth + ".pop_tmp.pt"
                    shutil.copy2(pth, tmp)
                    m_ = load_model(tmp, dev)
                    m_.eval()
                    ms.append(m_)
                except Exception:
                    pass
        return ms
    act_hist = [[] for _ in range(N)]               # --record-wins: the actions of the game each environment is in
    ep_count = np.zeros(N, dtype=np.int64)          # games each environment has finished: env e plays the seed
    wins_path = a.out.replace(".pt", "_wins.jsonl")  #   venv.seed0 + e + N * ep_count[e]
    t0 = time.time() - 60.0 * min0
    best = -1.0
    sim_s = 0.0
    games, wins = games0, wins0
    ante_hist = np.zeros(10, dtype=np.int64)        # games by ante reached, since the last log row
    # --money-target: the money curriculum. `extra` dollars are added at the start of money_share of the games;
    # it moves so that those games, on the target stake, are won about money_target of the time
    extra = float(resumed.get("money_extra", a.money_start))
    rich_rate = float(resumed.get("rich_win", a.money_target))      # their win rate, smoothed over iterations
    rich = (10.0 ** (3.0 * venv.obs["g"][:, 18]) - 1.0) > 4.5        # this game began with extra money
    normal_games, normal_wins = int(resumed.get("normal_games", 0)), int(resumed.get("normal_wins", 0))
    menu_prev = -1
    for it in range(it0 + 1, a.iters + 1):
        # linear schedules over the run: the learning rate to lr_final, the entropy bonus to ent_final
        frac = (it - 1) / max(1, a.iters - 1)
        # --shape-fade-iters: the shaping terms and the head starts fade over this many updates (then stay at their final values), not over the run
        shape_frac = min(1.0, (it - 1) / max(1, a.shape_fade_iters)) if a.shape_fade_iters > 0 else frac
        any_now = a.start_any + (a.start_any_final - a.start_any) * shape_frac if a.start_any_final is not None else a.start_any
        xm_now = (a.start_xmult + (a.start_xmult_final - a.start_xmult) * shape_frac if a.start_xmult_final is not None
                  else a.start_xmult)
        money_p = max(0.0, a.start_money_p + (a.start_money_p_final - a.start_money_p) * shape_frac
                      if a.start_money_p_final is not None else a.start_money_p)
        money2_p = max(0.0, a.start_money2_p + (a.start_money2_p_final - a.start_money2_p) * shape_frac
                       if a.start_money2_p_final is not None else a.start_money2_p)
        sg0 = max(0.0, a.start_grow + (a.start_grow_final - a.start_grow) * shape_frac
                  if a.start_grow_final is not None else a.start_grow)
        if a.money_target > 0:
            venv.set_start(xm_now, any_now, a.money_share, int(round(extra)), 0.0, 0, sg0)
        else:
            venv.set_start(xm_now, any_now, money_p, a.start_money, money2_p, a.start_money2, sg0)
        if a.menu_slots > 2:
            menu_now = a.menu_slots + (2 - a.menu_slots) * min(1.0, (it - 1) / max(1, a.menu_fade_iters))
            menu_now = int(round(menu_now))
            menu_now = menu_now if menu_now > 2 else 0
            if menu_now != menu_prev:
                venv.set_menu(menu_now)
                menu_prev = menu_now
        if a.archive_ante != "reverse" and (a.archive_p > 0 or a.archive_store > 0):
            last_out = venv.archive_outcomes(True).tolist()       # the archive games' outcomes by start blind, for the log (every band mode)
        if a.archive_ante == "reverse" and (a.archive_p > 0 or a.archive_store > 0):
            out_now = venv.archive_outcomes(True)
            band_acc += out_now
            last_out = out_now.tolist()
            b0 = 3 * (band_lo - 1)
            ends, wons = int(band_acc[0, b0:b0 + 3].sum()), int(band_acc[1, b0:b0 + 3].sum())
            band_wr = wons / ends if ends else None
            timed_out = a.archive_stage_max > 0 and it - band_since >= a.archive_stage_max
            if band_lo > 1 and ((ends >= a.archive_advance_games and band_wr >= a.archive_advance) or timed_out):
                print(json.dumps({"archive_band": {"extended_to": band_lo - 1, "win_rate": None if band_wr is None else round(band_wr, 3), "games": ends,
                                                   "iter": it, "timed_out": bool(timed_out and not (ends >= a.archive_advance_games and band_wr >= a.archive_advance))}}), flush=True)
                band_lo -= 1
                band_acc[:] = 0
                band_wr = None
                band_since = it
            if band_lo == 1 and fade_from is None:
                fade_from = it
            arch_p_now = a.archive_p
            if fade_from is not None and a.archive_p_final is not None:      # the help fades once the start has reached ante 1
                arch_p_now = a.archive_p + (a.archive_p_final - a.archive_p) * min(1.0, (it - fade_from) / max(1, a.archive_fade_iters))
            lo_a, hi_a = band_lo, 8
            venv.set_archive(arch_p_now, a.archive_store, lo_a, hi_a, a.archive_cap, a.archive_min, a.archive_winners)
        if a.archive_ante == "auto" and (a.archive_p > 0 or a.archive_store > 0):
            # the archive's start range follows the policy: from one ante below to two above the ante its games typically reach (blinds beaten / 3)
            front = int(stats.summary().get("blinds", 0.0) // 3) + 1
            lo_a, hi_a = max(1, front - 1), min(8, front + 2)
            venv.set_archive(a.archive_p, a.archive_store, lo_a, hi_a, a.archive_cap, a.archive_min, a.archive_winners)
        if scen_on:
            scn_p_now, scn_mix_now = scenario_schedule(a, it)
            venv.set_scenario(scn_p_now, *scn_mix_now, a.scenario_min_ante, a.scenario_max_ante, a.scenario_reroll_cap, a.scenario_couple)
        if a.immortal and a.immortal_fade:
            ante_mult = np.clip(1.0 - (ante_ema - a.fade_start) / max(1e-6, a.fade_end - a.fade_start), a.fade_floor, 1.0)
            venv.set_blind_mult(np.repeat(ante_mult, 3))
        clear_acc = np.zeros(24, dtype=np.int64)            # --immortal: games in which each blind was beaten, this update
        clear_games = 0
        scn_stat = np.zeros((4, 4), dtype=np.int64)       # scenario games finished this iteration, by kind: [games, cut off (truncated), lost, won]
        # the scaled head start (--start-xscale): its share and the multiplier it is handed over at fade together
        xf = min(1.0, (it - 1) / max(1, a.start_xscale_fade_iters))
        xs_share = max(0.0, a.start_xscale + (a.start_xscale_final - a.start_xscale) * xf
                       if a.start_xscale_final is not None else a.start_xscale)
        xs_val = max(1.0, a.start_xscale_val + (a.start_xscale_val_final - a.start_xscale_val) * xf
                     if a.start_xscale_val_final is not None else a.start_xscale_val)
        if a.start_xscale > 0:
            venv.set_xscale(xs_share, xs_val)
        lc_share = max(0.0, a.level_cap_share + (a.level_cap_share_final - a.level_cap_share) * frac
                       if a.level_cap_share_final is not None else a.level_cap_share)
        if a.level_cap_share > 0:
            venv.set_level_cap(lc_share, a.level_cap_n, a.level_cap_ante)
        if a.stake_curriculum and it % a.stake_window == 0:
            n_hard = int(target.sum())
            floor_n = int(round(a.stake_floor * N))
            if it > a.stake_warmup and cur_games >= a.stake_min_games:
                cur = cur_blinds / cur_games
                if cur_prev is not None and cur - cur_prev < a.stake_min_gain and n_hard > floor_n:
                    move = min(max(1, int(round(a.stake_step * N))), n_hard - floor_n)
                    _new = list(env_stake)
                    for _i in np.flatnonzero(target)[n_hard - move:]:
                        _new[_i] = a.stake_easy
                    venv.set_stakes(_new)
                    env_stake = np.array(venv.stakes)
                    target = env_stake == a.stake
                    print(f"stake curriculum: {a.stake} blinds {cur:.2f} (was {cur_prev:.2f}): {move} environments move to "
                          f"{a.stake_easy}, {int(target.sum())} left on {a.stake}", flush=True)
                cur_prev = cur
            cur_blinds, cur_games = 0.0, 0
        ban_now = max(0.0, a.ban_share + (a.ban_share_final - a.ban_share) * frac
                      if a.ban_share_final is not None else a.ban_share)
        tilt_now = max(0.0, a.tilt_share + (a.tilt_share_final - a.tilt_share) * frac
                       if a.tilt_share_final is not None else a.tilt_share)
        if a.ban_top > 0:
            if it % a.ban_every == 0 or (not ban_ids and buy_counts.sum() >= 2000):
                jc = buy_counts[jvocab]
                ban_ids = jc.topk(min(a.ban_top, len(joker_keys))).indices.tolist() if jc.sum() > 0 else ban_ids
                buy_counts.zero_()
            venv.set_ban(ban_ids, ban_now if ban_ids else 0.0)
        if a.tilt_share > 0:
            venv.set_tilt(tilt_now, a.tilt_levels)
        # the school starts scaled and is toned down: its share of the games and its bonus move linearly to their
        # final values over --school-fade-iters iterations
        sfrac = min(1.0, (it - 1) / max(1, a.school_fade_iters))
        school_share_now = max(0.0, a.school_share + (a.school_share_final - a.school_share) * sfrac
                               if a.school_share_final is not None else a.school_share)
        school_now = max(0.0, a.school_bonus + (a.school_bonus_final - a.school_bonus) * sfrac
                         if a.school_bonus_final is not None else a.school_bonus)
        if school_students:
            venv.set_school(school_share_now, school_students)
        ban_games = ban_blinds = norm_games = norm_blinds = 0
        rich_n = rich_w = 0
        kinds = torch.zeros(N_KINDS, device=dev)
        final_boss = 0
        if a.lr_steps:                                   # --lr-steps: AlphaZero-style step schedule, the rate cut by --lr-step-factor at each listed update
            lr_now = a.lr * a.lr_step_factor ** sum(it > s_ for s_ in a.lr_steps) * a.lr_scale
        else:
            lr_now = (a.lr + (a.lr_final - a.lr) * frac if a.lr_final is not None else a.lr) * a.lr_scale
        warm = (it - it0) <= a.value_warmup                  # the critic alone learns the reward's scale first
        model.value_detach = warm
        if warm:
            lr_now *= a.value_warmup_lr_mult
        ent_now = a.ent + (a.ent_final - a.ent) * frac if a.ent_final is not None else a.ent
        demo_now = (a.demo_coef + (a.demo_coef_final - a.demo_coef) * min(1.0, (it - 1) / max(1, a.demo_fade_iters))
                    if a.demo_coef_final is not None else a.demo_coef) if demos is not None else 0.0
        dl_, dacc_ = [], []
        ent_bump_now = 0.0
        if a.win_only:
            fade.w[:] = 0.0                                  # (the shaped stream never drives the policy)
        elif a.streams and not a.stream_equal:
            if reach.ready():
                if fade.update(reach.rates()):               # a fade begins: explore again for a while, as the greedy spending is no longer paid
                    ent_bump.trigger()
            ent_bump_now = ent_bump.value()                  # (its own term in the loss: decisions outside rounds only; ent_now stays the base)
            ent_bump.step()
        if ladder:                                           # the same kind of term after the stake moved up
            ent_bump_now += ladder_bump.value()
            ladder_bump.step()
        if a.stall_ent:                                      # ... and after the win rate stalled
            ent_bump_now += stall_bump.value()
            stall_bump.step()
        # --bank-bonus: reward bank_now * (gamma * phi(next) - phi(now)), phi the share of a full bank; it is
        # given as money is banked and taken back as it is spent or when the game ends, so over a game it
        # sums to (about) zero: it tells the critic early that money is worth something, no more
        la_coef[:] = [max(0.0, a.bank_bonus + (a.bank_bonus_final - a.bank_bonus) * shape_frac if a.bank_bonus_final is not None else a.bank_bonus),
                      max(0.0, a.build_bonus + (a.build_bonus_final - a.build_bonus) * shape_frac if a.build_bonus_final is not None else a.build_bonus)]
        bank_now = max(0.0, a.bank_bonus + (a.bank_bonus_final - a.bank_bonus) * shape_frac
                       if a.bank_bonus_final is not None else a.bank_bonus)
        # --oracle-bonus / --money-option-bonus: potentials on the ante-8 oracle's log best share and on the money-option curve (see attach_ex):
        # a stronger projected build and the money to fish with are rewarded as they are gained and taken back as they go
        orac_w1 = max(0.0, a.oracle_bonus + (a.oracle_bonus_final - a.oracle_bonus) * shape_frac
                      if a.oracle_bonus_final is not None else a.oracle_bonus)
        orac_w3 = max(0.0, a.money_option_bonus + (a.money_option_bonus_final - a.money_option_bonus) * shape_frac
                      if a.money_option_bonus_final is not None else a.money_option_bonus)
        orac_acc[:] = 0.0
        orac_n[0] = 0
        orac_calls[0] = 0
        # --build-bonus: the same kind of term for the jokers held: a multiplier joker is worth build_now, a
        # +mult joker half of it, a chips joker a third (once each, however many are held)
        build_now = max(0.0, a.build_bonus + (a.build_bonus_final - a.build_bonus) * shape_frac
                        if a.build_bonus_final is not None else a.build_bonus)
        # --power-bonus: paid when a play beats a blind, times how strongly (power, above): a stronger build
        # is worth more at every blind, not only at the one where a weaker build would have died
        power_now = max(0.0, a.power_bonus + (a.power_bonus_final - a.power_bonus) * shape_frac
                        if a.power_bonus_final is not None else a.power_bonus)
        # --strength-bonus: paid once a round, at its first play or discard, times the build's strength (above)
        strength_now = max(0.0, a.strength_bonus + (a.strength_bonus_final - a.strength_bonus) * shape_frac
                           if a.strength_bonus_final is not None else a.strength_bonus)
        rounds_started = torch.zeros((), device=dev)
        auto_n = sel_n = 0
        play_t = np.zeros((T, N), dtype=np.float32)          # the chips the play of step t scored (0: no play)
        buy_acc = torch.zeros(8, 6, device=dev)              # jokers bought / picked by ante (real games): +mult, xMult, scaling, economy, chips, all
        # --grow-bonus: times the change in growth() at every step: paid as a held growing joker's counter rises
        # and taken back when it falls or the joker goes; what is held when the game ends is kept
        grow_now = max(0.0, a.grow_bonus + (a.grow_bonus_final - a.grow_bonus) * shape_frac
                       if a.grow_bonus_final is not None else a.grow_bonus)
        sg_now = max(0.0, a.start_grow + (a.start_grow_final - a.start_grow) * shape_frac
                     if a.start_grow_final is not None else a.start_grow)
        growth_sum = torch.zeros((), device=dev)
        bs_sum = 0.0
        # --income-bonus: per dollar earned when a blind is beaten (the round's payout: the blind's reward, unused
        # hands, interest, jokers' and gold cards' payouts, less rental fees), up to --income-cap dollars
        income_now = max(0.0, a.income_bonus + (a.income_bonus_final - a.income_bonus) * shape_frac
                         if a.income_bonus_final is not None else a.income_bonus)
        bank_sum = torch.zeros((), device=dev)
        role_sum = torch.zeros(5, device=dev)
        x_offer = torch.zeros((), device=dev)
        e_offer = torch.zeros((), device=dev)
        e_take = torch.zeros((), device=dev)
        x_take = torch.zeros((), device=dev)
        for gp in opt.param_groups:
            gp["lr"] = lr_now
        model.eval()
        if partner_paths and (it - it0) % a.pop_reload == 1 % a.pop_reload:
            partner_models = reload_partners()
        own_np = np.ones((T, N), dtype=bool)
        act_state["bal"] = act_state["next"]
        for t in range(T):
            with torch.no_grad():
                logits, v = fwd(ob)
                if vloop is not None:
                    vloop.collect(ob["mask"].cpu().numpy(), v)
                xt = xmult_takes(ob)
                et = item_takes(ob)
                dist = policy(logits, ob["mask"], ob, xt, et)
                act = dist.sample()
                if partner_models:
                    at_sel = (ob["g"][:, 4] > 0.5).cpu().numpy()                      # a blind select
                    sw = (at_sel & (pop_rng.random(N) < a.pop_handoff)) | just_done
                    ctrl = np.where(sw, pop_rng.integers(-1, len(partner_models), size=N), ctrl)
                    for k_ in range(len(partner_models)):
                        sel = np.flatnonzero(ctrl == k_)
                        if len(sel):
                            st_ = torch.from_numpy(sel).to(dev)
                            sub = {kk: vv[st_] for kk, vv in ob.items()}
                            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                                lg_, _ = partner_models[k_](sub)
                            act[st_] = torch.multinomial(torch.softmax(lg_.float(), 1), 1).squeeze(1)
                    own_np[t] = ctrl == -1
                lp = dist.log_prob(act)
                e_offer += (et.sum(1) > 0).sum()
                e_take += et.gather(1, act[:, None]).sum()
                x_offer += (xt.sum(1) > 0).sum()
                x_take += xt.gather(1, act[:, None]).sum()
            for k in buf:
                buf[k][t] = ob[k]
            acts[t], lps[t] = act, lp
            kinds += torch.bincount(type_of[act], minlength=N_KINDS)
            if a.autoplay:                                # the share of blind selects the network hands to the solver
                at_sel = ob["g"][:, 4] > 0.5
                auto_n += int(((act == A_LEAVE_ID) & at_sel).sum())
                sel_n += int(at_sel.sum())
            if a.ban_top > 0:
                was_restricted = venv.restricted()
                # the list comes from what it buys when the whole pool is there (a banned game's purchases are
                # what it buys instead, and would put the stand-ins on the list)
                is_buy = torch.isin(type_of[act], buy_kinds) & ~torch.from_numpy(was_restricted).to(dev)
                buy_counts += torch.bincount(ob["aid"].gather(1, act[:, None]).squeeze(1)[is_buy], minlength=len(_VOCAB)).float()
            vals[t] = v.cpu().numpy()
            w_dec[t] = w_of(ob)
            kb = venv.scn_kinds() if scen_on else None        # the kind of the game each environment is in (read before the step: a finished one auto-resets)
            with torch.no_grad():
                real_t = torch.ones(N, dtype=torch.bool, device=dev) if kb is None else torch.from_numpy(kb < 0).to(dev)
                bought = torch.isin(type_of[act], buy_kinds) & real_t
                if bool(bought.any()):
                    ids = ob["aid"].gather(1, act[:, None]).squeeze(1)[bought].long()
                    ante_i = (ob["g"][:, 0] * 8.0).round().clamp(1, 8).long()[bought] - 1
                    buy_acc.index_add_(0, ante_i, joker_cat[ids])
            ts = time.time()
            act_np = act.cpu().numpy()
            with torch.no_grad():
                powers[t] = power(ob, act)
                strengths[t], fresh = strength(ob, act)
                if money_pot_on:
                    if t == 0:
                        mpot_delta[:] = 0.0
                    rent_n = ((ob["jf"][..., 8].float() > 0.5) & (ob["jid"] > 0)).sum(1).float()
                    debt_d = (-ob["g"][:, _MC.MONEY].float() * 50.0).clamp(min=0.0)
                    ph_m = (-(a.rental_penalty * rent_n + a.debt_penalty * debt_d)).cpu().numpy().astype(np.float32)
                    mpot_delta[t] = ph_m - mpot_last
                    mpot_last = ph_m
                if curve_on:                                 # the clear-chance curve's potentials, at each blind select
                    if t == 0:
                        curve_delta[:] = 0.0
                        curve_stat[:] = [0.0, 0.0, 0]
                        curve_gsum[0] = 0.0
                        grow_rew[:] = 0.0
                    g_c = np.asarray(venv.obs["g"])
                    sel_c = np.flatnonzero(np.argmax(g_c[:, _PCV.STATE:_PCV.STATE + 4], 1) == 0)
                    if len(sel_c):
                        shares = np.asarray(venv.v.power_curve([int(i) for i in sel_c], a.curve_samples), dtype=np.float64).reshape(-1, 8)
                        ante_c = np.clip(np.rint(g_c[sel_c, 0] * 8), 1, 8).astype(int)
                        blind_c = np.argmax(g_c[sel_c, 1:4], 1)
                        sv, bh, gr = curve_potential(shares, ante_c, blind_c)
                        if a.grow_reward > 0:                # paid, not refunded: W x clamp(growth + 1, 0, 3) (a flat build ~0, keeping pace ~W, pulling ahead up to 3W)
                            if t == 0:
                                grow_rew[:] = 0.0
                            grow_rew[t, sel_c] = a.grow_reward * np.clip(gr + 1.0, 0.0, 3.0)
                        ph_c = (a.surv_pot * sv - a.grow_pen * bh).astype(np.float32)
                        curve_delta[t, sel_c] = ph_c - curve_last[sel_c]
                        curve_last[sel_c] = ph_c
                        curve_stat[0] += float(sv.sum()); curve_stat[1] += float(bh.sum()); curve_stat[2] += len(sel_c)
                        curve_gsum[0] += float(gr.sum())
                if a.proj8 > 0:                              # the build projected to the ante-8 boss (scalers grown at their natural rates), at each blind select
                    if t == 0:
                        proj8_delta[:] = 0.0
                        proj8_sum[:] = [0.0, 0]                 # (the log share is reported per update)
                    g_np = np.asarray(venv.obs["g"])
                    at_sel = np.flatnonzero(np.argmax(g_np[:, _PC.STATE:_PC.STATE + 4], 1) == 0)
                    if len(at_sel):
                        pp = np.asarray(venv.v.projected_power([int(i) for i in at_sel], a.proj8_samples), dtype=np.float64)
                        ph = np.clip((np.log10(np.maximum(pp, 1e-9)) + a.proj8_span) / a.proj8_span, 0.0, 1.0).astype(np.float32)
                        proj8_delta[t, at_sel] = a.proj8 * (ph - proj8_last[at_sel])
                        proj8_last[at_sel] = ph
                        proj8_sum[0] += float(np.log10(np.maximum(pp, 1e-9)).sum()); proj8_sum[1] += len(at_sel)
                if a.phi8 > 0:
                    if t == 0:
                        phi8_delta[:] = 0.0
                    p8 = phi8_val(ob).cpu().numpy()
                    fr = fresh.cpu().numpy() > 0
                    phi8_delta[t] = np.where(fr, a.phi8 * (p8 - phi8_last), 0.0)
                    phi8_last = np.where(fr, p8, phi8_last).astype(np.float32)
                rounds_started += fresh.sum()
            if split_step:
                r, d, info, sparse_r[t], truncs[t] = venv.step_split(act_np)
            else:
                r, d, info = venv.step(act_np)
            just_done = np.asarray(d, dtype=bool)
            if use_play:
                play_t[t] = venv.play_scores()
            sim_s += time.time() - ts
            rews[t], dones[t] = r, d
            if curve_on:                                     # a finished game gives its potential back
                ddc = np.asarray(d, dtype=bool)
                curve_delta[t] -= np.where(ddc, curve_last, 0.0)
                curve_last[ddc] = 0.0
            if money_pot_on:                                 # a finished game gives its potential back (the next game starts from 0)
                ddm = np.asarray(d, dtype=bool)
                mpot_delta[t] -= np.where(ddm, mpot_last, 0.0)
                mpot_last[ddm] = 0.0
            if a.proj8 > 0:                                  # a finished game gives its potential back
                dd8 = np.asarray(d, dtype=bool)
                proj8_delta[t] -= np.where(dd8, a.proj8 * proj8_last, 0.0)
                proj8_last[dd8] = 0.0
            if a.phi8 > 0:                                   # a finished game gives its potential back (phi of the terminal state is 0)
                dd = np.asarray(d, dtype=bool)
                phi8_delta[t] -= np.where(dd, a.phi8 * phi8_last, 0.0)
                phi8_last[dd] = 0.0
            beaten[t] = (info["blinds"] > prev_blinds) & (act_np < N_SUB)
            cleared[t] = info["blinds"] > prev_blinds
            finals[t] = info["blinds"]
            fin_ante[t] = info["ante"]
            fin_won[t] = info["won"]
            prev_blinds = np.where(d, 0, info["blinds"])
            ep_ret += r
            if a.record_wins:
                for e in range(N):
                    act_hist[e].append(int(act_np[e]))
            if d.any():
                began_rich = (10.0 ** (3.0 * venv.obs["g"][:, 18]) - 1.0) > 4.5    # of the games that start now
            if a.immortal and d.any():
                bits = venv.last_bits()
                for e in np.flatnonzero(d):
                    clear_acc += (int(bits[e]) >> np.arange(24)) & 1
                    clear_games += 1
            for e in np.flatnonzero(d):
                is_real = kb is None or kb[e] < 0
                tg = bool(target[e]) and is_real           # (a scenario is no game of the stake being learned: its outcome is not the win rate)
                if not is_real:
                    k_ = int(kb[e])
                    scn_stat[k_] += [1, int(truncs[t][e]), int(not truncs[t][e] and not info["won"][e]), int(info["won"][e])]
                if a.ban_top > 0 and tg:
                    if was_restricted[e]:
                        ban_games += 1; ban_blinds += int(info["blinds"][e])
                    else:
                        norm_games += 1; norm_blinds += int(info["blinds"][e])
                if tg:
                    if rich[e]:
                        rich_n += 1
                        rich_w += int(info["won"][e])
                    else:
                        normal_games += 1
                        normal_wins += int(info["won"][e])
                rich[e] = began_rich[e]
                games += 1
                games_by[str(env_stake[e])] = games_by.get(str(env_stake[e]), 0) + 1
                if tg:
                    cur_blinds += float(info["blinds"][e]); cur_games += 1
                if tg:
                    stats.returns.append(float(ep_ret[e]))
                    stats.add({"episode_end": True, "blinds": int(info["blinds"][e]), "ante": int(info["ante"][e]),
                               "won": bool(info["won"][e])})
                    final_boss += int(info["blinds"][e] >= 23)   # reached the ante-8 boss (23 beaten: lost to it)
                    reach.add(int(info["ante"][e]), bool(info["won"][e]))
                ep_ret[e] = 0.0
                if info["won"][e] and is_real:
                    with open(wins_path, "a") as f:
                        f.write(json.dumps({"iter": it, "games": games, "seed": int(venv.seed0 + e + N * ep_count[e]),
                                            "stake": str(env_stake[e]), "start_any": round(any_now, 3),
                                            "start_xmult": round(xm_now, 3),
                                            **({"actions": act_hist[e]} if a.record_wins else {})}) + "\n")
                ep_count[e] += 1
                act_hist[e] = []
                if tg:
                    wins += int(info["won"][e])
                    ante_hist[min(int(info["ante"][e]), 8) + (1 if info["won"][e] else 0)] += 1
                elif info["won"][e] and is_real:
                    wins_other[str(env_stake[e])] = wins_other.get(str(env_stake[e]), 0) + 1
            ob = current()
            with torch.no_grad():
                lasting[t] = changed_lasting({k: buf[k][t] for k in ("jid", "cid", "jf", "g")}, ob) & torch.from_numpy(~d).to(dev)
                growth_new = growth(ob)
                w_new = w_of(ob)
                grows[t] = (w_new * growth_new - w_prev * growth_prev) if a.shape_relative else growth_new - growth_prev
                if a.build_score_bonus > 0:
                    bs_new = build_score(ob)
                    bs_cur = torch.where(bs_new < 0, bs_last, bs_new)      # in a round: no new reading
                    bs_rews[t] = bs_cur - bs_last
                    bs_last = bs_cur
                    bs_sum += float(bs_cur.mean())
                if school_students:
                    _, sv_now = student_vocab()
                    sg_new = school_growth(ob, sv_now)
                    school_rews[t] = sg_new - sg_prev
                    sg_prev = sg_new
                growth_sum += growth_prev.mean()
                growth_prev = growth_new
                money_new = money_of(ob)
                incomes[t] = money_new - money_prev
                money_prev = money_new
                phi_new, psi_new, held = potentials(ob)
                live = torch.from_numpy(~d).to(dev).float()          # the game goes on (its potential counts)
                if a.shape_relative:                            # the potential is W(s) x the quantity: it still telescopes to zero over a game
                    shaped[t] = (bank_now * (a.gamma * live * w_new * phi_new - w_prev * phi_prev)
                                 + build_now * (a.gamma * live * w_new * psi_new - w_prev * psi_prev))
                else:
                    shaped[t] = (bank_now * (a.gamma * live * phi_new - phi_prev)
                                 + build_now * (a.gamma * live * psi_new - psi_prev))
                bank_sum += phi_prev.mean()
                role_sum += held
            phi_prev, psi_prev = phi_new, psi_new
            w_prev = w_new
            if orac_on:
                attach_ex(venv, ex_state, ob)
                ophi_new = ophi(ex_state)
                orac[t] = a.gamma * live * ophi_new - ophi_prev
                ophi_prev = ophi_new
        if bank_now > 0 or build_now > 0:
            rews += shaped.cpu().numpy()
        if orac_on and (orac_w1 > 0 or orac_w3 > 0):
            rews += orac.cpu().numpy()
        w_np = w_dec.cpu().numpy() if a.shape_relative else 1.0
        if strength_now > 0:
            rews += strength_now * strengths.cpu().numpy() * w_np
        if a.phi8 > 0:
            rews += phi8_delta
        if a.proj8 > 0:
            rews += proj8_delta
        if money_pot_on:
            rews += mpot_delta
        if curve_on:
            rews += curve_delta
            if a.grow_reward > 0:
                rews += grow_rew
        if grow_now > 0:
            rews += grow_now * grows.cpu().numpy() * (1.0 - dones)     # (a finished game's last step: the next game's)
        if a.build_score_bonus > 0:
            # the change in the build's expected score (decades), at the steps outside rounds; a finished game's last
            # step is the next game's first reading, so it is dropped
            rews += a.build_score_bonus * bs_rews.cpu().numpy() * (1.0 - dones)
        if school_students and school_now > 0:
            rews += school_now * school_rews.cpu().numpy() * (1.0 - dones)
        paid = beaten * (1.0 - dones)                    # (the money of a finished game's last step is the next game's)
        inc = np.clip(incomes.cpu().numpy(), 0.0, a.income_cap) * paid
        if income_now > 0:
            rews += income_now * inc * w_np
        if a.money_target > 0 and rich_n >= 20:
            rich_rate = 0.8 * rich_rate + 0.2 * rich_w / rich_n
            extra = extra * float(np.exp(-a.money_gain * (rich_rate - a.money_target))) + (1.0 if rich_rate < a.money_target else 0.0)
            extra = float(min(max(extra, 0.0), a.money_max))
        pw = powers.cpu().numpy() * beaten
        if power_now > 0:
            rews += power_now * pw * w_np
        if a.immortal and a.immortal_fade and clear_games > 0:
            rate_now = (clear_acc / clear_games).reshape(8, 3).mean(1)
            ante_ema = a.fade_ema * ante_ema + (1.0 - a.fade_ema) * rate_now
        ret_std = 1.0
        if a.ret_norm and not a.streams:
            norm_ret.update(rews, dones)
            ret_std = norm_ret.std
            rews = rews / ret_std
        bank_sum, role_sum = float(bank_sum), role_sum.cpu().numpy()
        x_offer, x_take = int(x_offer), int(x_take)
        if a.explore_actions > 0:                           # the take rate of every action outside rounds, for the next update's weights (the update just below still uses this one's)
            nr_ = buf["g"][:, :, 5] <= 0.5
            if bool(nr_.any()):
                cnt_ = torch.bincount(acts[nr_], minlength=_N_ACT).float()
                act_state["freq"] = 0.9 * act_state["freq"] + 0.1 * cnt_ / cnt_.sum().clamp(min=1)
                act_state["next"] = (1.0 / (act_state["freq"] + 0.002)).clamp(max=100.0)
        with torch.no_grad():
            if use_aux:
                _, last_v, last_x = fwd_aux(ob)
                nxt = torch.sigmoid(last_x[:, 0]).cpu().numpy()                # games still going at the end of
                fin = last_x[:, 1].clamp(0.0, 24.0 / AUX_BLINDS).cpu().numpy()   # the rollout: its own prediction
            else:
                _, last_v = fwd(ob)
            last_v = last_v.cpu().numpy()
            if use_reach:                                   # games still going at the end of the rollout: the heads' own prediction
                last_r = model.reach_probs().cpu().numpy()
        if use_aux:
            # targets, back from the end of the rollout: is the blind faced at step t beaten (before the game
            # ends), and the blinds the game of step t ends with
            aux_next = np.zeros((T, N), dtype=np.float32)
            aux_final = np.zeros((T, N), dtype=np.float32)
            for t in reversed(range(T)):
                over = dones[t] > 0.5
                nxt = np.where(cleared[t], 1.0, np.where(over, 0.0, nxt))
                fin = np.where(over, finals[t] / AUX_BLINDS, fin)
                aux_next[t], aux_final[t] = nxt, fin
            flat_next = torch.tensor(aux_next.reshape(-1), device=dev)
            flat_final = torch.tensor(aux_final.reshape(-1), device=dev)
        if use_reach:
            # targets, back from the end of the rollout: did the game of step t reach ante 5 / 6 / 7 / 8, was it won (a won game reached them all)
            rr = last_r.astype(np.float32)
            reach_t = np.zeros((T, N, 5), dtype=np.float32)
            for t in reversed(range(T)):
                over = dones[t] > 0.5
                won_t = fin_won[t] > 0.5
                y = np.stack([(fin_ante[t] >= k) | won_t for k in (5, 6, 7, 8)] + [won_t], 1).astype(np.float32)
                rr = np.where(over[:, None], y, rr)
                reach_t[t] = rr
            flat_reach = torch.tensor(reach_t.reshape(-1, 5), device=dev)
            r_pred_acc, r_bce_acc = [], []
        # --streams: each stream (the shaped one by ante, the sparse one) has its own normalised reward, value output and GAE; the policy's
        # advantage adds them back in reward units, the shaped components weighted by the fade (streams.py)
        adv_streams, ret_k, ev_sh, ev_sp, sh_std, sp_std = None, None, float("nan"), float("nan"), 1.0, 1.0
        if a.streams and a.stream_equal:
            sh = rews - sparse_r
            ante_t = np.clip(np.rint(buf["g"][:, :, 0].float().cpu().numpy() * 8.0), 1, 8)
            comp = by_ante(sh, ante_t)                                        # [T, N, 8]: each blind's reward in the column of the ante it was earned in
            for k in range(N_ANTES):
                norm_c[k].update(comp[..., k], dones)
            stds = np.array([n.std for n in norm_c])
            norm_sp.update(sparse_r, dones)
            sp_std = norm_sp.std
            sh_std = float(stds.mean())
            rk = np.concatenate([comp / stds, (sparse_r / sp_std)[..., None]], -1)
            adv_k = gae_streams(rk, vals, last_v, dones, truncs, a.gamma, a.lam)
            ret_k = adv_k + vals
            adv_streams = adv_k[..., :N_ANTES].sum(-1) + adv_k[..., N_ANTES]     # normalised units: every ante has an equal voice
            ev_sh = explained_variance(vals[..., :N_ANTES].sum(-1), ret_k[..., :N_ANTES].sum(-1))
            ev_sp = explained_variance(vals[..., N_ANTES], ret_k[..., N_ANTES])
        elif a.streams:
            sh = rews - sparse_r
            norm_sh.update(sh, dones)
            norm_sp.update(sparse_r, dones)
            sh_std, sp_std = norm_sh.std, norm_sp.std
            ante_t = np.clip(np.rint(buf["g"][:, :, 0].float().cpu().numpy() * 8.0), 1, 8)      # the ante each decision was taken in
            rk = np.concatenate([by_ante(sh, ante_t) / sh_std, (sparse_r / sp_std)[..., None]], -1)
            adv_k = gae_streams(rk, vals, last_v, dones, truncs, a.gamma, a.lam)
            ret_k = adv_k + vals
            adv_streams = (adv_k[..., :N_ANTES] * fade.w).sum(-1) * sh_std + adv_k[..., N_ANTES] * sp_std
            ev_sh = explained_variance(vals[..., :N_ANTES].sum(-1), ret_k[..., :N_ANTES].sum(-1))
            ev_sp = explained_variance(vals[..., N_ANTES], ret_k[..., N_ANTES])
        lam_build = a.lam_build if a.lam_build is not None and a.lam_build != a.lam and not a.streams else None
        long_in_round = 0.0
        adv_b = np.zeros((T, N), dtype=np.float32)
        gae_b = np.zeros(N, dtype=np.float32)
        adv = np.zeros((T, N), dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)
        lam_v = a.lam_value if (a.lam_value is not None and not a.streams) else None       # --lam-value: the critic's own (longer) target horizon
        adv_v = np.zeros((T, N), dtype=np.float32)
        gae_v = np.zeros(N, dtype=np.float32)
        for t in reversed(range(0 if a.streams else T)):
            nv = last_v if t == T - 1 else vals[t + 1]
            nonterm = 1.0 - dones[t]
            if split_step:                              # an episode cut off (a scenario's end, the step limit) is bootstrapped, not a zero
                cut = (truncs[t] & (dones[t] > 0.5))
                nv = np.where(cut, vals[t], nv)
                bootstrap = np.where(cut, 1.0, nonterm)
            else:
                bootstrap = nonterm
            delta = rews[t] + a.gamma * nv * bootstrap - vals[t]
            gae = delta + a.gamma * a.lam * nonterm * gae
            adv[t] = gae
            if lam_build is not None:
                gae_b = delta + a.gamma * lam_build * nonterm * gae_b
                adv_b[t] = gae_b
            if lam_v is not None:
                gae_v = delta + a.gamma * lam_v * nonterm * gae_v
                adv_v[t] = gae_v
        if lam_build is not None:
            # --lam-build: the longer horizon for every decision outside a round (g column 5: in a round) and for
            # those inside one that changed something lasting; the plain plays and discards keep --lam
            in_round = (buf["g"][:, :, 5] > 0.5)
            use_long = (~in_round | lasting).cpu().numpy()
            long_in_round = float((in_round & lasting).float().sum() / in_round.float().sum().clamp(min=1))
            adv = np.where(use_long, adv_b, adv)
        if a.streams:
            adv, ret = adv_streams.astype(np.float32), ret_k
        else:
            ret = (adv_v if lam_v is not None else adv) + vals
        value_ev = float(explained_variance(vals.reshape(-1), ret.reshape(-1))) if not a.streams else float("nan")
        big = {k: v.reshape((T * N,) + tuple(v.shape[2:])) for k, v in buf.items()}
        if use_play:
            flat_play = torch.tensor((np.log10(1.0 + play_t) / 6.0).reshape(-1), device=dev)
            flat_play_m = torch.tensor((play_t > 0).reshape(-1), device=dev)
        flat_act = acts.reshape(-1)
        flat_lp = lps.reshape(-1)
        flat_adv = torch.tensor(adv.reshape(-1), device=dev)
        flat_ret = torch.tensor(ret.reshape((-1,) + ret.shape[2:]), device=dev)
        # --tail-q: the risk-seeking update. Only advantages above the batch's tail-q quantile count as good (their
        # excess over it, rescaled to the batch's own spread); a share tail_lam, ramped up from --tail-from to
        # --tail-to, of the update uses that instead of the plain advantages. Win rate is a tail probability: this
        # pays a choice for the games that went far beyond expectation, not for the average
        tail_lam = 0.0
        if a.tail_q > 0:
            tail_lam = min(a.tail_max, max(0.0, (it - a.tail_from) / max(1, a.tail_to - a.tail_from)))
            if tail_lam > 0:
                thr = torch.quantile(flat_adv[::4], a.tail_q)
                up = (flat_adv - thr).clamp(min=0)
                up = up * (flat_adv.std() / (up.std() + 1e-8))
                flat_adv = (1.0 - tail_lam) * flat_adv + tail_lam * up
        own_flat = own_np.reshape(-1)
        if partner_models:                                    # the advantages are normalised over this policy's own decisions
            m_own = torch.from_numpy(own_flat).to(dev)
            flat_adv = (flat_adv - flat_adv[m_own].mean()) / (flat_adv[m_own].std() + 1e-8)
        else:
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        if use_amp and a.amp_recompute:
            # an extra forward over the rollout (17% of an iteration): the old log-probs as the update's
            # batches round them. Off by default: measured, the rollout's own log-probs differ from these by
            # far less than a PPO step moves the policy
            with torch.no_grad():
                chunks = []
                for k in range(0, T * N, a.batch):
                    sl = slice(k, k + a.batch)
                    lg, _ = fwd({key: t[sl] for key, t in big.items()})
                    chunks.append(policy(lg, big["mask"][sl], {key: t[sl] for key, t in big.items()})
                                  .log_prob(flat_act[sl]))
                flat_lp = torch.cat(chunks)
        model.train()
        idx = np.flatnonzero(own_flat) if partner_models else np.arange(T * N)       # (a partner's decisions are not trained on)
        pl, vl, ents, kls = [], [], [], []
        vl_sh, vl_sp = [], []
        e_round, e_build = [], []
        ax_next, ax_final, ax_acc, ax_mae = [], [], [], []
        ax_play, ax_loo = [], []
        for _ in range(a.epochs):
            np.random.shuffle(idx)
            for k in range(0, len(idx), a.batch):
                mb = torch.from_numpy(idx[k:k + a.batch]).to(dev)
                mbo = {key: t[mb] for key, t in big.items()}
                if use_aux:
                    logits, v, x = fwd_aux(mbo)
                else:
                    logits, v = fwd(mbo)
                if use_play or use_loo:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        heads = model.aux_heads(mbo, flat_act[mb])
                dist = policy(logits, mbo["mask"], mbo)
                lp = dist.log_prob(flat_act[mb])
                ratio = torch.exp(lp - flat_lp[mb])
                s1 = ratio * flat_adv[mb]
                s2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * flat_adv[mb]
                ploss = -torch.min(s1, s2).mean()
                if a.streams:                                     # one loss per head: the shaped one (its ante components) and the sparse one
                    tgt = flat_ret[mb]
                    l_sh = F.mse_loss(v[:, :N_ANTES], tgt[:, :N_ANTES])
                    l_sp = F.mse_loss(v[:, N_ANTES], tgt[:, N_ANTES])
                    vloss = l_sh + l_sp
                    vl_sh.append(l_sh.detach()); vl_sp.append(l_sp.detach())
                else:
                    vloss = F.mse_loss(v, flat_ret[mb])
                ent_all = dist.entropy()
                ent = ent_all.mean()
                in_round = mbo["g"][:, 5] > 0.5                  # obs "g" column 5: state == SELECTING_HAND
                # the bonus counts decisions outside rounds ent_build times: per decision a round has hundreds
                # of near-equal plays to be vague about, a shop a handful of choices that decide the run
                ent_w = torch.where(in_round, torch.ones_like(ent_all), torch.full_like(ent_all, a.ent_build))
                loss = ploss + a.vf * vloss - ent_now * (ent_w * ent_all).mean()
                if ent_bump_now > 0:                                 # --streams: the exploration bump after a fade begins, on the build decisions only
                    loss = loss - ent_bump_now * ((~in_round).float() * a.ent_build * ent_all).mean()
                if use_reach:                                     # the outcome heads (on the state the forward just computed)
                    rlog = model.reach(model._s.float()).float()
                    l_reach = F.binary_cross_entropy_with_logits(rlog, flat_reach[mb])
                    loss = loss + a.reach_coef * l_reach
                    r_bce_acc.append(l_reach.detach()); r_pred_acc.append(torch.sigmoid(rlog).mean(0).detach())
                if warm:
                    loss = a.vf * vloss
                if use_aux:
                    tn, tf = flat_next[mb], flat_final[mb]
                    l_next = F.binary_cross_entropy_with_logits(x[:, 0], tn)
                    l_final = F.mse_loss(x[:, 1], tf)
                    loss = loss + a.aux_coef * (l_next + l_final)
                    ax_next.append(l_next.detach()); ax_final.append(l_final.detach())
                    ax_acc.append((((x[:, 0] > 0) == (tn > 0.5)).float().mean()).detach())
                    ax_mae.append(((x[:, 1] - tf).abs().mean() * AUX_BLINDS).detach())
                if use_play:                                  # the score of the play that was made (log10 / 6), on the steps that were plays
                    pm = flat_play_m[mb]
                    l_play = (((heads["play"].float() - flat_play[mb]) ** 2) * pm).sum() / pm.sum().clamp(min=1)
                    loss = loss + a.aux_play_coef * l_play
                    ax_play.append(l_play.detach())
                if use_loo:                                   # each held joker's leave-one-out contribution (the engine's columns the network does not read)
                    from .env import F_JOKER as _FJ, F_JOKER_W as _FJW
                    tgt = mbo["jf"][..., _FJ:_FJW].float()
                    present = (mbo["jf"][..., 0] > 0.5) & ~in_round[:, None]
                    l_loo = (((heads["loo"].float() - tgt) ** 2).sum(-1) * present).sum() / present.sum().clamp(min=1)
                    loss = loss + a.aux_loo_coef * l_loo
                    ax_loo.append(l_loo.detach())
                if demo_now > 0:                              # --demo: the imitation loss on a batch of demonstrated build decisions
                    dob, dact = demos.sample(a.demo_batch)
                    dlogits, _ = fwd(dob)
                    l_demo = F.cross_entropy(dlogits.float(), dact)
                    loss = loss + demo_now * l_demo
                    dl_.append(l_demo.detach())
                    dacc_.append((dlogits.argmax(1) == dact).float().mean().detach())
                e_round.append(ent_all[in_round].mean().detach() if in_round.any() else ent.detach())
                e_build.append(ent_all[~in_round].mean().detach() if (~in_round).any() else ent.detach())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                pl.append(ploss.detach()); vl.append(vloss.detach()); ents.append(ent.detach())
                kls.append((flat_lp[mb] - lp).mean().detach())
        vl_extra = []
        for _ in range(a.value_epochs if not a.streams else 0):          # --value-epochs: more passes for the critic alone (it learns from noisy returns)
            np.random.shuffle(idx)
            for k in range(0, len(idx), a.batch):
                mb = torch.from_numpy(idx[k:k + a.batch]).to(dev)
                mbo = {key: t[mb] for key, t in big.items()}
                _, v = fwd(mbo)
                vloss_x = F.mse_loss(v, flat_ret[mb])
                opt.zero_grad()
                (a.vf * vloss_x).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                vl_extra.append(vloss_x.detach())
        pl, vl, ents, kls = (torch.stack(x).double().cpu().numpy() for x in (pl, vl, ents, kls))
        if cuda:
            # the legal-action path makes tensors of a different size every call, and the allocator's cache of
            # them grows until the card is full (then Windows pages GPU memory and everything crawls)
            torch.cuda.empty_cache()
        # --ladder: look at the stake ahead; when its win signal is dense enough on enough previews in a row, train on it
        advanced = None
        if ladder:
            ladder_since += 1
            if ladder_i + 1 < len(ladder) and it % a.ladder_every == 0:
                nxt = ladder[ladder_i + 1]
                prev = fixed_eval(nxt, a.ladder_games)
                ladder_preview = prev["win_rate"]
                ladder_hits = ladder_hits + 1 if (ladder_preview >= a.ladder_dense and ladder_since >= a.ladder_min_iters) else 0
                print(json.dumps({"ladder_preview": nxt, "win_rate": ladder_preview, "blinds": prev["blinds"], "hits": ladder_hits,
                                  "since": ladder_since, "iter": it}), flush=True)
                if ladder_hits >= a.ladder_confirm:
                    ladder_i += 1
                    advanced = (ladder[ladder_i - 1], ladder[ladder_i], ladder_preview)
                    apply_ladder(ladder_i)
                    ladder_since, ladder_hits = 0, 0
                    ladder_bump.trigger()                                   # a harder stake: try other builds for a while
                    stall.reset()                                           # (the win rate drops by itself on a harder stake)
                    stats = EpisodeStats(n=a.stats_window)                # (the logged window is the new stake's from here)
                    cur_blinds, cur_games = 0.0, 0
                    print(json.dumps({"ladder": "advance", "from": advanced[0], "to": advanced[1], "preview_win": advanced[2], "iter": it}), flush=True)
        eval_res = fixed_eval() if (a.eval_every > 0 and (it % a.eval_every == 0 or it == it0 + 1)) else None
        el = time.time() - t0
        row = {"iter": it, "pi": round(float(np.mean(pl)), 4), "v": round(float(np.mean(vl)), 3),
               "ent": round(float(np.mean(ents)), 3), "kl": round(float(np.mean(kls)), 4),
               "min": round(el / 60, 1), "phase": a.phase, "env_steps": it * T * N,
               "sps": round(it * T * N / max(el, 1e-9)),
               "sim%": round(100 * sim_s / max(time.time() - t0 - 60.0 * min0, 1e-9), 1),
               "games": games, "wins": wins, "antes": ante_hist[1:].tolist(),   # antes: lost in 1..8, then won
               "wins_other": dict(wins_other),             # wins on the other stakes of the mix, in total
               **({"level_cap_share": round(lc_share, 3)} if a.level_cap_share > 0 else {}),
               **({"xscale_share": round(xs_share, 3), "xscale_val": round(xs_val, 3)} if a.start_xscale > 0 else {}),
               **({"n_hard": int(target.sum()), "hard_window": round(cur_blinds / max(1, cur_games), 3),
                   "hard_prev": None if cur_prev is None else round(cur_prev, 3)} if a.stake_curriculum else {}),
               "games_by_stake": dict(games_by),           # (both since the last start of the process)
               "lr": lr_now, "ent_coef": ent_now,
               "ent_round": round(float(torch.stack(e_round).mean()), 3),
               "ent_build": round(float(torch.stack(e_build).mean()), 3),
               "start_any": round(any_now, 3), "start_xmult": round(xm_now, 3), "menu": max(menu_prev, 0), "final_boss": final_boss,
               **({"scn_p": round(scn_p_now, 3), "scn_mix": [round(x, 2) for x in scn_mix_now],
                   "scn": {n: dict(zip(("games", "cut", "lost", "won"), [int(x) for x in scn_stat[k]]))
                           for k, n in enumerate(("one_blind", "few_blinds", "full_run", "archive"))}} if scen_on else {}),
               **({"archive_range": [lo_a, hi_a], "archive_lo": band_lo, "archive_out": last_out, **({"archive_p_now": round(float(arch_p_now), 3)} if a.archive_ante == "reverse" else {}), "archive_band_wr": None if band_wr is None else round(band_wr, 3), "archive_starts": [int(x) for x in venv.archive_starts(True)]} if (a.archive_p > 0 or a.archive_store > 0) else {}),
               **({"archive_fill": [int(x) for x in venv.archive_counts(-1)]} if (a.archive_p > 0 or a.archive_store > 0) else {}),
               **({"pop_partners": len(partner_models), "pop_own_share": round(float(own_np.mean()), 3)} if partner_paths else {}),
               **({"auto_share": round(auto_n / max(1, sel_n), 4), "blind_selects": sel_n} if a.autoplay else {}),
               **({"clear_rate": [round(float(x) / max(1, clear_games), 4) for x in clear_acc], "clear_games": clear_games} if a.immortal else {}),
               **({"ante_ema": [round(float(x), 4) for x in ante_ema], "ante_mult": [round(float(x), 3) for x in ante_mult]} if (a.immortal and a.immortal_fade) else {}),
               "buy_share": [[round(float(x), 3) for x in row[:5] / row[5].clamp(min=1)] + [int(row[5])] for row in buy_acc.cpu()],
               **({"aux_play": round(float(torch.stack(ax_play).mean()), 5)} if ax_play else {}),
               **({"aux_loo": round(float(torch.stack(ax_loo).mean()), 5)} if ax_loo else {}),
               "start_money_p": round(money_p, 3), "start_money2_p": round(money2_p, 4),
               **({"money_extra": round(extra, 2), "rich_win": round(rich_rate, 4),      # the money curriculum; the
                   "normal_games": normal_games, "normal_wins": normal_wins}             # target stake from the
                  if a.money_target > 0 else {}),                                        # normal start
               **({"aux_next": round(float(torch.stack(ax_next).mean()), 4),          # prediction heads: loss of
                   "aux_next_acc": round(float(torch.stack(ax_acc).mean()), 3),       # "this blind is beaten" and
                   "aux_next_base": round(float(max(aux_next.mean(), 1 - aux_next.mean())), 3),   # its accuracy
                   "aux_final": round(float(torch.stack(ax_final).mean()), 4),        # (base: always the commoner
                   "aux_final_mae": round(float(torch.stack(ax_mae).mean()), 3)}      # answer); final blinds: the
                  if use_aux else {}),                                                # loss and the error in blinds
               "bank_coef": round(bank_now, 4), "bank": round(25 * bank_sum / T, 2),
               "oracle": ({"w_build": round(orac_w1, 3), "w_money": round(orac_w3, 3), "clear": round(float(orac_acc[0] / max(1, orac_n[0])), 4),
                           "logshare": round(float(orac_acc[1] / max(1, orac_n[0])), 4), "pfind": round(float(orac_acc[2] / max(1, orac_n[0])), 4),
                           "states": int(orac_n[0]), "calls": int(orac_calls[0])} if orac_on else None),   # $ held (capped at 25), mean
               "x_offer": x_offer, "x_take": x_take,       # decisions with a multiplier joker to take; taken
               "e_offer": int(e_offer), "e_take": int(e_take),   # ... with an exposed item to take (or Luchador
               "ret_std": round(ret_std, 3), "lam": a.lam, "lam_build": a.lam_build if a.lam_build is not None else a.lam,
               "long_in_round": round(long_in_round, 3),   # share of in-round decisions on the longer horizon                                     #     to sell on a boss); taken
               "build_score": round(bs_sum / T, 3), "build_score_coef": a.build_score_bonus,
               "grow_coef": round(grow_now, 4), "start_grow": round(sg_now, 3),
               **({"school_share": round(school_share_now, 3), "school_bonus": round(school_now, 3),
                   # finished school games by student: [games, the multiplier its counter ended on, geometric mean]
                   "school": {_skeys[k]: [int(n), round(float(np.exp(sl / n)), 2)]
                              for k, (n, sl) in enumerate(venv.school_stats()) if n > 0}}
                  if school_students else {}),
               "growth": round(float(growth_sum) / T, 3),  # growth(): mean over decisions
               "income_coef": round(income_now, 4),        # income: dollars earned per blind beaten, mean
               "income": round(float(inc.sum() / max(1.0, paid.sum())), 2),
               "demo_coef": round(demo_now, 4), "demo_loss": round(float(torch.stack(dl_).mean()), 4) if dl_ else None,
               "demo_acc": round(float(torch.stack(dacc_).mean()), 3) if dacc_ else None,
               "strength_coef": round(strength_now, 4), "phi8_mean": round(float(phi8_last.mean()), 4), "value_ev": round(value_ev, 4),
               **({"reach_bce": round(float(torch.stack(r_bce_acc).mean()), 4), "reach_pred": [round(float(x), 4) for x in torch.stack(r_pred_acc).mean(0)],
                   "reach_rate": [round(float(x), 4) for x in flat_reach.mean(0)]} if use_reach and r_bce_acc else {}),
               **({"vloss_extra": round(float(torch.stack(vl_extra).mean()), 4)} if vl_extra else {}),
               **({"rental_debt_pot": round(float(mpot_last.mean()), 4)} if money_pot_on else {}),
               **({"curve_surv": round(curve_stat[0] / max(1, curve_stat[2]), 3), "curve_behind": round(curve_stat[1] / max(1, curve_stat[2]), 3),
                   "curve_growth": round(curve_gsum[0] / max(1, curve_stat[2]), 3), "grow_reward_paid": round(float(grow_rew.sum()) / max(1, N), 4)} if curve_on else {}),
               **({"proj8_mean": round(float(proj8_last.mean()), 4), "proj8_log": round(proj8_sum[0] / max(1, proj8_sum[1]), 3)} if a.proj8 > 0 else {}),    # strength: what the build can score at a round's start
               "strength": round(float(strengths.sum() / rounds_started.clamp(min=1)), 3),   # mean (0..1)
               "power_coef": round(power_now, 4),          # power: how strongly blinds were beaten, mean (0..1)
               "power": round(float(pw.sum() / max(1.0, beaten.sum())), 3),
               "build_coef": round(build_now, 4),          # holds: share of all decisions with a working
               "holds": [round(float(x), 3) for x in role_sum / T],   # multiplier / +mult / chips / scaling joker; 2+ multipliers
               "kinds": {k: int(v) for k, v in zip(kind_names, kinds.cpu().tolist()) if v > 0},
               **({"tail_lam": round(tail_lam, 3)} if a.tail_q > 0 else {}),
               **({"ban_share": round(ban_now, 3), "ban_ids": ban_ids,      # the shifting world: the ban list,
                   "ban_names": [joker_keys[j] for j in ban_ids[:10]],      # share of joker purchases off it,
                   "off_list": round(float(1.0 - buy_counts[jvocab][ban_ids].sum() / buy_counts[jvocab].sum().clamp(min=1)), 3) if ban_ids else 1.0,
                   "ban_games": ban_games,                                  # and the banned games' blinds
                   "ban_blinds": round(ban_blinds / max(1, ban_games), 2), # against the normal games'
                   "normal_blinds": round(norm_blinds / max(1, norm_games), 2),
                   "tilt_share": round(tilt_now, 3)} if a.ban_top > 0 or a.tilt_share > 0 else {}),
               "gpu_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1) if cuda else 0.0,
               "gpu_used_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if cuda else 0.0}
        if eval_res is not None:
            row["eval"] = eval_res
            if a.eval_lookahead > 0:                      # the same seeds with the one-step lookahead outside rounds
                row["eval_la"] = fixed_eval(n=a.eval_la_games or a.eval_games, la_k=a.eval_lookahead)
            row["eval_stake"] = a.stake
        if ladder:
            row.update({"ladder_stake": a.stake, "ladder_next": ladder[ladder_i + 1] if ladder_i + 1 < len(ladder) else None,
                        "ladder_preview": ladder_preview, "ladder_hits": ladder_hits, "ladder_since": ladder_since,
                        "ladder_dense": a.ladder_dense, "ladder_bump": round(ladder_bump.value(), 5), "ladder_bump_left": ladder_bump.left,
                        **({"ladder_advanced": list(advanced)} if advanced else {})})
        if a.streams:
            nan = lambda x: None if x != x else round(float(x), 3)
            row.update({"streams": True, "fade_w": [round(float(x), 4) for x in fade.w],        # each ante's shaping weight
                        "reach": [round(float(x), 3) for x in reach.rates()],                   # the share of recent games past each ante
                        "reach_n": len(reach.furthest), "fading": [int(x) for x in fade.fading],
                        "ent_bump": round(ent_bump.value(), 5), "ent_bump_left": ent_bump.left,
                        "v_shaped": round(float(torch.stack(vl_sh).mean()), 4), "v_sparse": round(float(torch.stack(vl_sp).mean()), 4),
                        "ev_shaped": nan(ev_sh), "ev_sparse": nan(ev_sp),                       # explained variance of the two heads
                        "ret_norm": {"shaped_std": round(sh_std, 3), "sparse_std": round(sp_std, 3),
                                     "shaped_mean": round(norm_sh.stat.mean, 3), "sparse_mean": round(norm_sp.stat.mean, 3)},
                        "sparse_rew": round(float(sparse_r.sum() / max(1, dones.sum())), 3),     # sparse reward per finished game
                        "truncs": int(truncs.sum())})
        ante_hist[:] = 0
        row.update(stats.summary())
        if a.stall_ent and row.get("win%") is not None:
            level, cut = stall.update(float(row["win%"]), stall_bump.left > 0)
            if cut:                                                   # the win rate fell after a bump: end it early
                stall_bump.left = 0
                row["stall_cut"] = True
                print(json.dumps({"stall": "cut", "iter": it, "win%": row["win%"]}), flush=True)
            if level is not None:                                     # a stall: explore on the build decisions
                stall_bump.peak = level
                stall_bump.trigger()
                row["stall_fired"] = round(level, 5)
                print(json.dumps({"stall": "fired", "iter": it, "level": level, "k": stall.k, "win%": row["win%"]}), flush=True)
            row.update({"stall_k": stall.k, "stall_bump": round(stall_bump.value(), 5), "stall_bump_left": stall_bump.left})
        if vloop is not None and it % a.vloop_every == 0:
            row.update(vloop.fit())
        log(a.log, row)
        if it % a.save_every == 0 or it == a.iters:
            save_model(model, a.out, {"mode": "ppo", "iter": it, "engine": "cpp"})
            s_ = stats.summary().get("blinds", -1)
            if s_ > best:
                best = s_
                save_model(model, a.out.replace(".pt", "_best.pt"), {"mode": "ppo", "iter": it, "blinds": s_})
    venv.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["bc", "ppo"])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=64, help="env steps per env per iteration")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--buffer", type=int, default=6000, help="BC replay size")
    p.add_argument("--updates", type=int, default=6, help="BC gradient steps per iteration (0 = full epochs)")
    p.add_argument("--beta-frac", type=float, default=0.5, help="BC: fraction of iters to anneal expert mixing")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.002)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--reach-coef", type=float, default=0.0, help="the outcome heads (model.add_reach_head: P(reach ante 5..8), P(win)) trained with this weight "
                   "(binary cross-entropy on each game's outcome, bootstrapped from the heads at the rollout's end); 0: off")
    p.add_argument("--value-epochs", type=int, default=0, help="extra passes per update training the critic alone (after the PPO epochs)")
    p.add_argument("--lam-value", type=float, default=None, help="the GAE lambda of the critic's target (1.0: Monte Carlo returns); the policy's advantages keep "
                   "--lam / --lam-build")
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--init", default=None, help="checkpoint to start from")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--log", default=None)
    p.add_argument("--pipeline", action="store_true",
                   help="PPO: overlap simulation and GPU work with two groups of envs "
                        "(use about twice your core count for --envs)")
    p.add_argument("--ante-weight", type=float, default=1.0,
                   help="PPO reward: an ante-8 blind is worth this many times an ante-1 blind, rising linearly "
                        "(the 24 blinds still add up to 24)")
    p.add_argument("--win-bonus", type=float, default=10.0, help="PPO reward for winning the run")
    p.add_argument("--shape-chips", type=float, default=0.0,
                   help="PPO potential-based shaping: reward for progress through a new blind, as a fraction of "
                        "that blind's weight (e.g. 0.5). Paid back when the blind ends, so it only speeds learning")
    p.add_argument("--shape-phi", type=float, default=0.0,
                   help="PPO potential-based shaping: weight of the build potential (rewards/potential.py: "
                        "headroom against the next boss + blinds beaten / 24), e.g. 1.0")
    p.add_argument("--reward-config", default=None, help="YAML/JSON reward config (rewards/config.py)")
    p.add_argument("--strategic", action="store_true",
                   help="PPO only makes strategic decisions; a tactical network plays the cards (see strategic.py)")
    p.add_argument("--tactical", default=None, help="--strategic: checkpoint that plays the cards (default: --init)")
    p.add_argument("--margin", type=float, default=0.2,
                   help="--strategic: bonus x how comfortably a blind was cleared (score/target - 1, capped at 1)")
    p.add_argument("--win-ante", type=int, default=8,
                   help="end runs as won after this ante's boss (cheap experiments); 8 is the real game")
    p.add_argument("--joker-pool-file", default=None,
                   help="JSON list of joker keys that shops and packs may offer (cheap experiments)")
    p.add_argument("--phase", default=None, help="label written to every log row (e.g. a curriculum stage)")
    p.add_argument("--amp", action="store_true",
                   help="PPO, experimental: run the network in bfloat16 mixed precision (~12%% faster on an RTX 4080, "
                        "but in a short test updates moved the policy ~2x further and entropy rose faster)")
    p.add_argument("--engine", choices=["py", "cpp"], default="py",
                   help="PPO: cpp runs the environments on the C++ engine (same game, observations and rewards; "
                        "many times faster; use hundreds of --envs)")
    p.add_argument("--threads", type=int, default=0, help="--engine cpp: simulation threads (0: one per core)")
    p.add_argument("--stats-window", type=int, default=5000, help="--engine cpp: games the logged averages cover")
    p.add_argument("--lr-final", type=float, default=None, help="--engine cpp: anneal the learning rate linearly to this")
    p.add_argument("--lr-steps", type=lambda x: [int(v) for v in x.split(",") if v], default=[],
                   help="--engine cpp: step schedule instead of the linear one, e.g. 400,1200: the rate is cut by --lr-step-factor after each of these updates")
    p.add_argument("--lr-step-factor", type=float, default=0.1, help="--lr-steps: the cut at each step")
    p.add_argument("--ent-final", type=float, default=None, help="--engine cpp: anneal the entropy bonus linearly to this")
    p.add_argument("--explore-types", type=float, default=0.0,
                   help="--engine cpp: share of the behaviour policy spread evenly over the legal kinds of action")
    p.add_argument("--start-grow", type=float, default=0.0,
                   help="--engine cpp: share of training games that start holding one random growing joker, eternal")
    p.add_argument("--start-grow-final", type=float, default=None)
    p.add_argument("--grow-bonus", type=float, default=0.0,
                   help="--engine cpp: reward for how far the growing jokers held have grown (their counters, log "
                        "scale): paid as they rise, taken back as they fall, kept at the end of the game")
    p.add_argument("--grow-bonus-final", type=float, default=None)
    p.add_argument("--income-bonus", type=float, default=0.0,
                   help="--engine cpp: reward per dollar earned at the end of a round (when a play beats a blind)")
    p.add_argument("--income-bonus-final", type=float, default=None,
                   help="--engine cpp: --income-bonus moves linearly to this over the run")
    p.add_argument("--income-cap", type=float, default=30.0,
                   help="--engine cpp: dollars of a round's earnings that --income-bonus counts at most")
    p.add_argument("--fixed-seed", type=int, default=-1, help="--engine cpp: every game (training and evaluation) is this one seed (with --exact: the real game's streams)")
    p.add_argument("--shop-only", action="store_true", help="--engine cpp: the shop-only game: blinds before the final one are won without play (scaling counters advance), the final boss is played by the exact round solver (training and evaluation)")
    p.add_argument("--shop-only-nodes", type=int, default=1000)
    p.add_argument("--partial-weight", type=float, default=0.5, help="--engine cpp: part credit of a lost blind = this x its weight x the share reached (0.5: the original)")
    p.add_argument("--no-stickers", action="store_true", help="--engine cpp: no eternal / perishable / rental jokers at any stake (training and evaluation)")
    p.add_argument("--small-blind-pay", action="store_true", help="--engine cpp: the small blind pays its $3 at every stake (training and evaluation)")
    p.add_argument("--demo", default="", help="--engine cpp: a demonstrations file (tools/make_demos.py): an imitation loss on its decisions is added to PPO's")
    p.add_argument("--demo-coef", type=float, default=0.5, help="--demo: weight of the imitation (cross-entropy) loss")
    p.add_argument("--demo-coef-final", type=float, default=None, help="--demo: the weight moves linearly to this over --demo-fade-iters")
    p.add_argument("--demo-fade-iters", type=int, default=2000)
    p.add_argument("--demo-batch", type=int, default=256)
    p.add_argument("--demo-won-weight", type=float, default=1.0, help="--demo: sampling weight of decisions from won lines (others 1)")
    p.add_argument("--demo-pretrain", type=int, default=0, help="--demo: imitation-only updates before PPO starts")
    p.add_argument("--phi8", type=float, default=0.0,
                   help="--engine cpp: potential-based shaping toward the ante-8 boss: phi = this x clamp((log10(best single play score) - log10(ante-8 boss target) + span) / span, 0, 1), "
                        "measured at each round's first decision; the change is paid when it is measured and what is held is taken back when the game ends (policy-invariant)")
    p.add_argument("--proj8", type=float, default=0.0, help="potential on the build projected to the ante-8 boss (VecEnv.projected_power: every growing joker moved by "
                   "its natural rate over the rounds left, the boss faced with the exact scorer on redrawn hands; no solver), measured at each blind select: "
                   "phi = this x clamp((log10(share of the target) + span) / span, 0, 1); taken back when the game ends. Works with --autoplay-force (unlike --phi8)")
    p.add_argument("--proj8-span", type=float, default=3.0)
    p.add_argument("--rental-penalty", type=float, default=0.0, help="potential: -this per rental joker held (a rental costs $3 a round after its $1 price); "
                   "charged when bought, given back when sold or the game ends")
    p.add_argument("--debt-penalty", type=float, default=0.0, help="potential: -this per dollar of debt")
    p.add_argument("--surv-pot", type=float, default=0.0, help="potential: this x the expected reward of every blind still to come if the build stopped changing "
                   "(the clear-chance curve: the build projected to each ante's boss, P(clear) fitted on solver clears), measured at blind selects")
    p.add_argument("--grow-pen", type=float, default=0.0, help="potential: -this x the decades the build falls behind the targets over the next two antes (0..2)")
    p.add_argument("--grow-reward", type=float, default=0.0, help="a reward (not a potential: paid, never refunded) at each blind select: this x clamp(growth + 1, 0, 3), "
                   "growth = log10 of the build's projected share two antes ahead over its share now (a flat build ~ -1 -> 0; keeping pace -> this; ahead -> up to 3x this)")
    p.add_argument("--curve-samples", type=int, default=2, help="--surv-pot / --grow-pen: redrawn hands per projected boss")
    p.add_argument("--proj8-samples", type=int, default=4)
    p.add_argument("--phi8-span", type=float, default=3.0, help="--phi8: the window in decades below the ante-8 boss target over which phi rises from 0 to 1")
    p.add_argument("--strength-bonus", type=float, default=0.0,
                   help="--engine cpp: reward at the first play / discard of each round, times the build's strength "
                        "(0..1: the best single play's score as a multiple of the target, log scale)")
    p.add_argument("--reroll-cap", type=int, default=0, help="rerolls allowed per shop visit (0: the default, 8)")
    p.add_argument("--flash-cap", type=int, default=0, help="Flash Card grows from only the first N rerolls of a game (0: every reroll)")
    p.add_argument("--hand-level-cap", type=int, default=0, help="the highest level a planet or Black Hole takes a hand to (0: no limit)")
    p.add_argument("--win-ratio", type=float, default=0.0,
                   help="the win bonus must exceed this many times the most shaping reward one episode can pay (shaping_budget), else every shaping term "
                        "is scaled down until it does (0: not checked); e.g. 10")
    p.add_argument("--immortal", action="store_true",
                   help="--engine cpp: a blind that is not beaten does not end the run: no pay, the next blind and its shop follow, so the agent builds through every ante and "
                        "meets the late game with what it made; only the ante-8 boss ends it. The reward is each blind actually beaten (once) and the win; the logged 'blinds' is the number beaten")
    p.add_argument("--partners", default="",
                   help="a population: comma list of the checkpoints of the other members (each trained by its own run of this trainer). A game is played by a changing team: at a blind select "
                        "control may pass (--pop-handoff) to a random member of {this policy, the partners}; this policy learns only from its own decisions (the others' are masked out of "
                        "the loss), with advantages from its own value function over the whole game, so it must play well from the states and builds the others produce")
    p.add_argument("--pop-handoff", type=float, default=0.25, help="chance that control is handed to a random member at a blind select (a new game always draws one)")
    p.add_argument("--pop-reload", type=int, default=10, help="updates between reloads of the partners' checkpoints")
    p.add_argument("--autoplay", action="store_true",
                   help="--engine cpp: at a blind select the action 'leave shop' (never legal there otherwise) means 'select it and let the exact round solver play it' (the network keeps "
                        "the final say: the ordinary select plays by hand). The solver sees the draw order: a training aid, not the real game; the evaluation never has it")
    p.add_argument("--autoplay-nodes", type=int, default=400)
    p.add_argument("--autoplay-force", action="store_true", help="--autoplay: every round is played by the exact solver (the ordinary select hands it over too), in "
                   "training and in the evaluation: the network makes only the decisions outside rounds")
    p.add_argument("--solver-smart", action="store_true", help="--autoplay: the round solver tries set-up discards first where the greedy line cannot clear (ordering only)")
    p.add_argument("--solver-lite", action="store_true", help="--autoplay: the round solver's cheap steps")
    p.add_argument("--solver-tt", action="store_true", help="--autoplay: the round solver skips positions already reached by another order of moves")
    p.add_argument("--solver-tarots", action="store_true", help="the round solver (autoplay) also tries held consumables that need target cards (Chariot, "
                   "Death, Strength ...) mid-round, on its --solver-targets best target sets")
    p.add_argument("--solver-targets", type=int, default=3)
    p.add_argument("--vloop-every", type=int, default=0, help="the V(s) loop (vloop.py): every this many updates the engine's value network is fitted to the critic's "
                   "V(s) on recent states outside rounds and loaded, so the solver keeps lines by money + --round-eval-w x V(after the round) (0: off)")
    p.add_argument("--vloop-steps", type=int, default=300)
    p.add_argument("--vloop-per-step", type=int, default=96, help="states sampled per rollout step")
    p.add_argument("--vloop-buffer", type=int, default=200_000)
    p.add_argument("--vloop-init", default="", help="a balatro_rl/nnue.py checkpoint to start the engine's network from (default: a new one)")
    p.add_argument("--round-nnue", default="", help="an exported value network (balatro_rl/nnue.py export) for the solver's round objective (VecEnv.load_nnue)")
    p.add_argument("--round-eval-w", type=float, default=0.0, help="--round-nnue: the solver keeps a clearing line by its money + this x the network's expected blinds after it")
    p.add_argument("--d", type=int, default=256, help="a new network: width of the state vector (the value and the heads read it)")
    p.add_argument("--tok", type=int, default=128, help="a new network: width of the attention tokens")
    p.add_argument("--aw", type=int, default=128, help="a new network: width of the action path (each move's row and the head that scores it)")
    p.add_argument("--vw", type=int, default=128, help="a new network: hidden width of the value head")
    p.add_argument("--res", type=int, default=0, help="a new network: residual blocks (pre-norm MLP, model.ResBlock) on the state vector")
    p.add_argument("--emb", type=int, default=32, help="a new network: width of the item embeddings")
    p.add_argument("--heads", type=int, default=4, help="a new network: attention heads")
    p.add_argument("--immortal-pay", action="store_true",
                   help="--immortal: a failed blind still pays its cash-out (reward money, interest, joker income, a rental's fee), so the economy of the late antes stays normal; "
                        "it is not counted as beaten and earns no reward")
    p.add_argument("--stream-equal", action="store_true",
                   help="--streams with --immortal: every ante's reward has its own value head AND its own return normalisation, and the policy's advantage is the sum of the "
                        "normalised advantages (each ante has an equal voice; one whose return hardly varies, because it is mastered or never reached, is quiet); no "
                        "reach-based fade of the components")
    p.add_argument("--immortal-fade", action="store_true",
                   help="--immortal: an ante's blinds pay less as it is mastered: the multiplier of its reward falls linearly from 1 (clear rate <= --fade-start) to --fade-floor "
                        "(>= --fade-end) and rises again if the rate drops, so the pay moves to the antes not yet mastered and an early build that clears easily is not what is paid for")
    p.add_argument("--fade-start", type=float, default=0.5)
    p.add_argument("--fade-end", type=float, default=0.85)
    p.add_argument("--fade-floor", type=float, default=0.1)
    p.add_argument("--fade-ema", type=float, default=0.8, help="the clear rates are smoothed over updates: rate = ema * rate + (1 - ema) * this update's")
    p.add_argument("--weight-geom", type=float, default=0.0,
                   help="the blind rewards rise by this factor every ante (1, r, r^2 ... ; 1.5-2 makes one deep run outweigh many mid ones), scaled to add up to 24; "
                        "overrides --ante-weight / --weight-power (0: those)")
    p.add_argument("--ret-norm", action="store_true",
                   help="single-stream path: rewards are divided by the running std of the discounted return, so the value loss does not depend on the reward's units")
    p.add_argument("--archive-p", type=float, default=0.0,
                   help="--engine cpp: share of games that start from an archived state (the start of a blind of --archive-ante, from the policy's own earlier games)")
    p.add_argument("--archive-winners", action="store_true", help="the archive keeps the blind starts only of games that went on to win")
    p.add_argument("--archive-seed-ckpt", default="", help="before training, fill the archive with the blind starts of games this network wins (greedy, from --archive-seed-money)")
    p.add_argument("--archive-seed-money", default="250,500,1000", help="the extra starting money of the seeding games, one pass each")
    p.add_argument("--archive-seed-games", type=int, default=300, help="games per seeding pass")
    p.add_argument("--archive-seed-ante", type=int, default=6,
                   help="the seeding games keep only blind starts from this ante on: the earlier antes of the archive come from the policy's own normal-rule games, "
                        "so the band's extension is tested on the states the policy actually reaches")
    p.add_argument("--archive-advance", type=float, default=0.3, help="--archive-ante reverse: extend the band back one ante when this share of the games started in its earliest ante are won")
    p.add_argument("--archive-advance-games", type=int, default=200, help="... over at least this many games")
    p.add_argument("--archive-ante", default="4-7",
                   help="the antes games may start from, lo-hi; 'auto': from one ante below to two above the ante the policy's games typically reach (re-set every update); "
                        "'reverse': from ante 8 back: the band [lo, 8] starts at lo = 8 and lo drops one ante whenever --archive-advance of the games started in ante lo are won")
    p.add_argument("--archive-store", type=float, default=0.0, help="chance that the start of a blind in those antes is offered to the archive")
    p.add_argument("--archive-cap", type=int, default=64, help="snapshots kept per stake and blind (a reservoir)")
    p.add_argument("--plant-file", default="", help="planted builds (tools/plant_builds.py) put into the archive before training, at every blind of their ante")
    p.add_argument("--plant-per-ante", type=int, default=100000, help="--plant-file: at most this many builds per ante (the strongest first)")
    p.add_argument("--archive-stage-max", type=int, default=0, help="--archive-ante reverse: move the start back one ante after this many updates even if "
                   "--archive-advance is not reached (0: never)")
    p.add_argument("--archive-p-final", type=float, default=None, help="--archive-ante reverse: once the start has reached ante 1, --archive-p moves "
                   "linearly to this over --archive-fade-iters (the planted help fades; the rest of the games start from nothing)")
    p.add_argument("--archive-fade-iters", type=int, default=2000)
    p.add_argument("--archive-min", type=int, default=8, help="a blind is started from only once it holds this many snapshots")
    p.add_argument("--shape-relative", action="store_true",
                   help="every shaping term is in units of the reward of the blind being faced (the strength, income and power bonuses times it; the bank, "
                        "build and growth potentials as W(s) x quantity, which still telescopes to zero; the Verdant Leaf bonus too)")
    p.add_argument("--shape-fade-iters", type=int, default=0,
                   help="the shaping terms and head starts that fade (--*-final) do so over this many updates instead of over --iters (0: over --iters)")
    p.add_argument("--strength-ahead", type=int, default=0,
                   help="--strength-bonus: measure the build against the blind targets this many antes later (the same blind, up to ante 8) instead of "
                        "its own ante's, from 1/30 of that target (0) to the target (1); 0: the ante's own target")
    p.add_argument("--strength-bonus-final", type=float, default=None,
                   help="--engine cpp: --strength-bonus moves linearly to this over the run")
    p.add_argument("--power-bonus", type=float, default=0.0,
                   help="--engine cpp: reward when a play beats a blind, times how strongly it was beaten (0..1: "
                        "the score as a multiple of the target, and the hands left unused)")
    p.add_argument("--power-bonus-final", type=float, default=None,
                   help="--engine cpp: --power-bonus moves linearly to this over the run")
    p.add_argument("--marginal", action="store_true",
                   help="--engine cpp: a new network also reads the marginal-value inputs (marginal.py): the build's expected "
                        "score, what each held joker and each joker / planet on offer is worth to it. A resumed or --init "
                        "network keeps the setting it was made with")
    p.add_argument("--school-share", type=float, default=0.0,
                   help="--engine cpp: the joker school (0: off): this share of the games gives one growing joker (the "
                        "student, --school-students) as an eternal, with a kit, and a shelf item for its chain in every "
                        "shop; the growth of its counter is rewarded (--school-bonus per unit of log multiplier)")
    p.add_argument("--school-share-final", type=float, default=None)
    p.add_argument("--school-students", default="vampire,lucky_cat,glass,hologram,constellation")
    p.add_argument("--school-bonus", type=float, default=3.0)
    p.add_argument("--school-bonus-final", type=float, default=None)
    p.add_argument("--school-fade-iters", type=int, default=3000,
                   help="iterations over which the school's share and bonus move to their -final values")
    p.add_argument("--build-score-bonus", type=float, default=0.0,
                   help="--engine cpp with --marginal/--potential: reward this * the change in the build's expected "
                        "best-play score in decades (log10), read outside rounds, capped at --build-score-cap")
    p.add_argument("--build-score-cap", type=float, default=5.0,
                   help="decades of per-hand score beyond which more is not rewarded (the ante-8 boss needs ~5)")
    p.add_argument("--death-money-penalty", type=float, default=0.0,
                   help="--engine cpp: a game that ends in a loss costs this * min(money, --death-money-cap) / cap")
    p.add_argument("--death-money-cap", type=float, default=100.0)
    p.add_argument("--exact", action="store_true",
                   help="--engine cpp: play the real game's named random streams (shop, packs, bosses, tags, deck shuffles and rolls)")
    p.add_argument("--priv", action="store_true",
                   help="--engine cpp --exact: an asymmetric critic. The value head also reads the privileged observation (the next shop, the "
                        "next ante's boss, the top of the deck); the policy does not, so the network that results needs none of it")
    p.add_argument("--streams", action="store_true",
                   help="--engine cpp: two reward streams (streams.py). Shaped (per-blind rewards and every shaping bonus, by ante) and sparse (the win, the "
                        "ante-8 bonus, blinds from --sparse-from-ante) with a value output each and their own return normalisers; the policy's advantage is "
                        "sum_k w_k A_shaped_k + A_sparse, and w_k fades per milestone (--stream-trigger ...)")
    p.add_argument("--win-only", action="store_true",
                   help="--streams: the only reward is the win bonus (--stream-win-bonus): the shaped stream's weights stay at 0 and no blind reward is paid "
                        "into the sparse stream. For a network that already wins often (a dense signal): fine-tune it from --init, then raise the stake")
    p.add_argument("--stream-win-bonus", type=float, default=50.0, help="--streams: the terminal reward for winning (instead of --win-bonus)")
    p.add_argument("--stream-ante-weight", type=float, default=3.0,
                   help="--streams: the per-blind reward rises from 1 at ante 1 to this at ante 8 (instead of --ante-weight; --weight-power shapes it)")
    p.add_argument("--sparse-from-ante", type=int, default=7, help="--streams: the blind rewards from this ante on are in the sparse stream")
    p.add_argument("--stream-trigger", type=float, default=0.25, help="--streams: a shaping weight starts to fade when this share of the recent episodes got past its ante")
    p.add_argument("--stream-fade-iters", type=int, default=200, help="--streams: updates a weight takes to fade from 1 to --stream-floor (linearly)")
    p.add_argument("--stream-window", type=int, default=500, help="--streams: finished episodes the reach shares are taken over")
    p.add_argument("--stream-reverse", type=float, default=0.5, help="--streams: a faded weight rises again when its reach share falls under trigger x this")
    p.add_argument("--stream-floor", type=float, default=0.0, help="--streams: the lowest a shaping weight fades to")
    p.add_argument("--fade-ent-bump", type=float, default=0.004,
                   help="--streams: extra entropy coefficient when a fade begins, on decisions outside rounds only (shop, packs, blind select; "
                        "x --ent-build as the base one is) ...")
    p.add_argument("--fade-ent-decay", type=int, default=150, help="--streams: ... decaying linearly back to nothing over this many updates")
    p.add_argument("--ret-norm-floor", type=float, default=0.5, help="--streams: the lowest std a stream's return is divided by")
    p.add_argument("--ladder", default="", help="--engine cpp: a stake ladder, e.g. WHITE,RED,GREEN,BLACK,BLUE,PURPLE,ORANGE,GOLD. Every --ladder-every updates "
                   "the network plays the NEXT stake greedily (--ladder-games games, fixed seeds) and the preview's win rate is logged; once it has "
                   "been at least --ladder-dense on --ladder-confirm previews in a row (and the current stake has had --ladder-min-iters updates) the "
                   "training stake moves up, with --ladder-keep of the environments staying on the stake below")
    p.add_argument("--ladder-every", type=int, default=25)
    p.add_argument("--ladder-games", type=int, default=150)
    p.add_argument("--ladder-dense", type=float, default=0.15, help="the preview win rate at which the next stake's win signal counts as dense enough")
    p.add_argument("--ladder-confirm", type=int, default=2)
    p.add_argument("--ladder-min-iters", type=int, default=100)
    p.add_argument("--ladder-keep", type=float, default=0.25, help="share of the environments on the stake below the training stake")
    p.add_argument("--ladder-ahead", type=float, default=0.0,
                   help="share of the environments on the stake AHEAD of the training stake (harder games, so it does not get complacent and the "
                        "next stake's win signal grows); the training stake keeps the rest")
    p.add_argument("--ladder-ent-bump", type=float, default=0.003,
                   help="--ladder: an extra entropy coefficient when the stake moves up, on decisions outside rounds only (x --ent-build), "
                        "decaying linearly over --ladder-ent-decay updates: a harder stake needs other builds tried")
    p.add_argument("--ladder-ent-decay", type=int, default=100)
    p.add_argument("--stall-ent", action="store_true",
                   help="--streams: when the training win rate stops improving (the mean of the last --stall-iters updates less than --stall-delta points "
                        "over the window before), start an entropy bump on decisions outside rounds: --stall-ent-bump, doubling for every stall in a "
                        "row up to --stall-ent-cap, decaying over --stall-ent-decay updates; cut short if the win rate then falls --stall-guard points")
    p.add_argument("--start-ante", type=int, default=1,
                   help="--engine cpp: every game begins in the shop of this ante (blinds before it counted as beaten), e.g. 8 with --start-money / "
                        "--start-money-p and --level-cap-* (a limit on planet use): a sandbox for what is strong late in a run")
    p.add_argument("--level-cap-mode", type=int, default=0,
                   help="--level-cap-*: 0 blocks planets after --level-cap-n uses; 1 lets planets be bought and used without limit (Constellation grows) but "
                        "they raise no hand level past --level-cap-n level-ups")
    p.add_argument("--no-part-credit", action="store_true", help="--engine cpp: a lost blind earns nothing for the chips scored in it: the blind rewards are the potential W(furthest blind beaten) and the win bonus only")
    p.add_argument("--log-partial", action="store_true",
                   help="--engine cpp: the part credit for a lost blind is on a log scale of the chips reached (route-agnostic and dense where a linear share is ~0)")
    p.add_argument("--pack-restock", type=int, default=0, help="the sandbox: a pack bought is replaced by another, this many times per shop visit")
    p.add_argument("--scaled-start", type=int, default=0,
                   help="the sandbox: every game starts holding 1..N growing jokers (random) whose counters are already grown --scaled-rounds rounds (no ceiling)")
    p.add_argument("--scaled-shop", action="store_true",
                   help="the sandbox: a growing joker made for a shop or a pack also arrives with its counter grown --scaled-rounds rounds")
    p.add_argument("--scaled-rounds", type=int, default=30)
    p.add_argument("--pack-bias", default="", help="the sandbox: pack kind weights multiplied, e.g. arcana:3,standard:3,celestial:2")
    p.add_argument("--strength", action="store_true",
                   help="--joker-desc --spec, a new network: the item descriptors also carry each joker's measured strength (joker_strength.json, tools/joker_strength.py)")
    p.add_argument("--spec", action="store_true",
                   help="--joker-desc, a new network: the item descriptors also carry joker_spec's exact numbers (magnitudes, growth, conditions)")
    p.add_argument("--scenario-p", type=float, default=0.0,
                   help="--engine cpp: share of games that are scenarios, a sampled build at a sampled ante (engine Env::apply_scenario); the rest are played from ante 1")
    p.add_argument("--scenario-mix", default="1,0,0", help="the kinds of scenario, weights: one blind, three to six blinds with shops, the rest of the run")
    p.add_argument("--scenario-phases", default="",
                   help="P1,P2,P3 iterations: mostly one-blind scenarios, then mostly multi-blind ones, then real runs rising from --scenario-real-start to --scenario-real-end")
    p.add_argument("--scenario-real-start", type=float, default=0.2)
    p.add_argument("--scenario-real-end", type=float, default=0.9)
    p.add_argument("--scenario-min-ante", type=int, default=1)
    p.add_argument("--scenario-max-ante", type=int, default=7)
    p.add_argument("--scenario-reroll-cap", type=int, default=3, help="rerolls per shop in the few-blinds scenarios")
    p.add_argument("--scenario-couple", type=float, default=0.5, help="share of scenarios whose deck theme and jokers are made to fit each other")
    p.add_argument("--aux-play-coef", type=float, default=0.0,
                   help="auxiliary head: the score of the play that was chosen (log10 / 6), from the state and the action")
    p.add_argument("--aux-loo-coef", type=float, default=0.0,
                   help="auxiliary head: each held joker's leave-one-out contribution, the engine's marginal columns (the network must not read them: no --potential)")
    p.add_argument("--record-wins", action="store_true",
                   help="--engine cpp: each won game's actions are written with it to <out>_wins.jsonl (replay from its seed, stake and the run's settings)")
    p.add_argument("--stall-now", action="store_true",
                   help="--stall-ent, one-shot: start a stall bump right now at the first level (keep the flag out of the config, or every restart repeats it)")
    p.add_argument("--stall-iters", type=int, default=150)
    p.add_argument("--stall-delta", type=float, default=1.0, help="points of win rate (percent) a window must gain over the one before it")
    p.add_argument("--stall-ent-bump", type=float, default=0.003)
    p.add_argument("--stall-ent-cap", type=float, default=0.012)
    p.add_argument("--stall-ent-decay", type=int, default=100)
    p.add_argument("--stall-guard", type=float, default=3.0)
    p.add_argument("--ladder-set", default="", help="--ladder, one-shot: move the training stake to this one now (the update counter on it starts again and "
                   "the entropy bump fires); take the flag out of the config after the start, or every restart repeats it")
    p.add_argument("--eval-every", type=int, default=0, help="a fixed evaluation (greedy, fixed seeds) every this many updates; 0: off")
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--eval-seed0", type=int, default=9_000_000)
    p.add_argument("--eval-lookahead", type=int, default=0, help="also evaluate with a one-step lookahead outside rounds (eval_lookahead.py): the policy's this many "
                   "likeliest moves (+ leave / skip / select) each tried on a copy, scored by reward + V(next position); logged as eval_la (0: off)")
    p.add_argument("--eval-la-games", type=int, default=0, help="--eval-lookahead: games (0: --eval-games)")
    p.add_argument("--priv-actor", action="store_true",
                   help="with --priv: the policy reads the privileged observation too (a network that needs the future: exact-seed games only)")
    p.add_argument("--potential", action="store_true",
                   help="--engine cpp: with the marginal-value inputs, also what each growing joker would be worth once "
                        "its enabling events happen (marginal.py POTENTIAL); implies --marginal")
    p.add_argument("--blind-scale", type=float, default=1.0,
                   help="--engine cpp: the reward for each blind beaten (and the part-credit for the one lost) times "
                        "this; 0 leaves a game paid for winning and --ante8-bonus only")
    p.add_argument("--ante8-bonus", type=float, default=0.0, help="--engine cpp: reward when ante 8 is reached")
    p.add_argument("--cap-jokers", default="", help="--engine cpp: groups of jokers ('+' between groups; each comma names, or flat_mult / flat_chips / grow_mult) of which at most --cap-max may be held "
                   "at once, in training and evaluation games")
    p.add_argument("--cap-max", default="1")
    p.add_argument("--ban-jokers", default="", help="--engine cpp: jokers (names, comma separated) out of the pool in "
                   "every game, e.g. bull,bootstraps")
    p.add_argument("--ban-top", type=int, default=0,
                   help="--engine cpp: the shifting world (0: off): the N jokers the policy buys most, recomputed "
                        "every --ban-every iterations from its own purchases, are out of the pool in --ban-share of games")
    p.add_argument("--ban-share", type=float, default=0.3)
    p.add_argument("--ban-share-final", type=float, default=None)
    p.add_argument("--ban-every", type=int, default=50)
    p.add_argument("--tilt-share", type=float, default=0.0,
                   help="--engine cpp: share of games in which one random hand type starts --tilt-levels levels up")
    p.add_argument("--tilt-share-final", type=float, default=None)
    p.add_argument("--tilt-levels", type=int, default=2)
    p.add_argument("--tail-q", type=float, default=0.0,
                   help="--engine cpp: the risk-seeking update (0: off): advantages above this quantile of the batch "
                        "(0.7: the top 30%%) count as good, the rest as none; blended in from --tail-from to --tail-to")
    p.add_argument("--tail-max", type=float, default=1.0, help="the most of the update that is risk-seeking (0.5: half)")
    p.add_argument("--tail-from", type=int, default=400, help="iteration where the risk-seeking share starts to rise")
    p.add_argument("--tail-to", type=int, default=1200, help="iteration where it reaches 1")
    p.add_argument("--weight-power", type=float, default=1.0,
                   help="--engine cpp: blind rewards rise with the ante as (ante / 7) ** power (1: linear; >1: convex), "
                        "from 1 at ante 1 to --ante-weight at ante 8")
    p.add_argument("--value-warmup", type=int, default=0,
                   help="--engine cpp: for this many first iterations only the value head trains (the policy and the "
                        "encoder are frozen), so a reward that changed does not meet a baseline calibrated to the old one")
    p.add_argument("--value-warmup-lr-mult", type=float, default=10.0)
    p.add_argument("--aux-coef", type=float, default=0.0,
                   help="--engine cpp: weight of two prediction heads trained beside the policy and the value (0: "
                        "none; a new network gets them only when this is set): whether the blind now faced is "
                        "beaten, and how many blinds the game will beat. A learning signal for the state at every "
                        "step; changes no reward")
    p.add_argument("--money-target", type=float, default=0.0,
                   help="--engine cpp: the money curriculum (0: off): --money-share of training games start with extra "
                        "money, and the amount moves so that those games on the target stake are won this often")
    p.add_argument("--money-share", type=float, default=0.7, help="share of games that get the curriculum's money")
    p.add_argument("--money-start", type=float, default=300.0, help="the extra dollars at the start of the run")
    p.add_argument("--money-max", type=float, default=500.0)
    p.add_argument("--money-gain", type=float, default=0.3,
                   help="how hard the amount reacts to the gap between the win rate and --money-target")
    p.add_argument("--start-money", type=int, default=0,
                   help="--engine cpp: extra starting money for a share (--start-money-p) of training games")
    p.add_argument("--start-money-p", type=float, default=0.0,
                   help="--engine cpp: share of training games that start with --start-money extra")
    p.add_argument("--start-money-p-final", type=float, default=None,
                   help="--engine cpp: --start-money-p moves linearly to this over the run")
    p.add_argument("--start-money2", type=int, default=0,
                   help="--engine cpp: a second, separate tier of extra starting money (--start-money2-p of games)")
    p.add_argument("--start-money2-p", type=float, default=0.0)
    p.add_argument("--start-money2-p-final", type=float, default=None)
    p.add_argument("--attn", type=int, default=0,
                   help="--engine cpp, a new network: layers of self-attention over jokers, hand, consumables "
                        "and a global token (0: the mean / max summaries of the older networks)")
    p.add_argument("--joker-desc", action="store_true",
                   help="--engine cpp, a new network: item embeddings also see what the item does (joker_desc.py)")
    p.add_argument("--gpu-fraction", type=float, default=0.0,
                   help="--engine cpp: ceiling on this process's share of the GPU's memory (0: none)")
    p.add_argument("--lam-build", type=float, default=None,
                   help="--engine cpp: the GAE lambda for decisions outside a round and for those inside one that "
                        "change something lasting (a joker's counter, jokers / consumables held, the deck, hand "
                        "levels, vouchers); --lam then applies to the other plays and discards. Default: --lam")
    p.add_argument("--explore-items", type=float, default=0.0,
                   help="--engine cpp: share of the behaviour policy spread over taking an economy joker, a boss "
                        "reroll voucher, a deck tarot or a growing joker on offer (or rerolling the boss), where possible")
    p.add_argument("--explore-groups", default="econ,boss,deck",
                   help="--engine cpp: what --explore-items covers: econ (economy jokers), boss (the boss-reroll "
                        "vouchers and the reroll itself), luchador (Luchador, Mr. Bones, selling Luchador on a live "
                        "boss), deck (Hanged Man, Death, Strength), grow (growing jokers)")
    p.add_argument("--explore-actions", type=float, default=0.0,
                   help="--engine cpp: share of the behaviour policy spread over EVERY individual legal action outside rounds (each buy slot, sale, reroll, skip, pack, voucher, swap, replace, pick, use of a consumable), not over kinds: no such action can fall to probability zero")
    p.add_argument("--explore-balance", action="store_true",
                   help="with --explore-actions: the share is weighted toward the actions taken least (inverse of a running take rate, fixed within an update so the ratio stays exact): rare actions are tried the most")
    p.add_argument("--explore-xmult", type=float, default=0.0,
                   help="--engine cpp: share of the behaviour policy spread over taking a multiplier joker (shop, "
                        "pack, replace) at decisions where one is on offer")
    p.add_argument("--explore-use", type=float, default=1.0,
                   help="--engine cpp: shares of the --explore-types floor that go to using a consumable when that "
                        "is legal (every other legal kind of action gets one share)")
    p.add_argument("--start-xmult", type=float, default=0.0,
                   help="--engine cpp: share of training games that start with one random multiplier joker")
    p.add_argument("--arch", default="transformer", choices=["transformer", "nnue"], help="nnue: the Stockfish-shaped actor-critic (balatro_rl/nnue_ac.py)")
    p.add_argument("--nnue-width", type=int, default=256, help="--arch nnue: the accumulator's width")
    p.add_argument("--topk", "--nnue-topk", dest="topk", type=int, default=0, help="a new network: only this many plays and this many discards are scored "
                   "(the highest predicted scores; the discards keeping the most toward a flush / straight / pairs); stored in the network (0: all)")
    p.add_argument("--resume", action="store_true",
                   help="--engine cpp: continue the run in --out from its checkpoint and log")
    p.add_argument("--lr-scale", type=float, default=1.0, help="--engine cpp: multiplies the learning-rate schedule")
    p.add_argument("--start-xmult-final", type=float, default=None, help="--engine cpp: fade --start-xmult to this")
    p.add_argument("--start-any", type=float, default=0.0,
                   help="--engine cpp: share of training games that start with one random joker of any kind")
    p.add_argument("--catalogue", type=float, default=0.0, help="--engine cpp: share of training games whose every shop holds every eligible non-legendary joker "
                   "(Game::catalogue: 24 to a page, a reroll turns the page for free and counts for nothing else; jokers only in the card slots, packs and vouchers as usual). "
                   "The evaluation plays the usual shop")
    p.add_argument("--blind-bases", default="", help="training games: each ante's base chips (comma list; small 1x, big 1.5x, boss the boss's multiple), replacing the "
                   "stake's table; use with --win-ante for a longer game (e.g. 12 antes reaching 150k base = a 300k boss)")
    p.add_argument("--start-any-eternal", action="store_true", help="--start-any: the random starting joker is eternal (cannot be sold)")
    p.add_argument("--start-any2", type=float, default=0.0, help="--engine cpp: of the games that start with a random joker (--start-any), this share start with a "
                   "second one too (drawn the same way, never the same joker); training games only")
    p.add_argument("--start-any-final", type=float, default=None, help="--engine cpp: fade --start-any to this")
    p.add_argument("--verdant-bonus", type=float, default=0.0,
                   help="--engine cpp: reward, once, for the sale that lifts Verdant Leaf's debuff")
    p.add_argument("--bank-bonus", type=float, default=0.0,
                   help="--engine cpp: potential on money held: a full bank ($25, where interest stops) is worth "
                        "this much reward, given as it is banked and taken back as it is spent or the game ends")
    p.add_argument("--build-bonus", type=float, default=0.0,
                   help="--engine cpp: potential on the jokers held: a working multiplier joker is worth this much, "
                        "a +mult joker half, a chips joker a third (once each); given when acquired, taken back "
                        "when it goes or the game ends")
    p.add_argument("--build-roles", default="xmult,mult,chips",
                   help="--build-bonus: the joker roles it pays for, comma separated among xmult (full), mult (half), chips (a third), scaler (full: a joker that grows over the run)")
    p.add_argument("--build-second", type=float, default=0.0,
                   help="--engine cpp: with --build-bonus, a second working multiplier joker is worth this share "
                        "of the first on top")
    p.add_argument("--build-bonus-final", type=float, default=None,
                   help="--engine cpp: --build-bonus moves linearly to this over the run (below zero: clamped)")
    p.add_argument("--oracle-bonus", type=float, default=0.0,
                   help="--engine cpp: potential on the ante-8 oracle (Env::final_boss_clear: the build with its counters projected to the ante-8 boss, the round "
                        "played by the exact solver): a build at the boss's target is worth this much, on a log scale from 1/10000 of it")
    p.add_argument("--oracle-bonus-final", type=float, default=None, help="--oracle-bonus moves linearly to this over the run")
    p.add_argument("--money-option-bonus", type=float, default=0.0,
                   help="--engine cpp: potential on the money-option curve (balatro_rl/data/money_option.json, tools/money_option.py): the chance of fishing a strong "
                        "joker with the money held, within two antes, is worth this much")
    p.add_argument("--money-option-bonus-final", type=float, default=None, help="--money-option-bonus moves linearly to this over the run")
    p.add_argument("--oracle-inputs", action="store_true", help="the network also reads the oracle and the curve (ob['ex'], a zero-initialised input path: a resumed network is unchanged)")
    p.add_argument("--oracle-samples", type=int, default=1, help="redrawn decks per oracle measurement")
    p.add_argument("--oracle-nodes", type=int, default=150, help="the solver's node cap per oracle round")
    p.add_argument("--bank-bonus-final", type=float, default=None,
                   help="--engine cpp: --bank-bonus moves linearly to this over the run (below zero: clamped)")
    p.add_argument("--ent-build", type=float, default=1.0,
                   help="--engine cpp: the entropy bonus counts decisions outside rounds this many times")
    p.add_argument("--start-xscale", type=float, default=0.0,
                   help="--engine cpp: share of games that start holding one scaling multiplier joker (Constellation, "
                        "Hologram, Vampire, Lucky Cat, Glass) with its counter already grown to --start-xscale-val")
    p.add_argument("--start-xscale-final", type=float, default=None)
    p.add_argument("--start-xscale-val", type=float, default=2.0, help="the multiplier it is handed over at")
    p.add_argument("--start-xscale-val-final", type=float, default=None)
    p.add_argument("--menu-slots", type=int, default=0,
                   help="--engine cpp: a curriculum with a big shop: every shop shows this many jokers (up to 24; only jokers) instead of its usual two cards; it shrinks linearly to the usual shop "
                   "over --menu-fade-iters updates (0: off). The fixed evaluation always plays the usual shop")
    p.add_argument("--menu-fade-iters", type=int, default=3000, help="updates the big shop takes to shrink to the usual one")
    p.add_argument("--start-xscale-fade-iters", type=int, default=3000,
                   help="iterations over which the share and the multiplier move to their -final values")
    p.add_argument("--level-cap-share", type=float, default=0.0,
                   help="--engine cpp: share of games where, until --level-cap-ante, at most --level-cap-n planet cards "
                        "may be bought, picked or used (0: off)")
    p.add_argument("--level-cap-share-final", type=float, default=None)
    p.add_argument("--level-cap-n", type=int, default=2)
    p.add_argument("--level-cap-ante", type=int, default=4, help="the cap holds in antes below this")
    p.add_argument("--stake-curriculum", action="store_true",
                   help="--engine cpp: every --stake-window iterations, if the blinds on --stake rose by less than "
                        "--stake-min-gain over the last window, --stake-step of the environments move to --stake-easy "
                        "(down to --stake-floor left on --stake)")
    p.add_argument("--stake-easy", default="GOLD")
    p.add_argument("--stake-window", type=int, default=100)
    p.add_argument("--stake-warmup", type=int, default=300)
    p.add_argument("--stake-min-gain", type=float, default=0.15)
    p.add_argument("--stake-step", type=float, default=0.1)
    p.add_argument("--stake-floor", type=float, default=0.1)
    p.add_argument("--stake-min-games", type=int, default=2000, help="games on --stake a window needs to count")
    p.add_argument("--stake-mix", default="",
                   help="--engine cpp: shares of the environments on other stakes, e.g. WHITE:0.06,GREEN:0.05; the "
                        "rest play --stake, whose results are the ones logged")
    p.add_argument("--amp-recompute", action="store_true",
                   help="--engine cpp --amp: recompute the rollout's log-probs at the update's batch size")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.strength:                                  # the strength columns live in the descriptor table: --joker-desc --spec --strength
        a.joker_desc = a.spec = True
    if a.epochs is None:
        a.epochs = 2 if a.mode == "bc" else 4
    if a.lr is None:
        a.lr = 1e-3 if a.mode == "bc" else (1e-4 if a.init else 3e-4)
    if a.phase is None:
        a.phase = a.mode
    if a.pipeline and a.envs < 2:
        p.error("--pipeline needs --envs 2 or more")
    if a.strategic and not (a.tactical or a.init):
        p.error("--strategic needs --tactical or --init")
    if a.log is None:
        a.log = a.out.replace(".pt", "_log.jsonl")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.engine == "cpp":
        if a.mode != "ppo" or a.pipeline or a.strategic:
            p.error("--engine cpp runs plain PPO (no bc, --pipeline or --strategic)")
        return train_ppo_cpp(a)
    (train_bc if a.mode == "bc" else train_ppo)(a)


if __name__ == "__main__":
    main()
