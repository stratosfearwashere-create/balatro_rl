"""Self-play with search, training, evaluation.

Iteration: worker processes play games with the agent (search on, Gumbel noise at the root), recording at
each decision the network's inputs (with the potential Phi), the search's improved policy and, once the
game is over, the raw outcomes (rewards/targets.py). The learner then trains on a window of recent
iterations:
    cross-entropy to the improved policy (searched decisions only; logged as loss_policy, and as
        loss_policy_kl = KL(improved policy || network), which is the part the network can remove)
  + kappa * KL(pi || pi_solver) on in-round decisions            (kappa anneals to exactly 0)
  + MSE(V, clip(z + novelty, 0, 1)), V = Phi + R                    (z = (1 - lam) win + lam progress)
        (with potential.value_bound = floor_sigmoid: cross-entropy on the bounded V's logit, az/net.py)
  + weighted auxiliary losses: P(clear), ante reached (8 classes), log(blind score / required),
    next blind's headroom
Targets are computed at training time from the schedule at the current step (step = decisions played by
self-play so far; lam on its own clock, which only runs while the recent self-play win rate clears
lambda_gate_win_rate, see rewards.config.LambdaGate), so lam, beta and kappa always match
rewards.RewardConfig.schedule. Warm-up iterations
play without search and train the heads first.

Diagnostics (one JSON line per iteration, *_log.jsonl): the schedule, each shaping component per episode,
calibration of Phi and V against actual wins, headroom cost per decision, losses; with --eval-every, the
held-out win rate (the measure of success) by ante reached and boss type, calibration on held-out games,
and a warning when the shaped return (mean value target) rises while the held-out win rate doesn't.

    python -m balatro_rl.az.train run --iters 50 --games 64 --workers 8 --out checkpoints/az.pt --eval-every 5
    python -m balatro_rl.az.train run ... --reward-config rewards.yaml
    python -m balatro_rl.az.train eval --model checkpoints/az.pt --games 100
    python -m balatro_rl.az.train eval --model none --games 50 --no-search      # the priors alone
Novelty ablation (keep novelty only if it beats search's own exploration on held-out win rate): run twice,
the second time with a config that sets `novelty: {beta: 0}`, and compare the evaluations.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import pickle
import time
from collections import Counter

import numpy as np

from ..rewards.config import LambdaGate, RewardConfig, PotentialConfig
from ..rewards.diagnostics import HackAlarm, breakdown, calibration, mean_or_nan
from ..rewards.novelty import NoveltyCounter
from ..rewards.targets import GameRecorder, N_ANTE_CLASSES, blind_index, z_target

EVAL_SEED0 = 10_000
GEN_SEED0 = 900_000
STAKE = "WHITE"


def _compact(enc):
    s, c = enc
    s = {k: (v.astype(np.float16) if (v.dtype == np.float32 and k not in ("phi", "prog")) else v)
         for k, v in s.items()}
    c = {k: (v.astype(np.float16) if (v.dtype == np.float32 and k != "c_prior") else v) for k, v in c.items()}
    return s, c


def play_game(agent, seed: int, deck: str = "RED", stake: str = STAKE, explore: bool = True, record: bool = True,
              verbose: bool = False):
    """Play one game. Returns (rows, info). Every decision the network saw gets a row (phi, V, targets);
    with `record`, rows also carry the network inputs and the search policy for training."""
    from ..sim.game import Game
    from .world import World
    w = World(Game(seed=seed, deck_type=deck, stake=stake))
    agent.reseed(seed * 7919 + 17)
    rec = GameRecorder(agent.potential)
    while not w.done:
        g = w.g
        d = agent.decide(w, explore=explore)
        if d.action is None:
            g.state = "GAME_OVER"
            break
        if verbose:
            print(f"  ante {g.ante} {g.state:15s} ${g.money:<4} {d.reason:10s} {describe(w, d.action)}")
        if d.enc is not None:
            extra = {"value": d.value, "searched": d.searched}
            if record:
                extra.update(enc=_compact(d.enc), pi=d.policy.astype(np.float32))
            rec.decision(w, **extra)
        prev = (g.state, blind_index(g), g.boss)
        w.step(d.action)
        rec.transition(*prev, w)
    g = w.g
    rows = rec.finish(g)
    info = {"seed": seed, "won": g.state == "WON", "blinds": g.furthest_blind, "ante": g.ante, "steps": w.steps,
            "bosses": rec.bosses}
    return rows, info


def describe(w, a) -> str:
    g = w.g
    if a.kind in ("play", "discard"):
        return f"{a.kind} " + " ".join(repr(g.hand[i]) for i in a.cards)
    if a.kind == "use":
        s = f"use {g.consumables[a.idx].key}"
        return s + (" on " + " ".join(repr(g.hand[i]) for i in a.cards) if a.cards else "")
    if a.kind == "buy":
        return f"buy {g.shop[a.idx].key} ${g.shop[a.idx].cost}"
    if a.kind == "buy_pack":
        return f"buy {g.shop_packs[a.idx].key}"
    if a.kind == "pick":
        x = g.pack_cards[a.idx]
        k = getattr(x, "key", None)
        s = f"pick {k if isinstance(k, str) else repr(x)}"
        return s + (" on " + " ".join(repr(g.pack_hand[i]) for i in a.cards) if a.cards else "")
    if a.kind == "sell_joker":
        return f"sell j_{g.jokers[a.idx].key}"
    if a.kind == "sell_cons":
        return f"sell {g.consumables[a.idx].key}"
    if a.kind == "move_joker":
        return f"move j_{g.jokers[a.idx].key} to slot {a.to}"
    return a.kind


# ------------------------------------------------------------------ workers
def _make_agent(model, search: bool, seed: int, cfg_over: dict | None = None, pot: dict | None = None):
    import torch
    torch.set_num_threads(1)
    from .agent import Agent, AgentConfig
    from .net import AZNet, load_net
    cfg = AgentConfig(search=search, **(cfg_over or {}))
    pcfg = PotentialConfig(**pot) if pot else None
    if model and model != "none":
        net = load_net(model, "cpu")
    else:                                   # untrained: the same weights in every worker and every run
        torch.manual_seed(0)
        net = new_net(pcfg or PotentialConfig()).eval()
    return Agent(net, cfg, seed=seed, potential=pcfg)


def new_net(pcfg: PotentialConfig):
    from .net import AZNet
    return AZNet(value_residual=pcfg.value_residual, value_bound=pcfg.value_bound,
                 value_init_scale=pcfg.value_init_scale, value_init_bias=pcfg.value_init_bias)


def _headroom_stats(agent) -> dict:
    st = agent.potential.headroom.stats
    return {"headroom_calls": st["calls"], "headroom_sec": st["seconds"]}


def _light(r: dict) -> dict:
    return {k: r[k] for k in ("phi", "value", "win", "headroom", "game")}


def _gen_worker(args):
    model, seeds, search, out_path, deck, stake, cfg_over, pot = args
    agent = _make_agent(model, search, seeds[0], cfg_over, pot)
    rows, infos = [], []
    for s in seeds:
        r, info = play_game(agent, s, deck, stake, explore=True)
        for x in r:
            x["game"] = s
        rows += r
        infos.append(info)
    with open(out_path, "wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    return infos, {**dict(agent.stats), **_headroom_stats(agent)}, [_light(x) for x in rows]


def _eval_worker(args):
    model, seeds, search, deck, stake, cfg_over, pot = args
    agent = _make_agent(model, search, seeds[0], cfg_over, pot)
    infos, light = [], []
    for s in seeds:
        t = time.process_time()
        rows, info = play_game(agent, s, deck, stake, explore=False, record=False)
        info["cpu_sec"] = time.process_time() - t
        infos.append(info)
        light += [_light({**r, "game": s}) for r in rows]
    return infos, {**dict(agent.stats), **_headroom_stats(agent)}, light


def _bench_worker(args):
    """Fixed games with search on a fixed (seeded, untrained) network: identical work on every version."""
    seeds, cfg_over = args
    import torch
    torch.set_num_threads(1)
    from .agent import Agent, AgentConfig
    from .net import AZNet
    torch.manual_seed(0)
    net = AZNet().eval()
    out = []
    for s in seeds:
        agent = Agent(net, AgentConfig(**(cfg_over or {})), seed=s)
        t = time.perf_counter()
        _, info = play_game(agent, s, "RED", STAKE, explore=True, record=False)
        out.append({"sec": time.perf_counter() - t, "decisions": agent.stats["decisions"],
                    "sims": agent.stats["sims"], **_headroom_stats(agent), **_cache_stats(agent), **info})
    return out


def _cache_stats(agent) -> dict:
    return {k: v for k, v in agent.stats.items() if k.startswith("cache_")}


def bench(games: int, workers: int, seed0: int = 950_000, cfg_over=None) -> dict:
    """games/hour with search, on fixed seeds (the same games whatever the code version, as long as the
    decisions are unchanged)."""
    seeds = list(range(seed0, seed0 + games))
    t = time.time()
    with mp.get_context("spawn").Pool(min(workers, games)) as pool:
        parts = pool.map(_bench_worker, [(c, cfg_over) for c in _split(seeds, workers)])
    wall = time.time() - t
    games_ = sum(parts, [])
    sec = sum(g["sec"] for g in games_)
    res = {"games": len(games_), "workers": min(workers, games), "wall_min": round(wall / 60, 2),
           "games_per_hour": round(len(games_) / wall * 3600, 1),
           "sec_per_game_single_core": round(sec / len(games_), 2),
           "decisions": sum(g["decisions"] for g in games_), "sims": sum(g["sims"] for g in games_),
           "blinds": round(float(np.mean([g["blinds"] for g in games_])), 3)}
    hits = sum(g.get("cache_hits", 0) for g in games_)
    looks = hits + sum(g.get("cache_misses", 0) for g in games_)
    if looks:
        res["score_cache_hit%"] = round(100 * hits / looks, 1)
    return res


def _split(seeds, n):
    chunks = [seeds[i::n] for i in range(n)]
    return [c for c in chunks if c]


def summarize(infos) -> dict:
    return {"games": len(infos), "win%": 100.0 * float(np.mean([i["won"] for i in infos])),
            "blinds": float(np.mean([i["blinds"] for i in infos])),
            "ante": float(np.mean([min(i["ante"], 8) for i in infos]))}


def _checkpoint_meta(model) -> dict:
    if not model or model == "none":
        return {}
    import torch
    return torch.load(model, map_location="cpu", weights_only=False).get("extra", {})


def evaluate(model, games: int, workers: int, search: bool, deck="RED", stake=STAKE, seed0=EVAL_SEED0,
             cfg_over=None, rcfg: RewardConfig | None = None) -> dict:
    """Held-out games (greedy, no exploration). The value of a finished game in the search uses lam from
    the reward schedule at the checkpoint's training step and lambda clock, matching what its value head
    was trained on. Also returns the override rates by phase (agent.override_summary), the result of each
    game (per_game, for paired comparisons on the same seeds) and the cost in CPU seconds per game."""
    from .agent import override_summary
    meta = _checkpoint_meta(model)
    rcfg = rcfg or RewardConfig.from_dict(meta.get("rewards"))
    cfg_over = {"lam": rcfg.schedule(meta.get("step", 0), meta.get("lambda_clock")).lam, **(cfg_over or {})}
    seeds = list(range(seed0, seed0 + games))
    jobs = [(model, c, search, deck, stake, cfg_over, dict(rcfg.potential.__dict__)) for c in _split(seeds, workers)]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(_eval_worker, jobs)
    infos = sum((p[0] for p in parts), [])
    light = sum((p[2] for p in parts), [])
    stats = Counter()
    for p in parts:
        stats.update(p[1])
    res = summarize(infos)
    res["autoplay%"] = 100.0 * stats["autoplay"] / max(1, stats["decisions"])
    res["sims/decision"] = stats["sims"] / max(1, stats["decisions"])
    res["headroom_ms/decision"] = 1e3 * stats["headroom_sec"] / max(1, stats["decisions"])
    res["breakdown"] = breakdown(infos)
    res["override"] = override_summary(stats)
    res["decisions"] = stats["decisions"]
    res["cpu_sec/game"] = float(np.mean([i["cpu_sec"] for i in infos]))
    res["per_game"] = sorted(([i["seed"], i["blinds"], int(i["won"]), min(i["ante"], 8)] for i in infos))
    games = [x["game"] for x in light]
    res["calibration_phi"] = calibration([x["phi"] for x in light], [x["win"] for x in light], games)
    res["calibration_v"] = calibration([x["value"] for x in light], [x["win"] for x in light], games)
    return res


# ------------------------------------------------------------------ learner
def load_rows_jsonl(path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rows(paths) -> list:
    rows = []
    for p in paths:
        with open(p, "rb") as f:
            rows += pickle.load(f)
    return rows


def value_target(r: dict, sched, novelty: NoveltyCounter | None) -> tuple[float, float]:
    """(training target, novelty bonus in it): clip(z + bonus, 0, 1); exactly z once beta is 0.
    Training only: evaluation and the auxiliary heads never see the bonus."""
    z = z_target(r["win"], r["progress"], sched.lam)
    bonus = novelty.bonus(r["sig"], sched.beta) if novelty is not None else 0.0
    if bonus == 0.0:
        return z, 0.0
    return min(1.0, max(0.0, z + bonus)), bonus


def train_on(net, rows, steps: int, batch: int, lr: float, device: str, rcfg: RewardConfig, sched,
             novelty: NoveltyCounter | None = None, opt=None):
    import torch
    import torch.nn.functional as F
    from .net import collate
    net.train()
    opt = opt or torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    aw = rcfg.aux_loss_weights
    rng = np.random.default_rng(len(rows) + sched.step)
    logs = Counter()
    n = 0
    for _ in range(steps):
        idx = rng.choice(len(rows), size=min(batch, len(rows)), replace=False)
        mb = [rows[i] for i in idx]
        s, c, m = collate([r["enc"] for r in mb], device)
        logits, adj, out = net(s, c, m)
        A = m.shape[1]
        logp = torch.log_softmax(logits, -1)
        loss = torch.zeros((), device=device)
        # policy: the search's improved policy
        pi = np.zeros((len(mb), A), np.float32)
        for i, r in enumerate(mb):
            pi[i, :len(r["pi"])] = r["pi"]
        pi = torch.from_numpy(pi).to(device)
        searched = torch.tensor([bool(r["searched"]) and len(r["pi"]) > 0 for r in mb], device=device)
        if searched.any():
            pl = -(pi[searched] * logp[searched]).sum(-1).mean()
            loss = loss + pl
            logs["policy"] += pl.item()
            ent = -(pi[searched] * torch.log(pi[searched].clamp(min=1e-12))).sum(-1).mean()
            logs["policy_kl"] += (pl - ent).item()          # KL(target || network) = cross-entropy - H(target)
        # closeness to the solver, in-round decisions only, while kappa > 0
        in_round = torch.tensor([bool(r["in_round"]) for r in mb], device=device)
        if sched.kappa > 0 and in_round.any():
            lps = torch.log_softmax(c["c_prior"].float().masked_fill(~m, -1e9), -1)
            kl = (logp.exp() * (logp - lps)).masked_fill(~m, 0.0).sum(-1)[in_round].mean()
            loss = loss + sched.kappa * kl
            logs["solver_kl"] += kl.item()
        # value: V = Phi + R towards clip(z + novelty, 0, 1)
        tg = [value_target(r, sched, novelty) for r in mb]
        zt = torch.tensor([t[0] for t in tg], dtype=torch.float32, device=device)
        v = net.value(out, s["phi"], s["prog"], sched.lam)
        if net.value_bound == "none":
            vl = F.mse_loss(v, zt)
        else:                                               # V = lo + (1 - lo) sigmoid(u): cross-entropy on u
            lo = sched.lam * s["prog"].float()              # towards where the target sits between lo and 1
            t01 = ((zt - lo) / (1.0 - lo).clamp(min=1e-6)).clamp(0.0, 1.0)
            vl = F.binary_cross_entropy_with_logits(net.value_logit(out, s["phi"]), t01)
            logs["value_mse"] += F.mse_loss(v, zt).item()
        loss = loss + vl
        logs["value"] += vl.item()
        # auxiliary heads (predicted, never rewarded)
        f = lambda k: torch.tensor([float(r[k]) for r in mb], dtype=torch.float32, device=device)
        cl = F.binary_cross_entropy_with_logits(out[:, 1], f("clear"))
        al = F.cross_entropy(out[:, 4:4 + N_ANTE_CLASSES],
                             torch.tensor([r["ante_cls"] for r in mb], dtype=torch.long, device=device))
        aux = aw.p_clear_blind * cl + aw.ante_reached * al
        logs["clear"] += cl.item()
        logs["ante"] += al.item()
        for col, key, wgt in ((2, "ratio", aw.blind_score_ratio), (3, "next_head", aw.next_headroom)):
            t = f(key)
            ok = ~torch.isnan(t)
            if ok.any():
                l = F.mse_loss(out[ok, col], t[ok])
                aux = aux + wgt * l
                logs[key] += l.item()
        loss = loss + aux
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        logs["adj"] += adj[m].abs().mean().item()
        n += 1
    net.eval()
    return {k: v / max(1, n) for k, v in logs.items()}, opt


def shaping_components(rows, sched, novelty, w_head: float) -> dict:
    """Each shaping component, averaged per episode (diagnostics only)."""
    games = {}
    for r in rows:
        games.setdefault(r.get("game"), []).append(r)
    comp = {"phi": [], "phi_headroom_term": [], "phi_progress_term": [], "lambda_term": [], "novelty": [],
            "value_target": []}
    for rs in games.values():
        phi = np.mean([r["phi"] for r in rs])
        head = np.mean([w_head * r["headroom"] for r in rs])
        comp["phi"].append(phi)
        comp["phi_headroom_term"].append(head)
        comp["phi_progress_term"].append(phi - head)
        comp["lambda_term"].append(np.mean([z_target(r["win"], r["progress"], sched.lam) - r["win"] for r in rs]))
        vt = [value_target(r, sched, novelty) for r in rs]
        comp["novelty"].append(np.mean([t[1] for t in vt]))
        comp["value_target"].append(np.mean([t[0] for t in vt]))
    return {k: round(float(np.mean(v)), 5) if v else 0.0 for k, v in comp.items()}


def run(a):
    import torch
    from .agent import override_summary
    from .net import load_net, save_net
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.data, exist_ok=True)
    log_path = a.out.replace(".pt", "_log.jsonl")
    done_path = a.out.replace(".pt", "_done.txt")
    opt = None
    first = 1
    if os.path.exists(a.out) and not a.resume:
        raise SystemExit(f"{a.out} already exists: pass --resume to continue that run, or choose another --out")
    if a.resume and os.path.exists(a.out):
        # continue where the run stopped: network, optimizer, schedule step, reward config, novelty counts
        # and alarm history; iterations count on from the last one saved (--iters is the run's total)
        ck = torch.load(a.out, map_location=device, weights_only=False)
        meta0 = ck.get("extra", {})
        rcfg = RewardConfig.from_dict(meta0.get("rewards"))
        net = load_net(a.out, device)
        step = meta0.get("step", 0)
        first = meta0.get("iter", 0) + 1
        # runs saved before the lambda gate existed faded lambda on the step count: carry that on
        gate_state = meta0.get("lambda_gate", {"clock": meta0.get("lambda_clock", step)})
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        print(f"resuming {a.out} after iteration {first - 1} ({step:,} decisions)", flush=True)
    else:
        rcfg = RewardConfig.load(a.reward_config)
        step = 0
        gate_state = {"clock": 0}
        if a.init:
            net = load_net(a.init, device)
            step = _checkpoint_meta(a.init).get("step", 0)
        else:
            torch.manual_seed(0)
            net = new_net(rcfg.potential).to(device)
    gate = LambdaGate.from_state(rcfg.lambda_gate_win_rate, rcfg.lambda_gate_games, gate_state)
    meta = lambda it: {"iter": it, "step": step, "rewards": rcfg.to_dict(), "lambda_clock": gate.clock,
                       "lambda_gate": gate.state()}
    if first == 1:
        save_net(net, a.out, meta(0))
    cfg_over = json.loads(a.cfg) if a.cfg else {}
    novelty = NoveltyCounter(rcfg.novelty.window)
    alarm = HackAlarm(a.alarm_n)
    if first > 1:
        logged = [r for r in load_rows_jsonl(log_path) if r["iter"] < first] if os.path.exists(log_path) else []
        oldest = 1                              # novelty counts cover the last `window` decisions: load
        prev = {r["iter"]: r["step"] for r in logged}   # only the iterations that fall inside it
        for r in reversed(logged):
            if step - prev.get(r["iter"] - 1, 0) >= rcfg.novelty.window:
                oldest = r["iter"]
                break
        for s in sorted(glob.glob(os.path.join(a.data, "it*_w*.pkl"))):
            if oldest <= int(os.path.basename(s)[2:6]) < first:
                novelty.add(r["sig"] for r in load_rows([s]))
        if "recent" not in gate_state:          # old checkpoint: recent win rates from its log
            gate.recent = [(round(r["win%"] * r["games"] / 100), r["games"]) for r in logged][-50:]
        for row in logged:
            if "eval" in row:
                alarm.update(row["shaping"]["value_target"], row["eval"]["breakdown"]["win_rate"])
    if os.path.exists(done_path) and first <= a.iters:
        os.remove(done_path)                    # resumed with more iterations: not finished any more
    pot = dict(rcfg.potential.__dict__)
    for it in range(first, a.iters + 1):
        for stale in glob.glob(os.path.join(a.data, f"it{it:04d}_w*.pkl")):
            os.remove(stale)                    # games of an iteration that was cut off: played again
        t0 = time.time()
        search = it > a.warmup
        sched = rcfg.schedule(step, gate.clock)
        seeds = list(range(GEN_SEED0 + (it - 1) * a.games, GEN_SEED0 + it * a.games))
        jobs = [(a.out, c, search, os.path.join(a.data, f"it{it:04d}_w{k}.pkl"), a.deck, a.stake,
                 {"lam": sched.lam, **cfg_over}, pot)
                for k, c in enumerate(_split(seeds, a.workers))]
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            parts = pool.map(_gen_worker, jobs)
        infos = sum((p[0] for p in parts), [])
        light = sum((p[2] for p in parts), [])
        stats = Counter()
        for p in parts:
            stats.update(p[1])
        step += stats["decisions"]
        gate_open = gate.update(sum(i["won"] for i in infos), len(infos), stats["decisions"])
        t_gen = time.time() - t0
        new_rows = load_rows(sorted(glob.glob(os.path.join(a.data, f"it{it:04d}_w*.pkl"))))
        novelty.add(r["sig"] for r in new_rows)
        shards = sorted(glob.glob(os.path.join(a.data, "it*_w*.pkl")))
        rows = load_rows([s for s in shards if it - a.window < int(os.path.basename(s)[2:6]) <= it])
        sched = rcfg.schedule(step, gate.clock)
        losses, opt = train_on(net, rows, a.steps, a.batch, a.lr, device, rcfg, sched, novelty, opt)
        save_net(net, a.out, meta(it), opt)
        comp = shaping_components(new_rows, sched, novelty, rcfg.potential.w_head)
        annealed = sched.lam == 0.0 and sched.beta == 0.0
        exact = not annealed or all(value_target(r, sched, novelty)[0] == r["win"] for r in new_rows)
        gate_rate = gate.rate()
        row = {"iter": it, "step": step, "search": search, "lam": round(sched.lam, 5), "lambda_clock": gate.clock,
               "lambda_gate": {"open": gate_open, "win_rate": None if gate_rate is None else round(gate_rate, 4),
                               "threshold": gate.win_rate},
               "beta": round(sched.beta, 5),
               "kappa": round(sched.kappa, 5), **{k: round(v, 3) for k, v in summarize(infos).items()},
               "autoplay%": round(100.0 * stats["autoplay"] / max(1, stats["decisions"]), 2),
               "sims/decision": round(stats["sims"] / max(1, stats["decisions"]), 2),
               "headroom_ms/decision": round(1e3 * stats["headroom_sec"] / max(1, stats["decisions"]), 3),
               "override": override_summary(stats),      # self-play: includes the exploration noise
               "samples": len(rows), "gen_min": round(t_gen / 60, 2),
               "train_min": round((time.time() - t0 - t_gen) / 60, 2),
               "shaping": comp, "value_target_is_win": annealed and exact,
               "calibration_phi": calibration([x["phi"] for x in light], [x["win"] for x in light],
                                              [x["game"] for x in light]),
               "calibration_v": calibration([x["value"] for x in light], [x["win"] for x in light],
                                            [x["game"] for x in light]),
               **{f"loss_{k}": round(v, 5) for k, v in losses.items()}}
        if not exact:
            row["WARNING"] = "value target differs from win although lambda and beta have reached 0"
        for name in ("calibration_phi", "calibration_v"):
            if row[name]["status"] == "FLAT_OR_FALLING":
                row.setdefault("flags", []).append(f"{name}: actual win rate does not rise with it")
        if a.eval_every and it % a.eval_every == 0:
            ev = evaluate(a.out, a.eval_games, a.workers, True, a.deck, a.stake, cfg_over=cfg_over, rcfg=rcfg)
            row["eval"] = ev
            warn = alarm.update(comp["value_target"], ev["breakdown"]["win_rate"])
            if warn:
                row["ALARM"] = warn
                print("WARNING:", warn, flush=True)
            for name in ("calibration_phi", "calibration_v"):
                if ev[name]["status"] == "FLAT_OR_FALLING":
                    row.setdefault("flags", []).append(f"held-out {name}: actual win rate does not rise with it")
        with open(log_path, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        short = {k: v for k, v in row.items() if not k.startswith("calibration") and k not in ("eval", "override")}
        short["override%"] = {ph: v["final%"] for ph, v in row["override"].items()}
        if "eval" in row:
            short["eval"] = {k: row["eval"][k] for k in ("games", "win%", "blinds", "ante")}
            short["eval"]["override%"] = {ph: v["final%"] for ph, v in row["eval"]["override"].items()}
        print(json.dumps(short, default=float), flush=True)
    with open(done_path, "w") as f:
        f.write(f"finished {a.iters} iterations, {step} decisions\n")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="self-play with search, then train; repeat")
    r.add_argument("--iters", type=int, default=50)
    r.add_argument("--games", type=int, default=64, help="self-play games per iteration")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--warmup", type=int, default=2, help="first iterations play without search")
    r.add_argument("--window", type=int, default=8, help="train on this many recent iterations")
    r.add_argument("--steps", type=int, default=400, help="gradient steps per iteration")
    r.add_argument("--batch", type=int, default=128)
    r.add_argument("--lr", type=float, default=3e-4)
    r.add_argument("--deck", default="RED")
    r.add_argument("--stake", default=STAKE)
    r.add_argument("--init", default=None, help="start a new run from this checkpoint's network")
    r.add_argument("--resume", action="store_true",
                   help="if --out exists, continue that run from its last saved iteration (else start it)")
    r.add_argument("--data", default="checkpoints/az_data")
    r.add_argument("--out", default="checkpoints/az.pt")
    r.add_argument("--reward-config", default=None, help="YAML / JSON reward config (rewards/config.py)")
    r.add_argument("--eval-every", type=int, default=0)
    r.add_argument("--eval-games", type=int, default=100)
    r.add_argument("--alarm-n", type=int, default=3, help="evaluations of rising shaped return before the alarm")
    r.add_argument("--cfg", default="", help="AgentConfig overrides as JSON, e.g. '{\"budget_shop\": 16}'")
    b = sub.add_parser("bench", help="games/hour with search on fixed seeds and a fixed untrained network")
    b.add_argument("--games", type=int, default=14)
    b.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    b.add_argument("--cfg", default="")
    e = sub.add_parser("eval", help="play unseen seeds greedily")
    e.add_argument("--model", default="none", help="checkpoint, or 'none' for an untrained network (the priors)")
    e.add_argument("--games", type=int, default=100)
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    e.add_argument("--no-search", action="store_true")
    e.add_argument("--deck", default="RED")
    e.add_argument("--stake", default=STAKE)
    e.add_argument("--cfg", default="")
    e.add_argument("--reward-config", default=None)
    e.add_argument("--verbose", action="store_true", help="play one game and print every move")
    a = p.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "bench":
        print(json.dumps(bench(a.games, a.workers, cfg_over=json.loads(a.cfg) if a.cfg else None)))
    elif a.cmd == "eval":
        cfg_over = json.loads(a.cfg) if a.cfg else None
        rcfg = RewardConfig.load(a.reward_config) if a.reward_config else None
        if a.verbose:
            pot = dict((rcfg or RewardConfig()).potential.__dict__)
            agent = _make_agent(a.model, not a.no_search, 0, cfg_over, pot)
            _, info = play_game(agent, EVAL_SEED0, a.deck, a.stake, explore=False, record=False, verbose=True)
            print(info, dict(agent.stats))
            return
        t = time.time()
        res = evaluate(a.model, a.games, a.workers, not a.no_search, a.deck, a.stake, cfg_over=cfg_over, rcfg=rcfg)
        print(json.dumps(res, indent=1, default=float), f"\n({(time.time() - t) / 60:.1f} min)")


if __name__ == "__main__":
    main()
