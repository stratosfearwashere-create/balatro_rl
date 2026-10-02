"""Search at strategic decisions, and expert iteration (search -> train the network to copy it -> repeat).

At every strategic decision with more than one legal action (blind select, shop, packs, blind start), the
policy's top-k actions are each tried on N re-drawn futures. A future is a copy of the game with a fresh
random number generator, and inside a blind also a reshuffled draw pile and fresh face-down cards, so
the search never sees the real seed's future. Each copy is played on by the current policy (greedy), with
the frozen tactical network playing the cards, to the end of the run (or `horizon` more decisions, then
the critic's estimate). Every candidate gets the same N futures (common random numbers), so differences
between candidates come mostly from the action itself. The search overrules the network only when the
best candidate beats the network's choice by more than `z` standard errors of the paired difference.

Expert iteration: games played with the search record, at each searched decision, the candidates and
their average outcomes. The network is trained towards the search's choices (KL to its old policy
elsewhere, the critic towards the returns the search achieved), then the search runs again with the
new network, and so on.

    python -m balatro_rl.shopsearch bench --policy checkpoints/baseline_strategic.pt --games 100
    python -m balatro_rl.shopsearch iterate --policy checkpoints/baseline_strategic.pt --iters 3 --out checkpoints/exit
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np

TACTICAL = "checkpoints/ppo_gold1_best.pt"
# the reward baseline_strategic.pt was trained with, so its critic's estimates are on the same scale
REWARD = {"ante_weight": 3.0, "win_bonus": 50.0, "margin": 0.2}
EVAL_SEED0 = 10_000
GEN_SEED0 = 700_000


# ------------------------------------------------------------------ environments
def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_env(tactical, stake: str = "GOLD", reward: dict | None = None, dev: str = "cpu"):
    """A strategic environment; `tactical` is a checkpoint path or an already loaded model."""
    from .strategic import StrategicEnv
    return StrategicEnv("RED", stake, tactical=tactical, device=dev, **(reward or REWARD))


def redraw(g):
    """Inside a blind: reshuffle the draw pile and deal new face-down cards from it, using g.rng."""
    hidden = [i for i, c in enumerate(g.hand) if c.hidden]
    pool = list(g.deck) + [g.hand[i] for i in hidden]
    for c in pool:
        c.hidden = False
    g.rng.shuffle(pool)
    for i in hidden:
        c = pool.pop()
        c.hidden = True
        g.hand[i] = c
    g.deck = pool


def fork(senv, seed: int):
    """An independent copy of a strategic environment with a re-drawn future."""
    new = copy.copy(senv)                      # shares the tactical network
    new.env = copy.deepcopy(senv.env)
    g = new.env.g
    g.rng = random.Random(int(seed))
    if g.state == "SELECTING_HAND":
        redraw(g)
    return new


class Net:
    """The strategic policy/critic, batched, on the CPU of a worker process."""

    def __init__(self, path_or_model, device: str = "cpu"):
        import torch
        from .model import load_model, batch_obs
        self.torch, self.batch_obs, self.device = torch, batch_obs, device
        self.model = load_model(path_or_model, device) if isinstance(path_or_model, str) else path_or_model

    def __call__(self, obs_list):
        with self.torch.no_grad():
            lg, v = self.model(self.batch_obs(obs_list, self.device))
        return lg.float().cpu().numpy(), v.float().cpu().numpy()


def playout(net: Net, envs, first, horizon: int = 0):
    """Take `first[i]` in `envs[i]`, then play on greedily with the policy. Returns (return, blinds
    reached, won) per env; truncated playouts add the critic's estimate.

    Same result as calling StrategicEnv.step on each env in turn, but all envs advance in lockstep so
    that the strategic network and the tactical network each see one batch per round instead of one
    position at a time (the batching is where the time goes)."""
    from .env import A_SELECT
    from .tactical import in_blind
    n = len(envs)
    tac = envs[0].tactics
    ret, blinds, won = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    done = np.zeros(n, bool)
    beaten = [0] * n
    obs = [None] * n
    act = {i: int(a) for i, a in enumerate(first)}
    steps = 0

    def base_step(i, a):
        o, r, d, info = envs[i].env.step(a)
        ret[i] += r
        blinds[i], won[i] = info["blinds"], info["won"]
        done[i] = d

    def tactical_turn(i):
        e = envs[i]
        return not done[i] and e.handed_over and in_blind(e.env)

    while act:
        fin, tac_live = [], []
        for i, a in act.items():                          # the strategic action (StrategicEnv.step's first half)
            e = envs[i]
            g = e.env.g
            beaten[i] = g.blinds_beaten
            was_targeting = g.targeting is not None
            base_step(i, a)
            e._track_blind()
            if a < A_SELECT and not was_targeting:
                e.handed_over = True
            (tac_live if tactical_turn(i) else fin).append(i)
        while tac_live:                                   # the tactical network plays the blinds out
            lg = tac.logits([envs[i].env.obs for i in tac_live])
            nxt = []
            for i, row in zip(tac_live, lg):
                base_step(i, int(row.argmax()))
                (nxt if tactical_turn(i) else fin).append(i)
            tac_live = nxt
        need = []
        for i in fin:                                     # margin bonus, then the next decision's observation
            e = envs[i]
            g = e.env.g
            if g.blinds_beaten > beaten[i] and g.target > 0:
                ret[i] += e.margin * min(1.0, max(0.0, g.chips / g.target - 1.0))
            if done[i]:
                continue
            if in_blind(e.env) and g.targeting is None:
                e._track_blind()
                need.append(i)
            else:
                obs[i] = e.env.obs
        if need:                                          # blind start: the tactical suggestion is the only card action
            lg = tac.logits([envs[i].env.obs for i in need])
            for i, row in zip(need, lg):
                o = dict(envs[i].env.obs)
                m = o["mask"].copy()
                m[:A_SELECT] = False
                m[int(row.argmax())] = True
                o["mask"] = m
                obs[i] = o
        steps += 1
        live = [i for i in fin if not done[i]]
        if not live:
            break
        if horizon > 0 and steps >= horizon:
            _, v = net([obs[i] for i in live])
            ret[live] += v
            break
        lg, _ = net([obs[i] for i in live])
        act = {i: int(a) for i, a in zip(live, lg.argmax(-1))}
    return ret, blinds, won


class StrategicSearch:
    def __init__(self, net: Net, samples: int = 16, top_k: int = 6, z: float = 2.0, horizon: int = 0,
                 objective: str = "return", seed: int = 0):
        self.net, self.samples, self.top_k, self.z, self.horizon = net, samples, top_k, z, horizon
        self.objective = objective
        self.rng = np.random.default_rng(seed)

    def act(self, senv, obs):
        """(action, record); record is None when there was nothing to search."""
        legal = np.flatnonzero(obs["mask"])
        lg, _ = self.net([obs])
        prior = lg[0]
        if len(legal) == 1:
            return int(legal[0]), None
        cands = legal[np.argsort(-prior[legal], kind="stable")][:self.top_k]     # cands[0]: the network's choice
        seeds = self.rng.integers(0, 2 ** 31, self.samples)
        envs = [fork(senv, s) for _ in cands for s in seeds]
        first = [a for a in cands for _ in seeds]
        ret, blinds, _ = playout(self.net, envs, first, self.horizon)
        val = (ret if self.objective == "return" else blinds).reshape(len(cands), self.samples)
        q = val.mean(1)
        diff = val - val[0]
        se = diff.std(1, ddof=1) / math.sqrt(self.samples)
        best = int(q.argmax())
        pick = best if best != 0 and diff[best].mean() > self.z * se[best] else 0
        return int(cands[pick]), {"cands": cands.astype(np.int64), "q": q.astype(np.float32),
                                  "picked": pick, "best": best}


# ------------------------------------------------------------------ playing games
def play_game(senv, seed: int, choose, record: bool = False):
    """Play one game; choose(senv, obs) -> (action, search record or None)."""
    obs = senv.reset(seed)
    done, steps, rewards, recs = False, 0, [], []
    info = {"blinds": 0, "won": False}
    while not done:
        a, rec = choose(senv, obs)
        if record and rec is not None:
            rec = dict(rec, obs=obs, t=steps)
            recs.append(rec)
        obs, r, done, info = senv.step(a)
        rewards.append(r)
        steps += 1
    return {"seed": seed, "blinds": int(info["blinds"]), "won": bool(info["won"]), "ante": senv.g.ante,
            "decisions": steps, "searched": sum(1 for x in recs) if record else None,
            "return": float(sum(rewards))}, rewards, recs


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _bench_worker(args):
    _init_worker()
    policy, tactical, stake, seeds, kw = args
    dev = device()
    net = Net(policy, dev)
    senv = make_env(tactical, stake, dev=dev)
    search = StrategicSearch(net, seed=seeds[0], **kw)
    greedy = lambda e, o: (int(net([o])[0][0].argmax()), None)
    out = []
    for seed in seeds:
        plain, _, _ = play_game(senv, seed, greedy)
        t = time.perf_counter()
        stats = {"n": 0, "over": 0}

        def choose(e, o):
            a, rec = search.act(e, o)
            if rec is not None:
                stats["n"] += 1
                stats["over"] += rec["picked"] != 0
            return a, rec
        res, _, _ = play_game(senv, seed, choose)
        out.append({"seed": seed, "plain": plain["blinds"], "search": res["blinds"], "plain_won": plain["won"],
                    "search_won": res["won"], "searched": stats["n"], "overruled": stats["over"],
                    "sec": time.perf_counter() - t})
        print(f"  seed {seed}: network {plain['blinds']:2d}  search {res['blinds']:2d}  "
              f"({stats['over']}/{stats['n']} overruled, {out[-1]['sec']:.0f}s)", flush=True)
    return out


def boot_ci(x, reps=4000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def bench(policy, tactical, stake, games, procs, kw, seed0=EVAL_SEED0, out=None):
    seeds = list(range(seed0, seed0 + games))
    chunks = [seeds[i::procs] for i in range(procs) if seeds[i::procs]]
    t = time.time()
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        parts = pool.map(_bench_worker, [(policy, tactical, stake, c, kw) for c in chunks])
    res = sorted((r for p in parts for r in p), key=lambda r: r["seed"])
    p = np.array([r["plain"] for r in res])
    s = np.array([r["search"] for r in res])
    print(f"\n{len(res)} {stake} games, same seeds for both ({(time.time() - t) / 60:.0f} min; search settings {kw})")
    print("  network alone  blinds %.2f [%.2f, %.2f]" % boot_ci(p), f" wins {sum(r['plain_won'] for r in res)}")
    print("  with search    blinds %.2f [%.2f, %.2f]" % boot_ci(s), f" wins {sum(r['search_won'] for r in res)}")
    print("  difference     %+.2f [%+.2f, %+.2f]" % boot_ci(s - p),
          f" (search better in {int((s > p).sum())} games, worse in {int((s < p).sum())})")
    n, o = sum(r["searched"] for r in res), sum(r["overruled"] for r in res)
    print(f"  searched decisions {n}, overruled the network {o} ({100 * o / max(1, n):.0f}%), "
          f"{np.mean([r['sec'] for r in res]):.0f} s per searched game per process")
    if out:
        with open(out, "w") as f:
            json.dump({"settings": kw, "policy": policy, "games": res}, f)
    return res


# ------------------------------------------------------------------ expert iteration
def _gen_worker(args):
    _init_worker()
    from .train import compact
    policy, tactical, stake, seeds, kw, out_path, gamma = args
    dev = device()
    net = Net(policy, dev)
    senv = make_env(tactical, stake, dev=dev)
    search = StrategicSearch(net, seed=seeds[0] * 7 + 3, **kw)
    rows, cands, q, picked, logp, rtg, games = [], [], [], [], [], [], []
    k = kw.get("top_k", 6)
    for seed in seeds:
        res, rewards, recs = play_game(senv, seed, search.act, record=True)
        games.append(res)
        g_ret = np.zeros(len(rewards) + 1)
        for t in range(len(rewards) - 1, -1, -1):
            g_ret[t] = rewards[t] + gamma * g_ret[t + 1]
        for rec in recs:
            lg, _ = net([rec["obs"]])
            lp = lg[0] - lg[0].max()
            lp = lp - np.log(np.exp(lp).sum())
            rows.append(compact(rec["obs"]))
            c = np.full(k, -1, np.int64)
            c[:len(rec["cands"])] = rec["cands"]
            qq = np.full(k, np.nan, np.float32)
            qq[:len(rec["q"])] = rec["q"]
            cands.append(c)
            q.append(qq)
            picked.append(rec["picked"])
            logp.append(lp.astype(np.float32))
            rtg.append(g_ret[rec["t"]])
    data = {key: np.stack([o[key] for o in rows]) for key in rows[0]} if rows else {}
    np.savez(out_path, cands=np.array(cands), q=np.array(q), picked=np.array(picked, np.int64),
             logp=np.stack(logp) if logp else np.zeros((0,)), rtg=np.array(rtg, np.float32),
             **{"obs_" + key: v for key, v in data.items()})
    return games, len(rows)


def generate(policy, tactical, stake, games, procs, kw, out_dir, seed0=GEN_SEED0, gamma=0.995):
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(range(seed0, seed0 + games))
    jobs = [(policy, tactical, stake, seeds[i::procs], kw, os.path.join(out_dir, f"shard{i}.npz"), gamma)
            for i in range(procs) if seeds[i::procs]]
    t = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        done = pool.map(_gen_worker, jobs)
    games_ = [g for d in done for g in d[0]]
    n = sum(d[1] for d in done)
    b = boot_ci([g["blinds"] for g in games_])
    print(f"{n} searched decisions from {len(games_)} games in {(time.time() - t) / 60:.0f} min; "
          f"blinds with search %.2f [%.2f, %.2f] -> {out_dir}" % b, flush=True)
    return games_


def distill(init, data_dirs, out, epochs=3, lr=1e-4, batch=256, tau=1.0, target="soft", kl_weight=0.5,
            v_weight=0.5, device="cuda"):
    """Train the policy towards the search: cross-entropy to the search's target over the candidates
    (soft: softmax of their average outcomes / tau; hard: the action the search played), KL to the
    old policy over all actions, and the critic towards the returns-to-go of the searched games."""
    import torch
    import torch.nn.functional as F
    from .model import load_model, save_model, OBS_KEYS
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    shards = [np.load(f) for d in data_dirs for f in sorted(glob.glob(f"{d}/shard*.npz"))]
    shards = [s for s in shards if len(s["picked"])]
    D = {k: np.concatenate([s[k] for s in shards]) for k in shards[0].files}
    N = len(D["picked"])
    cands, q = D["cands"], D["q"]
    valid = cands >= 0
    if target == "soft":
        z = np.where(valid, np.nan_to_num(q, nan=-1e9), -1e9) / tau
        z = z - z.max(1, keepdims=True)
        tw = np.exp(z) * valid
        tw = tw / tw.sum(1, keepdims=True)
    else:
        tw = np.zeros_like(q)
        tw[np.arange(N), D["picked"]] = 1.0
    tw = tw.astype(np.float32)
    old_choice = D["logp"].argmax(-1)
    differs = cands[np.arange(N), tw.argmax(1)] != old_choice
    print(f"{N} searched decisions; the search's favourite differs from the old policy's in "
          f"{100 * differs.mean():.0f}%; played an override in {100 * (D['picked'] != 0).mean():.0f}%")
    model = load_model(init, device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    ci = torch.from_numpy(np.maximum(cands, 0)).to(device)
    tw_t = torch.from_numpy(tw).to(device)
    for ep in range(epochs):
        idx = rng.permutation(N)
        tot = {"ce": [], "acc": [], "kl": [], "v": []}
        for k in range(0, N, batch):
            mb = idx[k:k + batch]
            ob = {key: torch.from_numpy(D["obs_" + key][mb]).to(device) for key in OBS_KEYS}
            logits, v = model(ob)
            logp = torch.log_softmax(logits.float(), -1)
            mbt = torch.from_numpy(mb).to(device)
            lp_c = torch.gather(logp, 1, ci[mbt])
            t = tw_t[mbt]
            ce = -(t * lp_c).sum(1).mean()
            lp0 = torch.from_numpy(D["logp"][mb]).to(device)
            kl = (lp0.exp() * (lp0 - logp)).sum(-1).mean()
            vl = F.mse_loss(v, torch.from_numpy(D["rtg"][mb]).to(device))
            loss = ce + kl_weight * kl + v_weight * vl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot["ce"].append(ce.item())
            tot["acc"].append((lp_c.argmax(1) == t.argmax(1)).float().mean().item())
            tot["kl"].append(kl.item())
            tot["v"].append(vl.item())
        print(f"  epoch {ep + 1}: target loss {np.mean(tot['ce']):.3f}, matches the search's favourite "
              f"{100 * np.mean(tot['acc']):.0f}%, KL to old {np.mean(tot['kl']):.4f}, value loss {np.mean(tot['v']):.2f}",
              flush=True)
    save_model(model, out, {"mode": "expert_iteration", "from": init, "data": list(data_dirs)})
    print("  saved", out)


def _plain_worker(args):
    _init_worker()
    policies, tactical, stake, seeds = args
    dev = device()
    nets = [Net(p, dev) for p in policies]
    senv = make_env(tactical, stake, dev=dev)
    out = []
    for seed in seeds:
        row = []
        for net in nets:
            res, _, _ = play_game(senv, seed, lambda e, o, net=net: (int(net([o])[0][0].argmax()), None))
            row.append((res["blinds"], res["won"]))
        out.append(row)
    return out


def compare_plain(policies, tactical, stake, games, procs, seed0=EVAL_SEED0):
    """The networks alone (no search), greedy, on the same seeds."""
    seeds = list(range(seed0, seed0 + games))
    jobs = [(policies, tactical, stake, seeds[i::procs]) for i in range(procs) if seeds[i::procs]]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rows = [r for part in pool.map(_plain_worker, jobs) for r in part]
    B = np.array([[b for b, _ in r] for r in rows], float)
    W = np.array([[w for _, w in r] for r in rows])
    for j, p in enumerate(policies):
        extra = "" if j == 0 else "   vs first: %+.2f [%+.2f, %+.2f]" % boot_ci(B[:, j] - B[:, 0])
        print(f"  {os.path.basename(p):28s} blinds %.2f [%.2f, %.2f]" % boot_ci(B[:, j]),
              f" wins {int(W[:, j].sum())}{extra}", flush=True)
    return B


def iterate(policy, tactical, stake, iters, games, procs, kw, out, eval_games, **dkw):
    os.makedirs(out, exist_ok=True)
    chain = [policy]
    data = []
    for it in range(1, iters + 1):
        print(f"\n=== iteration {it}: search games with {chain[-1]}", flush=True)
        d = os.path.join(out, f"data{it}")
        generate(chain[-1], tactical, stake, games, procs, kw, d, seed0=GEN_SEED0 + 100_000 * it)
        data.append(d)
        new = os.path.join(out, f"iter{it}.pt")
        print(f"=== iteration {it}: training {new}", flush=True)
        distill(chain[-1], data[-2:], new, **dkw)                  # this and the previous round's data
        chain.append(new)
        print(f"=== iteration {it}: networks alone on {eval_games} evaluation games", flush=True)
        compare_plain(chain, tactical, stake, eval_games, procs)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, games):
        s.add_argument("--policy", required=True)
        s.add_argument("--tactical", default=TACTICAL)
        s.add_argument("--stake", default="GOLD")
        s.add_argument("--games", type=int, default=games)
        s.add_argument("--procs", type=int, default=8)
        s.add_argument("--samples", type=int, default=16)
        s.add_argument("--top-k", type=int, default=6)
        s.add_argument("--z", type=float, default=2.0, help="overrule the network only by this many standard errors")
        s.add_argument("--horizon", type=int, default=0, help="decisions per playout (0 = to the end of the run)")
        s.add_argument("--objective", default="return", choices=["return", "blinds"])

    b = sub.add_parser("bench", help="how many blinds the search clears, against the network alone")
    common(b, 100)
    b.add_argument("--out", default=None, help="save per-game results as JSON")
    it = sub.add_parser("iterate", help="expert iteration: search games -> train -> repeat")
    common(it, 400)
    it.add_argument("--iters", type=int, default=3)
    it.add_argument("--out", required=True, help="folder for data and checkpoints")
    it.add_argument("--eval-games", type=int, default=1000)
    it.add_argument("--epochs", type=int, default=3)
    it.add_argument("--lr", type=float, default=1e-4)
    it.add_argument("--tau", type=float, default=1.0)
    it.add_argument("--target", default="soft", choices=["soft", "hard"])
    it.add_argument("--kl-weight", type=float, default=0.5)
    cp = sub.add_parser("compare", help="networks alone (no search) on the same evaluation seeds")
    cp.add_argument("policies", nargs="+")
    cp.add_argument("--tactical", default=TACTICAL)
    cp.add_argument("--stake", default="GOLD")
    cp.add_argument("--games", type=int, default=1000)
    cp.add_argument("--procs", type=int, default=8)
    a = p.parse_args()
    if a.cmd == "compare":
        compare_plain(a.policies, a.tactical, a.stake, a.games, a.procs)
        return
    kw = {"samples": a.samples, "top_k": a.top_k, "z": a.z, "horizon": a.horizon, "objective": a.objective}
    if a.cmd == "bench":
        bench(a.policy, a.tactical, a.stake, a.games, a.procs, kw, out=a.out)
    else:
        iterate(a.policy, a.tactical, a.stake, a.iters, a.games, a.procs, kw, a.out, a.eval_games,
                epochs=a.epochs, lr=a.lr, tau=a.tau, target=a.target, kl_weight=a.kl_weight)


if __name__ == "__main__":
    main()
