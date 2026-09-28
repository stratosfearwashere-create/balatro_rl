"""Tactical layer: which cards to play or discard inside a blind.

Given the hand, the cards left in the deck, the jokers and the hands/discards left, the score of
any play is exactly computable (Game.predict_many); the only uncertainty is what gets drawn next.
TacticalSearch handles that by sampling redraws. Each candidate play/discard (the policy
network's top choices) is tried on N sampled deck orders - the same orders for every candidate -
and the rest of the blind is played out by the network itself; the candidate with the best
average outcome wins. Picking the best action by rollouts of a policy is, given enough samples,
never worse than that policy (the policy-improvement step), so the search can be distilled back
into the network. It never looks at the real deck order or at face-down cards; the remaining
deck's contents are public in Balatro.

Approximations inside rollouts only (real games always use the full rules): cards drawn during a
rollout are not turned face down, The Hook's random discards are ignored, and jokers do not scale
within the blind.

    python -m balatro_rl.tactical bench --policy checkpoints/ppo_gold1_best.pt --stake GOLD --games 40
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import time

import numpy as np

from .env import A_DISC, A_SELECT, BalatroEnv, encode, subset_of
from .sim.cards import sort_hand
from .sim.scoring import Plan


def tactical_actions(obs: dict) -> tuple[dict, dict]:
    """({positions: action} for legal plays, same for legal discards)."""
    plays, discs = {}, {}
    for a in np.flatnonzero(obs["mask"][:A_SELECT]):
        (plays if a < A_DISC else discs)[tuple(subset_of(obs, int(a)))] = int(a)
    return plays, discs


def in_blind(env: BalatroEnv) -> bool:
    return env.obs is not None and env.g.state == "SELECTING_HAND"


def blind_value(cleared: bool, hands_left: int, chips: float, target: float) -> float:
    """What a finished blind is worth: clearing it, then hands left over ($1 each), else closeness."""
    if cleared:
        return 1.0 + 0.03 * hands_left
    return 0.3 * min(1.0, chips / target) if target > 0 else 0.0


class NetTactics:
    """A policy network restricted to play/discard actions (greedy), batched."""

    def __init__(self, path_or_model, device: str = "cpu"):
        import torch
        from .model import load_model, batch_obs
        self.torch, self.batch_obs = torch, batch_obs
        self.model = load_model(path_or_model, device) if isinstance(path_or_model, str) else path_or_model
        self.device = device

    def logits(self, obs_list: list[dict]):
        """Play/discard logits (other actions masked out) as a numpy array."""
        obs2 = []
        for ob in obs_list:
            ob = dict(ob)
            m = ob["mask"].copy()
            m[A_SELECT:] = False
            ob["mask"] = m
            obs2.append(ob)
        with self.torch.no_grad():
            lg, _ = self.model(self.batch_obs(obs2, self.device))
        return lg.float().cpu().numpy()

    def act(self, env: BalatroEnv) -> int:
        return int(self.logits([env.obs])[0].argmax())


class _Roll:
    """One simulated rest-of-blind: its own hand, draw pile and counters."""
    __slots__ = ("hand", "draw", "hl", "dl", "chips", "value")

    def __init__(self, hand, draw, hl, dl, chips):
        self.hand, self.draw, self.hl, self.dl, self.chips = hand, draw, hl, dl, chips
        self.value = None


class TacticalSearch:
    """Overrules the network only when it has to: a play that clears the blind now when the
    network's choice doesn't, or a candidate whose rollouts beat the network's choice by at least
    `margin` cleared blinds in total over the samples (smaller differences are noise)."""

    def __init__(self, net: NetTactics, samples: int = 16, top_k: int = 8, seed: int = 0, margin: float = 1.0):
        self.net = net
        self.samples = samples
        self.top_k = top_k
        self.margin = margin
        self.rng = np.random.default_rng(seed)

    def act(self, env: BalatroEnv) -> int:
        g, obs = env.g, env.obs
        plays, discs = tactical_actions(obs)
        plan = Plan(g)
        need = g.target - g.chips
        lg = self.net.logits([obs])[0]
        legal = list(plays.values()) + list(discs.values())
        cands = sorted(legal, key=lambda a: -lg[a])[:self.top_k]   # cands[0] is the network's choice
        psubs = list(plays)
        pscore = dict(zip(psubs, (p[0] for p in g.predict_many(psubs, plan))))
        winners = {plays[s]: pscore[s] for s in psubs if pscore[s] >= need}
        if winners:                                            # a play that clears the blind now
            return cands[0] if cands[0] in winners else max(winners, key=winners.get)
        if len(cands) == 1:
            return cands[0]
        self._hand_size = g.effective_hand_size()
        pool = list(g.deck)
        orders = [self.rng.permutation(len(pool)) for _ in range(self.samples)]
        rolls, owner = [], []
        for ci, a in enumerate(cands):
            pos = subset_of(obs, a)
            for order in orders:
                r = _Roll(list(g.hand), [pool[i] for i in order], g.hands_left, g.discards_left, g.chips)
                self._apply(g, plan, env, r, a < A_DISC, pos, pscore.get(tuple(pos)))
                rolls.append(r)
                owner.append(ci)
        self._rollout(env, plan, rolls)
        totals = np.zeros(len(cands))
        for ci, r in zip(owner, rolls):
            totals[ci] += r.value
        best = int(totals.argmax())
        return cands[best] if totals[best] - totals[0] >= self.margin else cands[0]

    # ------------------------------------------------------------------ simulation
    def _apply(self, g, plan, env, r: _Roll, is_play: bool, pos, score=None):
        """Carry out a play or discard in a simulated blind (sets r.value when it ends)."""
        if is_play:
            if score is None:
                score = self._score(g, plan, r, pos)
            r.chips += score
            r.hl -= 1
            if r.chips >= g.target:
                r.value = blind_value(True, r.hl, r.chips, g.target)
                return
            if r.hl <= 0:
                r.value = blind_value(False, 0, r.chips, g.target)
                return
        else:
            r.dl -= 1
        keep = [c for i, c in enumerate(r.hand) if i not in pos]
        n = 3 if g.boss_active() == "serpent" else self._hand_size - len(keep)
        new = [r.draw.pop() for _ in range(min(max(0, n), len(r.draw)))]
        r.hand = sort_hand(keep + new)
        if not r.hand:
            r.value = blind_value(False, r.hl, r.chips, g.target)

    def _score(self, g, plan, r: _Roll, pos) -> float:
        saved = (g.hand, g.hands_left, g.discards_left, g.deck)
        g.hand, g.hands_left, g.discards_left, g.deck = r.hand, r.hl, r.dl, r.draw
        try:
            return g.predict_many([tuple(pos)], plan)[0][0]
        finally:
            g.hand, g.hands_left, g.discards_left, g.deck = saved

    def _rollout(self, env, plan, rolls):
        """Play every simulated blind to its end with the network, all in lockstep."""
        g = env.g
        saved = (g.hand, g.hands_left, g.discards_left, g.deck, g.chips)
        try:
            for _ in range(40):                                # a blind never needs more steps
                live = [r for r in rolls if r.value is None]
                if not live:
                    break
                obs_list = []
                for r in live:
                    g.hand, g.hands_left, g.discards_left, g.deck, g.chips = r.hand, r.hl, r.dl, r.draw, r.chips
                    obs_list.append(encode(g, env.cnt))
                lg = self.net.logits(obs_list)
                for r, ob, row in zip(live, obs_list, lg):
                    a = int(row.argmax())
                    self._apply(g, plan, env, r, a < A_DISC, subset_of(ob, a))
            for r in rolls:
                if r.value is None:
                    r.value = blind_value(False, r.hl, r.chips, g.target)
        finally:
            g.hand, g.hands_left, g.discards_left, g.deck, g.chips = saved


# ------------------------------------------------------------------ comparing tactical players
def blind_starts(policy_path: str, stake: str, seeds, device: str = "cpu"):
    """Play games with a checkpoint and copy the environment at the start of every blind."""
    from .evaluate import make_policy
    pol = make_policy(policy_path, device)
    snaps = []
    for seed in seeds:
        env = BalatroEnv("RED", stake)
        obs = env.reset(seed)
        done, prev = False, None
        while not done:
            if env.g.state == "SELECTING_HAND" and prev != "SELECTING_HAND":
                snaps.append(copy.deepcopy(env))
            prev = env.g.state
            obs, _, done, _ = env.step(pol(env, obs))
    return snaps


def play_blind(env: BalatroEnv, tactics) -> dict:
    """Play one blind to its end with a tactical player. Returns whether it was cleared and how."""
    g = env.g
    before = g.blinds_beaten
    target = g.target
    steps = 0
    while in_blind(env):
        _, _, done, _ = env.step(tactics.act(env))
        steps += 1
        if done:
            break
    return {"cleared": g.blinds_beaten > before, "frac": g.chips / target if target else 0.0,
            "steps": steps, "ante": env.g.ante}


def _bench_worker(args):
    import torch
    torch.set_num_threads(1)
    policy, stake, seeds, samples, top_k, device, compare = args
    snaps = blind_starts(policy, stake, seeds)
    net = NetTactics(policy, device)
    players = {"network": net}
    for i, path in enumerate(compare):
        players[f"compare{i + 1}"] = NetTactics(path, device)
    if samples > 0:
        players["search"] = TacticalSearch(net, samples=samples, top_k=top_k, seed=seeds[0])
    out = {name: [] for name in players}
    for snap in snaps:
        for name, p in players.items():
            env = copy.deepcopy(snap)                      # same deck order and randomness for both
            t = time.perf_counter()
            r = play_blind(env, p)
            r["sec"] = time.perf_counter() - t
            out[name].append(r)
    return out


def bench(policy: str, stake: str, games: int, samples: int, top_k: int, procs: int = 8, seed0: int = 20_000,
          device: str = "cuda", compare=()):
    """Blinds come from games played by `policy`; every player then plays exactly the same blinds."""
    chunks = [list(range(seed0 + i, seed0 + games, procs)) for i in range(procs)]
    chunks = [c for c in chunks if c]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(chunks)) as pool:
        parts = pool.map(_bench_worker, [(policy, stake, c, samples, top_k, device, list(compare)) for c in chunks])
    res = {k: sum((p[k] for p in parts), []) for k in parts[0]}
    names = {"network": policy.replace("\\", "/").split("/")[-1], "search": "search"}
    names.update({f"compare{i + 1}": c.replace("\\", "/").split("/")[-1] for i, c in enumerate(compare)})
    base = np.array([r["cleared"] for r in res["network"]])
    print(f"{len(base)} blinds from {games} {stake} games (same draws for every player)")
    for name, rs in res.items():
        cl = np.array([r["cleared"] for r in rs])
        fr = np.array([min(r["frac"], 3.0) for r in rs])
        sec = sum(r["sec"] for r in rs) / max(1, sum(r["steps"] for r in rs))
        vs = "" if name == "network" else \
            f"   vs network: +{int((cl & ~base).sum())} / -{int((base & ~cl).sum())} blinds"
        print(f"  {names[name]:24s} cleared {100 * cl.mean():5.1f}%   median score/target {np.median(fr):.2f}   "
              f"{1e3 * sec:6.1f} ms/decision{vs}")
    by_ante = {}
    for i, r in enumerate(res["network"]):
        a = by_ante.setdefault(r["ante"], {k: 0 for k in ["n"] + list(res)})
        a["n"] += 1
        for k in res:
            a[k] += res[k][i]["cleared"]
    print("  cleared by ante:", {a: " / ".join(str(v[k]) for k in ["n"] + list(res)) for a, v in sorted(by_ante.items())},
          f"(blinds / {' / '.join(names[k] for k in res)})")
    return res


# ------------------------------------------------------------------ distillation
def _gen_worker(args):
    """Play games: the search makes the card decisions, the network (sampling) everything else.
    Saves every position with the label distillation needs."""
    import torch
    from .model import load_model, batch_obs
    from .train import compact
    torch.set_num_threads(1)
    policy, stake, seeds, samples, top_k, device, out_path = args
    model = load_model(policy, device)
    net = NetTactics(model, device)
    search = TacticalSearch(net, samples=samples, top_k=top_k, seed=seeds[0] * 7 + 1)
    obs_rows, act, logp, value = [], [], [], []
    for seed in seeds:
        env = BalatroEnv("RED", stake)
        obs = env.reset(seed)
        done = False
        while not done:
            with torch.no_grad():
                lg, v = model(batch_obs([obs], device))
            lp = torch.log_softmax(lg[0].float(), -1)
            if in_blind(env):
                a = search.act(env)
                act.append(a)
            else:
                a = int(torch.distributions.Categorical(logits=lg[0]).sample())
                act.append(-1)
            obs_rows.append(compact(obs))
            logp.append(lp.cpu().numpy().astype(np.float32))
            value.append(float(v[0]))
            obs, _, done, _ = env.step(a)
    data = {k: np.stack([o[k] for o in obs_rows]) for k in obs_rows[0]}
    np.savez(out_path, act=np.array(act, dtype=np.int64), logp=np.stack(logp), value=np.array(value, np.float32),
             **{"obs_" + k: v for k, v in data.items()})
    return out_path, len(act), int((np.array(act) >= 0).sum())


def generate(policy, stake, games, samples, top_k, out_dir, procs=8, seed0=500_000, device="cuda"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    chunks = [list(range(seed0 + i, seed0 + games, procs)) for i in range(procs)]
    jobs = [(policy, stake, c, samples, top_k, device, os.path.join(out_dir, f"shard{i}.npz"))
            for i, c in enumerate(chunks) if c]
    t = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        done = pool.map(_gen_worker, jobs)
    n, nt = sum(d[1] for d in done), sum(d[2] for d in done)
    print(f"{n} positions ({nt} card decisions labelled by the search) from {games} games "
          f"in {(time.time() - t) / 60:.1f} min -> {out_dir}")


def distill(init, data_dir, out, epochs=4, lr=1e-4, batch=256, kl_weight=1.0, v_weight=0.1, device="cuda",
            override_weight=1.0):
    """Train the card-play outputs to copy the search, keeping everything else as it was:
    cross-entropy to the search's choice on card decisions (decisions where the search overruled
    the original network count `override_weight` times), KL to the old policy elsewhere, and the
    value head held to its old predictions."""
    import glob
    import torch
    import torch.nn.functional as F
    from .model import load_model, save_model, OBS_KEYS
    shards = [np.load(f) for f in sorted(glob.glob(f"{data_dir}/shard*.npz"))]
    D = {k: np.concatenate([s[k] for s in shards]) for k in shards[0].files}
    N = len(D["act"])
    tac = D["act"] >= 0
    overruled = tac & (D["act"] != D["logp"].argmax(-1))
    D["w"] = np.where(overruled, override_weight, 1.0).astype(np.float32)
    print(f"{N} positions, {int(tac.sum())} card decisions, {int(overruled.sum())} where the search overruled")
    model = load_model(init, device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        idx = rng.permutation(N)
        tot = {"ce": [], "acc": [], "kl": [], "v": []}
        for k in range(0, N, batch):
            mb = idx[k:k + batch]
            ob = {key: torch.from_numpy(D["obs_" + key][mb]).to(device) for key in OBS_KEYS}
            logits, v = model(ob)
            logp = torch.log_softmax(logits.float(), -1)
            act = torch.from_numpy(D["act"][mb]).to(device)
            t = act >= 0
            loss = torch.zeros((), device=device)
            if t.any():
                w = torch.from_numpy(D["w"][mb]).to(device)[t]
                ce = (F.nll_loss(logp[t], act[t], reduction="none") * w).sum() / w.sum()
                loss = loss + ce
                tot["ce"].append(ce.item())
                tot["acc"].append((logp[t].argmax(-1) == act[t]).float().mean().item())
            if (~t).any():
                lp0 = torch.from_numpy(D["logp"][mb]).to(device)[~t]
                kl = (lp0.exp() * (lp0 - logp[~t])).sum(-1).mean()
                loss = loss + kl_weight * kl
                tot["kl"].append(kl.item())
            vl = F.mse_loss(v, torch.from_numpy(D["value"][mb]).to(device))
            loss = loss + v_weight * vl
            tot["v"].append(vl.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f"epoch {ep + 1}: card-play loss {np.mean(tot['ce']):.4f}, matches search {100 * np.mean(tot['acc']):.1f}%, "
              f"drift elsewhere (KL) {np.mean(tot['kl']):.4f}, value drift {np.mean(tot['v']):.4f}", flush=True)
    save_model(model, out, {"mode": "distill", "from": init, "data": data_dir})
    print("saved", out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="compare the search with a network's own card play on real blinds")
    b.add_argument("--policy", required=True)
    b.add_argument("--stake", default="GOLD")
    b.add_argument("--games", type=int, default=40)
    b.add_argument("--samples", type=int, default=16)
    b.add_argument("--top-k", type=int, default=8)
    b.add_argument("--procs", type=int, default=8)
    b.add_argument("--device", default="cuda")
    b.add_argument("--compare", nargs="*", default=[], help="other checkpoints to play the same blinds")
    gn = sub.add_parser("gen", help="play games with the search to make distillation data")
    gn.add_argument("--policy", required=True)
    gn.add_argument("--stake", default="GOLD")
    gn.add_argument("--games", type=int, default=400)
    gn.add_argument("--samples", type=int, default=16)
    gn.add_argument("--top-k", type=int, default=8)
    gn.add_argument("--procs", type=int, default=8)
    gn.add_argument("--out", default="checkpoints/distill_data")
    ds = sub.add_parser("distill", help="train a checkpoint's card play to copy the search")
    ds.add_argument("--init", required=True)
    ds.add_argument("--data", default="checkpoints/distill_data")
    ds.add_argument("--out", required=True)
    ds.add_argument("--epochs", type=int, default=4)
    ds.add_argument("--lr", type=float, default=1e-4)
    ds.add_argument("--override-weight", type=float, default=1.0,
                    help="weight of card decisions where the search overruled the original network")
    a = p.parse_args()
    if a.cmd == "bench":
        bench(a.policy, a.stake, a.games, a.samples, a.top_k, a.procs, device=a.device, compare=a.compare)
    elif a.cmd == "gen":
        generate(a.policy, a.stake, a.games, a.samples, a.top_k, a.out, a.procs)
    elif a.cmd == "distill":
        distill(a.init, a.data, a.out, a.epochs, a.lr, override_weight=a.override_weight)


if __name__ == "__main__":
    main()
