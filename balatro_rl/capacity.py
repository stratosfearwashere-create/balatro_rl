"""Build strength ("capacity"): how much a build scores in one blind, learned from random builds.

gen    random builds (ante, jokers with editions and stickers, planet levels, enhanced / edition / sealed
       cards, suit-converted, added and removed cards, money), each played through a boss-free blind by the
       frozen tactical network `draws` times with different shuffles. The target is 4x the ante's base (the
       size of The Wall, so the network is in familiar territory). Label: mean log10 of the chips scored,
       capped at the target, so "clears 2x a normal boss" is the ceiling.
train  a small network from build features (independent of which screen the game is on) to that label.
check  in real games played by a strategic policy: does predicted capacity at the start of an ante tell
       whether the run clears that ante's boss? And how close is it to a fresh measurement of the same
       real build (real builds are not random, so this is where a model trained on random ones can fail)?
shop   the idea on its own: the network plays, but a shop or pack action that raises predicted capacity
       by more than `threshold` (log10 units) overrules it.

Known gaps: scaling jokers start at their initial value (a grown Hologram looks like a new one), and money,
interest and economy jokers are worth nothing here. Those are the policy's job, not the build model's.

    python -m balatro_rl.capacity gen --builds 200000 --out checkpoints/capacity_data
    python -m balatro_rl.capacity train --data checkpoints/capacity_data --out checkpoints/capacity.pt
    python -m balatro_rl.capacity check --model checkpoints/capacity.pt --policy checkpoints/baseline_strategic.pt
    python -m balatro_rl.capacity shop --model checkpoints/capacity.pt --policy checkpoints/baseline_strategic.pt
"""
from __future__ import annotations

import argparse
import copy
import glob
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np

from .env import BalatroEnv, Counters, encode, encode_joker_feats, blind_weights, F_JOKER, VOCAB_SIZE
from .env import A_BUY, A_BUY_PACK, A_SELL_J, A_USE_C, A_PICK, A_PSKIP, A_SWAP, N_ACTIONS, MAX_SHOP
from .sim.cards import ENHANCEMENTS, EDITIONS, SEALS, Card
from .sim.game import Game, MAX_JOKERS
from .sim.hands import N_HANDS, HC, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS
from .sim.items import BLIND_BASE, VOUCHERS, item_id

TACTICAL = "checkpoints/ppo_gold1_best.pt"
TARGET_X = 4                     # blind target = 4x the ante's base
COMMON_HANDS = [HC, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS]
VOUCHER_KEYS = list(VOUCHERS)
F_POS = MAX_JOKERS


# ------------------------------------------------------------------ features
def build_features(g: Game):
    """(global vector, joker ids [8], joker features [8, F_JOKER + position]) for the current build."""
    f = [float(g.ante == a) for a in range(1, 9)] + [min(g.ante, 8) / 8]
    f += [lv / 10 for lv in g.hand_levels] + [math.log1p(max(lv - 1, 0)) / 3 for lv in g.hand_levels]
    rc, sc = [0] * 13, [0] * 4
    ec, edc, slc = [0] * len(ENHANCEMENTS), [0] * len(EDITIONS), [0] * len(SEALS)
    extra = 0
    for c in g.full_deck:
        if not c.is_stone:
            rc[c.rank - 2] += 1
            sc[c.suit] += 1
        if c.enh in ENHANCEMENTS:
            ec[ENHANCEMENTS.index(c.enh)] += 1
        if c.edition in EDITIONS:
            edc[EDITIONS.index(c.edition)] += 1
        slc[SEALS.index(c.seal)] += 1
        extra += c.extra_chips
    n = max(1, len(g.full_deck))
    f += [x / 4 for x in rc] + [x / 13 for x in sc] + [max(sc) / n]
    f += [x / 10 for x in ec[1:]] + [x / 5 for x in edc[1:]] + [x / 5 for x in slc[1:]]
    f += [len(g.full_deck) / 52, extra / 100]
    saved = g.state
    g.state = "BLIND_SELECT"                               # no boss effects in the hand size
    f += [g.effective_hand_size() / 8, g.round_hands() / 5, g.round_discards() / 5]
    g.state = saved
    f += [min(max(g.money, 0), 100) / 50, math.log1p(max(g.money, 0)) / 4]
    f += [g.joker_slots / 7, len(g.jokers) / 7]
    f += [float(v in g.vouchers) for v in VOUCHER_KEYS]
    jid = np.zeros(MAX_JOKERS, np.int64)
    jf = np.zeros((MAX_JOKERS, F_JOKER + F_POS), np.float32)
    for i, j in enumerate(g.jokers[:MAX_JOKERS]):
        jj = copy.copy(j)
        jj.hidden = False
        jj.debuffed = False
        jid[i] = item_id(f"j_{j.key}")
        jf[i, :F_JOKER] = encode_joker_feats(jj)
        jf[i, F_JOKER + i] = 1.0
    return np.asarray(f, np.float32), jid, jf


def g_size() -> int:
    return len(build_features(Game(seed=0))[0])


# ------------------------------------------------------------------ random builds and measuring them
def random_build(rng: random.Random, seed: int) -> Game:
    g = Game(seed=seed, stake="GOLD")
    g.ante = rng.choices(range(1, 9), weights=[3, 3, 3, 3, 2, 2, 1, 1])[0]
    g.blind_idx = 0
    g.new_ante()
    n_j = min(g.joker_slots, max(0, int(rng.gauss(1 + 0.6 * g.ante, 1.5))))
    for _ in range(n_j):
        g.add_joker(g.random_joker())
    fav = rng.choice(COMMON_HANDS)
    for _ in range(rng.randint(0, 2 * g.ante)):
        h = fav if rng.random() < 0.7 else rng.choice(COMMON_HANDS)
        g.hand_levels[h] += 1
    if rng.random() < 0.3:                                 # flush-style deck: convert cards to one suit
        s = rng.randrange(4)
        for c in rng.sample(g.full_deck, rng.randint(3, 20)):
            c.suit = s
    for _ in range(rng.randint(0, 3 * g.ante)):
        if not g.full_deck:
            break
        c = rng.choice(g.full_deck)
        r = rng.random()
        if r < 0.55:
            c.enh = rng.choice(ENHANCEMENTS[1:])
        elif r < 0.68:
            c.edition = rng.choice(["FOIL", "HOLO", "POLYCHROME"])
        elif r < 0.8:
            c.seal = rng.choice(SEALS[1:])
        elif r < 0.9 and len(g.full_deck) > 30:
            g.full_deck.remove(c)
        else:
            g.full_deck.append(Card(c.rank, c.suit, enh=c.enh, edition=c.edition, seal=c.seal))
    g.money = rng.randint(0, 40)
    return g


def measure(g: Game, tactics, draws: int, rng: random.Random):
    """Mean log10 chips (capped at the target) over `draws` plays of a boss-free blind, and the share cleared."""
    ys, cleared = [], 0
    for _ in range(draws):
        h = copy.deepcopy(g)
        h.rng = random.Random(rng.getrandbits(32))
        h.state, h.blind_idx = "BLIND_SELECT", 0
        h.select_blind()
        h.target = TARGET_X * BLIND_BASE[h.scaling()][min(h.ante, 8) - 1]
        env = BalatroEnv.__new__(BalatroEnv)
        env.g, env.steps, env.cnt = h, 0, Counters()
        env.weights, env.win_bonus = blind_weights(), 0.0
        env.shape_chips = env.shape_phi = 0.0                  # no reward shaping (BalatroEnv.step reads these)
        env.obs = encode(h, env.cnt)
        best, target, before = 0.0, h.target, h.blinds_beaten
        while env.obs is not None and h.state == "SELECTING_HAND":
            env.step(tactics.act(env))
            best = max(best, h.chips)
        won = h.blinds_beaten > before
        cleared += won
        ys.append(math.log10(1 + (target if won else min(best, target))))
    return float(np.mean(ys)), cleared / draws


def _gen_worker(args):
    import torch
    torch.set_num_threads(1)
    from .tactical import NetTactics
    tactical, seeds, draws, out_path = args
    tactics = NetTactics(tactical)
    G, JI, JF, Y, C, A = [], [], [], [], [], []
    for s in seeds:
        rng = random.Random(s)
        g = random_build(rng, s)
        gv, jid, jf = build_features(g)
        y, c = measure(g, tactics, draws, rng)
        G.append(gv), JI.append(jid), JF.append(jf), Y.append(y), C.append(c), A.append(g.ante)
    np.savez(out_path, g=np.stack(G), jid=np.stack(JI), jf=np.stack(JF), y=np.array(Y, np.float32),
             cleared=np.array(C, np.float32), ante=np.array(A, np.int64))
    return len(seeds)


def generate(builds, out, draws=4, procs=8, tactical=TACTICAL, seed0=0, chunk=2000):
    os.makedirs(out, exist_ok=True)
    start = len(glob.glob(f"{out}/shard*.npz"))
    jobs = []
    for k, lo in enumerate(range(seed0, seed0 + builds, chunk)):
        jobs.append((tactical, list(range(lo, min(lo + chunk, seed0 + builds))), draws,
                     os.path.join(out, f"shard{start + k:04d}.npz")))
    t = time.time()
    done = 0
    with mp.get_context("spawn").Pool(procs) as pool:
        for n in pool.imap_unordered(_gen_worker, jobs):
            done += n
            rate = done / (time.time() - t)
            print(f"  {done}/{builds} builds, {rate:.0f}/s, {(builds - done) / rate / 60:.0f} min left", flush=True)
    print(f"{builds} builds in {(time.time() - t) / 60:.1f} min -> {out}")


# ------------------------------------------------------------------ model
def make_net(g_dim: int, emb: int = 32, d: int = 256):
    import torch
    import torch.nn as nn

    class CapacityNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"g_dim": g_dim, "emb": emb, "d": d}
            self.emb = nn.Embedding(VOCAB_SIZE, emb)
            self.joker = nn.Sequential(nn.Linear(emb + F_JOKER + F_POS, 128), nn.ReLU(), nn.Linear(128, 128))
            self.out = nn.Sequential(nn.Linear(g_dim + 256, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(),
                                     nn.Linear(d, 1))

        def forward(self, g, jid, jf):
            m = jf[..., :1]                                          # present flag
            je = self.joker(torch.cat([self.emb(jid), jf], -1)) * m
            jsum = je.sum(1)
            jmax = (je - (1 - m) * 1e4).max(1).values.clamp(min=-10)
            return self.out(torch.cat([g, jsum, jmax], -1)).squeeze(-1)
    return CapacityNet()


def load_capacity(path, device="cpu"):
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    m = make_net(**ck["config"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


class Capacity:
    """Predicted log10 chips in one blind for a batch of games."""

    def __init__(self, path, device="cpu"):
        import torch
        self.torch, self.device = torch, device
        self.model = load_capacity(path, device)

    def __call__(self, games):
        feats = [build_features(g) for g in games]
        t = self.torch
        with t.no_grad():
            y = self.model(t.from_numpy(np.stack([f[0] for f in feats])).to(self.device),
                           t.from_numpy(np.stack([f[1] for f in feats])).to(self.device),
                           t.from_numpy(np.stack([f[2] for f in feats])).to(self.device))
        return y.cpu().numpy()


def train(data, out, epochs=30, lr=1e-3, batch=512, device="cuda", val_frac=0.05):
    import torch
    import torch.nn.functional as F
    shards = [np.load(f) for f in sorted(glob.glob(f"{data}/shard*.npz"))]
    D = {k: np.concatenate([s[k] for s in shards]) for k in shards[0].files}
    N = len(D["y"])
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    nv = max(1000, int(N * val_frac))
    val, tr = idx[:nv], idx[nv:]
    T = {k: torch.from_numpy(v).to(device) for k, v in D.items()}
    model = make_net(D["g"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    base_var = float(D["y"][val].var())
    print(f"{N} builds ({nv} held out); label spread (variance) {base_var:.3f}")
    for ep in range(epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(tr)).to(device)
        for k in range(0, len(perm), batch):
            mb = perm[k:k + batch]
            loss = F.mse_loss(model(T["g"][mb], T["jid"][mb], T["jf"][mb]), T["y"][mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = torch.from_numpy(val).to(device)
            pv = model(T["g"][v], T["jid"][v], T["jf"][v])
            mse = F.mse_loss(pv, T["y"][v]).item()
        if ep % 5 == 4 or ep == epochs - 1:
            print(f"  epoch {ep + 1}: held-out error {mse:.4f} (R^2 {1 - mse / base_var:.3f}, "
                  f"typical miss x{10 ** math.sqrt(mse):.2f} in chips)", flush=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, out)
    print("saved", out)


# ------------------------------------------------------------------ checks against real games
def _auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _check_worker(args):
    import torch
    torch.set_num_threads(1)
    from .shopsearch import Net, make_env, device
    model, policy, tactical, seeds, draws = args
    dev = device()
    cap = Capacity(model, dev)
    net = Net(policy, dev)
    senv = make_env(tactical, dev=dev)
    tac = senv.tactics
    rows = []
    for seed in seeds:
        obs = senv.reset(seed)
        done, seen = False, set()
        starts = []
        while not done:
            g = senv.g
            if g.state == "BLIND_SELECT" and g.blind_idx == 0 and g.ante not in seen:
                seen.add(g.ante)
                boss_t = g.blind_target(2)
                m, _ = measure(g, tac, draws, random.Random(seed * 100 + g.ante))
                starts.append({"ante": g.ante, "pred": float(cap([g])[0]), "measured": m,
                               "boss": math.log10(boss_t), "jokers": len(g.jokers)})
            obs, r, done, info = senv.step(int(net([obs])[0][0].argmax()))
        for s in starts:
            s["cleared_boss"] = info["blinds"] >= 3 * s["ante"]
            rows.append(s)
    return rows


def check(model, policy, games=200, procs=8, draws=4, tactical=TACTICAL, seed0=10_000):
    seeds = list(range(seed0, seed0 + games))
    jobs = [(model, policy, tactical, seeds[i::procs], draws) for i in range(procs)]
    with mp.get_context("spawn").Pool(procs) as pool:
        rows = [r for part in pool.map(_check_worker, jobs) for r in part]
    P = np.array([r["pred"] for r in rows])
    M = np.array([r["measured"] for r in rows])
    B = np.array([r["boss"] for r in rows])
    J = np.array([r["jokers"] for r in rows])
    A = np.array([r["ante"] for r in rows])
    L = np.array([r["cleared_boss"] for r in rows])
    err = P - M
    print(f"{len(rows)} ante starts from {games} real games")
    print(f"  predicted vs freshly measured capacity of the same real build: R^2 "
          f"{1 - err.var() / max(M.var(), 1e-9):.3f}, bias {err.mean():+.3f} (log10), typical miss x{10 ** np.abs(err).mean():.2f}")
    print("  how well each score tells whether the run clears that ante's boss (AUC, 0.5 = no better than chance):")
    print(f"    predicted capacity / boss target  {_auc(P - B, L):.3f}")
    print(f"    measured capacity / boss target   {_auc(M - B, L):.3f}")
    print(f"    number of jokers                  {_auc(J, L):.3f}")
    print(f"    ante (earlier = safer)            {_auc(-A, L):.3f}")
    for a in sorted(set(A)):
        s = A == a
        if s.sum() >= 20 and 0 < L[s].mean() < 1:
            print(f"    ante {a}: {s.sum()} starts, {100 * L[s].mean():.0f}% cleared the boss; AUC predicted "
                  f"{_auc(P[s] - B[s], L[s]):.3f}, measured {_auc(M[s] - B[s], L[s]):.3f}, jokers {_auc(J[s], L[s]):.3f}")


# ------------------------------------------------------------------ capacity-guided shopping
BUILD_ACTIONS = list(range(A_BUY, A_BUY + MAX_SHOP)) + list(range(A_SELL_J, A_SELL_J + MAX_JOKERS)) + \
    list(range(A_USE_C, A_USE_C + 3)) + list(range(A_PICK, A_PSKIP)) + list(range(A_SWAP, N_ACTIONS))


def capacity_choice(senv, obs, net_action, cap, threshold):
    """The network's action, unless a build action raises predicted capacity by more than `threshold`."""
    g = senv.g
    if g.state not in ("SHOP", "PACK"):
        return net_action, False
    cands = [a for a in BUILD_ACTIONS if obs["mask"][a] and a != net_action]
    if not cands:
        return net_action, False
    after = []
    for a in cands + [net_action]:
        e = copy.deepcopy(senv.env)
        e.g.rng = random.Random(0)
        e.step(a)
        after.append(e.g)
    vals = cap(after + [g])
    now, net_val = vals[-1], vals[-2]
    base = max(now, net_val)
    i = int(np.argmax(vals[:len(cands)]))
    if vals[i] - base > threshold and after[i].money >= 0:
        return cands[i], True
    return net_action, False


def _shop_worker(args):
    import torch
    torch.set_num_threads(1)
    from .shopsearch import Net, make_env, play_game, device
    model, policy, tactical, seeds, thresholds = args
    dev = device()
    cap = Capacity(model, dev)
    net = Net(policy, dev)
    senv = make_env(tactical, dev=dev)
    out = []
    for seed in seeds:
        row = [play_game(senv, seed, lambda e, o: (int(net([o])[0][0].argmax()), None))[0]["blinds"]]
        for th in thresholds:
            n = [0]

            def choose(e, o, th=th):
                a, over = capacity_choice(e, o, int(net([o])[0][0].argmax()), cap, th)
                n[0] += over
                return a, None
            row.append(play_game(senv, seed, choose)[0]["blinds"])
            row.append(n[0])
        out.append(row)
    return out


def shop(model, policy, games=200, procs=8, thresholds=(0.03, 0.1, 0.2), tactical=TACTICAL, seed0=10_000):
    from .shopsearch import boot_ci
    seeds = list(range(seed0, seed0 + games))
    jobs = [(model, policy, tactical, seeds[i::procs], list(thresholds)) for i in range(procs)]
    with mp.get_context("spawn").Pool(procs) as pool:
        R = np.array([r for part in pool.map(_shop_worker, jobs) for r in part], float)
    print(f"{len(R)} games, same seeds")
    print("  network alone                blinds %.2f [%.2f, %.2f]" % boot_ci(R[:, 0]))
    for k, th in enumerate(thresholds):
        col = R[:, 1 + 2 * k]
        print(f"  capacity overrules at +{th:<5} blinds %.2f [%.2f, %.2f]" % boot_ci(col),
              "  difference %+.2f [%+.2f, %+.2f]" % boot_ci(col - R[:, 0]),
              f"  {R[:, 2 + 2 * k].mean():.1f} overrules per game")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    gn = sub.add_parser("gen")
    gn.add_argument("--builds", type=int, default=200_000)
    gn.add_argument("--draws", type=int, default=4)
    gn.add_argument("--procs", type=int, default=8)
    gn.add_argument("--seed0", type=int, default=0)
    gn.add_argument("--out", default="checkpoints/capacity_data")
    gn.add_argument("--tactical", default=TACTICAL)
    tr = sub.add_parser("train")
    tr.add_argument("--data", default="checkpoints/capacity_data")
    tr.add_argument("--out", default="checkpoints/capacity.pt")
    tr.add_argument("--epochs", type=int, default=30)
    ck = sub.add_parser("check")
    ck.add_argument("--model", default="checkpoints/capacity.pt")
    ck.add_argument("--policy", required=True)
    ck.add_argument("--games", type=int, default=200)
    ck.add_argument("--procs", type=int, default=8)
    sh = sub.add_parser("shop")
    sh.add_argument("--model", default="checkpoints/capacity.pt")
    sh.add_argument("--policy", required=True)
    sh.add_argument("--games", type=int, default=200)
    sh.add_argument("--procs", type=int, default=8)
    sh.add_argument("--thresholds", type=float, nargs="+", default=[0.03, 0.1, 0.2])
    a = p.parse_args()
    if a.cmd == "gen":
        generate(a.builds, a.out, a.draws, a.procs, a.tactical, a.seed0)
    elif a.cmd == "train":
        train(a.data, a.out, a.epochs)
    elif a.cmd == "check":
        check(a.model, a.policy, a.games, a.procs)
    else:
        shop(a.model, a.policy, a.games, a.procs, a.thresholds)


if __name__ == "__main__":
    main()
