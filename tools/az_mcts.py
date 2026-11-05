"""AlphaZero-style search and training on a list of seeds whose whole future is known (the real game's named random streams: exact mode).

    python tools/az_mcts.py --seeds 100001-100064 --out checkpoints/az1.pt

THE GAME AS A TREE. A node is a state at a blind select. Its children are the ways to reach the next blind select: skip the blind, or play it (the exact round solver finds the
plays that clear it) and then shop (a layer-by-layer shop search; the engine's `expand_level`), at most --ends of them per node, each an end state with the complete list of the
actions that lead to it. Every edge pays the reward of the blinds it passes (geometric x1.6 per ante, summing to 24) and beating the ante-8 boss pays --win-bonus more.
Nothing is hidden: the engine is exact and deterministic, so the tree is the real game, and the network is given the whole future (the privileged `priv` observation: the next
shops, the boss, the top of the deck, the list for the next eight antes) as an input.

THE SEARCH. PUCT over those nodes. The value of a node is the network's value (the reward still to come, in units of 1 / (24 + win bonus)); the prior of a child is a softmax over
the siblings of (the engine's build-score ranking / tau_s + the child's value / tau_v). A leaf is expanded by one `expand_level` and its children are all valued by the network in
one batch; the value backed up is the best of the children's (edge reward + value). One wave advances every game's search by one simulation, so one engine call and one network call
serve all the games at once. After --sims simulations the game moves to the most visited child and the search starts again from there; a winning line, once found, ends the game.

THE LEARNING. When a game ends, every state it was in is a training example: the reward it actually collected from there on (its search's own result) is the value target. The value
network is fitted to those (replay buffer), and the next searches use it. The score reported is how many of the seed list the search wins, and how many blinds it beats.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from balatro_rl.engine_env import F_PRIV             # noqa: E402
from balatro_rl.model import ActorCritic, load_model  # noqa: E402
from balatro_rl.train import blind_weights           # noqa: E402
from solve_fast import make                          # noqa: E402

OBS = ("g", "cards", "jid", "jf", "cid", "priv")


class Node:
    __slots__ = ("slot", "parent", "path", "r_in", "prior", "v", "N", "W", "children", "expanded", "win", "b", "obs", "score")

    def __init__(self, slot, parent, path, r_in, prior, v, b, obs, score):
        self.slot, self.parent, self.path, self.r_in, self.prior, self.v = slot, parent, path, r_in, prior, v
        self.N, self.W, self.children, self.expanded, self.win, self.b, self.obs, self.score = 0, 0.0, [], False, None, b, obs, score


class Game:
    def __init__(self, seed, block, a):
        self.seed, self.block = seed, block                  # the pool slots this game owns: block * B .. (block + 1) * B - 1
        self.free = []
        self.root = None
        self.actions = []                                    # the whole line played so far
        self.examples = []                                   # (obs, b) of every state the game was in
        self.done = False
        self.won = False
        self.solution = None
        self.sims = 0


def compact(ob):
    return {k: (x.astype(np.float16) if x.dtype == np.float32 else x) for k, x in ob.items()}


def state_b(g_row):
    """The blind about to be faced, 0..23 (obs "g": column 0 is ante / 8, columns 1-3 the blind's one-hot)."""
    ante = int(round(float(g_row[0]) * 8))
    return 3 * (max(1, ante) - 1) + int(np.argmax(g_row[1:4]))


def run_search(net, a, dev):
    W = blind_weights(1.0, 1.0, geom=a.weight_geom)
    cum = np.concatenate([[0.0], np.cumsum(W)])              # reward of blinds [i, j) = cum[j] - cum[i]
    scale = 24.0 + a.win_bonus
    seeds = a.seed_list
    G = a.games
    B = 1 + a.sims * a.ends + 8
    pool = make(G * B, a.deck, a.stake, 1, 1)
    dest = make(G * a.ends, a.deck, a.stake, 1, 1)
    dest.set_priv(True)
    roots_v = make(G, a.deck, a.stake, 1, 1)
    roots_v.set_priv(True)
    rng = np.random.default_rng(a.seed)
    buf_obs = {k: deque(maxlen=a.buffer) for k in OBS}
    buf_z = deque(maxlen=a.buffer)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    amp = str(dev).startswith("cuda")
    log = a.out.replace(".pt", "_log.jsonl")
    solved_ema, blinds_ema = None, None
    seen = {}
    t_start = time.time()
    batch_i = 0
    while batch_i < a.batches:
        batch_i += 1
        t0 = time.time()
        # ------------------------------------------------------------ the games of this batch: seeds cycle through the list
        games = []
        for gi in range(G):
            sd = seeds[((batch_i - 1) * G + gi) % len(seeds)]
            games.append(Game(sd, gi, a))
        # each game's start state: environment 0 of an engine playing that seed (one root environment per game)
        # environment gi of an engine with seed0 = s plays seed s + gi: the batch's seeds are consecutive (the list is a range)
        s0 = games[0].seed
        src = make(G, a.deck, a.stake, s0, 1)
        src.set_priv(True)
        pool.copy_many(src, list(range(G)), [gm.block * B for gm in games])
        for gm in games:
            gm.free = list(range(gm.block * B + 1, (gm.block + 1) * B))
            ob = compact({k: np.array(src.obs[k][gm.block], copy=True) for k in OBS})
            gm.root = Node(gm.block * B, None, [], 0.0, 1.0, 0.0, state_b(ob["g"]), ob, 0.0)
        active = list(games)
        move = 0
        while active and move < a.max_moves:
            move += 1
            for gm in active:
                gm.sims = 0
                if gm.root.expanded:
                    gm.root.parent = None
            for sim in range(a.sims + 1):
                leaves = []
                for gm in active:
                    if gm.done or gm.solution is not None:
                        continue
                    node = gm.root
                    path = []
                    while node.expanded and node.children and node.win is None:
                        sq = math.sqrt(max(1, node.N))
                        best, bs = None, -1e9
                        for c in node.children:
                            q = (c.W / c.N) if c.N > 0 else c.v
                            sc = c.r_in / scale + q + a.c_puct * c.prior * sq / (1 + c.N)
                            if sc > bs:
                                best, bs = c, sc
                        node = best
                        path.append(node)
                    if node.expanded and not node.children:                             # a dead end (or a winning leaf, already counted): nothing more to gain
                        v = node.v if node.win is not None else 0.0
                        for n in reversed(path):
                            n.N += 1
                            n.W += v
                            v = n.r_in / scale + v
                        continue
                    leaves.append((gm, node, path))
                if not leaves:
                    continue
                res = pool.expand_level([n.slot for _, n, _ in leaves], dest, a.ends, a.sw, a.sl, a.node_cap, 0.0, 0.0, 0.0, [], 0.0)
                new_nodes = []                                                          # (gm, parent, cand)
                for (gm, node, path), (win, wpath, cands) in zip(leaves, res):
                    node.expanded = True
                    if win:
                        node.win = list(wpath)
                        w_final = W[min(23, node.b)] if node.b < 24 else 0.0
                        r_win = (w_final + a.win_bonus) / scale
                        node.v = r_win
                        v = r_win
                        for n in reversed(path):
                            n.N += 1
                            n.W += v
                            v = n.r_in / scale + v
                        gm.solution = (node, list(wpath))
                        continue
                    for c in cands:
                        new_nodes.append((gm, node, c))
                    node.children = []
                if new_nodes:
                    dslots = [c[3] for _, _, c in new_nodes]
                    ob = {k: torch.as_tensor(np.asarray(dest.obs[k])[dslots]).to(dev) for k in OBS}
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        vals = net.values(ob).float().cpu().numpy()
                    if vals.ndim > 1:
                        vals = vals.sum(-1)
                    ob_np = {k: np.asarray(dest.obs[k])[dslots] for k in OBS}
                    pslots = []
                    for gm, node, c in new_nodes:
                        pslots.append(gm.free.pop())
                    pool.copy_many(dest, dslots, pslots)
                    by_parent = {}
                    for i, (gm, node, c) in enumerate(new_nodes):
                        by_parent.setdefault(id(node), (node, []))[1].append(i)
                    for node, idxs in by_parent.values():
                        sc = np.array([new_nodes[i][2][1] for i in idxs], dtype=np.float64)
                        vv = vals[idxs]
                        logit = (sc - sc.max()) / a.tau_s + (vv - vv.max()) / a.tau_v
                        pri = np.exp(logit - logit.max())
                        pri /= pri.sum()
                        for j, i in enumerate(idxs):
                            gm, _, c = new_nodes[i]
                            b_child = state_b(ob_np["g"][i])
                            r_in = cum[min(24, max(b_child, node.b))] - cum[min(24, node.b)]
                            ch = Node(pslots[i], node, list(c[0]), float(r_in), float(pri[j]), float(vv[j]), b_child, {k: ob_np[k][i].astype(np.float16) if ob_np[k].dtype == np.float32 else ob_np[k][i].copy() for k in OBS}, float(c[1]))
                            node.children.append(ch)
                # back up: the best of the children's (edge reward + value) from every leaf that was expanded (a winning leaf was backed up above)
                for (gm, node, path) in leaves:
                    if node.win is not None:
                        continue
                    if node.children:
                        v = max(c.r_in / scale + c.v for c in node.children)
                    else:
                        v = 0.0
                    node.v = v
                    for n in reversed(path):
                        n.N += 1
                        n.W += v
                        v = n.r_in / scale + v
                    if not path:
                        node.N += 1
            # ------------------------------------------------------------ every active game moves
            still = []
            for gm in active:
                root = gm.root
                if gm.solution is not None:
                    node, wpath = gm.solution
                    chain = []
                    n = node
                    while n is not None and n.parent is not None:
                        chain.append(n.path)
                        n = n.parent
                    for p_ in reversed(chain):
                        gm.actions += p_
                    gm.actions += wpath
                    gm.done, gm.won = True, True
                    gm.final_b = 24
                    continue
                if not root.children:
                    gm.done = True
                    gm.final_b = root.b
                    continue
                gm.examples.append((root.obs, root.b))
                visits = np.array([c.N for c in root.children], dtype=np.float64)
                if move <= a.sample_moves and a.temp > 0:
                    pr = visits ** (1.0 / a.temp)
                    pr /= pr.sum()
                    k = int(rng.choice(len(visits), p=pr))
                else:
                    k = int(np.argmax(visits))
                child = root.children[k]
                gm.actions += child.path
                pool.copy_many(pool, [child.slot], [gm.block * B])
                old_slots = [gm.block * B + 1 + i for i in range(B - 1)]
                gm.free = old_slots
                child.parent = None
                newroot = Node(gm.block * B, None, [], 0.0, 1.0, child.v, child.b, child.obs, child.score)
                gm.root = newroot
                still.append(gm)
            active = still
        for gm in active:                                    # out of moves
            gm.done = True
            gm.final_b = gm.root.b
        # ------------------------------------------------------------ the batch is over: examples, training
        n_win = 0
        for gm in games:
            if gm.won:
                n_win += 1
            fb = getattr(gm, "final_b", gm.root.b if gm.root else 0)
            for obs, b in gm.examples:
                z = (cum[min(24, fb)] - cum[min(24, b)] + (a.win_bonus if gm.won else 0.0)) / scale
                for k in OBS:
                    buf_obs[k].append(obs[k])
                buf_z.append(z)
            seen[gm.seed] = max(seen.get(gm.seed, 0), 24 if gm.won else fb)
        mean_b = float(np.mean([getattr(gm, "final_b", 0) for gm in games]))
        t1 = time.time()
        losses = []
        if len(buf_z) >= 64:
            net.train()
            n = len(buf_z)
            arrs = {k: np.stack(list(buf_obs[k])) for k in OBS}
            zs = np.array(buf_z, dtype=np.float32)
            for _ in range(a.train_steps):
                idx = rng.integers(0, n, size=min(a.batch, n))
                ob = {k: torch.as_tensor(arrs[k][idx]).to(dev) for k in OBS}
                ob = {k: (x.float() if x.dtype == torch.float16 else x) for k, x in ob.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    v = net.values(ob).float()
                if v.dim() > 1:
                    v = v.sum(-1)
                loss = F.mse_loss(v, torch.as_tensor(zs[idx]).to(dev))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                losses.append(float(loss))
            net.eval()
        t2 = time.time()
        solved_ema = n_win / G if solved_ema is None else 0.8 * solved_ema + 0.2 * (n_win / G)
        blinds_ema = mean_b if blinds_ema is None else 0.8 * blinds_ema + 0.2 * mean_b
        row = {"batch": batch_i, "games": batch_i * G, "wins": n_win, "win_rate": round(n_win / G, 3), "win_rate_ema": round(solved_ema, 3), "blinds": round(mean_b, 2),
               "blinds_ema": round(blinds_ema, 2), "seeds_solved_ever": sum(1 for v in seen.values() if v >= 24), "seeds_seen": len(seen),
               "loss": round(float(np.mean(losses)), 5) if losses else None, "buffer": len(buf_z), "secs": {"search": round(t1 - t0, 1), "train": round(t2 - t1, 1)},
               "min": round((time.time() - t_start) / 60, 1)}
        print(json.dumps(row), flush=True)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        torch.save({"config": net.config, "state_dict": net.state_dict()}, a.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="100001-100064", help="the seed list lo-hi (consecutive: the games of a batch are consecutive seeds)")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--out", default="checkpoints/az1.pt")
    p.add_argument("--init", default="")
    p.add_argument("--games", type=int, default=16, help="games searched at once")
    p.add_argument("--sims", type=int, default=32, help="simulations per move")
    p.add_argument("--ends", type=int, default=8, help="children per node")
    p.add_argument("--sw", type=int, default=8)
    p.add_argument("--sl", type=int, default=8)
    p.add_argument("--node-cap", type=int, default=1500)
    p.add_argument("--c-puct", type=float, default=0.15)
    p.add_argument("--tau-s", type=float, default=0.5, help="the prior's temperature on the engine's build score (decades)")
    p.add_argument("--tau-v", type=float, default=0.05, help="... and on the network's value (units of 1 / (24 + win bonus))")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--sample-moves", type=int, default=6, help="the first moves of a game are sampled from the visits")
    p.add_argument("--max-moves", type=int, default=40)
    p.add_argument("--weight-geom", type=float, default=1.6)
    p.add_argument("--win-bonus", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--buffer", type=int, default=40000)
    p.add_argument("--batches", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    lo, hi = [int(x) for x in a.seeds.split("-")]
    a.seed_list = list(range(lo, hi + 1))
    assert len(a.seed_list) % a.games == 0, "the seed list must be a whole number of batches (consecutive seeds per batch)"
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    if a.init:
        net = load_model(a.init, dev)
    else:
        net = ActorCritic(attn=2, desc=1, priv=F_PRIV, priv_actor=1).to(dev)
    net.eval()
    run_search(net, a, dev)


if __name__ == "__main__":
    main()
