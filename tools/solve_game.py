"""Solve one game: search its tree until a winning line is found (exact-seed mode: the game is deterministic).

A beam search over decisions. Every state in the beam is tried with every legal action outside blinds (shop, packs, blind select, consumables)
and the student's best few inside blinds (which cards to play or discard); the children are scored by the reward earned so far plus the
critic's estimate of the rest (and a bonus for the build's expected score), and the best `beam` of them, without duplicates, go on.
A line that wins ends the search. The line is written as seed + actions and replays identically.

    python tools/solve_game.py checkpoints/ppo_gold_v22.pt --seed 777 --beam 256 --out traces/solved_777.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from balatro_rl import env as E                                # noqa: E402
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS          # noqa: E402
from balatro_rl.model import load_model                        # noqa: E402


def make(n, deck, stake, seed, model, ante_weight):
    v = CppVecEnv(n, deck, stake, seed0=seed, threads=0)
    v.set_exact(True)
    v.v.restart()
    v.set_weight_shape(ante_weight, 1.0)
    if getattr(model, "marg", 0):
        v.set_marginal(int(model.marg))
    return v


CHUNK = 1536                                              # states per forward pass (the whole beam at once overflows the GPU)


def forward(model, venv, n, device):
    """-> (the observations of the first n environments as CPU tensors, logits, values)."""
    ob = {k: torch.from_numpy(np.array(venv.obs[k][:n], copy=True)) for k in OBS_KEYS}
    lg, vl = [], []
    with torch.no_grad():
        for i in range(0, n, CHUNK):
            part = {k: v[i:i + CHUNK].to(device) for k, v in ob.items()}
            logits, value = model(part)
            lg.append(logits.float().cpu())
            vl.append(value.float().cpu().reshape(-1))
    return ob, torch.cat(lg), torch.cat(vl)


def signature(ob, i):
    """What makes two states the same for the search: the global features, the jokers and the consumables."""
    return hash((ob["g"][i].round(3).tobytes(), ob["jid"][i].tobytes(), ob["jf"][i].round(2).tobytes(), ob["cid"][i].tobytes()))


def solve(model, deck, stake, seed, beam, kids_per, k_round, build_bonus, ante_weight, device, max_depth=900, log=print):
    root = make(1, deck, stake, seed, model, ante_weight)
    B, C = beam, kids_per
    cur = make(B, deck, stake, seed, model, ante_weight)
    kid = make(B * C, deck, stake, seed, model, ante_weight)
    cur.copy_from(root, 0, 0)
    paths = [[]]
    ret = np.zeros(1)
    n = 1
    t0 = time.time()
    nodes = 0
    best_blinds = 0
    for depth in range(max_depth):
        ob, logits, _ = forward(model, cur, n, device)
        in_round = ob["g"][:, 5].cpu().numpy() > 0.5
        mask = ob["mask"].cpu().numpy()
        a0 = np.full(B * C, -1, dtype=np.int64)
        parent = np.full(B * C, -1)
        c = 0
        for s in range(n):
            legal = np.flatnonzero(mask[s])
            if len(legal) == 0:
                continue
            order = legal[np.argsort(-logits[s, legal].numpy())]
            take = order[:k_round] if in_round[s] else order[:C]
            for a in take:
                if c >= B * C:
                    break
                kid.copy_from(cur, s, c)
                a0[c] = a
                parent[c] = s
                c += 1
        nodes += c
        r, done, info = kid.step(a0)
        live = []
        for j in range(c):
            if done[j]:
                if info["won"][j]:
                    path = paths[parent[j]] + [int(a0[j])]
                    return path, {"depth": depth + 1, "nodes": nodes, "secs": time.time() - t0, "blinds": int(info["blinds"][j])}
                best_blinds = max(best_blinds, int(info["blinds"][j]))
            else:
                live.append(j)
        if not live:
            return None, {"depth": depth + 1, "nodes": nodes, "secs": time.time() - t0, "blinds": best_blinds}
        ob2, _, value = forward(model, kid, c, device)
        phi = np.clip(7.0 * ob2["g"][:, 205].cpu().numpy(), 0.0, 5.0) if ob2["g"].shape[1] > 205 else np.zeros(c)
        score = np.array([ret[parent[j]] + r[j] for j in range(c)]) + value.numpy() + build_bonus * phi
        obn = {k: ob2[k].cpu().numpy() for k in ("g", "jid", "jf", "cid")}
        seen, chosen = set(), []
        for j in sorted(live, key=lambda j: -score[j]):
            sg = signature(obn, j)
            if sg in seen:
                continue
            seen.add(sg)
            chosen.append(j)
            if len(chosen) >= B:
                break
        new_paths, new_ret = [], np.zeros(len(chosen))
        for q, j in enumerate(chosen):
            cur.copy_from(kid, j, q)
            new_paths.append(paths[parent[j]] + [int(a0[j])])
            new_ret[q] = ret[parent[j]] + r[j]
        paths, ret, n = new_paths, new_ret, len(chosen)
        if depth % 25 == 0:
            gb = ob2["g"][chosen[0]]
            log(f"  depth {depth}: {n} states, {nodes} nodes, best score {score[chosen[0]]:.2f}, {time.time() - t0:.0f}s")
    return None, {"depth": max_depth, "nodes": nodes, "secs": time.time() - t0, "blinds": best_blinds}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--beam", type=int, default=256)
    p.add_argument("--kids", type=int, default=14, help="children tried per state outside blinds (best by the student's ranking)")
    p.add_argument("--round-k", type=int, default=5, help="children per state inside blinds")
    p.add_argument("--build-bonus", type=float, default=0.5)
    p.add_argument("--ante-weight", type=float, default=2.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="traces/solved.json")
    a = p.parse_args()
    seed = a.seed if a.seed is not None else random.randrange(100000, 999999)
    model = load_model(a.checkpoint, a.device)
    model.eval()
    print(f"solving seed {seed}: {a.deck} deck, {a.stake} stake, beam {a.beam}", flush=True)
    path, info = solve(model, a.deck, a.stake, seed, a.beam, a.kids, a.round_k, a.build_bonus, a.ante_weight, a.device,
                       log=lambda s: print(s, flush=True))
    print(("SOLVED" if path else "no win found"), info)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"seed": seed, "deck": a.deck, "stake": a.stake, "solved": path is not None, "actions": path, "info": info},
              open(a.out, "w"))


if __name__ == "__main__":
    main()
