"""Teacher games: searched play with the real seed's future (exact-seed mode).

In exact-seed mode the shops, packs, bosses, tags, draws and every random roll come from the game's named streams, so the future
of a seed is a function of the actions alone: a copy of the game can be played forward and shows exactly what each choice leads
to (no re-drawn futures, no chance nodes). At every decision outside a blind (blind select, shop, packs, using consumables) the
student's top-k actions are each tried on a copy that the student then plays on, greedily, to the end of the run (or `horizon`
more steps and the critic's estimate). The candidate with the best outcome is played. Inside blinds the student plays.

The games are written as seeds + the teacher's actions + the search log: the engine is deterministic, so the observations are
rebuilt by replaying (distill.py); a student that does not see the future learns from them with DAgger (play, relabel).

    python tools/teacher_games.py checkpoints/ppo_gold_v22.pt --games 8 --k 6 --stake GOLD --out traces/teacher_pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from balatro_rl.engine_env import CppVecEnv, OBS_KEYS      # noqa: E402
from balatro_rl.model import load_model                      # noqa: E402


def make_env(n, deck, stake, seed0, threads, model, ante_weight):
    v = CppVecEnv(n, deck, stake, seed0=seed0, threads=threads)
    v.set_exact(True)
    v.v.restart()                                           # (the first games are made again, now with the real streams)
    v.set_weight_shape(ante_weight, 1.0)
    if getattr(model, "marg", 0):
        v.set_marginal(int(model.marg))
    return v


def forward(model, venv, device):
    ob = {k: torch.from_numpy(np.array(venv.obs[k], copy=True)).to(device) for k in OBS_KEYS}
    with torch.no_grad():
        logits, value = model(ob)
    return ob, logits.float().cpu(), value.float().cpu().reshape(-1)


def greedy_baseline(model, deck, stake, seeds, threads, device, ante_weight):
    """The student alone on the same seeds (the control): blinds beaten and wins per seed."""
    n = len(seeds)
    env = make_env(n, deck, stake, seeds[0], threads, model, ante_weight)
    blinds = np.full(n, -1)
    won = np.zeros(n, bool)
    while (blinds < 0).any():
        _, logits, _ = forward(model, env, device)
        _, done, info = env.step(logits.argmax(1).numpy())
        for e in np.flatnonzero(done):
            if blinds[e] < 0:
                blinds[e], won[e] = info["blinds"][e], info["won"][e]
    return blinds, won


def search_games(model, deck, stake, seeds, k, horizon, margin, threads, device, ante_weight, gamma=0.999, log=print, rounds=False):
    M = len(seeds)
    main = make_env(M, deck, stake, seeds[0], threads, model, ante_weight)
    roll = make_env(M * k, deck, stake, seeds[0], threads, model, ante_weight)
    actions = [[] for _ in range(M)]                        # the teacher's actions per game
    searches = [[] for _ in range(M)]                       # the search log per game
    result = [None] * M
    alive = np.ones(M, bool)
    step = 0
    t0 = time.time()
    n_search = n_changed = 0
    while alive.any():
        ob, logits, _ = forward(model, main, device)
        act = logits.argmax(1).numpy().copy()
        in_round = ob["g"][:, 5].cpu().numpy() > 0.5
        legal = ob["mask"].cpu().numpy().sum(1)
        todo = [m for m in range(M) if alive[m] and (rounds or not in_round[m]) and legal[m] > 1]
        if todo:
            cands = {}
            a0 = np.full(M * k, -1, dtype=np.int64)
            for m in todo:
                top = torch.topk(logits[m], int(min(k, legal[m]))).indices.numpy()
                cands[m] = top
                for c, a in enumerate(top):
                    roll.copy_from(main, m, m * k + c)
                    a0[m * k + c] = a
            used = np.zeros(M * k, bool)
            used[a0 >= 0] = True
            r, done, info = roll.step(a0)
            ret = np.where(used, r, 0.0).astype(np.float64)
            fin = used & done
            blinds = np.where(fin, info["blinds"], -1)
            won = np.where(fin, info["won"], False)
            disc = np.ones(M * k)
            t = 0
            while not fin[used].all() and t < horizon:
                _, lg, val = forward(model, roll, device)
                a = lg.argmax(1).numpy().astype(np.int64)
                run = used & ~fin
                a[~run] = -1
                disc[run] *= gamma
                r, done, info = roll.step(a)
                ret += np.where(run, r * disc, 0.0)
                new = run & done
                blinds[new] = info["blinds"][new]
                won[new] = info["won"][new]
                fin |= new
                t += 1
            if not fin[used].all():                         # the horizon: the critic's estimate of the rest
                _, _, val = forward(model, roll, device)
                run = used & ~fin
                ret[run] += (disc * gamma)[run] * val.numpy()[run]
            for m in todo:
                idx = np.arange(m * k, m * k + len(cands[m]))
                out = ret[idx]
                best = int(np.argmax(out))
                student = 0                                 # the top candidate is the student's own choice
                choose = best if out[best] > out[student] + margin else student
                searches[m].append({"step": step, "state": ob["g"][m, 5].item(), "n": len(idx),
                                    "cands": cands[m].tolist(), "out": [round(float(x), 3) for x in out],
                                    "blinds": blinds[idx].tolist(), "chose": int(choose)})
                n_search += 1
                n_changed += int(choose != student)
                act[m] = cands[m][choose]
        for m in range(M):
            if alive[m]:
                actions[m].append(int(act[m]))
            else:
                act[m] = -1
        _, done, info = main.step(act)
        for m in np.flatnonzero(done & alive):
            result[m] = {"blinds": int(info["blinds"][m]), "won": bool(info["won"][m]), "ante": int(info["ante"][m])}
            alive[m] = False
        step += 1
        if step % 50 == 0:
            log(f"  step {step}: {int(alive.sum())} games left, {n_search} searches ({n_changed} changed), {time.time() - t0:.0f}s")
    return actions, searches, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--seed0", type=int, default=20000)
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--k", type=int, default=6, help="candidates tried at each decision (the student's top k)")
    p.add_argument("--horizon", type=int, default=400, help="steps a copy is played before the critic's estimate is used")
    p.add_argument("--margin", type=float, default=0.05, help="a candidate must beat the student's choice by this much")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--ante-weight", type=float, default=2.5)
    p.add_argument("--rounds", action="store_true", help="search inside blinds too (which hands to play and discard)")
    p.add_argument("--device", default="cpu", help="cpu: do not use the GPU beside a running trainer")
    p.add_argument("--chunk", type=int, default=100, help="games searched together (memory); each chunk is appended to --out")
    p.add_argument("--resume", action="store_true", help="skip the chunks already in --out")
    p.add_argument("--out", default="traces/teacher_games.jsonl")
    a = p.parse_args()
    torch.set_num_threads(max(1, a.threads))
    model = load_model(a.checkpoint, a.device)
    model.eval()
    print(f"student {os.path.basename(a.checkpoint)}, {a.games} games from seed {a.seed0}, {a.deck} deck, {a.stake} stake, k={a.k}"
          f"{', searching inside blinds' if a.rounds else ''}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done_seeds = set()
    if a.resume and os.path.exists(a.out):                  # (a chunk is written whole: the seeds in the file are finished)
        done_seeds = {json.loads(l)["seed"] for l in open(a.out)}
    elif os.path.exists(a.out):
        open(a.out, "w").close()
    all_sb, all_tb, all_tw, all_sw = [], [], [], []
    for c0 in range(0, a.games, a.chunk):
        seeds = [a.seed0 + i for i in range(c0, min(a.games, c0 + a.chunk))]
        if all(s in done_seeds for s in seeds):
            continue
        t = time.time()
        sb, sw = greedy_baseline(model, a.deck, a.stake, seeds, a.threads, a.device, a.ante_weight)
        actions, searches, result = search_games(model, a.deck, a.stake, seeds, a.k, a.horizon, a.margin, a.threads, a.device,
                                                 a.ante_weight, log=lambda s: None, rounds=a.rounds)
        tb = np.array([r["blinds"] for r in result])
        tw = np.array([r["won"] for r in result])
        with open(a.out, "a") as f:
            for m, seed in enumerate(seeds):
                f.write(json.dumps({"seed": seed, "deck": a.deck, "stake": a.stake, "student_blinds": int(sb[m]),
                                    "teacher": result[m], "actions": actions[m], "searches": searches[m]}) + "\n")
        all_sb += list(sb); all_tb += list(tb); all_tw += list(tw); all_sw += list(sw)
        d = np.array(all_tb) - np.array(all_sb)
        print(f"seeds {seeds[0]}..{seeds[-1]} [{time.time() - t:.0f}s]  so far {len(all_tb)} games: student {np.mean(all_sb):.2f} "
              f"-> teacher {np.mean(all_tb):.2f} blinds (paired {d.mean():+.2f} +- {d.std() / max(1, np.sqrt(len(d))):.2f}), "
              f"wins {int(np.sum(all_sw))} -> {int(np.sum(all_tw))}", flush=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
