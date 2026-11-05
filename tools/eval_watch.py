"""Greedy evaluation of a training run's checkpoint on fixed seeds: the clean measure of progress.

The training log's numbers come from games where actions are sampled and the entropy bonus keeps play a
little random. Here the checkpoint plays the same seeds every time and always takes its most likely action,
so two readings differ only by what the network learned.

    python tools/eval_watch.py checkpoints/ppo_cpp_white3.pt              # evaluate now, print the history
    python tools/eval_watch.py checkpoints/ppo_cpp_white3.pt --show       # only print the history
    python tools/eval_watch.py checkpoints/ppo_cpp_white3.pt --every 10   # keep evaluating every 10 minutes
        --logs checkpoints/ppo_cpp_white_log.jsonl checkpoints/ppo_cpp_white2_log.jsonl checkpoints/ppo_cpp_white3_log.jsonl

Results are appended to <checkpoint>_eval.jsonl. --logs (the run's logs in order) adds the games and steps
trained so far to each row. --baseline scores another checkpoint once on the same seeds for reference.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from eval_policy import play as _play                        # noqa: E402


def play(ckpt, games, seed0, deck, stake, greedy, device, chunk=250):
    """eval_policy.play in chunks of games (the same seeds), and the GPU's cache released after: a thousand
    games at once left several GB cached beside the trainer, which then spilled to shared memory and crawled."""
    bl, wn = [], []
    for s0 in range(seed0, seed0 + games, chunk):
        b, w = _play(ckpt, min(chunk, seed0 + games - s0), s0, deck, stake, greedy, device)
        bl.append(b); wn.append(w)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(bl), np.concatenate(wn)


def trained(logs: list[str]) -> dict:
    """Games, steps and minutes trained so far, summed over a run's logs."""
    tot = {"games": 0, "env_steps": 0, "min": 0.0}
    for path in logs:
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 4000))
                last = json.loads(f.read().decode().strip().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            continue
        for k in tot:
            tot[k] += last.get(k, 0)
    return tot


def evaluate(ckpt: str, a) -> dict:
    snap = ckpt + ".evalcopy"
    for _ in range(5):                         # the trainer may be writing the file right now
        try:
            shutil.copy(ckpt, snap)
            blinds, won = play(snap, a.games, a.seed0, a.deck, a.stake, True, a.device)
            break
        except Exception:
            time.sleep(2)
    else:
        raise SystemExit(f"could not read {ckpt}")
    # the checkpoint with the best greedy score so far is kept beside the run (<name>_best_greedy.pt): the
    # trainer's own "_best" goes by training games, which include any training aids
    out = ckpt.replace(".pt", "_eval.jsonl")
    prev = [json.loads(l)["blinds"] for l in open(out) if l.strip()] if os.path.exists(out) else []
    if not prev or float(blinds.mean()) > max(prev):
        shutil.copy(snap, ckpt.replace(".pt", "_best_greedy.pt"))
    os.remove(snap)
    ante = 1 + np.minimum(blinds, 23) // 3
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "blinds": round(float(blinds.mean()), 3), "wins": int(won.sum()),
           "win%": round(100 * float(won.mean()), 2), "ante": round(float(ante.mean()), 3),
           "se": round(float(blinds.std() / np.sqrt(len(blinds))), 3),
           "antes": [int((ante[~won] == k).sum()) for k in range(1, 9)] + [int(won.sum())],
           "games_eval": a.games, "seed0": a.seed0, "stake": a.stake, "per_game": blinds.tolist()}
    row.update({f"trained_{k}": v for k, v in trained(a.logs).items()})
    return row


STAKES = ("WHITE", "RED", "GREEN", "BLACK", "BLUE", "PURPLE", "ORANGE", "GOLD", "DIAMOND")   # DIAMOND: simulator only


def evaluate_stakes(ckpt: str, a) -> dict:
    """Greedy win rate and blinds on every stake (the same seeds on each), for a run that trains on a mix."""
    snap = ckpt + ".stakecopy"
    shutil.copy(ckpt, snap)
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "games_eval": a.games, "seed0": a.seed0, "stakes": {}}
    row.update({f"trained_{k}": v for k, v in trained(a.logs).items()})
    for st in STAKES:
        b, w = play(snap, a.games, a.seed0, a.deck, st, True, a.device)
        ante = 1 + np.minimum(b, 23) // 3
        row["stakes"][st] = {"win%": round(100 * float(w.mean()), 2), "wins": int(w.sum()),
                             "blinds": round(float(b.mean()), 3), "ante6+%": round(100 * float((ante >= 6).mean()), 1)}
    os.remove(snap)
    return row


def show_stakes(path: str):
    rows = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
    if not rows:
        print("no per-stake evaluations yet")
        return
    print(f"greedy win% by stake ({rows[-1]['games_eval']} games each on seeds {rows[-1]['seed0']}.., real game)")
    print(f"{'when':>16} {'trained games':>14} | " + " ".join(f"{s[:6]:>7}" for s in STAKES))
    for r in rows:
        print(f"{r['time']:>16} {r.get('trained_games', 0):>14,} | "
              + " ".join(f"{r['stakes'][s]['win%']:7.1f}" if s in r["stakes"] else f"{'-':>7}" for s in STAKES))
    last = rows[-1]["stakes"]
    print(f"{'latest blinds':>31} | " + " ".join(f"{last[s]['blinds']:7.2f}" if s in last else f"{'-':>7}" for s in STAKES))
    print(f"{'latest ante 6+ %':>31} | " + " ".join(f"{last[s]['ante6+%']:7.1f}" if s in last else f"{'-':>7}" for s in STAKES))


def show(path: str, baseline: dict | None = None):
    rows = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
    if not rows:
        print("no evaluations yet")
        return
    first = np.asarray(rows[0]["per_game"])
    print(f"{rows[-1]['games_eval']} greedy games on seeds {rows[-1]['seed0']}.., {rows[-1]['stake']} stake")
    print(f"{'when':>16} {'trained games':>14} {'steps':>12} {'blinds':>7} {'+-':>5} {'wins':>5} {'win%':>6} "
          f"{'ante':>5} | {'a1-2':>5} {'a3-5':>5} {'a6-8':>5} | {'vs first':>14}")
    if baseline:
        print(f"{'baseline':>16} {'':>14} {'':>12} {baseline['blinds']:7.2f} {baseline['se']:5.2f} {baseline['wins']:5d} "
              f"{baseline['win%']:6.1f} {baseline['ante']:5.2f} |   {baseline['name']}")
    for r in rows:
        h = np.asarray(r["antes"], float)
        n = max(h.sum(), 1)
        d = np.asarray(r["per_game"]) - first
        vs = f"{d.mean():+.2f} +- {d.std() / np.sqrt(len(d)):.2f}" if len(d) == len(first) and r is not rows[0] else ""
        print(f"{r['time']:>16} {r.get('trained_games', 0):>14,} {r.get('trained_env_steps', 0):>12,} {r['blinds']:7.2f} "
              f"{r['se']:5.2f} {r['wins']:5d} {r['win%']:6.1f} {r['ante']:5.2f} | {100 * h[0:2].sum() / n:5.1f} "
              f"{100 * h[2:5].sum() / n:5.1f} {100 * h[5:8].sum() / n:5.1f} | {vs:>14}")
    best = max(rows, key=lambda r: r["blinds"])
    print(f"best: {best['blinds']:.2f} blinds, {best['wins']} wins at {best.get('trained_games', 0):,} trained games")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--logs", nargs="*", default=[], help="the run's training logs in order")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="WHITE")
    p.add_argument("--every", type=float, default=0.0, help="minutes between evaluations (0: once)")
    p.add_argument("--show", action="store_true", help="print the history without evaluating")
    p.add_argument("--stakes-every", type=int, default=0,
                   help="also score every stake on each N-th evaluation (0: never), into <checkpoint>_stakes.jsonl")
    p.add_argument("--stakes", action="store_true", help="score every stake now (or with --show: print that history)")
    p.add_argument("--baseline", default="", help="another checkpoint, scored once on the same seeds")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    out = a.checkpoint.replace(".pt", "_eval.jsonl")
    base = None
    bpath = out.replace("_eval.jsonl", "_eval_baseline.json")
    if a.baseline and not a.show:
        b, w = play(a.baseline, a.games, a.seed0, a.deck, a.stake, True, a.device)
        base = {"name": os.path.basename(a.baseline), "blinds": float(b.mean()), "wins": int(w.sum()),
                "win%": 100 * float(w.mean()), "ante": float((1 + np.minimum(b, 23) // 3).mean()),
                "se": float(b.std() / np.sqrt(len(b)))}
        json.dump(base, open(bpath, "w"))
    elif os.path.exists(bpath):
        base = json.load(open(bpath))
    sout = a.checkpoint.replace(".pt", "_stakes.jsonl")
    if a.show:
        if a.stakes:
            return show_stakes(sout)
        return show(out, base)
    if a.stakes and a.every <= 0:
        with open(sout, "a") as f:
            f.write(json.dumps(evaluate_stakes(a.checkpoint, a)) + "\n")
        return show_stakes(sout)
    cycle = 0
    last_mtime = -1.0
    while True:
        while not os.path.exists(a.checkpoint):         # a run that has not saved yet
            time.sleep(15)
        m = os.path.getmtime(a.checkpoint)
        if m != last_mtime:
            last_mtime = m
            row = evaluate(a.checkpoint, a)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            show(out, base)
            cycle += 1
            if a.stakes_every > 0 and cycle % a.stakes_every == 0:
                with open(sout, "a") as f:
                    f.write(json.dumps(evaluate_stakes(a.checkpoint, a)) + "\n")
        if a.every <= 0:
            break
        time.sleep(a.every * 60)


if __name__ == "__main__":
    main()
