"""One blind run (the seed is not used: an ordinary game with hidden draws) played by the run eval as a one-step policy: at every decision outside a
round, each legal action is applied on --samples copies of the game whose future is redrawn, the states it leads to are scored by the eval
(a won copy scores 1, a lost one 0), and the action with the best mean score is taken. The scripted round player plays the rounds. Prints the
decisions and the build ante by ante; with --ckpt the network plays the same seed for comparison.

    python tools/runeval_play.py --seed 777 --samples 8 --ckpt checkpoints/ppo_gold_v30_best.pt
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from balatro_rl.engine_env import CppVecEnv                                                                     # noqa: E402
from balatro_rl.env import (A_SELECT, A_SKIP, A_REROLL_BOSS, A_BUY, A_BUY_PACK, A_VOUCHER, A_REROLL, A_LEAVE, A_SELL_J, A_SELL_C, A_USE_C, A_PICK,
                            A_PSKIP, A_SWAP, A_SWAP_END, A_REPLACE)                                              # noqa: E402
from balatro_rl.runeval import RunEval, Cols, I_BLIND_SELECT, I_SHOP, I_PACK, from_vecenv, HAND_NAMES         # noqa: E402
from balatro_rl.sim.items import VOCAB                                                                           # noqa: E402


def describe(a, aid, aid2, jid):
    name = lambda x: VOCAB[int(x)][2:] if 0 <= int(x) < len(VOCAB) and VOCAB[int(x)][:2] in ("j_", "c_", "v_", "p_") else VOCAB[int(x)] if 0 <= int(x) < len(VOCAB) else "?"
    if a == A_SELECT:
        return "select"
    if a == A_SKIP:
        return "skip"
    if a == A_REROLL_BOSS:
        return "reroll boss"
    if A_BUY <= a < A_BUY_PACK:
        return f"buy {name(aid[a])}"
    if A_BUY_PACK <= a < A_VOUCHER:
        return f"pack {name(aid[a])}"
    if A_VOUCHER <= a < A_REROLL:
        return f"voucher {name(aid[a])}"
    if a == A_REROLL:
        return "reroll"
    if a == A_LEAVE:
        return "leave"
    if A_SELL_J <= a < A_SELL_C:
        return f"sell {name(jid[a - A_SELL_J])}"
    if A_SELL_C <= a < A_USE_C:
        return f"sell consumable {name(aid[a])}"
    if A_USE_C <= a < A_PICK:
        return f"use {name(aid[a])}"
    if A_PICK <= a < A_PSKIP:
        return f"pick {name(aid[a])}"
    if a == A_PSKIP:
        return "skip pack"
    if A_SWAP <= a < A_SWAP_END:
        return f"swap {a - A_SWAP}"
    if a >= A_REPLACE:
        return f"replace {name(aid2[a])} with {name(aid[a])}"
    return f"action {a}"


def money_of(g):
    return float(g[Cols.MONEY] * 50.0 if g[Cols.MONEY] < 1.99 else 10.0 ** (3.0 * g[Cols.LOG_MONEY]) - 1.0)


def eval_run(seed, samples, stake="GOLD", max_steps=3000, verbose=True):
    ev = RunEval()
    v = CppVecEnv(1, "RED", stake, seed0=seed, threads=0)
    v.set_auto_round(True, 1.5, 0)
    v.v.restart()
    cap = 64
    scratch = CppVecEnv(cap * samples, "RED", stake, seed0=seed + 1, threads=0)
    scratch.set_auto_round(True, 1.5, 0)
    t0 = time.time()
    salt = 0
    for step in range(max_steps):
        g = np.asarray(v.obs["g"])[0]
        mask = np.asarray(v.obs["mask"])[0]
        aid = np.asarray(v.obs["aid"])[0]
        aid2 = np.asarray(v.obs["aid2"])[0]
        jid = np.asarray(v.obs["jid"])[0]
        st = int(np.argmax(g[Cols.STATE:Cols.STATE + 4]))
        ante, blind = int(round(g[Cols.ANTE] * 8)), int(np.argmax(g[Cols.BLIND:Cols.BLIND + 3]))
        legal = np.flatnonzero(mask)[:cap]
        if len(legal) == 0:
            break
        if len(legal) == 1:
            best = int(legal[0])
            scores = {best: float("nan")}
        else:
            n = len(legal) * samples
            scratch.copy_many(v, [0] * n, list(range(n)))
            salt += 1
            scratch.reseed_rng(list(range(n)), [seed * 1000 + salt * 100_003 + k for k in range(n)])
            acts = np.full(scratch.n, -1, dtype=np.int64)
            for k, a in enumerate(legal):
                acts[k * samples:(k + 1) * samples] = a
            _, d, info = scratch.step(acts)
            sc = ev(**from_vecenv(scratch, list(range(n))))["score"]
            won = np.asarray(info["won"])[:n] > 0.5
            sc = np.where(d[:n], np.where(won, 1.0, 0.0), sc)
            per = sc.reshape(len(legal), samples).mean(1)
            order = np.argsort(-per)
            best = int(legal[order[0]])
            scores = {int(legal[k]): float(per[k]) for k in range(len(legal))}
        if verbose and st in (I_BLIND_SELECT, I_SHOP, I_PACK) and (st != I_BLIND_SELECT or blind == 0 or len(legal) > 1):
            jokers = [VOCAB[x][2:] for x in jid if x > 0]
            top = sorted(scores.items(), key=lambda kv: -kv[1] if kv[1] == kv[1] else 0)[:3]
            alt = ", ".join(f"{describe(a, aid, aid2, jid)} {s:.2f}" for a, s in top[1:]) if len(top) > 1 else ""
            print(f"ante {ante} blind {blind} {['select', 'round', 'shop', 'pack'][st]:6s} ${money_of(g):>4.0f} | {describe(best, aid, aid2, jid):34s} {scores[best]:.2f}"
                  + (f"   (next: {alt})" if alt else "") + (f"   jokers {jokers}" if st == I_BLIND_SELECT and blind == 0 else ""))
        _, d, info = v.step(np.array([best]))
        if d[0]:
            return dict(won=bool(np.asarray(info["won"])[0] > 0.5), blinds=int(np.asarray(info["blinds"])[0]), secs=time.time() - t0, steps=step + 1)
    return dict(won=False, blinds=-1, secs=time.time() - t0, steps=max_steps)


def net_run(seed, ckpt, stake="GOLD"):
    from runeval_data import Policy, play_out
    v = CppVecEnv(1, "RED", stake, seed0=seed, threads=0)
    v.set_auto_round(True, 1.5, 0)
    v.v.restart()
    out = play_out(v, Policy(ckpt, "cuda"), np.array([0]), watch_ante4=False)
    return dict(won=bool(out["won"][0]), blinds=int(out["blinds"][0]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--samples", type=int, default=8, help="redrawn copies per option")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--ckpt", default="", help="also play the same seed with this network, for comparison")
    a = p.parse_args()
    r = eval_run(a.seed, a.samples, a.stake)
    print(f"\nEVAL run, seed {a.seed}: {'WON' if r['won'] else 'lost'}; blinds beaten {r['blinds']} (ante {min(r['blinds'] // 3 + 1, 8)}); {r['steps']} decisions, {r['secs']:.0f}s")
    if a.ckpt:
        n = net_run(a.seed, a.ckpt, a.stake)
        print(f"network {os.path.basename(a.ckpt)} on the same seed: {'WON' if n['won'] else 'lost'}; blinds beaten {n['blinds']} (ante {min(n['blinds'] // 3 + 1, 8)})")


if __name__ == "__main__":
    main()
