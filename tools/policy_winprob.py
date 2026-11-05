"""A policy guided by the analytic win probability (balatro_rl/winprob.py): at every decision outside a round, each candidate action (the
policy's --k most likely, plus leaving / skipping when legal) is applied on a copy and the state it leads to is scored by the model's expected
blinds; a chance action (reroll, buying a pack) is scored on --samples copies with their future redrawn (so the copy does not see the real
next shop or pack). The policy's own choice is kept unless another action is better by --margin blinds. Rounds are played by the policy.

    python tools/policy_winprob.py --ckpt checkpoints/ppo_gold_v7_best_greedy.pt --games 1000
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import env as E                                    # noqa: E402
from balatro_rl.engine_env import CppVecEnv                         # noqa: E402
from balatro_rl.runeval import from_vecenv                          # noqa: E402
from balatro_rl.winprob import WinProb                              # noqa: E402
from policy_search import Pol                                       # noqa: E402

CHANCE = set(range(E.A_BUY_PACK, E.A_VOUCHER)) | {E.A_REROLL}
ONLY_SHOP_ALL = False
SWITCHES = {}


def kind(a):
    from balatro_rl.env import ACTION_TYPES, _TYPE_OF
    return "replace" if a >= E.A_REPLACE else ACTION_TYPES[_TYPE_OF[a]]


def play(pol, wp, seeds, guided, k, samples, margin, max_steps=4000, objective="blinds"):
    G = len(seeds)
    v = CppVecEnv(G, "RED", "GOLD", seed0=seeds[0], threads=0)
    v.v.restart()
    S = CppVecEnv(G * (k + 2) * samples, "RED", "GOLD", seed0=1, threads=0) if guided else None
    done = np.zeros(G, dtype=bool)
    bl = np.zeros(G)
    won = np.zeros(G, dtype=bool)
    salt = 0
    overrides = decisions = 0
    for _ in range(max_steps):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        mask = np.asarray(v.obs["mask"])
        lg = pol.logits(v, live)
        act = np.full(G, -1, dtype=np.int64)
        act[live] = lg.argmax(1)
        if guided:
            outr = ~mask[live, :E.A_SELECT].any(1) & (mask[live].sum(1) >= 2)
            sg = live[outr]
            if len(sg):
                lgo = lg[outr]
                src, acts, own = [], [], []
                for r, gi in enumerate(sg):
                    if mask[gi, E.A_SELECT] or mask[gi, E.A_PSKIP] and not ONLY_SHOP_ALL:     # blind decisions stay with the policy
                        if mask[gi, E.A_SELECT]:
                            continue
                    cands = [c for c in np.argsort(-lgo[r])[:k] if not (E.A_BUY_PACK <= c < E.A_VOUCHER)]   # packs: the policy decides
                    if not cands:
                        continue
                    for extra in (E.A_LEAVE,):
                        if mask[gi, extra] and extra not in cands:
                            cands.append(extra)
                    cands = [int(c) for c in cands if mask[gi, c]]
                    for c in cands:
                        reps = samples if c in CHANCE else 1
                        for _ in range(reps):
                            src.append(int(gi))
                            acts.append(c)
                            own.append((r, c))
                n = len(src)
                if n == 0:
                    _, d, info = v.step(act)
                    f = d & ~done
                    bl[f], won[f] = info["blinds"][f], info["won"][f]
                    done |= d
                    continue
                S.copy_many(v, src, list(range(n)))
                salt += 1
                S.reseed_rng(list(range(n)), [salt * 1_000_003 + j * 7919 + 5 for j in range(n)])   # (every copy's unseen future redrawn)
                a0 = np.full(S.n, -1, dtype=np.int64)
                a0[:n] = acts
                sel = np.array([c == E.A_SELECT for c in acts], dtype=bool)
                a0[:n][sel] = -1                                           # selecting: the state as it is (the blind about to be played)
                _, d, info = S.step(a0)
                out = wp(**from_vecenv(S, list(range(n))))
                val = out["e_blinds"] if objective == "blinds" else out["p_win"] * 100 + out["e_blinds"]
                val = np.where(d[:n], info["blinds"][:n] + 10.0 * info["won"][:n], val)
                tot = {}
                for j, key in enumerate(own):
                    tot.setdefault(key, []).append(val[j])
                for r, gi in enumerate(sg):
                    base = int(act[gi])
                    if (r, base) not in tot:                              # (a decision the model does not judge: the policy's)
                        continue
                    decisions += 1
                    vb = float(np.mean(tot.get((r, base), [-1e9])))
                    best, bv = base, vb
                    for (rr, c), vs in tot.items():
                        if rr == r and np.mean(vs) > bv + 1e-9:
                            best, bv = c, float(np.mean(vs))
                    if best != base and bv > vb + margin:
                        act[gi] = best
                        overrides += 1
                        key = f"{kind(base)} -> {kind(best)}"
                        SWITCHES[key] = SWITCHES.get(key, 0) + 1
        _, d, info = v.step(act)
        f = d & ~done
        bl[f], won[f] = info["blinds"][f], info["won"][f]
        done |= d
    return bl, won, overrides, decisions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v7_best_greedy.pt")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed0", type=int, default=700_000)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--objective", default="blinds", choices=["blinds", "win"])
    a = p.parse_args()
    pol = Pol(a.ckpt, "cuda")
    wp = WinProb()
    print("win-probability parameters:", wp.p)
    seeds = list(range(a.seed0, a.seed0 + a.games))
    t0 = time.time()
    bl0, w0, _, _ = play(pol, wp, seeds, False, a.k, a.samples, a.margin)
    print(f"policy alone : mean blinds {bl0.mean():.2f}, wins {int(w0.sum())}/{a.games} ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    bls, ws, ov, de = [], [], 0, 0
    for c0 in range(0, a.games, a.batch):
        b, w, o, dd = play(pol, wp, seeds[c0:c0 + a.batch], True, a.k, a.samples, a.margin, objective=a.objective)
        bls.append(b)
        ws.append(w)
        ov += o
        de += dd
        print(f"  guided {c0 + len(b)} games: mean blinds {np.concatenate(bls).mean():.2f}, wins {int(np.concatenate(ws).sum())} ({time.time() - t0:.0f}s)", flush=True)
    bl1, w1 = np.concatenate(bls), np.concatenate(ws)
    d = bl1 - bl0
    print(f"policy+model : mean blinds {bl1.mean():.2f}, wins {int(w1.sum())}/{a.games}; the model changed {ov} of {de} decisions ({100 * ov / max(de, 1):.0f}%)")
    print(f"paired: {d.mean():+.2f} blinds a game (SE {d.std(ddof=1) / np.sqrt(len(d)):.2f}); better on {(d > 0).sum()}, worse on {(d < 0).sum()}")
    print("the model's overrides (policy's choice -> model's):", dict(sorted(SWITCHES.items(), key=lambda kv: -kv[1])[:12]))
    print("ante reached, alone :", np.bincount(np.minimum(bl0 // 3 + 1, 9).astype(int), minlength=10)[1:])
    print("ante reached, model :", np.bincount(np.minimum(bl1 // 3 + 1, 9).astype(int), minlength=10)[1:])


if __name__ == "__main__":
    main()
