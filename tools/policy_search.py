"""A policy alone vs the same policy with search (no foresight), on the same Gold seeds.

search: at every decision outside a round with two or more legal actions, the policy's --k most likely actions are each tried on --rollouts copies
of the game whose future is redrawn (VecEnv.reseed_rng: new draws, shops and bosses; what is on the table stays), and each copy is played to the
end by the policy itself (greedy); the action with the most blinds beaten on average (a win counts --win-bonus) is taken. Rounds are played by
the policy in both modes.

    python tools/policy_search.py --ckpt checkpoints/ppo_gold_v7_best_greedy.pt --games 200 --k 4 --rollouts 8
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from balatro_rl import azc                                       # noqa: E402
from balatro_rl import env as E                                  # noqa: E402
from balatro_rl.engine_env import CppVecEnv                      # noqa: E402
from balatro_rl.model import load_model                          # noqa: E402


class Pol:
    def __init__(self, ckpt, dev):
        self.m = load_model(ckpt, dev).eval()
        self.dev = dev

    @torch.no_grad()
    def logits(self, v, idx, chunk=4096):
        out = []
        mask_all = np.asarray(v.obs["mask"])
        for c0 in range(0, len(idx), chunk):
            ii = idx[c0:c0 + chunk]
            lg, _ = self.m(azc.obs_batch(v, list(ii), self.dev))
            lg = lg.float().masked_fill(~torch.from_numpy(mask_all[ii]).to(self.dev), -1e9)
            out.append(lg.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, E.N_ACTIONS), dtype=np.float32)


def play(pol, seeds, search, k, K, win_bonus, max_steps=4000):
    G = len(seeds)
    v = CppVecEnv(G, "RED", "GOLD", seed0=seeds[0], threads=0)
    v.v.restart()
    S = CppVecEnv(G * k * K, "RED", "GOLD", seed0=1, threads=0) if search else None
    done = np.zeros(G, dtype=bool)
    bl = np.zeros(G)
    won = np.zeros(G, dtype=bool)
    salt = 0
    n_search = 0
    for _ in range(max_steps):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        mask = np.asarray(v.obs["mask"])
        lg = pol.logits(v, live)
        act = np.full(G, -1, dtype=np.int64)
        act[live] = lg.argmax(1)
        if search:
            outr = ~mask[live, :E.A_SELECT].any(1) & (mask[live].sum(1) >= 2)
            sg = live[outr]
            if len(sg):
                n_search += len(sg)
                cand = np.argsort(-lg[outr], axis=1)[:, :k]
                n_leg = mask[sg].sum(1)
                slots_src, slots_act, owner = [], [], []
                for r, gi in enumerate(sg):
                    for c in range(min(k, int(n_leg[r]))):
                        for _ in range(K):
                            slots_src.append(int(gi))
                            slots_act.append(int(cand[r, c]))
                            owner.append((r, c))
                n = len(slots_src)
                S.copy_many(v, slots_src, list(range(n)))
                salt += 1
                S.reseed_rng(list(range(n)), [salt * 1_000_003 + j * 7919 + 11 for j in range(n)])
                a0 = np.full(S.n, -1, dtype=np.int64)
                a0[:n] = slots_act
                _, d, info = S.step(a0)
                sdone = np.ones(S.n, dtype=bool)
                sdone[:n] = False
                sbl = np.zeros(S.n)
                swon = np.zeros(S.n, dtype=bool)
                f = d & ~sdone
                sbl[f], swon[f] = info["blinds"][f], info["won"][f]
                sdone |= d
                for _ in range(max_steps):
                    sl = np.flatnonzero(~sdone)
                    if not len(sl):
                        break
                    sa = np.full(S.n, -1, dtype=np.int64)
                    sa[sl] = pol.logits(S, sl).argmax(1)
                    _, d, info = S.step(sa)
                    f = d & ~sdone
                    sbl[f], swon[f] = info["blinds"][f], info["won"][f]
                    sdone |= d
                score = sbl[:n] + win_bonus * swon[:n]
                tot = {}
                for j, (r, c) in enumerate(owner):
                    tot.setdefault((r, c), []).append(score[j])
                for r, gi in enumerate(sg):
                    best_c, best_v = 0, -1e9
                    for c in range(min(k, int(n_leg[r]))):
                        m_ = float(np.mean(tot[(r, c)]))
                        if m_ > best_v + 1e-9:
                            best_c, best_v = c, m_
                    act[gi] = cand[r, best_c]
        _, d, info = v.step(act)
        f = d & ~done
        bl[f], won[f] = info["blinds"][f], info["won"][f]
        done |= d
    return bl, won, n_search


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v7_best_greedy.pt")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed0", type=int, default=700_000)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--win-bonus", type=float, default=10.0)
    p.add_argument("--batch", type=int, default=100, help="games searched at once")
    a = p.parse_args()
    pol = Pol(a.ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(a.seed0, a.seed0 + a.games))
    t0 = time.time()
    bl0, w0, _ = play(pol, seeds, False, a.k, a.rollouts, a.win_bonus)
    print(f"policy alone : mean blinds {bl0.mean():.2f}, wins {int(w0.sum())}/{a.games}  ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    bls, ws, ns = [], [], 0
    for c0 in range(0, a.games, a.batch):
        b, w, n = play(pol, seeds[c0:c0 + a.batch], True, a.k, a.rollouts, a.win_bonus)
        bls.append(b)
        ws.append(w)
        ns += n
        print(f"  searched {c0 + len(b)} games: so far mean blinds {np.concatenate(bls).mean():.2f}, wins {int(np.concatenate(ws).sum())}  ({time.time() - t0:.0f}s)", flush=True)
    bl1, w1 = np.concatenate(bls), np.concatenate(ws)
    print(f"policy+search: mean blinds {bl1.mean():.2f}, wins {int(w1.sum())}/{a.games}  ({ns} searched decisions, {time.time() - t0:.0f}s)")
    d = bl1 - bl0
    print(f"paired: search beats the policy alone by {d.mean():+.2f} blinds a game (SE {d.std(ddof=1) / np.sqrt(len(d)):.2f}); better on {(d > 0).sum()}, worse on {(d < 0).sum()}")
    print("ante reached, alone :", np.bincount(np.minimum(bl0 // 3 + 1, 9).astype(int), minlength=10)[1:])
    print("ante reached, search:", np.bincount(np.minimum(bl1 // 3 + 1, 9).astype(int), minlength=10)[1:])


if __name__ == "__main__":
    main()
