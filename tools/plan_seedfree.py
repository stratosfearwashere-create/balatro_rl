"""A seed-free shop planner: the agent never sees the seed's future, but plans with the engine as a model.

At every decision outside a round (blind select, shop, packs) the legal actions are each tried on K copies of the real game whose unseen future
has been redrawn (shops, packs, bosses, tags and shuffles still to come, and every random roll: VecEnv.reseed), and a small beam (the exact round
solver plus the shop search, as in solve_novel) plays on from each copy for H blinds. The action that goes deepest on average is played in the real
game, and the planner is asked again after it (a receding horizon). Rounds, and the choice of cards for a consumable, are played by a network.

    python tools/plan_seedfree.py --stake GOLD --seeds 60001-60008 --model checkpoints/ppo_gold_v22.pt
    python tools/plan_seedfree.py --stake GOLD --seeds 60001-60008 --no-plan        # the network alone, in the same harness
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import make                                            # noqa: E402
from balatro_rl.engine_env import OBS_KEYS                              # noqa: E402
from balatro_rl.model import load_model                                 # noqa: E402

from balatro_rl import env as E                                         # noqa: E402

N_SUB = E.N_SUB
A_SELECT, A_SKIP, A_REROLL_BOSS = E.A_SELECT, E.A_SKIP, E.A_REROLL_BOSS
A_BUY, A_BUY_PACK, A_VOUCHER, A_REROLL, A_LEAVE = E.A_BUY, E.A_BUY_PACK, E.A_VOUCHER, E.A_REROLL, E.A_LEAVE
A_SELL_J, A_SELL_C, A_USE_C, A_PICK, A_PSKIP = E.A_SELL_J, E.A_SELL_C, E.A_USE_C, E.A_PICK, E.A_PSKIP
BLIND_SELECT, SELECTING_HAND, SHOP, PACK = 0, 1, 2, 3


def state_of(obs, i=0):
    return int(np.argmax(np.asarray(obs["g"])[i, 4:8]))


class Planner:
    def __init__(self, deck, stake, seed, k=4, width=4, horizon=3, a_max=10, ends_per=4, shop_width=4, shop_layers=5):
        self.k, self.w, self.h, self.a_max, self.ep = k, width, horizon, a_max, ends_per
        self.sw, self.sl = shop_width, shop_layers
        r = a_max * k
        self.S = make(r, deck, stake, seed, 2)
        self.cur = make(r * width, deck, stake, seed, 2)
        self.dest = make(r * width * ends_per, deck, stake, seed, 2)
        self.nodes = 0
        self.decisions = 0

    # --------------------------------------------------------------------------------------------------------------- imagination
    def imagine(self, n_roots, first_select=None):
        """Beam over self.h blinds from the states in S[0:n_roots]. -> value per root (blinds survived + a little for the build's score)."""
        H, W = self.h, self.w
        val = np.zeros(n_roots)
        done = np.zeros(n_roots, dtype=bool)
        src, idxs = self.S, list(range(n_roots))
        owner = list(range(n_roots))
        for level in range(H):
            res = src.expand_level(idxs, self.dest, self.ep, self.sw, self.sl, 1000, 0.0, 0.0, 1.0, [], 0.0)
            pool = defaultdict(list)
            for j, (win, wpath, cands) in enumerate(res):
                r = owner[j]
                if win:
                    val[r] = H + 2
                    done[r] = True
                    continue
                if level == 0 and first_select is not None and first_select[r]:
                    cands = [c for c in cands if c[0] and c[0][0] == A_SELECT]
                for path, score, sig, slot, jk in cands:
                    pool[r].append((score, sig, slot))
            src_slots, dst, new_owner = [], [], []
            for r in range(n_roots):
                if done[r] or r not in pool:
                    continue
                cs = sorted(pool[r], key=lambda c: -c[0])
                seen, keep = set(), []
                for c in cs:
                    if c[1] in seen:
                        continue
                    seen.add(c[1])
                    keep.append(c)
                    if len(keep) >= W:
                        break
                val[r] = level + 1 + 0.001 * keep[0][0]
                for t, c in enumerate(keep):
                    src_slots.append(c[2])
                    dst.append(r * W + t)
                    new_owner.append(r)
            if not dst:
                break
            self.cur.copy_many(self.dest, src_slots, dst)
            src, idxs, owner = self.cur, dst, new_owner
        return val

    def candidates(self, obs, st):
        m = np.asarray(obs["mask"])[0]
        acts = [int(a) for a in np.flatnonzero(m) if a >= A_SELECT]
        if st == BLIND_SELECT:
            return [a for a in acts if a in (A_SELECT, A_SKIP, A_REROLL_BOSS)]
        if st == PACK:
            return [a for a in acts if A_PICK <= a <= A_PSKIP]
        out = [a for a in acts if A_BUY <= a < A_LEAVE or a == A_LEAVE or A_SELL_J <= a < A_PICK]
        if len(out) > self.a_max:                                            # leave first, then the shop's own items by slot, then the rest
            keep = [A_LEAVE] + [a for a in out if A_REROLL == a]
            rest = [a for a in out if a not in keep]
            out = keep + rest[:self.a_max - len(keep)]
        return out

    def decide(self, R, obs, st, salt0, default=None, margin=0.5):
        acts = self.candidates(obs, st)
        if len(acts) <= 1:
            return acts[0] if acts else None
        self.decisions += 1
        K = self.k
        n = len(acts) * K
        idx_src = [0] * n
        self.S.copy_many(R, idx_src, list(range(n)))
        salts = [salt0 + (q % K) for q in range(n)]                          # the same K futures for every action (common random numbers)
        self.S.v.reseed(list(range(n)), salts)
        first_select = np.zeros(n, dtype=bool)
        step = np.full(self.S.n, -1, dtype=np.int64)
        for c, a in enumerate(acts):
            for k in range(K):
                q = c * K + k
                if a == A_SELECT:
                    first_select[q] = True                                   # imagined from the blind select, the first move forced
                else:
                    step[q] = a
        if (step[:n] >= 0).any():
            self.S.step(step)
        vals = self.imagine(n, first_select)
        score = vals.reshape(len(acts), K).mean(1)
        best = int(np.argmax(score))
        pick = acts[best]
        if default in acts and score[best] - score[acts.index(default)] < margin:       # not clearly better than what the network would do: stay with it
            pick = default
        return pick, dict(zip(acts, score.round(3).tolist()))


def run_game(seed, a, model, planner_on, log):
    R = make(1, a.deck, a.stake, seed, int(getattr(model, "marg", 0)))
    pl = Planner(a.deck, a.stake, seed, a.k, a.width, a.horizon, a.a_max) if planner_on else None
    t0 = time.time()
    info = None
    for step in range(6000):
        obs = {k: np.array(R.obs[k]) for k in OBS_KEYS}
        st = state_of(obs)
        mask = obs["mask"][0]
        targeting = st in (SHOP, PACK) and bool(mask[:2 * N_SUB].any())
        if pl is None or st == SELECTING_HAND or targeting:
            with torch.no_grad():
                logits, _ = model({k: torch.from_numpy(v) for k, v in obs.items() if k in OBS_KEYS})
            act = int(logits[0].argmax())
        else:
            with torch.no_grad():
                logits, _ = model({k: torch.from_numpy(v) for k, v in obs.items() if k in OBS_KEYS})
            default = int(logits[0].argmax())
            out = pl.decide(R, obs, st, 1_000_003 * (step + 1) + 17 * seed, default, a.margin)
            act = out[0] if isinstance(out, tuple) else out
            if act is None:
                act = A_LEAVE if st == SHOP else A_PSKIP if st == PACK else A_SELECT
        _, done, inf = R.step(np.array([act]))
        if done[0]:
            info = dict(blinds=int(inf["blinds"][0]), won=bool(inf["won"][0]), ante=int(inf["ante"][0]))
            break
    secs = time.time() - t0
    return info or dict(blinds=-1, won=False, ante=-1), secs, (pl.decisions if pl else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--seeds", default="60001-60008")
    p.add_argument("--model", default="checkpoints/ppo_gold_v22.pt")
    p.add_argument("--no-plan", action="store_true")
    p.add_argument("--k", type=int, default=4, help="redrawn futures per action")
    p.add_argument("--width", type=int, default=4, help="beam width inside an imagined future")
    p.add_argument("--horizon", type=int, default=3, help="blinds an imagined future is played on")
    p.add_argument("--margin", type=float, default=0.5, help="blinds an action must beat the network's own choice by to override it")
    p.add_argument("--a-max", type=int, default=10, help="actions tried at a decision (leave and reroll always)")
    p.add_argument("--out", default="")
    a = p.parse_args()
    torch.set_num_threads(2)
    model = load_model(a.model, "cpu")
    lo, hi = [int(x) for x in a.seeds.split("-")] if "-" in a.seeds else [int(a.seeds)] * 2
    rows = []
    for sd in range(lo, hi + 1):
        info, secs, dec = run_game(sd, a, model, not a.no_plan, print)
        rows.append(dict(seed=sd, secs=secs, decisions=dec, **info))
        print(f"seed {sd}: {info['blinds']:2d} blinds, ante {info['ante']}, {'WON' if info['won'] else 'lost'}  ({secs:.0f}s, {dec} planned decisions)", flush=True)
    b = [r["blinds"] for r in rows]
    print(f"\n{'network alone' if a.no_plan else 'planner'}: {len(rows)} games, mean blinds {np.mean(b):.2f}, wins {sum(r['won'] for r in rows)}, mean {np.mean([r['secs'] for r in rows]):.0f}s a game")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
