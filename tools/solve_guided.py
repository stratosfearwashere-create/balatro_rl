"""The shop-only beam search with better ranking: a learned value, and a short lookahead.

The seed's future is known (exact-seed mode); the cards are played by the scripted player (mode 2), so the search is only over what to buy, sell, reroll and skip. Each level keeps the `beam` best
states of the next blind select. How they are ranked decides which builds survive:

    score     the engine's own number: the build's expected best play (as in solve_novel)
    value     the PPO network's value head on the state (it predicted how far a game goes better than the exact scorer: rank correlation 0.76 against 0.69 at ante 4)
    look      the next --look blinds are played on a copy with the scripted player (no shopping in between): how many are cleared, and by what margin
    both      look first, then value, then the engine score

    python tools/solve_guided.py --rank both --beam 32 --seeds 42001-42020
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import make                                                     # noqa: E402
from solve_novel import pick                                                    # noqa: E402
from balatro_rl.engine_env import OBS_KEYS                                      # noqa: E402
from balatro_rl.env import A_LEAVE, A_SELECT                                    # noqa: E402
from balatro_rl.model import load_model                                         # noqa: E402

KEYS = [k for k in OBS_KEYS if k != "priv"]


class Ranker:
    def __init__(self, mode, model_path, look, dev, deck, stake, seed, cap):
        self.mode, self.look, self.dev = mode, look, dev
        self.model = load_model(model_path, dev) if mode in ("value", "both") else None
        self.scr = make(cap, deck, stake, seed, 1) if mode in ("value", "both", "look") else None
        self.la = make(cap, deck, stake, seed, 0) if mode in ("look", "both") else None

    @torch.no_grad()
    def values(self, n):
        o = {k: torch.from_numpy(np.ascontiguousarray(np.asarray(self.scr.obs[k])[:n])).to(self.dev) for k in KEYS}
        out = []
        for c0 in range(0, n, 512):
            sub = {k: v[c0:c0 + 512] for k, v in o.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.dev).startswith("cuda")):
                _, v = self.model(sub)
            out.append(v.float().cpu().numpy())
        return np.concatenate(out)

    def lookahead(self, n):
        """Plays the next blinds of the n states in self.la: -> (blinds cleared, summed margin (the best share of the target, at most 1.5 each))."""
        la = self.la
        la.set_auto_round(True, 1.5, 2)
        la.set_modes([2] * la.n)
        cleared, margin = np.zeros(n), np.zeros(n)
        alive = np.ones(n, dtype=bool)
        for _ in range(self.look):
            act = np.full(la.n, -1, dtype=np.int64)
            act[:n] = np.where(alive, A_SELECT, -1)
            _, d, _, _, _ = la.step_split(act)
            rep = la.round_report()
            ok = alive & rep["cleared"][:n] & ~d[:n]
            won = alive & rep["cleared"][:n] & d[:n]
            margin += np.where(alive, np.minimum(rep["ratio"][:n], 1.5), 0.0)
            cleared += (ok | won)
            alive = ok
            if not alive.any():
                break
            mask = np.asarray(la.obs["mask"])
            leave = np.full(la.n, -1, dtype=np.int64)
            leave[:n] = np.where(alive & mask[:n, A_LEAVE], A_LEAVE, -1)
            la.step_split(leave)
        return cleared, margin


def solve(deck, stake, seed, beam, elite, floor, ends_per, sw, sl, node_cap, rank, model_path, look, dev, max_levels=40):
    cur = make(beam, deck, stake, seed, 2)                  # (marginal 2: the columns the scaling bonus of the engine score reads, as in solve_novel)
    dest = make(beam * ends_per, deck, stake, seed, 2)
    root = make(1, deck, stake, seed, 2)
    cur.copy_many(root, [0], [0])
    rk = Ranker(rank, model_path, look, dev, deck, stake, seed, beam * ends_per) if rank != "score" else None
    paths, n, t0 = [[]], 1, time.time()
    for level in range(max_levels):
        res = cur.expand_level(list(range(n)), dest, ends_per, sw, sl, node_cap, 0.0, 0.0, 1.0, [], 0.0)
        pool = []
        for b, (win, wpath, cands) in enumerate(res):
            if win:
                return paths[b] + list(wpath), {"level": level, "secs": time.time() - t0}
            for path, score, sig, slot, jk in cands:
                pool.append([score, sig, paths[b] + list(path), slot, jk])
        if not pool:
            return None, {"level": level, "secs": time.time() - t0, "dead": True}
        if rk is not None:
            slots = [c[3] for c in pool]
            m = len(pool)
            if rk.scr is not None:
                rk.scr.copy_many(dest, slots, list(range(m)))
            if rk.la is not None:
                rk.la.copy_many(dest, slots, list(range(m)))
            v = rk.values(m) if rk.model is not None else np.zeros(m)
            cl, mg = rk.lookahead(m) if rk.la is not None else (np.zeros(m), np.zeros(m))
            sc = np.array([c[0] for c in pool])
            for i, c in enumerate(pool):                                  # the ranking number: look-ahead first, then the value, then the engine's own score
                if rank == "value":
                    c[0] = v[i] + 1e-3 * sc[i]
                elif rank == "look":
                    c[0] = 10.0 * cl[i] + mg[i] + 1e-3 * sc[i]
                else:
                    c[0] = 10.0 * cl[i] + mg[i] + v[i] + 1e-3 * sc[i]
        keep = pick([tuple(c) for c in pool], beam, elite, floor)
        cur.copy_many(dest, [c[3] for c in keep], list(range(len(keep))))
        paths, n = [c[2] for c in keep], len(keep)
    return None, {"level": max_levels, "secs": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42001-42020")
    p.add_argument("--rank", default="score", choices=["score", "look"])
    p.add_argument("--beam", type=int, default=32)
    p.add_argument("--shop-layers", type=int, default=4)
    p.add_argument("--look", type=int, default=2)
    p.add_argument("--mode", type=int, default=2, help="the scripted player's mode (rounds are played by it: the cards are not searched)")
    p.add_argument("--model", default="checkpoints/ppo_gold_v30_watch.pt")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    make(1, a.deck, a.stake, 1, 1).set_level_scripted(a.mode)
    lo, hi = [int(x) for x in a.seeds.split("-")]
    solved, levels, t0 = [], [], time.time()
    for sd in range(lo, hi + 1):
        path, info = solve(a.deck, a.stake, sd, a.beam, a.beam // 2, 0.5, 8, 4, a.shop_layers, 1000, a.rank, a.model, a.look, a.device)
        if path:
            solved.append(sd)
        levels.append(24 if path else info["level"])
    print(f"rank by {a.rank:6s} (look {a.look if a.rank in ('look', 'both') else '-'}), beam {a.beam}, layers {a.shop_layers}: solved {len(solved)}/{hi - lo + 1} {solved}, mean level {np.mean(levels):.2f}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
