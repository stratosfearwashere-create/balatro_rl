"""What does the policy build when every joker is on offer? (1) its preference over every joker at the first shop of a seed: the shop copied many times, each copy
rerolled into a different 24-joker menu (VecEnv.set_menu) on a redrawn future, the buy logit of each joker minus the leave logit averaged over its appearances;
(2) the seed played through with a 24-joker shop at every visit (greedy, every round by the exact solver), logging each purchase.

    python tools/joker_menu.py --ckpt checkpoints/ppo_gold_v47.pt --seed 43003
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import env as E                                  # noqa: E402
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS            # noqa: E402
from balatro_rl.model import load_model                          # noqa: E402
from balatro_rl.runeval import Cols                              # noqa: E402
from balatro_rl.sim.items import VOCAB                           # noqa: E402


def name(i):
    return VOCAB[int(i)][2:] if 0 < int(i) < len(VOCAB) else "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v47.pt")
    p.add_argument("--seed", type=int, default=43003)
    p.add_argument("--copies", type=int, default=512)
    p.add_argument("--money", type=int, default=50, help="extra starting money for part 1 (pays the rerolls; the policy has seen +$50 starts in training)")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    net = load_model(a.ckpt, "cpu").eval()

    def mk(n, seed, money=0):
        v = CppVecEnv(n, "RED", "GOLD", seed0=seed, threads=a.threads)
        if getattr(net, "marg", 0):
            v.set_marginal(int(net.marg))
        v.set_autoplay(True, 400, True)
        v.v.set_solver_tarots(True, 3)
        if money:
            v.set_start(0.0, 0.0, 1.0, money)
        v.set_menu(24)
        v.v.restart()
        return v

    def obs(v):
        return {k: torch.from_numpy(np.asarray(v.obs[k])) for k in OBS_KEYS}

    def in_shop(v, i=0):
        return int(np.argmax(np.asarray(v.obs["g"])[i, Cols.STATE:Cols.STATE + 4])) == 2

    # part 1: preferences at the first shop
    v = mk(1, a.seed, a.money)
    for _ in range(50):
        if in_shop(v):
            break
        with torch.no_grad():
            lg, _ = net(obs(v))
        v.step(np.array([int(lg.argmax(1))]))
    S = mk(a.copies, a.seed + 1)
    S.copy_many(v, [0] * a.copies, list(range(a.copies)))
    S.reseed_rng(list(range(a.copies)), [1_000_003 * (k + 1) for k in range(a.copies)])
    S.step(np.full(a.copies, E.A_REROLL, dtype=np.int64))          # a fresh 24-joker menu in every copy
    o = obs(S)
    with torch.no_grad():
        lg, _ = net(o)
    lg = lg.numpy()
    aid = o["aid"].numpy()
    mask = o["mask"].numpy()
    pref = defaultdict(list)
    for c in range(a.copies):
        if not mask[c, E.A_LEAVE]:
            continue
        for s in range(E.MAX_SHOP):
            act = E.A_BUY + s
            if mask[c, act] and VOCAB[int(aid[c, act])].startswith("j_"):
                pref[name(aid[c, act])].append(lg[c, act] - lg[c, E.A_LEAVE])
    rows = sorted(((np.mean(x), k, len(x)) for k, x in pref.items()), reverse=True)
    print(f"part 1: seed {a.seed}, first shop (ante 1, ${a.money} extra to pay the reroll), {len(rows)} jokers seen in {a.copies} rerolled 24-joker menus")
    print("  most wanted (buy logit - leave logit, mean over appearances):")
    for m, k, n in rows[:15]:
        print(f"    {k:22s} {m:+6.2f}  ({n})")
    print("  least wanted:", ", ".join(f"{k} {m:+.1f}" for m, k, n in rows[-8:]))

    # part 2: the seed played with a 24-joker shop every visit
    v = mk(1, a.seed)
    buys, ante_seen = [], set()
    for _ in range(4000):
        o = obs(v)
        ante = int(np.clip(round(float(o["g"][0, Cols.ANTE]) * 8), 1, 8))
        with torch.no_grad():
            lg, _ = net(o)
        act = int(lg.argmax(1))
        if E.A_BUY <= act < E.A_BUY + E.MAX_SHOP or act >= E.A_REPLACE:
            sl = act - E.A_BUY if act < E.A_BUY + E.MAX_SHOP else (act - E.A_REPLACE) // E.MAX_JOKERS
            buys.append((ante, name(o["aid"][0, act]) if act < E.A_BUY + E.MAX_SHOP else "replace -> " + name(o["aid"][0, act])))
        if int(np.argmax(o["g"][0, Cols.STATE:Cols.STATE + 4].numpy())) == 0 and ante not in ante_seen:
            ante_seen.add(ante)
            js = [name(j) for j in o["jid"][0].numpy() if j > 0 and VOCAB[int(j)].startswith("j_")]
            print(f"  entering ante {ante}: jokers {js}")
        _, d, info = v.step(np.array([act]))
        if d[0]:
            print(f"part 2 result: {'WON' if info['won'][0] else 'lost'} after {int(info['blinds'][0])} blinds (furthest ante {int(info['ante'][0])})")
            break
    by = defaultdict(list)
    for an, k in buys:
        by[an].append(k)
    for an in sorted(by):
        print(f"  ante {an} buys: {', '.join(by[an])}")


if __name__ == "__main__":
    main()
