"""Where a policy's money comes from and where it goes, by ante.

Greedy games of a checkpoint; the money before and after every action is read (g column 17 x 50) and the change is put down to the action's kind: spent on jokers / cards, packs, rerolls,
vouchers, replaces (buy and sell in one); got from selling, and from playing a round (the blind's payout, interest, unused hands and jokers' money all arrive with the round's end, which
this counts as "round income"). Also the money held when leaving each shop, against the interest steps ($5 each, up to $25).

    python tools/money_flow.py checkpoints/ppo_gold_v30_best.pt --games 300
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import env as E                                                  # noqa: E402
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS                            # noqa: E402
from balatro_rl.model import load_model                                          # noqa: E402
from loss_analysis import action_kind                                            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    model = load_model(a.checkpoint, a.device)
    N = a.games
    v = CppVecEnv(N, "RED", a.stake, seed0=a.seed0, threads=0)
    if getattr(model, "marg", 0):
        v.set_marginal(int(model.marg))
    done_ep = np.zeros(N, dtype=bool)
    flow = defaultdict(lambda: defaultdict(float))          # ante -> kind -> dollars (+ got, - spent)
    count = defaultdict(lambda: defaultdict(int))
    games_at = defaultdict(set)
    leave_money = defaultdict(list)
    for _ in range(10000):
        if done_ep.all():
            break
        g = np.asarray(v.obs["g"])
        money0 = np.rint(g[:, 17] * 50).astype(int)
        ante = np.rint(g[:, 0] * 8).astype(int)
        ob = {k: torch.from_numpy(np.asarray(v.obs[k])).to(a.device) for k in OBS_KEYS}
        with torch.no_grad():
            logits, _ = model(ob)
        act = logits.argmax(1).cpu().numpy()
        act[done_ep] = -1
        _, done, _ = v.step(act)
        money1 = np.rint(np.asarray(v.obs["g"])[:, 17] * 50).astype(int)
        for e in range(N):
            if done_ep[e] or done[e]:
                continue
            k = action_kind(int(act[e]))
            if k in ("play", "discard"):
                k = "round income (payout, interest, jokers)" if money1[e] != money0[e] else None
            if k is None:
                continue
            if k == "leave":
                leave_money[ante[e]].append(money0[e])
            flow[ante[e]][k] += money1[e] - money0[e]
            count[ante[e]][k] += 1
            games_at[ante[e]].add(e)
        done_ep |= done
    kinds = ["round income (payout, interest, jokers)", "buy", "buy_pack", "reroll", "voucher", "replace", "sell_joker", "sell_cons", "select", "skip", "use", "pick"]
    print(f"{os.path.basename(a.checkpoint)}: dollars per game that reached each ante (+ got, - spent), {N} greedy Gold games")
    print(f"{'ante':>4s} {'games':>5s}  " + "  ".join(f"{k[:12]:>12s}" for k in kinds))
    for an in sorted(flow):
        n = len(games_at[an])
        print(f"{an:4d} {n:5d}  " + "  ".join(f"{flow[an][k] / n:+12.1f}" for k in kinds))
    print("\nmoney held when leaving the shop (the interest steps are $5, $10 ... $25):")
    for an in sorted(leave_money):
        m = np.array(leave_money[an])
        print(f"  ante {an}: median ${np.median(m):.0f}, mean ${m.mean():.1f}; under $5 {100 * (m < 5).mean():.0f}%, $25+ {100 * (m >= 25).mean():.0f}%  (n {len(m)})")


if __name__ == "__main__":
    main()
