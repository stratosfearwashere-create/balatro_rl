"""When a strong joker is on offer, does the policy take it? And what does it do with its money and its slots?

Greedy games of a checkpoint on fixed seeds. At every shop step the jokers the policy could buy (legal buy actions, so affordable with a free slot) or swap in (legal replace actions) are looked up
in the measured strength table (balatro_rl/joker_strength.json: the gain of a joker at the ante-4 boss). A joker is STRONG when its gain at ante 4 is at least --strong (default 0.4). For each such step:
what the policy did (bought it / replaced a joker with it / bought another strong one / bought something else / rerolled / left), by ante. Also: money at the shop when a strong joker was offered but
the policy left, and how many of the strong jokers on offer were not legal to buy (the item is visible in the shop but unaffordable or no slot, read from the engine's shop list).

    python tools/offer_take.py checkpoints/ppo_gold_v30_best.pt --games 100
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import env as E                                                  # noqa: E402
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS                            # noqa: E402
from balatro_rl.env import item_id                                               # noqa: E402
from balatro_rl.model import load_model                                          # noqa: E402
from balatro_rl.sim.jokers import JOKERS                                         # noqa: E402
from loss_analysis import action_kind                                            # noqa: E402

ITEM2KEY = {item_id("j_" + k): k for k in JOKERS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--strong", type=float, default=0.4)
    p.add_argument("--sample", action="store_true", help="sample from the policy instead of the greedy action")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    tab = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "balatro_rl", "joker_strength.json")))["jokers"]
    strong = {k for k, v in tab.items() if v["measured"] and v["gains"][1] >= a.strong}
    print(f"{len(strong)} strong jokers (gain at the ante-4 boss >= {a.strong}): {', '.join(sorted(strong, key=lambda k: -tab[k]['gains'][1])[:14])} ...")
    model = load_model(a.checkpoint, a.device)
    N = a.games
    venv = CppVecEnv(N, "RED", a.stake, seed0=a.seed0, threads=0)
    if getattr(model, "marg", 0):
        venv.set_marginal(int(model.marg))
    gen = torch.Generator(device=a.device).manual_seed(0)
    finished = np.zeros(N, dtype=bool)
    res = defaultdict(Counter)                           # ante -> outcome counts, over shop steps where a strong joker was legal to take
    left_money = defaultdict(list)
    free_slots = defaultdict(list)
    seen_visits = defaultdict(int)
    for _ in range(8000):
        if finished.all():
            break
        ob = {k: torch.from_numpy(np.asarray(venv.obs[k])).to(a.device) for k in OBS_KEYS}
        mask, aid, g = np.asarray(venv.obs["mask"]), np.asarray(venv.obs["aid"]), np.asarray(venv.obs["g"])
        with torch.no_grad():
            logits, _ = model(ob)
        if a.sample:
            act = torch.multinomial(torch.softmax(logits.float(), 1), 1, generator=gen).squeeze(1).cpu().numpy()
        else:
            act = logits.argmax(1).cpu().numpy()
        for e in range(N):
            if finished[e] or not mask[e, E.A_LEAVE]:
                continue
            ante = int(round(g[e, 0] * 8))
            offered = []
            for s in range(E.MAX_SHOP):
                ai = E.A_BUY + s
                if mask[e, ai] and int(aid[e, ai]) in ITEM2KEY and ITEM2KEY[int(aid[e, ai])] in strong:
                    offered.append(ai)
            rep = []
            for s in range(E.MAX_SHOP):
                for j in range(E.MAX_JOKERS):
                    ai = E.A_REPLACE + s * E.MAX_JOKERS + j
                    if mask[e, ai] and int(aid[e, ai]) in ITEM2KEY and ITEM2KEY[int(aid[e, ai])] in strong:
                        rep.append(ai)
            if not offered and not rep:
                continue
            seen_visits[ante] += 1
            ai = int(act[e])
            kind = action_kind(ai)
            if ai in offered:
                out = "bought a strong joker"
            elif ai in rep:
                out = "swapped a strong joker in"
            elif kind == "buy":
                out = "bought something else"
            elif kind == "replace":
                out = "swapped in something else"
            elif kind == "reroll":
                out = "rerolled"
            elif kind == "leave":
                out = "left"
            elif kind in ("buy_pack", "voucher"):
                out = "bought a pack / voucher"
            else:
                out = "other (sell, use, pick ...)"
            res[ante][out] += 1
            if out == "left":
                left_money[ante].append(int(round(g[e, 17] * 50)))
        _, done, _ = venv.step(act)
        finished |= done
    outs = ["bought a strong joker", "swapped a strong joker in", "bought something else", "swapped in something else", "bought a pack / voucher", "rerolled", "left", "other (sell, use, pick ...)"]
    print(f"\nshop steps where a strong joker was legal to take (bought or swapped in), by ante, {a.games} greedy games of {os.path.basename(a.checkpoint)}")
    print(f"{'ante':>4s} {'steps':>6s}  " + "  ".join(f"{o[:16]:>16s}" for o in outs))
    for ante in sorted(res):
        n = sum(res[ante].values())
        print(f"{ante:4d} {n:6d}  " + "  ".join(f"{100 * res[ante][o] / n:15.0f}%" for o in outs))
    tot = Counter()
    for ante in res:
        tot.update(res[ante])
    n = sum(tot.values())
    if n:
        print(f"all  {n:6d}  " + "  ".join(f"{100 * tot[o] / n:15.0f}%" for o in outs))
    for ante in sorted(left_money):
        print(f"ante {ante}: money held on the steps it LEFT with a strong joker legal to take: median {np.median(left_money[ante]):.0f}, n {len(left_money[ante])}")


if __name__ == "__main__":
    main()
