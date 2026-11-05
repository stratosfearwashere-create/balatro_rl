"""What builds a checkpoint makes: greedy games, optionally under the rule changes (--no-stickers, --small-blind-pay).

For every game the build at its first blind select of ante 4 and its last build before the game ended are read (VecEnv.cell_info): jokers, hand levels, money.
Reported: blinds and wins; where games end; the most common jokers at ante 4 and at the end, split into games that reached ante 6+ and the rest, with each joker's
measured strength role (seed_report.role_of); the share of builds holding a multiplier / scaler / copy joker; the top hand level.

    python tools/builds_report.py checkpoints/ppo_gold_v39.pt --games 300 --no-stickers --small-blind-pay
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("BALATRO_DEV_ENGINE", "")
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS                            # noqa: E402
from balatro_rl.env import A_SELECT                                              # noqa: E402
from balatro_rl.model import load_model                                          # noqa: E402
from balatro_rl.sim.jokers import JOKERS                                         # noqa: E402
from seed_report import role_of                                                  # noqa: E402

KEYS = list(JOKERS)


def jokers_of(inf):
    return [KEYS[inf[8 + 3 * i]] for i in range(inf[7])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--no-stickers", action="store_true")
    p.add_argument("--small-blind-pay", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    model = load_model(a.checkpoint, a.device)
    N = a.games
    v = CppVecEnv(N, "RED", a.stake, seed0=a.seed0, threads=0)
    if getattr(model, "marg", 0):
        v.set_marginal(int(model.marg))
    if a.no_stickers or a.small_blind_pay:
        v.set_rules(a.no_stickers, a.small_blind_pay)
        v.v.restart()
    at4 = [None] * N
    last = [None] * N
    peak_money = np.zeros(N)
    done_ep = np.zeros(N, dtype=bool)
    blinds = np.zeros(N, dtype=int)
    won = np.zeros(N, dtype=bool)
    ante_end = np.zeros(N, dtype=int)
    for _ in range(10000):
        if done_ep.all():
            break
        mask, g = np.asarray(v.obs["mask"]), np.asarray(v.obs["g"])
        sel = [e for e in range(N) if not done_ep[e] and mask[e, A_SELECT]]
        if sel:
            infos = v.cell_info(sel)
            lv = v.hand_levels(sel)
            for e, inf, l in zip(sel, infos, lv):
                last[e] = (inf, l)
                if at4[e] is None and inf[0] == 4:
                    at4[e] = (inf, l)
        live = ~done_ep
        peak_money[live] = np.maximum(peak_money[live], np.rint(g[live, 17] * 50))
        ob = {k: torch.from_numpy(np.asarray(v.obs[k])).to(a.device) for k in OBS_KEYS}
        with torch.no_grad():
            logits, _ = model(ob)
        act = logits.argmax(1).cpu().numpy()
        act[done_ep] = -1
        _, done, info = v.step(act)
        for e in np.flatnonzero(done):
            if not done_ep[e]:
                done_ep[e] = True
                blinds[e], won[e], ante_end[e] = info["blinds"][e], info["won"][e], info["ante"][e]
    rules = ", ".join(x for x, on in (("no stickers", a.no_stickers), ("small blind pays", a.small_blind_pay)) if on) or "normal rules"
    print(f"{os.path.basename(a.checkpoint)}: {N} greedy {a.stake} games ({rules}): {blinds.mean():.2f} blinds, {int(won.sum())} wins ({100 * won.mean():.1f}%)")
    print("  games end at ante: " + ", ".join(f"{k}: {int((ante_end == k).sum())}" for k in range(1, 9)) + f"; peak money median ${np.median(peak_money):.0f}")
    deep = ante_end >= 6

    def summary(which, label, mask_games):
        cnt, roles, n, lvl = Counter(), Counter(), 0, []
        for e in np.flatnonzero(mask_games):
            b = which[e]
            if b is None:
                continue
            inf, l = b
            js = jokers_of(inf)
            n += 1
            cnt.update(js)
            rs = {role_of(j) for j in js}
            for r in ("xmult", "scaler", "copy", "econ"):
                roles[r] += r in rs
            lvl.append(max(l))
        if not n:
            return
        print(f"  {label} ({n} games): holds a multiplier {100 * roles['xmult'] / n:.0f}%, scaler {100 * roles['scaler'] / n:.0f}%, copy {100 * roles['copy'] / n:.0f}%, "
              f"economy {100 * roles['econ'] / n:.0f}%; top hand level {np.mean(lvl):.1f}")
        print("     " + ", ".join(f"{k} ({role_of(k)}) {c}" for k, c in cnt.most_common(12)))
    summary(at4, "at ante 4, games that went on to reach ante 6+", deep)
    summary(at4, "at ante 4, games that died before ante 6", ~deep)
    summary(last, "last build, games that reached ante 6+", deep)


if __name__ == "__main__":
    main()
