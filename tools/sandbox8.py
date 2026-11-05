"""How well can a policy build when nothing holds it back? The ante-8 boss with unlimited resources.

Every game starts in the shop of ante 8 (VecEnv.set_start_ante) with $1,000,000 (Bull and Bootstraps banned: they turn money into mult), rerolls capped only at --rerolls per
shop visit (so every joker can be found), packs restocked --restock times per visit and weighted toward Arcana and Spectral (the tarots and spectrals that change the deck).
When the policy leaves the shop the game is put straight at the ante-8 boss (VecEnv.set_state). The policy plays everything, greedily.

Reported per checkpoint: the share of games that beat the boss, the build's POWER at the boss (the exact solver's best share of the target with that build: it sees the draws,
an upper bound on play), what was bought, and the most common jokers.

    python tools/sandbox8.py checkpoints/backup_v30/ppo_gold_v30_best.pt checkpoints/ppo_gold_v37.pt --games 200
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS                            # noqa: E402
from balatro_rl.env import A_SELECT                                              # noqa: E402
from balatro_rl.model import load_model                                          # noqa: E402
from balatro_rl.sim.jokers import JOKERS                                         # noqa: E402
from loss_analysis import action_kind                                            # noqa: E402

KEYS = list(JOKERS)


def run(path, games, seed0, stake, rerolls, restock, money, device, start_ante=8, level_cap=0, ban=("bull", "bootstraps")):
    model = load_model(path, device)
    v = CppVecEnv(games, "RED", stake, seed0=seed0, threads=0)
    if getattr(model, "marg", 0):
        v.set_marginal(int(model.marg))
    v.set_start_ante(start_ante)
    v.set_start(0.0, 0.0, 1.0, money)
    v.set_ban([KEYS.index(k) for k in ban], 1.0)
    v.set_caps(rerolls, level_cap, 0)
    v.set_sandbox(0, restock, (3.0, 1.0, 1.0, 1.0, 3.0))
    v.v.restart()
    scratch = CppVecEnv(games, "RED", stake, seed0=1, threads=0)
    if getattr(model, "marg", 0):
        scratch.set_marginal(int(model.marg))
    at_boss = np.zeros(games, dtype=bool)
    done_ep = np.zeros(games, dtype=bool)
    won = np.zeros(games, dtype=bool)
    acts = [Counter() for _ in range(games)]
    power = np.full(games, np.nan)
    builds = [None] * games
    died_ante = np.zeros(games, dtype=int)
    for _ in range(4000):
        if done_ep.all():
            break
        mask = np.asarray(v.obs["mask"])
        jump = [e for e in range(games) if not done_ep[e] and not at_boss[e] and mask[e, A_SELECT]] if start_ante == 8 else []
        if start_ante < 8:                                # played from start_ante: the build at the first blind select of ante 8
            g = np.asarray(v.obs["g"])
            reach = [e for e in range(games) if not done_ep[e] and not at_boss[e] and mask[e, A_SELECT] and int(round(g[e, 0] * 8)) == 8]
            if reach:
                at_boss[reach] = True
                for e, inf in zip(reach, v.cell_info(reach)):
                    builds[e] = (inf, v.hand_levels([e])[0])
        if jump:                                          # out of the shop: straight to the ante-8 boss
            v.set_state(jump, 8, 2)
            at_boss[jump] = True
            info = v.cell_info(jump)
            for e, inf in zip(jump, info):
                builds[e] = (inf, v.hand_levels([e])[0])
        ob = {k: torch.from_numpy(np.asarray(v.obs[k])).to(device) for k in OBS_KEYS}
        with torch.no_grad():
            logits, _ = model(ob)
        act = logits.argmax(1).cpu().numpy()
        act[done_ep] = -1
        g_now = np.asarray(v.obs["g"])
        sel = [e for e in range(games) if at_boss[e] and not done_ep[e] and act[e] == A_SELECT and np.isnan(power[e])
               and (start_ante == 8 or (int(round(g_now[e, 0] * 8)) == 8 and g_now[e, 3] > 0.5))]       # (from ante 4: measured at the ante-8 boss)
        for e in range(games):
            if not done_ep[e] and (not at_boss[e] or start_ante < 8):
                acts[e][action_kind(int(act[e]))] += 1
        _, done, info = v.step(act)
        live = [e for e in sel if not done[e]]
        if live:                                          # the boss just began: the build's power, on a copy
            scratch.copy_many(v, live, live)
            r = scratch.solve_best(live, 1000)
            power[live] = r[:, 0]
        for e in np.flatnonzero(done):
            if not done_ep[e]:
                done_ep[e] = True
                won[e] = bool(info["won"][e])
                died_ante[e] = int(info["ante"][e])
    ok = ~np.isnan(power)
    jc = Counter()
    nj, lv = [], []
    for b in builds:
        if b is None:
            continue
        inf, levels = b
        nj.append(inf[7])
        lv.append(max(levels))
        jc.update(KEYS[inf[8 + 3 * i]] for i in range(inf[7]))
    mean_acts = {k: np.mean([a[k] for a in acts]) for k in ("buy", "reroll", "buy_pack", "pick", "use", "voucher", "sell_joker", "replace", "swap", "leave")}
    where = f"from the ante-{start_ante} shop" if start_ante < 8 else "ante-8 boss"
    print(f"\n{os.path.basename(path)}: {games} games, {where} with ${money:,}: WON {int(won.sum())} ({100 * won.mean():.1f}%)")
    if start_ante < 8:
        lost = died_ante[~won]
        print("  losses by ante: " + ", ".join(f"{k}: {int((lost == k).sum())}" for k in range(start_ante, 9)) + f"; reached ante 8: {int(at_boss.sum())}")
    if ok.any():
        lp = np.log10(np.maximum(power[ok], 1e-3))
        print(f"  build power at the boss (log10 best share, solver sees the draws): median {np.median(lp):+.2f}, p25 {np.percentile(lp, 25):+.2f}, p75 {np.percentile(lp, 75):+.2f}; "
              f"{100 * (power[ok] >= 1).mean():.0f}% of builds CAN beat it")
    print(f"  jokers held {np.mean(nj):.1f}, top hand level {np.mean(lv):.1f}; shop actions per game: " + ", ".join(f"{k} {v_:.1f}" for k, v_ in mean_acts.items()))
    print("  most held jokers: " + ", ".join(f"{k} {c}" for k, c in jc.most_common(15)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--rerolls", type=int, default=200)
    p.add_argument("--restock", type=int, default=20)
    p.add_argument("--money", type=int, default=1_000_000)
    p.add_argument("--ban", default="bull,bootstraps", help="jokers out of the pool")
    p.add_argument("--level-cap", type=int, default=0, help="the highest level a planet takes a hand to (0: none)")
    p.add_argument("--start-ante", type=int, default=8, help="8: straight to the ante-8 boss; lower: played from this ante's shop to the end")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    for ck in a.checkpoints:
        run(ck, a.games, a.seed0, a.stake, a.rerolls, a.restock, a.money, a.device, a.start_ante, a.level_cap, a.ban.split(","))


if __name__ == "__main__":
    main()
