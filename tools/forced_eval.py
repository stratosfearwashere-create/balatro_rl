"""A causal test before training: does a policy do better if it always takes a kind of joker when it can?

The checkpoint plays greedy Gold games; a rule overrides it at every shop / pack step where a joker of the chosen kind is legal to take: buy it if there is a slot, else replace the
held joker the rule values least (one that is not of the kind). Everything else is the network's choice. Run on the same seeds as the plain policy (the --rule none line).

  rules: none | xmult (joker_desc 'xmult' jokers) | strong (measured gain at the ante-4 boss >= 0.4, balatro_rl/joker_strength.json) | scaler (joker_desc 'grows') | econ (joker_desc 'economy')

    python tools/forced_eval.py checkpoints/ppo_gold_v30_best.pt --games 300 --rules none,xmult,strong
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import env as E                                                  # noqa: E402
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS                            # noqa: E402
from balatro_rl.env import item_id                                               # noqa: E402
from balatro_rl.joker_desc import FEATURES, item_descriptors                     # noqa: E402
from balatro_rl.model import load_model                                          # noqa: E402
from balatro_rl.sim.jokers import JOKERS                                         # noqa: E402

D = item_descriptors()


def kind_set(rule):
    if rule == "none":
        return set()
    if rule == "strong":
        tab = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "balatro_rl", "joker_strength.json")))["jokers"]
        return {item_id("j_" + k) for k, v in tab.items() if v["measured"] and v["gains"][1] >= 0.4}
    col = FEATURES.index({"xmult": "xmult", "scaler": "grows", "econ": "economy"}[rule])
    return {item_id("j_" + k) for k in JOKERS if D[item_id("j_" + k), col] > 0.5}


def sticker_cols():
    """Columns of the action features: an offered joker's eternal, perishable (rounds left / 5) and rental flags (env._item_feats)."""
    off = len(E.ACTION_TYPES) + 1 + E.N_HANDS + 5 + 6
    base = off + 2 + len(E.EDITIONS)
    return base, base + 1, base + 2


ACQUIRE = None


def play(model, rule, games, seed0, stake, device):
    global ACQUIRE
    if ACQUIRE is None:
        ACQUIRE = np.zeros(E.N_ACTIONS, dtype=bool)
        ACQUIRE[E.A_BUY:E.A_BUY + E.MAX_SHOP] = True
        ACQUIRE[E.A_PICK:E.A_PSKIP] = True
        ACQUIRE[E.A_REPLACE:] = True
    mask_cols = []
    reserve = int(rule.split("_")[1]) if rule.startswith("reserve_") else 0
    SPEND = np.zeros(E.N_ACTIONS, dtype=bool)
    SPEND[E.A_BUY:E.A_VOUCHER + 2] = True
    SPEND[E.A_REROLL] = True
    SPEND[E.A_REPLACE:] = True
    if rule.startswith("mask_"):                          # never take a joker with these stickers: the network's best other action
        c_et, c_pe, c_re = sticker_cols()
        mask_cols = {"mask_rental": [c_re], "mask_perish": [c_pe], "mask_both": [c_re, c_pe], "mask_all": [c_re, c_pe, c_et]}[rule]
    want = kind_set(rule) if not (rule.startswith("mask_") or rule.startswith("reserve_") or rule in ("no_packs", "no_rerolls") or rule.startswith("skip_")) else set()
    v = CppVecEnv(games, "RED", stake, seed0=seed0, threads=0)
    if getattr(model, "marg", 0):
        v.set_marginal(int(model.marg))
    blinds = np.full(games, -1)
    won = np.zeros(games, dtype=bool)
    forced = 0
    for _ in range(10000):
        if (blinds >= 0).all():
            break
        ob = {k: torch.from_numpy(np.asarray(v.obs[k])).to(device) for k in OBS_KEYS}
        with torch.no_grad():
            logits, _ = model(ob)
        if mask_cols:
            af = torch.from_numpy(np.asarray(v.obs["af"])[:, :, mask_cols].astype(np.float32)).to(device)
            bad = (af > 1e-6).any(-1) & torch.from_numpy(ACQUIRE).to(device)[None, :]
            logits = logits.masked_fill(bad, -1e9)
        if rule.startswith("skip_"):                      # skip a small / big blind when its tag is in the set
            from balatro_rl.env import STATES, BOSS_KEYS
            from balatro_rl.sim.items import TAGS
            off = 1 + 3 + len(STATES) + 17 + len(BOSS_KEYS) + 3
            sets = {"skip_money": {"economy", "investment"}, "skip_good": {"economy", "investment", "rare", "negative", "polychrome", "holo", "double", "coupon", "charm"},
                    "skip_inv": {"investment"}}
            ids = [TAGS.index(t) for t in sets[rule]]
            g = np.asarray(v.obs["g"])
            mask = np.asarray(v.obs["mask"])
            hit = (g[:, [off + i for i in ids]] > 0.5).any(1) & mask[:, E.A_SKIP]
            act_override = np.where(hit, E.A_SKIP, -1)
        else:
            act_override = None
        if rule in ("no_packs", "no_rerolls"):
            bad = np.zeros(E.N_ACTIONS, dtype=bool)
            if rule == "no_packs":
                bad[E.A_BUY_PACK:E.A_VOUCHER] = True
            else:
                bad[E.A_REROLL] = True
            logits = logits.masked_fill(torch.from_numpy(bad).to(device)[None, :], -1e9)
        if reserve:                                       # from ante 2: no spend that leaves less than the reserve (rerolls priced at $5)
            g = np.asarray(v.obs["g"])
            money = np.rint(g[:, 17] * 50)
            ante = np.rint(g[:, 0] * 8)
            cost = np.asarray(v.obs["af"])[:, :, 40] * 10.0
            cost[:, E.A_REROLL] = np.maximum(cost[:, E.A_REROLL], 5.0)
            bad = SPEND[None, :] & ((money[:, None] - cost) < reserve) & (ante[:, None] >= 2)
            logits = logits.masked_fill(torch.from_numpy(bad).to(device), -1e9)
        act = logits.argmax(1).cpu().numpy()
        if act_override is not None:
            forced += int((act_override >= 0).sum())
            act = np.where(act_override >= 0, act_override, act)
        if want:
            mask, aid, aid2 = np.asarray(v.obs["mask"]), np.asarray(v.obs["aid"]), np.asarray(v.obs["aid2"])
            for e in range(games):
                if blinds[e] >= 0:
                    continue
                if int(aid[e, act[e]]) in want and mask[e, act[e]]:
                    continue                                     # the network takes one already
                pick = None
                for s in range(E.MAX_SHOP):                      # a buy with a free slot
                    ai = E.A_BUY + s
                    if mask[e, ai] and int(aid[e, ai]) in want:
                        pick = ai
                        break
                if pick is None:
                    for i in range(E.A_PICK, E.A_PSKIP):         # a pack pick
                        if mask[e, i] and int(aid[e, i]) in want:
                            pick = i
                            break
                if pick is None:                                 # a replace that gives up a joker not of the kind (the cheapest to sell: the first such)
                    for s in range(E.MAX_SHOP):
                        for j in range(E.MAX_JOKERS):
                            ai = E.A_REPLACE + s * E.MAX_JOKERS + j
                            if mask[e, ai] and int(aid[e, ai]) in want and int(aid2[e, ai]) not in want:
                                pick = ai
                                break
                        if pick is not None:
                            break
                if pick is not None:
                    act[e] = pick
                    forced += 1
        act[blinds >= 0] = -1
        _, done, info = v.step(act)
        for e in np.flatnonzero(done):
            if blinds[e] < 0:
                blinds[e] = info["blinds"][e]
                won[e] = info["won"][e]
    return blinds, won, forced


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--rules", default="none,xmult,strong,scaler")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    model = load_model(a.checkpoint, a.device)
    base = None
    for rule in a.rules.split(","):
        b, w, f = play(model, rule, a.games, a.seed0, a.stake, a.device)
        line = f"{rule:7s} {b.mean():6.2f} blinds  {int(w.sum()):3d} wins  ante>=5 {100 * (b >= 12).mean():4.0f}%  forced {f / a.games:5.2f}/game"
        if base is None:
            base = b
        else:
            d = b - base
            line += f"   paired {d.mean():+.2f} +- {d.std() / np.sqrt(len(d)):.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
