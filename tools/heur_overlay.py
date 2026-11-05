"""Gap-study rules on top of a PPO policy: the policy decides outside rounds; these rules override it where they apply (GAP_STUDY.md).

    xmult    from ante 2: buy an affordable xMult joker in the shop (or swap out the weakest flat joker when the slots are full); pick one from a
             Buffoon pack
    planet   buy the main hand's planet when affordable and a consumable slot is free; pick it from a Celestial pack; use a held one at once
             (main hand: the most played so far, Pair before anything is played)
    rental   no rental joker bought before ante 5
    noskip   no blind skipped (except for the Investment tag)
    bank     from ante 3, nothing that leaves less than $25 except the xmult and planet buys above
    xmultm   like xmult, but only a buy / swap / pick that raises the build's measured strength against the boss (fast_power, common random
             numbers) by --gain or more: a multiplier that will fire with this build

    python tools/heur_overlay.py --games 1000 --rules xmult,planet,rental,noskip,bank --mode honest
    python tools/heur_overlay.py --games 1000 --rules none                       # the policy alone
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS                                  # noqa: E402
from balatro_rl.env import (A_BUY, A_BUY_PACK, A_PICK, A_PSKIP, A_REPLACE, A_REROLL, A_SELECT, A_SKIP,  # noqa: E402
                            A_USE_C, A_VOUCHER, MAX_JOKERS)
from balatro_rl.model import load_model                                                # noqa: E402
from balatro_rl.sim.game import HAND_TO_PLANET                                         # noqa: E402
from gap_study import Honest, arg_of, roles                                            # noqa: E402

ALL_RULES = ("xmult", "xmultm", "planet", "rental", "noskip", "bank")


class Overlay:
    def __init__(self, rules, scratch=None, gain=1.25):
        self.rules = set(rules)
        self.scr, self.gain = scratch, gain
        self.V = None
        self.role = roles()
        self.fired = Counter()

    def main_planet(self, d):
        pl = d["hand_played"]
        h = int(np.argmax(pl)) if max(pl) > 0 else 1
        return HAND_TO_PLANET[h]

    def weak_slot(self, d):
        for k, j in enumerate(d["jokers"]):
            if not j["eternal"] and self.role(j["key"]) in ("mult", "chips"):
                return k
        return -1

    def is_rental(self, it):
        return it.get("kind") == "joker" and it["joker"]["rental"]

    def cost_of(self, a, d):
        """Money an action spends (0 for free ones)."""
        if A_BUY <= a < A_BUY_PACK:
            i = a - A_BUY
            return d["shop"][i]["cost"] if i < len(d["shop"]) else 0
        if A_BUY_PACK <= a < A_VOUCHER:
            i = a - A_BUY_PACK
            return d["shop_packs"][i]["cost"] if i < len(d["shop_packs"]) else 0
        if A_VOUCHER <= a < A_REROLL:
            i = a - A_VOUCHER
            return d["shop_vouchers"][i]["cost"] if i < len(d["shop_vouchers"]) else 0
        if a == A_REROLL:
            return 0 if d["free_rerolls"] > 0 else d["reroll_cost"]
        if a >= A_REPLACE:
            i, j = divmod(a - A_REPLACE, MAX_JOKERS)
            if i < len(d["shop"]) and j < len(d["jokers"]):
                return d["shop"][i]["cost"] - max(1, d["jokers"][j]["cost"] // 2)
        return 0

    def vetoed(self, a, d):
        """The rules that forbid action a here."""
        st, an = d["state"], d["ante"]
        if st == "BLIND_SELECT" and a == A_SKIP and "noskip" in self.rules and d["tags_offered"][d["blind_idx"]] != "investment":
            return "noskip"
        if st == "SHOP":
            item = None
            if A_BUY <= a < A_BUY_PACK and a - A_BUY < len(d["shop"]):
                item = d["shop"][a - A_BUY]
            elif a >= A_REPLACE and (a - A_REPLACE) // MAX_JOKERS < len(d["shop"]):
                item = d["shop"][(a - A_REPLACE) // MAX_JOKERS]
            if item is not None and "rental" in self.rules and an < 5 and self.is_rental(item):
                return "rental"
            if "bank" in self.rules and an >= 3:
                c = self.cost_of(a, d)
                prio = item is not None and ((item["key"].startswith("j_") and self.role(item["key"][2:]) == "xmult")
                                             or item["key"] == "c_" + self.main_planet(d))
                if c > 0 and d["money"] - c < 25 and not prio:
                    return "bank"
        return None

    def strength_gain(self, i, acts):
        """For game i of self.V: the boss share after each action / before (fast_power with common random numbers)."""
        S = self.scr
        k = len(acts) + 1
        S.copy_many(self.V, [i] * k, list(range(k)))
        st = np.full(S.n, -1, dtype=np.int64)
        st[1:k] = acts
        S.step(st)
        S.v.set_curve_crn(4242, False)
        pw = S.v.fast_power(list(range(k)), 16)
        S.v.set_curve_crn(0, False)
        base = max(pw[0][4], 1e-6)
        return [pw[r][4] / base for r in range(1, k)]

    def forced(self, d, mask, i=None):
        """An action the rules require here (or None)."""
        st, an, m = d["state"], d["ante"], d["money"]
        if "xmultm" in self.rules and an >= 2 and i is not None and st in ("SHOP", "PACK"):
            cands = []
            if st == "SHOP":
                for k, it in enumerate(d["shop"]):
                    if it["kind"] != "joker" or self.role(it["key"][2:]) != "xmult":
                        continue
                    if mask[A_BUY + k]:
                        cands.append(A_BUY + k)
                    else:
                        cands += [A_REPLACE + k * MAX_JOKERS + j for j in range(len(d["jokers"])) if mask[A_REPLACE + k * MAX_JOKERS + j]]
            else:
                cands = [A_PICK + k for k, x in enumerate(d["pack_cards"]) if mask[A_PICK + k] and isinstance(x, dict) and self.role(x["key"]) == "xmult"]
            if cands:
                gains = self.strength_gain(int(i), cands)
                b = int(np.argmax(gains))
                if gains[b] >= self.gain:
                    return cands[b], "xmultm:" + ("pick" if st == "PACK" else ("buy" if cands[b] < A_BUY_PACK else "replace"))
        if st == "SHOP":
            if "planet" in self.rules:
                mp = self.main_planet(d)
                for i, c in enumerate(d["consumables"]):
                    if c[0] == mp and mask[A_USE_C + i]:
                        return A_USE_C + i, "planet:use"
            if "xmult" in self.rules and an >= 2:
                for i, it in enumerate(d["shop"]):
                    if it["kind"] != "joker" or self.role(it["key"][2:]) != "xmult":
                        continue
                    if "rental" in self.rules and an < 5 and self.is_rental(it):
                        continue
                    if mask[A_BUY + i]:
                        return A_BUY + i, "xmult:buy"
                    j = self.weak_slot(d)
                    if j >= 0 and mask[A_REPLACE + i * MAX_JOKERS + j]:
                        return A_REPLACE + i * MAX_JOKERS + j, "xmult:replace"
            if "planet" in self.rules:
                mp = "c_" + self.main_planet(d)
                for i, it in enumerate(d["shop"]):
                    if it["key"] == mp and mask[A_BUY + i]:
                        return A_BUY + i, "planet:buy"
        elif st == "PACK":
            for i, x in enumerate(d["pack_cards"]):
                if not mask[A_PICK + i]:
                    continue
                if "xmult" in self.rules and isinstance(x, dict) and self.role(x["key"]) == "xmult":
                    return A_PICK + i, "xmult:pick"
                if "planet" in self.rules and isinstance(x, str) and x == "c_" + self.main_planet(d):
                    return A_PICK + i, "planet:pick"
        return None

    def choose(self, d, mask, logits, i=None):
        f = self.forced(d, mask, i)
        if f is not None:
            self.fired[f[1]] += 1
            return f[0]
        order = np.argsort(-np.where(mask, logits, -1e30))
        first = int(order[0])
        for a in order:
            a = int(a)
            if not mask[a]:
                break
            why = self.vetoed(a, d)
            if why is None:
                if a != first:
                    self.fired["veto:" + (self.vetoed(first, d) or "?")] += 1
                return a
        return first


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v51.pt")
    p.add_argument("--config", default="checkpoints/ppo_gold_v51.json")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed0", type=int, default=9_500_000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--rules", default=",".join(ALL_RULES))
    p.add_argument("--mode", choices=["honest", "foresight"], default="honest")
    p.add_argument("--honest", type=int, default=5)
    p.add_argument("--gain", type=float, default=1.25, help="xmultm: the strength ratio a multiplier buy must reach")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    rules = [] if a.rules == "none" else [r for r in a.rules.split(",") if r]
    assert all(r in ALL_RULES for r in rules), rules
    args = json.load(open(a.config))["args"]
    dev = a.device
    net = load_model(a.ckpt, dev).eval()
    n = a.games
    V = CppVecEnv(n, "RED", a.stake, seed0=a.seed0, threads=a.threads)
    if getattr(net, "marg", 0):
        V.set_marginal(int(net.marg))
    V.v.set_solver_tarots(True, arg_of(args, "--solver-targets", 3))
    V.v.set_solver_order(True)
    V.v.set_solver_speed(True, False, 0.0)
    V.v.set_solver_tt(True)
    honest = None
    if a.mode == "foresight":
        V.set_autoplay(True, arg_of(args, "--autoplay-nodes", 400), True)
    else:
        honest = Honest(V, a.honest, 400, 4096, "RED", a.stake, a.threads)
    V.v.restart()
    scr = None
    if "xmultm" in rules:
        scr = CppVecEnv(48, "RED", a.stake, seed0=5, threads=a.threads)
        if getattr(net, "marg", 0):
            scr.set_marginal(int(net.marg))
        scr.v.restart()
    ov = Overlay(rules, scr, a.gain)
    ov.V = V
    fin = np.zeros(n, bool)
    bl, won, ante = np.zeros(n), np.zeros(n, bool), np.zeros(n, int)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(20000):
            obs = V.obs
            g = np.asarray(obs["g"])
            mask = np.asarray(obs["mask"])
            inround = (g[:, 5] > 0.5) & ~fin
            outside = np.flatnonzero(~fin & ~inround)
            act = np.full(n, -1, dtype=np.int64)
            if len(outside):
                o = {k: torch.from_numpy(np.asarray(obs[k])[outside]).to(dev) for k in OBS_KEYS if k in obs}
                lg, _ = net(o)
                lg = lg.float().cpu().numpy()
                for r, i in enumerate(outside):
                    act[i] = ov.choose(V.v.dump(int(i)), mask[i], lg[r], int(i)) if rules else int(np.argmax(np.where(mask[i], lg[r], -1e30)))
            ids = np.flatnonzero(inround)
            if len(ids):
                act[ids] = honest.moves(V, ids, mask, np.asarray(obs["af"]))
            _, done, info = V.step(act)
            f = done & ~fin
            bl[f], won[f], ante[f] = info["blinds"][f], info["won"][f], info["ante"][f]
            fin |= done
            if fin.all():
                break
    furthest = np.where(won, 9, ante)
    print(json.dumps({"rules": rules or "none", "mode": a.mode, "games": n, "blinds": round(float(bl.mean()), 3), "wins": int(won.sum()),
                      "reach_ante": {k: round(float((furthest >= k).mean()), 3) for k in range(3, 9)}, "fired": dict(ov.fired),
                      "s": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
