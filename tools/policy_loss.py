"""Loss analysis of a PPO policy (greedy, every round by the exact solver as in training): the build at each game's last decision outside a round (jokers by
role, hand levels, money, the coming boss, the build's power against the coming blind and this ante's boss, projected to the ante-8 boss) and what it did over
the run; each loss gets cause tags, the summary counts them. Normal Gold games (no head-start joker, the usual shop).

    python tools/policy_loss.py --ckpt checkpoints/ppo_gold_v50.pt --config checkpoints/ppo_gold_v50.json --games 100 --out traces/v50_loss.txt
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.az import guide as G                               # noqa: E402
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS              # noqa: E402
from balatro_rl.env import BOSS_KEYS                               # noqa: E402
from balatro_rl.heur import describe                               # noqa: E402
from balatro_rl.model import load_model                            # noqa: E402
from balatro_rl.runeval import Cols                                # noqa: E402
from balatro_rl.sim.jokers import JOKERS                           # noqa: E402

K = list(JOKERS)
HANDS = ["high card", "pair", "two pair", "three", "straight", "flush", "full house", "four", "straight flush", "five", "flush house", "flush five"]
ROLE = {"xmult": G.XMULT, "mult": G.MULT, "chips": G.CHIPS, "scaler": G.SCALERS, "econ": G.ECON}
BL = ["small", "big", "boss"]


def roles_of(k):
    return [r for r, s in ROLE.items() if k in s]


def arg_of(args, k, d):
    return type(d)(args[args.index(k) + 1]) if k in args else d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v50.pt")
    p.add_argument("--config", default="checkpoints/ppo_gold_v50.json")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed0", type=int, default=9_000_000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="traces/policy_loss.txt")
    p.add_argument("--no-overrule", default="", help="--lookahead: action kinds the search may not switch to (skip,sell)")
    p.add_argument("--no-pot", action="store_true", help="--lookahead: leave the shaping potentials out of the move scores (the old, biased search)")
    p.add_argument("--lookahead", type=int, default=0, help="decisions outside rounds by the one-step lookahead (eval_lookahead.py) over this many policy moves (0: greedy)")
    a = p.parse_args()
    args = json.load(open(a.config))["args"]
    torch.set_num_threads(a.threads)
    net = load_model(a.ckpt, "cpu").eval()
    n = a.games
    kw = {"ante_weight": arg_of(args, "--ante-weight", 1.0), "win_bonus": arg_of(args, "--win-bonus", 10.0), "win_ante": 8, "joker_pool": None, "shape_chips": 0.0,
          "shape_phi": 0.0, "reward_config": None, "shape_gamma": arg_of(args, "--gamma", 0.995)}

    def make(m, seed):
        e = CppVecEnv(m, "RED", "GOLD", seed0=seed, threads=a.threads, reward_kw=kw, verdant_bonus=arg_of(args, "--verdant-bonus", 0.0))
        if arg_of(args, "--weight-power", 1.0) != 1.0:
            e.set_weight_shape(arg_of(args, "--ante-weight", 1.0), arg_of(args, "--weight-power", 1.0))
        if arg_of(args, "--weight-geom", 0.0) > 0:
            e.set_weight_geom(arg_of(args, "--weight-geom", 0.0))
        if getattr(net, "marg", 0):
            e.set_marginal(int(net.marg))
        e.set_autoplay(True, arg_of(args, "--autoplay-nodes", 400), True)
        e.v.restart()
        return e
    if "--solver-tarots" in args:
        CppVecEnv(1, "RED", "GOLD").v.set_solver_tarots(True, 3)        # (engine globals)
    v = make(n, a.seed0)
    if "--solver-smart" in args:
        v.v.set_solver_order(True)
    if "--solver-lite" in args:
        v.v.set_solver_speed(True, False, 0.0)
    if "--solver-tt" in args:
        v.v.set_solver_tt(True)
    la = None
    if a.lookahead:
        from balatro_rl.eval_lookahead import Lookahead
        from balatro_rl.shaping_pot import ShapingPot
        pot = ShapingPot("cpu", build_roles=arg_of(args, "--build-roles", "xmult,mult,chips"), build_second=arg_of(args, "--build-second", 0.0),
                         bank=arg_of(args, "--bank-bonus", 0.0), build=arg_of(args, "--build-bonus", 0.0),
                         rental=arg_of(args, "--rental-penalty", 0.0), debt=arg_of(args, "--debt-penalty", 0.0))
        la = Lookahead(lambda m: make(m, a.seed0 + 77), net, "cpu", k=a.lookahead, slots=256, pot=None if a.no_pot else pot,
                       no_overrule=tuple(x for x in a.no_overrule.split(",") if x))
    last, acts, buys = {}, defaultdict(Counter), defaultdict(Counter)
    fin = np.zeros(n, bool)
    bl, won = np.zeros(n), np.zeros(n, bool)
    for _ in range(6000):
        o = {k: np.asarray(v.obs[k]) for k in OBS_KEYS}
        with torch.no_grad():
            lg, _ = net({k: torch.from_numpy(x) for k, x in o.items()})
        act = la.act(v, lg, fin) if la is not None else np.where(fin, -1, lg.argmax(1).numpy())
        st = np.argmax(o["g"][:, Cols.STATE:Cols.STATE + 4], 1)
        idx = [int(i) for i in np.flatnonzero(~fin & ((st == 0) | (st == 2)))]       # (blind selects and shops: projected_power crashes in a pack)
        if idx:
            fp = v.fast_power(idx, 4)
            pp = v.v.projected_power(idx, 2)
            for k, (i, inf) in enumerate(zip(idx, v.v.cell_info(idx))):
                nj = inf[7]
                jok = [K[inf[8 + 3 * j]] for j in range(nj)]
                rentals = sum(1 for j in range(nj) if inf[8 + 3 * j + 2] & 2)
                eternals = sum(1 for j in range(nj) if inf[8 + 3 * j + 2] & 1)
                lv = list(inf[8 + 3 * nj: 8 + 3 * nj + 12])
                bv = o["g"][i, 25:25 + len(BOSS_KEYS)]
                boss = BOSS_KEYS[int(np.argmax(bv))] if bv.max() > 0.5 else "?"
                last[i] = dict(rentals=rentals, eternals=eternals, ante=inf[0], blind=inf[1], money=inf[3], slots=inf[5], jokers=jok, levels=lv, boss=boss, power=float(fp[k, 0]),
                               bpower=float(fp[k, 4]) if fp.shape[1] > 4 else float("nan"), proj=float(pp[k]))
                d = describe(int(act[i]), o["aid"][i])
                kind = d.split(" ")[0]
                acts[i][kind] += 1
                if kind in ("buy", "replace", "pick") and "j_" in d:
                    for r in roles_of(d.split("j_", 1)[1]) or ["other"]:
                        buys[i][r] += 1
        _, dn, info = v.step(act)
        f = dn & ~fin
        bl[f], won[f] = info["blinds"][f], info["won"][f]
        fin |= dn
        if fin.all():
            break
    tags_all, death_ante = Counter(), Counter()
    mode = f"one-step lookahead over {a.lookahead} moves (overruled the policy on {la.n_changed}/{la.n_decisions} decisions)" if la is not None else "greedy"
    lines = [f"{n} Gold games ({a.ckpt}, {mode}, solver rounds): {bl.mean():.2f} blinds, {int(won.sum())} wins"]
    for i in range(n):
        s = last.get(i)
        if s is None:
            continue
        tags = []
        if not won[i]:
            death_ante[s["ante"]] += 1
            js = s["jokers"]
            rl = Counter(r for k in js for r in roles_of(k))
            tags.append(f"died to the boss ({s['boss']})" if s["blind"] == 2 else f"died to the {BL[s['blind']]} blind")
            if s["power"] < 1.0:
                tags.append("underpowered: under 1x the blind (optimistic measure)")
            elif s["power"] < 2.0:
                tags.append("thin: 1-2x the blind (optimistic measure)")
            else:
                tags.append("had 2x+ the blind on paper but lost the round")
            if s["blind"] < 2 and s["bpower"] < 1.0:
                tags.append("this ante's boss out of reach (under 1x)")
            if rl["xmult"] == 0 and s["ante"] >= 3:
                tags.append("no xMult joker (ante 3+)")
            if len(js) < s["slots"] and s["ante"] >= 3:
                tags.append("empty joker slots (ante 3+)")
            if max(s["levels"]) <= 1 and s["ante"] >= 3:
                tags.append("no levelled hand (ante 3+)")
            if s["money"] >= 20:
                tags.append("died holding $20+")
            if rl["mult"] + rl["chips"] >= 3 and rl["xmult"] == 0:
                tags.append("flat build (3+ flat jokers, no xMult)")
            if s["rentals"] >= 1:
                tags.append(f"holding {s['rentals']} rental joker(s) ($3 a round each)")
            if s["money"] < 0:
                tags.append("in debt at the last shop")
            if rl["econ"] >= 2:
                tags.append("2+ economy jokers held")
            if acts[i]["skip"] >= 2 and s["ante"] <= 3:
                tags.append("skipped 2+ blinds and died by ante 3")
            tags_all.update(tags)
        lv = s["levels"]
        lines.append(f"\ngame {i} (seed {a.seed0 + i}): {int(bl[i])} blinds{' WON' if won[i] else ''}; last state ante {s['ante']} {BL[s['blind']]}, boss {s['boss']}, "
                     f"${s['money']}, power {s['power']:.2f} of the blind / {s['bpower']:.2f} of the boss, projected ante-8 {s['proj']:.3f}")
        lines.append(f"   rentals {s['rentals']}, eternals {s['eternals']}; jokers ({len(s['jokers'])}/{s['slots']}): " + ", ".join(k + "[" + ("/".join(roles_of(k)) or "-") + "]" for k in s["jokers"]))
        lines.append("   levels: " + (", ".join(f"{HANDS[h]} {lv[h]}" for h in range(12) if lv[h] > 1) or "none")
                     + f"; actions {dict(acts[i])}; joker buys by role {dict(buys[i])}")
        if tags:
            lines.append("   causes: " + "; ".join(tags))
    nl = n - int(won.sum())
    lines.append(f"\n=== summary over {nl} losses ===\nfurthest ante: " + ", ".join(f"ante {k}: {c}" for k, c in sorted(death_ante.items())))
    for t, c in tags_all.most_common():
        lines.append(f"  {c:3d}  {t}")
    tot = Counter()
    for i in range(n):
        tot.update(buys[i])
    lines.append("joker buys by role over all games: " + str(dict(tot)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    text = "\n".join(lines)
    open(a.out, "w", encoding="utf-8").write(text)
    print(lines[0])
    print(text[text.index("=== summary"):])


if __name__ == "__main__":
    main()
