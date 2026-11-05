"""Where do Gold runs fail, and what does money buy? (python tools/failure_table.py --ckpt X --games 300 --money 0,250,500,1000 --out traces/ft_X.json)

For each starting-money level the same seeds are played greedily (so the levels are paired). Per game it records the per-ante state at the first blind select of each ante
(money, jokers and their kinds, hand level-ups, deck size) and, at the end, how the game ended: the blind it was lost at, how much of the target the lost blind reached, the boss,
the build, the money held, hand level-ups, planets used. The summary has the win rate and blinds by money, and for each level a failure table by ante.
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppEnv, OBS_KEYS      # noqa: E402
from balatro_rl.joker_desc import FEATURES, item_descriptors    # noqa: E402
from balatro_rl.model import load_model                 # noqa: E402
from balatro_rl.sim.items import VOCAB                  # noqa: E402
from balatro_rl.sim.jokers import JOKERS                # noqa: E402

KINDS = ("mult", "xmult", "chips", "grows", "economy")


def joker_kinds():
    d = item_descriptors()
    col = {n: FEATURES.index(n) for n in KINDS}
    return {k[2:]: {n: bool(d[i, col[n]]) for n in KINDS} for i, k in enumerate(VOCAB) if k.startswith("j_") and k[2:] in JOKERS}


def play(model, deck, stake, money, seeds, kinds, autoplay=False):
    envs = []
    for _ in seeds:
        e = CppEnv(deck, stake)
        e.e.autoplay = bool(autoplay)
        if money:
            e.e.start_money_p, e.e.start_money = 1.0, money
        if getattr(model, "marg", 0):
            e.e.marginal = int(model.marg)
        envs.append(e)
    obs = [e.reset(s) for e, s in zip(envs, seeds)]
    n = len(envs)
    per_ante = [dict() for _ in range(n)]
    out = [None] * n
    live = list(range(n))
    while live:
        batch = {k: torch.from_numpy(np.stack([obs[i][k] for i in live])) for k in OBS_KEYS}
        with torch.no_grad():
            logits, _ = model(batch)
        act = logits.argmax(1).numpy()
        nxt = []
        for i, a in zip(live, act):
            e = envs[i]
            st = e.game.state
            if st == "BLIND_SELECT" and e.game.furthest_blind == 3 * (e.game.ante - 1) and e.game.ante not in per_ante[i]:
                d = e.game.dump()
                js = [j["key"] for j in d["jokers"]]
                per_ante[i][d["ante"]] = {"money": d["money"], "jokers": len(js), "levels": int(sum(l - 1 for l in d["hand_levels"])), "deck": len(d["full_deck"]),
                                          **{k: sum(kinds[j][k] for j in js) for k in KINDS}}
            o, r, done, info = e.step(int(a))
            if done:
                d = e.game.dump()
                js = [j["key"] for j in d["jokers"]]
                out[i] = {"won": d["state"] == "WON", "blinds": d["furthest_blind"], "ante": d["ante"], "blind_idx": d["blind_idx"], "boss": d["boss"],
                          "reach": min(1.0, d["chips"] / max(1.0, d["target"])), "money": d["money"], "jokers": js, "levels": int(sum(l - 1 for l in d["hand_levels"])),
                          "planets": d.get("planet_uses", 0), "deck": len(d["full_deck"]), "vouchers": d["vouchers"] if isinstance(d["vouchers"], list) else [],
                          "per_ante": per_ante[i], **{k: sum(kinds[j][k] for j in js) for k in KINDS}}
            else:
                obs[i] = o
                nxt.append(i)
        live = nxt
    return out


def summarise(games):
    n = len(games)
    wins = sum(g["won"] for g in games)
    lost = [g for g in games if not g["won"]]
    s = {"games": n, "wins": wins, "win_rate": round(wins / n, 3), "blinds_mean": round(float(np.mean([g["blinds"] for g in games])), 2),
         "money_left_mean": round(float(np.mean([g["money"] for g in games])), 1), "levels_mean": round(float(np.mean([g["levels"] for g in games])), 1),
         "planets_mean": round(float(np.mean([g["planets"] for g in games])), 1), "deck_mean": round(float(np.mean([g["deck"] for g in games])), 1)}
    fail = {}
    for ante in range(1, 9):
        gs = [g for g in lost if g["ante"] == ante]
        if not gs:
            continue
        by_blind = Counter(("small", "big", "boss")[g["blind_idx"]] for g in gs)
        fail[ante] = {"lost": len(gs), "at": dict(by_blind), "reach_median": round(float(np.median([g["reach"] for g in gs])), 2),
                      "reach_under_half": round(float(np.mean([g["reach"] < 0.5 for g in gs])), 2), "money_at_death": round(float(np.mean([g["money"] for g in gs])), 1),
                      "jokers": round(float(np.mean([len(g["jokers"]) for g in gs])), 2), "levels": round(float(np.mean([g["levels"] for g in gs])), 1),
                      "has_xmult": round(float(np.mean([g["xmult"] > 0 for g in gs])), 2), "mult": round(float(np.mean([g["mult"] for g in gs])), 2),
                      "xmult": round(float(np.mean([g["xmult"] for g in gs])), 2), "scaling": round(float(np.mean([g["grows"] for g in gs])), 2),
                      "bosses": dict(Counter(g["boss"] for g in gs if g["blind_idx"] == 2).most_common(4))}
    s["failure_by_ante"] = fail
    traj = defaultdict(lambda: defaultdict(list))
    for g in games:
        for ante, v in g["per_ante"].items():
            for k, x in v.items():
                traj[ante][k].append(x)
    s["per_ante_start"] = {a: {"games": len(v["money"]), **{k: round(float(np.mean(x)), 2) for k, x in v.items()}} for a, v in sorted(traj.items())}
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--money", default="0,250,500,1000")
    p.add_argument("--seed0", type=int, default=9_500_000)
    p.add_argument("--autoplay", action="store_true", help="the round solver is available (the network may hand a round to it); default: it plays every hand itself")
    p.add_argument("--out", default="")
    a = p.parse_args()
    tmp = a.ckpt.replace(".pt", "_ft_snap.pt")
    shutil.copy2(a.ckpt, tmp)
    model = load_model(tmp, "cpu")
    kinds = joker_kinds()
    seeds = [a.seed0 + i for i in range(a.games)]
    res = {"ckpt": a.ckpt, "levels": {}}
    for m in [int(x) for x in a.money.split(",")]:
        games = play(model, a.deck, a.stake, m, seeds, kinds, a.autoplay)
        res["levels"][m] = summarise(games)
        print(m, res["levels"][m]["win_rate"], res["levels"][m]["blinds_mean"], flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"))
    os.remove(tmp)
    print(json.dumps({m: {k: v for k, v in s.items() if k not in ("failure_by_ante", "per_ante_start")} for m, s in res["levels"].items()}, indent=1))


if __name__ == "__main__":
    main()
