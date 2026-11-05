"""Loss analysis of the heuristic shop bot (balatro_rl/heur.py): plays N Gold games (the exact solver plays the rounds) and, for each game, the build at its
last shop decision (jokers and their roles, hand levels, money, the coming boss, the build's power against the coming blind and projected to the ante-8 boss)
and what it did over the run (buys by role, rerolls, packs, sales, skips). Each loss gets cause tags; the summary counts them.

    python tools/heur_loss.py --ckpt checkpoints/heur4_best.pt --games 50 --out traces/heur_loss.txt
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import heur as H                                   # noqa: E402
from balatro_rl.az import guide as G                               # noqa: E402
from balatro_rl.env import A_SELECT, BOSS_KEYS                     # noqa: E402
from balatro_rl.sim.items import VOCAB                             # noqa: E402
from balatro_rl.sim.jokers import JOKERS                           # noqa: E402

K = list(JOKERS)
HANDS = ["high card", "pair", "two pair", "three", "straight", "flush", "full house", "four", "straight flush", "five", "flush house", "flush five"]
ROLE = {"xmult": G.XMULT, "mult": G.MULT, "chips": G.CHIPS, "scaler": G.SCALERS, "econ": G.ECON}
BL = ["small", "big", "boss"]


def roles_of(k):
    return [r for r, s in ROLE.items() if k in s]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/heur4_best.pt")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--seed0", type=int, default=9_000_000)
    p.add_argument("--node-cap", type=int, default=150)
    p.add_argument("--out", default="traces/heur_loss.txt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    net = H.HeurNet()
    w = net.init() if a.ckpt == "start" else net.from_checkpoint(a.ckpt)
    n = a.games
    ev = H.SolverEvaluator(net, 1, n, "RED", "GOLD", 0, a.device, 10.0, a.node_cap)
    last = {}
    acts = defaultdict(Counter)
    buys = defaultdict(Counter)

    def rec(v, act, scores):
        o = v.obs
        mask = np.asarray(o["mask"])
        aid = np.asarray(o["aid"])
        g = np.asarray(o["g"])
        idx = [i for i in range(n) if act[i] >= A_SELECT and not mask[i, :A_SELECT].any()]
        if not idx:
            return
        pw = v.fast_power(idx, 4)
        pp = v.projected_power(idx, 2)
        for k, (i, inf) in enumerate(zip(idx, v.cell_info(idx))):
            nj = inf[7]
            jok = [K[inf[8 + 3 * j]] for j in range(nj)]
            lv = inf[8 + 3 * nj: 8 + 3 * nj + 12]
            boss = BOSS_KEYS[int(np.argmax(g[i, 25:25 + len(BOSS_KEYS)]))] if g[i, 25:25 + len(BOSS_KEYS)].max() > 0.5 else "?"
            last[i] = dict(ante=inf[0], blind=inf[1], money=inf[3], slots=inf[5], jokers=jok, levels=lv, boss=boss, power=float(pw[k, 0]), proj=float(pp[k]))
            d = H.describe(int(act[i]), aid[i])
            kind = d.split(" ")[0]
            acts[i][kind] += 1
            if kind in ("buy", "replace", "pick") and "j_" in d:
                key = d.split("j_", 1)[1]
                for r in roles_of(key) or ["other"]:
                    buys[i][r] += 1

    _, b, wn = ev.run(w[None].float().to(a.device), seed0=a.seed0, recorder=rec)
    b, wn = b[0], wn[0]
    tags_all = Counter()
    lines = [f"{n} Gold games ({a.ckpt}): {b.mean():.2f} blinds, {int(wn.sum())} wins"]
    death_ante = Counter()
    for i in range(n):
        s = last.get(i)
        if s is None:
            continue
        nb = int(b[i])
        tags = []
        if not wn[i]:
            death_ante[s["ante"]] += 1
            js = s["jokers"]
            rl = Counter(r for k in js for r in roles_of(k))
            main = int(np.argmax(s["levels"]))
            if s["blind"] == 2:
                tags.append(f"died to the boss ({s['boss']})" if s["boss"] in G.KILLERS else "died to the boss")
            else:
                tags.append(f"died to the {BL[s['blind']]} blind")
            if s["power"] < 0.6:
                tags.append("underpowered (<60% of the blind with every hand)")
            elif s["power"] >= 1.0:
                tags.append("had the power on paper (>=100%) but lost the round")
            if rl["xmult"] == 0 and s["ante"] >= 3:
                tags.append("no xMult joker (ante 3+)")
            if len(js) < s["slots"] and s["ante"] >= 3:
                tags.append("empty joker slots")
            if max(s["levels"]) <= 1 and s["ante"] >= 3:
                tags.append("no levelled hand (ante 3+)")
            if s["money"] >= 20:
                tags.append("died holding $20+")
            if rl["mult"] + rl["chips"] >= 3 and rl["xmult"] == 0:
                tags.append("flat build (3+ flat jokers, no xMult)")
            if acts[i]["reroll"] >= 8:
                tags.append("many rerolls (8+)")
            if acts[i]["sell"] + acts[i]["sell_joker"] >= 4:
                tags.append("many sales (4+)")
            if s["proj"] < 0.01 and s["ante"] >= 3:
                tags.append("projected ante-8 power < 1% of the target")
            tags_all.update(tags)
        lines.append(f"\ngame {i} (seed {a.seed0 + i}): {nb} blinds{' WON' if wn[i] else ''}; last shop ante {s['ante']} {BL[s['blind']]} next, boss {s['boss']}, ${s['money']}, "
                     f"power {s['power']:.2f} of the blind, projected ante-8 {s['proj']:.3f}")
        lines.append(f"   jokers ({len(s['jokers'])}/{s['slots']}): " + ", ".join(f"{k}[{'/'.join(roles_of(k)) or '-'}]" for k in s["jokers"]))
        lv = s["levels"]
        lines.append(f"   levels: " + (", ".join(f"{HANDS[h]} {lv[h]}" for h in range(12) if lv[h] > 1) or "none") + f"; actions {dict(acts[i])}; joker buys by role {dict(buys[i])}")
        if tags:
            lines.append("   causes: " + "; ".join(tags))
    lines.append(f"\n=== summary over {n - int(wn.sum())} losses ===\nfurthest ante: " + ", ".join(f"ante {k}: {v}" for k, v in sorted(death_ante.items())))
    for t, c in tags_all.most_common():
        lines.append(f"  {c:3d}  {t}")
    tot = Counter()
    for i in range(n):
        tot.update(buys[i])
    lines.append("joker buys by role over all games: " + str(dict(tot)))
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(text)
    print(text[text.index("=== summary"):])


if __name__ == "__main__":
    main()
