"""Gap study: what our agent's Gold builds look like ante by ante, against what winning human builds need (GOLD_STAKE_RESEARCH.md).

    collect   play --games Gold games with a PPO checkpoint deciding outside rounds; rounds played honestly (--mode honest: --honest copies with the
              future hidden are solved and vote on every move, Env::hide_future) or with foresight (--mode foresight: the engine's autoplay solver,
              which sees the draws, as in training). Writes one JSON line per game: the result, a build snapshot at the first blind select of every
              ante and at the end (jokers, hand levels and plays, deck suits / ranks / enhancements / seals, money, vouchers, consumables, measured
              power), and every decision outside rounds with what was on offer.

    python tools/gap_study.py collect --ckpt checkpoints/ppo_gold_v51.pt --config checkpoints/ppo_gold_v51.json --games 1000 --mode honest \
        --out traces/gap/v51_honest.jsonl
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS                                  # noqa: E402
from balatro_rl.env import (A_BUY, A_BUY_PACK, A_LEAVE, A_PICK, A_PSKIP, A_REPLACE, A_REROLL, A_REROLL_BOSS, A_SELECT,  # noqa: E402
                            A_SELL_C, A_SELL_J, A_SKIP, A_USE_C, A_VOUCHER, MAX_JOKERS, N_SUB, RATIO_COL)
from balatro_rl.model import load_model                                                # noqa: E402

GOLD_BASE = [300, 1000, 3200, 9000, 25000, 60000, 110000, 200000]        # Purple scaling (Gold): small blind base per ante; big x1.5, boss x2


def arg_of(args, k, d):
    return type(d)(args[args.index(k) + 1]) if k in args else d


def snapshot(d, power=None):
    """The build in a game dump: what the gap study compares with the human targets."""
    deck = d["full_deck"]
    suits = Counter(c[1] for c in deck if c[2] != "STONE")
    ranks = Counter(c[0] for c in deck if c[2] != "STONE")
    out = {
        "ante": d["ante"], "blind": d["blind_idx"], "money": d["money"], "state": d["state"],
        "jokers": [[j["key"], j["edition"], int(j["eternal"]), int(j["rental"]), j["perishable"], {k: round(float(v), 3) for k, v in j["state"].items()}]
                   for j in d["jokers"]],
        "joker_slots": d["joker_slots"],
        "levels": list(d["hand_levels"]), "played": list(d["hand_played"]),
        "deck_size": len(deck), "suits": [suits.get(s, 0) for s in range(4)], "ranks": [ranks.get(r, 0) for r in range(2, 15)],
        "enh": dict(Counter(c[2] for c in deck if c[2])), "eds": dict(Counter(c[3] for c in deck if c[3])), "seals": dict(Counter(c[4] for c in deck if c[4])),
        "consumables": [c[0] for c in d["consumables"]], "vouchers": list(d["vouchers"]) if not isinstance(d["vouchers"], int) else d["vouchers"],
        "hand_size": d["hand_size"], "planets_used": d["planets_used"], "tarots_used": d["tarots_used"],
    }
    if power is not None:
        out["power"] = {"share_next": round(power[0], 4), "mean_best": round(power[2], 1), "max_best": round(power[3], 1), "share_boss": round(power[4], 4)}
    return out


def describe(a, d):
    """A decision outside a round, in words: (kind, item, extra)."""
    shop, jok, cons, pack = d["shop"], d["jokers"], d["consumables"], d["pack_cards"]
    name = lambda x: x["key"] if isinstance(x, dict) else (x if isinstance(x, str) else "card")
    if a == A_SELECT:
        return ["select"]
    if a == A_SKIP:
        return ["skip_blind", d["tags_offered"][0]]
    if a == A_REROLL_BOSS:
        return ["reroll_boss"]
    if a == A_LEAVE:
        return ["leave"]
    if a == A_REROLL:
        return ["reroll"]
    if a == A_PSKIP:
        return ["skip_pack"]
    if A_BUY <= a < A_BUY_PACK:
        i = a - A_BUY
        return ["buy", name(shop[i]) if i < len(shop) else "?"]
    if A_BUY_PACK <= a < A_VOUCHER:
        i = a - A_BUY_PACK
        return ["pack", d["shop_packs"][i]["key"] if i < len(d["shop_packs"]) else "?"]
    if A_VOUCHER <= a < A_REROLL:
        i = a - A_VOUCHER
        return ["voucher", d["shop_vouchers"][i]["key"] if i < len(d["shop_vouchers"]) else "?"]
    if A_SELL_J <= a < A_SELL_J + MAX_JOKERS:
        i = a - A_SELL_J
        return ["sell", jok[i]["key"] if i < len(jok) else "?"]
    if A_USE_C <= a < A_USE_C + 3:
        i = a - A_USE_C
        return ["use", cons[i][0] if i < len(cons) else "?"]
    if A_SELL_C <= a < A_SELL_C + 3:
        i = a - A_SELL_C
        return ["sell_cons", cons[i][0] if i < len(cons) else "?"]
    if A_PICK <= a < A_PSKIP:
        i = a - A_PICK
        return ["pick", name(pack[i]) if i < len(pack) else "?"]
    if a >= A_REPLACE:
        i, j = divmod(a - A_REPLACE, MAX_JOKERS)
        return ["replace", name(shop[i]) if i < len(shop) else "?", jok[j]["key"] if j < len(jok) else "?"]
    return ["other", int(a)]


def offers(d):
    if d["state"] == "PACK":
        return {"pack": d["pack_kind"], "cards": [x["key"] if isinstance(x, dict) else (x if isinstance(x, str) else "card") for x in d["pack_cards"]]}
    if d["state"] == "SHOP":
        sh = []
        for x in d["shop"]:
            e = [x["key"], x["cost"]]
            if x["kind"] == "joker":
                j = x["joker"]
                e.append("".join(t for t, on in (("E", j["eternal"]), ("R", j["rental"]), ("P", j["perishable"] is not None)) if on) + (("/" + j["edition"]) if j["edition"] else ""))
            sh.append(e)
        return {"shop": sh, "packs": [p["key"] for p in d["shop_packs"]], "vouchers": [p["key"] for p in d["shop_vouchers"]]}
    if d["state"] == "BLIND_SELECT":
        return {"boss": d["boss"], "tags": d["tags_offered"]}
    return {}


class Honest:
    """The honest round player (as heur.SolverEvaluator._honest_moves): K copies with the future hidden are solved; their first moves vote."""

    def __init__(self, v, K, node_cap, scratch, deck, stake, threads):
        self.K, self.node_cap = K, node_cap
        self.H = CppVecEnv(scratch, deck, stake, seed0=2, threads=threads)
        self.H.v.restart()
        self.salt = 0
        self.pending = {}                     # game -> the rest of the voted line after a consumable use (its target cards)

    def moves(self, v, ids, mask, af):
        K, H = self.K, self.H
        per = max(1, H.n // K)
        out = np.zeros(len(ids), dtype=np.int64)
        todo = []
        for r, i in enumerate(ids):
            q = self.pending.pop(int(i), None)
            if q and v.v.dump(int(i))["targeting"]:
                out[r] = q[0]
                if len(q) > 1:
                    self.pending[int(i)] = q[1:]
            else:
                todo.append(r)
        sub = [ids[r] for r in todo]
        res = self._solve(v, sub, mask, af)
        for r, a in zip(todo, res):
            out[r] = a
        return out

    def _solve(self, v, ids, mask, af):
        K, H = self.K, self.H
        per = max(1, H.n // K)
        out = np.zeros(len(ids), dtype=np.int64)
        for c0 in range(0, len(ids), per):
            ch = ids[c0:c0 + per]
            src = [int(i) for i in ch for _ in range(K)]
            dst = list(range(len(src)))
            H.copy_many(v, src, dst)
            self.salt += 1
            H.v.hide_future(dst, [self.salt * 1_000_003 + 7919 * k + 11 for k in dst])
            sols = H.solve_rounds(dst, 1, self.node_cap)
            for r, i in enumerate(ch):
                lines = [p[0] for p in sols[r * K:(r + 1) * K] if p and p[0]]
                votes = Counter(int(ln[0]) for ln in lines)
                a = votes.most_common(1)[0][0] if votes else -1
                if A_USE_C <= a < A_USE_C + 3:                               # a consumable use: its target cards follow in the line
                    ln = next(ln for ln in lines if int(ln[0]) == a)
                    if len(ln) > 1:
                        self.pending[int(i)] = [int(x) for x in ln[1:]]
                if a < 0 or not mask[i, a]:
                    sc = np.where(mask[i, :N_SUB], af[i, :N_SUB, RATIO_COL + 1], -1e9)
                    a = int(np.argmax(sc)) if sc.max() > -1e8 else int(np.argmax(mask[i]))
                out[c0 + r] = a
        return out


def collect(a):
    args = json.load(open(a.config))["args"]
    dev = a.device
    net = load_model(a.ckpt, dev).eval()
    n = a.games
    v = CppVecEnv(n, "RED", a.stake, seed0=a.seed0, threads=a.threads)
    if getattr(net, "marg", 0):
        v.set_marginal(int(net.marg))
    v.v.set_solver_tarots(True, arg_of(args, "--solver-targets", 3))
    v.v.set_solver_order(True)
    v.v.set_solver_speed(True, False, 0.0)
    v.v.set_solver_tt(True)
    honest = None
    if a.mode == "foresight":
        v.set_autoplay(True, arg_of(args, "--autoplay-nodes", 400), True)
    else:
        honest = Honest(v, a.honest, a.node_cap, a.scratch, "RED", a.stake, a.threads)
    v.v.restart()
    fin = np.zeros(n, bool)
    res = [None] * n
    snaps = [[] for _ in range(n)]
    decisions = [[] for _ in range(n)]
    last_ante = np.zeros(n, int)
    last_dump = [None] * n
    t0 = time.time()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with torch.no_grad():
        for step in range(a.max_steps):
            obs = v.obs
            g = np.asarray(obs["g"])
            mask = np.asarray(obs["mask"])
            live = ~fin
            inround = (g[:, 5] > 0.5) & live
            outside = [int(i) for i in np.flatnonzero(live & ~inround)]
            act = np.full(n, -1, dtype=np.int64)
            dumps = {i: v.v.dump(i) for i in outside}
            for i, d in dumps.items():
                last_dump[i] = d
            # snapshots: the first blind select of each ante
            sel = [i for i in outside if dumps[i]["state"] == "BLIND_SELECT" and dumps[i]["ante"] != last_ante[i]]
            if sel:
                v.v.set_curve_crn(12345, False)
                pw = v.v.fast_power(sel, 16)
                v.v.set_curve_crn(0, False)
                for k, i in enumerate(sel):
                    snaps[i].append(snapshot(dumps[i], pw[k]))
                    last_ante[i] = dumps[i]["ante"]
            if outside:
                o = {k: torch.from_numpy(np.asarray(obs[k])[outside]).to(dev) for k in OBS_KEYS if k in obs}
                lg, _ = net(o)
                pa = lg.argmax(1).cpu().numpy()
                for k, i in enumerate(outside):
                    act[i] = int(pa[k])
                    d = dumps[i]
                    decisions[i].append({"a": d["ante"], "b": d["blind_idx"], "s": d["state"][0], "m": d["money"], "off": offers(d), "do": describe(int(pa[k]), d)})
            ids = np.flatnonzero(inround)
            if len(ids):
                if honest is None:
                    raise RuntimeError("in a round with the autoplay solver on")
                act[ids] = honest.moves(v, ids, mask, np.asarray(obs["af"]))
            _, done, info = v.step(act)
            for i in np.flatnonzero(done & ~fin):
                d = last_dump[i]
                res[i] = {"blinds": float(info["blinds"][i]), "won": bool(info["won"][i]), "ante": int(info["ante"][i])}
                if d is not None:
                    snaps[i].append(dict(snapshot(d), end=True))
            fin |= done
            if step % 200 == 0:
                print(json.dumps({"step": step, "done": int(fin.sum()), "s": round(time.time() - t0)}), flush=True)
            if fin.all():
                break
    with open(a.out, "w") as f:
        for i in range(n):
            f.write(json.dumps({"game": i, "seed0": a.seed0, "mode": a.mode, "result": res[i], "snaps": snaps[i], "decisions": decisions[i]}) + "\n")
    bl = np.array([r["blinds"] if r else 0 for r in res])
    print(json.dumps({"games": n, "mode": a.mode, "blinds": round(float(bl.mean()), 3), "wins": sum(1 for r in res if r and r["won"]),
                      "s": round(time.time() - t0), "out": a.out}), flush=True)


# ------------------------------------------------------------------ report
def roles():
    from balatro_rl.az import guide as G

    def role(k):
        for name, st in (("xmult", G.XMULT), ("scaler", G.SCALERS), ("econ", G.ECON), ("mult", G.MULT), ("chips", G.CHIPS)):
            if k in st:
                return name
        return "other"
    return role


def ante_table(games, pr, role):
    from balatro_rl.sim.hands import HAND_NAMES
    pr("| ante | games | reach next | money med | >=$25 | rentals | xMult jokers | >=1 / >=2 xMult | +mult / chips / scaler / econ / other | main hand (share) "
       "| its level | best level | deck | max suit | max rank | enhanced | blue seals | tarots used | planets used | best hand med | target/hand (boss/4) "
       "| share vs boss med | >=1 |")
    pr("|" + "---|" * 23)
    for an in range(1, 9):
        rows = [(g, sn) for g in games for sn in g["snaps"] if sn["ante"] == an and not sn.get("end")]
        if len(rows) < 10:
            continue
        reach = np.mean([any(s2["ante"] > an for s2 in g["snaps"]) or g["result"]["won"] for g, _ in rows])
        money = np.array([sn["money"] for _, sn in rows])
        rent = np.array([sum(j[3] for j in sn["jokers"]) for _, sn in rows])
        rc = [Counter(role(j[0]) for j in sn["jokers"]) for _, sn in rows]
        xm = np.array([c["xmult"] for c in rc])
        mains = [int(np.argmax(sn["played"])) for _, sn in rows]
        mc = Counter(mains).most_common(1)[0]
        lv = np.array([sn["levels"][h] for (_, sn), h in zip(rows, mains)])
        bestlv = np.array([max(sn["levels"]) for _, sn in rows])
        deck = np.array([sn["deck_size"] for _, sn in rows])
        msuit = np.array([max(sn["suits"]) for _, sn in rows])
        mrank = np.array([max(sn["ranks"]) for _, sn in rows])
        enh = np.array([sum(sn["enh"].values()) for _, sn in rows])
        blue = np.array([sn["seals"].get("BLUE", 0) for _, sn in rows])
        tar = np.array([sn["tarots_used"] for _, sn in rows])
        pla = np.array([len(sn["planets_used"]) if isinstance(sn["planets_used"], (list, dict)) else sn["planets_used"] for _, sn in rows])
        best = np.array([sn["power"]["mean_best"] for _, sn in rows if "power" in sn])
        shb = np.array([sn["power"]["share_boss"] for _, sn in rows if "power" in sn])
        boss = 2 * GOLD_BASE[an - 1]
        mix = " / ".join(f"{np.mean([c[k] for c in rc]):.1f}" for k in ("mult", "chips", "scaler", "econ", "other"))
        pr(f"| {an} | {len(rows)} | {reach:.0%} | {np.median(money):.0f} | {np.mean(money >= 25):.0%} | {rent.mean():.2f} | {xm.mean():.2f} | "
           f"{np.mean(xm >= 1):.0%} / {np.mean(xm >= 2):.0%} | {mix} | {HAND_NAMES[mc[0]]} ({mc[1] / len(rows):.0%}) | {lv.mean():.1f} | {bestlv.mean():.1f} | "
           f"{deck.mean():.0f} | {msuit.mean():.1f} | {mrank.mean():.1f} | {enh.mean():.1f} | {blue.mean():.2f} | {tar.mean():.1f} | {pla.mean():.1f} | "
           f"{np.median(best):,.0f} | {boss / 4:,.0f} | {np.median(shb):.2f} | {np.mean(shb >= 1):.0%} |")


def shop_table(games, pr, role):
    from balatro_rl.sim.game import HAND_TO_PLANET
    pr("| ante | shops | money at leave med | left with >=$25 | rerolls/shop | jokers bought | of them rental | xMult offered+affordable : bought "
       "| main-hand planet offered+affordable : bought | Hanging Chad offered : bought | packs opened | blind skips |")
    pr("|" + "---|" * 12)
    for an in range(1, 9):
        st = Counter()
        leaves = []
        for g in games:
            cur = None
            main = 1
            seen = set()
            for d in g["decisions"]:
                if d["a"] != an:
                    continue
                if d["s"] == "B" and d["do"][0] == "skip_blind":
                    st["skips"] += 1
                if d["s"] != "S":
                    continue
                if (d["a"], d["b"]) != cur:
                    cur = (d["a"], d["b"])
                    st["shops"] += 1
                    sn = [s2 for s2 in g["snaps"] if s2["ante"] <= an]
                    main = int(np.argmax(sn[-1]["played"])) if sn else 1
                    seen = set()
                m = d["m"]
                for item in d["off"].get("shop", []):
                    k, cost = item[0], item[1]
                    if k in seen:
                        continue
                    if k.startswith("j_") and role(k[2:]) == "xmult" and cost <= m:
                        seen.add(k)
                        st["x_off"] += 1
                    elif k == "c_" + HAND_TO_PLANET[main] and cost <= m:
                        seen.add(k)
                        st["p_off"] += 1
                    elif k == "j_hanging_chad":
                        seen.add(k)
                        st["c_off"] += 1
                do = d["do"]
                if do[0] == "reroll":
                    st["rerolls"] += 1
                elif do[0] == "pack":
                    st["packs"] += 1
                elif do[0] == "leave":
                    leaves.append(m)
                elif do[0] in ("buy", "replace") and do[1].startswith("j_"):
                    st["jbuy"] += 1
                    tag = next((it[2] for it in d["off"].get("shop", []) if it[0] == do[1] and len(it) > 2), "")
                    st["jrent"] += "R" in tag.split("/")[0]
                    st["x_buy"] += role(do[1][2:]) == "xmult"
                    st["c_buy"] += do[1] == "j_hanging_chad"
                elif do[0] == "buy" and do[1] == "c_" + HAND_TO_PLANET[main]:
                    st["p_buy"] += 1
        if st["shops"] < 10:
            continue
        lv = np.array(leaves) if leaves else np.zeros(1)
        f = lambda o, b: f"{st[o]} : {st[b]}"
        pr(f"| {an} | {st['shops']} | {np.median(lv):.0f} | {np.mean(lv >= 25):.0%} | {st['rerolls'] / st['shops']:.2f} | {st['jbuy']} | {st['jrent']} | "
           f"{f('x_off', 'x_buy')} | {f('p_off', 'p_buy')} | {f('c_off', 'c_buy')} | {st['packs']} | {st['skips']} |")


def deep_vs_dead(games, pr, role):
    groups = {"reached ante 7+": [], "died at ante 5-6": []}
    for g in games:
        s5 = [sn for sn in g["snaps"] if sn["ante"] == 5 and not sn.get("end")]
        if s5:
            deep = any(sn["ante"] >= 7 for sn in g["snaps"]) or g["result"]["won"]
            groups["reached ante 7+" if deep else "died at ante 5-6"].append(s5[0])
    pr("| ante-5 build of games that | games | money | rentals | xMult | +mult | chips | scaler | econ | main level | best level | enhanced | share vs boss med |")
    pr("|" + "---|" * 13)
    for k, sl in groups.items():
        if not sl:
            continue
        rc = [Counter(role(j[0]) for j in sn["jokers"]) for sn in sl]
        pr(f"| {k} | {len(sl)} | {np.mean([sn['money'] for sn in sl]):.1f} | {np.mean([sum(j[3] for j in sn['jokers']) for sn in sl]):.2f} | "
           + " | ".join(f"{np.mean([c[r] for c in rc]):.2f}" for r in ("xmult", "mult", "chips", "scaler", "econ"))
           + f" | {np.mean([sn['levels'][int(np.argmax(sn['played']))] for sn in sl]):.1f} | {np.mean([max(sn['levels']) for sn in sl]):.1f} | "
           f"{np.mean([sum(sn['enh'].values()) for sn in sl]):.1f} | {np.median([sn['power']['share_boss'] for sn in sl if 'power' in sn]):.2f} |")


def report(a):
    role = roles()
    out = []
    pr = out.append
    for path in a.files:
        games = [json.loads(line) for line in open(path)]
        bl = np.array([g["result"]["blinds"] for g in games])
        pr(f"\n## {os.path.basename(path)}: {len(games)} games, {bl.mean():.2f} blinds, {sum(g['result']['won'] for g in games)} wins\n")
        pr("### Builds at the start of each ante\n")
        ante_table(games, pr, role)
        pr("\n### Shop behaviour\n")
        shop_table(games, pr, role)
        pr("\n### What the deep runs had at ante 5\n")
        deep_vs_dead(games, pr, role)
        for an in (4, 6):
            c = Counter(j[0] for g in games for sn in g["snaps"] if sn["ante"] == an and not sn.get("end") for j in sn["jokers"])
            nn = sum(1 for g in games for sn in g["snaps"] if sn["ante"] == an and not sn.get("end"))
            if nn:
                pr(f"\nMost held jokers at the start of ante {an} ({nn} games): " + ", ".join(f"{k} {v / nn:.0%} ({role(k)})" for k, v in c.most_common(15)))
    text = "\n".join(out)
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text + "\n")


# ------------------------------------------------------------------ levers
LEVERS = {                      # name -> list of edits (op, a, b); a joker named by key is resolved to its id, slot "weak" to the slot it takes
    "base": [],
    "levels+3": [(0, -1, 3)],
    "levels+6": [(0, -1, 6)],
    "money+25": [(1, 25, -1)],
    "xmult (Cavendish x3)": [(2, "cavendish", "weak")],
    "Hanging Chad": [(2, "hanging_chad", "weak")],
    "thin 10 low cards": [(4, 10, -1)],
    "13 cards to main suit": [(3, 13, -1)],
    "no rentals": [(5, 0, -1)],
    "levels+3, xmult, thin 10": [(0, -1, 3), (2, "cavendish", "weak"), (4, 10, -1)],
}


def weak_slot(d, role):
    """Where an added joker goes: a free slot (-1), else the first flat (+mult / chips) joker that is not eternal, else the first that is not eternal."""
    js = d["jokers"]
    if len(js) < d["joker_slots"]:
        return -1
    for k, j in enumerate(js):
        if not j["eternal"] and role(j["key"]) in ("mult", "chips"):
            return k
    for k, j in enumerate(js):
        if not j["eternal"]:
            return k
    return -2                                                   # (no room: the edit is skipped)


def play_out(v, net, dev, n, max_steps=6000):
    fin = np.zeros(n, bool)
    bl, won, ante = np.zeros(n), np.zeros(n, bool), np.zeros(n)
    with torch.no_grad():
        for _ in range(max_steps):
            o = {k: torch.from_numpy(np.asarray(v.obs[k])).to(dev) for k in OBS_KEYS if k in v.obs}
            lg, _ = net(o)
            _, d, info = v.step(np.where(fin, -1, lg.argmax(1).cpu().numpy()))
            f = d & ~fin
            bl[f], won[f], ante[f] = info["blinds"][f], info["won"][f], info["ante"][f]
            fin |= d
            if fin.all():
                break
    return bl, won, np.where(won, 9, ante)


def levers(a):
    from balatro_rl.runeval import Cols
    from balatro_rl.sim.jokers import JOKERS
    keys = list(JOKERS)
    role = roles()
    args = json.load(open(a.config))["args"]
    dev = a.device
    net = load_model(a.ckpt, dev).eval()

    def make(n, seed):
        e = CppVecEnv(n, "RED", a.stake, seed0=seed, threads=a.threads)
        if getattr(net, "marg", 0):
            e.set_marginal(int(net.marg))
        e.set_autoplay(True, arg_of(args, "--autoplay-nodes", 400), True)
        e.v.set_solver_tarots(True, arg_of(args, "--solver-targets", 3))
        e.v.set_solver_order(True)
        e.v.set_solver_speed(True, False, 0.0)
        e.v.set_solver_tt(True)
        return e

    antes = [int(x) for x in a.antes.split(",")]
    banks = {an: make(a.bank, 1) for an in antes}
    filled = {an: 0 for an in antes}
    for b in banks.values():
        b.v.restart()
    # 1. the positions: the first blind select of each ante in antes, from fresh games
    v = make(a.games, a.seed0)
    v.v.restart()
    n = a.games
    fin = np.zeros(n, bool)
    taken = {an: np.zeros(n, bool) for an in antes}
    with torch.no_grad():
        for _ in range(6000):
            g = np.asarray(v.obs["g"])
            st = np.argmax(g[:, Cols.STATE:Cols.STATE + 4], 1)
            ante = np.rint(g[:, Cols.ANTE] * 8).astype(int)
            blind = np.argmax(g[:, Cols.BLIND:Cols.BLIND + 3], 1)
            for an in antes:
                ids = [int(i) for i in np.flatnonzero((st == 0) & (ante == an) & (blind == 0) & ~fin & ~taken[an])]
                ids = ids[:max(0, a.bank - filled[an])]
                if ids:
                    banks[an].copy_many(v, ids, list(range(filled[an], filled[an] + len(ids))))
                    filled[an] += len(ids)
                    taken[an][ids] = True
            o = {k: torch.from_numpy(np.asarray(v.obs[k])).to(dev) for k in OBS_KEYS if k in v.obs}
            lg, _ = net(o)
            _, d, _ = v.step(np.where(fin, -1, lg.argmax(1).cpu().numpy()))
            fin |= d
            if fin.all():
                break
    v.close()
    print(json.dumps({"positions": filled}), flush=True)
    out = []
    R = a.repeats
    for an in antes:
        B = filled[an]
        if B < 10:
            continue
        bank = banks[an]
        dumps = [bank.v.dump(k) for k in range(B)]
        P = make(B * R, 2)
        P.v.restart()
        src = [k for k in range(B) for _ in range(R)]
        dst = list(range(B * R))
        salts = [7_000_003 * (k + 1) + 101 * r for k in range(B) for r in range(R)]
        base = None
        start = 3 * (an - 1)
        for name, edits in LEVERS.items():
            P.copy_many(bank, src, dst)
            applied = np.ones(B * R, bool)
            for op, x, y in edits:
                if op == 2:
                    kid = keys.index(x)
                    slots = [weak_slot(dumps[k], role) for k in range(B)]
                    for sl in set(slots):
                        ids = [k * R + r for k in range(B) if slots[k] == sl for r in range(R)]
                        if sl == -2:
                            applied[ids] = False
                            continue
                        P.v.edit(ids, 2, kid, sl)
                else:
                    P.v.edit(dst, op, int(x), int(y))
            P.v.hide_future(dst, salts)
            bl, won, furthest = play_out(P, net, dev, B * R)
            extra = bl - start
            row = {"ante": an, "lever": name, "positions": B, "games": B * R, "applied": round(float(applied.mean()), 3),
                   "blinds_after": round(float(extra.mean()), 3), "reach_ante7": round(float((furthest >= 7).mean()), 4),
                   "reach_ante8": round(float((furthest >= 8).mean()), 4), "wins": int(won.sum())}
            if base is None:
                base = extra
            else:
                dlt = extra - base
                row["gain"] = round(float(dlt.mean()), 3)
                row["gain_se"] = round(float(dlt.std() / np.sqrt(len(dlt))), 3)
            out.append(row)
            print(json.dumps(row), flush=True)
        P.close()
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--ckpt", default="checkpoints/ppo_gold_v51.pt")
    c.add_argument("--config", default="checkpoints/ppo_gold_v51.json")
    c.add_argument("--games", type=int, default=1000)
    c.add_argument("--stake", default="GOLD")
    c.add_argument("--seed0", type=int, default=9_500_000)
    c.add_argument("--mode", choices=["honest", "foresight"], default="honest")
    c.add_argument("--honest", type=int, default=5)
    c.add_argument("--node-cap", type=int, default=400)
    c.add_argument("--scratch", type=int, default=4096)
    c.add_argument("--threads", type=int, default=0)
    c.add_argument("--max-steps", type=int, default=20000)
    c.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c.add_argument("--out", required=True)
    r = sp.add_parser("report")
    r.add_argument("files", nargs="+")
    r.add_argument("--out", default="")
    lv = sp.add_parser("levers")
    lv.add_argument("--ckpt", default="checkpoints/ppo_gold_v51.pt")
    lv.add_argument("--config", default="checkpoints/ppo_gold_v51.json")
    lv.add_argument("--games", type=int, default=2000, help="fresh games the positions are taken from")
    lv.add_argument("--antes", default="4,5")
    lv.add_argument("--bank", type=int, default=400, help="positions kept per ante")
    lv.add_argument("--repeats", type=int, default=8, help="redrawn futures per position (the same for every lever)")
    lv.add_argument("--stake", default="GOLD")
    lv.add_argument("--seed0", type=int, default=9_600_000)
    lv.add_argument("--threads", type=int, default=0)
    lv.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    lv.add_argument("--out", default="")
    a = p.parse_args()
    if a.cmd == "collect":
        collect(a)
    elif a.cmd == "report":
        report(a)
    elif a.cmd == "levers":
        levers(a)


if __name__ == "__main__":
    main()
