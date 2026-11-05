"""Greedy games on the C++ engine with every decision outside a round written down (and consumables used
inside one), plus a summary of what the money went on.

    python tools/trace_games.py checkpoints/saved/gold_v7_final/final.pt --games 300 --stake GOLD --money 254 --out traces/v7_gold_250.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from balatro_rl import env as E                              # noqa: E402
from balatro_rl.engine_env import CppEnv, OBS_KEYS          # noqa: E402
from balatro_rl.model import load_model                      # noqa: E402
from balatro_rl.az.guide import XMULT, MULT, CHIPS, ECON     # noqa: E402

BLIND = ("small", "big", "boss")


def jname(j: dict) -> str:
    s = j["key"]
    if j.get("edition"):
        s += f"[{j['edition'].lower()}]"
    if j.get("rental"):
        s += "[rent]"
    if j.get("perishable") is not None:
        s += "[perish]"
    if j.get("eternal"):
        s += "[eternal]"
    if j.get("debuffed"):
        s += "[debuffed]"
    return s


def card(c) -> str:
    if isinstance(c, dict) and "rank" in c:
        s = f"{c.get('rank')}{str(c.get('suit', ''))[:1]}"
        for k in ("enh", "enhancement", "edition", "seal"):
            if c.get(k):
                s += f"[{c[k]}]"
        return s
    if isinstance(c, dict):
        return c.get("key", str(c))
    return str(c)


def role(key: str) -> str:
    k = key[2:] if key.startswith("j_") else key
    return "x" if k in XMULT else "+" if k in MULT else "c" if k in CHIPS else "$" if k in ECON else "-"


def describe(d: dict, a: int, obs: dict) -> tuple[str, str, int]:
    """-> (kind, text, cost) of action a in the dumped game d."""
    if d["targeting"]:
        return "target", "choose target cards", 0
    if a < E.A_SELECT:
        return ("play" if a < E.A_DISC else "discard"), "", 0
    if a == E.A_SELECT:
        return "select", f"select {BLIND[d['blind_idx']]} blind" + (f" ({d['boss']})" if d["blind_idx"] == 2 else ""), 0
    if a == E.A_SKIP:
        return "skip", f"skip {BLIND[d['blind_idx']]} blind for tag {d['tags_offered'][d['blind_idx']]}", 0
    if a == E.A_REROLL_BOSS:
        return "reroll_boss", f"reroll boss ({d['boss']})", 0
    if a < E.A_BUY_PACK:
        it = d["shop"][a - E.A_BUY]
        return "buy", f"buy {it['key']} ${it['cost']}" + (f" ({role(it['key'])})" if it["kind"] == "joker" else ""), it["cost"]
    if a < E.A_VOUCHER:
        it = d["shop_packs"][a - E.A_BUY_PACK]
        return "buy_pack", f"buy {it['key']} ${it['cost']}", it["cost"]
    if a < E.A_REROLL:
        it = d["shop_vouchers"][a - E.A_VOUCHER]
        return "voucher", f"buy {it['key']} ${it['cost']}", it["cost"]
    if a == E.A_REROLL:
        c = 0 if d["free_rerolls"] else d["reroll_cost"]
        return "reroll", f"reroll ${c}", c
    if a == E.A_LEAVE:
        return "leave", "leave shop", 0
    if a < E.A_SELL_C:
        j = d["jokers"][a - E.A_SELL_J]
        return "sell_joker", f"sell {jname(j)}", -max(1, j["cost"] // 2 + j.get("sell_bonus", 0))
    if a < E.A_USE_C:
        c = d["consumables"][a - E.A_SELL_C]
        return "sell_cons", f"sell {c['key'] if isinstance(c, dict) else c}", 0
    if a < E.A_PICK:
        c = d["consumables"][a - E.A_USE_C]
        return "use", f"use {c['key'] if isinstance(c, dict) else c}" + (" (in round)" if d["state"] == "SELECTING_HAND" else ""), 0
    if a < E.A_PSKIP:
        x = d["pack_cards"][a - E.A_PICK]
        return "pick", f"pick {jname(x) if isinstance(x, dict) and 'key' in x and 'rank' not in x else card(x)} from {d['pack_kind']} pack", 0
    if a == E.A_PSKIP:
        return "pack_skip", f"skip {d['pack_kind']} pack ({', '.join(jname(x) if isinstance(x, dict) and 'rank' not in x else card(x) for x in d['pack_cards'])})", 0
    if a < E.A_SWAP_END:
        i = a - E.A_SWAP
        return "swap", f"swap {jname(d['jokers'][i])} <-> {jname(d['jokers'][i + 1])}", 0
    i, j = divmod(a - E.A_REPLACE, E.MAX_JOKERS)
    it = d["shop"][i]
    return "replace", f"replace {jname(d['jokers'][j])} with {it['key']} ${it['cost']} ({role(it['key'])})", it["cost"]


def shop_line(d: dict) -> str:
    items = [(f"j_{jname(it['joker'])}${it['cost']}({role(it['key'])})" if it["kind"] == "joker" else f"{it['key']}${it['cost']}") for it in d["shop"]]
    items += [f"{p['key']}${p['cost']}" for p in d["shop_packs"]]
    items += [f"{v['key']}${v['cost']}" for v in d["shop_vouchers"]]
    return " ".join(items)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--money", type=int, default=None, help="every game starts with this much")
    p.add_argument("--out", default="")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    model = load_model(a.checkpoint, a.device)
    n = a.games
    envs = [CppEnv(a.deck, a.stake) for _ in range(n)]
    if getattr(model, "marg", 0):                              # a network made with the marginal-value inputs
        for e in envs:
            e.e.marginal = int(model.marg)
    obs = [e.reset(a.seed0 + i) for i, e in enumerate(envs)]
    if a.money is not None:
        for i, e in enumerate(envs):
            e.game.money = a.money
            obs[i] = e.e.observe()
    live = list(range(n))
    lines = [[] for _ in range(n)]
    spent = [Counter() for _ in range(n)]            # dollars by kind
    counts = [Counter() for _ in range(n)]
    bought = [[] for _ in range(n)]                  # (ante, key, role)
    hands = [Counter() for _ in range(n)]            # plays / discards per round, by (ante, blind)
    last_shop = [None] * n
    result = [None] * n
    rich_at = [{} for _ in range(n)]                 # money at the start of each ante
    while live:
        batch = {k: torch.from_numpy(np.stack([obs[i][k] for i in live])).to(a.device) for k in OBS_KEYS}
        with torch.no_grad():
            logits, _ = model(batch)
        acts = logits.argmax(1).cpu().numpy()
        nxt = []
        for i, ai in zip(live, acts):
            ai = int(ai)
            d = envs[i].game.dump()
            kind, text, cost = describe(d, ai, obs[i])
            where = f"a{d['ante']}.{BLIND[d['blind_idx']] if d['blind_idx'] < 3 else '?'}"
            if d["state"] == "BLIND_SELECT" and d["blind_idx"] == 0:
                rich_at[i].setdefault(d["ante"], d["money"])
            if kind in ("play", "discard"):
                hands[i][(d["ante"], d["blind_idx"], kind)] += 1
            else:
                if d["state"] == "SHOP" and last_shop[i] != (d["ante"], d["blind_idx"], d["shops_seen"], len(lines[i])):
                    key = (d["ante"], d["blind_idx"], d["shops_seen"])
                    if last_shop[i] is None or last_shop[i][:3] != key:
                        lines[i].append(f"  {where:9s} ${d['money']:<4} jokers [{' '.join(jname(j) + role(j['key']) for j in d['jokers'])}]"
                                        f" cons [{' '.join(c['key'] if isinstance(c, dict) else str(c) for c in d['consumables'])}]"
                                        f" levels {[l for l in d['hand_levels']]}")
                        lines[i].append(f"            shop: {shop_line(d)}")
                    last_shop[i] = key + (len(lines[i]),)
                if kind == "reroll":
                    pass
                lines[i].append(f"  {where:9s} ${d['money']:<4} {text}")
                if kind == "reroll" and d["state"] == "SHOP":
                    pass
                counts[i][kind] += 1
                spent[i][kind] += cost
                if kind in ("buy", "replace") and d["state"] == "SHOP":
                    it = d["shop"][(ai - E.A_BUY) if kind == "buy" else (ai - E.A_REPLACE) // E.MAX_JOKERS]
                    if it["kind"] == "joker":
                        bought[i].append((d["ante"], it["key"], role(it["key"])))
            o, r, done, info = envs[i].step(ai)
            if kind == "reroll" and envs[i].game.state == "SHOP":
                dd = envs[i].game.dump()
                lines[i].append(f"            shop: {shop_line(dd)}")
            if done:
                dd = envs[i].game.dump()
                result[i] = dd
            else:
                obs[i] = o
                nxt.append(i)
        live = nxt
    # ---- write
    out = []
    blinds = np.array([r["furthest_blind"] for r in result])
    won = np.array([r["state"] == "WON" for r in result])
    ante = 1 + np.minimum(blinds, 23) // 3
    out.append(f"{a.checkpoint}  {a.stake}  {n} games, seeds {a.seed0}.., start money ${a.money if a.money is not None else 4}")
    out.append(f"{blinds.mean():.2f} blinds, {int(won.sum())} wins ({100 * won.mean():.1f}%), ante 4+ {100 * (ante >= 4).mean():.1f}%, "
               f"ante 6+ {100 * (ante >= 6).mean():.1f}%")
    out.append("")
    tot_spent = Counter(); tot_cnt = Counter()
    for s, c in zip(spent, counts):
        tot_spent.update(s); tot_cnt.update(c)
    out.append("MONEY PER GAME (dollars spent by kind, and how many times)")
    for k in ("buy", "replace", "buy_pack", "voucher", "reroll", "sell_joker"):
        out.append(f"  {k:12s} ${tot_spent[k] / n:6.1f}   x{tot_cnt[k] / n:5.1f}")
    out.append(f"  money at the end: ${np.mean([r['money'] for r in result]):.1f}; at the start of ante 2/3/4/5/6: "
               + " / ".join(f"${np.mean([m[k] for m in rich_at if k in m]):.0f}" if any(k in m for m in rich_at) else "-" for k in (2, 3, 4, 5, 6)))
    out.append("")
    out.append("JOKERS BOUGHT FROM THE SHOP (per game, by ante; x = xMult, + = +mult, c = chips, $ = economy)")
    for k in range(1, 9):
        c = Counter(r for b in bought for (an, _, r) in b if an == k)
        tot = sum(c.values())
        if tot:
            out.append(f"  ante {k}: {tot / n:4.2f} jokers/game  " + " ".join(f"{r}:{c[r] / n:.2f}" for r in ("x", "+", "c", "$", "-")))
    top = Counter(key for b in bought for (_, key, _) in b)
    out.append("  most bought: " + ", ".join(f"{k} {v}" for k, v in top.most_common(20)))
    out.append("")
    out.append("END OF GAME: jokers held (wins / losses)")
    for name, sel in (("wins", won), ("losses at ante 6-8", (~won) & (ante >= 6)), ("losses at ante 4-5", (~won) & (ante >= 4) & (ante <= 5)), ("losses before ante 4", (~won) & (ante < 4))):
        rs = [r for r, s in zip(result, sel) if s]
        if rs:
            c = Counter(j["key"] for r in rs for j in r["jokers"])
            x = np.mean([sum(j["key"] in XMULT for j in r["jokers"]) for r in rs])
            out.append(f"  {name} ({len(rs)}): {x:.2f} xMult jokers; " + ", ".join(f"{k} {v}" for k, v in c.most_common(12)))
    out.append("")
    out.append("=" * 100)
    for i in range(n):
        r = result[i]
        out.append(f"GAME {i} seed {a.seed0 + i}: {'WON' if r['state'] == 'WON' else 'lost at ante ' + str(r['ante']) + ' ' + BLIND[min(r['blind_idx'], 2)] + ' blind' + (' (' + r['boss'] + ')' if r['blind_idx'] == 2 else '')}, "
                   f"{r['furthest_blind']} blinds, ${r['money']} left; jokers [{' '.join(jname(j) + role(j['key']) for j in r['jokers'])}]")
        out.extend(lines[i])
        out.append("")
    text = "\n".join(out)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print("\n".join(out[: out.index("=" * 100)]))
    if a.out:
        print(f"full trace: {a.out}")


if __name__ == "__main__":
    main()
