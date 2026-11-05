"""Phase 2, the seed oracle: what does a seed hold, and is it worth a deep search?

For a seed (exact-seed mode, so every shop, pack, boss and tag is a function of the seed) it lists, per ante: the boss, the skip tags of the small and big blinds, the ante's voucher, the shop
cards after 0, 1, 2 ... rerolls (with price and edition / sticker) and the contents of the packs on offer. Then it scores the seed by the pieces that can be reached in the first antes:
multiplier jokers, scaling jokers, Blueprint / Brainstorm, economy jokers, key vouchers, and whether an early shop can be afforded: the feasibility screen.

    python tools/seed_report.py --seed 42010                  # the readable report of one seed
    python tools/seed_report.py --seeds 43001-43200 --out traces/seed_screen.jsonl   # the screen over many seeds (cheap: no search)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.env import swap_role                                              # noqa: E402
from balatro_rl.potential import GROW                                             # noqa: E402
from balatro_rl.sim.game import Game                                              # noqa: E402
from balatro_rl.sim.seedrng import seed_string                                    # noqa: E402

ECON = {"golden", "rocket", "business", "mail", "reserved_parking", "trading", "faceless", "delayed_grat", "to_the_moon", "egg", "cloud_9", "hallucination", "cartomancer",
        "8_ball", "certificate", "dna", "midas_mask", "gift", "satellite", "vagabond", "riff_raff", "todo_list", "rough_gem", "matador", "ticket", "chaos", "credit_card"}
COPY = {"blueprint", "brainstorm"}
SCALERS = {k for k, g in GROW.items() if g[1] > 0}
KEY_VOUCHERS = {"hieroglyph", "petroglyph", "antimatter", "overstock_plus", "overstock_norm", "clearance_sale", "liquidation", "directors_cut", "retcon", "hone", "glow_up", "grabber", "wasteful",
                "paint_brush", "palette", "crystal_ball", "tarot_tycoon", "planet_tycoon", "telescope", "observatory", "money_tree", "seed_money", "blank", "magic_trick", "illusion", "recyclomancy"}
BUDGET = {1: 10, 2: 18, 3: 26, 4: 34, 5: 40, 6: 45, 7: 50, 8: 55}               # money a run can plausibly spend by the shops of each ante (start $4, blind pay, interest)
WEIGHT = {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.45}                                      # earlier pieces are worth more: they work for longer


def role_of(key):
    if key in COPY:
        return "copy"
    try:
        if swap_role(key) == 2:
            return "xmult"
    except Exception:
        pass
    if key in SCALERS:
        return "scaler"
    if key in ECON:
        return "econ"
    return "plain"


def describe(it):
    if it.kind == "joker" and it.joker is not None:
        j = it.joker
        tags = "".join(f" [{x}]" for x, on in (("negative", getattr(j, "edition", None) == "NEGATIVE"), ("eternal", getattr(j, "eternal", False)), ("rental", getattr(j, "rental", False)),
                                               ("perishable", getattr(j, "perishable", None) not in (None, False, 0))) if on)
        ed = getattr(j, "edition", None)
        return f"{it.key[2:]} ${it.cost}" + (f" ({str(ed).lower()})" if ed else "") + tags
    return f"{it.key[2:] if it.key[1] == '_' else it.key} ${it.cost}"


def pack_contents(g):
    """The contents of the packs on offer, opened one after the other in the order shown (opening one moves the streams on: the second pack's contents are what it holds after the first)."""
    c = g.clone()
    c.money = 10_000
    out = []
    for _ in range(len(c.shop_packs)):
        try:
            c.buy_pack(0)
        except Exception:
            break
        names = []
        for x in getattr(c, "pack_cards", []):
            k = getattr(x, "key", None)
            name = k() if callable(k) else (k or getattr(x, "name", None) or str(x))
            names.append(str(name).replace("j_", "").replace("c_", ""))
        out.append(names)
        try:
            c.pack_skip()
        except Exception:
            break
    return out


def scan(seed, deck="RED", stake="GOLD", antes=8, pairs=6):
    g = Game(seed=1, deck_type=deck, stake=stake, exact_seed=seed_string(seed))
    g.money = 10_000
    res = []
    for ante in range(1, antes + 1):
        if ante > 1:
            g.ante = ante
            g.new_ante()
        rec = {"ante": ante, "boss": g.boss, "tags": list(g.tags_offered), "pairs": [], "packs": [], "voucher": None}
        g.shop, g.shop_packs, g.shop_vouchers, g.ante_voucher = [], [], [], None
        g.open_shop()
        rec["voucher"] = [(it.key[2:], it.cost) for it in g.shop_vouchers]
        contents = pack_contents(g)
        rec["packs"] = [{"key": it.key[2:], "cost": it.cost, "contents": contents[i] if i < len(contents) else []} for i, it in enumerate(g.shop_packs)]
        for k in range(pairs):
            if k:
                g.free_rerolls = 1
                g.reroll()
            rec["pairs"].append([{"kind": it.kind, "key": it.key[2:], "cost": it.cost, "text": describe(it), "role": role_of(it.key[2:]) if it.kind == "joker" else None,
                                  "bad": bool(it.kind == "joker" and it.joker is not None and (getattr(it.joker, "rental", False) or getattr(it.joker, "perishable", None) not in (None, False, 0)))}
                                 for it in g.shop])
        res.append(rec)
    return res


def screen(rep, last_ante=4):
    """The feasibility score of a scanned seed and its breakdown: the pieces reachable by ante `last_ante` within the money a run can plausibly spend, each worth more the earlier it is."""
    found = {"xmult": [], "scaler": [], "copy": [], "econ": [], "voucher": [], "plain_early": 0}
    for rec in rep:
        a = rec["ante"]
        if a > last_ante:
            break
        budget = BUDGET[a]
        for k, pair in enumerate(rec["pairs"]):
            for it in pair:
                if it["kind"] == "joker" and not it["bad"] and it["cost"] <= budget and it["role"] in ("xmult", "scaler", "copy", "econ"):
                    found[it["role"]].append((a, k, it["key"], it["cost"]))
                elif it["kind"] == "joker" and not it["bad"] and a == 1 and k < 3 and it["cost"] <= budget:
                    found["plain_early"] += 1
        for pk in rec["packs"]:
            if pk["key"].startswith("buffoon") and pk["cost"] <= budget:                 # a joker out of a pack costs only the pack
                for name in pk["contents"]:
                    r = role_of(name)
                    if r in ("xmult", "scaler", "copy", "econ"):
                        found[r].append((a, -1, name, pk["cost"]))
        for v, c in rec["voucher"] or []:
            if v in KEY_VOUCHERS and c <= budget:
                found["voucher"].append((a, v))

    def pts(items, per, cap):
        seen, tot = set(), 0.0
        for it in sorted(items):
            key = it[2] if len(it) > 2 else it[1]
            if key in seen:
                continue
            seen.add(key)
            tot += per * WEIGHT[it[0]]
            if len(seen) >= cap:
                break
        return tot
    parts = {"xmult": pts(found["xmult"], 3.0, 3), "scaler": pts(found["scaler"], 2.0, 3), "copy": pts(found["copy"], 3.0, 2), "econ": pts(found["econ"], 1.0, 2),
             "voucher": pts(found["voucher"], 1.0, 2), "early": min(found["plain_early"], 3) * 0.7}
    return sum(parts.values()), parts, found


def text_report(seed, rep, sc, parts, found):
    L = [f"SEED {seed}   feasibility score {sc:.1f}   " + "  ".join(f"{k} {v:.1f}" for k, v in parts.items())]
    for rec in rep:
        L.append(f"ante {rec['ante']}: boss {rec['boss']}   skip tags: small {rec['tags'][0]}, big {rec['tags'][1]}   voucher: " + (", ".join(f"{v} ${c}" for v, c in rec["voucher"]) or "-"))
        for k, pair in enumerate(rec["pairs"]):
            L.append(f"    shop after {k} rerolls: " + "; ".join(it["text"] + (f" <{it['role']}>" if it["role"] in ("xmult", "scaler", "copy", "econ") else "") for it in pair))
        for pk in rec["packs"]:
            L.append(f"    pack {pk['key']} ${pk['cost']}: " + ", ".join(pk["contents"]))
    L.append("reachable by ante 4: " + "; ".join(f"{k}: " + ", ".join(f"{x[2] if len(x) > 2 else x[1]}@a{x[0]}" for x in v) for k, v in found.items() if k != "plain_early" and v))
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="")
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--out", default="")
    a = p.parse_args()
    if a.seed:
        rep = scan(a.seed, a.deck, a.stake, pairs=a.pairs)
        sc, parts, found = screen(rep)
        print(text_report(a.seed, rep, sc, parts, found))
    if a.seeds:
        lo, hi = [int(x) for x in a.seeds.split("-")]
        f = open(a.out, "w") if a.out else None
        for sd in range(lo, hi + 1):
            sc, parts, _ = screen(scan(sd, a.deck, a.stake, pairs=a.pairs))
            row = {"seed": sd, "score": round(sc, 2), **{k: round(v, 2) for k, v in parts.items()}}
            if f:
                f.write(json.dumps(row) + "\n")
            else:
                print(row)


if __name__ == "__main__":
    main()
