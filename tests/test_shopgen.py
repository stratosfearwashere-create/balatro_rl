"""balatro_rl.sim.shopgen reproduces the shops the game's own Lua code makes (frozen from its LuaJIT; see tests/data/shopgen_golden.json).

Each run: the shop cards, a pack and a voucher for antes 1-8 with nothing owned, for stakes 1/4/7/8."""
import json
import os

from balatro_rl.sim.shopgen import ShopGen

GOLD = json.load(open(os.path.join(os.path.dirname(__file__), "data", "shopgen_golden.json")))


def fmt(c):
    e = {"negative": "neg", "polychrome": "poly", "holo": "holo", "foil": "foil"}.get(c["edition"], "")
    return "%s|%s|%s%s%s" % (c["key"], e, "E" if c["eternal"] else "", "P" if c["perishable"] else "", "R" if c["rental"] else "")


def test_shops_match_the_game():
    for run in GOLD:
        g = ShopGen(run["seed"], stake=run["stake"])
        for row in run["shops"]:
            g.ante = row["ante"]
            g.used = set()
            assert [fmt(g.shop_card()) for _ in range(2)] == row["cards"], (run["seed"], row["ante"])
            assert g.pack() == row["pack"], (run["seed"], row["ante"])
            assert g.next_voucher() == row["voucher"], (run["seed"], row["ante"])
