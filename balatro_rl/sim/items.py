"""Consumables, vouchers, booster packs, tags and boss blinds."""
from __future__ import annotations

from .hands import (HC, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS,
                    STRAIGHT_FLUSH, FIVE_KIND, FLUSH_HOUSE, FLUSH_FIVE)

# ------------------------------------------------------------------ planets
PLANETS = {
    "pluto": HC, "mercury": PAIR, "uranus": TWO_PAIR, "venus": TRIPS, "saturn": STRAIGHT,
    "jupiter": FLUSH, "earth": FULL_HOUSE, "mars": QUADS, "neptune": STRAIGHT_FLUSH,
    "planet_x": FIVE_KIND, "ceres": FLUSH_HOUSE, "eris": FLUSH_FIVE,
}
HAND_TO_PLANET = {v: k for k, v in PLANETS.items()}

# ------------------------------------------------------------------ tarots
# target: number of hand cards needed (0 = none). Targeting is chosen by a fixed
# heuristic (see game.auto_targets) so the agent only decides *whether/which* to use.
TAROTS = {
    "fool": 0, "magician": 2, "high_priestess": 0, "empress": 2, "emperor": 0, "heirophant": 2,
    "lovers": 1, "chariot": 1, "justice": 1, "hermit": 0, "wheel_of_fortune": 0, "strength": 2,
    "hanged_man": 2, "death": 2, "temperance": 0, "devil": 1, "tower": 1, "star": 3, "moon": 3,
    "sun": 3, "judgement": 0, "world": 3,
}
TAROT_ENH = {"magician": "LUCKY", "empress": "MULT", "heirophant": "BONUS", "lovers": "WILD",
             "chariot": "STEEL", "justice": "GLASS", "devil": "GOLD", "tower": "STONE"}
TAROT_SUIT = {"world": 0, "sun": 1, "moon": 2, "star": 3}

# ------------------------------------------------------------------ spectrals (all 18)
SPECTRALS = {
    "familiar": 0, "grim": 0, "incantation": 0, "talisman": 1, "aura": 1, "wraith": 0, "sigil": 0,
    "ouija": 0, "ectoplasm": 0, "immolate": 0, "ankh": 0, "deja_vu": 1, "hex": 0, "trance": 1,
    "medium": 1, "cryptid": 1, "black_hole": 0, "soul": 0,
}
# The Soul and Black Hole never appear in the normal pool; each spectral/arcana (Soul) or
# celestial/spectral (Black Hole) slot has a 0.3% chance to be replaced by them.
HIDDEN_SPECTRALS = ("soul", "black_hole")
SPECTRAL_POOL = [k for k in SPECTRALS if k not in HIDDEN_SPECTRALS]

# ------------------------------------------------------------------ vouchers
# name -> prerequisite (tier-2 vouchers need tier-1)
VOUCHERS = {
    "overstock_norm": None, "clearance_sale": None, "hone": None, "reroll_surplus": None,
    "crystal_ball": None, "telescope": None, "grabber": None, "wasteful": None,
    "tarot_merchant": None, "planet_merchant": None, "seed_money": None, "blank": None,
    "hieroglyph": None, "magic_trick": None, "directors_cut": None, "paint_brush": None,
    "overstock_plus": "overstock_norm", "liquidation": "clearance_sale", "glow_up": "hone",
    "reroll_glut": "reroll_surplus", "observatory": "telescope", "nacho_tong": "grabber",
    "recyclomancy": "wasteful", "tarot_tycoon": "tarot_merchant", "planet_tycoon": "planet_merchant",
    "money_tree": "seed_money", "antimatter": "blank", "petroglyph": "hieroglyph",
    "illusion": "magic_trick", "retcon": "directors_cut", "palette": "paint_brush",
    "omen_globe": "crystal_ball",
}
VOUCHER_COST = 10

# ------------------------------------------------------------------ packs
# kind, size -> (cards shown, picks, cost, weight)
PACKS = {
    ("arcana", "normal"): (3, 1, 4, 4.0), ("arcana", "jumbo"): (5, 1, 6, 2.0), ("arcana", "mega"): (5, 2, 8, 0.5),
    ("celestial", "normal"): (3, 1, 4, 4.0), ("celestial", "jumbo"): (5, 1, 6, 2.0), ("celestial", "mega"): (5, 2, 8, 0.5),
    ("standard", "normal"): (3, 1, 4, 4.0), ("standard", "jumbo"): (5, 1, 6, 2.0), ("standard", "mega"): (5, 2, 8, 0.5),
    ("buffoon", "normal"): (2, 1, 4, 1.2), ("buffoon", "jumbo"): (4, 1, 6, 0.6), ("buffoon", "mega"): (4, 2, 8, 0.15),
    ("spectral", "normal"): (2, 1, 4, 0.6), ("spectral", "jumbo"): (4, 1, 6, 0.3), ("spectral", "mega"): (4, 2, 8, 0.07),
}
PACK_KEYS = list(PACKS)

# ------------------------------------------------------------------ tags (subset)
TAGS = ["economy", "investment", "uncommon", "rare", "foil", "holo", "polychrome", "negative",
        "buffoon", "meteor", "charm", "coupon", "d_six", "top_up", "speed", "handy", "garbage",
        "orbital", "double", "voucher", "standard", "ethereal", "boss", "juggle"]      # all 24
TAG_MIN_ANTE = {"negative": 2, "buffoon": 2, "meteor": 2, "top_up": 2, "handy": 2, "garbage": 2,
                "orbital": 2, "ethereal": 2, "standard": 2, "rare": 1}
DECKS = ["RED", "BLUE", "YELLOW", "GREEN", "BLACK", "MAGIC", "NEBULA", "GHOST", "ABANDONED",
         "CHECKERED", "ZODIAC", "PAINTED", "ANAGLYPH", "PLASMA", "ERRATIC"]

# ------------------------------------------------------------------ boss blinds
# key -> (display name, min ante, blind multiplier)
BOSSES = {
    "hook": ("The Hook", 1, 2), "ox": ("The Ox", 6, 2), "wall": ("The Wall", 2, 4),
    "arm": ("The Arm", 2, 2), "club": ("The Club", 1, 2), "goad": ("The Goad", 1, 2),
    "window": ("The Window", 1, 2), "head": ("The Head", 1, 2), "psychic": ("The Psychic", 1, 2),
    "water": ("The Water", 2, 2), "manacle": ("The Manacle", 1, 2), "eye": ("The Eye", 3, 2),
    "mouth": ("The Mouth", 2, 2), "plant": ("The Plant", 4, 2), "serpent": ("The Serpent", 5, 2),
    "pillar": ("The Pillar", 1, 2), "needle": ("The Needle", 2, 1), "flint": ("The Flint", 2, 2),
    "tooth": ("The Tooth", 3, 2),
    # face-down bosses: cards are dealt hidden (the agent can't see them)
    "house": ("The House", 2, 2), "wheel": ("The Wheel", 2, 2), "fish": ("The Fish", 2, 2),
    "mark": ("The Mark", 2, 2),
}
FINISHERS = {
    "violet_vessel": ("Violet Vessel", 8, 6), "verdant_leaf": ("Verdant Leaf", 8, 2),
    "crimson_heart": ("Crimson Heart", 8, 2), "amber_acorn": ("Amber Acorn", 8, 2),
    "cerulean_bell": ("Cerulean Bell", 8, 2),
}
ALL_BOSSES = {**BOSSES, **FINISHERS}
BOSS_KEYS = list(ALL_BOSSES)
BOSS_BY_NAME = {v[0]: k for k, v in ALL_BOSSES.items()}
SUIT_BOSS = {"goad": 0, "head": 1, "club": 2, "window": 3}

# Balatro blind base chip requirement by ante for scaling levels 1/2/3 (Gold Stake uses 3)
BLIND_BASE = {
    1: [300, 800, 2000, 5000, 11000, 20000, 35000, 50000],
    2: [300, 900, 2600, 8000, 20000, 36000, 60000, 100000],
    3: [300, 1000, 3200, 9000, 25000, 60000, 110000, 200000],
}
STAKES = ["WHITE", "RED", "GREEN", "BLACK", "BLUE", "PURPLE", "ORANGE", "GOLD"]

# ------------------------------------------------------------------ global item vocabulary
# Used to give every buyable/holdable thing an integer id for the neural net's embedding.
def _vocab():
    from .jokers import JOKER_KEYS
    v = ["<none>", "<unknown>", "<playing_card>"]
    v += [f"j_{k}" for k in JOKER_KEYS] + ["j_unknown"]
    v += [f"c_{k}" for k in PLANETS] + [f"c_{k}" for k in TAROTS] + [f"c_{k}" for k in SPECTRALS]
    v += ["c_unknown"]
    v += [f"v_{k}" for k in VOUCHERS] + ["v_unknown"]
    v += [f"p_{k}_{s}" for k, s in PACK_KEYS]
    v += [f"tag_{t}" for t in TAGS]
    return v


VOCAB = _vocab()
VOCAB_INDEX = {k: i for i, k in enumerate(VOCAB)}


def item_id(key: str) -> int:
    if key in VOCAB_INDEX:
        return VOCAB_INDEX[key]
    if key.startswith("j_"):
        return VOCAB_INDEX["j_unknown"]
    if key.startswith("c_"):
        return VOCAB_INDEX["c_unknown"]
    if key.startswith("v_"):
        return VOCAB_INDEX["v_unknown"]
    return VOCAB_INDEX["<unknown>"]
