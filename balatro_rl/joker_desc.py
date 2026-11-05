"""What each item does, as a fixed feature vector per vocabulary entry (model.ActorCritic, desc=True).

An item id alone tells the network nothing until it has held that item many times; these features say what
kind of thing it is (a multiplier, a +mult, chips, economy), what it depends on (a hand type, suits, ranks,
discards, the deck, money ...), when it acts (the joker's hooks in sim/jokers.py) and whether it grows over a
run, so what is learned about one joker carries over to the ones like it.

    item_descriptors() -> float32 [VOCAB_SIZE, K];  FEATURES: the K column names
"""
from __future__ import annotations

import numpy as np

from .az.guide import CHIPS, MULT, XMULT, ECON, NARROW, GENERIC_EARLY, SLOW_SCALERS
from .sim.hands import HAND_NAMES
from .sim.items import VOCAB, PLANETS
from .sim.jokers import JOKERS

HOOKS = ("before", "card", "retrig", "held", "main", "after", "discard", "end_round", "init", "blind_select", "on_sell")

# value that builds up over a run (and the ones that start strong and run down)
GROWS = {"runner", "square", "castle", "wee", "hiker", "green_joker", "ride_the_bus", "red_card", "trousers", "flash",
         "ceremonial", "supernova", "fortune_teller", "constellation", "hologram", "vampire", "obelisk", "madness",
         "lucky_cat", "glass", "campfire", "hit_the_road", "throwback", "caino", "yorick", "rocket", "egg", "bull",
         "bootstraps", "erosion", "steel_joker", "stone", "satellite"}
SHRINKS = {"ice_cream", "popcorn", "ramen", "turtle_bean", "selzer", "gros_michel", "invisible", "diet_cola"}
# the hand type it pays for
HAND = {"pair": {"jolly", "sly", "duo"}, "two_pair": {"mad", "clever", "trousers"},
        "three": {"zany", "wily", "trio"}, "four": {"family", "square"},
        "straight": {"crazy", "devious", "order", "runner", "shortcut", "four_fingers", "superposition"},
        "flush": {"droll", "crafty", "tribe", "four_fingers", "smeared"}}
SUITS = {"greedy_joker", "lusty_joker", "wrathful_joker", "gluttenous_joker", "rough_gem", "bloodstone", "arrowhead",
         "onyx_agate", "blackboard", "flower_pot", "seeing_double", "ancient", "castle", "smeared"}
RANKS = {"scary_face", "smiley", "photograph", "sock_and_buskin", "pareidolia", "business", "midas_mask",
         "reserved_parking", "faceless", "baron", "shoot_the_moon", "triboulet", "even_steven", "odd_todd", "fibonacci",
         "scholar", "walkie_talkie", "hack", "wee", "8_ball", "sixth_sense", "hit_the_road", "mail", "idol",
         "raised_fist", "caino", "superposition"}
BOSS = {"luchador", "chicot", "mr_bones", "matador"}
COPIES = {"blueprint", "brainstorm"}
CREATES = {"8_ball", "sixth_sense", "seance", "superposition", "cartomancer", "vagabond", "hallucination", "riff_raff",
           "marble", "certificate", "dna", "perkeo", "invisible", "space", "burnt"}
ROOM = {"juggler", "drunkard", "troubadour", "merry_andy", "burglar", "turtle_bean", "stuntman", "credit_card", "chaos",
        "splash", "oops", "ring_master", "astronomer"}                 # hands, discards, hand size, rules, prices
DECK = {"steel_joker", "stone", "glass", "lucky_cat", "drivers_license", "erosion", "hologram", "ticket", "marble",
        "blue_joker", "vampire", "midas_mask", "dna", "certificate", "trading", "caino"}
MONEY = {"bull", "bootstraps", "to_the_moon", "vagabond", "credit_card"}
JOKERS_HELD = {"abstract", "stencil", "swashbuckler", "baseball", "blueprint", "brainstorm", "gift", "ceremonial",
               "madness", "riff_raff", "campfire", "invisible"}
DISCARDS = {"banner", "mystic_summit", "castle", "hit_the_road", "ramen", "green_joker", "faceless", "mail", "trading",
            "burnt", "yorick", "drunkard", "merry_andy", "delayed_grat"}
CONSUMABLES = {"constellation", "fortune_teller", "satellite", "astronomer", "cartomancer", "8_ball", "sixth_sense",
               "seance", "superposition", "vagabond", "hallucination", "perkeo"}
CHANCE = {"oops", "bloodstone", "business", "8_ball", "space", "misprint", "lucky_cat", "gros_michel", "cavendish",
          "hallucination", "reserved_parking", "glass"}
DESTROYS = {"ceremonial", "madness", "trading", "sixth_sense", "gros_michel", "cavendish", "glass", "caino"}

_SETS = [("chips", CHIPS), ("mult", MULT), ("xmult", XMULT), ("economy", ECON), ("grows", GROWS),
         ("grows_slowly", SLOW_SCALERS), ("shrinks", SHRINKS), ("narrow", NARROW), ("generic_early", GENERIC_EARLY),
         ("suits", SUITS), ("ranks", RANKS), ("boss_tool", BOSS), ("copies", COPIES), ("creates", CREATES),
         ("room", ROOM), ("deck", DECK), ("money", MONEY), ("jokers_held", JOKERS_HELD), ("discards", DISCARDS),
         ("consumables", CONSUMABLES), ("chance", CHANCE), ("destroys", DESTROYS)]
_TYPES = (("j_", "joker"), ("c_", "consumable"), ("v_", "voucher"), ("p_", "pack"), ("tag_", "tag"),
          ("<playing_card>", "playing_card"))

FEATURES = ([f"is_{n}" for _, n in _TYPES] + [f"rarity_{r}" for r in (1, 2, 3, 4)] + ["cost"]
            + [n for n, _ in _SETS] + [f"hook_{h}" for h in HOOKS] + ["eternal_ok", "perish_ok", "copyable"]
            + [f"hand_{h}" for h in HAND] + [f"planet_{h}" for h in HAND_NAMES])


def features_spec() -> list:
    """Column names of item_descriptors(spec=True)."""
    from .joker_spec import FEATURE_NAMES
    return FEATURES + list(FEATURE_NAMES)


def _check():
    """Every name in the sets above is a joker of this game (a typo would silently describe nothing)."""
    names = set().union(GROWS, SHRINKS, SUITS, RANKS, BOSS, COPIES, CREATES, ROOM, DECK, MONEY, JOKERS_HELD, DISCARDS,
                        CONSUMABLES, CHANCE, DESTROYS, *HAND.values())
    bad = sorted(names - set(JOKERS))
    if bad:
        raise KeyError(f"joker_desc: not jokers: {bad}")


STRENGTH_NAMES = ("str_a2", "str_a4", "str_a6", "str_a8", "str_measured")


def strength_table() -> np.ndarray:
    """float32 [len(VOCAB), 5]: each joker's measured strength (tools/joker_strength.py, exported by tools/export_strength.py) at the bosses of antes 2, 4, 6, 8 and a flag that it was measured."""
    import json
    import os
    out = np.zeros((len(VOCAB), len(STRENGTH_NAMES)), dtype=np.float32)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joker_strength.json")
    data = json.load(open(path))["jokers"]
    for i, key in enumerate(VOCAB):
        if key.startswith("j_") and key[2:] in data:
            e = data[key[2:]]
            out[i, :4] = e["gains"]
            out[i, 4] = e["measured"]
    return out


def item_descriptors(spec: bool = False, strength: bool = False) -> np.ndarray:
    """spec=True: the exact numeric per-joker columns of joker_spec (FEATURES_SPEC) follow the columns above, so K
    grows; spec=False is the table every older network was trained with, unchanged. strength=True (with spec): five more columns, the measured strength of each joker."""
    out = _item_descriptors()
    if not spec:
        return out
    from .joker_spec import vectorise
    out = np.concatenate([out, vectorise()], axis=1)
    if strength:
        out = np.concatenate([out, strength_table()], axis=1)
    return out


def _item_descriptors() -> np.ndarray:
    _check()
    col = {n: i for i, n in enumerate(FEATURES)}
    out = np.zeros((len(VOCAB), len(FEATURES)), dtype=np.float32)
    for i, key in enumerate(VOCAB):
        for prefix, name in _TYPES:
            if key.startswith(prefix):
                out[i, col[f"is_{name}"]] = 1.0
        if key.startswith("j_") and key[2:] in JOKERS:
            k, d = key[2:], JOKERS[key[2:]]
            out[i, col[f"rarity_{d.rarity}"]] = 1.0
            out[i, col["cost"]] = d.cost / 10.0
            for n, s in _SETS:
                out[i, col[n]] = float(k in s)
            for h in HOOKS:
                out[i, col[f"hook_{h}"]] = float(getattr(d, h) is not None)
            out[i, col["eternal_ok"]], out[i, col["perish_ok"]], out[i, col["copyable"]] = (
                float(d.eternal_ok), float(d.perish_ok), float(d.copyable))
            for h, s in HAND.items():
                out[i, col[f"hand_{h}"]] = float(k in s)
        elif key.startswith("c_") and key[2:] in PLANETS:
            hand = PLANETS[key[2:]]
            hi = HAND_NAMES.index(hand) if hand in HAND_NAMES else (hand if isinstance(hand, int) else -1)
            if 0 <= hi < len(HAND_NAMES):
                out[i, col[f"planet_{HAND_NAMES[hi]}"]] = 1.0
    return out
