"""A rule player for decisions outside rounds built from GOLD_STAKE_HEURISTICS.md (Balatro University's
Gold Stake lessons) and the per-item verdicts in GOLD_STAKE_ITEM_VERDICTS.md. Rounds stay with the exact
solver. Used through AgentConfig(shop_prior="guide"): the chosen option gets +heur_bonus logits.

The rules it carries (section numbers refer to the heuristics file):
  1  survive first: with the next boss not safe, buy immediate scoring and ignore the interest line;
     safe, prefer a value generator and keep the $25 line (2, 3)
  2  slots by role: one chips source, one +mult, one scaler, value generators; generic beats narrow; no
     slow scalers late; stop buying chips jokers once pair levels carry (5)
  3  stickers: at most one rental and only early, never eternal+rental, eternal fine on generic power and on
     one value generator but not on chips jokers, perishable only as an early bridge (4)
  4  vouchers first in a shop; Overstock, Clearance Sale, Grabber, boss reroll, Tarot Merchant, Crystal
     Ball, Telescope when the plan is one hand; none before a scoring joker (8)
  5  packs: open before buying jokers; Buffoon early, Arcana from about $19, Standard until ante 5 for
     seals and gold cards, Celestial when a level is needed, Spectral only when safe (6)
  6  rerolls: $5-7 fine, spread across shops, not with two rentals, not while holding Death or Fool (3)
  7  sell spent jokers (Popcorn, Ice Cream, expired perishables) and spare consumables that cross an
     interest step (3, 14)
  8  skip only for Investment on the big blind of antes 1-2 when the boss is already beatable, Top-up with no
     jokers; boss reroll on the hand-restricting killers (9, 12)
  9  default hand is pairs; planets go on the hand played; use Hermit near $20, Temperance when it pays;
     hold Fool for a good copy (7, 10)
"""
from __future__ import annotations

import random
from collections import Counter

import numpy as np

from ..rewards.potential import _seed
from ..rewards.strength import clear_chances
from ..sim.game import Consumable, Game
from ..sim.hands import N_HANDS, PAIR, HC, FLUSH, STRAIGHT
from ..sim.items import PLANETS
from ..sim.jokers import Joker

# ------------------------------------------------------------------ item knowledge
# base value of a joker in dollars-equivalent (verdict table: how often he takes it, what he says it does)
VALUE = {
    "hanging_chad": 9, "mail": 9, "triboulet": 9, "runner": 8, "ride_the_bus": 8, "castle": 8, "wee": 8,
    "trading": 8, "square": 7, "hologram": 7, "red_card": 7, "golden": 7, "blueprint": 7, "abstract": 6,
    "green_joker": 6, "supernova": 6, "burglar": 6, "certificate": 6, "dna": 6, "campfire": 6, "card_sharp": 6,
    "stuntman": 6, "cavendish": 6, "caino": 6, "chicot": 6, "perkeo": 6, "banner": 5, "mystic_summit": 5,
    "raised_fist": 5, "blue_joker": 5, "blackboard": 5, "vampire": 5, "constellation": 5, "hallucination": 5,
    "business": 5, "faceless": 5, "ceremonial": 5, "fibonacci": 5, "burnt": 5, "gros_michel": 5, "idol": 5,
    "ancient": 5, "duo": 5, "rocket": 5, "photograph": 5, "odd_todd": 4, "popcorn": 4, "mime": 4,
    "reserved_parking": 4, "midas_mask": 4, "bull": 4, "shoot_the_moon": 4, "bootstraps": 4, "8_ball": 4,
    "obelisk": 4, "misprint": 4, "gift": 4, "cartomancer": 4, "even_steven": 4, "riff_raff": 4, "todo_list": 4,
    "scary_face": 4, "drunkard": 4, "sly": 4, "hit_the_road": 4, "ramen": 4, "diet_cola": 4, "arrowhead": 4,
    "baseball": 4, "crafty": 4, "walkie_talkie": 4, "flash": 4, "onyx_agate": 4, "satellite": 4,
    "seeing_double": 4, "delayed_grat": 4, "trousers": 4, "droll": 4, "hiker": 4, "space": 4, "half": 4,
    "clever": 4, "jolly": 4, "selzer": 4, "scholar": 4, "baron": 4, "to_the_moon": 4, "bloodstone": 4,
    "chaos": 4, "stencil": 4, "yorick": 4, "fortune_teller": 3, "acrobat": 3, "brainstorm": 3, "shortcut": 3,
    "four_fingers": 3, "madness": 3, "swashbuckler": 3, "superposition": 3, "juggler": 3, "wrathful_joker": 3,
    "lusty_joker": 3, "lucky_cat": 3, "drivers_license": 3, "devious": 3, "egg": 3, "hack": 3, "astronomer": 3,
    "zany": 3, "luchador": 3, "mad": 3, "steel_joker": 3, "troubadour": 3, "erosion": 3, "sock_and_buskin": 3,
    "greedy_joker": 3, "matador": 3, "cloud_9": 3, "glass": 3, "rough_gem": 3, "vagabond": 3, "smiley": 3,
    "order": 3, "throwback": 3, "mr_bones": 3, "smeared": 3, "flower_pot": 3, "trio": 3, "gluttenous_joker": 3,
    "loyalty_card": 3, "family": 3, "tribe": 3, "pareidolia": 3, "joker": 3, "ice_cream": 3, "merry_andy": 3,
    "turtle_bean": 3, "credit_card": 2, "splash": 2, "dusk": 2, "marble": 2, "stone": 2, "invisible": 2,
    "seance": 2, "oops": 2, "sixth_sense": 2, "ring_master": 2,
}
CHIPS = {"sly", "wily", "clever", "devious", "crafty", "banner", "scary_face", "odd_todd", "blue_joker", "ice_cream",
         "runner", "square", "bull", "arrowhead", "stone", "castle", "wee", "stuntman", "hiker", "fibonacci",
         "walkie_talkie", "misprint"}
MULT = {"joker", "jolly", "zany", "mad", "crazy", "droll", "half", "abstract", "mystic_summit", "raised_fist", "popcorn",
        "gros_michel", "even_steven", "greedy_joker", "lusty_joker", "wrathful_joker", "gluttenous_joker", "smiley",
        "scholar", "swashbuckler", "fortune_teller", "ride_the_bus", "green_joker", "red_card", "supernova",
        "ceremonial", "flash", "trousers", "onyx_agate", "bootstraps", "shoot_the_moon", "erosion", "misprint",
        "blackboard", "walkie_talkie"}
XMULT = {"duo", "trio", "family", "order", "tribe", "baron", "ancient", "obelisk", "cavendish", "card_sharp", "loyalty_card",
         "acrobat", "blackboard", "flower_pot", "seeing_double", "idol", "constellation", "ramen", "stencil",
         "photograph", "bloodstone", "madness", "vampire", "hologram", "steel_joker", "lucky_cat", "baseball",
         "campfire", "throwback", "glass", "hit_the_road", "drivers_license", "caino", "triboulet", "yorick"}
SCALERS = {"ride_the_bus", "green_joker", "red_card", "supernova", "ceremonial", "flash", "trousers", "square",
           "runner", "castle", "bull", "wee", "hologram", "constellation", "vampire", "lucky_cat", "obelisk",
           "madness", "campfire", "yorick", "glass", "hit_the_road", "spare_trousers", "fortune_teller", "ice_cream",
           "popcorn", "throwback", "canio", "caino"}
SLOW_SCALERS = {"hologram", "lucky_cat", "obelisk", "constellation", "vampire", "madness"}
ECON = {"golden", "rocket", "business", "mail", "reserved_parking", "trading", "faceless", "delayed_grat", "to_the_moon",
        "egg", "cloud_9", "hallucination", "cartomancer", "8_ball", "certificate", "dna", "midas_mask", "gift",
        "satellite", "vagabond", "riff_raff", "hanging_chad", "todo_list", "rough_gem", "matador", "ticket",
        "superposition", "sixth_sense", "seance", "astronomer", "chaos", "credit_card"}
NARROW = {"lusty_joker", "greedy_joker", "wrathful_joker", "gluttenous_joker", "onyx_agate", "arrowhead", "rough_gem",
          "bloodstone", "flower_pot", "seeing_double", "order", "tribe", "family", "trio", "duo", "crazy", "droll",
          "devious", "clever", "wily", "sly", "mad", "zany", "jolly", "shoot_the_moon", "walkie_talkie", "scary_face",
          "smiley", "odd_todd", "even_steven", "fibonacci", "hack", "dusk", "sock_and_buskin"}
GENERIC_EARLY = {"abstract", "banner", "jolly", "half", "mystic_summit", "raised_fist", "popcorn", "gros_michel", "misprint",
                 "joker", "blue_joker", "odd_todd", "scary_face", "even_steven"}

VOUCHERS = {"overstock_norm": 10, "overstock_plus": 8, "clearance_sale": 10, "liquidation": 10, "grabber": 12, "nacho_tong": 8,
            "directors_cut": 8, "retcon": 4, "tarot_merchant": 7, "tarot_tycoon": 4, "crystal_ball": 8, "telescope": 7,
            "observatory": 9, "paint_brush": 7, "palette": 5, "wasteful": 5, "recyclomancy": 3, "seed_money": 6,
            "money_tree": 7, "hone": 3, "glow_up": 3, "reroll_surplus": 5, "reroll_glut": 4, "hieroglyph": 0,
            "petroglyph": 0, "antimatter": 12, "blank": 1, "magic_trick": 0, "illusion": 0, "planet_merchant": 0,
            "planet_tycoon": 0, "omen_globe": 2}
KILLERS = {"needle", "eye", "mouth", "psychic", "water", "arm", "wall", "violet_vessel", "hook", "manacle", "flint"}
TAROT_PICK = {"hanged_man": 8, "hermit": 8, "death": 7, "fool": 7, "temperance": 6, "judgement": 6, "justice": 5,
              "chariot": 5, "devil": 5, "emperor": 5, "strength": 4, "high_priestess": 4, "empress": 4,
              "heirophant": 4, "hierophant": 4, "magician": 3, "lovers": 3, "moon": 2, "sun": 2, "star": 2, "world": 2,
              "tower": 2, "wheel_of_fortune": 1}
SPECTRAL_PICK = {"soul": 9, "black_hole": 8, "hex": 8, "wraith": 6, "deja_vu": 5, "cryptid": 5, "talisman": 4,
                 "trance": 4, "medium": 3, "ankh": 3, "immolate": 3, "ectoplasm": 3, "aura": 3, "familiar": 2,
                 "grim": 2, "incantation": 2, "ouija": 2, "sigil": 1}
HOLD_TAROTS = {"death", "fool", "hanged_man", "hermit", "temperance", "justice", "chariot", "strength", "judgement"}
HAND_PLANETS = {h: p for p, h in PLANETS.items()}


# ------------------------------------------------------------------ the state read
class View:
    """What the rules look at, computed once per decision."""

    def __init__(self, w, pricer, rng: random.Random):
        g: Game = w.g
        self.w, self.g = w, g
        self.money, self.ante, self.stake = g.money, g.ante, g.stake
        self.jokers = list(g.jokers)
        keys = [j.key for j in self.jokers]
        self.has_chips = any(k in CHIPS for k in keys)
        self.has_mult = any(k in MULT for k in keys)
        self.has_xmult = any(k in XMULT for k in keys)
        self.has_scaler = any(k in SCALERS for k in keys)
        self.econ = [j for j in self.jokers if j.key in ECON]
        self.scoring = [j for j in self.jokers if (j.key in CHIPS or j.key in MULT or j.key in XMULT) and not j.debuffed]
        self.rentals = sum(1 for j in self.jokers if j.rental)
        self.eternal_econ = sum(1 for j in self.econ if j.eternal)
        self.slots_free = g.joker_slots - len(self.jokers)
        self.cons_free = g.consumable_slots - len(g.consumables)
        self.main = main_hand(g)
        self.main_level = g.hand_levels[self.main]
        # how safe the next blinds are: the pricer's shared sampled hands
        ctx = pricer.context(g)
        self.p_boss = ctx.p_boss
        scfg = pricer.potential.cfg.strength
        seed = _seed(("guide", ctx.seed))
        nxt = g.blind_idx if g.state == "BLIND_SELECT" else min(g.blind_idx + 1, 2)
        if g.state == "BLIND_SELECT":
            targets = [g.blind_target(i) for i in range(g.blind_idx, 3)]
        else:
            targets = [g.blind_target(i) for i in range(nxt, 3)]
        p = clear_chances(ctx.scores, g.round_hands(), g.round_discards(), targets, scfg, seed) if targets else [1.0]
        self.p_next = float(p[0])
        self.safe = self.p_boss >= 0.8 and self.p_next >= 0.85
        self.reserve = 0 if (self.ante <= 1 and not self.scoring) else (20 if self.ante <= 2 else 25)
        if not self.safe:
            self.reserve = min(self.reserve, 5)
        self.deck_n = len(g.full_deck)
        ranks = Counter(c.rank for c in g.full_deck if not c.is_stone)
        self.top_ranks = {r for r, _ in ranks.most_common(3)}
        suits = Counter(c.suit for c in g.full_deck if not c.is_stone)
        self.top_suit = suits.most_common(1)[0][0] if suits else 0
        self.holding = {c.name for c in g.consumables}


def main_hand(g: Game) -> int:
    """Pairs unless the build says otherwise (10)."""
    keys = {j.key for j in g.jokers}
    if keys & {"stuntman", "square", "wee", "supernova"} and g.hand_levels[HC] >= g.hand_levels[PAIR]:
        return HC
    if keys & {"four_fingers", "shortcut", "runner"} and g.hand_levels[STRAIGHT] >= g.hand_levels[PAIR] + 1:
        return STRAIGHT
    best = max(range(N_HANDS), key=lambda h: (g.hand_levels[h], g.hand_played[h]))
    if g.hand_levels[best] >= g.hand_levels[PAIR] + 2 and best in (HC, FLUSH, STRAIGHT, 2, 3, 6, 7):
        return best
    return PAIR


# ------------------------------------------------------------------ joker pricing
def joker_value(v: View, j: Joker, in_shop: bool = True) -> float:
    """Dollars-equivalent value of holding j in this build (2, 3, 4, 5)."""
    k = j.key
    val = float(VALUE.get(k, {1: 3, 2: 4, 3: 5, 4: 7}.get(getattr(j.d, "rarity", 1), 3)))
    scoring = k in CHIPS or k in MULT or k in XMULT
    if not v.safe:                                        # survive first: immediate scoring counts double
        if scoring and k not in SLOW_SCALERS:
            val += 4 if k not in SCALERS or k in ("ride_the_bus", "green_joker", "red_card", "supernova") else 2
        if k in ECON and not scoring:
            val -= 3
    else:
        if k in ECON and len(v.econ) == 0:
            val += 3
        if k in ECON and v.eternal_econ >= 1 and j.eternal:
            val -= 4
    if k in CHIPS and not v.has_chips:
        val += 3
    elif k in CHIPS and v.has_chips:
        val -= 2                                          # two chips jokers are redundant
    if k in MULT and not v.has_mult:
        val += 2
    if k in XMULT and v.has_xmult and k not in SCALERS:
        val -= 2                                          # one xMult is enough
    if k in SCALERS and not v.has_scaler and v.ante >= 2:
        val += 3
    if k in SLOW_SCALERS and v.ante >= 5:
        val -= 4
    if k in NARROW and v.ante >= 3:
        val -= 1
    if k in CHIPS and v.main_level >= 6:
        val -= 3                                          # levels supply chips by then
    if k == "hanging_chad" or k == "photograph":
        if any(x.key in ("photograph", "hanging_chad") for x in v.jokers if x is not j):
            val += 3
    val += {"": 0, "FOIL": 2, "HOLO": 3, "POLYCHROME": 5, "NEGATIVE": 7}.get(j.edition, 0)
    if j.rental and j.eternal:
        return -50.0
    if j.rental:
        val -= 4 if (v.rentals == 0 and v.ante <= 2 and scoring) else 9
    if j.eternal:
        if k in CHIPS:
            val -= 4
        elif k in GENERIC_EARLY or (k in ECON and v.eternal_econ == 0):
            val -= 1
        else:
            val -= 3
    if j.perishable is not None:
        val -= 2 if v.ante <= 2 else 5
    if j.debuffed:
        val -= 3
    return val


def spend_weight(v: View, cost: int, scoring: bool) -> float:
    """Dollars given up per dollar spent: more than one when it breaks the interest line (3)."""
    if cost <= 0:
        return 0.0
    if not v.safe and scoring:
        return 0.7
    if v.money - cost >= v.reserve:
        return 1.0
    return 1.6 if v.ante <= 5 else 1.2


# ------------------------------------------------------------------ decisions
def shop_choice(v: View, cands, w, pricer, rng) -> int | None:
    g = v.g
    by_kind = {}
    for i, c in enumerate(cands):
        by_kind.setdefault(c.kind, []).append(i)
    # 7. sells first: spent jokers
    for i in by_kind.get("sell_joker", []):
        j = g.jokers[cands[i].action.idx]
        val = j.state.get("val")
        spent = (j.key == "popcorn" and isinstance(val, (int, float)) and val <= 4) or \
                (j.key == "ice_cream" and isinstance(val, (int, float)) and val <= 20) or \
                (j.perishable is not None and j.perishable <= 0) or \
                (j.key in ("egg", "gift", "hallucination") and v.ante >= 6 and not v.safe)
        if spent and len(v.jokers) > 1:
            return i
    # 4. vouchers first
    best, best_i = 0.0, None
    for i in by_kind.get("voucher", []):
        it = g.shop_vouchers[cands[i].action.idx]
        key = it.key[2:]
        val = VOUCHERS.get(key, 2)
        if key == "telescope" and v.main_level >= 3:
            val += 3
        if key in ("hieroglyph", "petroglyph") and not v.safe and v.scoring:
            val = 8
        if not v.scoring and v.ante <= 2:
            val -= 6                                      # no vouchers before a scoring joker
        net = val - it.cost * spend_weight(v, it.cost, False)
        if net > best:
            best, best_i = net, i
    if best_i is not None:
        return best_i
    # 9. consumables that pay now
    for i in by_kind.get("use", []):
        a = cands[i].action
        if a.cards:
            continue
        c = g.consumables[a.idx]
        n = c.name
        if c.kind == "planet" and PLANETS.get(n) == v.main:
            return i
        if n == "hermit" and (v.money >= 15 or v.cons_free == 0):
            return i
        if n == "temperance":
            pay = min(50, sum(j.sell_value() for j in g.jokers))
            if pay >= 6 or v.cons_free == 0:
                return i
        if n == "fool" and g.last_consumable is not None and g.last_consumable.name in HOLD_TAROTS - {"fool"} and v.cons_free >= 1:
            return i
        if n == "judgement" and v.slots_free >= 1:
            return i
        if n in ("emperor", "high_priestess") and v.cons_free >= 1:
            return i
        if c.kind == "planet" and v.cons_free == 0 and PLANETS.get(n) != v.main and v.ante <= 2:
            return i
    # 5. packs before jokers
    best, best_i = 0.5, None
    for i in by_kind.get("buy_pack", []):
        it = g.shop_packs[cands[i].action.idx]
        kind, size = it.pack
        cost = it.cost
        left = v.money - cost
        val = -1.0
        if kind == "buffoon":
            if v.slots_free >= 1 and (not v.scoring or left >= 15 or not v.safe):
                val = 7 if v.ante <= 2 else (5 if not v.safe else 3)
        elif kind == "arcana":
            gate = 15 if ("fool" in v.holding or any(j.key == "golden" for j in v.jokers)) else 19
            if left >= gate and v.cons_free >= 1:
                val = 6
        elif kind == "standard":
            if left >= 20 and v.ante <= 5:
                val = 5
        elif kind == "celestial":
            need = (not v.safe) or (v.ante <= 2 and v.main_level == 1)
            if (need and left >= 3) or left >= 25 or any(j.key in ("constellation", "astronomer") for j in v.jokers):
                val = 6 if need else 3
        elif kind == "spectral":
            if v.safe and left >= 25 and v.slots_free >= 1:
                val = 5
        if val < 0:
            continue
        val += {"normal": 0, "jumbo": 1, "mega": 2}.get(size, 0)
        net = val - cost * spend_weight(v, cost, kind in ("buffoon", "celestial"))
        if net > best:
            best, best_i = net, i
    if best_i is not None:
        return best_i
    # 2/3. jokers
    worst = min(v.jokers, key=lambda j: joker_value(v, j, False), default=None)
    worst_val = joker_value(v, worst, False) if worst is not None else None
    best, best_i, want_sell = 1.0, None, None
    for i in by_kind.get("buy", []):
        it = g.shop[cands[i].action.idx]
        if it.kind == "joker" and it.joker is not None:
            jv = joker_value(v, it.joker)
            net = jv - it.cost * spend_weight(v, it.cost, it.joker.key in CHIPS or it.joker.key in MULT or it.joker.key in XMULT)
            if v.slots_free <= 0:
                if worst is None or worst.eternal or jv - worst_val < 3:
                    continue
                net -= 1
            if net > best:
                best, best_i = net, i
                want_sell = (v.slots_free <= 0)
        elif it.kind == "planet":
            n = it.key[2:]
            val = 5 if PLANETS.get(n) == v.main else (1 if v.ante <= 2 else -1)
            net = val - it.cost * spend_weight(v, it.cost, not v.safe)
            if v.cons_free >= 1 and net > best:
                best, best_i, want_sell = net, i, False
        elif it.kind in ("tarot", "spectral"):
            n = it.key[2:]
            val = (TAROT_PICK if it.kind == "tarot" else SPECTRAL_PICK).get(n, 2)
            if n == "hermit" and v.money < 10:
                val -= 4
            if v.ante <= 1 and not v.scoring:
                val -= 4
            net = val - 1 - it.cost * spend_weight(v, it.cost, False)
            if v.cons_free >= 1 and net > best:
                best, best_i, want_sell = net, i, False
        elif it.kind == "card" and it.card is not None:
            c = it.card
            val = 6 if c.seal == "BLUE" else 4 if c.seal in ("GOLD", "PURPLE") else 4 if c.enh == "GOLD" else \
                3 if c.enh == "STEEL" else 1 if (c.rank in v.top_ranks and v.main == PAIR) else -1
            net = val - it.cost * spend_weight(v, it.cost, False)
            if net > best:
                best, best_i, want_sell = net, i, False
    if best_i is not None:
        if want_sell and worst is not None:
            for i in by_kind.get("sell_joker", []):
                if g.jokers[cands[i].action.idx] is worst:
                    return i
        else:
            return best_i
    # 6. reroll
    for i in by_kind.get("reroll", []):
        cost = 0 if g.free_rerolls > 0 else g.reroll_cost
        rich = v.money >= 60
        ok = (cost <= 7 or (rich and cost <= 15)) and v.money - cost >= v.reserve and v.rentals < 2 \
            and w.rerolls < (3 if rich else 2) and not (v.holding & {"death", "fool"})
        if cost == 0 or (ok and (v.ante <= 6)):
            return i
    # 7. spare consumables that cross an interest step
    if v.money < 25:
        for i in by_kind.get("sell_cons", []):
            c = g.consumables[cands[i].action.idx]
            if c.name in HOLD_TAROTS or (c.kind == "planet" and PLANETS.get(c.name) == v.main):
                continue
            if (v.money + c.sell_value()) // 5 > v.money // 5:
                return i
    return by_kind.get("leave", [None])[0]


def pack_choice(v: View, cands, w, pricer, rng) -> int | None:
    g = v.g
    kind = g.pack_kind
    scored = {}
    for i, c in enumerate(cands):
        if c.kind != "pick":
            continue
        x = g.pack_cards[c.action.idx]
        val = -1.0
        if isinstance(x, Joker):
            if v.slots_free >= 1:
                val = joker_value(v, x) - 1
        elif isinstance(x, Consumable):
            n = x.name
            if x.kind == "planet":
                val = 7 if PLANETS.get(n) == v.main else (2 if v.ante <= 2 else 0)
                if PLANETS.get(n) in (9, 10, 11) and g.hand_played[PLANETS[n]] == 0:
                    val = -1
            elif x.kind == "tarot":
                val = TAROT_PICK.get(n, 2)
                if n == "hermit":
                    val = 8 if v.money >= 10 else 4
                if n == "fool":
                    val = 7 if (g.last_consumable is not None and g.last_consumable.name in HOLD_TAROTS - {"fool"}) else 2
                if n == "death" and not any(c2.seal or c2.enh in ("STEEL", "GOLD", "GLASS") for c2 in g.full_deck):
                    val = 3
                if n in ("moon", "sun", "star", "world") and v.main == FLUSH:
                    val = 5
                if n == "judgement" and v.slots_free <= 0:
                    val = 1
            else:
                val = SPECTRAL_PICK.get(n, 2)
                if n == "hex" and not any(not (j.rental or j.eternal or j.perishable is not None) and j.key in (CHIPS | MULT | XMULT) for j in v.jokers):
                    val = 2
                if n == "wraith" and v.slots_free <= 0:
                    val = 1
                if n == "immolate" and v.deck_n < 40:
                    val = 0
            if not c.action.cards and v.cons_free <= 0 and n not in ("hermit", "temperance", "judgement", "fool"):
                val = -1                                  # no room to keep it
        else:                                             # a playing card
            cd = x
            val = 9 if cd.seal == "BLUE" else 7 if cd.seal == "GOLD" else 6 if cd.seal == "PURPLE" else \
                6 if cd.enh == "GOLD" else 5 if cd.enh == "STEEL" else 3 if cd.enh == "GLASS" else \
                2 if cd.seal == "RED" else (1 if (cd.rank in v.top_ranks or (v.main == FLUSH and cd.suit == v.top_suit)) else -1)
            if v.deck_n >= 52 + 4 and val < 5:
                val = -1                                  # cap plain additions (16)
        scored[i] = val
    if not scored:
        return next((i for i, c in enumerate(cands) if c.kind == "pack_skip"), None)
    top = max(scored.values())
    if top < 1.0:
        return next((i for i, c in enumerate(cands) if c.kind == "pack_skip"), None)
    tied = [i for i, s in scored.items() if s >= top - 1e-9]
    if len(tied) == 1:
        return tied[0]
    prices = pricer.prices(w, _sub(cands, tied), rng, root=False).sum(axis=1)     # targets: the pricer decides
    return tied[int(np.argmax(prices))]


class _Sub:
    def __init__(self, cands):
        self.cands = cands


def _sub(cands, idx):
    return _Sub([cands[i] for i in idx])


def blind_choice(v: View, cands) -> int | None:
    g = v.g
    kinds = {c.kind: i for i, c in enumerate(cands)}
    if g.blind_idx == 2 and "reroll_boss" in kinds and g.boss in KILLERS and v.money >= 10 and not v.safe:
        return kinds["reroll_boss"]
    if "skip" in kinds and g.blind_idx < 2:
        tag = g.tags_offered[g.blind_idx]
        survive = v.p_boss >= 0.5 and v.p_next >= 0.7
        if tag == "investment" and v.ante <= 2 and g.blind_idx == 1 and survive and len(v.econ) < 2 \
                and "investment" not in g.pending_tags and not (v.stake >= 3 and not v.scoring):
            return kinds["skip"]
        if tag == "top_up" and v.ante <= 2 and not v.jokers and v.p_next >= 0.8 and v.slots_free >= 2:
            return kinds["skip"]
        if tag == "orbital" and survive and v.ante <= 5 and v.main_level <= 4:
            return kinds["skip"]
    return kinds.get("select")


def guide_choice(w, choice, pricer, rng: random.Random) -> int | None:
    v = View(w, pricer, rng)
    st = w.g.state
    if st == "SHOP":
        return shop_choice(v, choice.cands, w, pricer, rng)
    if st == "PACK":
        return pack_choice(v, choice.cands, w, pricer, rng)
    if st == "BLIND_SELECT":
        return blind_choice(v, choice.cands)
    return None


def guide_prior(w, choice, bonus: float, pricer, rng: random.Random) -> np.ndarray:
    pri = np.zeros(len(choice.cands))
    try:
        i = guide_choice(w, choice, pricer, rng)
    except Exception:
        i = None
    if i is not None:
        pri[i] = bonus
    return pri
