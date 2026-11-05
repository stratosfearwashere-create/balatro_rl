"""An exact numeric description of every joker, read off the engine's own definitions (sim/jokers.py and the
growth code in sim/game.py), as network input. Where the engine and the real game differ, this follows the ENGINE.

    SPEC[key] -> dict of the non-default fields below (a field left out is 0 / -1 / "none" as FIELDS says)
    vectorise() -> float32 [len(VOCAB), len(FEATURE_NAMES)]  (rows of items that are not jokers are zero)

Schema (FIELDS gives the defaults):

  The effect (per trigger, at counter zero; a scaler's counter then adds to it, see counter_unit)
    chips, mult         flat bonus.  With scales_with != none: the bonus per `unit` of that quantity.
    xmult               multiplier MINUS 1 (0 = none), same convention.
    chance              probability the effect happens (0 = it is certain); Bloodstone 0.5, Misprint's mult is its mean.
    unit                the divisor of the scaled quantity (Bootstraps: +2 mult per 5 dollars -> unit 5)
    trigs               the events where it acts (any of TRIGGERS):
        hand (the joker step of a played hand, including "before" hooks), scored_card, held_card, retrigger,
        discard, end_round, blind_select, on_sell, shop, pack, passive (a rule or room change, always on), boss
  Condition
    retriggers          extra triggers it gives the cards it applies to (Hanging Chad 2, Hack 1)
    hand                the hand type that must be contained (index into HAND_NAMES, -1 none)
    suit                suit index (0 S, 1 H, 2 C, 3 D), -1 none
    rank_class          none / face / even / odd / ace / specific / set          rank_mask: ranks 2..14 as 13 bits
    max_cards / exact_cards   hand may hold at most / exactly this many cards played (0 none)
    discards_zero, last_hand, first_hand, first_discard, repeat_hand   situation flags
    needs_enh           needs enhanced cards (Vampire eats them, Driver's License counts them)
    threshold           a count that must be reached (Loyalty Card: every 6th hand, Driver's License 16 enhanced
                        cards, Faceless 3 faces, Invisible 2 rounds)
    target_state        the joker picks a new random target (rank / suit / hand) every round
  Growth
    scales              1 if the joker has a counter that moves during the run
    counter_unit        what the counter is: chips / mult / xmult / money / count / hand_size
    start_value         the counter's value when the joker is made (xmult counters start at 1.0)
    grow_amount, grow_event, grow_per    counter gain per event (per `grow_per` units of the event, Yorick 23 cards)
    decay_amount, decay_event            counter loss per event (the joker runs down)
    reset_event         the counter goes back to its start on this event
    expires, expire_at, expire_prob      destroyed when the counter reaches expire_at / by chance each round
  Economy
    income_round, income_other   dollars at the end of the round / on some other event (per unit, see scales_with)
    interest, debt, free_rerolls, free_planets, sell_gain, sell_all_gain
  Room
    d_hands, d_discards, d_hand_size, d_joker_slots, d_consumable_slots, zero_discards
  Creation / destruction
    creates_tarot/planet/spectral/joker/card/tag, levels_hand, destroys_card, destroys_joker, copies_joker,
    copies_consumable, boss_effect, saves_run
  Dependence
    scales_with         one of SCALES_WITH; uses_deck / uses_jokers / uses_money / uses_consumables / uses_deck_count
                        are derived from it
  rule                 passive rule changing the game (RULES)
  role                 position role as env.swap_role (3 neighbour, 2 multiplier, 1 additive mult, 0 none)
"""
from __future__ import annotations

import math

import numpy as np

from .az.guide import MULT, XMULT
from .sim.hands import HAND_NAMES, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, QUADS
from .sim.items import VOCAB
from .sim.jokers import JOKERS

TRIGGERS = ("hand", "scored_card", "held_card", "retrigger", "discard", "end_round", "blind_select", "on_sell",
            "shop", "pack", "passive", "boss")
GROW_EVENTS = ("none", "reroll", "planet_used", "hand_played", "hand_of_type_played", "card_scored", "discard",
               "card_discarded", "card_sold", "joker_sold", "round_end", "blind_skipped", "boss_defeated",
               "consumable_used", "money_spent", "card_destroyed", "card_added", "shop_item_bought", "per_ante",
               "blind_select", "pack_skipped", "lucky_trigger", "other")
SCALES_WITH = ("none", "jokers_held", "money", "deck_size", "discards_left", "hands_left", "hand_level",
               "planets_used", "tarots_used", "cards_of_suit", "cards_of_rank", "enhanced_cards", "deck_missing",
               "times_hand_played", "blinds_skipped", "empty_slots", "joker_sell_value", "stakes")
COUNTER_UNITS = ("none", "chips", "mult", "xmult", "money", "count", "hand_size")
RANK_CLASSES = ("none", "face", "even", "odd", "ace", "specific", "set")
RULES = ("none", "all_faces", "four_fingers", "shortcut", "smeared", "splash", "double_odds", "duplicates",
         "held_retrigger", "uncommon_x", "copy_right", "copy_left", "no_boss", "free_reroll", "debt")

FIELDS: dict = dict(
    chips=0.0, mult=0.0, xmult=0.0, chance=0.0, unit=1.0, trigs=(), retriggers=0,
    hand=-1, suit=-1, rank_class="none", rank_mask=(), max_cards=0, exact_cards=0,
    discards_zero=0, last_hand=0, first_hand=0, first_discard=0, repeat_hand=0, needs_enh=0, threshold=0.0,
    target_state=0,
    scales=0, counter_unit="none", start_value=0.0, grow_amount=0.0, grow_event="none", grow_per=1.0,
    decay_amount=0.0, decay_event="none", reset_event="none", expires=0, expire_at=0.0, expire_prob=0.0,
    income_round=0.0, income_other=0.0, interest=0, debt=0.0, free_rerolls=0, free_planets=0, sell_gain=0.0,
    sell_all_gain=0.0,
    d_hands=0, d_discards=0, d_hand_size=0, d_joker_slots=0, d_consumable_slots=0, zero_discards=0,
    creates_tarot=0, creates_planet=0, creates_spectral=0, creates_joker=0, creates_card=0, creates_tag=0,
    levels_hand=0, destroys_card=0, destroys_joker=0, copies_joker=0, copies_consumable=0, boss_effect=0,
    saves_run=0,
    scales_with="none", rule="none",
)
_ENUMS = dict(grow_event=GROW_EVENTS, decay_event=GROW_EVENTS, reset_event=GROW_EVENTS, scales_with=SCALES_WITH,
              counter_unit=COUNTER_UNITS, rank_class=RANK_CLASSES, rule=RULES)

EVEN = (2, 4, 6, 8, 10)
ODD = (3, 5, 7, 9, 14)
FACE = (11, 12, 13)

SPEC: dict[str, dict] = {}


def S(key, **kw):
    bad = set(kw) - set(FIELDS)
    if bad:
        raise KeyError(f"joker_spec {key}: unknown fields {sorted(bad)}")
    for f, vals in _ENUMS.items():
        if f in kw and kw[f] not in vals:
            raise ValueError(f"joker_spec {key}: {f}={kw[f]!r}")
    if not set(kw.get("trigs", ())) <= set(TRIGGERS):
        raise ValueError(f"joker_spec {key}: trigs {kw['trigs']}")
    SPEC[key] = kw


def suit_mult(key, suit, **kw):
    S(key, trigs=("scored_card",), suit=suit, **kw)


# ---------------------------------------------------------------- flat jokers
S("joker", mult=4, trigs=("hand",))
suit_mult("greedy_joker", 3, mult=3)                 # Diamonds
suit_mult("lusty_joker", 1, mult=3)                  # Hearts
suit_mult("wrathful_joker", 0, mult=3)               # Spades
suit_mult("gluttenous_joker", 2, mult=3)             # Clubs
for _k, _h, _m, _c in (("jolly", PAIR, 8, 0), ("zany", TRIPS, 12, 0), ("mad", TWO_PAIR, 10, 0),
                       ("crazy", STRAIGHT, 12, 0), ("droll", FLUSH, 10, 0), ("sly", PAIR, 0, 50),
                       ("wily", TRIPS, 0, 100), ("clever", TWO_PAIR, 0, 80), ("devious", STRAIGHT, 0, 100),
                       ("crafty", FLUSH, 0, 80)):
    S(_k, trigs=("hand",), hand=_h, mult=_m, chips=_c)
for _k, _h, _x in (("duo", PAIR, 2), ("trio", TRIPS, 3), ("family", QUADS, 4), ("order", STRAIGHT, 3),
                   ("tribe", FLUSH, 2)):
    S(_k, trigs=("hand",), hand=_h, xmult=_x - 1)
S("half", trigs=("hand",), mult=20, max_cards=3)
S("banner", trigs=("hand",), chips=30, scales_with="discards_left")
S("mystic_summit", trigs=("hand",), mult=15, discards_zero=1)
S("misprint", trigs=("hand",), mult=11.5)           # the mean of 0..23
S("raised_fist", trigs=("held_card",), rank_class="set", mult=2)   # 2 x the lowest held card's chips
S("scary_face", trigs=("scored_card",), rank_class="face", rank_mask=FACE, chips=30)
S("abstract", trigs=("hand",), mult=3, scales_with="jokers_held")
S("even_steven", trigs=("scored_card",), rank_class="even", rank_mask=EVEN, mult=4)
S("odd_todd", trigs=("scored_card",), rank_class="odd", rank_mask=ODD, chips=31)
S("scholar", trigs=("scored_card",), rank_class="ace", rank_mask=(14,), chips=20, mult=4)
S("supernova", trigs=("hand",), mult=1, scales_with="times_hand_played")     # + the count of this hand type played
S("blue_joker", trigs=("hand",), chips=2, scales_with="deck_size")
S("walkie_talkie", trigs=("scored_card",), rank_class="set", rank_mask=(10, 4), chips=10, mult=4)
S("smiley", trigs=("scored_card",), rank_class="face", rank_mask=FACE, mult=5)
S("shoot_the_moon", trigs=("held_card",), rank_class="specific", rank_mask=(12,), mult=13)
S("photograph", trigs=("scored_card",), rank_class="face", rank_mask=FACE, xmult=1)     # the first face card only
S("fibonacci", trigs=("scored_card",), rank_class="set", rank_mask=(14, 2, 3, 5, 8), mult=8)
S("fortune_teller", trigs=("hand",), mult=1, scales_with="tarots_used")
S("bull", trigs=("hand",), chips=2, scales_with="money")
S("bootstraps", trigs=("hand",), mult=2, unit=5, scales_with="money")
S("card_sharp", trigs=("hand",), xmult=2, repeat_hand=1)         # x3 when this hand type was already played this round
S("acrobat", trigs=("hand",), xmult=2, last_hand=1)
S("arrowhead", trigs=("scored_card",), suit=0, chips=50)
S("onyx_agate", trigs=("scored_card",), suit=2, mult=7)
S("bloodstone", trigs=("scored_card",), suit=1, xmult=0.5, chance=0.5)
S("rough_gem", trigs=("scored_card",), suit=3, income_other=1)
S("blackboard", trigs=("hand",), xmult=2)                       # every held card Spades or Clubs
S("flower_pot", trigs=("hand",), xmult=2)                       # all four suits among the scoring cards
S("seeing_double", trigs=("hand",), xmult=1)                    # a Club and a card of another suit
S("stencil", trigs=("hand",), xmult=1, scales_with="empty_slots")     # x(1 + empty joker slots)
S("baron", trigs=("held_card",), rank_class="specific", rank_mask=(13,), xmult=0.5)
S("ancient", trigs=("scored_card", "end_round"), xmult=0.5, target_state=1)    # a new suit each round
S("idol", trigs=("scored_card", "end_round"), xmult=1, target_state=1)         # a new rank and suit each round
S("triboulet", trigs=("scored_card",), rank_class="set", rank_mask=(12, 13), xmult=1)
S("erosion", trigs=("hand",), mult=4, scales_with="deck_missing")
S("stone", trigs=("hand",), chips=25, scales_with="enhanced_cards", needs_enh=1)    # per Stone card
S("steel_joker", trigs=("hand",), xmult=0.2, scales_with="enhanced_cards", needs_enh=1)   # per Steel card
S("swashbuckler", trigs=("hand",), mult=1, scales_with="joker_sell_value")
S("throwback", trigs=("hand",), xmult=0.25, scales_with="blinds_skipped")
S("stuntman", trigs=("hand", "passive"), chips=250, d_hand_size=-2)
S("drivers_license", trigs=("hand",), xmult=2, threshold=16, needs_enh=1)
S("gros_michel", trigs=("hand", "end_round"), mult=15, expires=1, expire_prob=1 / 6)
S("cavendish", trigs=("hand", "end_round"), xmult=2, expires=1, expire_prob=1 / 1000)

# ---------------------------------------------------------------- retriggers
S("hanging_chad", trigs=("retrigger",), retriggers=2)                    # the first scoring card, 2 more times
S("hack", trigs=("retrigger",), retriggers=1, rank_class="set", rank_mask=(2, 3, 4, 5))
S("dusk", trigs=("retrigger",), retriggers=1, last_hand=1)
S("sock_and_buskin", trigs=("retrigger",), retriggers=1, rank_class="face", rank_mask=FACE)
S("selzer", trigs=("retrigger", "hand"), retriggers=1, scales=1, counter_unit="count", start_value=10, decay_amount=1,
  decay_event="hand_played", expires=1, expire_at=0)
S("mime", trigs=("passive",), retriggers=1, rule="held_retrigger")

# ---------------------------------------------------------------- scalers (counter in the joker)
S("ride_the_bus", trigs=("hand",), scales=1, counter_unit="mult", grow_amount=1, grow_event="hand_played",
  reset_event="hand_played")                                # +1 per hand without a scoring face card, back to 0 on one
S("green_joker", trigs=("hand", "discard"), scales=1, counter_unit="mult", grow_amount=1, grow_event="hand_played",
  decay_amount=1, decay_event="discard")
S("ice_cream", trigs=("hand",), scales=1, counter_unit="chips", start_value=100, decay_amount=5,
  decay_event="hand_played", expires=1, expire_at=0)
S("popcorn", trigs=("hand", "end_round"), scales=1, counter_unit="mult", start_value=20, decay_amount=4,
  decay_event="round_end", expires=1, expire_at=0)
S("ramen", trigs=("hand", "discard"), scales=1, counter_unit="xmult", start_value=1.0 + 1.0, decay_amount=0.01,
  decay_event="card_discarded", expires=1, expire_at=1.0)
S("runner", trigs=("hand",), scales=1, counter_unit="chips", grow_amount=15, grow_event="hand_of_type_played",
  hand=STRAIGHT)
S("square", trigs=("hand",), scales=1, counter_unit="chips", grow_amount=4, grow_event="hand_played", exact_cards=4)
S("trousers", trigs=("hand",), scales=1, counter_unit="mult", grow_amount=2, grow_event="hand_of_type_played",
  hand=TWO_PAIR)
S("flash", trigs=("hand", "shop"), scales=1, counter_unit="mult", grow_amount=2, grow_event="reroll")
S("constellation", trigs=("hand",), scales=1, counter_unit="xmult", start_value=1.0, grow_amount=0.1,
  grow_event="planet_used")
S("hologram", trigs=("hand",), scales=1, counter_unit="xmult", start_value=1.0, grow_amount=0.25,
  grow_event="card_added")
S("vampire", trigs=("hand",), scales=1, counter_unit="xmult", start_value=1.0, grow_amount=0.1,
  grow_event="card_scored", needs_enh=1, destroys_card=0)    # per scored enhanced card, whose enhancement it eats
S("obelisk", trigs=("hand",), scales=1, counter_unit="xmult", start_value=1.0, grow_amount=0.2,
  grow_event="hand_played", reset_event="hand_played")       # unless this is the most played hand type: then reset
S("madness", trigs=("hand", "blind_select"), scales=1, counter_unit="xmult", start_value=1.0, grow_amount=0.5,
  grow_event="blind_select", destroys_joker=1)
S("lucky_cat", trigs=("hand", "scored_card"), scales=1, counter_unit="xmult", start_value=1.0, grow_amount=0.25,
  grow_event="lucky_trigger")
S("glass", trigs=("hand",), scales=1, counter_unit="xmult", start_value=1.0, grow_amount=0.75,
  grow_event="card_destroyed")                               # per Glass card destroyed
S("caino", trigs=("hand",), scales=1, counter_unit="xmult", start_value=1.0, grow_amount=1.0,
  grow_event="card_destroyed")                               # per face card destroyed
S("campfire", trigs=("hand",), scales=1, counter_unit="xmult", start_value=1.0, grow_amount=0.25,
  grow_event="card_sold", reset_event="boss_defeated")
S("ceremonial", trigs=("hand", "blind_select"), scales=1, counter_unit="mult", grow_amount=2,
  grow_event="blind_select", destroys_joker=1, scales_with="joker_sell_value")    # +2 x the sell value of the joker to its right
S("red_card", trigs=("hand", "pack"), scales=1, counter_unit="mult", grow_amount=3, grow_event="pack_skipped")
S("wee", trigs=("hand", "scored_card"), scales=1, counter_unit="chips", grow_amount=8, grow_event="card_scored",
  rank_class="specific", rank_mask=(2,))
S("castle", trigs=("hand", "discard", "end_round"), scales=1, counter_unit="chips", grow_amount=3,
  grow_event="card_discarded", target_state=1)               # per discarded card of the round's suit
S("hit_the_road", trigs=("hand", "discard", "end_round"), scales=1, counter_unit="xmult", start_value=1.0,
  grow_amount=0.5, grow_event="card_discarded", rank_class="specific", rank_mask=(11,), reset_event="round_end")
S("yorick", trigs=("hand", "discard"), scales=1, counter_unit="xmult", start_value=1.0, grow_amount=1.0,
  grow_event="card_discarded", grow_per=23)
S("hiker", trigs=("scored_card",), scales=1, counter_unit="chips", grow_amount=5, grow_event="card_scored")   # lives on the cards
S("rocket", trigs=("end_round",), scales=1, counter_unit="money", start_value=1, grow_amount=2,
  grow_event="boss_defeated", income_round=1)
S("egg", trigs=("end_round",), sell_gain=3, scales=1, counter_unit="money", grow_amount=3, grow_event="round_end")
S("gift", trigs=("end_round",), sell_all_gain=1, scales=1, counter_unit="money", grow_amount=1, grow_event="round_end")
S("invisible", trigs=("end_round", "on_sell"), scales=1, counter_unit="count", grow_amount=1, grow_event="round_end",
  threshold=2, copies_joker=1)
S("turtle_bean", trigs=("passive", "end_round"), scales=1, counter_unit="hand_size", start_value=5, d_hand_size=5,
  decay_amount=1, decay_event="round_end", expires=1, expire_at=0)
S("loyalty_card", trigs=("hand",), xmult=3, threshold=6, scales=1, counter_unit="count", start_value=6,
  decay_amount=1, decay_event="hand_played")                # x4 on every 6th hand played
S("satellite", trigs=("end_round",), income_round=1, scales_with="planets_used")
S("cloud_9", trigs=("end_round",), income_round=1, scales_with="cards_of_rank", rank_class="specific", rank_mask=(9,))

# ---------------------------------------------------------------- economy
S("golden", trigs=("end_round",), income_round=4)
S("delayed_grat", trigs=("end_round",), income_round=2, scales_with="discards_left", discards_zero=0)
S("to_the_moon", trigs=("passive", "end_round"), interest=1)
S("credit_card", trigs=("passive",), debt=20, rule="debt")
S("business", trigs=("scored_card",), rank_class="face", rank_mask=FACE, income_other=2, chance=0.5)
S("faceless", trigs=("discard",), rank_class="face", rank_mask=FACE, income_other=5, threshold=3)
S("reserved_parking", trigs=("held_card",), rank_class="face", rank_mask=FACE, income_other=1, chance=0.5)
S("todo_list", trigs=("hand", "end_round"), income_other=4, target_state=1)    # when the target hand type is played
S("mail", trigs=("discard", "end_round"), rank_class="specific", income_other=5, target_state=1)   # per discarded card of the rank
S("trading", trigs=("discard",), income_other=3, first_discard=1, destroys_card=1)
S("ticket", trigs=("scored_card",), income_other=4, needs_enh=1)    # per scored Gold card
S("matador", trigs=("boss",), income_other=8)
S("chaos", trigs=("passive", "shop"), free_rerolls=1, rule="free_reroll")
S("astronomer", trigs=("passive", "shop"), free_planets=1)
S("diet_cola", trigs=("on_sell",), creates_tag=1)
S("luchador", trigs=("on_sell",), boss_effect=1)
S("chicot", trigs=("passive", "boss"), boss_effect=1, rule="no_boss")
S("mr_bones", trigs=("passive",), saves_run=1)

# ---------------------------------------------------------------- room
S("drunkard", trigs=("passive",), d_discards=1)
S("juggler", trigs=("passive",), d_hand_size=1)
S("troubadour", trigs=("passive",), d_hand_size=2, d_hands=-1)
S("merry_andy", trigs=("passive",), d_discards=3, d_hand_size=-1)
S("burglar", trigs=("blind_select",), d_hands=3, zero_discards=1)

# ---------------------------------------------------------------- creation
S("8_ball", trigs=("scored_card",), rank_class="specific", rank_mask=(8,), creates_tarot=1, chance=0.25)
S("sixth_sense", trigs=("hand",), rank_class="specific", rank_mask=(6,), exact_cards=1, first_hand=1,
  creates_spectral=1, destroys_card=1)
S("superposition", trigs=("hand",), hand=STRAIGHT, rank_class="ace", rank_mask=(14,), creates_tarot=1)
S("seance", trigs=("hand",), creates_spectral=1)           # on a Straight Flush
S("vagabond", trigs=("hand",), creates_tarot=1)            # when money <= 4
S("hallucination", trigs=("pack",), creates_tarot=1, chance=0.5)
S("cartomancer", trigs=("blind_select",), creates_tarot=1)
S("riff_raff", trigs=("blind_select",), creates_joker=2)
S("marble", trigs=("blind_select",), creates_card=1)       # a Stone card into the deck
S("certificate", trigs=("blind_select",), creates_card=1)  # a card with a random seal into the hand
S("dna", trigs=("hand",), exact_cards=1, first_hand=1, creates_card=1)
S("perkeo", trigs=("shop",), copies_consumable=1)
S("space", trigs=("hand",), levels_hand=1, chance=0.25)
S("burnt", trigs=("discard",), levels_hand=1, first_discard=1)
S("midas_mask", trigs=("hand",), rank_class="face", rank_mask=FACE)       # scoring faces turn Gold

# ---------------------------------------------------------------- rules
S("pareidolia", trigs=("passive",), rule="all_faces")
S("four_fingers", trigs=("passive",), rule="four_fingers")
S("shortcut", trigs=("passive",), rule="shortcut")
S("smeared", trigs=("passive",), rule="smeared")
S("splash", trigs=("passive",), rule="splash")
S("oops", trigs=("passive",), rule="double_odds")
S("ring_master", trigs=("passive",), rule="duplicates")
S("baseball", trigs=("passive", "hand"), xmult=0.5, rule="uncommon_x")      # x1.5 per Uncommon joker
S("blueprint", trigs=("hand", "scored_card", "held_card", "retrigger"), copies_joker=1, rule="copy_right")
S("brainstorm", trigs=("hand", "scored_card", "held_card", "retrigger"), copies_joker=1, rule="copy_left")

def full(key: str) -> dict:
    """The complete record of a joker: SPEC's fields over the defaults."""
    return {**FIELDS, **SPEC[key]}


def role(key: str) -> int:
    """env.swap_role: 3 acts on a neighbour, 2 multiplier, 1 additive mult, 0 none."""
    return 3 if key in ("blueprint", "brainstorm", "ceremonial") else 2 if key in XMULT else 1 if key in MULT else 0


def _check():
    miss = sorted(set(JOKERS) - set(SPEC))
    extra = sorted(set(SPEC) - set(JOKERS))
    if miss or extra:
        raise KeyError(f"joker_spec: missing {miss}, unknown {extra}")


# ---------------------------------------------------------------- the vector
_NUM = ("chips", "mult", "xmult", "chance", "unit", "retriggers", "max_cards", "exact_cards", "threshold", "start_value",
        "grow_amount", "grow_per", "decay_amount", "expire_at", "expire_prob", "income_round", "income_other",
        "debt", "sell_gain", "sell_all_gain", "d_hands", "d_discards", "d_hand_size", "d_joker_slots",
        "d_consumable_slots", "creates_joker", "creates_card")
_FLAG = ("scales", "discards_zero", "last_hand", "first_hand", "first_discard", "repeat_hand", "needs_enh",
         "target_state", "expires", "interest", "free_rerolls", "free_planets", "zero_discards", "creates_tarot",
         "creates_planet", "creates_spectral", "creates_tag", "levels_hand", "destroys_card", "destroys_joker",
         "copies_joker", "copies_consumable", "boss_effect", "saves_run")
_DERIVED = ("uses_deck", "uses_jokers", "uses_money", "uses_consumables", "uses_deck_count")
_HANDS = len(HAND_NAMES)

FEATURE_NAMES = (["sp_" + n for n in _NUM] + ["sp_" + n for n in _FLAG] + ["sp_" + n for n in _DERIVED]
                 + ["sp_hand_" + h.replace(" ", "_").lower() for h in HAND_NAMES] + ["sp_hand_none"]
                 + [f"sp_suit_{s}" for s in "SHCD"] + ["sp_suit_none"]
                 + [f"sp_rankclass_{r}" for r in RANK_CLASSES] + [f"sp_rank_{r}" for r in range(2, 15)]
                 + [f"sp_trig_{t}" for t in TRIGGERS]
                 + [f"sp_grow_{e}" for e in GROW_EVENTS] + [f"sp_decay_{e}" for e in GROW_EVENTS]
                 + [f"sp_reset_{e}" for e in GROW_EVENTS]
                 + [f"sp_counter_{u}" for u in COUNTER_UNITS] + [f"sp_scales_{s}" for s in SCALES_WITH]
                 + [f"sp_rule_{r}" for r in RULES] + [f"sp_role_{r}" for r in range(4)])


def _unit(v: float, unit: str) -> float:
    """A value in the counter's unit, scaled to about 0..1 (chips by log, the rest by their usual maximum)."""
    if unit == "chips":
        return math.log1p(abs(v)) / math.log1p(300) * (1 if v >= 0 else -1)
    return v / {"mult": 20.0, "xmult": 3.0, "money": 10.0, "count": 10.0, "hand_size": 5.0, "none": 10.0}[unit]


def _row(key: str) -> np.ndarray:
    f = full(key)
    cu = f["counter_unit"]
    v = {}
    v["chips"] = _unit(f["chips"], "chips")
    v["mult"] = f["mult"] / 20.0
    v["xmult"] = f["xmult"] / 3.0
    v["chance"] = f["chance"]
    v["unit"] = math.log(f["unit"]) / math.log(23.0) if f["unit"] > 1 else 0.0
    v["retriggers"] = f["retriggers"] / 2.0
    v["max_cards"] = f["max_cards"] / 5.0
    v["exact_cards"] = f["exact_cards"] / 5.0
    v["threshold"] = f["threshold"] / 16.0
    v["start_value"] = _unit(f["start_value"] - (1.0 if cu == "xmult" else 0.0), cu)
    v["grow_amount"] = _unit(f["grow_amount"], cu)
    v["grow_per"] = math.log(f["grow_per"]) / math.log(23.0) if f["grow_per"] > 1 else 0.0
    v["decay_amount"] = _unit(f["decay_amount"], cu)
    v["expire_at"] = _unit(f["expire_at"] - (1.0 if cu == "xmult" else 0.0), cu)
    v["expire_prob"] = f["expire_prob"]
    v["income_round"] = f["income_round"] / 10.0
    v["income_other"] = f["income_other"] / 10.0
    v["debt"] = f["debt"] / 20.0
    v["sell_gain"] = f["sell_gain"] / 5.0
    v["sell_all_gain"] = f["sell_all_gain"] / 5.0
    v["d_hands"] = f["d_hands"] / 3.0
    v["d_discards"] = f["d_discards"] / 3.0
    v["d_hand_size"] = f["d_hand_size"] / 3.0
    v["d_joker_slots"] = f["d_joker_slots"] / 3.0
    v["d_consumable_slots"] = f["d_consumable_slots"] / 3.0
    v["creates_joker"] = f["creates_joker"] / 2.0
    v["creates_card"] = f["creates_card"] / 2.0
    sw = f["scales_with"]
    d = {"uses_deck": float(sw in ("deck_size", "cards_of_suit", "cards_of_rank", "enhanced_cards", "deck_missing")
                            or f["needs_enh"] == 1 or f["threshold"] == 16),
         "uses_jokers": float(sw in ("jokers_held", "empty_slots", "joker_sell_value") or f["copies_joker"] > 0),
         "uses_money": float(sw == "money" or f["interest"] > 0),
         "uses_consumables": float(sw in ("planets_used", "tarots_used") or f["copies_consumable"] > 0),
         "uses_deck_count": float(sw in ("deck_size", "deck_missing", "enhanced_cards", "cards_of_rank"))}
    out = [v[n] for n in _NUM] + [float(f[n]) for n in _FLAG] + [d[n] for n in _DERIVED]
    hv = [0.0] * (_HANDS + 1)
    hv[f["hand"] if f["hand"] >= 0 else _HANDS] = 1.0
    sv = [0.0] * 5
    sv[f["suit"] if f["suit"] >= 0 else 4] = 1.0
    out += hv + sv
    out += [float(f["rank_class"] == r) for r in RANK_CLASSES]
    out += [float(r in f["rank_mask"]) for r in range(2, 15)]
    out += [float(t in f["trigs"]) for t in TRIGGERS]
    out += [float(f["grow_event"] == e) for e in GROW_EVENTS]
    out += [float(f["decay_event"] == e) for e in GROW_EVENTS]
    out += [float(f["reset_event"] == e) for e in GROW_EVENTS]
    out += [float(cu == u) for u in COUNTER_UNITS]
    out += [float(sw == s) for s in SCALES_WITH]
    out += [float(f["rule"] == r) for r in RULES]
    out += [float(role(key) == r) for r in range(4)]
    return np.asarray(out, dtype=np.float32)


def vectorise() -> np.ndarray:
    """float32 [len(VOCAB), len(FEATURE_NAMES)] in VOCAB order; only the joker rows are non-zero."""
    _check()
    out = np.zeros((len(VOCAB), len(FEATURE_NAMES)), dtype=np.float32)
    for i, key in enumerate(VOCAB):
        if key.startswith("j_") and key[2:] in SPEC:
            out[i] = _row(key[2:])
    return out
