"""joker_spec: every joker is described, the vectors are sane, and the numbers match what the engine does.

Engine-consistency covers (scored with sim.scoring.score_hand, a joker against none):
  hand-type jokers, per-scored-card suit / rank jokers, held-card jokers, the quantity-scaled ones, retriggers,
  and the scalers' counters (reroll, planet, hand played, discard, round end, card added / destroyed / scored ...).
Skipped (need exotic setup; the spec for them is from reading jokers.py): ceremonial, castle, mail, todo_list,
  trading, faceless, space, burnt, vagabond, sixth_sense, superposition, seance, 8_ball, dna, certificate, marble,
  riff_raff, cartomancer, burglar, perkeo, hallucination, luchador, chicot, mr_bones, matador, diet_cola, invisible
  (on_sell copy), rocket / campfire boss resets, loyalty_card, lucky_cat, raised_fist, swashbuckler, flower_pot
  variants with Wild cards, blueprint / brainstorm, baseball, rules (pareidolia, four_fingers, shortcut, smeared,
  splash, oops, ring_master, mime), and the passive room / economy jokers (their numbers are read from game.py).
"""
import random

import numpy as np
import pytest

from balatro_rl import joker_spec as JS
from balatro_rl.joker_desc import FEATURES, item_descriptors, features_spec
from balatro_rl.sim.cards import Card
from balatro_rl.sim.game import Game, Consumable
from balatro_rl.sim.hands import PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, QUADS, HC, HAND_NAMES
from balatro_rl.sim.items import VOCAB
from balatro_rl.sim.jokers import JOKERS, Joker
from balatro_rl.sim.scoring import score_hand


def C(r, s=0, **kw):
    return Card(r, s, **kw)


def game(*keys, seed=1):
    g = Game(seed=seed)
    for k in keys:
        g.add_joker(Joker(k, base_cost=JOKERS[k].cost))
    return g


def sc(g, played, held=()):
    ctx = score_hand(g, list(played), list(held))[1]
    return ctx.chips, ctx.mult, ctx.money


def val(g, key):
    return next(j for j in g.jokers if j.key == key).state.get("val", 0)


# ---------------------------------------------------------------- the table itself
def test_every_joker_has_an_entry():
    assert set(JS.SPEC) == set(JOKERS)


def test_vectors_finite_and_shaped():
    v = JS.vectorise()
    assert v.shape == (len(VOCAB), len(JS.FEATURE_NAMES)) and v.dtype == np.float32
    assert np.isfinite(v).all() and np.abs(v).max() <= 2.0
    assert len(set(JS.FEATURE_NAMES)) == len(JS.FEATURE_NAMES)
    for i, k in enumerate(VOCAB):
        if not k.startswith("j_") or k == "j_unknown":
            assert not v[i].any()
        else:
            assert v[i].any()


def test_jokers_are_distinguishable():
    v = JS.vectorise()
    rows = {k: v[i].tobytes() for i, k in enumerate(VOCAB) if k.startswith("j_")}
    same = {}
    for k, b in rows.items():
        same.setdefault(b, []).append(k)
    # only jokers whose effect lies in a state the spec does not hold may coincide (suit_mult jokers differ by suit)
    dup = [g for g in same.values() if len(g) > 1]
    assert all(set(g) <= {"j_seance", "j_vagabond"} or len(g) == 2 for g in dup), dup


def test_roles_match_env_swap_role():
    from balatro_rl.env import swap_role
    for k in JOKERS:
        assert JS.role(k) == swap_role(k), k


def test_descriptor_columns():
    base = item_descriptors()
    spec = item_descriptors(spec=True)
    assert spec.shape == (base.shape[0], base.shape[1] + len(JS.FEATURE_NAMES))
    assert np.array_equal(spec[:, :base.shape[1]], base)
    assert len(features_spec()) == spec.shape[1] and base.shape[1] == len(FEATURES)


def test_enums_and_fields():
    for k in JS.SPEC:
        f = JS.full(k)
        assert f["hand"] < len(HAND_NAMES) and -1 <= f["suit"] <= 3
        assert set(f["trigs"]) <= set(JS.TRIGGERS)
        if f["scales"]:
            assert f["counter_unit"] != "none" or f["grow_event"] != "none" or f["decay_event"] != "none"


def test_model_flag_roundtrip(tmp_path):
    import torch
    from balatro_rl.model import ActorCritic, save_model, load_model
    m = ActorCritic(desc=1, spec=1, attn=1)
    assert m.config["spec"] == 1 and m.desc.shape[1] == len(FEATURES) + len(JS.FEATURE_NAMES)
    save_model(m, str(tmp_path / "m.pt"))
    m2 = load_model(str(tmp_path / "m.pt"))
    assert torch.equal(m.desc, m2.desc)
    old = ActorCritic(desc=1, attn=1)
    assert "spec" not in old.config and old.desc.shape[1] == len(FEATURES)


# ---------------------------------------------------------------- flat effects against the engine
HANDS = {  # hand type -> (a hand that has it, a hand that does not)
    PAIR: ([C(9, 0), C(9, 1), C(4, 2), C(6, 3), C(2, 0)], [C(2, 0), C(4, 1), C(7, 2), C(9, 3), C(3, 0)]),
    TWO_PAIR: ([C(9, 0), C(9, 1), C(4, 2), C(4, 3), C(2, 0)], [C(9, 0), C(9, 1), C(4, 2), C(6, 3), C(2, 0)]),
    TRIPS: ([C(9, 0), C(9, 1), C(9, 2), C(4, 3), C(2, 0)], [C(9, 0), C(9, 1), C(4, 2), C(4, 3), C(2, 0)]),
    STRAIGHT: ([C(5, 0), C(6, 1), C(7, 2), C(8, 3), C(9, 0)], [C(2, 0), C(4, 1), C(7, 2), C(9, 3), C(3, 0)]),
    FLUSH: ([C(2, 1), C(4, 1), C(6, 1), C(8, 1), C(10, 1)], [C(2, 0), C(4, 1), C(7, 2), C(9, 3), C(3, 0)]),
    QUADS: ([C(9, 0), C(9, 1), C(9, 2), C(9, 3), C(2, 0)], [C(9, 0), C(9, 1), C(9, 2), C(4, 3), C(2, 0)]),
}
HC5 = HANDS[PAIR][1]


def delta(key, played, held=(), setup=None):
    """(d chips, d mult, mult ratio, d money) of having the joker, for this hand."""
    g0, g1 = game(), game(key)
    if setup:
        setup(g0)
        setup(g1)
    c0, m0, d0 = sc(g0, played, held)
    c1, m1, d1 = sc(g1, played, held)
    return c1 - c0, m1 - m0, (m1 / m0 if m0 else 1.0), d1 - d0


def expect(f, units=1.0, ratio_only=False):
    p = f["chance"] or 1.0
    return f["chips"] * units * p, f["mult"] * units * p, 1 + f["xmult"] * units * p, f["income_other"] * p


def check(key, got, units=1.0):
    f = JS.full(key)
    dc, dm, r, dmoney = got
    ec, em, er, emoney = expect(f, units)
    assert dc == pytest.approx(ec, abs=1e-6), (key, "chips", got)
    if f["xmult"] and not f["mult"]:
        assert r == pytest.approx(er, rel=1e-6), (key, "xmult", got)
    else:
        assert dm == pytest.approx(em, abs=1e-6), (key, "mult", got)
    if f["income_other"] and "scored_card" in f["trigs"]:
        assert dmoney == pytest.approx(emoney, abs=1e-6), (key, "money", got)


HAND_JOKERS = [k for k, s in JS.SPEC.items()
               if s.get("hand", -1) in HANDS and not s.get("scales") and s.get("trigs") == ("hand",)
               and (s.get("chips") or s.get("mult") or s.get("xmult"))]


def test_hand_type_jokers_found():
    assert set(HAND_JOKERS) == {"jolly", "zany", "mad", "crazy", "droll", "sly", "wily", "clever", "devious",
                                "crafty", "duo", "trio", "family", "order", "tribe"}


@pytest.mark.parametrize("key", HAND_JOKERS)
def test_hand_type_joker(key):
    f = JS.full(key)
    yes, no = HANDS[f["hand"]]
    check(key, delta(key, yes))
    dc, dm, r, _ = delta(key, no)
    assert dc == 0 and dm == 0 and r == 1


def pick_rank(f, inside):
    if inside:
        return f["rank_mask"][0] if f["rank_mask"] else 7
    return next(r for r in range(2, 11) if r not in f["rank_mask"] and r != 7 or (not f["rank_mask"] and r == 7))


SCORED = [k for k, s in JS.SPEC.items()
          if s.get("trigs") == ("scored_card",) and not s.get("scales") and not s.get("target_state")
          and not any(s.get(c) for c in ("creates_tarot", "needs_enh")) and not s.get("retriggers")
          and (s.get("chips") or s.get("mult") or s.get("xmult") or s.get("income_other"))]


def test_scored_card_jokers_found():
    assert {"greedy_joker", "lusty_joker", "wrathful_joker", "gluttenous_joker", "scary_face", "even_steven",
            "odd_todd", "scholar", "walkie_talkie", "smiley", "photograph", "fibonacci", "arrowhead", "onyx_agate",
            "bloodstone", "rough_gem", "business", "triboulet"} <= set(SCORED)


@pytest.mark.parametrize("key", SCORED)
def test_scored_card_joker(key):
    f = JS.full(key)
    mask = f["rank_mask"]
    r_in = mask[0] if mask else 7
    r_out = next(r for r in range(2, 11) if r not in mask) if mask else 7
    s_in = f["suit"] if f["suit"] >= 0 else 0
    s_out = (s_in + 1) % 4 if f["suit"] >= 0 else 0
    check(key, delta(key, [C(r_in, s_in)]))
    dc, dm, r, dmoney = delta(key, [C(r_out, s_out)])
    assert dc == 0 and dm == 0 and r == 1 and dmoney == 0
    if len(mask) > 1:                               # every rank of the mask, not only the first
        for rk in mask:
            check(key, delta(key, [C(rk, s_in)]))


def test_idol_and_ancient_with_their_target():
    def tgt(g):
        for j in g.jokers:
            j.state.update(rank=7, suit=1)
    g0, g1 = game(), game("idol")
    tgt(g1)
    assert sc(g1, [C(7, 1)])[1] / sc(g0, [C(7, 1)])[1] == pytest.approx(1 + JS.full("idol")["xmult"])
    assert sc(g1, [C(7, 2)])[1] == sc(g0, [C(7, 2)])[1]
    g1 = game("ancient")
    tgt(g1)
    assert sc(g1, [C(7, 1)])[1] / sc(g0, [C(7, 1)])[1] == pytest.approx(1 + JS.full("ancient")["xmult"])
    assert sc(g1, [C(7, 2)])[1] == sc(g0, [C(7, 2)])[1]


HELD = [k for k, s in JS.SPEC.items() if s.get("trigs") == ("held_card",) and s.get("rank_mask")
        and (s.get("mult") or s.get("xmult") or s.get("income_other"))]


@pytest.mark.parametrize("key", HELD)
def test_held_card_joker(key):
    f = JS.full(key)
    r_in = f["rank_mask"][0]
    r_out = next(r for r in range(2, 11) if r not in f["rank_mask"])
    got = delta(key, [C(2, 0)], held=[C(r_in, 0)])
    dc, dm, r, dmoney = got
    ec, em, er, emoney = expect(f)
    if f["xmult"]:
        assert r == pytest.approx(er)
    if f["mult"]:
        assert dm == pytest.approx(em)
    if f["income_other"]:
        assert dmoney == pytest.approx(emoney)
    dc, dm, r, dmoney = delta(key, [C(2, 0)], held=[C(r_out, 0)])
    assert dm == 0 and r == 1 and dmoney == 0


def test_always_and_conditions():
    for k in ("joker", "gros_michel", "cavendish", "stuntman", "misprint"):
        check(k, delta(k, HC5))
    # half joker: at most 3 cards
    f = JS.full("half")
    assert delta("half", [C(2), C(4, 1), C(7, 2)])[1] == f["mult"]
    assert delta("half", HC5[:4])[1] == 0 and f["max_cards"] == 3
    # banner: per discard left; mystic summit: only with none left
    f = JS.full("banner")
    for n in (0, 1, 3):
        assert delta("banner", HC5, setup=lambda g, n=n: setattr(g, "discards_left", n))[0] == f["chips"] * n
    f = JS.full("mystic_summit")
    assert delta("mystic_summit", HC5, setup=lambda g: setattr(g, "discards_left", 0))[1] == f["mult"]
    assert delta("mystic_summit", HC5, setup=lambda g: setattr(g, "discards_left", 1))[1] == 0
    # acrobat last hand, card sharp repeat hand type
    f = JS.full("acrobat")
    assert delta("acrobat", HC5, setup=lambda g: setattr(g, "hands_left", 1))[2] == pytest.approx(1 + f["xmult"])
    assert delta("acrobat", HC5, setup=lambda g: setattr(g, "hands_left", 2))[2] == 1
    f = JS.full("card_sharp")

    def seen(g):
        g.hand_played_round[HC] = 1
    assert delta("card_sharp", HC5, setup=seen)[2] == pytest.approx(1 + f["xmult"])
    assert delta("card_sharp", HC5)[2] == 1
    # blackboard (every held card Spades / Clubs), flower pot (four suits), seeing double (a Club and another suit)
    f = JS.full("blackboard")
    assert delta("blackboard", HC5[:1], held=[C(5, 0), C(6, 2)])[2] == pytest.approx(1 + f["xmult"])
    assert delta("blackboard", HC5[:1], held=[C(5, 0), C(6, 1)])[2] == 1
    f = JS.full("flower_pot")
    assert delta("flower_pot", HANDS[STRAIGHT][0])[2] == pytest.approx(1 + f["xmult"])
    assert delta("flower_pot", [C(5, 0), C(6, 0), C(7, 1), C(8, 1), C(9, 0)])[2] == 1
    f = JS.full("seeing_double")
    assert delta("seeing_double", [C(5, 2), C(6, 1), C(7, 2), C(8, 3), C(9, 0)])[2] == pytest.approx(1 + f["xmult"])
    assert delta("seeing_double", [C(5, 0), C(6, 1), C(7, 3), C(8, 3), C(9, 0)])[2] == 1


QTY = [  # key, setup, units of the scaled quantity, which delta
    ("abstract", lambda g: None, 1),                                 # itself
    ("blue_joker", lambda g: None, None),
    ("bull", lambda g: setattr(g, "money", 13), 13),
    ("bootstraps", lambda g: setattr(g, "money", 17), 17 // 5),
    ("fortune_teller", lambda g: setattr(g, "tarots_used", 3), 3),
    ("erosion", lambda g: [g.full_deck.pop() for _ in range(3)], 3),
    ("stone", lambda g: g.full_deck.extend(C(2, 0, enh="STONE") for _ in range(2)), 2),
    ("steel_joker", lambda g: g.full_deck.extend(C(2, 0, enh="STEEL") for _ in range(2)), 2),
    ("throwback", lambda g: setattr(g, "blinds_skipped", 2), 2),
    ("supernova", lambda g: g.hand_played.__setitem__(HC, 3), 4),         # times this hand type was played, + this one
    ("stencil", lambda g: setattr(g, "joker_slots", 5), 4),               # x(1 + empty slots): 4 empty
    ("drivers_license", lambda g: g.full_deck.__setitem__(slice(0, 16), [C(2, 0, enh="BONUS") for _ in range(16)]), 1),
]


@pytest.mark.parametrize("key,setup,units", QTY, ids=[q[0] for q in QTY])
def test_quantity_scaled(key, setup, units):
    f = JS.full(key)
    g0, g1 = game(), game(key)
    setup(g0)
    setup(g1)
    if key == "blue_joker":
        units = len(g1.deck)
    c0, m0, _ = sc(g0, HC5)
    c1, m1, _ = sc(g1, HC5)
    if key == "stencil":
        assert m1 / m0 == pytest.approx(1 + f["xmult"] * units)
        return
    if key == "drivers_license":
        assert f["threshold"] == 16 and m1 / m0 == pytest.approx(1 + f["xmult"])
        g2 = game(key)
        g2.full_deck.__setitem__(slice(0, 15), [C(2, 0, enh="BONUS") for _ in range(15)])
        assert sc(g2, HC5)[1] == m0
        return
    if key == "abstract":
        assert m1 - m0 == f["mult"] * len(g1.jokers)
        return
    assert c1 - c0 == pytest.approx(f["chips"] * units)
    if f["xmult"]:
        assert m1 / m0 == pytest.approx(1 + f["xmult"] * units)
    else:
        assert m1 - m0 == pytest.approx(f["mult"] * units)


def test_unit_field_is_the_divisor():
    assert JS.full("bootstraps")["unit"] == 5 and JS.full("bootstraps")["mult"] == 2


def test_retriggers():
    for key, played, applies in (("hack", [C(3)], True), ("hack", [C(7)], False), ("sock_and_buskin", [C(12)], True),
                                 ("sock_and_buskin", [C(5)], False), ("selzer", [C(5)], True),
                                 ("hanging_chad", [C(5)], True)):
        c0, _, _ = sc(game(), played)
        c1, _, _ = sc(game(key), played)
        assert c1 - c0 == (played[0].chip_value() * JS.full(key)["retriggers"] if applies else 0), (key, c0, c1)
    g = game("dusk")
    g.hands_left = 1
    assert sc(g, [C(5)])[0] - sc(game(), [C(5)])[0] == 5 * JS.full("dusk")["retriggers"]
    g.hands_left = 2
    assert sc(g, [C(5)])[0] == sc(game(), [C(5)])[0]


# ---------------------------------------------------------------- scalers
def committed(g, played, held=()):
    score_hand(g, list(played), list(held), rng=random.Random(1), commit=True)


def ready(*keys):
    g = game(*keys)
    g.select_blind()
    return g


def test_reroll_flash():
    f = JS.full("flash")
    g = game("flash")
    g.open_shop()
    g.money = 100
    v0 = val(g, "flash")
    g.reroll()
    assert val(g, "flash") - v0 == f["grow_amount"] and f["grow_event"] == "reroll"
    assert sc(g, HC5)[1] - sc(game(), HC5)[1] == val(g, "flash")


def test_planet_constellation():
    f = JS.full("constellation")
    g = game("constellation")
    v0 = g.jokers[0].state["val"]
    assert v0 == f["start_value"]
    c = Consumable("planet", "pluto")
    g.consumables.append(c)
    g.apply_consumable(c, [])
    assert g.jokers[0].state["val"] - v0 == pytest.approx(f["grow_amount"]) and f["grow_event"] == "planet_used"
    ref = game()
    ref.hand_levels = list(g.hand_levels)
    assert sc(g, HC5)[1] / sc(ref, HC5)[1] == pytest.approx(g.jokers[0].state["val"])


FOUR = [C(9, 0), C(9, 1), C(4, 2), C(6, 3)]
HAND_PLAYED = ["green_joker", "ride_the_bus", "square", "runner", "trousers"]


@pytest.mark.parametrize("key", HAND_PLAYED)
def test_hand_played_growth(key):
    f = JS.full(key)
    assert f["grow_event"] in ("hand_played", "hand_of_type_played")
    g = game(key)
    played = HANDS[f["hand"]][0] if f["hand"] >= 0 else FOUR
    if key == "square":
        assert f["exact_cards"] == len(played) == 4
    v0 = val(g, key)
    committed(g, played)
    assert val(g, key) - v0 == f["grow_amount"], key
    if f["hand"] >= 0:                                  # not this hand type: no growth
        g = game(key)
        committed(g, HANDS[f["hand"]][1])
        assert val(g, key) == v0
    # the counter is added to the score in its unit; a qualifying hand grows it first, then scores with it
    g2 = game(key)
    g2.jokers[0].state["val"] = 2 * f["grow_amount"]
    base = sc(game(), played)
    got = sc(g2, played)
    if f["counter_unit"] == "chips":
        assert got[0] - base[0] == 3 * f["grow_amount"]
    else:
        assert got[1] - base[1] == 3 * f["grow_amount"]


def test_ride_the_bus_resets_on_a_face():
    g = game("ride_the_bus")
    committed(g, FOUR)
    committed(g, FOUR)
    assert val(g, "ride_the_bus") == 2
    committed(g, [C(12, 0), C(12, 1), C(4, 2)])
    assert val(g, "ride_the_bus") == JS.full("ride_the_bus")["start_value"] == 0 and \
        JS.full("ride_the_bus")["reset_event"] == "hand_played"


def test_obelisk():
    f = JS.full("obelisk")
    g = game("obelisk")
    g.hand_played[HC] = 5
    committed(g, HANDS[PAIR][0])                       # not the most played hand: +0.2
    assert val(g, "obelisk") == pytest.approx(f["start_value"] + f["grow_amount"])
    g.hand_played[PAIR] = 9
    committed(g, HANDS[PAIR][0])                       # the most played: back to the start
    assert val(g, "obelisk") == f["start_value"]


def test_card_added_hologram_and_destroyed():
    f = JS.full("hologram")
    g = game("hologram")
    g.add_card(C(7))
    assert g.jokers[0].state["val"] == pytest.approx(f["start_value"] + f["grow_amount"]) and f["grow_event"] == "card_added"
    f = JS.full("glass")
    g = game("glass")
    c = C(7, enh="GLASS")
    g.full_deck.append(c)
    g.destroy_cards([c])
    assert val(g, "glass") == pytest.approx(f["start_value"] + f["grow_amount"])
    f = JS.full("caino")
    g = game("caino")
    c = C(12)
    g.full_deck.append(c)
    g.destroy_cards([c])
    assert val(g, "caino") == pytest.approx(f["start_value"] + f["grow_amount"])
    c = C(5)
    g.full_deck.append(c)
    g.destroy_cards([c])
    assert val(g, "caino") == pytest.approx(f["start_value"] + f["grow_amount"])      # not a face card


def test_card_sold_campfire_and_pack_skipped_red_card():
    f = JS.full("campfire")
    g = game("campfire", "joker")
    g.state = "SHOP"
    g.sell_joker(1)
    assert val(g, "campfire") == pytest.approx(f["start_value"] + f["grow_amount"]) and f["grow_event"] == "card_sold"
    f = JS.full("red_card")
    g = game("red_card")
    g.pack_skip()
    assert val(g, "red_card") == f["grow_amount"] and f["grow_event"] == "pack_skipped"


def test_card_scored_growth():
    f = JS.full("wee")
    g = game("wee")
    committed(g, [C(2, 0), C(2, 1), C(5, 2)])
    assert val(g, "wee") == 2 * f["grow_amount"]
    f = JS.full("hiker")
    g = game("hiker")
    c = C(5)
    committed(g, [c])
    assert c.extra_chips == f["grow_amount"]
    f = JS.full("vampire")
    g = game("vampire")
    c = C(5, enh="BONUS")
    committed(g, [c])
    assert val(g, "vampire") == pytest.approx(f["start_value"] + f["grow_amount"]) and c.enh == ""


def test_blind_select_madness():
    f = JS.full("madness")
    g = game("madness", "joker")
    g.select_blind()
    assert val(g, "madness") == pytest.approx(f["start_value"] + f["grow_amount"]) and f["destroys_joker"]
    assert [j.key for j in g.jokers] == ["madness"]


def test_discard_growth_and_decay():
    f = JS.full("green_joker")
    g = ready("green_joker")
    g.jokers[0].state["val"] = 3
    g.discard([0])
    assert val(g, "green_joker") == 3 - f["decay_amount"] and f["decay_event"] == "discard"
    f = JS.full("ramen")
    g = ready("ramen")
    g.discard([0, 1, 2])
    assert val(g, "ramen") == pytest.approx(f["start_value"] - 3 * f["decay_amount"])
    assert sc(g, HC5)[1] / sc(game(), HC5)[1] == pytest.approx(val(g, "ramen"))
    f = JS.full("hit_the_road")
    g = ready("hit_the_road")
    g.hand[0], g.hand[1], g.hand[2] = C(11, 0), C(11, 1), C(5, 0)
    g.discard([0, 1, 2])
    assert val(g, "hit_the_road") == pytest.approx(f["start_value"] + 2 * f["grow_amount"])
    f = JS.full("yorick")
    g = ready("yorick")
    g.jokers[0].state["left"] = 2
    g.discard([0, 1])
    assert val(g, "yorick") == pytest.approx(f["start_value"] + f["grow_amount"]) and f["grow_per"] == 23


def test_decay_on_hands_played():
    for key in ("ice_cream", "selzer"):
        f = JS.full(key)
        g = ready(key)
        v0 = val(g, key)
        assert v0 == f["start_value"]
        g.play([0])
        assert v0 - val(g, key) == f["decay_amount"] and f["decay_event"] == "hand_played", key
    assert JS.full("ice_cream")["counter_unit"] == "chips"
    assert sc(game("ice_cream"), HC5)[0] - sc(game(), HC5)[0] == JS.full("ice_cream")["start_value"]


def end_round(g):
    g.chips = g.target
    g.win_round()


@pytest.mark.parametrize("key", ["popcorn", "turtle_bean", "egg", "gift", "invisible", "gros_michel"])
def test_round_end(key):
    f = JS.full(key)
    g = ready(key)
    j = g.jokers[0]
    v0, sb0 = j.state.get("val", 0), j.sell_bonus
    random.seed(0)
    end_round(g)
    if key == "popcorn":
        assert v0 == f["start_value"] and v0 - val(g, key) == f["decay_amount"]
    elif key == "turtle_bean":
        assert v0 - val(g, key) == f["decay_amount"] and v0 == 5 == f["d_hand_size"]
    elif key == "egg":
        assert j.sell_bonus - sb0 == f["sell_gain"]
    elif key == "gift":
        assert j.sell_bonus - sb0 == f["sell_all_gain"]
    elif key == "invisible":
        assert val(g, key) - v0 == f["grow_amount"]
    else:
        assert f["expire_prob"] == pytest.approx(1 / 6)


def test_economy_income():
    def money(*keys, boss=False):
        g = ready(*keys)
        g.hands_left = 0
        m0 = g.money
        end_round(g)
        return g.money - m0
    base = money()
    assert money("golden") - base == JS.full("golden")["income_round"]
    assert money("rocket") - base == JS.full("rocket")["start_value"] == JS.full("rocket")["income_round"]
    g = ready("satellite")
    g.planets_used.update(["pluto", "mars"])
    m0 = g.money
    end_round(g)
    g2 = ready()
    g2.planets_used.update(["pluto", "mars"])
    m1 = g2.money
    end_round(g2)
    assert (g.money - m0) - (g2.money - m1) == 2 * JS.full("satellite")["income_round"]
    g = ready("cloud_9")
    n9 = sum(1 for c in g.full_deck if c.rank == 9)
    g.hands_left = 0
    m0 = g.money
    end_round(g)
    assert g.money - m0 - base == n9 * JS.full("cloud_9")["income_round"]


def test_room_and_economy_flags_against_game():
    g0 = game()
    for key, attr in (("drunkard", "round_discards"), ("merry_andy", "round_discards"),
                      ("troubadour", "round_hands"), ("juggler", "effective_hand_size"),
                      ("stuntman", "effective_hand_size"), ("merry_andy", "effective_hand_size"),
                      ("troubadour", "effective_hand_size")):
        f = JS.full(key)
        g = game(key)
        d = getattr(g, attr)() - getattr(g0, attr)()
        want = f["d_discards"] if attr == "round_discards" else f["d_hands"] if attr == "round_hands" else f["d_hand_size"]
        assert d == want, (key, attr)
    assert game("credit_card").debt_limit() == -JS.full("credit_card")["debt"]
    g = game("chaos")
    g.open_shop()
    assert g.free_rerolls == JS.full("chaos")["free_rerolls"]
    g = ready()
    g.money = 25
    g.hands_left = 0
    m0 = g.money
    end_round(g)
    base = g.money - m0
    g = ready("to_the_moon")
    g.money = 25
    g.hands_left = 0
    m0 = g.money
    end_round(g)
    assert g.money - m0 - base == 5 * JS.full("to_the_moon")["interest"]
