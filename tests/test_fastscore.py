"""The compiled scorer (sim/_fastscore.pyx) must give exactly the same predictions as Game.predict:
random game states with random jokers, cards and bosses, every candidate play compared."""
import os
import random
import sys
from itertools import combinations

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from balatro_rl.sim import fastscore  # noqa: E402
from balatro_rl.sim.cards import Card  # noqa: E402
from balatro_rl.sim.game import Game, Consumable  # noqa: E402
from balatro_rl.sim.items import PLANETS  # noqa: E402
from balatro_rl.sim.jokers import Joker, JOKERS  # noqa: E402
from balatro_rl.sim.scoring import Plan  # noqa: E402

pytestmark = pytest.mark.skipif(not fastscore.ENABLED, reason="compiled scorer not built")

KEYS = sorted(JOKERS)
HOT = ["blueprint", "brainstorm", "photograph", "vampire", "midas_mask", "lucky_cat", "raised_fist",
       "splash", "four_fingers", "shortcut", "smeared", "pareidolia", "mime", "oops", "hanging_chad",
       "baseball", "flower_pot", "seeing_double", "blackboard", "wee", "idol", "ancient", "obelisk",
       "loyalty_card", "swashbuckler", "stencil", "card_sharp", "sock_and_buskin", "ride_the_bus"]
ENH = ["", "", "", "BONUS", "MULT", "WILD", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY"]
EDS = ["", "", "", "FOIL", "HOLO", "POLYCHROME", "NEGATIVE"]
SEALS = ["", "", "", "RED", "RED", "BLUE", "GOLD", "PURPLE"]
BOSSES = ["", "flint", "arm", "psychic", "eye", "mouth", "crimson_heart", "club", "plant"]


def rand_card(rng, few_suits):
    c = Card(rng.randrange(2, 15), rng.randrange(2 if few_suits else 4), enh=rng.choice(ENH),
             edition=rng.choice(EDS), seal=rng.choice(SEALS))
    c.extra_chips = rng.choice([0, 0, 0, 5, 10, 35])
    c.debuffed = rng.random() < 0.1
    c.hidden = rng.random() < 0.08
    return c


def rand_state(rng):
    g = Game(seed=rng.randrange(10**6), deck_type=rng.choice(["RED", "PLASMA", "CHECKERED", "ERRATIC"]),
             stake=rng.choice(["WHITE", "GOLD"]))
    g.state = "SELECTING_HAND"
    g.blind_idx = rng.choice([0, 1, 2, 2])
    g.boss = rng.choice(BOSSES)
    g.boss_disabled = rng.random() < 0.15
    g.jokers = []
    for _ in range(rng.randint(0, 8)):
        key = rng.choice(HOT) if rng.random() < 0.4 else rng.choice(KEYS)
        j = Joker(key, edition=rng.choice(EDS), base_cost=rng.randint(1, 10))
        if j.d.init:
            j.d.init(g, j)
        if "val" in j.state or rng.random() < 0.3:
            j.state["val"] = rng.choice([0, 1, 2, 5, 13, 1.0, 1.5, 2.75, 0.6, 20])
        if rng.random() < 0.3:
            j.state["rank"], j.state["suit"] = rng.randrange(2, 15), rng.randrange(4)
        if rng.random() < 0.2:
            j.state.pop("val", None)       # missing state -> hook defaults
        j.debuffed = rng.random() < 0.1
        j.sell_bonus = rng.randint(0, 3)
        g.jokers.append(j)
    if g.jokers and rng.random() < 0.5:
        g.crimson_disabled = rng.choice(g.jokers).uid
    few = g.deck_type == "CHECKERED"
    g.hand = [rand_card(rng, few) for _ in range(rng.choice([rng.randint(1, 8), rng.randint(1, 16)]))]
    g.hand_levels = [rng.randint(0, 6) for _ in range(12)]
    g.hand_played = [rng.choice([0, 0, rng.randint(0, 12)]) for _ in range(12)]
    g.hand_played_round = [rng.choice([0, 0, 1, 2]) for _ in range(12)]
    g.discards_left = rng.randint(0, 4)
    g.hands_left = rng.randint(1, 4)
    g.money = rng.randint(-20, 80)
    g.full_deck = [rand_card(rng, few) for _ in range(rng.randint(20, 60))]
    g.deck = g.full_deck[:rng.randint(0, len(g.full_deck))]
    g.tarots_used = rng.randint(0, 9)
    g.blinds_skipped = rng.randint(0, 5)
    g.joker_slots = rng.randint(3, 7)
    if rng.random() < 0.3:
        g.vouchers = list(g.vouchers) + ["observatory"] if isinstance(g.vouchers, list) else set(g.vouchers) | {"observatory"}
    g.consumables = [Consumable("planet", rng.choice(list(PLANETS))) for _ in range(rng.randint(0, 3))]
    g.round_hand_types = set(rng.sample(range(12), rng.randint(0, 3)))
    g.mouth_hand = rng.choice([-1, -1, rng.randrange(12)])
    return g


def test_compiled_scorer_matches_python():
    rng = random.Random(7)
    for s in range(300):
        g = rand_state(rng)
        plan = Plan(g)
        view = g.hand_view()
        subs = [c for k in range(1, 6) for c in combinations(range(min(len(g.hand), 9)), k)]
        ref = [g.predict(list(c), plan, view) for c in subs]
        got = fastscore.build(g, plan, view).predict_many(subs)
        for c, a, b in zip(subs, ref, got):
            assert a == b and type(a[0]) is type(b[0]), (s, c, a, b, [j.key for j in g.jokers])