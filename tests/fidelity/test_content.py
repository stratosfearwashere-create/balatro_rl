"""Checks for the jokers and mechanics added to mirror the full game."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from balatro_rl.env import BalatroEnv, encode, A_PLAY, A_SWAP, A_REROLL_BOSS, N_SUB, subset_of  # noqa: E402
from balatro_rl.sim.cards import Card  # noqa: E402
from balatro_rl.sim.game import Game  # noqa: E402
from balatro_rl.sim.items import VOUCHERS, SPECTRALS, TAGS, DECKS  # noqa: E402
from balatro_rl.sim.jokers import JOKERS, Joker  # noqa: E402
from balatro_rl.sim.scoring import score_hand  # noqa: E402

S, H, C, D = 0, 1, 2, 3
PAIR_AA = (10 + 22)          # pair of aces: base 10 chips + 11 + 11


def game(*keys, stake="WHITE"):
    g = Game(seed=1, stake=stake)
    for k in keys:
        g.add_joker(Joker(k, base_cost=JOKERS[k].cost))
    return g


def sc(g, played, held=()):
    return score_hand(g, list(played), list(held))[0]


def test_content_counts():
    from collections import Counter
    assert Counter(d.rarity for d in JOKERS.values()) == Counter({1: 61, 2: 64, 3: 20, 4: 5})
    assert len(VOUCHERS) == 32 and len(SPECTRALS) == 18 and len(TAGS) == 24 and len(DECKS) == 15


def test_new_scoring_jokers():
    assert sc(game("triboulet"), [Card(13, S), Card(13, H)]) == (10 + 20) * 2 * 4
    assert sc(game("stuntman"), [Card(14, S), Card(14, H)]) == (PAIR_AA + 250) * 2
    # Baseball: each uncommon joker x1.5 (Hack is uncommon)
    assert sc(game("hack", "baseball"), [Card(14, S), Card(14, H)]) == int(PAIR_AA * 2 * 1.5)
    # Vampire strips the enhancement before scoring and gains X0.1 per card
    g = game("vampire")
    a = Card(14, S, enh="MULT")
    assert sc(g, [a, Card(14, H)]) == int(PAIR_AA * 2 * 1.1)
    assert a.enh == "MULT"                                   # prediction didn't mutate
    # Midas Mask turns scored face cards to Gold (Gold cards give nothing when played)
    g = game("midas_mask")
    k = Card(13, S, enh="MULT")
    assert sc(g, [k, Card(13, H)]) == (10 + 20) * 2
    # Erosion: +4 mult per card below starting deck size
    g = game("erosion")
    g.full_deck = g.full_deck[:50]
    assert sc(g, [Card(14, S), Card(14, H)]) == PAIR_AA * (2 + 8)
    # Driver's License needs 16 enhanced cards
    g = game("drivers_license")
    for c in g.full_deck[:16]:
        c.enh = "BONUS"
    assert sc(g, [Card(14, S), Card(14, H)]) == PAIR_AA * 2 * 3
    # Oops doubles Lucky card odds: expected +20 mult * 2/5
    g = game("oops")
    assert abs(sc(g, [Card(14, S, enh="LUCKY"), Card(14, H)]) - int(PAIR_AA * (2 + 8))) <= 1


def test_game_event_jokers():
    g = game("hologram")
    g.add_card(Card(5, S))
    assert g.jokers[0].state["val"] == 1.25
    g = game("glass", "caino")
    g.destroy_cards([Card(13, S, enh="GLASS")])          # not in deck -> ignored
    c = g.full_deck[0]
    c.enh = "GLASS"
    c.rank = 12
    g.destroy_cards([c])
    assert g.jokers[0].state["val"] == 1.75 and g.jokers[1].state["val"] == 2.0
    g = game("campfire")
    g.state = "SHOP"
    g.add_joker(Joker("joker", base_cost=2))
    g.sell_joker(1)
    assert g.jokers[0].state["val"] == 1.25


def test_mr_bones_saves():
    g = game("mr_bones")
    g.select_blind()
    g.hands_left = 1
    g.chips = int(g.target * 0.3)
    g.hand = [Card(2, S), Card(3, H), Card(9, C)]
    g.play([0])
    assert g.state == "SHOP" and not g.jokers


def test_hand_size_and_candidates():
    g = game("juggler", "troubadour")
    assert g.effective_hand_size() == 11
    g.select_blind()
    assert len(g.hand) == 11
    o = encode(g)
    subs = [subset_of(o, A_PLAY + i) for i in range(N_SUB) if o["mask"][A_PLAY + i]]
    assert len(subs) == N_SUB and max(max(s) for s in subs) >= 8


def test_face_down_cards_hidden_from_prediction():
    g = Game(seed=3, stake="WHITE")
    g.blind_idx, g.boss = 2, "house"
    g.select_blind()
    assert all(c.hidden for c in g.hand)
    sc_, _ = g.predict([0, 1])
    assert sc_ <= 10                                    # hidden cards give no chips in predictions
    o = encode(g)
    assert o["cards"][0, -1] == 1.0                     # hidden flag set, rank unknown
    assert o["cards"][0, 1:14].sum() == 0


def test_joker_swap_and_boss_reroll():
    env = BalatroEnv(stake="WHITE")
    env.reset(1)
    g = env.g
    g.add_joker(Joker("joker", base_cost=2))
    g.add_joker(Joker("duo", base_cost=8))
    g.vouchers.add("directors_cut")
    g.money = 30
    o = env.obs = encode(g, env.cnt)
    assert o["mask"][A_REROLL_BOSS]
    old = g.boss
    env.step(A_REROLL_BOSS)
    assert g.boss != old and g.money == 20 and not env.obs["mask"][A_REROLL_BOSS]
    env.g.state = "SHOP"
    env.obs = encode(g, env.cnt)
    assert env.obs["mask"][A_SWAP]
    env.step(A_SWAP)
    assert [j.key for j in g.jokers] == ["duo", "joker"]


def test_decks():
    for d in DECKS:
        g = Game(seed=0, deck_type=d)
        assert len(g.full_deck) in (40, 52)
    assert Game(0, "PAINTED").effective_hand_size() == 10
    assert Game(0, "GHOST").consumables[0].name == "hex"
    ranks = {c.rank for c in Game(0, "ERRATIC").full_deck}
    assert len(ranks) > 5
