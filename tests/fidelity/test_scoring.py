import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from balatro_rl.sim.cards import Card
from balatro_rl.sim.game import Game
from balatro_rl.sim.jokers import Joker, JOKERS
from balatro_rl.sim.hands import evaluate, PAIR, FLUSH, STRAIGHT, FULL_HOUSE, TWO_PAIR, HC
from balatro_rl.sim.scoring import score_hand

S, H, C, D = 0, 1, 2, 3


def g_with(*jkeys):
    g = Game(seed=1, stake="WHITE")
    for k in jkeys:
        g.jokers.append(Joker(k, base_cost=JOKERS[k].cost))
    return g


def sc(g, played, held=()):
    return score_hand(g, list(played), list(held))[0]


def test_hand_types():
    assert evaluate([Card(14, S), Card(14, H)]).hand == PAIR
    assert evaluate([Card(2, H), Card(5, H), Card(7, H), Card(9, H), Card(13, H)]).hand == FLUSH
    assert evaluate([Card(14, S), Card(2, H), Card(3, C), Card(4, D), Card(5, S)]).hand == STRAIGHT
    assert evaluate([Card(3, S), Card(3, H), Card(3, C), Card(9, D), Card(9, S)]).hand == FULL_HOUSE
    assert evaluate([Card(3, S), Card(3, H), Card(9, D), Card(9, S), Card(4, S)]).hand == TWO_PAIR
    assert evaluate([Card(2, H), Card(5, H), Card(7, H), Card(9, H)], four_fingers=True).hand == FLUSH
    assert evaluate([Card(2, S), Card(4, H), Card(6, C), Card(8, D), Card(10, S)], shortcut=True).hand == STRAIGHT
    r = evaluate([Card(14, S), Card(14, H), Card(5, D)])
    assert r.scoring == [0, 1]
    assert evaluate([Card(9, S), Card(4, H)]).scoring == [0]


def test_base_scores():
    g = g_with()
    assert sc(g, [Card(14, S), Card(14, H)]) == (10 + 22) * 2
    assert sc(g, [Card(2, H), Card(5, H), Card(7, H), Card(9, H), Card(13, H)]) == (35 + 33) * 4
    assert sc(g, [Card(14, S), Card(2, H), Card(3, C), Card(4, D), Card(5, S)]) == (30 + 25) * 4


def test_jokers():
    assert sc(g_with("joker"), [Card(14, S), Card(14, H)]) == 32 * 6
    assert sc(g_with("jolly"), [Card(14, S), Card(14, H)]) == 32 * 10
    assert sc(g_with("duo"), [Card(14, S), Card(14, H)]) == 32 * 4
    # Lusty: +3 mult per heart scored
    assert sc(g_with("lusty_joker"), [Card(14, S), Card(14, H)]) == 32 * 5
    # Hack retriggers 2-5
    assert sc(g_with("hack"), [Card(5, S), Card(5, H)]) == (10 + 20) * 2
    # Baron: kings held x1.5 each
    assert sc(g_with("baron"), [Card(14, S), Card(14, H)], [Card(13, S), Card(13, D)]) == int(32 * 2 * 2.25)
    # Joker order: +mult then xmult vs xmult then +mult
    assert sc(g_with("joker", "duo"), [Card(14, S), Card(14, H)]) == 32 * 12
    assert sc(g_with("duo", "joker"), [Card(14, S), Card(14, H)]) == 32 * 8
    # Blueprint copies the joker to its right
    assert sc(g_with("blueprint", "duo"), [Card(14, S), Card(14, H)]) == 32 * 8


def test_enhancements():
    g = g_with()
    a = Card(14, S, enh="GLASS")
    assert sc(g, [a, Card(14, H)]) == 32 * 4
    b = Card(14, S, enh="BONUS", seal="RED")
    assert sc(g, [b, Card(14, H)]) == (10 + 11 + 11 + 11 + 30 + 30) * 2
    assert sc(g, [Card(14, S), Card(14, H)], [Card(3, D, enh="STEEL")]) == 32 * 3


def test_game_runs():
    import random
    for seed in range(20):
        g = Game(seed=seed)
        rng = random.Random(seed)
        steps = 0
        while not g.done and steps < 3000:
            steps += 1
            if g.state == "BLIND_SELECT":
                g.select_blind()
            elif g.state == "SELECTING_HAND":
                n = rng.randint(1, min(5, len(g.hand)))
                pos = sorted(rng.sample(range(len(g.hand)), n))
                if g.discards_left and rng.random() < 0.3:
                    g.discard(pos)
                else:
                    g.play(pos)
            elif g.state == "SHOP":
                g.leave_shop()
            elif g.state == "PACK":
                g.pack_skip()
        assert g.done
