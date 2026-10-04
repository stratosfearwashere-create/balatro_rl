"""Verdant Leaf: with every card debuffed, selling a joker is the move that makes scoring possible, so the
solver gives in-round sells a playout value and the agent picks one. The bridge transmits the agent's card
order (Hanging Chad, Photograph ...) through a hand rearrangement before the play."""
import random
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from balatro_rl.az import solver as S                                 # noqa: E402
from balatro_rl.az.actions import enumerate_candidates, Action        # noqa: E402
from balatro_rl.az.agent import solver_prior                          # noqa: E402
from balatro_rl.az.world import World                                 # noqa: E402
from balatro_rl.sim.game import Game                                  # noqa: E402
from balatro_rl.sim.jokers import Joker                               # noqa: E402
from balatro_rl import bridge as B                                    # noqa: E402


def _verdant_game(seed=5):
    g = Game(seed=seed, stake="WHITE")
    g.ante = 2                                    # at ante 1 the base hand chips clear the boss even debuffed
    for k in ("joker", "greedy_joker"):
        g.add_joker(Joker(k, base_cost=4))
    g.boss, g.blind_idx = "verdant_leaf", 2
    g.select_blind()
    assert g.flags.get("verdant") and g.state == "SELECTING_HAND"
    assert all(c.debuffed for c in g.hand)
    return g


@pytest.mark.parametrize("solver_cls", [S.PyRoundSolver] + ([S.CoreRoundSolver] if S.CORE else []))
def test_sells_get_a_playout_value_under_verdant_leaf(solver_cls):
    g = _verdant_game()
    ch = enumerate_candidates(World(g), random.Random(1))
    sells = [c for c in ch.cands if c.kind == "sell_joker"]
    assert sells, "in-round sells are offered"
    solver_cls(samples=6, joker_state=True, use_playout=True).solve(g, ch, random.Random(2), depth=1)
    assert all(c.p_clear >= 0 for c in sells)
    best_play = max([c.p_clear for c in ch.cands if c.kind == "play"], default=0.0)
    assert max(c.p_clear for c in sells) > best_play, "selling clears the debuff; a debuffed play cannot score"
    prior = solver_prior(ch, 1.0)
    assert ch.cands[int(prior.argmax())].kind == "sell_joker"


def test_no_playout_for_sells_outside_verdant_leaf():
    g = Game(seed=5, stake="WHITE")
    g.add_joker(Joker("joker", base_cost=4))
    g.select_blind()
    ch = enumerate_candidates(World(g), random.Random(1))
    S.PyRoundSolver(samples=6, joker_state=True, use_playout=True).solve(g, ch, random.Random(2), depth=1)
    assert all(c.p_clear < 0 for c in ch.cands if c.kind == "sell_joker")


def test_rearrange_puts_the_played_cards_in_the_agents_order():
    g = Game(seed=1, stake="WHITE")
    # sim hand positions 0..4 live in game slots 3,0,4,1,2 (the bridge sorts the hand)
    hand_map = [3, 0, 4, 1, 2]
    a = Action("play", cards=(2, 0, 3))          # game slots 4, 3, 1 in that order
    order = B.az_rearrange(g, a, hand_map, 5)
    assert sorted(order) == [0, 1, 2, 3, 4]
    slots = sorted(hand_map[p] for p in a.cards)  # the slots they occupy: 1, 3, 4
    assert [order[s] for s in slots] == [4, 3, 1]
    assert order[0] == 0 and order[2] == 2       # the other cards stay put
    assert B.az_rearrange(g, Action("play", cards=(0, 1, 2)), hand_map, 5) is None
    assert B.az_rearrange(g, Action("discard", cards=(2, 0)), hand_map, 5) is None
    _, params = B.az_to_rpc(g, a, hand_map)
    assert params == {"cards": [1, 3, 4]}
