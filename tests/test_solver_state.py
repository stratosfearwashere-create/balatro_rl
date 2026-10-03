"""The solver's playouts carry joker state and hands played (az/solver.py joker_state).

- the C++ core and the Python reference give the same P(clear) / expected chips on the regression states;
- the playouts' hook table does what the game's real hooks do (Green Joker, Castle, Hit the Road, Ramen,
  Ice Cream, and the scorer's "before" hooks through score_hand);
- carrying state changes the estimate where it should (Green Joker over several hands) and a hand the
  boss forbids changes nothing;
- joker_state=False is the old, stateless solver.
"""
import random
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from balatro_rl.az import solver as S                                 # noqa: E402
from balatro_rl.az.actions import enumerate_candidates                # noqa: E402
from balatro_rl.az.solver import PyRoundSolver, _Round, base_state      # noqa: E402
from balatro_rl.az.world import World                                 # noqa: E402
from balatro_rl.sim.cards import Card                                 # noqa: E402
from balatro_rl.sim.game import Game                                  # noqa: E402
from balatro_rl.sim.jokers import Joker                               # noqa: E402
from balatro_rl.sim.scoring import Plan, score_hand                   # noqa: E402
import test_regression as T                                           # noqa: E402


def _cands(ch):
    return [(str(c.action), repr(c.p_clear), repr(c.e_chips)) for c in ch.cands]


@pytest.mark.skipif(not S.CORE, reason="the C++ core is not built")
def test_core_matches_python_reference_with_state():
    n = 0
    for i, g in enumerate(T.states()):
        for depth in (1, 2):
            a = enumerate_candidates(World(g), random.Random(i))
            S.CoreRoundSolver(samples=6, joker_state=True).solve(g, a, random.Random(100 + i), depth=depth)
            b = enumerate_candidates(World(g), random.Random(i))
            PyRoundSolver(samples=6, joker_state=True).solve(g, b, random.Random(100 + i), depth=depth)
            assert _cands(a) == _cands(b), f"state {i} depth {depth}"
            n += 1
    assert n >= 60


def test_stateless_flag_is_the_old_solver():
    """joker_state=False: identical to the Cython solver / the golden file's solver section."""
    import json
    gold = json.load(open(T.GOLDEN))
    for i, g in enumerate(T.states()[:6]):
        ch = enumerate_candidates(World(g), random.Random(i))
        S.RoundSolver(samples=6, joker_state=False, use_playout=False).solve(g, ch, random.Random(100 + i), depth=1)
        assert T._cands(ch) == gold["states"][i]["depth1"], f"state {i}"


def _white(seed, jokers, hand=None, boss=None):
    g = Game(seed=seed, stake="WHITE")
    for k, st in jokers:
        j = Joker(k, base_cost=4)
        g.add_joker(j)
        j.state.update(st)
    if boss:
        g.boss, g.blind_idx = boss, 2
    g.select_blind()
    if hand is not None:
        g.hand = hand
    return g


def _solver_round(sv, g):
    sg = sv._setup(g)
    return sg, _Round(list(sg.hand), list(sg.deck), sg.hands_left, sg.discards_left, 0, frozenset(), -1, base_state(sg))


@pytest.mark.parametrize("key,state,discard_cards", [
    ("green_joker", {"val": 3}, 2),
    ("green_joker", {"val": 0}, 1),
    ("castle", {"val": 6, "suit": 1}, 4),
    ("hit_the_road", {"val": 1.0}, 5),
    ("ramen", {"val": 1.03}, 5),
])
def test_discard_table_matches_the_real_hook(key, state, discard_cards):
    g = _white(3, [(key, state)])
    hand = [Card(11, 1), Card(11, 0), Card(5, 1), Card(9, 2), Card(13, 3), Card(2, 1), Card(7, 0), Card(4, 2)]
    g.hand = hand
    sv = PyRoundSolver(joker_state=True)
    sg, r = _solver_round(sv, g)
    pos = tuple(range(discard_cards))
    sv._advance_discard(r, pos)
    j = sg.jokers[0]
    expect = g.clone()
    jj = expect.jokers[0]
    jj.d.discard(expect, jj, [expect.hand[i] for i in pos])
    want = jj.state.get("val") if jj in expect.jokers else {"ramen": 1.0}.get(key, 0.0)
    assert r.st[0][0] == pytest.approx(float(want)), (key, r.st[0][0], want)


@pytest.mark.parametrize("key,state", [
    ("green_joker", {"val": 2}), ("ride_the_bus", {"val": 3}), ("square", {"val": 8}), ("runner", {"val": 15}),
    ("loyalty_card", {"val": 1}), ("wee", {"val": 8}), ("ice_cream", {"val": 100}),
])
def test_play_state_matches_the_real_hooks(key, state):
    g = _white(5, [(key, state)])
    g.hand = [Card(2, 1), Card(2, 0), Card(3, 1), Card(4, 2), Card(5, 3), Card(13, 1), Card(7, 0), Card(9, 2)]
    sv = PyRoundSolver(joker_state=True)
    sg, r = _solver_round(sv, g)
    pos = (0, 1, 2, 3)                                           # 2 2 3 4: a Square Joker hand with two 2s
    sv._advance_play(r, pos, 1)
    expect = g.clone()
    played = [expect.hand[i] for i in pos]
    held = [c for i, c in enumerate(expect.hand) if i not in pos]
    score_hand(expect, played, held, None, True, Plan(expect))      # commit: the before / card hooks write back
    jj = expect.jokers[0]
    if jj.d.after:
        jj.d.after(expect, jj)
    want = jj.state.get("val") if jj in expect.jokers else 0.0
    assert r.st[0][0] == pytest.approx(float(want)), (key, r.st[0][0], want)
    assert r.st[3][1] == g.hand_played_round[1] + 1 and r.st[2][1] == g.hand_played[1] + 1


def test_green_joker_raises_the_estimate_over_a_round():
    """With Green Joker at 0 and a target no single hand reaches, the stateful playouts (each hand +1 Mult)
    expect more chips than the stateless ones."""
    g = _white(8, [("green_joker", {"val": 0})])
    g.target = 10 ** 9                                             # never cleared: compare expected chips
    ch_a = enumerate_candidates(World(g), random.Random(0))
    S.RoundSolver(samples=8, joker_state=True).solve(g, ch_a, random.Random(1))
    ch_b = enumerate_candidates(World(g), random.Random(0))
    S.RoundSolver(samples=8, joker_state=False).solve(g, ch_b, random.Random(1))
    ea = {str(c.action): c.e_chips for c in ch_a.cands if c.kind == "play" and c.e_chips >= 0}
    eb = {str(c.action): c.e_chips for c in ch_b.cands if c.kind == "play" and c.e_chips >= 0}
    common = set(ea) & set(eb)
    assert common and all(ea[k] >= eb[k] for k in common) and any(ea[k] > eb[k] for k in common)


def test_forbidden_hand_changes_no_state():
    g = _white(9, [("green_joker", {"val": 2})], boss="psychic")
    sv = PyRoundSolver(joker_state=True)
    sg, r = _solver_round(sv, g)
    sv._advance_play(r, (0, 1), 1)                                 # two cards: The Psychic forbids it
    assert r.st[0][0] == 2.0 and r.st[3][1] == 1                   # no +1 Mult; still a hand played
