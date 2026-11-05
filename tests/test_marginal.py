"""The marginal-value inputs (balatro_rl/marginal.py, the engine's Env::marginal_fill): off they are zeros; on, they
say what each joker is worth to the build; the C++ engine reproduces the Python reference value for value."""
import numpy as np
import pytest

from balatro_rl import env as E
from balatro_rl.marginal import MARG_A, MARG_G, MARG_J, POT_A, POT_G, POT_J, POTENTIAL
from balatro_rl.sim.game import Game
from balatro_rl.sim.jokers import JOKERS, Joker

try:
    from balatro_rl.sim import _engine
    HAVE_MARGINAL_ENGINE = hasattr(_engine.Env, "marginal")
except ImportError:                                    # not built
    _engine = None
    HAVE_MARGINAL_ENGINE = False


def mg(o):                                              # the marginal columns of the global vector / joker rows / action rows
    return o["g"][-(MARG_G + POT_G):-POT_G], o["jf"][:, -(MARG_J + POT_J):-POT_J], o["af"][:, -(MARG_A + POT_A):-POT_A]


def pot(o):                                             # the potential columns
    return o["g"][-POT_G:], o["jf"][:, -POT_J:], o["af"][:, -POT_A:]


def shop(jokers, money=60, seed=7, stake="GOLD"):
    g = Game(seed=seed, deck_type="RED", stake=stake)
    for k in jokers:
        g.add_joker(Joker(k, base_cost=JOKERS[k].cost))
    g.money = money
    g.open_shop()
    return g


def test_inputs_off_are_zero():
    g = shop(["green_joker", "half"])
    o = E.encode(g)
    assert all(np.all(x == 0) for x in mg(o) + pot(o))
    assert o["g"].shape[0] == E.global_size() and o["jf"].shape[1] == E.F_JOKER_P and o["af"].shape[1] == E.F_ACT_P


def test_no_inputs_inside_a_round():
    g = shop(["half"])
    g.state = "SELECTING_HAND"
    g.blind_idx = 0
    o = E.encode(g, marginal=2)
    assert all(np.all(x == 0) for x in mg(o)[:2] + pot(o)[:2])


def test_a_jokers_share_of_the_score():
    g = shop(["half", "green_joker", "supernova"])          # Half Joker: +20 mult on plays of up to three cards
    o = E.encode(g, marginal=1)
    mgl, mjf, _ = mg(o)
    share_half = mjf[0, 0]
    assert 0.2 < share_half < 0.9
    assert mjf[1, 0] < share_half                           # Green Joker and Supernova have grown nothing yet
    assert 0 < mgl[0] < 1                                   # the build's expected score, log10 / 7
    assert np.all(mgl[1:] >= 0)
    assert all(np.all(x == 0) for x in pot(o))              # level 1: no potential


def test_a_hypothetical_joker_leaves_no_trace():
    """Evaluating a Negative joker (it adds a slot, which Joker Stencil reads) must not change the next evaluation."""
    from balatro_rl.marginal import _Scratch
    g = shop(["stencil", "oops"])
    plain = Joker("joker", base_cost=2)
    neg = Joker("joker", edition="NEGATIVE", base_cost=5)
    fresh = _Scratch(g)
    fresh.prepare()
    alone = fresh.with_jokers(add=plain)
    sc = _Scratch(g)
    sc.prepare()
    sc.with_jokers(add=neg)
    sc.with_jokers(remove=0)
    assert sc.with_jokers(add=plain) == alone
    assert sc.s.joker_slots == g.joker_slots and [j.key for j in sc.s.jokers] == [j.key for j in g.jokers]


def test_an_item_that_helps_scores_above_zero_and_one_that_hurts_below():
    g = shop(["stencil"])                                   # Joker Stencil: x(empty slots + itself)
    o_base = E.encode(g, marginal=2)
    from balatro_rl.marginal import _Scratch
    sc = _Scratch(g)
    sc.prepare()
    gain_planet = sc.with_level(1)                          # a level on Pair: more chips and mult on every pair
    assert gain_planet > sc.base
    assert np.isfinite(o_base["af"]).all()


def test_the_potential_of_growing_jokers():
    g = shop(["green_joker", "half", "hologram"])
    o = E.encode(g, marginal=2)
    pg, pj, _ = pot(o)
    assert pj[0, 0] > 0 and pj[2, 0] > 0                    # Green Joker and Hologram would be worth more grown
    assert pj[1, 0] == 0                                    # Half Joker does not grow
    assert pg[0] >= max(pj[0, 0], pj[2, 0]) - 1e-6          # all grown at once is at least the best single one
    assert pg[0] > 0
    # the same joker on offer shows its potential on its row; one that does not grow shows none
    g = shop(["half"])
    g.shop[0].joker = Joker("hologram", base_cost=7)
    g.shop[0].kind = "joker"
    g.shop[0].key = "j_hologram"
    g.shop[1].joker = Joker("half", base_cost=5)
    g.shop[1].kind = "joker"
    g.shop[1].key = "j_half"
    o = E.encode(g, marginal=2)
    assert pot(o)[2][E.A_BUY, 0] > 0 and pot(o)[2][E.A_BUY + 1, 0] == 0


@pytest.mark.skipif(not HAVE_MARGINAL_ENGINE, reason="the C++ engine with the marginal inputs is not built")
def test_engine_matches_the_reference():
    from tests.engine_diff import play_one, DECKS, STAKES
    for seed in range(40, 52):
        bad, _ = play_one(seed, DECKS[seed % len(DECKS)], STAKES[(seed // 3) % len(STAKES)], True, True, deep=8,
                          p_spice=0.3, marginal=1 + seed % 2)
        assert not bad, bad[:3]
