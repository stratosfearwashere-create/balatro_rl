"""The C++ core (sim/cpp, built by python setup_cython.py build): each ported piece gives exactly what the
Python code gives. Random: CPython's random.Random bit for bit; evaluate: hands.evaluate on random cards;
the solver core: the compiled solver's P(clear) and expected chips on the regression states, with the
agent's random generator left in the same state."""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from balatro_rl.sim import hands  # noqa: E402
from balatro_rl.sim.cards import Card  # noqa: E402

try:
    from balatro_rl.sim import _core
except ImportError:
    _core = None

pytestmark = pytest.mark.skipif(_core is None or os.environ.get("BALATRO_PURE") == "1",
                                reason="the C++ core is not built")


def test_random_matches_cpython():
    for seed in (0, 1, 12345, 2 ** 40 + 7, 70001):
        a, b = random.Random(seed), _core.Random(seed)
        assert a.getstate() == b.getstate()
        for _ in range(2000):
            op = a.randrange(0, 9)
            assert op == b.randrange(0, 9)
            if op == 0:
                assert a.random() == b.random()
            elif op == 1:
                assert a.getrandbits(53) == b.getrandbits(53)
            elif op == 2:
                assert a.randint(-5, 100) == b.randint(-5, 100)
            elif op == 3:
                n = a.randrange(1, 70)
                assert n == b.randrange(1, 70)
                assert a.choice(range(n)) == b.choice(range(n))
            elif op == 4:
                x = list(range(a.randrange(0, 60)))
                y = list(x)
                assert b.randrange(0, 60) == len(x)
                a.shuffle(x)
                b.shuffle(y)
                assert x == y
            elif op == 5:
                n = a.randrange(1, 80)
                assert n == b.randrange(1, 80)
                k = a.randrange(0, n + 1)
                assert k == b.randrange(0, n + 1)
                assert a.sample(range(n), k) == b.sample(range(n), k)
            elif op == 6:
                w = [a.random() for _ in range(5)]
                assert w == [b.random() for _ in range(5)]
                assert a.choices(range(5), weights=w, k=3) == b.choices(range(5), weights=w, k=3)
                assert a.choices(range(5), k=4) == b.choices(range(5), k=4)
            elif op == 7:
                assert a.uniform(1.5, 9.5) == b.uniform(1.5, 9.5)
            else:
                assert a.getrandbits(7) == b.getrandbits(7)
        assert a.getstate() == b.getstate()
        c = _core.Random(1)
        c.setstate(a.getstate())
        assert c.random() == a.random()


def test_evaluate_matches_python():
    rng = random.Random(5)
    enh = ["", "", "", "BONUS", "MULT", "WILD", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY", "HIDDEN"]
    for _ in range(30000):
        cards = [Card(rng.randrange(2, 15), rng.randrange(4), enh=rng.choice(enh)) for _ in range(rng.randint(1, 5))]
        for c in cards:
            c.debuffed = rng.random() < 0.1
        ff, sc, sm = rng.random() < 0.3, rng.random() < 0.3, rng.random() < 0.3
        a = hands._py_evaluate(cards, ff, sc, sm)
        b = hands.evaluate(cards, ff, sc, sm)
        assert (a.hand, a.scoring, a.contains) == (b.hand, b.scoring, b.contains)


def test_solver_core_matches_compiled_solver():
    from balatro_rl.az import solver as S
    from balatro_rl.az.actions import enumerate_candidates
    from balatro_rl.az.world import World
    import test_regression as T
    if not S.CORE:
        pytest.skip("the solver core is not built")

    def rows(ch):
        return [(str(c.action), repr(c.p_clear), repr(c.e_chips)) for c in ch.cands]
    n = 0
    for i, g in enumerate(T.states()):
        for depth in (1, 2):
            a = enumerate_candidates(World(g), random.Random(i))
            b = enumerate_candidates(World(g), random.Random(i))
            ra, rb = random.Random(100 + i), random.Random(100 + i)
            S.CompiledRoundSolver(samples=6).solve(g, a, ra, depth=depth)
            S.CoreRoundSolver(samples=6).solve(g, b, rb, depth=depth)
            assert rows(a) == rows(b), (i, g.boss, depth)
            assert ra.getstate() == rb.getstate()
            n += 1
    assert n == 60
