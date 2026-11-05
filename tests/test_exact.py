"""Exact-seed mode: the Python game and the C++ engine draw the shop, packs, bosses, tags and the deck shuffles from the real game's
named streams and stay identical (tests/engine_diff.py --exact on more games)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_diff                                                      # noqa: E402

try:
    from balatro_rl.sim import _engine
    HAVE = hasattr(_engine.Env, "exact")
except ImportError:
    _engine = None
    HAVE = False


@pytest.mark.skipif(not HAVE, reason="the C++ engine with the exact-seed mode is not built")
@pytest.mark.parametrize("seed,deck,stake", [(1, "BLUE", "WHITE"), (2, "RED", "GOLD"), (3, "YELLOW", "ORANGE"), (4, "ABANDONED", "GREEN"),
                                              (5, "CHECKERED", "GOLD"), (6, "MAGIC", "BLACK"), (7, "GHOST", "RED"), (8, "ZODIAC", "GOLD")])
def test_python_and_cpp_agree_in_exact_mode(seed, deck, stake):
    bad, _ = engine_diff.play_one(seed, deck, stake, do_spice=True, check_obs=True, exact=True, deep=6)
    assert not bad, bad[:3]
