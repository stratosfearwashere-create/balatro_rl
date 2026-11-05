"""balatro_rl.sim.seedrng: self-consistency always, and agreement with real LuaJIT when `lupa` (bundles LuaJIT) is installed."""
import pytest

from balatro_rl.sim.seedrng import LuaRandom, Streams, pseudohash

try:
    import lupa.luajit21 as lupa_lj
except ImportError:
    try:
        import lupa as lupa_lj
    except ImportError:
        lupa_lj = None


def test_random_is_in_unit_interval_and_reproducible():
    a, b = LuaRandom(0.3), LuaRandom(0.3)
    xs = [a.random() for _ in range(1000)]
    assert xs == [b.random() for _ in range(1000)]
    assert all(0.0 <= x < 1.0 for x in xs)
    assert 0.4 < sum(xs) / len(xs) < 0.6


def test_randint_bounds():
    r = LuaRandom(0.7)
    vals = [r.randint(3, 9) for _ in range(2000)]
    assert min(vals) == 3 and max(vals) == 9
    assert {r.randint(5) for _ in range(500)} == {1, 2, 3, 4, 5}


def test_streams_are_independent_of_other_keys():
    a, b = Streams("ABCD1234"), Streams("ABCD1234")
    b.pseudorandom("shop_pack1")                        # an unrelated key used in b only
    b.pseudorandom("cdt1")
    assert [a.pseudorandom("Joker1sho1", 1, 100) for _ in range(5)] == [b.pseudorandom("Joker1sho1", 1, 100) for _ in range(5)]


def test_streams_differ_by_seed_and_key():
    a, b = Streams("AAAAAAAA"), Streams("AAAAAAAB")
    assert a.pseudorandom("x") != b.pseudorandom("x")
    assert Streams("AAAAAAAA").pseudorandom("x") != Streams("AAAAAAAA").pseudorandom("y")


def test_shuffle_is_a_permutation():
    s = Streams("ZZZZ")
    items = list(range(52))
    s.shuffle(items, "shuffle1")
    assert sorted(items) == list(range(52)) and items != list(range(52))


LUA = r"""
function pseudohash(str)
  local num = 1
  for i = #str, 1, -1 do
    num = ((1.1239285023 / num) * string.byte(str, i) * math.pi + math.pi * i) % 1
  end
  return num
end
local P = {}
local hashed
function setup(seed) P = {}; hashed = pseudohash(seed); P.seed = seed end
function pseudoseed(key)
  if not P[key] then P[key] = pseudohash(key .. P.seed) end
  P[key] = math.abs(tonumber(string.format("%.13f", (2.134453429141 + P[key] * 1.72431234) % 1)))
  return (P[key] + hashed) / 2
end
function pseudorandom(key, lo, hi)
  math.randomseed(pseudoseed(key))
  if lo and hi then return math.random(lo, hi) end
  return math.random()
end
function raw(seed, n) math.randomseed(seed); local t = {}; for i = 1, n do t[i] = math.random() end; return t end
"""


@pytest.mark.skipif(lupa_lj is None, reason="lupa (LuaJIT) not installed: the generator is unverified against the real one")
def test_matches_luajit():
    lua = lupa_lj.LuaRuntime(unpack_returned_tuples=True)
    lua.execute(LUA)
    g = lua.globals()
    for seed in (0.123, 0.987654321, 3.5, 1234.5678):
        ref = list(g.raw(seed, 50).values())
        r = LuaRandom(seed)
        assert [r.random() for _ in range(50)] == ref, seed
    for s in ("ABCDEFGH", "7LB2WVPK", "1"):
        assert pseudohash(s) == g.pseudohash(s)
    g.setup("7LB2WVPK")
    st = Streams("7LB2WVPK")
    for key, lo, hi in (("Joker1sho1", 1, 150), ("shuffle1", None, None), ("rarity1sho", None, None), ("Joker1sho1", 1, 150)):
        assert st.pseudorandom(key, lo, hi) == g.pseudorandom(key, lo, hi)
