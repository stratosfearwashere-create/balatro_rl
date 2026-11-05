"""Balatro's seeded random numbers: LuaJIT's math.random (the tw223 generator) and the game's pseudohash / pseudoseed /
pseudorandom / pseudoshuffle on top of it. Written from the algorithms (LuaJIT 2.1 lj_prng, Balatro functions/misc_functions.lua)
and NOT yet checked against the real thing: tests/test_seedrng.py compares it with LuaJIT itself when `lupa` is installed.

Every random thing in the game asks for a named stream, e.g. pseudorandom('Joker1' + source + ante). The key's state
advances only when that key is used, so what a shop shows depends on the seed, the ante and the key, not on how many
other random numbers earlier actions used (unlike this repository's single-stream Game.rng).
"""
import math
import struct

M64 = (1 << 64) - 1


def _d2u(d: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", d))[0]


def _u2d(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & M64))[0]


class LuaRandom:
    """LuaJIT 2.1's PRNG state: four 64-bit words, advanced by the Tausworthe step of L'Ecuyer's tw223."""
    # (word, k, q, s) of the four component generators
    PARAMS = ((0, 63, 31, 18), (1, 58, 19, 28), (2, 55, 24, 7), (3, 47, 21, 8))

    def __init__(self, seed: float = 0.0):
        self.u = [0, 0, 0, 0]
        self.seed(seed)

    def _step(self) -> int:
        r = 0
        for i, k, q, s in self.PARAMS:
            z = self.u[i]
            z = ((((z << q) & M64) ^ z) >> (k - s)) ^ (((z & ((M64 << (64 - k)) & M64)) << s) & M64)
            r ^= z
            self.u[i] = z
        return r

    def seed(self, d: float):
        """math.randomseed(d): the words are made from d by an affine map, then ten steps are discarded."""
        r = 0x11090601                                  # 64 - k for k = 63, 58, 55, 47
        for i in range(4):
            m = 1 << (r & 255)
            r >>= 8
            d = d * 3.14159265358979323846 + 2.7182818284590452354
            u = _d2u(d)
            if u < m:
                u += m
            self.u[i] = u
        for _ in range(10):
            self._step()

    def random(self) -> float:
        """math.random(): a double in [0, 1)."""
        return _u2d((self._step() & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000) - 1.0

    def randint(self, lo: int, hi: int | None = None) -> int:
        """math.random(m, n) (or math.random(n) for 1..n)."""
        if hi is None:
            lo, hi = 1, lo
        return int(self.random() * (hi - lo + 1)) + lo


def pseudohash(s: str) -> float:
    num = 1.0
    for i in range(len(s), 0, -1):
        num = ((1.1239285023 / num) * ord(s[i - 1]) * math.pi + math.pi * i) % 1.0
    return num


class Streams:
    """G.GAME.pseudorandom: the per-key states of one run and the run's seed."""

    def __init__(self, seed: str):
        self.seed = seed
        self.state: dict[str, float] = {}
        self.hashed_seed = pseudohash(seed)
        self.rng = LuaRandom()                          # the global math.random state (seeded on every pseudorandom call)

    def pseudoseed(self, key: str) -> float:
        if key not in self.state:
            self.state[key] = pseudohash(key + self.seed)
        v = (2.134453429141 + self.state[key] * 1.72431234) % 1.0
        self.state[key] = abs(float("%.13f" % v))
        return (self.state[key] + self.hashed_seed) / 2

    def pseudorandom(self, key, lo: int | None = None, hi: int | None = None):
        """pseudorandom(key[, min, max]): key is a stream name (str) or an already derived seed (float)."""
        s = self.pseudoseed(key) if isinstance(key, str) else key
        self.rng.seed(s)
        return self.rng.randint(lo, hi) if lo is not None and hi is not None else self.rng.random()

    def shuffle(self, items: list, key=None):
        """pseudoshuffle(list, seed): Fisher-Yates from the end with math.random(i) (the caller sorts by sort_id first)."""
        if key is not None:
            self.rng.seed(self.pseudoseed(key) if isinstance(key, str) else key)
        for i in range(len(items), 1, -1):
            j = self.rng.randint(i)
            items[i - 1], items[j - 1] = items[j - 1], items[i - 1]


class ExactRng:
    """What scoring and effects see as `rng` in exact mode: keyed draws on the game's named streams (`keyed`, `keyed_int`), and the
    plain generator for anything not keyed (yet)."""

    def __init__(self, streams, fallback):
        self.streams = streams
        self.fb = fallback

    def keyed(self, key: str) -> float:
        return self.streams.pseudorandom(key)

    def keyed_int(self, key: str, lo: int, hi: int) -> int:
        return self.streams.pseudorandom(key, lo, hi)

    def __getattr__(self, name):
        return getattr(self.fb, name)


_SEED_ALPHABET = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"


def seed_string(n: int) -> str:
    """The 8-character seed of a run from an integer (the same mapping as the C++ engine's seed_string)."""
    out = []
    for _ in range(8):
        out.append(_SEED_ALPHABET[n % 34])
        n //= 34
    return "".join(out)
