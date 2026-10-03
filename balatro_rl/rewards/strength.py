"""Build strength: how likely the current build is to clear the bosses ahead. Calculated, not learned.

    rep = Strength().report(g)
    rep["clear"]      clear chance of this ante's boss, the next three antes' and ante 8's   (HORIZON)
    rep["survives"]   "survives through ante": the last ante, counting on from this one, whose boss the build
                      clears with a chance of at least 50% (ante - 1 when it would not clear this one)
    rep["score"]      the scalar strength score, in [0, 1] (config `score`)

How:
  - Probe: the game as the first hand of a fresh round (potential.fresh_round): full deck, hands and discards
    at their per-round values, round-dependent jokers at their start-of-round values, no boss effect.
  - K hands are sampled from the full deck and the best play of each is scored by the compiled scorer in one
    batched call: exactly the hands and scores of the headroom (Headroom.best_scores).
  - Whole rounds are bootstrapped: B times, `hands per round` of the K best-hand scores are drawn with
    replacement and summed; P(clear) = share of the B totals that reach the target.
    Discards (config `discards`): "none" ignores them; "extra_draws" draws hands + discard_weight * discards
    scores and keeps the best `hands` of them (a discard is another look at the deck). `score_scale`
    multiplies every score. Both are corrections fitted against the round solver and real games
    (rewards/strength_check.py); what is still not modelled: deck depletion within a round, keeping cards
    across a discard, jokers that grow within the round, and every boss effect.
  - Targets: this ante's boss is known, so its real requirement is used (g.blind_target(2): The Wall x4,
    The Needle x1, Violet Vessel x6); later antes use the standard boss (x2 the ante's base, x2 again on the
    Plasma deck).
  - Growth (config `growth`, off by default): each scaling joker's `val` is moved forward by
    table[key] * (1 + d + d^2 + ... over the rounds until that boss) before scoring (rewards/growth.py).
  - Determinism and leakage: the K hands and the bootstrap are both seeded from a hash of the
    scoring-relevant state (potential.scoring_key), the result is cached on that state and the targets, and
    only the deck's contents are read, in a canonical order: never the draw order or the game's RNG.

The scalar score, with p_a the clear chance of ante a's boss and c the current ante:
    horizon          sum_{a=c..min(c+3,8)} prod_{b=c..a} p_b / (number of terms): the expected share survived
                     of this ante and the next three (the default: the best spread and calibration of the
                     four on real games, STAGE2_REPORT.md)
    expected_antes   the same over all the antes left, a = c..8, divided by 9 - c
    expected_total   (c - 1 + sum_{a=c..8} prod_{b=c..a} p_b) / 8   antes behind + expected antes survived
    current          p_c
All are in [0, 1] (the headroom term they can replace in Phi is tanh, in [-1, 1]).
"""
from __future__ import annotations

import time

import numpy as np

from ..sim.game import Game
from ..sim.items import BLIND_BASE
from .config import PotentialConfig, StrengthConfig
from .potential import Headroom, _seed, scoring_key

LAST_ANTE = 8
HORIZON = (0, 1, 2, 3)                     # antes ahead reported, plus ante 8
N_CLEAR = len(HORIZON) + 1
N_SURVIVE = LAST_ANTE + 1                  # classes of "survives through ante": 0 .. 8
SCORES = ("expected_antes", "expected_total", "horizon", "current")
DISCARDS = ("none", "extra_draws")


def current_ante(g: Game) -> int:
    return min(max(g.ante, 1), LAST_ANTE)


def boss_targets(g: Game) -> list[int]:
    """Chips required by the boss of each ante from the current one to ante 8: the current ante's real boss
    (with its size multiplier), the standard boss for the later ones."""
    a = current_ante(g)
    base = BLIND_BASE[g.scaling()]
    mult = 2 * (2 if g.deck_type == "PLASMA" else 1)
    return [g.blind_target(2)] + [int(base[b - 1] * mult) for b in range(a + 1, LAST_ANTE + 1)]


_INDEX_CACHE: dict = {}


def _draw_indices(rng, n: int, rows: int, cols: int) -> np.ndarray:
    """rng.integers(0, n, size=(rows, cols)), memoised by the generator's state: every strength evaluation
    of a shop decision draws from a fresh generator with the decision's seed, so the same rounds are drawn
    again and again. A generator in the same state draws the same numbers, so the cache is exact; the
    generator is left advanced as if it had drawn."""
    key = _state_key(rng)
    if key is None:
        return rng.integers(0, n, size=(rows, cols))
    key = key + (n, rows, cols)
    got = _INDEX_CACHE.get(key)
    if got is None:
        if len(_INDEX_CACHE) >= 4096:
            _INDEX_CACHE.clear()
        idx = rng.integers(0, n, size=(rows, cols))
        _INDEX_CACHE[key] = got = (idx, rng.bit_generator.state)
    else:
        rng.bit_generator.state = got[1]
    return got[0]


def _state_key(rng) -> tuple | None:
    try:
        st = rng.bit_generator.state
        if st.get("bit_generator") != "PCG64" or st.get("has_uint32", 0) != 0:
            return None
        return (st["state"]["state"], st["state"]["inc"])
    except (AttributeError, KeyError, TypeError):
        return None


def round_totals(scores, hands: int, discards: int, cfg: StrengthConfig, rng: np.random.Generator) -> np.ndarray:
    """B bootstrapped whole-round totals from the best-hand scores of K sampled hands."""
    s = np.asarray(scores, dtype=float) * cfg.score_scale
    if cfg.discards == "none":
        extra = 0
    elif cfg.discards == "extra_draws":
        extra = max(0, int(round(cfg.discard_weight * discards)))
    else:
        raise ValueError(f"unknown strength discards model {cfg.discards!r}")
    hands = max(1, int(hands))
    if len(s) == 0:
        return np.zeros(cfg.bootstrap)
    draws = s[_draw_indices(rng, len(s), cfg.bootstrap, hands + extra)]
    if extra:
        draws = np.sort(draws, axis=1)[:, extra:]             # the best `hands` of each row
    return draws.sum(axis=1)


def clear_chances(score_sets, hands: int, discards: int, targets, cfg: StrengthConfig, seed: int) -> list[float]:
    """P(round total >= target) for each target. `score_sets`: one list of K best-hand scores for all the
    targets, or one list per target (growth). The same resampled rounds are used for every target."""
    per_target = len(score_sets) > 0 and isinstance(score_sets[0], (list, tuple, np.ndarray))
    totals = {}
    out = []
    for i, t in enumerate(targets):
        s = score_sets[i] if per_target else score_sets
        tot = totals.get(id(s))
        if tot is None:
            tot = totals[id(s)] = round_totals(s, hands, discards, cfg, np.random.default_rng(seed))
        out.append(float((tot >= t).mean()))
    return out


def survives_through(ante: int, chances) -> int:
    """The last ante, counting on from `ante`, whose clear chance is at least 50%; ante - 1 if the first
    one is already below."""
    last = ante - 1
    for p in chances:
        if p < 0.5:
            break
        last += 1
    return min(last, LAST_ANTE)


def strength_score(ante: int, chances, kind: str) -> float:
    """The scalar strength score from the clear chances of antes `ante` .. 8 (module doc); in [0, 1]."""
    p = np.asarray(chances, dtype=float)
    if kind == "current":
        return float(p[0])
    alive = np.cumprod(p)                                     # P(still alive after ante a's boss)
    if kind == "expected_antes":
        return float(alive.sum() / len(p))
    if kind == "expected_total":
        return float((ante - 1 + alive.sum()) / LAST_ANTE)
    if kind == "horizon":
        n = min(len(p), len(HORIZON))
        return float(alive[:n].sum() / n)
    raise ValueError(f"unknown strength score {kind!r}")


def horizon_antes(ante: int) -> list[int]:
    """The antes whose clear chance is reported: ante, +1, +2, +3 (capped at 8) and ante 8."""
    return [min(ante + k, LAST_ANTE) for k in HORIZON] + [LAST_ANTE]


def summarize(ante: int, targets, chances, cfg: StrengthConfig) -> dict:
    hz = horizon_antes(ante)
    return {"ante": ante, "antes": hz, "targets": [int(targets[a - ante]) for a in hz],
            "clear": [float(chances[a - ante]) for a in hz], "clear_all": [float(p) for p in chances],
            "survives": survives_through(ante, chances), "score": strength_score(ante, chances, cfg.score)}


class Strength:
    """The strength report of a game state, cached. `stats`: calls, cache hits, seconds spent computing."""

    def __init__(self, cfg: StrengthConfig | None = None, max_cache: int = 50_000):
        self.cfg = cfg or StrengthConfig()
        if self.cfg.score not in SCORES:
            raise ValueError(f"unknown strength score {self.cfg.score!r}")
        if self.cfg.discards not in DISCARDS:
            raise ValueError(f"unknown strength discards model {self.cfg.discards!r}")
        self._sampler = Headroom(PotentialConfig(headroom_samples=self.cfg.samples))
        self.cache: dict = {}
        self.max_cache = max_cache
        self.stats = {"calls": 0, "hits": 0, "seconds": 0.0}
        self._table = None

    # ------------------------------------------------------------------ pieces
    def table(self) -> dict:
        if self._table is None:
            from .growth import load_table
            self._table = load_table(self.cfg.growth_table) if self.cfg.growth else {}
        return self._table

    def best_scores(self, g: Game, key: tuple | None = None, rounds_ahead: int = 0) -> list[float]:
        """Best-play score of each of the K sampled fresh-round hands; with rounds_ahead > 0 and growth on,
        the scaling jokers are first moved that many rounds forward."""
        seed = _seed(scoring_key(g) if key is None else key)
        if rounds_ahead > 0 and self.cfg.growth:
            from .growth import project
            g = project(g, self.table(), rounds_ahead, self.cfg.growth_discount)
        return self._sampler.best_scores(g, seed=seed)

    def _growing(self, g: Game) -> bool:
        if not self.cfg.growth:
            return False
        t = self.table()
        return any(j.key in t and isinstance(j.state.get("val"), (int, float)) for j in g.jokers)

    def _key(self, g: Game, targets) -> tuple:
        sk = scoring_key(g)
        return sk, (sk, tuple(targets), min(g.blind_idx, 2) if self._growing(g) else -1)

    def _cached(self, key, compute):
        self.stats["calls"] += 1
        got = self.cache.get(key)
        if got is not None:
            self.stats["hits"] += 1
            return got
        t = time.perf_counter()
        val = compute()
        self.stats["seconds"] += time.perf_counter() - t
        if len(self.cache) >= self.max_cache:
            self.cache.clear()
        self.cache[key] = val
        return val

    # ------------------------------------------------------------------ public
    def report(self, g: Game) -> dict:
        targets = boss_targets(g)
        sk, key = self._key(g, targets)
        return self._cached(key, lambda: self._compute(g, sk, targets))

    def value(self, g: Game) -> float:
        return self.report(g)["score"]

    def clear_chance(self, g: Game, target: float) -> float:
        """P(a fresh round's total >= target) for any target (calibration; no growth)."""
        sk = scoring_key(g)
        return self._cached((sk, "target", target), lambda: clear_chances(
            self.best_scores(g, sk), g.round_hands(), g.round_discards(), [target], self.cfg,
            _seed(("bootstrap", sk)))[0])

    def _compute(self, g: Game, sk: tuple, targets) -> dict:
        ante = current_ante(g)
        if self._growing(g):                      # one scoring pass per ante: the jokers as they will be
            first = max(0, 2 - min(g.blind_idx, 2))            # rounds played before this ante's boss
            sets = [self.best_scores(g, sk, first + 3 * k) for k in range(len(targets))]
        else:
            sets = self.best_scores(g, sk)
        chances = clear_chances(sets, g.round_hands(), g.round_discards(), targets, self.cfg,
                                _seed(("bootstrap", sk)))
        return summarize(ante, targets, chances, self.cfg)
