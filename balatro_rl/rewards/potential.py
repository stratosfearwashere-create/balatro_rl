"""The potential Phi(s) = w_head * Phi_headroom(s) + w_prog * Phi_prog(s); Phi(terminal) = 0.

Phi_headroom = tanh(headroom / scale), headroom = log(E[best-hand score]) - log(target of the next boss).
  - Hands: K samples of a hand (current hand size) from the full deck, the build's cards whatever the
    current draw pile holds. Only the deck's contents are read, in a canonical order, so the true draw
    order cannot matter.
  - Scoring: the simulator's own scorer (Game.predict_many, as for real plays), in the context of the first
    hand of a fresh round: hands and discards at their per-round values, nothing played this round, no
    boss effect, no debuffed or face-down cards. So Mystic Summit, Banner, Acrobat, Dusk, Card Sharp and
    other round-dependent jokers read as they would at the start of a round; jokers' runtime state
    (Green Joker's mult, Ride the Bus ...), hand levels, enhancements, editions and seals all count.
  - Target: the current ante's boss is always known (the simulator draws it when the ante starts, and the
    game shows it), so its real requirement is used (The Wall counts x4, The Needle x1); its effect on
    scoring is ignored.
  - Randomness: the K samples are seeded from a hash of the scoring-relevant state, so the same build
    always gets the same Phi (sampling noise would otherwise be shaping reward by itself), and results are
    cached on that state.
  - `aggregate`: "mean" uses log(mean score) (the same as log-mean-exp of the log scores); "geomean" uses
    the mean of the log scores. `round_hands` compares E[best hand] x hands per round with the target.
Phi_prog = furthest blind beaten / 24 (blinds replayed after Hieroglyph / Petroglyph don't count).
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from itertools import combinations

from ..sim import fastscore
from ..sim.cards import Card
from ..sim.game import Game
from ..sim.hands import N_HANDS
from ..sim.scoring import Plan
from .config import PotentialConfig

_SUBS = {}


def _subsets(n: int):
    s = _SUBS.get(n)
    if s is None:
        s = _SUBS[n] = [c for k in range(1, min(5, n) + 1) for c in combinations(range(n), k)]
    return s


def _card_sig(c: Card) -> tuple:
    return (c.rank, c.suit, c.enh, c.edition, c.seal, c.extra_chips)


def scoring_key(g: Game) -> tuple:
    """Everything the fresh-round scorer reads (and nothing it doesn't, e.g. the draw order or the target).
    Seeds the samples, so the same build always gets the same sampled hands."""
    jok = tuple((j.key, j.edition, j.debuffed, j.sell_value(), j.eternal,
                 tuple(sorted((k, repr(v)) for k, v in j.state.items() if not k.startswith("_"))))
                for j in g.jokers)
    return (tuple(sorted(_card_sig(c) for c in g.full_deck)), jok, tuple(g.hand_levels), tuple(g.hand_played),
            tuple(sorted(g.vouchers)), tuple(sorted(c.key for c in g.consumables)), g.money,
            fresh_hand_size(g), g.round_hands(), g.round_discards(), g.tarots_used, g.blinds_skipped,
            g.joker_slots, g.starting_deck_size, g.deck_type)


def fresh_hand_size(g: Game) -> int:
    """Hand size at the start of a round, without any boss's effect (The Manacle) or Juggle Tag."""
    saved = (g.state, g.juggle, g.hand_size_override)
    g.state, g.juggle = "SHOP", 0
    try:
        return g.effective_hand_size()
    finally:
        g.state, g.juggle, g.hand_size_override = saved


def next_boss_target(g: Game) -> int:
    """Chips required by the current ante's boss (known), with its real size multiplier."""
    return g.blind_target(2)


def best_play(probe: Game, plan: Plan, hand: list) -> tuple[float, tuple]:
    """(score, positions) of the best play from `hand`, scored by Game.predict_many (the simulator's scorer)."""
    subs = _subsets(len(hand))
    if not subs:
        return 0.0, ()
    preds = probe.predict_many(subs, plan, hand)
    i = max(range(len(subs)), key=lambda k: preds[k][0])
    return preds[i][0], subs[i]


def _seed(key: tuple) -> int:
    return int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], "little")


class Headroom:
    """Phi_headroom with a cache. `stats` holds calls, cache hits and seconds spent computing."""

    def __init__(self, cfg: PotentialConfig | None = None, max_cache: int = 50_000):
        self.cfg = cfg or PotentialConfig()
        self.cache: dict = {}
        self.max_cache = max_cache
        self.stats = {"calls": 0, "hits": 0, "seconds": 0.0}

    def raw(self, g: Game) -> float:
        """headroom = log(E[best-hand score]) - log(next boss target), before tanh."""
        self.stats["calls"] += 1
        key = (scoring_key(g), next_boss_target(g))
        got = self.cache.get(key)
        if got is not None:
            self.stats["hits"] += 1
            return got
        t = time.perf_counter()
        val = self._compute(g, key)
        self.stats["seconds"] += time.perf_counter() - t
        if len(self.cache) >= self.max_cache:
            self.cache.clear()
        self.cache[key] = val
        return val

    def value(self, g: Game) -> float:
        return math.tanh(self.raw(g) / self.cfg.headroom_scale)

    def best_scores(self, g: Game, samples: int | None = None, seed: int | None = None) -> list[float]:
        """Best-play score for each of the K sampled fresh-round hands."""
        probe = fresh_round(g)
        k = samples or self.cfg.headroom_samples
        rng = random.Random(_seed(scoring_key(g)) if seed is None else seed)
        pool = sorted(probe.full_deck, key=lambda c: (_card_sig(c), c.uid))
        n = min(fresh_hand_size(g), len(pool))
        plan = Plan(probe)
        fs = fastscore.pool_scorer(probe, plan, pool) if n <= 16 else None
        if fs is not None:                   # all K hands in one compiled call
            pos = {id(c): i for i, c in enumerate(pool)}
            hands = [[pos[id(c)] for c in rng.sample(pool, n)] for _ in range(k)]
            best = fs.best_two_many(hands, probe.hands_left, probe.discards_left, len(pool) - n,
                                    fastscore.hand_types_mask(probe.round_hand_types), probe.mouth_hand)
            return [b[0][0] if b else 0.0 for b in best]
        out = []
        for _ in range(k):
            hand = rng.sample(pool, n)
            probe.hand = hand
            ids = {id(c) for c in hand}
            probe.deck = [c for c in pool if id(c) not in ids]
            out.append(best_play(probe, plan, hand)[0])
        return out

    def _compute(self, g: Game, key) -> float:
        scores = self.best_scores(g)
        cfg = self.cfg
        mult = g.round_hands() if cfg.round_hands else 1
        if cfg.aggregate == "geomean":
            est = sum(math.log(max(s, 1.0)) for s in scores) / len(scores) + math.log(mult)
        elif cfg.aggregate == "mean":
            est = math.log(max(sum(scores) / len(scores), 1.0) * mult)
        else:
            raise ValueError(f"unknown headroom aggregate {cfg.aggregate!r}")
        return est - math.log(max(next_boss_target(g), 1))


def fresh_round(g: Game) -> Game:
    """A copy of g set up as the first hand of a new round with no boss effect."""
    p = g.clone()
    p.rng = random.Random(0)
    p.state = "SELECTING_HAND"
    p.blind_idx = 0                         # boss_active() is "" off the boss blind: no boss effect
    p.boss_disabled = True
    p.hands_left = p.round_hands()
    p.discards_left = p.round_discards()
    p.discards_used_round = 0
    p.hand_played_round = [0] * N_HANDS
    p.round_hand_types = set()
    p.mouth_hand = -1
    p.chips = 0
    p.juggle = 0
    p.flags.pop("verdant", None)
    for c in p.full_deck:
        c.debuffed = False
        c.hidden = False
    for j in p.jokers:
        j.hidden = False
    return p


class Potential:
    """Phi(s) = w_head * Phi_headroom + w_prog * Phi_prog; 0 at terminal states."""

    def __init__(self, cfg: PotentialConfig | None = None):
        self.cfg = cfg or PotentialConfig()
        self.headroom = Headroom(self.cfg)

    def progress(self, g: Game) -> float:
        return g.furthest_blind / 24.0

    def __call__(self, g: Game) -> float:
        if g.done:
            return 0.0
        return self.cfg.w_head * self.headroom.value(g) + self.cfg.w_prog * self.progress(g)

    def components(self, g: Game) -> dict:
        if g.done:
            return {"phi": 0.0, "headroom": 0.0, "prog": 0.0}
        h, p = self.headroom.value(g), self.progress(g)
        return {"phi": self.cfg.w_head * h + self.cfg.w_prog * p, "headroom": h, "prog": p}
