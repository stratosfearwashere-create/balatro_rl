"""Poker hand detection, matching Balatro's rules (incl. Four Fingers, Shortcut, Smeared)."""
from __future__ import annotations

import os
from itertools import combinations

from .cards import Card

try:                                   # the C++ core (python setup_cython.py build); BALATRO_PURE=1 disables
    if os.environ.get("BALATRO_PURE") == "1":
        raise ImportError
    from . import _core
except ImportError:
    _core = None

HAND_NAMES = ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush",
              "Full House", "Four of a Kind", "Straight Flush", "Five of a Kind",
              "Flush House", "Flush Five"]
HC, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH, FIVE_KIND, FLUSH_HOUSE, FLUSH_FIVE = range(12)
N_HANDS = 12

# (base chips, base mult, chips per level, mult per level)
HAND_BASE = {
    HC: (5, 1, 10, 1), PAIR: (10, 2, 15, 1), TWO_PAIR: (20, 2, 20, 1), TRIPS: (30, 3, 20, 2),
    STRAIGHT: (30, 4, 30, 3), FLUSH: (35, 4, 15, 2), FULL_HOUSE: (40, 4, 25, 2),
    QUADS: (60, 7, 30, 3), STRAIGHT_FLUSH: (100, 8, 40, 4), FIVE_KIND: (120, 12, 35, 3),
    FLUSH_HOUSE: (140, 14, 40, 4), FLUSH_FIVE: (160, 16, 50, 3),
}
SECRET_HANDS = (FIVE_KIND, FLUSH_HOUSE, FLUSH_FIVE)

# API names used by BalatroBot's `hands` dict
API_HAND_NAMES = {name: i for i, name in enumerate(HAND_NAMES)}


def hand_base(hand: int, level: int) -> tuple[int, int]:
    c, m, dc, dm = HAND_BASE[hand]
    lv = max(level, 1)
    return c + dc * (lv - 1), m + dm * (lv - 1)


def _is_straight_ranks(ranks: list[int], shortcut: bool) -> bool:
    rs = sorted(set(ranks))
    if len(rs) != len(ranks):
        return False
    variants = [rs]
    if 14 in rs:
        variants.append(sorted([1 if r == 14 else r for r in rs]))
    maxgap = 2 if shortcut else 1
    for v in variants:
        if all(0 < v[i + 1] - v[i] <= maxgap for i in range(len(v) - 1)):
            return True
    return False


class HandResult:
    __slots__ = ("hand", "scoring", "contains")

    def __init__(self, hand: int, scoring: list[int], contains: set[int]):
        self.hand = hand          # best poker hand index
        self.scoring = scoring    # indices (into played list) of scoring cards
        self.contains = contains  # set of hand types "contained" (for Jolly Joker etc.)


def evaluate(cards: list[Card], four_fingers=False, shortcut=False, smeared=False) -> HandResult:
    if _core is not None and len(cards) <= 5:
        hand, scoring, contains = _core.evaluate(cards, four_fingers, shortcut, smeared)
        return HandResult(hand, scoring, contains)
    return _py_evaluate(cards, four_fingers, shortcut, smeared)


def _py_evaluate(cards: list[Card], four_fingers=False, shortcut=False, smeared=False) -> HandResult:
    """The reference implementation (any number of cards)."""
    need = 4 if four_fingers else 5
    normal: list[int] = []
    stones: list[int] = []
    cnt: dict[int, int] = {}          # rank groups
    for i, c in enumerate(cards):
        if c.is_stone:
            stones.append(i)
        else:
            normal.append(i)
            cnt[c.rank] = cnt.get(c.rank, 0) + 1
    groups = sorted(cnt.items(), key=lambda kv: (-kv[1], -kv[0]))
    counts = [g[1] for g in groups]

    # --- flush (same test as Card.has_suit, inlined: this runs for every candidate play)
    flush_idx: list[int] = []
    if len(normal) >= need:
        # count first, then build the index list only for the winning suit
        n_suit = [0, 0, 0, 0]
        wild = 0
        for i in normal:
            c = cards[i]
            if c.enh == "WILD" and not c.debuffed:        # a debuffed Wild card only has its own suit
                wild += 1
            else:
                n_suit[c.suit] += 1
        if smeared:
            n_suit = [n_suit[s % 2] + n_suit[s % 2 + 2] for s in range(4)]
        best_s = -1
        for s in range(4):                                # the first suit (Spades, Hearts, Clubs, Diamonds)
            if n_suit[s] + wild >= need:                  # that makes a flush, as the game checks them
                best_s = s
                break
        if best_s >= 0:
            if smeared:
                flush_idx = [i for i in normal if (cards[i].enh == "WILD" and not cards[i].debuffed)
                             or cards[i].suit % 2 == best_s % 2]
            else:
                flush_idx = [i for i in normal if (cards[i].enh == "WILD" and not cards[i].debuffed)
                             or cards[i].suit == best_s]
    is_flush = bool(flush_idx)

    # --- straight (needs `need` distinct ranks, so most hands with a pair skip the search)
    straight_idx: set[int] = set()
    if len(normal) >= need and len(cnt) >= need:
        for size in range(need, min(5, len(normal)) + 1):
            for combo in combinations(normal, size):
                if _is_straight_ranks([cards[i].rank for i in combo], shortcut):
                    straight_idx.update(combo)
    is_straight = bool(straight_idx)

    def idx_of_ranks(ranks):
        return [i for i in normal if cards[i].rank in ranks]

    top = counts[0] if counts else 0
    second = counts[1] if len(counts) > 1 else 0

    contains = {HC}
    if top >= 2:
        contains.add(PAIR)
    if top >= 3:
        contains.add(TRIPS)
    if top >= 4:
        contains.add(QUADS)
    if top >= 5:
        contains.add(FIVE_KIND)
    if top >= 2 and second >= 2:                   # two separate groups: Four / Five of a Kind don't count
        contains.add(TWO_PAIR)
    if is_straight:
        contains.add(STRAIGHT)
    if is_flush:
        contains.add(FLUSH)
    if is_straight and is_flush:
        contains.add(STRAIGHT_FLUSH)
    full = top >= 3 and second >= 2
    if full:
        contains.add(FULL_HOUSE)

    if top >= 5 and is_flush:
        hand, sc = FLUSH_FIVE, idx_of_ranks({groups[0][0]})
    elif full and is_flush:
        hand, sc = FLUSH_HOUSE, idx_of_ranks({groups[0][0], groups[1][0]})
    elif top >= 5:
        hand, sc = FIVE_KIND, idx_of_ranks({groups[0][0]})
    elif is_straight and is_flush:
        hand, sc = STRAIGHT_FLUSH, sorted(set(flush_idx) | straight_idx)
    elif top >= 4:
        hand, sc = QUADS, idx_of_ranks({groups[0][0]})
    elif full:
        hand, sc = FULL_HOUSE, idx_of_ranks({groups[0][0], groups[1][0]})
    elif is_flush:
        hand, sc = FLUSH, flush_idx
    elif is_straight:
        hand, sc = STRAIGHT, sorted(straight_idx)
    elif top >= 3:
        hand, sc = TRIPS, idx_of_ranks({groups[0][0]})
    elif top >= 2 and second >= 2:
        hand, sc = TWO_PAIR, idx_of_ranks({groups[0][0], groups[1][0]})
    elif top >= 2:
        hand, sc = PAIR, idx_of_ranks({groups[0][0]})
    else:
        if normal:
            best = max(normal, key=lambda i: (cards[i].rank, -i))
            sc = [best]
        else:
            sc = []
        hand = HC
    if hand in (FLUSH_FIVE, FLUSH_HOUSE):
        contains.update({FLUSH, FULL_HOUSE if hand == FLUSH_HOUSE else FIVE_KIND})
    scoring = sorted(set(sc) | set(stones))
    return HandResult(hand, scoring, contains)
