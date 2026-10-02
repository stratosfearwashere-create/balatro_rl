"""Playing cards."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

SUITS = ["S", "H", "C", "D"]            # index 0..3, same letters as the BalatroBot API
RANK_CHARS = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
              10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
CHAR_RANKS = {v: k for k, v in RANK_CHARS.items()}

ENHANCEMENTS = ["", "BONUS", "MULT", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY"]
EDITIONS = ["", "FOIL", "HOLO", "POLYCHROME", "NEGATIVE"]
SEALS = ["", "RED", "BLUE", "GOLD", "PURPLE"]

_uid = itertools.count(1)


def next_uid() -> int:
    return next(_uid)


@dataclass
class Card:
    rank: int                  # 2..14 (14 = Ace)
    suit: int                  # 0..3 index into SUITS
    enh: str = ""
    edition: str = ""
    seal: str = ""
    extra_chips: int = 0       # permanent bonus chips (Hiker)
    debuffed: bool = False
    hidden: bool = False       # dealt face down (The House/Wheel/Fish/Mark)
    uid: int = field(default_factory=next_uid)

    # ---- helpers -------------------------------------------------------
    @property
    def is_stone(self) -> bool:
        # "HIDDEN" is only used for placeholder cards when predicting a face-down card
        return self.enh == "STONE" or self.enh == "HIDDEN"

    def chip_value(self) -> int:
        if self.enh == "HIDDEN":
            return 0
        if self.is_stone:
            return 50
        if self.rank == 14:
            return 11
        return min(self.rank, 10)

    def is_face(self, pareidolia: bool = False, from_boss: bool = False) -> bool:
        """A debuffed card is not a face card (Ride the Bus, Photograph, Midas Mask ...), except to the boss
        blind's own check (The Plant, The Mark: from_boss)."""
        if self.is_stone or (self.debuffed and not from_boss):
            return False
        return pareidolia or self.rank in (11, 12, 13)

    def has_suit(self, s: int, smeared: bool = False) -> bool:
        """Suit test that ignores debuffs: the suit bosses' own check and Flower Pot (a debuffed Wild card is
        still every suit here)."""
        if self.is_stone:
            return False
        if self.enh == "WILD":
            return True
        if smeared:
            return (self.suit % 2) == (s % 2)   # S/C share parity 0, H/D parity 1
        return self.suit == s

    def flush_suit(self, s: int, smeared: bool = False) -> bool:
        """Suit test of flushes and Blackboard: a debuffed card keeps its suit, but a debuffed Wild card is
        no longer wild."""
        if self.is_stone:
            return False
        if self.enh == "WILD" and not self.debuffed:
            return True
        if smeared:
            return (self.suit % 2) == (s % 2)
        return self.suit == s

    def live_suit(self, s: int, smeared: bool = False) -> bool:
        """Suit test of the jokers that look at cards (Seeing Double ...): a debuffed card has no suit."""
        return not self.debuffed and self.has_suit(s, smeared)

    def key(self) -> str:
        return f"{SUITS[self.suit]}_{RANK_CHARS[self.rank]}"

    def copy(self, new_uid: bool = True) -> "Card":
        c = Card(self.rank, self.suit, self.enh, self.edition, self.seal, self.extra_chips, self.debuffed)
        if not new_uid:
            c.uid = self.uid
        return c

    def __repr__(self) -> str:
        mods = "".join(f"[{m}]" for m in (self.enh, self.edition, self.seal) if m)
        return f"{RANK_CHARS[self.rank]}{SUITS[self.suit].lower()}{mods}"


def standard_deck(deck_type: str = "RED", rng=None) -> list[Card]:
    cards = []
    if deck_type == "ERRATIC":
        import random as _r
        rng = rng or _r.Random()
        return [Card(rng.randrange(2, 15), rng.randrange(4)) for _ in range(52)]
    for s in range(4):
        for r in range(2, 15):
            if deck_type == "ABANDONED" and r in (11, 12, 13):
                continue
            suit = s
            if deck_type == "CHECKERED":
                suit = 0 if s in (0, 2) else 1
            cards.append(Card(r, suit))
    return cards


def sort_hand(cards: list[Card]) -> list[Card]:
    """Deterministic order used for the 8 hand slots: rank desc, then suit."""
    return sorted(cards, key=lambda c: (-(0 if c.is_stone else c.rank), c.suit, c.uid))
