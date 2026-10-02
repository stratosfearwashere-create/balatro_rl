# cython: language_level=3, boundscheck=False, wraparound=False
"""Compiled playing card: the same class as cards.Card (a dataclass), with C fields.

cards.py imports it when built. Field order, defaults, equality (all fields, including uid), repr and the
shared uid counter (cards and jokers number from one sequence) are those of the Python class, so
decisions are bit for bit the same with or without it."""
cimport cython

SUITS = ["S", "H", "C", "D"]
RANK_CHARS = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
              10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}

cdef long long _next = 1


def _extras(o):
    """A card's ad-hoc attributes (Cython does not allow __dict__ on a typed reference, so: untyped)."""
    return o.__dict__


def next_uid():
    """Ids for cards and jokers, from one counter (as itertools.count(1) in cards.py)."""
    global _next
    cdef long long u = _next
    _next += 1
    return u


cdef class Card:
    FIELDS = ("rank", "suit", "enh", "edition", "seal", "extra_chips", "debuffed", "hidden", "uid")

    def __init__(self, int rank, int suit, str enh="", str edition="", str seal="", int extra_chips=0,
                 bint debuffed=False, bint hidden=False, uid=None):
        self.rank, self.suit = rank, suit
        self.enh, self.edition, self.seal = enh, edition, seal
        self.extra_chips, self.debuffed, self.hidden = extra_chips, debuffed, hidden
        self.uid = next_uid() if uid is None else uid

    # ---- helpers -------------------------------------------------------
    cdef inline bint stone(self):
        return self.enh == "STONE" or self.enh == "HIDDEN"

    @property
    def is_stone(self):
        # "HIDDEN" is only used for placeholder cards when predicting a face-down card
        return self.enh == "STONE" or self.enh == "HIDDEN"

    cpdef int chip_value(self):
        if self.enh == "HIDDEN":
            return 0
        if self.enh == "STONE":
            return 50
        if self.rank == 14:
            return 11
        return self.rank if self.rank < 10 else 10

    cpdef bint is_face(self, bint pareidolia=False, bint from_boss=False):
        """A debuffed card is not a face card (Ride the Bus, Photograph, Midas Mask ...), except to the boss
        blind's own check (The Plant, The Mark: from_boss)."""
        if self.stone() or (self.debuffed and not from_boss):
            return False
        return pareidolia or self.rank == 11 or self.rank == 12 or self.rank == 13

    cpdef bint has_suit(self, int s, bint smeared=False):
        """Suit test that ignores debuffs: the suit bosses' own check and Flower Pot (a debuffed Wild card is
        still every suit here)."""
        if self.stone():
            return False
        if self.enh == "WILD":
            return True
        if smeared:
            return (self.suit % 2) == (s % 2)   # S/C share parity 0, H/D parity 1
        return self.suit == s

    cpdef bint flush_suit(self, int s, bint smeared=False):
        """Suit test of flushes and Blackboard: a debuffed card keeps its suit, but a debuffed Wild card is
        no longer wild."""
        if self.stone():
            return False
        if self.enh == "WILD" and not self.debuffed:
            return True
        if smeared:
            return (self.suit % 2) == (s % 2)
        return self.suit == s

    cpdef bint live_suit(self, int s, bint smeared=False):
        """Suit test of the jokers that look at cards (Seeing Double ...): a debuffed card has no suit."""
        return not self.debuffed and self.has_suit(s, smeared)

    def key(self):
        return f"{SUITS[self.suit]}_{RANK_CHARS[self.rank]}"

    def copy(self, bint new_uid=True):
        c = Card(self.rank, self.suit, self.enh, self.edition, self.seal, self.extra_chips, self.debuffed)
        if not new_uid:
            c.uid = self.uid
        return c

    cdef Card dup(self):
        """An exact copy (every field, same uid, any ad-hoc attributes): what Game.clone makes."""
        cdef Card c = Card.__new__(Card)
        c.rank, c.suit = self.rank, self.suit
        c.enh, c.edition, c.seal = self.enh, self.edition, self.seal
        c.extra_chips, c.debuffed, c.hidden, c.uid = self.extra_chips, self.debuffed, self.hidden, self.uid
        extra = _extras(self)
        if extra:
            _extras(c).update(extra)
        return c

    def _dup(self):
        return self.dup()

    # ---- dataclass protocol ---------------------------------------------
    def _tuple(self):
        return (self.rank, self.suit, self.enh, self.edition, self.seal, self.extra_chips, self.debuffed,
                self.hidden, self.uid)

    def __eq__(self, other):
        if type(other) is not Card:
            return NotImplemented
        return self._tuple() == (<Card>other)._tuple()

    __hash__ = None                                    # as a mutable dataclass: unhashable

    def __reduce__(self):
        extra = _extras(self)
        return (Card, self._tuple(), dict(extra)) if extra else (Card, self._tuple())

    def __repr__(self):
        mods = "".join(f"[{m}]" for m in (self.enh, self.edition, self.seal) if m)
        return f"{RANK_CHARS[self.rank]}{SUITS[self.suit].lower()}{mods}"


cdef inline tuple _sort_key(Card c):
    return (-(0 if c.stone() else c.rank), c.suit, c.uid)


def sort_hand(cards):
    """Deterministic order used for the 8 hand slots: rank desc, then suit."""
    cdef list keyed = [(_sort_key(<Card>c), i, c) for i, c in enumerate(cards)]
    keyed.sort()
    return [t[2] for t in keyed]
