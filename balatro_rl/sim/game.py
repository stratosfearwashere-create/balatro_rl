"""Balatro game engine (Gold Stake rules by default).

A re-implementation of Balatro's rules with all of its content: 150 jokers, 12 planets,
22 tarots, 18 spectrals, 32 vouchers, 24 tags, 15 decks, 8 stakes, 23 boss blinds and
5 finishers. The random number generator differs from the real game, so seeds don't match.
"""
from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass, field
from typing import Optional

from .cards import Card, standard_deck, exact_deck, sort_hand, SUITS
from .hands import N_HANDS, SECRET_HANDS, HAND_NAMES, evaluate
from .shopgen import ShopGen
from .seedrng import ExactRng
from .jokers import JOKERS, Joker, EDITION_COST, NOT_IN_POOL
from .items import (PLANETS, HAND_TO_PLANET, TAROTS, TAROT_ENH, TAROT_SUIT, SPECTRALS, VOUCHERS,
                    VOUCHER_COST, PACKS, PACK_KEYS, TAGS, TAG_MIN_ANTE, BOSSES, FINISHERS, ALL_BOSSES,
                    SUIT_BOSS, BLIND_BASE, STAKES, SPECTRAL_POOL)
from .scoring import Plan, copies, score_hand

_FINAL_NAMES = {"bl_final_acorn": "amber_acorn", "bl_final_bell": "cerulean_bell", "bl_final_heart": "crimson_heart",
                "bl_final_leaf": "verdant_leaf", "bl_final_vessel": "violet_vessel"}
_P_CARD_KEYS = sorted(f"{s}_{r}" for s in "CDHS" for r in "23456789TJQKA")     # G.P_CARDS in key order
_RANK_OF = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
_SUIT_OF = {s: i for i, s in enumerate(SUITS)}
_ENH_OF = {"m_bonus": "BONUS", "m_mult": "MULT", "m_wild": "WILD", "m_glass": "GLASS", "m_steel": "STEEL",
           "m_stone": "STONE", "m_gold": "GOLD", "m_lucky": "LUCKY"}
_ENH_KEY = {v: k for k, v in _ENH_OF.items()}
MAX_HAND = 16          # hand slots exposed to the agent (hand size can grow past 8)
MAX_JOKERS = 8
MAX_CONSUMABLES = 3
MAX_SHOP = 24
MAX_PACK = 5
MAX_VOUCHERS = 2                    # the ante's voucher, plus one from a Voucher tag
WIN_ANTE = 8
# consumables whose effect depends on which cards they target (the rest act on random or all cards)
TARGETED = frozenset(TAROT_ENH) | frozenset(TAROT_SUIT) | {
    "strength", "hanged_man", "death", "talisman", "deja_vu", "trance", "medium", "aura", "cryptid"}


@dataclass
class ShopItem:
    kind: str                       # joker | tarot | planet | spectral | voucher | pack | card
    key: str                        # e.g. "j_joker", "c_mars", "v_grabber", "p_buffoon_normal"
    cost: int
    joker: Optional[Joker] = None
    card: Optional[Card] = None
    pack: Optional[tuple] = None    # (kind, size)


@dataclass
class Consumable:
    kind: str                       # tarot | planet | spectral
    name: str                       # key without prefix, e.g. "mars"
    negative: bool = False
    sell_bonus: int = 0

    @property
    def key(self):
        return f"c_{self.name}"

    def sell_value(self):
        return (1 if self.kind != "spectral" else 2) + self.sell_bonus


_ATOMS = (int, float, str, bool, type(None), frozenset)


def _clone(v, memo: dict):
    """Game.clone's copier: fast paths for the game's own types, memoised so sharing is preserved."""
    t = type(v)
    if t in _ATOMS:
        return v
    got = memo.get(id(v))
    if got is not None:
        return got
    if t is list:
        new = []
        memo[id(v)] = new
        new.extend(_clone(x, memo) for x in v)
        return new
    if t is Card:
        new = v._dup()
    elif t is Consumable:                            # a dataclass of immutable fields
        new = object.__new__(t)
        new.__dict__ = dict(v.__dict__)
    elif t is Joker:
        new = object.__new__(t)
        new.__dict__ = dict(v.__dict__)             # its JokerDef `d` is shared, as deepcopy does
        new.state = _clone(v.state, memo)
    elif t is ShopItem:
        new = object.__new__(t)
        new.__dict__ = {k: _clone(x, memo) for k, x in v.__dict__.items()}
    elif t is dict:
        new = {k: _clone(x, memo) for k, x in v.items()}
    elif t is set:
        new = set(v) if all(type(x) in _ATOMS for x in v) else copy.deepcopy(v)
    elif t is tuple:
        new = tuple(_clone(x, memo) for x in v)
    elif t is random.Random:
        new = random.Random()
        new.setstate(v.getstate())
    else:
        new = copy.deepcopy(v)
    memo[id(v)] = new
    return new


try:                                   # compiled copier (python setup_cython.py build); BALATRO_PURE=1 disables
    if os.environ.get("BALATRO_PURE") == "1":
        raise ImportError
    from . import _clone as _cclone
    _cclone.register(Consumable, Joker, ShopItem)
    _clone_game = _cclone.clone_game
except ImportError:
    _clone_game = None


_VALUE_ENH = {"": 0, "BONUS": 8, "MULT": 10, "WILD": 6, "GLASS": 20, "STEEL": 15, "STONE": -5,
              "GOLD": 5, "LUCKY": 10, "HIDDEN": 0}
_VALUE_EDITION = {"": 0, "FOIL": 8, "HOLO": 12, "POLYCHROME": 25, "NEGATIVE": 0}
_VALUE_SEAL = {"": 0, "RED": 15, "BLUE": 5, "GOLD": 8, "PURPLE": 5}


class Game:
    def __init__(self, seed: Optional[int] = None, deck_type: str = "RED", stake: str = "GOLD",
                 win_ante: int = WIN_ANTE, joker_pool=None, exact_seed: Optional[str] = None):
        """win_ante < 8 ends the run as won after that ante's boss; joker_pool (keys) restricts the
        jokers shops and packs can offer. Both exist for cheap experiments; defaults are the real game."""
        self.win_ante = win_ante
        self.joker_pool = frozenset(joker_pool) if joker_pool else None
        self.rng = random.Random(seed)
        self.seed = seed
        self.deck_type = deck_type
        self.stake = STAKES.index(stake)
        # exact mode: the shop's random draws come from the real game's named streams, seeded by this 8-character seed
        self.sg = ShopGen(exact_seed, stake=self.stake + 1) if exact_seed else None
        self.targets: dict = {}                 # exact mode: the round targets of Idol, Mail-In Rebate, Ancient Joker, Castle
        self.orbital_choices: list = []
        self.ante = 1
        self.blind_idx = 0          # 0 small, 1 big, 2 boss
        self.state = "BLIND_SELECT"
        self.money = 4 + (10 if deck_type == "YELLOW" else 0)
        self.full_deck: list[Card] = standard_deck(deck_type, self.rng) if exact_seed is None else exact_deck(deck_type, self.rng)
        self.starting_deck_size = len(self.full_deck)
        self.deck: list[Card] = []
        if self.sg is not None:                      # the game's first shuffle, before any blind is chosen
            self.deck = sorted(self.full_deck, key=lambda c: c.uid)
            self.sg.streams.shuffle(self.deck, "shuffle")
        self.hand: list[Card] = []
        self.discard_pile: list[Card] = []
        self.jokers: list[Joker] = []
        self.consumables: list[Consumable] = []
        self.joker_slots = 5 + (1 if deck_type == "BLACK" else 0) - (1 if deck_type == "PAINTED" else 0)
        self.consumable_slots = 2
        self.hand_size = 8 + (2 if deck_type == "PAINTED" else 0)
        self.base_hands = 4 + (1 if deck_type == "BLUE" else 0) - (1 if deck_type == "BLACK" else 0)
        self.base_discards = 3 + (1 if deck_type == "RED" else 0) - (1 if self.stake >= 4 else 0)
        self.hand_levels = [1] * N_HANDS
        self.hand_played = [0] * N_HANDS
        self.hand_played_round = [0] * N_HANDS
        self.vouchers: set[str] = set()
        self.pending_tags: list[str] = []
        self.flags: dict = {}
        self.planets_used: set[str] = set()
        self.tarots_used = 0
        self.last_consumable: Optional[Consumable] = None
        self.blinds_skipped = 0
        self.blinds_beaten = 0
        self.furthest_blind = 0     # furthest blind beaten, 1..24 (ante 8 boss); replays after -1 Ante don't count
        self.total_hands_played = 0
        self.last_hand = 0
        self.total_unused_discards = 0
        self.used_bosses: set[str] = set()
        # round state
        self.chips = 0
        self.target = 0
        self.hands_left = 0
        self.discards_left = 0
        self.discards_used_round = 0
        self.crimson_disabled = -1
        self.forced_uid = -1
        self.ante_played_uids: set[int] = set()
        self.round_hand_types: set[int] = set()
        self.mouth_hand = -1
        self.boss_disabled = False
        self.juggle = 0
        self.hand_size_override = 0
        self.targeting = None               # {"cons": Consumable, "from_pack": bool} while targets are chosen
        self.first_draw_done = False
        self.boss_rerolled_ante = 0
        self.ox_hand = -1                   # The Ox: the most played hand, fixed when its round starts
        # shop state
        self.shop: list[ShopItem] = []
        self.shop_packs: list[ShopItem] = []
        self.ante_voucher: Optional[ShopItem] = None    # this ante's voucher until bought (kept across shops)
        self.shop_vouchers: list[ShopItem] = []          # vouchers in the current shop (ante's first)
        self.reroll_cost = 5
        self.free_rerolls = 0
        self.shops_seen = 0
        # pack state
        self.pack_cards: list = []
        self.pack_kind = ""
        self.pack_picks = 0
        self.pack_hand: list[Card] = []
        self.pack_return = "SHOP"
        # blinds for this ante
        self.boss = ""
        self.tags_offered: list[str] = []
        self.new_ante()
        if self.sg is not None:
            self.reset_round_targets()
        if deck_type == "MAGIC":
            self.vouchers.add("crystal_ball")
            self.consumable_slots += 1
            self.consumables += [Consumable("tarot", "fool"), Consumable("tarot", "fool")]
        elif deck_type == "NEBULA":
            self.vouchers.add("telescope")
            self.consumable_slots -= 1
        elif deck_type == "ZODIAC":
            self.vouchers.update({"tarot_merchant", "planet_merchant", "overstock_norm"})
        elif deck_type == "GHOST":
            self.consumables.append(Consumable("spectral", "hex"))

    # ------------------------------------------------------------------ copying
    def clone(self) -> "Game":
        """An independent copy, equal to copy.deepcopy(self) but several times faster (searches copy the
        game constantly). Every card, joker, consumable and shop item is copied once, so objects shared
        between lists (a card in both the hand and the full deck) stay shared in the copy, and card / joker
        ids are kept. Anything of a type not handled here falls back to copy.deepcopy."""
        if _clone_game is not None:
            return _clone_game(self)
        memo: dict = {}
        new = object.__new__(Game)
        new.__dict__ = {k: _clone(v, memo) for k, v in self.__dict__.items()}
        return new

    # ------------------------------------------------------------------ helpers
    def has(self, key: str) -> bool:
        return any(j.key == key and not j.debuffed for j in self.jokers)

    def count(self, key: str) -> int:
        return sum(1 for j in self.jokers if j.key == key and not j.debuffed)

    def count_with_copies(self, key: str) -> int:
        """Jokers `key` plus Blueprints / Brainstorms copying one (for effects the copies repeat)."""
        return sum(1 for j in self.jokers if not j.debuffed and (j.key == key or copies(self, j) == key))

    def showman(self) -> bool:
        return self.has("ring_master")

    @property
    def shop_voucher(self) -> Optional[ShopItem]:
        """The first voucher in the shop (kept for callers that know only one voucher slot)."""
        return self.shop_vouchers[0] if self.shop_vouchers else None

    @shop_voucher.setter
    def shop_voucher(self, it: Optional[ShopItem]):
        self.ante_voucher = it
        self.shop_vouchers = [it] if it is not None else []

    def _voucher_options(self) -> list[str]:
        shown = {it.key[2:] for it in self.shop_vouchers}
        return [v for v, pre in VOUCHERS.items()
                if v not in self.vouchers and v not in shown and (pre is None or pre in self.vouchers)]

    def prob(self, n: int, d: int) -> float:
        return min(1.0, n * (2 ** self.count("oops")) / d)

    def debt_limit(self) -> int:
        return -20 if self.has("credit_card") else 0

    def can_afford(self, cost: int) -> bool:
        return self.money - cost >= self.debt_limit()

    def cons_used(self) -> int:
        return sum(1 for c in self.consumables if not c.negative)

    def cons_room(self) -> bool:
        return self.cons_used() < self.consumable_slots

    def create_consumable(self, kind: str, negative: bool = False, name: Optional[str] = None, src: str = "sho"):
        if negative or self.cons_room():
            c = self.random_consumable(kind, src=src) if name is None else Consumable(kind, name)
            c.negative = negative
            self.consumables.append(c)

    @property
    def done(self) -> bool:
        return self.state in ("GAME_OVER", "WON")

    def scaling(self) -> int:
        return 3 if self.stake >= 5 else (2 if self.stake >= 2 else 1)

    def blind_target(self, idx: int) -> int:
        a = min(max(self.ante, 1), 8)
        base = BLIND_BASE[self.scaling()][a - 1]
        mult = [1.0, 1.5, ALL_BOSSES[self.boss][2] if self.boss else 2][idx]
        if self.deck_type == "PLASMA":
            mult *= 2
        if self.stake >= 8:                      # Diamond
            mult *= 1.25
        return int(base * mult)

    def boss_active(self) -> str:
        return self.boss if (self.blind_idx == 2 and self.state == "SELECTING_HAND" and not self.boss_disabled) else ""

    def roll(self, key: str) -> float:
        """A probability roll on the game's named stream `key` in exact mode, else the plain generator."""
        return self.sg.streams.pseudorandom(key) if self.sg is not None else self.rng.random()

    def xpick(self, key: str, seq):
        """pseudorandom_element(seq, pseudoseed(key)): the cards and jokers in creation order, anything else as listed."""
        if self.sg is None:
            return self.rng.choice(seq)
        s = sorted(seq, key=lambda x: x.uid) if seq and hasattr(seq[0], "uid") else list(seq)
        return s[self.sg.streams.pseudorandom(key, 1, len(s)) - 1]

    def marble_card(self) -> Card:
        if self.sg is None:
            return Card(2, self.rng.randrange(4), enh="STONE")
        r, su = self.front_of("marb_fr")
        return Card(r, su, enh="STONE")

    def front_of(self, key: str) -> tuple[int, int]:
        """A random playing-card front (rank, suit) from G.P_CARDS on the stream `key`."""
        k = _P_CARD_KEYS[self.sg.streams.pseudorandom(key, 1, 52) - 1]
        return _RANK_OF[k[2]], _SUIT_OF[k[0]]

    def edition_poll(self, key: str) -> str:
        """The guaranteed edition of Aura / Wheel of Fortune (no Negative): foil 50 / holo 35 / polychrome 15."""
        if self.sg is None:
            return self.rng.choices(["FOIL", "HOLO", "POLYCHROME"], [50, 35, 15])[0]
        return (self.sg.poll_edition(key, 1, True, True) or "").upper()

    # the game's pairs() order of G.GAME.hands (the order lists of "visible hands" come in), as hand indices
    _HANDS_PAIRS = (10, 6, 5, 1, 0, 8, 4, 2, 11, 9, 3, 7)

    def visible_hands(self, exclude: int = -1) -> list[int]:
        return [h for h in self._HANDS_PAIRS if (h not in SECRET_HANDS or self.hand_played[h] > 0) and h != exclude]

    def reset_round_targets(self):
        """reset_idol_card / reset_mail_rank / reset_ancient_card / reset_castle_card: at the start of the run and after every round."""
        ante = self.ante
        valid = [c for c in sorted(self.full_deck, key=lambda c: c.uid) if not c.is_stone]
        t = self.targets
        t["idol"] = (14, 0)
        t["mail"] = 14
        if valid:
            c = self.xpick("idol%d" % ante, valid)
            t["idol"] = (c.rank, c.suit)
            t["mail"] = self.xpick("mail%d" % ante, valid).rank
        t["ancient"] = self.xpick("anc%d" % ante, [s for s in range(4) if s != t.get("ancient", -1)])
        t["castle"] = self.xpick("cas%d" % ante, valid).suit if valid else 0
        for j in self.jokers:
            self.sync_target_state(j)

    def sync_target_state(self, j):
        t = self.targets
        if j.key == "idol":
            j.state["rank"], j.state["suit"] = t["idol"]
        elif j.key == "mail":
            j.state["rank"] = t["mail"]
        elif j.key == "ancient":
            j.state["suit"] = t["ancient"]
        elif j.key == "castle":
            j.state["suit"] = t["castle"]

    def draw_orbital_choices(self):
        """The hand each blind's Orbital tag levels up, drawn when the ante's blind choices are first shown (Small, Big, Boss)."""
        self.orbital_choices = [self.xpick("orbital", self.visible_hands()) for _ in range(3)]

    def _crimson_pick(self) -> int:
        if self.sg is None:
            return self.rng.choice(self.jokers).uid
        cand = [j for j in self.jokers if (not j.debuffed and j.uid != self.crimson_disabled) or len(self.jokers) < 2]
        return self.xpick("crimson_heart", cand).uid if cand else -1

    @property
    def xrng(self):
        return ExactRng(self.sg.streams, self.rng) if self.sg is not None else self.rng

    def random_deck_rank(self) -> int:
        normal = [c for c in self.full_deck if not c.is_stone]
        return self.rng.choice(normal).rank if normal else 14

    def random_deck_card(self):
        normal = [c for c in self.full_deck if not c.is_stone]
        c = self.rng.choice(normal) if normal else Card(14, 0)
        return c.rank, c.suit

    def interest_cap(self) -> int:
        return 20 if "money_tree" in self.vouchers else (10 if "seed_money" in self.vouchers else 5)

    def discount(self) -> float:
        return 0.5 if "liquidation" in self.vouchers else (0.25 if "clearance_sale" in self.vouchers else 0.0)

    def price(self, base: int) -> int:
        # Balatro: max(1, floor((base + 0.5) * (100 - discount%) / 100))
        return max(1, int((base + 0.5) * (1 - self.discount())))

    def effective_hand_size(self) -> int:
        if self.hand_size_override:        # set by the real-game bridge (the game reports it)
            return self.hand_size_override
        hs = (self.hand_size + ("paint_brush" in self.vouchers) + ("palette" in self.vouchers)
              + self.count("juggler") + 2 * self.count("troubadour") - 2 * self.count("stuntman")
              - self.count("merry_andy") + self.juggle
              + sum(j.state.get("val", 0) for j in self.jokers if j.key == "turtle_bean" and not j.debuffed)
              - (1 if self.boss_active() == "manacle" else 0))
        return max(1, min(MAX_HAND, int(hs)))

    def round_hands(self) -> int:
        return max(1, self.base_hands + ("grabber" in self.vouchers) + ("nacho_tong" in self.vouchers)
                   - ("hieroglyph" in self.vouchers) - self.count("troubadour"))

    def round_discards(self) -> int:
        return max(0, self.base_discards + self.count("drunkard") + 3 * self.count("merry_andy")
                   + ("wasteful" in self.vouchers) + ("recyclomancy" in self.vouchers)
                   - ("petroglyph" in self.vouchers))

    # ------------------------------------------------------------------ antes / blinds
    def _exact_boss(self) -> str:
        self._sg_sync()
        key = self.sg.new_boss()
        return _FINAL_NAMES.get(key) or key[3:]

    def new_ante(self):
        if self.sg is not None:
            self.boss = self._exact_boss()
            self.used_bosses.add(self.boss)
            self.tags_offered = [self.random_tag(), self.random_tag()]
            self.draw_orbital_choices()
            self.ante_voucher = None
            self.shop_vouchers = []
            self.ante_played_uids = set()
            return
        if self.ante >= WIN_ANTE:
            pool = list(FINISHERS)
        else:
            pool = [k for k, v in BOSSES.items() if v[1] <= self.ante and k not in self.used_bosses]
            if not pool:
                self.used_bosses -= set(BOSSES)
                pool = [k for k, v in BOSSES.items() if v[1] <= self.ante]
        self.boss = self.rng.choice(pool)
        self.used_bosses.add(self.boss)
        self.tags_offered = [self.random_tag(), self.random_tag()]
        self.ante_voucher = None
        self.shop_vouchers = []
        self.ante_played_uids = set()

    def random_tag(self) -> str:
        if self.sg is not None:
            self._sg_sync()
            t = self.sg.next_tag()[4:]
            return "speed" if t == "skip" else t
        pool = [t for t in TAGS if TAG_MIN_ANTE.get(t, 1) <= self.ante]
        return self.rng.choice(pool)

    def reroll_boss(self):
        """Director's Cut (once per ante) / Retcon (unlimited): $10 to reroll the boss."""
        assert self.can_reroll_boss()
        self.money -= 10
        self.boss_rerolled_ante = self.ante
        if self.sg is not None:
            self.boss = self._exact_boss()
            return
        pool = list(FINISHERS) if self.ante >= WIN_ANTE else [k for k, v in BOSSES.items() if v[1] <= self.ante]
        pool = [b for b in pool if b != self.boss] or pool
        self.boss = self.rng.choice(pool)

    def can_reroll_boss(self) -> bool:
        if self.state != "BLIND_SELECT" or not self.can_afford(10):
            return False
        return "retcon" in self.vouchers or ("directors_cut" in self.vouchers and self.boss_rerolled_ante != self.ante)

    def disable_boss(self):
        """Everything the game undoes when a boss is disabled (Chicot, Luchador): The Water's discards and
        The Needle's hands come back, The Wall's target halves and Violet Vessel's thirds, The Manacle's
        hand size returns (and one card is drawn), face-down cards and jokers turn face up, debuffs and
        Cerulean Bell's forced card go."""
        if self.boss_disabled:
            return
        boss = self.boss if self.blind_idx == 2 else ""
        in_round = self.state == "SELECTING_HAND"
        self.boss_disabled = True
        self.flags.pop("verdant", None)
        if in_round and boss == "water":
            self.discards_left = max(self.discards_left, self.round_discards() - self.discards_used_round)
        if in_round and boss == "needle":
            self.hands_left = max(self.hands_left, self.round_hands() - sum(self.hand_played_round))
        if in_round and boss in ("wall", "violet_vessel"):
            t = self.target / (2 if boss == "wall" else 3)
            self.target = int(t) if t == int(t) else t
        if in_round and boss == "manacle" and self.first_draw_done:
            self.draw(1)
        self.forced_uid = -1
        for c in self.full_deck:
            c.debuffed = False
            c.hidden = False
        for j in self.jokers:
            j.hidden = False

    def select_blind(self):
        assert self.state == "BLIND_SELECT"
        self.state = "SELECTING_HAND"
        self.boss_disabled = False
        self.target = self.blind_target(self.blind_idx)
        self.chips = 0
        boss = self.boss_active()
        self.hands_left = self.round_hands()
        self.discards_left = self.round_discards()
        if boss == "needle":
            self.hands_left = 1
        if boss == "water":
            self.discards_left = 0
        self.discards_used_round = 0
        self.hand_played_round = [0] * N_HANDS
        self.round_hand_types = set()
        self.mouth_hand = -1
        self.first_draw_done = False
        if boss == "amber_acorn":
            if self.sg is not None:
                if len(self.jokers) > 1:                       # three shuffles, each starting from the creation order
                    base = sorted(self.jokers, key=lambda j: j.uid)
                    for _ in range(3):
                        shuffled = list(base)
                        self.sg.streams.shuffle(shuffled, "aajk")
                    self.jokers = shuffled
            else:
                self.rng.shuffle(self.jokers)
            for j in self.jokers:
                j.hidden = True
        if boss == "crimson_heart" and self.jokers:
            self.crimson_disabled = -1
            self.crimson_disabled = self._crimson_pick()
        if boss == "verdant_leaf":
            self.flags["verdant"] = True
        for j in list(self.jokers):
            if j.d.blind_select and not j.debuffed and j in self.jokers:
                j.d.blind_select(self, j)
        if boss and self.has("chicot"):             # after the target and the boss's own effects are set
            self.disable_boss()
            boss = ""
        if boss == "ox":                            # most played hand, ties to the higher-ranked hand
            self.ox_hand = max(range(N_HANDS), key=lambda h: (self.hand_played[h], h))
        for c in self.full_deck:
            c.debuffed = False
            c.hidden = False
        self.apply_debuffs()
        self.deck = list(self.full_deck)
        if self.sg is not None:                      # the game shuffles the cards in creation order with the 'nr' + ante stream
            self.deck.sort(key=lambda c: c.uid)
            self.sg.streams.shuffle(self.deck, "nr%d" % self.ante)
        else:
            self.rng.shuffle(self.deck)
        self.hand = []
        self.discard_pile = []
        self.draw()
        self.first_draw_done = True
        for _ in range(self.count_with_copies("certificate")):
            if self.sg is not None:
                r, su = self.front_of("cert_fr")
                t = self.sg.streams.pseudorandom("certsl")
                c = Card(r, su, seal="RED" if t > 0.75 else "BLUE" if t > 0.5 else "GOLD" if t > 0.25 else "PURPLE")
            else:
                c = Card(self.rng.randrange(2, 15), self.rng.randrange(4),
                         seal=self.rng.choice(["RED", "BLUE", "GOLD", "PURPLE"]))
            self.add_card(c, to_hand=True)

    def apply_debuffs(self):
        boss = self.boss_active()
        par = self.has("pareidolia")
        smeared = self.has("smeared")
        for c in self.full_deck:
            d = False
            if boss in SUIT_BOSS and c.has_suit(SUIT_BOSS[boss], smeared):   # Wild cards too
                d = True
            if boss == "plant" and c.is_face(par, from_boss=True):
                d = True
            if boss == "pillar" and c.uid in self.ante_played_uids:
                d = True
            if boss == "verdant_leaf" and self.flags.get("verdant"):
                d = True
            c.debuffed = d

    def skip_blind(self):
        assert self.state == "BLIND_SELECT" and self.blind_idx < 2
        tag = self.tags_offered[self.blind_idx]
        self.blinds_skipped += 1
        self.blind_idx += 1
        self.add_tag(tag)

    def add_tag(self, tag: str):
        if tag == "double":
            self.pending_tags.append("double")
            return
        n = 1 + self.pending_tags.count("double")         # every queued Double copies the next tag,
        self.pending_tags = [t for t in self.pending_tags if t != "double"]   # wherever it sits in the queue
        for _ in range(n):
            self.apply_tag(tag)

    def apply_tag(self, tag: str):
        if tag == "economy":
            self.money += min(40, max(0, self.money))
        elif tag == "speed":
            self.money += 5 * self.blinds_skipped
        elif tag == "handy":
            self.money += self.total_hands_played
        elif tag == "garbage":
            self.money += self.total_unused_discards
        elif tag == "top_up":
            for _ in range(2):
                if len(self.jokers) < self.joker_slots:
                    self.add_joker(self.random_joker(1, sticker=False, src="top"))
        elif tag == "orbital":
            if self.sg is not None:
                h = self.orbital_choices[max(0, min(2, self.blind_idx - 1))]
            else:
                h = self.rng.choice([i for i in range(N_HANDS) if i not in SECRET_HANDS or self.hand_played[i]])
            self.hand_levels[h] += 3
        elif tag == "juggle":
            self.pending_tags.append("juggle")
        elif tag == "boss":
            pool = [k for k, v in BOSSES.items() if v[1] <= self.ante and k != self.boss] if self.ante < WIN_ANTE \
                else [k for k in FINISHERS if k != self.boss]
            self.boss = self._exact_boss() if self.sg is not None else self.rng.choice(pool)
        elif tag in ("buffoon", "charm", "meteor", "standard", "ethereal"):
            kind = {"buffoon": "buffoon", "charm": "arcana", "meteor": "celestial",
                    "standard": "standard", "ethereal": "spectral"}[tag]
            size = "normal" if tag == "ethereal" else "mega"
            self.open_pack(kind, size, return_to="BLIND_SELECT")
        else:
            # applied when the next shop opens: uncommon, rare, foil, holo, polychrome, negative,
            # coupon, d_six, investment, voucher
            self.pending_tags.append(tag)

    # ------------------------------------------------------------------ drawing
    def draw(self, n: Optional[int] = None, after_play: bool = False):
        target = self.effective_hand_size()
        if n is None:
            n = target - len(self.hand)
        n = min(n, MAX_HAND - len(self.hand))
        boss = self.boss_active()
        par = self.has("pareidolia")
        for _ in range(max(0, n)):
            if not self.deck:
                break
            c = self.deck.pop()
            if boss == "house" and not self.first_draw_done:
                c.hidden = True
            elif boss == "wheel" and self.roll("wheel") < self.prob(1, 7):
                c.hidden = True
            elif boss == "fish" and after_play:
                c.hidden = True
            elif boss == "mark" and c.is_face(par, from_boss=True):
                c.hidden = True
            self.hand.append(c)
        self.hand = sort_hand(self.hand)
        if self.boss_active() == "cerulean_bell" and self.hand:
            if self.forced_uid not in [c.uid for c in self.hand]:
                self.forced_uid = self.xpick("cerulean_bell", self.hand).uid

    # ------------------------------------------------------------------ playing
    def forced_pos(self) -> int:
        if self.boss_active() != "cerulean_bell":
            return -1
        for i, c in enumerate(self.hand):
            if c.uid == self.forced_uid:
                return i
        return -1

    def violates_boss(self, played: list[Card]) -> bool:
        boss = self.boss_active()
        if boss == "psychic" and len(played) < 5:
            return True
        if boss in ("eye", "mouth"):
            h = evaluate(played, self.has("four_fingers"), self.has("shortcut"), self.has("smeared")).hand
            if boss == "eye" and h in self.round_hand_types:
                return True
            if boss == "mouth" and self.mouth_hand >= 0 and h != self.mouth_hand:
                return True
        return False

    def hand_view(self) -> list[Card]:
        """The hand as the player sees it: face-down cards become placeholders."""
        return [Card(2, 0, enh="HIDDEN", uid=c.uid) if c.hidden else c for c in self.hand]

    def predict(self, positions: list[int], plan=None, view=None) -> tuple[float, int]:
        """Expected score and hand type for playing the given hand slots (no side effects).
        Pass `view` (from hand_view) when predicting many plays from the same hand."""
        if view is None:
            view = self.hand_view()
        played = [view[i] for i in positions]
        pos = set(positions)
        held = [c for i, c in enumerate(view) if i not in pos]
        sc, ctx = score_hand(self, played, held, rng=None, commit=False, plan=plan)
        if self.violates_boss(played):
            sc = 0.0
        return sc, ctx.hand

    def predict_many(self, subsets, plan, view=None) -> list[tuple[float, int]]:
        """predict() for many plays from the same hand, using the compiled scorer when it is built."""
        from .fastscore import build
        if view is None:
            view = self.hand_view()
        fs = build(self, plan, view)
        if fs is not None:
            return fs.predict_many(subsets)
        return [self.predict(list(s), plan, view) for s in subsets]

    def play(self, positions: list[int]):
        """Play these hand slots, in this order. As in the game: The Arm lowers the hand's level first
        (above level 1 only); The Hook discards 2 random held cards before scoring; a hand the boss forbids
        (The Psychic, The Eye, The Mouth) scores nothing and triggers no scoring effects, but still counts
        as a hand played and still meets The Eye / The Mouth / The Arm / The Ox."""
        assert self.state == "SELECTING_HAND" and 1 <= len(positions) <= 5
        played = [self.hand[i] for i in positions]
        held = [c for i, c in enumerate(self.hand) if i not in positions]
        for c in played:
            c.hidden = False
        boss = self.boss_active()
        plan = Plan(self)
        violated = self.violates_boss(played)
        res = evaluate(played, plan.four_fingers, plan.shortcut, plan.smeared)
        h = res.hand
        arm_lowered = boss == "arm" and self.hand_levels[h] > 1
        if arm_lowered:
            self.hand_levels[h] -= 1
        if boss == "hook" and held:
            if self.sg is not None:
                pool, hooked = list(held), []
                for _ in range(min(2, len(held))):
                    c = self.xpick("hook", pool)
                    hooked.append(c)
                    pool = [x for x in pool if x is not c]
            else:
                hooked = self.rng.sample(held, min(2, len(held)))
            held = [c for c in held if all(c is not x for x in hooked)]
            self.hand = [c for c in self.hand if all(c is not x for x in hooked)]
            self._discard_effects(hooked, from_hook=True)
            self.discard_pile += hooked
            if "trading_destroy" in self.flags:
                self.destroy_cards([self.flags.pop("trading_destroy")])
        if violated:
            sc, money, events, scoring = 0, 0, [], res.scoring
        else:
            sc, ctx = score_hand(self, played, held, rng=self.xrng, commit=True, plan=plan, arm_applied=True,
                                 hooked=True)
            h, money, events, scoring = ctx.hand, ctx.money, ctx.events, ctx.scoring
        if boss and self.has("matador"):
            triggered = (violated or any(played[i].debuffed for i in scoring)
                         or boss in ("hook", "tooth", "flint", "crimson_heart")
                         or arm_lowered or (boss == "ox" and h == self.ox_hand))
            if triggered:
                self.money += 8 * self.count("matador")
        if boss == "ox" and h == self.ox_hand:
            self.money = 0
        if boss == "tooth":
            self.money -= len(played)
        if boss == "mouth" and self.mouth_hand < 0:
            self.mouth_hand = h
        self.round_hand_types.add(h)
        self.last_hand = h
        self.money += int(money)
        self.chips += sc
        self.hand_played[h] += 1
        self.hand_played_round[h] += 1
        self.total_hands_played += 1
        self.hands_left -= 1
        for u in (c.uid for c in played):
            self.ante_played_uids.add(u)
        for j in list(self.jokers):
            if j.d.after and not j.debuffed:
                j.d.after(self, j)
        # creations / copies triggered during scoring
        destroyed = []
        for ev in events:
            if ev[0] == "create":
                self.create_consumable(ev[1], src=ev[2] if len(ev) > 2 else "sho")
            elif ev[0] == "dna":                       # the copy was held while the hand scored, and
                self.add_card(ev[1], to_hand=True, grow=False)   # Hologram grew then (scoring._dna_before)
            elif ev[0] == "destroy":
                destroyed.append(ev[1])
        if destroyed:
            self.destroy_cards(destroyed)
        self.hand = [c for c in self.hand if c not in played]
        self.discard_pile += [c for c in played if c in self.full_deck]
        if boss == "crimson_heart" and self.jokers:
            self.crimson_disabled = self._crimson_pick()
        if self.chips >= self.target:
            self.win_round()
        elif self.hands_left <= 0:
            bones = next((j for j in self.jokers if j.key == "mr_bones" and not j.debuffed), None)
            if bones is not None and self.chips >= 0.25 * self.target:
                self.destroy_joker(bones)
                self.win_round(saved_by_bones=True)
            else:
                self.state = "GAME_OVER"
        else:
            if boss == "serpent":
                self.draw(3, after_play=True)
            else:
                self.draw(after_play=True)
            if not self.hand:
                self.state = "GAME_OVER"

    def _discard_effects(self, cards: list[Card], from_hook: bool = False):
        """Jokers' discard effects and Purple seals. The Hook's discards trigger them too (without using up a
        discard), except Burnt Joker."""
        for c in cards:
            c.hidden = False
        for j in list(self.jokers):
            if j.d.discard and not j.debuffed and j in self.jokers:
                if from_hook and (j.key == "burnt" or copies(self, j) == "burnt"):
                    continue
                j.d.discard(self, j, cards)
        for c in cards:
            if c.seal == "PURPLE" and not c.debuffed:
                self.create_consumable("tarot", src="8ba")

    def discard(self, positions: list[int]):
        assert self.state == "SELECTING_HAND" and self.discards_left > 0 and 1 <= len(positions) <= 5
        cards = [self.hand[i] for i in positions]
        self._discard_effects(cards)
        self.hand = [c for c in self.hand if c not in cards]
        self.discard_pile += cards
        if "trading_destroy" in self.flags:
            self.destroy_cards([self.flags.pop("trading_destroy")])
        self.discards_left -= 1
        self.discards_used_round += 1
        if self.boss_active() == "serpent":
            self.draw(3)
        else:
            self.draw()

    # ------------------------------------------------------------------ end of round
    def win_round(self, saved_by_bones: bool = False):
        """Round payout. Mr. Bones saving the run pays no blind reward (hands and interest still pay).
        Interest: (1 + To the Moon) x $1 per $5 held, up to the cap; none on the Green Deck."""
        was_boss = self.blind_idx == 2
        self.blinds_beaten += 1
        self.furthest_blind = max(self.furthest_blind, 3 * (self.ante - 1) + self.blind_idx + 1)
        earned = 0
        # at the round's end, before the cash-out: held Gold cards (and Blue seals, retriggered by Mime and Red seals) pay and rentals
        # are charged, so interest counts the money after both (the game's end_round, then evaluate_round)
        pre = 0
        mime = self.count_with_copies("mime")
        for c in self.hand:
            if c.debuffed:
                continue
            triggers = 1 + mime + (c.seal == "RED")
            if c.enh == "GOLD":
                pre += 3 * triggers
            if c.seal == "BLUE":
                for _ in range(triggers):
                    self.create_consumable("planet", name=HAND_TO_PLANET[self.last_hand])
        pre -= 3 * sum(1 for j in self.jokers if j.rental)
        self.money += pre
        if saved_by_bones:
            pass
        elif self.blind_idx == 0:
            earned += 0 if self.stake >= 1 else 3
        else:
            earned += 4 if self.blind_idx == 1 else 5
        if self.deck_type == "GREEN":
            earned += 2 * self.hands_left + self.discards_left
        else:
            earned += self.hands_left
            earned += (1 + self.count("to_the_moon")) * min(max(0, self.money) // 5, self.interest_cap())
        for j in list(self.jokers):
            if j.debuffed:
                continue
            if was_boss and j.key == "rocket":
                j.state["val"] = j.state.get("val", 1) + 2
            if was_boss and j.key == "campfire":
                j.state["val"] = 1.0
            if j.d.end_round:
                earned += j.d.end_round(self, j) or 0
        for j in self.jokers:
            if j.perishable is not None:
                j.perishable -= 1
                if j.perishable <= 0:
                    j.debuffed = True
        self.total_unused_discards += self.discards_left
        self.money += earned
        self.flags.pop("verdant", None)
        for c in self.full_deck:
            c.debuffed = False
            c.hidden = False
        for j in self.jokers:
            j.hidden = False
        self.juggle = 0
        if was_boss:
            if self.deck_type == "ANAGLYPH":
                self.add_tag("double")
            if "investment" in self.pending_tags:
                self.pending_tags.remove("investment")
                self.money += 25
            if self.ante >= self.win_ante:
                self.state = "WON"
                return
            self.ante += 1
            self.blind_idx = 0
            self.new_ante()
        else:
            self.blind_idx += 1
        if self.sg is not None:
            self.reset_round_targets()
        self.open_shop()

    # ------------------------------------------------------------------ shop (exact mode)
    def _sg_sync(self, exclude=()):
        """Tells the stream generator what the game's pools depend on: the cards that exist (the game marks every
        created card as used and the pools skip used keys unless Showman), vouchers, played hands, flags, rates."""
        sg = self.sg
        sg.ante = self.ante
        sg.used = ({f"j_{j.key}" for j in self.jokers} | {c.key for c in self.consumables}
                   | {it.key for it in self.shop} | {f"j_{k}" for k in exclude}
                   | {(f"j_{c.key}" if isinstance(c, Joker) else c.key) for c in self.pack_cards if isinstance(c, (Joker, Consumable))})
        sg.used_vouchers = {f"v_{v}" for v in self.vouchers}
        sg.shop_vouchers = [it.key for it in self.shop_vouchers]
        sg.pool_flags = {"gros_michel_extinct"} if self.flags.get("gros_michel_extinct") else set()
        sg.hands_played = {HAND_NAMES[h]: self.hand_played[h] for h in range(N_HANDS)}
        sg.showman = self.showman()
        sg.deck_enh = {_ENH_KEY[c.enh] for c in self.full_deck if c.enh in _ENH_KEY}
        sg.banned = ({f"j_{k}" for k in JOKERS if k not in self.joker_pool} if self.joker_pool is not None else set())
        tw = 32 if "tarot_tycoon" in self.vouchers else 9.6 if "tarot_merchant" in self.vouchers else 4
        pw = 32 if "planet_tycoon" in self.vouchers else 9.6 if "planet_merchant" in self.vouchers else 4
        sg.rates = {"Joker": 20, "Tarot": tw, "Planet": pw, "Base": 4 if "magic_trick" in self.vouchers else 0,
                    "Spectral": 2 if self.deck_type == "GHOST" else 0}
        sg.edition_rate = 4 if "glow_up" in self.vouchers else 2 if "hone" in self.vouchers else 1

    def _joker_from_card(self, card: dict, sticker: bool) -> Joker:
        key = card["key"][2:]
        d = JOKERS[key]
        edition = (card["edition"] or "").upper()
        j = Joker(key=key, edition=edition, base_cost=d.cost + EDITION_COST[edition])
        if sticker:
            if card["eternal"] and d.eternal_ok:
                j.eternal = True
            elif card["perishable"] and d.perish_ok and not j.eternal:
                j.perishable = 5
            if card["rental"]:
                j.rental = True
        if d.init:
            d.init(self, j)
        j.cost = 1 if j.rental else self.price(j.base_cost)
        return j

    def _exact_joker(self, rarity, sticker, exclude, src, area="shop") -> Joker:
        self._sg_sync(exclude)
        # a rarity given as 1/2/3 is the game's threshold value (0, 0.9 and 1 as the tags and Wraith pass them); 4: legendary
        thr = {None: None, 1: 0.0, 2: 0.9, 3: 1.0, 4: None}[rarity]
        card = self.sg.create_card("Joker", area if sticker else "none", legendary=(rarity == 4), rarity=thr, key_append=src)
        return self._joker_from_card(card, sticker)

    def _exact_consumable(self, kind: str, exclude=(), src="sho", area="shop", soulable=False, forced=None) -> Consumable:
        self._sg_sync()
        self.sg.used |= {f"c_{k}" for k in exclude}
        typ = {"tarot": "Tarot", "planet": "Planet", "spectral": "Spectral"}[kind]
        card = self.sg.create_card(typ, area, key_append=src, soulable=soulable, forced_key=forced)
        name = card["key"][2:]
        k = "tarot" if name in TAROTS else "planet" if name in PLANETS else "spectral"
        return Consumable(k, name)

    def _exact_shop_card(self, exclude=()) -> ShopItem:
        self._sg_sync()
        self.sg.used |= set(exclude)
        # with Illusion every shop card creation draws the 'illusion' stream (is a playing card enhanced?)
        illusion = self.sg.streams.pseudorandom("illusion") if "illusion" in self.vouchers else None
        c = self.sg.shop_card()
        if c["type"] == "Base":
            card = self._exact_shop_playing_card(illusion)
            return ShopItem("card", "<playing_card>", self.price(1 + EDITION_COST[card.edition]), card=card)
        key = c["key"]
        if key.startswith("j_"):
            j = self._joker_from_card(c, True)
            return ShopItem("joker", key, j.cost, joker=j)
        name = key[2:]
        if name in TAROTS:
            return ShopItem("tarot", key, self.price(3))
        if name in PLANETS:
            return ShopItem("planet", key, 0 if self.has("astronomer") else self.price(3))
        return ShopItem("spectral", key, self.price(4))

    def _exact_open_pack(self, kind: str, shown: int):
        sg = self.sg
        for i in range(shown):
            self._sg_sync()
            if kind == "buffoon":
                x = self._exact_joker(None, True, (), "buf", area="pack")
            elif kind == "arcana":
                if "omen_globe" in self.vouchers and sg.streams.pseudorandom("omen_globe") > 0.8:
                    x = self._exact_consumable("spectral", src="ar2", area="pack", soulable=True)
                else:
                    x = self._exact_consumable("tarot", src="ar1", area="pack", soulable=True)
            elif kind == "celestial":
                forced = None
                if "telescope" in self.vouchers and i == 0 and max(self.hand_played) > 0:
                    forced = "c_" + HAND_TO_PLANET[max(range(N_HANDS), key=lambda h: self.hand_played[h])]
                x = self._exact_consumable("planet", src="pl1", area="pack", soulable=True, forced=forced)
            elif kind == "spectral":
                x = self._exact_consumable("spectral", src="spe", area="pack", soulable=True)
            else:
                x = self._exact_standard_card()
            self.pack_cards.append(x)

    def _exact_shop_playing_card(self, illusion) -> Card:
        """A playing card in the shop (Magic Trick); with Illusion it may be enhanced and carry an edition."""
        sg, s, ante = self.sg, self.sg.streams, self.ante
        center = sg.pick("Enhanced", append="sho") if illusion is not None and illusion > 0.6 else None
        key = _P_CARD_KEYS[int(s.pseudorandom("frontsho%d" % ante, 1, 52)) - 1]
        c = Card(_RANK_OF[key[2]], _SUIT_OF[key[0]], enh=_ENH_OF[center] if center else "")
        if illusion is not None and s.pseudorandom("illusion") > 0.8:
            ep = s.pseudorandom("illusion")
            c.edition = "POLYCHROME" if ep > 1 - 0.15 else "HOLO" if ep > 0.5 else "FOIL"
        return c

    def _exact_standard_card(self) -> Card:
        sg, s, ante = self.sg, self.sg.streams, self.ante
        enhanced = s.pseudorandom("stdset%d" % ante) > 0.6
        center = sg.pick("Enhanced", append="sta") if enhanced else None
        key = _P_CARD_KEYS[int(s.pseudorandom("front" + "sta" + str(ante), 1, 52)) - 1]
        c = Card(_RANK_OF[key[2]], _SUIT_OF[key[0]], enh=_ENH_OF[center] if center else "")
        c.edition = (sg.poll_edition("standard_edition%d" % ante, 2, True) or "").upper()
        if s.pseudorandom("stdseal%d" % ante) > 1 - 0.02 * 10:
            t = s.pseudorandom("stdsealtype%d" % ante)
            c.seal = "RED" if t > 0.75 else "BLUE" if t > 0.5 else "GOLD" if t > 0.25 else "PURPLE"
        return c

    def _exact_pack(self) -> ShopItem:
        self._sg_sync()
        _, kind, size = self.sg.pack("shop_pack").split("_")[:3]
        cost = 0 if (kind == "celestial" and self.has("astronomer")) else self.price(PACKS[(kind, size)][2])
        return ShopItem("pack", f"p_{kind}_{size}", cost, pack=(kind, size))

    def _exact_voucher(self, from_tag: bool = False) -> Optional[str]:
        self._sg_sync()
        v = self.sg.next_voucher(from_tag)
        return None if v == "v_blank" and "blank" in self.vouchers else v[2:]

    # ------------------------------------------------------------------ shop
    def random_joker(self, rarity: Optional[int] = None, sticker: bool = True, exclude=(),
                     allow_empty: bool = False, src: str = "sho") -> Optional[Joker]:
        """A random joker, never one already owned or listed in `exclude` (keys; e.g. the rest of the
        shop or pack being generated), unless Showman is owned. The price it counts as paid (Joker.cost, for
        its sell value) is its shop price: discounted, $1 for Rental. allow_empty: None when no joker of
        the rarity is left (else a plain Joker)."""
        if self.sg is not None:
            return self._exact_joker(rarity, sticker, exclude, src)
        if rarity is None:
            r = self.rng.random()
            rarity = 3 if r > 0.95 else (2 if r > 0.7 else 1)
        owned = set() if self.showman() else {j.key for j in self.jokers} | set(exclude)
        pool = [k for k, d in JOKERS.items() if d.rarity == rarity and k not in owned
                and (k not in NOT_IN_POOL or self.flags.get("gros_michel_extinct"))]
        if self.flags.get("gros_michel_extinct") and "gros_michel" in pool:
            pool.remove("gros_michel")
        if self.joker_pool is not None:
            allowed = [k for k in pool if k in self.joker_pool]
            pool = allowed or [k for k in sorted(self.joker_pool) if k not in owned and JOKERS[k].rarity < 4]
        if not pool:
            if allow_empty:
                return None
            pool = ["joker"]
        key = self.rng.choice(pool)
        d = JOKERS[key]
        ed_mult = 4 if "glow_up" in self.vouchers else (2 if "hone" in self.vouchers else 1)
        r = self.rng.random()
        edition = ""
        if r > 1 - 0.003:
            edition = "NEGATIVE"
        elif r > 1 - 0.006 * ed_mult:
            edition = "POLYCHROME"
        elif r > 1 - 0.02 * ed_mult:
            edition = "HOLO"
        elif r > 1 - 0.04 * ed_mult:
            edition = "FOIL"
        j = Joker(key=key, edition=edition, base_cost=d.cost + EDITION_COST[edition])
        if sticker:
            r1 = self.rng.random()
            if self.stake >= 3 and r1 > 0.7 and d.eternal_ok:
                j.eternal = True
            elif self.stake >= 6 and 0.4 < r1 <= 0.7 and d.perish_ok:
                j.perishable = 5
            if self.stake >= 7 and self.rng.random() > 0.7:
                j.rental = True
        if d.init:
            d.init(self, j)
        j.cost = 1 if j.rental else self.price(j.base_cost)
        return j

    def random_consumable(self, kind: str, exclude=(), src: str = "sho") -> Consumable:
        """A random consumable of `kind`, not one named in `exclude` (unless Showman is owned)."""
        if self.sg is not None:
            return self._exact_consumable(kind, exclude, src)
        if kind == "planet":
            opts = [p for p, h in PLANETS.items() if h not in SECRET_HANDS or self.hand_played[h] > 0]
        elif kind == "tarot":
            opts = list(TAROTS)
        else:
            opts = list(SPECTRAL_POOL)
        if not self.showman():
            opts = [o for o in opts if o not in exclude] or opts
        return Consumable(kind, self.rng.choice(opts))

    def shop_card(self, exclude=()) -> ShopItem:
        """One shop card; `exclude`: keys already in this shop (no duplicates unless Showman)."""
        if self.sg is not None:
            return self._exact_shop_card(exclude)
        tw = 32 if "tarot_tycoon" in self.vouchers else 9.6 if "tarot_merchant" in self.vouchers else 4
        pw = 32 if "planet_tycoon" in self.vouchers else 9.6 if "planet_merchant" in self.vouchers else 4
        sw = 2 if self.deck_type == "GHOST" else 0
        cw = 4 if "magic_trick" in self.vouchers else 0
        r = self.rng.random() * (20 + tw + pw + sw + cw)
        names = {k[2:] for k in exclude}
        if r < 20:
            j = self.random_joker(exclude=names)
            return ShopItem("joker", f"j_{j.key}", j.cost, joker=j)
        r -= 20
        if r < tw:
            c = self.random_consumable("tarot", names)
            return ShopItem("tarot", c.key, self.price(3))
        r -= tw
        if r < pw:
            c = self.random_consumable("planet", names)
            return ShopItem("planet", c.key, 0 if self.has("astronomer") else self.price(3))
        r -= pw
        if r < sw:
            c = self.random_consumable("spectral", names)
            return ShopItem("spectral", c.key, self.price(4))
        card = self.shop_playing_card()
        return ShopItem("card", "<playing_card>", self.price(1 + EDITION_COST[card.edition]), card=card)

    def poll_edition(self, mod: float = 1.0, no_neg: bool = False, guaranteed: bool = False) -> str:
        """The game's poll_edition. Hone / Glow Up raise the rates (edition_rate 2 / 4)."""
        rate = 4 if "glow_up" in self.vouchers else 2 if "hone" in self.vouchers else 1
        r = self.rng.random()
        if guaranteed:
            neg, poly, holo, foil = 0.003 * 25, 0.006 * 25, 0.02 * 25, 0.04 * 25
        else:
            neg, poly, holo, foil = 0.003 * mod, 0.006 * rate * mod, 0.02 * rate * mod, 0.04 * rate * mod
        if r > 1 - neg and not no_neg:
            return "NEGATIVE"
        if r > 1 - poly:
            return "POLYCHROME"
        if r > 1 - holo:
            return "HOLO"
        if r > 1 - foil:
            return "FOIL"
        return ""

    def shop_playing_card(self) -> Card:
        """A playing card in the shop (Magic Trick). With Illusion: 40% enhanced, 20% with an edition (foil 50 /
        holo 35 / polychrome 15); never a seal."""
        c = Card(self.rng.randrange(2, 15), self.rng.randrange(4))
        if "illusion" in self.vouchers:
            if self.rng.random() > 0.6:
                c.enh = self.rng.choice(["BONUS", "MULT", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY"])
            if self.rng.random() > 0.8:
                c.edition = self.poll_edition(no_neg=True, guaranteed=True)
        return c

    def random_playing_card(self, enhanced: bool) -> Card:
        """A Standard pack card: 40% enhanced, edition by poll_edition(mod 2, no Negative), 20% sealed."""
        c = Card(self.rng.randrange(2, 15), self.rng.randrange(4))
        if enhanced:
            if self.rng.random() < 0.4:
                c.enh = self.rng.choice(["BONUS", "MULT", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY"])
            c.edition = self.poll_edition(mod=2, no_neg=True)
            if self.rng.random() < 0.2:
                c.seal = self.rng.choice(["RED", "BLUE", "GOLD", "PURPLE"])
        return c

    def random_pack(self) -> ShopItem:
        if self.sg is not None:
            return self._exact_pack()
        weights = [PACKS[k][3] for k in PACK_KEYS]
        kind, size = self.rng.choices(PACK_KEYS, weights)[0]
        cost = 0 if (kind == "celestial" and self.has("astronomer")) else self.price(PACKS[(kind, size)][2])
        return ShopItem("pack", f"p_{kind}_{size}", cost, pack=(kind, size))

    def shop_slots(self) -> int:
        return 2 + ("overstock_norm" in self.vouchers) + ("overstock_plus" in self.vouchers)

    def _shop_cards(self, n: int, first: Optional[list] = None) -> list:
        """n shop cards without duplicates (unless Showman); `first`: cards already placed (tag jokers)."""
        cards = list(first or [])
        saved = self.shop
        if self.sg is not None:
            self.shop = []                       # a reroll removes the old cards first: they no longer count as used
        while len(cards) < n:
            cards.append(self.shop_card(exclude={it.key for it in cards}))
        self.shop = saved
        return cards

    def _apply_edition_tags(self):
        """Edition tags wait for a joker: each goes to the first plain joker in the shop, or stays queued
        (for rerolls and later shops) while there is none."""
        eds = {"foil": "FOIL", "holo": "HOLO", "polychrome": "POLYCHROME", "negative": "NEGATIVE"}
        rest = []
        for t in self.pending_tags:
            it = next((it for it in self.shop if it.kind == "joker" and it.joker.edition == ""), None) \
                if t in eds else None
            if it is None:
                rest.append(t)
                continue
            it.joker.edition = eds[t]
            it.joker.base_cost += EDITION_COST[eds[t]]
            it.cost = it.joker.cost = 0
        self.pending_tags = rest

    def open_shop(self):
        self.state = "SHOP"
        self.shops_seen += 1
        if self.sg is not None:                      # cashing out shuffles the deck ('cashout' + ante): packs draw from it
            self.deck = sorted(self.full_deck, key=lambda c: c.uid)
            self.sg.streams.shuffle(self.deck, "cashout%d" % self.ante)
        self.reroll_cost = max(0, 5 - 2 * ("reroll_surplus" in self.vouchers) - 2 * ("reroll_glut" in self.vouchers))
        self.free_rerolls = self.count("chaos")
        # Uncommon / Rare tags: their free joker takes one of the normal shop slots (the Rare tag makes
        # nothing when every Rare is owned)
        tagged, rest = [], []
        for t in self.pending_tags:
            if t in ("uncommon", "rare") and len(tagged) < self.shop_slots():
                j = self.random_joker(2 if t == "uncommon" else 3, exclude={it.key[2:] for it in tagged},
                                      allow_empty=True, src="uta" if t == "uncommon" else "rta")
                if j is not None:
                    j.cost = 0
                    tagged.append(ShopItem("joker", f"j_{j.key}", 0, joker=j))
            else:
                rest.append(t)
        self.pending_tags = rest
        self.shop = self._shop_cards(self.shop_slots(), tagged)
        if self.shops_seen == 1 and self.sg is None:
            self.shop_packs = [ShopItem("pack", "p_buffoon_normal", self.price(4), pack=("buffoon", "normal")),
                               self.random_pack()]
        else:
            self.shop_packs = [self.random_pack(), self.random_pack()]
        if self.ante_voucher is None and not self.flags.get("voucher_bought_ante") == self.ante:
            self.shop_vouchers = []
            opts = self._voucher_options()
            if opts:
                v = self._exact_voucher() if self.sg is not None else self.rng.choice(opts)
                self.ante_voucher = ShopItem("voucher", f"v_{v}", self.price(VOUCHER_COST))
        self.shop_vouchers = [self.ante_voucher] if self.ante_voucher is not None else []
        # pending shop tags
        self._apply_edition_tags()
        tags, self.pending_tags = self.pending_tags, []
        for t in tags:
            if t in ("foil", "holo", "polychrome", "negative"):
                self.pending_tags.append(t)                # waiting for a joker
            elif t == "coupon":
                for it in self.shop + self.shop_packs:
                    it.cost = 0
                    if it.joker is not None:
                        it.joker.cost = 0
            elif t == "d_six":
                self.reroll_cost = 0
            elif t == "voucher":                         # one more voucher, whatever the shop already has
                opts = self._voucher_options()
                if opts and len(self.shop_vouchers) < MAX_VOUCHERS:
                    v = self._exact_voucher(True) if self.sg is not None else self.rng.choice(opts)
                    self.shop_vouchers.append(ShopItem("voucher", f"v_{v}", self.price(VOUCHER_COST)))
            else:                                        # not a shop tag, or no room for it yet: it stays
                self.pending_tags.append(t)              # queued (Investment, Juggle, a Double waiting for
        self.shop = self.shop[:MAX_SHOP]                 # the next tag, an Uncommon / Rare tag past the slots)

    def can_buy(self, it: ShopItem) -> bool:
        if not self.can_afford(it.cost):
            return False
        if it.kind == "joker":
            neg = it.joker.edition == "NEGATIVE"
            return neg or len(self.jokers) < self.joker_slots
        if it.kind in ("tarot", "planet", "spectral"):
            return self.cons_room()
        return True

    def buy_card(self, i: int):
        it = self.shop[i]
        assert self.can_buy(it)
        self.money -= it.cost
        self.shop.pop(i)
        if it.kind == "joker":
            it.joker.cost = it.cost                         # its sell value follows the price paid
            self.add_joker(it.joker)
        elif it.kind == "card":
            self.add_card(it.card)
        else:
            self.consumables.append(Consumable(it.kind, it.key[2:]))

    def buy_pack(self, i: int):
        it = self.shop_packs[i]
        assert self.can_afford(it.cost)
        self.money -= it.cost
        self.shop_packs.pop(i)
        self.open_pack(*it.pack, return_to="SHOP")

    def buy_voucher(self, i: int = 0):
        it = self.shop_vouchers[i]
        assert self.can_afford(it.cost)
        self.money -= it.cost
        self.shop_vouchers.pop(i)
        if it is self.ante_voucher:                     # the ante's own voucher: none again this ante
            self.ante_voucher = None
            self.flags["voucher_bought_ante"] = self.ante
        self.redeem_voucher(it.key[2:])

    def redeem_voucher(self, v: str):
        self.vouchers.add(v)
        if v == "crystal_ball":
            self.consumable_slots += 1
        elif v == "antimatter":
            self.joker_slots += 1
        elif v in ("hieroglyph", "petroglyph"):
            self.ante = max(1, self.ante - 1)
        elif v in ("overstock_norm", "overstock_plus"):
            self.shop.append(self.shop_card())
            self.shop = self.shop[:MAX_SHOP]
        elif v in ("clearance_sale", "liquidation"):
            pass

    def reroll(self):
        if self.free_rerolls > 0:
            self.free_rerolls -= 1
        else:
            assert self.can_afford(self.reroll_cost)
            self.money -= self.reroll_cost
            self.reroll_cost += 1
        for j in self.jokers:
            if j.key == "flash":
                j.state["val"] = j.state.get("val", 0) + 2
        self.shop = self._shop_cards(self.shop_slots())
        self._apply_edition_tags()

    def leave_shop(self):
        for _ in range(self.count("perkeo")):
            if self.consumables:
                src = self.xpick("perkeo", self.consumables)
                self.consumables.append(Consumable(src.kind, src.name, negative=True))
        if "juggle" in self.pending_tags:
            self.pending_tags.remove("juggle")
            self.juggle = 3
        self.state = "BLIND_SELECT"

    # ------------------------------------------------------------------ jokers / consumables
    def add_joker(self, j: Joker, force: bool = False):
        if j.edition == "NEGATIVE":
            self.joker_slots += 1
        self.jokers.append(j)
        if j.d.init and not j.state:
            j.d.init(self, j)

    def destroy_joker(self, j: Joker):
        if j in self.jokers:
            self.jokers.remove(j)
            if j.edition == "NEGATIVE":
                self.joker_slots -= 1

    def _on_sold(self):
        for o in self.jokers:
            if o.key == "campfire" and not o.debuffed:
                o.state["val"] = o.state.get("val", 1.0) + 0.25

    def sell_joker(self, i: int):
        j = self.jokers[i]
        assert not j.eternal
        self.money += j.sell_value()
        self.destroy_joker(j)
        if j.d.on_sell and not j.debuffed:
            j.d.on_sell(self, j)
        self._on_sold()
        if self.flags.get("verdant"):
            self.flags["verdant"] = False
            if self.state == "SELECTING_HAND":
                self.apply_debuffs()

    def sell_consumable(self, i: int):
        c = self.consumables.pop(i)
        self.money += c.sell_value()
        self._on_sold()

    def swap_jokers(self, i: int):
        self.jokers[i], self.jokers[i + 1] = self.jokers[i + 1], self.jokers[i]

    def destroy_card(self, c: Card):
        self.destroy_cards([c])

    def destroy_cards(self, cards: list[Card]):
        par = self.has("pareidolia")
        glass = faces = 0
        for c in cards:
            if c not in self.full_deck and c not in self.pack_hand:
                continue
            glass += c.enh == "GLASS"
            faces += c.is_face(par)
            for pile in (self.full_deck, self.hand, self.deck, self.discard_pile, self.pack_hand):
                if c in pile:
                    pile.remove(c)
        for j in self.jokers:
            if j.key == "glass" and glass:
                j.state["val"] = j.state.get("val", 1.0) + 0.75 * glass
            if j.key == "caino" and faces:
                j.state["val"] = j.state.get("val", 1.0) + 1.0 * faces

    def add_card(self, c: Card, to_hand: bool = False, grow: bool = True):
        """grow=False: Hologram has already counted this card."""
        self.full_deck.append(c)
        for j in self.jokers:
            if grow and j.key == "hologram" and not j.debuffed:
                j.state["val"] = j.state.get("val", 1.0) + 0.25
        if to_hand and self.state == "SELECTING_HAND" and len(self.hand) < MAX_HAND:
            self.hand.append(c)
            self.hand = sort_hand(self.hand)
        elif self.state == "SELECTING_HAND":
            self.deck.insert(0, c)

    def consumable_usable(self, c: Consumable, cards: list[Card], from_slot: bool = True) -> bool:
        """from_slot: used from the consumable slots (it frees its own slot) rather than picked from a pack."""
        if c.kind == "planet":
            return True
        need = TAROTS.get(c.name, 0) if c.kind == "tarot" else SPECTRALS.get(c.name, 0)
        if need and not cards:
            return False
        if c.name == "death" and len(cards) < 2:
            return False
        if c.name in ("judgement", "wraith", "soul") and len(self.jokers) >= self.joker_slots:
            return False
        if c.name in ("high_priestess", "emperor", "fool") and not from_slot and not self.cons_room():
            return False
        if c.name == "fool" and (self.last_consumable is None or self.last_consumable.name == "fool"):
            return False
        if c.name == "ankh" and not self.jokers:
            return False
        if c.name in ("hex", "ectoplasm", "wheel_of_fortune") and not any(j.edition == "" for j in self.jokers):
            return False
        if c.name == "ankh" and len(self.jokers) >= self.joker_slots:
            return False
        if c.name in ("familiar", "grim", "incantation", "immolate", "sigil", "ouija") and len(cards) < 2:
            return False
        if c.name == "aura" and not any(x.edition == "" for x in cards):
            return False
        return True

    def use_consumable(self, i: int, choose_targets: bool = False):
        """Use consumable i. With choose_targets, a consumable that needs target cards waits for
        apply_targets instead of picking them with the fixed rule (auto_targets)."""
        c = self.consumables.pop(i)
        cards = self.hand if self.state == "SELECTING_HAND" else []
        if choose_targets and c.name in TARGETED and cards:
            self.targeting = {"cons": c, "from_pack": False}
            return
        self.apply_consumable(c, cards)

    # ------------------------------------------------------------------ choosing target cards
    @staticmethod
    def target_range(c: Consumable) -> tuple[int, int]:
        """How many cards a consumable may target: Death converts exactly 2, the rest 1 up to their limit."""
        if c.name == "death":
            return 2, 2
        return 1, TAROTS.get(c.name, SPECTRALS.get(c.name, 0))

    def target_cards(self) -> list[Card]:
        return self.pack_hand if self.targeting["from_pack"] else self.hand

    def apply_targets(self, positions: list[int]):
        tg = self.targeting
        assert tg is not None, "no consumable is waiting for targets"
        cards = self.target_cards()
        lo, hi = self.target_range(tg["cons"])
        assert lo <= len(positions) <= hi and len(set(positions)) == len(positions)
        assert all(0 <= p < len(cards) for p in positions)
        targets = [cards[p] for p in sorted(positions)]
        self.targeting = None
        self.apply_consumable(tg["cons"], cards, targets=targets)
        if tg["from_pack"]:
            self._finish_pack_pick()

    # heuristic targeting shared with the real-game bridge
    @staticmethod
    def card_value(c: Card) -> float:
        v = c.chip_value() + c.rank * 0.1
        v += _VALUE_ENH.get(c.enh, 0)
        v += _VALUE_EDITION[c.edition]
        v += _VALUE_SEAL[c.seal]
        return v

    def main_suit(self) -> int:
        cnt = [0, 0, 0, 0]
        for c in self.full_deck:
            if not c.is_stone:
                cnt[c.suit] += 1
        return max(range(4), key=lambda s: cnt[s])

    def auto_targets(self, name: str, cards: list[Card]) -> list[Card]:
        """Pick target cards for a tarot/spectral. Returned in hand order (left to right)."""
        if not cards:
            return []
        order = {c.uid: i for i, c in enumerate(cards)}
        n = TAROTS.get(name, SPECTRALS.get(name, 0))
        plain = [c for c in cards if c.enh == ""]
        if name in TAROT_SUIT:
            s = TAROT_SUIT[name]
            cand = sorted([c for c in cards if not c.is_stone and c.suit != s], key=lambda c: -c.rank)
            t = cand[:n] or cards[:1]
        elif name == "hanged_man":
            t = sorted(cards, key=self.card_value)[:2]
        elif name == "death":
            best, pair = -1e9, None
            for a in range(len(cards)):
                for b in range(a + 1, len(cards)):
                    gain = self.card_value(cards[b]) - self.card_value(cards[a])
                    if gain > best:
                        best, pair = gain, [cards[a], cards[b]]
            t = pair if pair else cards[:2]
        elif name == "strength":
            t = sorted([c for c in cards if not c.is_stone and c.rank < 14], key=lambda c: -c.rank)[:2] or cards[:1]
        elif name == "tower":
            t = sorted(plain or cards, key=self.card_value)[:1]
        elif name in TAROT_ENH:
            t = sorted(plain or cards, key=lambda c: -self.card_value(c))[:n]
        elif name == "cryptid":
            t = sorted(cards, key=lambda c: -self.card_value(c))[:1]
        elif name == "aura":                               # only a card without an edition
            t = sorted([c for c in cards if c.edition == ""], key=lambda c: -self.card_value(c))[:1]
        else:
            t = sorted([c for c in cards if c.seal == "" and c.edition == ""] or cards,
                       key=lambda c: -self.card_value(c))[:max(1, n)]
        return sorted(t, key=lambda c: order[c.uid])

    def apply_consumable(self, c: Consumable, cards: list[Card], targets: Optional[list[Card]] = None):
        """Apply a consumable; its target cards are `targets` if given, else chosen by auto_targets."""
        n = c.name
        if c.kind == "planet":
            self.hand_levels[PLANETS[n]] += 1
            self.planets_used.add(n)
            for j in self.jokers:
                if j.key == "constellation":
                    j.state["val"] = j.state.get("val", 1.0) + 0.1
            self.last_consumable = c
            return
        if c.kind == "tarot":
            self.tarots_used += 1
            if n != "fool":
                self.last_consumable = c
        t = targets if targets is not None else self.auto_targets(n, cards)
        if n in TAROT_ENH:
            for x in t:
                x.enh = TAROT_ENH[n]
        elif n in TAROT_SUIT:
            for x in t:
                x.suit = TAROT_SUIT[n]
        elif n == "fool":
            if self.last_consumable:
                self.create_consumable(self.last_consumable.kind, name=self.last_consumable.name)
        elif n == "high_priestess":
            for _ in range(2):
                self.create_consumable("planet", src="pri")
        elif n == "emperor":
            for _ in range(2):
                self.create_consumable("tarot", src="emp")
        elif n == "hermit":
            self.money += min(20, max(0, self.money))
        elif n == "temperance":
            self.money += min(50, sum(j.sell_value() for j in self.jokers))
        elif n == "judgement":
            if len(self.jokers) < self.joker_slots:
                self.add_joker(self.random_joker(sticker=False, src="jud"))
        elif n == "wheel_of_fortune":
            cand = [j for j in self.jokers if j.edition == ""]
            if cand and self.roll("wheel_of_fortune") < self.prob(1, 4):
                j = self.xpick("wheel_of_fortune", cand)
                j.edition = self.edition_poll("wheel_of_fortune")
        elif n == "strength":
            for x in t:
                x.rank = 2 if x.rank == 14 else x.rank + 1
        elif n == "hanged_man":
            self.destroy_cards(t)
        elif n == "death":
            if len(t) == 2:
                a, b = t
                a.rank, a.suit, a.enh, a.edition, a.seal = b.rank, b.suit, b.enh, b.edition, b.seal
        # ---- spectrals
        elif n == "black_hole":
            self.hand_levels = [lv + 1 for lv in self.hand_levels]
        elif n in ("talisman", "deja_vu", "trance", "medium"):
            seal = {"talisman": "GOLD", "deja_vu": "RED", "trance": "BLUE", "medium": "PURPLE"}[n]
            for x in t:
                x.seal = seal
        elif n == "aura":
            for x in t:
                if x.edition == "":
                    x.edition = self.edition_poll("aura")
        elif n == "cryptid":
            for x in t:
                for _ in range(2):
                    self.add_card(x.copy(), to_hand=True)
        elif n == "soul":
            if len(self.jokers) < self.joker_slots:
                self.add_joker(self.random_joker(4, sticker=False, src="sou"))
        elif n in ("familiar", "grim", "incantation"):
            if cards:
                self.destroy_card(self.xpick("random_destroy", cards))
            ranks = {"familiar": [11, 12, 13], "grim": [14], "incantation": list(range(2, 11))}[n]
            k = {"familiar": 3, "grim": 2, "incantation": 4}[n]
            for _ in range(k):
                if self.sg is not None:
                    key = n + "_create"
                    rank = ranks[0] if n == "grim" else self.xpick(key, ranks)
                    suit = _SUIT_OF[self.xpick(key, ["S", "H", "D", "C"])]
                    enh = self.xpick("spe_card", ["BONUS", "MULT", "WILD", "GLASS", "STEEL", "GOLD", "LUCKY"])
                    self.add_card(Card(rank, suit, enh=enh), to_hand=True)
                    continue
                self.add_card(Card(self.rng.choice(ranks), self.rng.randrange(4),
                                   enh=self.rng.choice(["BONUS", "MULT", "WILD", "GLASS", "STEEL", "GOLD", "LUCKY"])),
                              to_hand=True)
        elif n == "immolate":
            if self.sg is not None:
                lst = sorted(cards, key=lambda c: c.uid)
                self.sg.streams.shuffle(lst, "immolate")
                self.destroy_cards(lst[:5])
            else:
                self.destroy_cards(self.rng.sample(cards, min(5, len(cards))))
            self.money += 20
        elif n == "sigil":
            s = _SUIT_OF[self.xpick("sigil", ["S", "H", "D", "C"])] if self.sg is not None else self.rng.randrange(4)
            for x in cards:
                x.suit = s
        elif n == "ouija":
            r = (_RANK_OF[self.xpick("ouija", ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"])]
                 if self.sg is not None else self.rng.randrange(2, 15))
            for x in cards:
                x.rank = r
            self.hand_size -= 1
        elif n == "ectoplasm":
            cand = [j for j in self.jokers if j.edition == ""]
            if cand:
                j = self.xpick("ectoplasm", cand)
                j.edition = "NEGATIVE"
                self.joker_slots += 1
            self.hand_size -= 1 + self.flags.get("ecto", 0)
            self.flags["ecto"] = self.flags.get("ecto", 0) + 1
        elif n == "wraith":
            if len(self.jokers) < self.joker_slots:
                self.add_joker(self.random_joker(3, sticker=False, src="wra"))
            self.money = 0                                  # exactly $0, even from debt
        elif n == "hex":
            if self.jokers:
                keep = self.xpick("hex", [j for j in self.jokers if j.edition == ""] or self.jokers)
                for j in list(self.jokers):
                    if j is not keep and not j.eternal:
                        self.destroy_joker(j)
                keep.edition = "POLYCHROME"
        elif n == "ankh":
            if self.jokers:
                src = self.xpick("ankh_choice", self.jokers)
                for j in list(self.jokers):
                    if j is not src and not j.eternal:
                        self.destroy_joker(j)
                cp = Joker(key=src.key, edition="" if src.edition == "NEGATIVE" else src.edition,
                           base_cost=src.base_cost, state=dict(src.state), cost=src.cost)
                self.add_joker(cp)
        self.hand = sort_hand([x for x in self.hand if x in self.full_deck])

    # ------------------------------------------------------------------ packs
    def open_pack(self, kind: str, size: str, return_to: str = "SHOP"):
        shown, picks, _, _ = PACKS[(kind, size)]
        self.pack_kind = kind
        self.pack_picks = picks
        self.pack_return = return_to
        self.pack_cards = []
        self.pack_hand = []
        if self.sg is not None:
            self._exact_open_pack(kind, shown)
            for _ in range(self.count_with_copies("hallucination")):
                if self.sg.streams.pseudorandom("halu%d" % self.ante) < self.prob(1, 2):
                    self.create_consumable("tarot", src="hal")
            if kind in ("arcana", "spectral"):               # the hand is drawn from the top of the deck
                n = min(MAX_HAND, self.effective_hand_size(), len(self.deck))
                self.pack_hand = sort_hand([self.deck.pop() for _ in range(n)])
            self.prev_state = self.state
            self.state = "PACK"
            return
        if kind == "buffoon":
            for _ in range(shown):                          # no duplicates within the pack
                self.pack_cards.append(self.random_joker(exclude={j.key for j in self.pack_cards}))
        elif kind == "celestial":
            opts = [p for p, h in PLANETS.items() if h not in SECRET_HANDS or self.hand_played[h] > 0]
            chosen = self.rng.sample(opts, min(shown, len(opts)))
            if "telescope" in self.vouchers:
                most = max(range(N_HANDS), key=lambda i: self.hand_played[i])
                p = HAND_TO_PLANET[most]
                if p not in chosen:
                    chosen[0] = p
            self.pack_cards = [Consumable("planet", p) for p in chosen]
        elif kind == "arcana":
            self.pack_cards = [Consumable("tarot", t) for t in self.rng.sample(list(TAROTS), shown)]
            if "omen_globe" in self.vouchers:
                for i in range(len(self.pack_cards)):
                    if self.rng.random() < 0.2:
                        self.pack_cards[i] = self.random_consumable(
                            "spectral", {x.name for x in self.pack_cards})
        elif kind == "spectral":
            self.pack_cards = [Consumable("spectral", t) for t in self.rng.sample(SPECTRAL_POOL, shown)]
        elif kind == "standard":
            self.pack_cards = [self.random_playing_card(enhanced=True) for _ in range(shown)]
        # The Soul (arcana/spectral) and Black Hole (celestial/spectral): 0.3% per card, once per run
        # (unless Showman)
        for i, x in enumerate(self.pack_cards):
            if isinstance(x, Consumable):
                for name, kinds in (("soul", ("arcana", "spectral")), ("black_hole", ("celestial", "spectral"))):
                    if kind in kinds and self.rng.random() < 0.003:
                        if self.showman() or not self.flags.get(f"seen_{name}"):
                            self.pack_cards[i] = Consumable("spectral", name)
                            self.flags[f"seen_{name}"] = True
                        break
        for _ in range(self.count_with_copies("hallucination")):
            if self.rng.random() < self.prob(1, 2):
                self.create_consumable("tarot", src="hal")
        if kind in ("arcana", "spectral"):
            pool = list(self.full_deck)
            self.rng.shuffle(pool)
            self.pack_hand = sort_hand(pool[:min(MAX_HAND, self.effective_hand_size())])
        self.prev_state = self.state
        self.state = "PACK"

    def pack_pick_ok(self, i: int) -> bool:
        x = self.pack_cards[i]
        if isinstance(x, Joker):
            return x.edition == "NEGATIVE" or len(self.jokers) < self.joker_slots
        if isinstance(x, Consumable):
            return self.consumable_usable(x, self.pack_hand, from_slot=False)
        return True

    def pack_pick(self, i: int, choose_targets: bool = False):
        x = self.pack_cards.pop(i)
        if isinstance(x, Joker):
            self.add_joker(x)
        elif isinstance(x, Consumable):
            if choose_targets and x.name in TARGETED and self.pack_hand:
                self.targeting = {"cons": x, "from_pack": True}     # finished by apply_targets
                return
            self.apply_consumable(x, self.pack_hand)
        else:
            self.add_card(x)
        self._finish_pack_pick()

    def _finish_pack_pick(self):
        self.pack_hand = sort_hand([c for c in self.pack_hand if c in self.full_deck])
        self.pack_picks -= 1
        if self.pack_picks <= 0 or not self.pack_cards:
            self.close_pack()

    def pack_skip(self):
        for j in self.jokers:
            if j.key == "red_card":
                j.state["val"] = j.state.get("val", 0) + 3
        self.close_pack()

    def close_pack(self):
        if self.sg is not None and self.pack_hand:           # the hand goes back under the deck, in order
            self.deck[0:0] = [c for c in self.pack_hand if c in self.full_deck]
        self.pack_cards = []
        self.pack_hand = []
        self.state = self.pack_return
