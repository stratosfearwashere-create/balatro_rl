"""The game as the unified agent sees it: structured actions, loop counters, and copies of the game whose
hidden information is resampled.

Hidden information in the simulator is (1) the order of the draw pile, (2) the identity of face-down
cards in hand, (3) the order of face-down jokers (Amber Acorn) and (4) everything the game's random number
generator will produce later: shop and pack contents, bosses and tags of later antes, and the outcomes of
random effects (Wheel of Fortune, Aura, Familiar, lucky cards ...). `determinize` makes a copy in which all
four are redrawn from the agent's own random generator. The redraw starts from a canonical order (sorted
by card id), so the copy does not depend on the true order either: two games that differ only in hidden
information give identical copies for the same agent seed. Nothing in the agent reads the real game's
draw order or random generator; everything that simulates the future works on such copies.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..sim.game import Game

MAX_REROLLS_PER_SHOP = 8
MAX_MOVES_PER_PHASE = 4
MAX_STEPS = 2500

KINDS = ("play", "discard", "use", "sell_joker", "sell_cons", "move_joker", "buy", "buy_pack", "voucher",
         "reroll", "leave", "pick", "pack_skip", "select", "skip", "reroll_boss")
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}
PHASES = ("BLIND_SELECT", "SELECTING_HAND", "SHOP", "PACK")


@dataclass(frozen=True)
class Action:
    """kind: one of KINDS. idx: the consumable / joker / shop / pack slot it acts on.
    cards: hand positions for play and discard, in the order they are played (joker and card effects
    trigger left to right); target positions (in the hand, or in the pack's hand) for use and pick.
    to: destination slot for move_joker."""
    kind: str
    idx: int = -1
    cards: tuple = ()
    to: int = -1

    def __str__(self):
        s = self.kind
        if self.idx >= 0:
            s += f" {self.idx}"
        if self.cards:
            s += " [" + " ".join(map(str, self.cards)) + "]"
        if self.to >= 0:
            s += f" -> {self.to}"
        return s


def apply(g: Game, a: Action):
    k = a.kind
    if k == "play":
        g.play(list(a.cards))
    elif k == "discard":
        g.discard(list(a.cards))
    elif k == "use":
        g.use_consumable(a.idx, choose_targets=True)
        if g.targeting is not None:
            g.apply_targets(list(a.cards) or _auto(g))
    elif k == "pick":
        g.pack_pick(a.idx, choose_targets=True)
        if g.targeting is not None:
            g.apply_targets(list(a.cards) or _auto(g))
    elif k == "sell_joker":
        g.sell_joker(a.idx)
    elif k == "sell_cons":
        g.sell_consumable(a.idx)
    elif k == "move_joker":
        j = g.jokers.pop(a.idx)
        g.jokers.insert(a.to, j)
    elif k == "buy":
        g.buy_card(a.idx)
    elif k == "buy_pack":
        g.buy_pack(a.idx)
    elif k == "voucher":
        g.buy_voucher()
    elif k == "reroll":
        g.reroll()
    elif k == "leave":
        g.leave_shop()
    elif k == "pack_skip":
        g.pack_skip()
    elif k == "select":
        g.select_blind()
    elif k == "skip":
        g.skip_blind()
    elif k == "reroll_boss":
        g.reroll_boss()
    else:
        raise ValueError(f"unknown action kind {k}")


def _auto(g: Game) -> list[int]:
    cards = g.target_cards()
    lo, _ = g.target_range(g.targeting["cons"])
    pos = [cards.index(c) for c in g.auto_targets(g.targeting["cons"].name, cards)]
    return pos if len(pos) >= lo else list(range(min(lo, len(cards))))


class World:
    """A game plus the counters that stop reroll / joker-reordering loops (as in env.BalatroEnv)."""
    __slots__ = ("g", "rerolls", "moves", "steps")

    def __init__(self, g: Game, rerolls: int = 0, moves: int = 0, steps: int = 0):
        self.g, self.rerolls, self.moves, self.steps = g, rerolls, moves, steps

    @property
    def done(self) -> bool:
        return self.g.done or self.steps >= MAX_STEPS

    def step(self, a: Action):
        g = self.g
        prev = g.state
        apply(g, a)
        self.steps += 1
        if a.kind == "reroll":
            self.rerolls += 1
        elif a.kind == "move_joker":
            self.moves += 1
        if g.state != prev:
            if g.state == "SHOP" and prev != "PACK":
                self.rerolls = 0
            if {prev, g.state} != {"SHOP", "PACK"}:
                self.moves = 0
        if g.state == "SELECTING_HAND" and not g.hand and not g.done:
            g.state = "GAME_OVER"          # nothing left to play (as the base environment does)

    def copy(self) -> "World":
        return World(self.g.clone(), self.rerolls, self.moves, self.steps)

    def determinize(self, rng: random.Random) -> "World":
        return World(determinize(self.g, rng), self.rerolls, self.moves, self.steps)


def redraw(g: Game, rng: random.Random):
    """Resample the draw pile's order and the face-down cards in hand, from a canonical order."""
    hidden = [i for i, c in enumerate(g.hand) if c.hidden]
    pool = sorted(list(g.deck) + [g.hand[i] for i in hidden], key=lambda c: c.uid)
    for c in pool:
        c.hidden = False
    rng.shuffle(pool)
    for i in hidden:
        c = pool.pop()
        c.hidden = True
        g.hand[i] = c
    g.deck = pool
    if any(j.hidden for j in g.jokers):               # Amber Acorn: the order of face-down jokers
        js = sorted(g.jokers, key=lambda j: j.uid)
        rng.shuffle(js)
        g.jokers = js


def determinize(g: Game, rng: random.Random) -> Game:
    """A copy of g with every piece of hidden information redrawn from `rng` (see the module doc)."""
    w = g.clone()
    w.rng = random.Random(rng.getrandbits(64))
    if w.state == "SELECTING_HAND":
        redraw(w, w.rng)
    return w


def card_sig(c) -> tuple:
    if c.hidden:
        return ("?",)
    return (c.rank, c.suit, c.enh, c.edition, c.seal, c.extra_chips, c.debuffed)


def deck_sig(g: Game) -> tuple:
    """The full deck's contents (public, whatever is face down), as a sorted multiset."""
    return tuple(sorted((c.rank, c.suit, c.enh, c.edition, c.seal, c.extra_chips) for c in g.full_deck))


def joker_sig(j) -> tuple:
    if j.hidden:
        return ("?",)
    return (j.key, j.edition, j.eternal, j.perishable, j.rental, j.debuffed, j.sell_value(),
            tuple(sorted((k, v) for k, v in j.state.items() if not k.startswith("_"))))


def infoset_key(w: World) -> tuple:
    """Everything a player can see (and nothing they can't): two games with the same key are the same
    decision for the agent. Used to merge chance outcomes in the search."""
    g = w.g
    shop = ()
    if g.state == "SHOP":
        shop = (tuple((it.key, it.cost, joker_sig(it.joker) if it.joker else None,
                       card_sig(it.card) if it.card else None) for it in g.shop),
                tuple((it.key, it.cost) for it in g.shop_packs),
                (g.shop_voucher.key, g.shop_voucher.cost) if g.shop_voucher else None,
                g.reroll_cost, g.free_rerolls)
    pack = ()
    if g.state == "PACK":
        pack = (g.pack_kind, g.pack_picks,
                tuple(joker_sig(x) if hasattr(x, "state") else (x.key if hasattr(x, "kind") else card_sig(x))
                      for x in g.pack_cards),
                tuple(card_sig(c) for c in g.pack_hand))
    return (g.state, g.ante, g.blind_idx, g.boss, tuple(g.tags_offered), g.money, g.chips, g.target,
            g.hands_left, g.discards_left, tuple(card_sig(c) for c in g.hand),
            tuple(joker_sig(j) for j in g.jokers), tuple((c.key, c.negative) for c in g.consumables),
            tuple(g.hand_levels), deck_sig(g), tuple(sorted(g.vouchers)),
            tuple(g.pending_tags), g.joker_slots, g.consumable_slots, g.hand_size, g.forced_pos(),
            g.furthest_blind, w.rerolls, w.moves, shop, pack)
