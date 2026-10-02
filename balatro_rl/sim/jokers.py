"""Joker definitions.

Each joker is a set of optional hooks called during scoring or at game events.
The scoring context (`ScoreCtx`, see scoring.py) supports an *expected-value* mode
(ctx.rng is None) so the same code gives the agent deterministic score predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .cards import next_uid
from .hands import (PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, QUADS, evaluate)


@dataclass
class JokerDef:
    key: str
    name: str
    rarity: int            # 1 common, 2 uncommon, 3 rare, 4 legendary
    cost: int
    before: Optional[Callable] = None     # (ctx, j, st) scaling before scoring
    card: Optional[Callable] = None       # (ctx, j, st, card) on each scored card
    retrig: Optional[Callable] = None     # (ctx, j, st, card, pos) -> int
    held: Optional[Callable] = None       # (ctx, j, st, card) held-in-hand effect
    main: Optional[Callable] = None       # (ctx, j, st) main joker effect
    after: Optional[Callable] = None      # (game, j) after a hand is played (real play only)
    discard: Optional[Callable] = None    # (game, j, cards) on discard
    end_round: Optional[Callable] = None  # (game, j) -> money earned
    init: Optional[Callable] = None       # (game, j) when created
    blind_select: Optional[Callable] = None  # (game, j) when a blind is selected
    on_sell: Optional[Callable] = None    # (game, j) when this joker is sold
    eternal_ok: bool = True
    perish_ok: bool = True
    copyable: bool = True

    def __deepcopy__(self, memo):
        return self                       # fixed definitions: copies of a game share them


@dataclass
class Joker:
    key: str
    edition: str = ""
    eternal: bool = False
    perishable: Optional[int] = None      # rounds left before debuff (None = not perishable)
    rental: bool = False
    base_cost: int = 0                    # list price (with its edition)
    sell_bonus: int = 0
    cost: Optional[int] = None            # price it counts as bought for (discounts, $1 Rental, $0 from tags)
    debuffed: bool = False
    hidden: bool = False                  # flipped face down (Amber Acorn)
    state: dict = field(default_factory=dict)
    uid: int = field(default_factory=next_uid)

    def __post_init__(self):
        self.d = JOKERS.get(self.key, UNKNOWN)

    def buy_cost(self) -> int:
        return self.base_cost

    def sell_value(self) -> int:
        """As in the game: half the price paid (at least $1), plus what Egg / Gift Card added."""
        paid = self.cost if self.cost is not None else self.base_cost
        return max(1, paid // 2) + self.sell_bonus


EDITION_COST = {"": 0, "FOIL": 2, "HOLO": 3, "POLYCHROME": 5, "NEGATIVE": 5}

JOKERS: dict[str, JokerDef] = {}


def J(key, name, rarity, cost, **hooks):
    JOKERS[key] = JokerDef(key, name, rarity, cost, **hooks)


# ---------------------------------------------------------------- helpers
def _suit_mult(suit, amt):
    def f(ctx, j, st, c):
        if c.has_suit(suit, ctx.smeared):
            ctx.add_mult(amt)
    return f


def _contains(hand, kind, amt):
    def f(ctx, j, st):
        if hand in ctx.contains:
            if kind == "m":
                ctx.add_mult(amt)
            elif kind == "c":
                ctx.add_chips(amt)
            else:
                ctx.x_mult(amt)
    return f


def _const(kind, amt):
    def f(ctx, j, st):
        if kind == "m":
            ctx.add_mult(amt)
        elif kind == "c":
            ctx.add_chips(amt)
        else:
            ctx.x_mult(amt)
    return f


def _val_mult(ctx, j, st):
    ctx.add_mult(st.get("val", 0))


def _val_chips(ctx, j, st):
    ctx.add_chips(st.get("val", 0))


def _val_x(ctx, j, st):
    ctx.x_mult(max(1.0, st.get("val", 1.0)))


def _init_val(v):
    def f(g, j):
        j.state.setdefault("val", v)
    return f


# ---------------------------------------------------------------- common
J("joker", "Joker", 1, 2, main=_const("m", 4))
J("greedy_joker", "Greedy Joker", 1, 5, card=_suit_mult(3, 3))
J("lusty_joker", "Lusty Joker", 1, 5, card=_suit_mult(1, 3))
J("wrathful_joker", "Wrathful Joker", 1, 5, card=_suit_mult(0, 3))
J("gluttenous_joker", "Gluttonous Joker", 1, 5, card=_suit_mult(2, 3))
J("jolly", "Jolly Joker", 1, 3, main=_contains(PAIR, "m", 8))
J("zany", "Zany Joker", 1, 4, main=_contains(TRIPS, "m", 12))
J("mad", "Mad Joker", 1, 4, main=_contains(TWO_PAIR, "m", 10))
J("crazy", "Crazy Joker", 1, 4, main=_contains(STRAIGHT, "m", 12))
J("droll", "Droll Joker", 1, 4, main=_contains(FLUSH, "m", 10))
J("sly", "Sly Joker", 1, 3, main=_contains(PAIR, "c", 50))
J("wily", "Wily Joker", 1, 4, main=_contains(TRIPS, "c", 100))
J("clever", "Clever Joker", 1, 4, main=_contains(TWO_PAIR, "c", 80))
J("devious", "Devious Joker", 1, 4, main=_contains(STRAIGHT, "c", 100))
J("crafty", "Crafty Joker", 1, 4, main=_contains(FLUSH, "c", 80))
J("half", "Half Joker", 1, 5, main=lambda ctx, j, st: ctx.add_mult(20) if len(ctx.played) <= 3 else None)
J("banner", "Banner", 1, 5, main=lambda ctx, j, st: ctx.add_chips(30 * ctx.g.discards_left))
J("mystic_summit", "Mystic Summit", 1, 5,
  main=lambda ctx, j, st: ctx.add_mult(15) if ctx.g.discards_left == 0 else None)
J("misprint", "Misprint", 1, 4,
  main=lambda ctx, j, st: ctx.add_mult(11.5 if ctx.rng is None else ctx.rng.randint(0, 23)))


def _raised_fist_held(ctx, j, st, c):
    """Held-card effect on the lowest-ranked held card (the rightmost one if tied): +2x its chips as mult.
    Scored in the held-card step, so Red seals and Mime retrigger it."""
    held = [x for x in ctx.held if not x.is_stone]
    if not held:
        return
    low = min(range(len(held)), key=lambda i: (held[i].rank, -i))
    if held[low] is c:
        ctx.add_mult(2 * c.chip_value())


J("raised_fist", "Raised Fist", 1, 5, held=_raised_fist_held)
J("scary_face", "Scary Face", 1, 4,
  card=lambda ctx, j, st, c: ctx.add_chips(30) if c.is_face(ctx.pareidolia) else None)
J("abstract", "Abstract Joker", 1, 4, main=lambda ctx, j, st: ctx.add_mult(3 * len(ctx.g.jokers)))
J("even_steven", "Even Steven", 1, 4,
  card=lambda ctx, j, st, c: ctx.add_mult(4) if (not c.is_stone and c.rank <= 10 and c.rank % 2 == 0) else None)
J("odd_todd", "Odd Todd", 1, 4,
  card=lambda ctx, j, st, c: ctx.add_chips(31) if (not c.is_stone and (c.rank == 14 or (c.rank <= 10 and c.rank % 2 == 1))) else None)


def _scholar(ctx, j, st, c):
    if not c.is_stone and c.rank == 14:
        ctx.add_chips(20)
        ctx.add_mult(4)


J("scholar", "Scholar", 1, 4, card=_scholar)
J("supernova", "Supernova", 1, 5,
  main=lambda ctx, j, st: ctx.add_mult(ctx.g.hand_played[ctx.hand] + 1))


def _bus_before(ctx, j, st):
    if any(ctx.played[i].is_face(ctx.pareidolia) for i in ctx.scoring):
        st["val"] = 0
    else:
        st["val"] = st.get("val", 0) + 1


J("ride_the_bus", "Ride the Bus", 1, 6, before=_bus_before, main=_val_mult, init=_init_val(0), eternal_ok=True)
J("blue_joker", "Blue Joker", 1, 5, main=lambda ctx, j, st: ctx.add_chips(2 * len(ctx.g.deck)))


def _green_before(ctx, j, st):
    st["val"] = st.get("val", 0) + 1


def _green_discard(g, j, cards):
    j.state["val"] = max(0, j.state.get("val", 0) - 1)


J("green_joker", "Green Joker", 1, 4, before=_green_before, main=_val_mult, discard=_green_discard, init=_init_val(0))


def _ice_after(g, j):
    j.state["val"] = j.state.get("val", 100) - 5
    if j.state["val"] <= 0:
        g.destroy_joker(j)


J("ice_cream", "Ice Cream", 1, 5, main=_val_chips, after=_ice_after, init=_init_val(100), eternal_ok=False)


def _popcorn_end(g, j):
    j.state["val"] = j.state.get("val", 20) - 4
    if j.state["val"] <= 0:
        g.destroy_joker(j)
    return 0


J("popcorn", "Popcorn", 1, 5, main=_val_mult, end_round=_popcorn_end, init=_init_val(20), eternal_ok=False)


def _walkie(ctx, j, st, c):
    if not c.is_stone and c.rank in (10, 4):
        ctx.add_chips(10)
        ctx.add_mult(4)


J("walkie_talkie", "Walkie Talkie", 1, 4, card=_walkie)
J("smiley", "Smiley Face", 1, 4, card=lambda ctx, j, st, c: ctx.add_mult(5) if c.is_face(ctx.pareidolia) else None)
J("golden", "Golden Joker", 1, 6, end_round=lambda g, j: 4)


def _egg_end(g, j):
    j.sell_bonus += 3
    return 0


J("egg", "Egg", 1, 4, end_round=_egg_end)


def _gros_end(g, j):
    if g.rng.random() < g.prob(1, 6):
        g.destroy_joker(j)
        g.flags["gros_michel_extinct"] = True
    return 0


J("gros_michel", "Gros Michel", 1, 5, main=_const("m", 15), end_round=_gros_end, eternal_ok=False)


def _cav_end(g, j):
    if g.rng.random() < g.prob(1, 1000):
        g.destroy_joker(j)
    return 0


J("cavendish", "Cavendish", 1, 4, main=_const("x", 3), end_round=_cav_end, eternal_ok=False)


def _delayed_end(g, j):
    return 2 * g.discards_left if g.discards_used_round == 0 else 0


J("delayed_grat", "Delayed Gratification", 1, 4, end_round=_delayed_end)


def _business(ctx, j, st, c):
    if c.is_face(ctx.pareidolia):
        ctx.money_chance(2, ctx.g.prob(1, 2))


J("business", "Business Card", 1, 4, card=_business)


def _faceless_discard(g, j, cards):
    if sum(1 for c in cards if c.is_face(g.has("pareidolia"))) >= 3:
        g.money += 5


J("faceless", "Faceless Joker", 1, 4, discard=_faceless_discard)
J("swashbuckler", "Swashbuckler", 1, 4,
  main=lambda ctx, j, st: ctx.add_mult(sum(o.sell_value() for o in ctx.g.jokers if o is not j)))
J("shoot_the_moon", "Shoot the Moon", 1, 5,
  held=lambda ctx, j, st, c: ctx.add_mult(13) if (not c.is_stone and c.rank == 12) else None)


def _photo(ctx, j, st, c):
    """x2 on every trigger of the first face card among the scoring cards (debuffed cards are not faces).
    A Blueprint / Brainstorm copying it gives its own x2 on each of those triggers."""
    first = next((ctx.played[i] for i in ctx.scoring if ctx.played[i].is_face(ctx.pareidolia)), None)
    if first is c:
        ctx.x_mult(2)


J("photograph", "Photograph", 1, 5, card=_photo)


def _runner_before(ctx, j, st):
    if STRAIGHT in ctx.contains:
        st["val"] = st.get("val", 0) + 15


J("runner", "Runner", 1, 5, before=_runner_before, main=_val_chips, init=_init_val(0))


def _square_before(ctx, j, st):
    if len(ctx.played) == 4:
        st["val"] = st.get("val", 0) + 4


J("square", "Square Joker", 1, 4, before=_square_before, main=_val_chips, init=_init_val(0))
J("drunkard", "Drunkard", 1, 4)          # +1 discard (passive, handled in game)
J("chaos", "Chaos the Clown", 1, 4)      # 1 free reroll (passive)
J("fortune_teller", "Fortune Teller", 1, 6,
  main=lambda ctx, j, st: ctx.add_mult(ctx.g.tarots_used))


def _todo_end(g, j):
    j.state["hand"] = g.rng.randrange(0, 9)
    return 0


def _todo_before(ctx, j, st):
    if ctx.hand == st.get("hand", 1):
        ctx.money_now(4)


J("todo_list", "To Do List", 1, 4, before=_todo_before, end_round=_todo_end,
  init=lambda g, j: j.state.setdefault("hand", g.rng.randrange(0, 9)))


def _mail_discard(g, j, cards):
    g.money += 5 * sum(1 for c in cards if not c.is_stone and c.rank == j.state.get("rank", 2))


def _mail_end(g, j):
    j.state["rank"] = g.random_deck_rank()
    return 0


J("mail", "Mail-In Rebate", 1, 4, discard=_mail_discard, end_round=_mail_end,
  init=lambda g, j: j.state.setdefault("rank", g.random_deck_rank()))
J("reserved_parking", "Reserved Parking", 1, 6,
  held=lambda ctx, j, st, c: ctx.money_chance(1, ctx.g.prob(1, 2)) if c.is_face(ctx.pareidolia) else None)
J("hanging_chad", "Hanging Chad", 1, 4,
  retrig=lambda ctx, j, st, c, pos: 2 if pos == 0 else 0)
J("splash", "Splash", 1, 3)              # passive: every played card scores
J("red_card", "Red Card", 1, 5, main=_val_mult, init=_init_val(0))


# ---------------------------------------------------------------- uncommon
def _fib(ctx, j, st, c):
    if not c.is_stone and c.rank in (14, 2, 3, 5, 8):
        ctx.add_mult(8)


J("fibonacci", "Fibonacci", 2, 8, card=_fib)
J("four_fingers", "Four Fingers", 2, 7)
J("hack", "Hack", 2, 6, retrig=lambda ctx, j, st, c, pos: 1 if (not c.is_stone and c.rank in (2, 3, 4, 5)) else 0)
J("dusk", "Dusk", 2, 5, retrig=lambda ctx, j, st, c, pos: 1 if ctx.g.hands_left == 1 else 0)


def _space_before(ctx, j, st):
    if ctx.rng is not None and ctx.rng.random() < ctx.g.prob(1, 4):
        ctx.level_up = ctx.level_up + 1


J("space", "Space Joker", 2, 5, before=_space_before)
J("constellation", "Constellation", 2, 6, main=_val_x, init=_init_val(1.0))


def _hiker(ctx, j, st, c):
    """+5 chips on the card for good, from its next trigger on (so a retrigger in this hand already has it).
    A prediction keeps the gain in the context instead of changing the card."""
    if ctx.rng is not None:
        c.extra_chips += 5
    else:
        ctx.hiked[c.uid] = ctx.hiked.get(c.uid, 0) + 5


J("hiker", "Hiker", 2, 5, card=_hiker)


def _sharp(ctx, j, st):
    if ctx.g.hand_played_round[ctx.hand] > 0:
        ctx.x_mult(3)


J("card_sharp", "Card Sharp", 2, 6, main=_sharp)


def _rocket_end(g, j):
    return j.state.get("val", 1)


J("rocket", "Rocket", 2, 6, end_round=_rocket_end, init=_init_val(1))
J("bull", "Bull", 2, 6, main=lambda ctx, j, st: ctx.add_chips(2 * max(0, ctx.g.money)))
J("bootstraps", "Bootstraps", 2, 7, main=lambda ctx, j, st: ctx.add_mult(2 * (max(0, ctx.g.money) // 5)))
J("smeared", "Smeared Joker", 2, 7)
J("shortcut", "Shortcut", 2, 7)
J("acrobat", "Acrobat", 2, 6, main=lambda ctx, j, st: ctx.x_mult(3) if ctx.g.hands_left == 1 else None)
J("sock_and_buskin", "Sock and Buskin", 2, 6,
  retrig=lambda ctx, j, st, c, pos: 1 if c.is_face(ctx.pareidolia) else 0)


def _blackboard(ctx, j, st):
    # the flush suit test: debuffed cards keep their suit, but a debuffed Wild card is not wild
    if all(c.is_stone is False and (c.flush_suit(0, ctx.smeared) or c.flush_suit(2, ctx.smeared)) for c in ctx.held):
        ctx.x_mult(3)


J("blackboard", "Blackboard", 2, 6, main=_blackboard)
J("mime", "Mime", 2, 5)                  # passive: retrigger held cards


FLOWER_ORDER = (1, 3, 0, 2)          # Hearts, Diamonds, Spades, Clubs (cards.SUITS = S, H, C, D)


def _flower(ctx, j, st):
    """x3 if the scoring cards cover all four suits. As in the game: non-Wild cards first, then Wild cards;
    each card fills the first still-empty suit it matches, in the order Hearts, Diamonds, Spades, Clubs
    (with Smeared Joker a card matches both suits of its colour)."""
    cards = [ctx.played[i] for i in ctx.scoring]
    filled = set()
    for wild_pass in (False, True):
        for c in cards:
            if (c.enh == "WILD") != wild_pass:
                continue
            for s in FLOWER_ORDER:
                if s not in filled and c.has_suit(s, ctx.smeared):
                    filled.add(s)
                    break
    if len(filled) == 4:
        ctx.x_mult(3)


J("flower_pot", "Flower Pot", 2, 6, main=_flower)


SEEING_WILD_ORDER = (2, 3, 0, 1)     # Clubs, Diamonds, Spades, Hearts


def _seeing(ctx, j, st):
    """x2 if the scoring cards hold a Club and a card of another suit. As in the game: debuffed cards count
    for nothing; every other non-Wild card counts for each suit it matches (both suits of its colour with
    Smeared Joker, so one Club is then enough); then each Wild card fills the first suit still missing, in
    the order Clubs, Diamonds, Spades, Hearts (so a lone Wild card is only a Club)."""
    n = [0, 0, 0, 0]
    wilds = 0
    for i in ctx.scoring:
        c = ctx.played[i]
        if c.debuffed or c.is_stone:
            continue
        if c.enh == "WILD":
            wilds += 1
            continue
        for s in range(4):
            if c.has_suit(s, ctx.smeared):
                n[s] += 1
    for _ in range(wilds):
        for s in SEEING_WILD_ORDER:
            if n[s] == 0:
                n[s] = 1
                break
    if n[2] and (n[0] or n[1] or n[3]):
        ctx.x_mult(2)


J("seeing_double", "Seeing Double", 2, 6, main=_seeing)


def _trousers_before(ctx, j, st):
    if TWO_PAIR in ctx.contains:
        st["val"] = st.get("val", 0) + 2


J("trousers", "Spare Trousers", 2, 6, before=_trousers_before, main=_val_mult, init=_init_val(0))


def _ramen_discard(g, j, cards):
    j.state["val"] = j.state.get("val", 2.0) - 0.01 * len(cards)
    if j.state["val"] <= 1.0:
        g.destroy_joker(j)


J("ramen", "Ramen", 2, 6, main=_val_x, discard=_ramen_discard, init=_init_val(2.0), eternal_ok=False)


def _selzer_after(g, j):
    j.state["val"] = j.state.get("val", 10) - 1
    if j.state["val"] <= 0:
        g.destroy_joker(j)


J("selzer", "Seltzer", 2, 6, retrig=lambda ctx, j, st, c, pos: 1, after=_selzer_after, init=_init_val(10), eternal_ok=False)
J("stencil", "Joker Stencil", 2, 8,
  main=lambda ctx, j, st: ctx.x_mult(max(1, ctx.g.joker_slots - len(ctx.g.jokers)
                                         + sum(1 for o in ctx.g.jokers if o.key == "stencil"))))


def _loyal_before(ctx, j, st):
    # val = hands left until the x4 hand; starts at 6, so it fires on hands 6, 12, 18 ... since creation
    st["val"] = st.get("val", 6) - 1 if st.get("val", 6) > 0 else 5


def _loyal_main(ctx, j, st):
    if st.get("val", 6) == 0:
        ctx.x_mult(4)


J("loyalty_card", "Loyalty Card", 2, 5, before=_loyal_before, main=_loyal_main, init=_init_val(6))
J("rough_gem", "Rough Gem", 2, 7, card=lambda ctx, j, st, c: ctx.money_now(1) if c.has_suit(3, ctx.smeared) else None)
J("arrowhead", "Arrowhead", 2, 7, card=lambda ctx, j, st, c: ctx.add_chips(50) if c.has_suit(0, ctx.smeared) else None)
J("onyx_agate", "Onyx Agate", 2, 7, card=lambda ctx, j, st, c: ctx.add_mult(7) if c.has_suit(2, ctx.smeared) else None)
J("bloodstone", "Bloodstone", 2, 7,
  card=lambda ctx, j, st, c: ctx.x_chance(1.5, ctx.g.prob(1, 2)) if c.has_suit(1, ctx.smeared) else None)
J("flash", "Flash Card", 2, 5, main=_val_mult, init=_init_val(0))
J("to_the_moon", "To the Moon", 2, 5)    # passive: extra interest
J("cloud_9", "Cloud 9", 2, 7, end_round=lambda g, j: sum(1 for c in g.full_deck if not c.is_stone and c.rank == 9))
J("satellite", "Satellite", 2, 6, end_round=lambda g, j: len(g.planets_used))
J("pareidolia", "Pareidolia", 2, 5)


def _idol(ctx, j, st, c):
    if not c.is_stone and c.rank == st.get("rank", 14) and c.has_suit(st.get("suit", 0), ctx.smeared):
        ctx.x_mult(2)


def _idol_end(g, j):
    r, s = g.random_deck_card()
    j.state["rank"], j.state["suit"] = r, s
    return 0


J("idol", "The Idol", 2, 6, card=_idol, end_round=_idol_end,
  init=lambda g, j: j.state.update(dict(zip(("rank", "suit"), g.random_deck_card()))) if "rank" not in j.state else None)


# ---------------------------------------------------------------- rare
J("duo", "The Duo", 3, 8, main=_contains(PAIR, "x", 2))
J("trio", "The Trio", 3, 8, main=_contains(TRIPS, "x", 3))
J("family", "The Family", 3, 8, main=_contains(QUADS, "x", 4))
J("order", "The Order", 3, 8, main=_contains(STRAIGHT, "x", 3))
J("tribe", "The Tribe", 3, 8, main=_contains(FLUSH, "x", 2))
J("baron", "Baron", 3, 8, held=lambda ctx, j, st, c: ctx.x_mult(1.5) if (not c.is_stone and c.rank == 13) else None)


def _ancient(ctx, j, st, c):
    if c.has_suit(st.get("suit", 0), ctx.smeared):
        ctx.x_mult(1.5)


def _ancient_end(g, j):
    j.state["suit"] = g.rng.choice([s for s in range(4) if s != j.state.get("suit")])
    return 0


J("ancient", "Ancient Joker", 3, 8, card=_ancient, end_round=_ancient_end,
  init=lambda g, j: j.state.setdefault("suit", g.rng.randrange(4)))


def _obelisk_before(ctx, j, st):
    # the game counts this play first: it resets when this hand is played at least as often as any other,
    # which includes the very first hand of a run (nothing played yet)
    hp = ctx.g.hand_played
    if hp[ctx.hand] == max(hp):
        st["val"] = 1.0
    else:
        st["val"] = st.get("val", 1.0) + 0.2


J("obelisk", "Obelisk", 3, 8, before=_obelisk_before, main=_val_x, init=_init_val(1.0))


def _copy_target(g, j, which=None):
    """The joker a Blueprint (the joker to its right) or Brainstorm (the leftmost joker) copies, or None.
    A copier pointing at another copier copies what that one copies, down the chain; a chain that comes back
    to a copier already on it (Blueprint -> Brainstorm -> the same Blueprint) copies nothing. A debuffed
    joker, or the one Crimson Heart has disabled, gives nothing to copy, and neither does a chain through it.
    `which` is kept for callers; the direction follows from each copier's own key."""
    js = g.jokers
    crimson = g.crimson_disabled if g.boss_active() == "crimson_heart" else -1
    seen = set()
    cur = j
    while True:
        if cur.uid in seen:
            return None
        seen.add(cur.uid)
        i = next((k for k, o in enumerate(js) if o is cur), -1)
        if i < 0:
            return None
        if cur.key == "blueprint":
            t = js[i + 1] if i + 1 < len(js) else None
        else:
            t = js[0]
        if t is None or t is cur or t.debuffed or t.uid == crimson:
            return None
        if t.key not in ("blueprint", "brainstorm"):
            return t if t.d.copyable else None
        cur = t


def _copier(which, hook):
    def f(ctx, j, st, *args):
        t = _copy_target(ctx.g, j, which)
        if t is None:
            return 0
        h = getattr(t.d, hook)
        if h is None:
            return 0
        ctx.copy = True                   # the copy gives the joker's effect; it never grows the joker
        try:
            return h(ctx, t, ctx.st(t), *args)
        finally:
            ctx.copy = False
    return f


# effects the copies repeat beyond scoring (the game's scaling jokers don't scale from a copy, so their
# "before" hooks are not in the list); Mime, Certificate and Hallucination are counted where they act
COPY_BEFORE = frozenset({"space", "dna", "seance", "superposition", "vagabond", "todo_list"})
COPY_DISCARD = frozenset({"faceless", "mail", "burnt"})
COPY_BLIND_SELECT = frozenset({"cartomancer", "marble", "riff_raff", "burglar"})


def _copier_limited(which, hook, allowed):
    def f(ctx, j, st, *args):
        t = _copy_target(ctx.g, j, which)
        if t is None or t.key not in allowed:
            return 0
        return getattr(t.d, hook)(ctx, t, ctx.st(t), *args)
    return f


def _game_copier(which, hook, allowed):
    """For game-event hooks (g, j, *args): run the copied joker's hook as that joker."""
    def f(g, j, *args):
        t = _copy_target(g, j, which)
        if t is None or t.key not in allowed:
            return 0
        return getattr(t.d, hook)(g, t, *args)
    return f


for _k, _w in (("blueprint", "right"), ("brainstorm", "left")):
    J(_k, "Blueprint" if _k == "blueprint" else "Brainstorm", 3, 10,
      before=_copier_limited(_w, "before", COPY_BEFORE),
      card=_copier(_w, "card"), retrig=_copier(_w, "retrig"),
      held=_copier(_w, "held"), main=_copier(_w, "main"),
      discard=_game_copier(_w, "discard", COPY_DISCARD),
      blind_select=_game_copier(_w, "blind_select", COPY_BLIND_SELECT), copyable=False)

# ---------------------------------------------------------------- remaining jokers
J("credit_card", "Credit Card", 1, 1)          # passive: go down to -$20 (game.debt_limit)


def _dagger_select(g, j):
    i = g.jokers.index(j)
    if i + 1 < len(g.jokers):
        r = g.jokers[i + 1]
        if not r.eternal:
            j.state["val"] = j.state.get("val", 0) + 2 * r.sell_value()
            g.destroy_joker(r)


J("ceremonial", "Ceremonial Dagger", 2, 6, blind_select=_dagger_select, main=_val_mult, init=_init_val(0))


def _marble_select(g, j):
    from .cards import Card
    g.add_card(Card(2, g.rng.randrange(4), enh="STONE"))


J("marble", "Marble Joker", 2, 6, blind_select=_marble_select)


def _eight_ball(ctx, j, st, c):
    if not c.is_stone and c.rank == 8 and ctx.real and ctx.rng.random() < ctx.g.prob(1, 4):
        ctx.event("create", "tarot")


J("8_ball", "8 Ball", 1, 5, card=_eight_ball)
J("steel_joker", "Steel Joker", 2, 7,
  main=lambda ctx, j, st: ctx.x_mult(1 + 0.2 * sum(1 for c in ctx.g.full_deck if c.enh == "STEEL")))


def _burglar_select(g, j):
    g.hands_left += 3
    g.discards_left = 0


J("burglar", "Burglar", 2, 6, blind_select=_burglar_select)


def _dna_before(ctx, j, st):
    """First hand of the round, exactly one card played: a copy of it joins the deck and the hand before the
    hand scores. So the copy is a held card of this very hand (rightmost), and Hologram has already grown."""
    if ctx.first_hand and len(ctx.played) == 1:
        c = ctx.played[0]
        cp = c.copy()
        cp.enh = ctx.enh(c)
        ctx.held = ctx.held + [cp]
        for h in ctx.plan.jokers:
            if h.key == "hologram":
                hs = ctx.st(h)
                hs["val"] = hs.get("val", 1.0) + 0.25
        ctx.event("dna", cp)


J("dna", "DNA", 3, 8, before=_dna_before)


def _sixth_before(ctx, j, st):
    c = ctx.played[0]
    if ctx.first_hand and len(ctx.played) == 1 and not c.is_stone and c.rank == 6:
        ctx.event("destroy", c)
        ctx.event("create", "spectral")


J("sixth_sense", "Sixth Sense", 2, 6, before=_sixth_before)


def _superpos_before(ctx, j, st):
    if STRAIGHT in ctx.contains and any(not ctx.played[i].is_stone and ctx.played[i].rank == 14 for i in ctx.scoring):
        ctx.event("create", "tarot")


J("superposition", "Superposition", 1, 4, before=_superpos_before)


def _madness_select(g, j):
    if g.blind_idx < 2:
        j.state["val"] = j.state.get("val", 1.0) + 0.5
        cand = [o for o in g.jokers if o is not j and not o.eternal]
        if cand:
            g.destroy_joker(g.rng.choice(cand))


J("madness", "Madness", 2, 7, blind_select=_madness_select, main=_val_x, init=_init_val(1.0))


def _seance_before(ctx, j, st):
    from .hands import STRAIGHT_FLUSH
    if STRAIGHT_FLUSH in ctx.contains:
        ctx.event("create", "spectral")


J("seance", "Seance", 2, 6, before=_seance_before)


def _riff_select(g, j):
    for _ in range(2):
        if len(g.jokers) < g.joker_slots:
            g.add_joker(g.random_joker(1, sticker=False))


J("riff_raff", "Riff-raff", 1, 6, blind_select=_riff_select)


def _vampire_before(ctx, j, st):
    n = 0
    for c in ctx.scoring_cards():
        if ctx.enh(c) not in ("",) and not c.debuffed:
            ctx.enh_override[c.uid] = ""
            n += 1
    st["val"] = st.get("val", 1.0) + 0.1 * n


J("vampire", "Vampire", 2, 7, before=_vampire_before, main=_val_x, init=_init_val(1.0))
J("hologram", "Hologram", 2, 7, main=_val_x, init=_init_val(1.0))    # grows in game.add_card


def _vagabond_before(ctx, j, st):
    if ctx.g.money <= 4:
        ctx.event("create", "tarot")


J("vagabond", "Vagabond", 3, 8, before=_vagabond_before)


def _midas_before(ctx, j, st):
    for c in ctx.scoring_cards():
        if c.is_face(ctx.pareidolia):
            ctx.enh_override[c.uid] = "GOLD"


J("midas_mask", "Midas Mask", 2, 7, before=_midas_before)


def _luchador_sell(g, j):
    if g.boss_active():
        g.disable_boss()


J("luchador", "Luchador", 2, 5, on_sell=_luchador_sell, eternal_ok=False)


def _gift_end(g, j):
    for o in g.jokers:
        o.sell_bonus += 1
    for c in g.consumables:
        c.sell_bonus += 1
    return 0


J("gift", "Gift Card", 2, 6, end_round=_gift_end)


def _bean_end(g, j):
    j.state["val"] = j.state.get("val", 5) - 1
    if j.state["val"] <= 0:
        g.destroy_joker(j)
    return 0


J("turtle_bean", "Turtle Bean", 2, 6, end_round=_bean_end, init=_init_val(5), eternal_ok=False)
J("erosion", "Erosion", 2, 6,
  main=lambda ctx, j, st: ctx.add_mult(4 * max(0, ctx.g.starting_deck_size - len(ctx.g.full_deck))))
J("hallucination", "Hallucination", 1, 4)          # game.open_pack
J("juggler", "Juggler", 1, 4)                      # passive +1 hand size
J("stone", "Stone Joker", 2, 6,
  main=lambda ctx, j, st: ctx.add_chips(25 * sum(1 for c in ctx.g.full_deck if c.enh == "STONE")))


def _lucky_cat_card(ctx, j, st, c):
    """Grows as the Lucky cards trigger (before any joker's own effect); a copy of it doesn't grow it."""
    if not ctx.copy and ctx.lucky_now:
        st["val"] = st.get("val", 1.0) + 0.25 * ctx.lucky_now


def _lucky_cat_main(ctx, j, st):
    ctx.x_mult(st.get("val", 1.0))


J("lucky_cat", "Lucky Cat", 2, 6, card=_lucky_cat_card, main=_lucky_cat_main, init=_init_val(1.0))


J("baseball", "Baseball Card", 3, 8)      # scoring.score_hand: x1.5 on each Uncommon joker's effect


def _cola_sell(g, j):
    g.add_tag("double")


J("diet_cola", "Diet Cola", 2, 6, on_sell=_cola_sell, eternal_ok=False)


def _trading_discard(g, j, cards):
    if g.discards_used_round == 0 and len(cards) == 1:
        g.money += 3
        g.flags["trading_destroy"] = cards[0]


J("trading", "Trading Card", 2, 6, discard=_trading_discard)


def _castle_discard(g, j, cards):
    j.state["val"] = j.state.get("val", 0) + 3 * sum(1 for c in cards if c.has_suit(j.state.get("suit", 0)))


def _castle_end(g, j):
    j.state["suit"] = g.random_deck_card()[1]
    return 0


J("castle", "Castle", 2, 6, discard=_castle_discard, end_round=_castle_end, main=_val_chips,
  init=lambda g, j: j.state.update({"val": j.state.get("val", 0), "suit": j.state.get("suit", g.random_deck_card()[1])}))
J("campfire", "Campfire", 3, 9, main=_val_x, init=_init_val(1.0))   # grows in game.sell_*
J("ticket", "Golden Ticket", 1, 5, card=lambda ctx, j, st, c: ctx.money_now(4) if ctx.enh(c) == "GOLD" else None)
J("mr_bones", "Mr. Bones", 2, 5, eternal_ok=False)                  # game.play
J("troubadour", "Troubadour", 2, 6)                                 # passive +2 hand size, -1 hand
J("certificate", "Certificate", 2, 6)                               # game.select_blind
J("throwback", "Throwback", 2, 6, main=lambda ctx, j, st: ctx.x_mult(1 + 0.25 * ctx.g.blinds_skipped))
J("glass", "Glass Joker", 2, 6, main=_val_x, init=_init_val(1.0))  # grows in game.destroy_cards
J("ring_master", "Showman", 2, 5)                                   # passive: duplicates allowed


def _wee(ctx, j, st, c):
    if not ctx.copy and not c.is_stone and c.rank == 2:      # a copy gives its chips but doesn't grow it
        st["val"] = st.get("val", 0) + 8


J("wee", "Wee Joker", 3, 8, card=_wee, main=_val_chips, init=_init_val(0))
J("merry_andy", "Merry Andy", 2, 7)                                 # passive +3 discards, -1 hand size
J("oops", "Oops! All 6s", 2, 4)                                     # passive: doubles probabilities
J("matador", "Matador", 2, 7)                                       # game.play


def _road_discard(g, j, cards):
    j.state["val"] = j.state.get("val", 1.0) + 0.5 * sum(1 for c in cards if not c.is_stone and c.rank == 11)


def _road_end(g, j):
    j.state["val"] = 1.0
    return 0


J("hit_the_road", "Hit the Road", 3, 8, discard=_road_discard, end_round=_road_end, main=_val_x, init=_init_val(1.0))
J("stuntman", "Stuntman", 3, 7, main=_const("c", 250))              # passive -2 hand size


def _invisible_end(g, j):
    j.state["val"] = j.state.get("val", 0) + 1
    return 0


def _invisible_sell(g, j):
    if j.state.get("val", 0) >= 2:
        cand = [o for o in g.jokers if o is not j]
        if cand:
            src = g.rng.choice(cand)
            g.add_joker(Joker(src.key, edition="" if src.edition == "NEGATIVE" else src.edition,
                              base_cost=src.base_cost, state=dict(src.state), cost=src.cost), force=True)


J("invisible", "Invisible Joker", 3, 8, end_round=_invisible_end, on_sell=_invisible_sell, init=_init_val(0),
  eternal_ok=False)
J("drivers_license", "Driver's License", 3, 7,
  main=lambda ctx, j, st: ctx.x_mult(3) if sum(1 for c in ctx.g.full_deck if c.enh) >= 16 else None)


def _carto_select(g, j):
    g.create_consumable("tarot")


J("cartomancer", "Cartomancer", 2, 6, blind_select=_carto_select)
J("astronomer", "Astronomer", 2, 8)                                 # shop prices


def _burnt_discard(g, j, cards):
    if g.discards_used_round == 0:
        h = evaluate(cards, g.has("four_fingers"), g.has("shortcut"), g.has("smeared")).hand
        g.hand_levels[h] += 1


J("burnt", "Burnt Joker", 3, 8, discard=_burnt_discard)


def _yorick_discard(g, j, cards):
    left = j.state.get("left", 23) - len(cards)
    while left <= 0:
        j.state["val"] = j.state.get("val", 1.0) + 1
        left += 23
    j.state["left"] = left


# legendaries (only from The Soul)
J("caino", "Canio", 4, 20, main=_val_x, init=_init_val(1.0))       # grows in game.destroy_cards
J("triboulet", "Triboulet", 4, 20,
  card=lambda ctx, j, st, c: ctx.x_mult(2) if (not c.is_stone and c.rank in (12, 13)) else None)
J("yorick", "Yorick", 4, 20, discard=_yorick_discard, main=_val_x,
  init=lambda g, j: j.state.update({"val": j.state.get("val", 1.0), "left": j.state.get("left", 23)}))
J("chicot", "Chicot", 4, 20)                                        # passive: disables boss blinds
J("perkeo", "Perkeo", 4, 20)                                        # game.leave_shop

UNKNOWN = JokerDef("unknown", "Unknown Joker", 1, 5)

# jokers that grow as the run goes on can't get the Perishable sticker (as in the game)
NO_PERISHABLE = ("ride_the_bus", "green_joker", "runner", "square", "red_card", "constellation", "rocket",
                 "trousers", "flash", "obelisk", "ceremonial", "madness", "vampire", "hologram", "lucky_cat",
                 "castle", "glass", "wee")
for _k in NO_PERISHABLE:
    JOKERS[_k].perish_ok = False

# Jokers that should not appear in the shop pool (for the simulator)
NOT_IN_POOL = {"cavendish"}   # only after Gros Michel goes extinct

JOKER_KEYS = sorted(JOKERS)
JOKER_INDEX = {k: i for i, k in enumerate(JOKER_KEYS)}
