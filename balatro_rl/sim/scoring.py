"""Hand scoring, in the game's order:
  "before" effects (scaling jokers, Vampire, Midas Mask ...)
  -> base chips / mult of the hand (after The Arm lowered its level, then Space Joker's level-up)
  -> each scored card, repeated per retrigger: chips (base, then Bonus), Mult / Lucky +20 mult, Gold seal /
     Lucky money, Glass x2, edition (Foil +50 / Holo +10 / Polychrome x1.5), then the jokers' per-card effects
  -> each held card (Steel, Raised Fist, Baron, Shoot the Moon ...), repeated by Red seals and Mime only when
     it had an effect
  -> each joker: edition Foil / Holo, its effect, Baseball Card (x1.5 per Baseball if it is Uncommon), then
     edition Polychrome
  -> held planets (Observatory).

The Hook discards 2 random held cards before any of this. A real play passes the held cards that are left;
an expected-value prediction (rng=None) averages over every pair the Hook could take, when the held cards
can change the score at all (hook_variants)."""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Optional

from .cards import Card
from .hands import evaluate, hand_base

if TYPE_CHECKING:
    from .game import Game


class Plan:
    """Per-state precomputation (which jokers have which hooks); reused across many predictions."""

    def __init__(self, g: "Game"):
        boss = g.boss_active()
        self.boss = boss
        self.jokers = [j for j in g.jokers
                       if not j.debuffed and not (boss == "crimson_heart" and j.uid == g.crimson_disabled)]
        keys = {j.key for j in self.jokers}
        self.four_fingers = "four_fingers" in keys
        self.shortcut = "shortcut" in keys
        self.smeared = "smeared" in keys
        self.pareidolia = "pareidolia" in keys
        self.splash = "splash" in keys
        self.blackboard = "blackboard" in keys      # its effect reads the held cards
        # Mime, and Blueprint / Brainstorm copying Mime, each retrigger held cards once more
        self.mime = sum(1 for j in self.jokers if j.key == "mime" or copies(g, j) == "mime")
        # Baseball Card, and copies of it: x1.5 each on every Uncommon joker's effect
        self.baseball = sum(1 for j in self.jokers if j.key == "baseball" or copies(g, j) == "baseball")
        self.before = [(j, j.d.before) for j in self.jokers if j.d.before]
        self.card = [(j, j.d.card) for j in self.jokers if j.d.card]
        self.retrig = [(j, j.d.retrig) for j in self.jokers if j.d.retrig]
        self.held = [(j, j.d.held) for j in self.jokers if j.d.held]
        self.main = [(j, j.d.main, j.edition) for j in self.jokers]


class ScoreCtx:
    def __init__(self, g: "Game", played: list[Card], held: list[Card], rng: Optional[random.Random],
                 plan: Plan):
        self.g = g
        self.played = played
        self.held = held
        self.rng = rng
        self.plan = plan
        self.four_fingers = plan.four_fingers
        self.shortcut = plan.shortcut
        self.smeared = plan.smeared
        self.pareidolia = plan.pareidolia
        res = evaluate(played, self.four_fingers, self.shortcut, self.smeared)
        self.hand = res.hand
        self.contains = res.contains
        self.scoring = list(range(len(played))) if plan.splash else res.scoring
        self.chips = 0.0
        self.mult = 0.0
        self.money = 0.0
        self.level_up = 0
        self.jstate: dict[int, dict] = {}
        self.trigger_id = 0
        self.enh_override: dict[int, str] = {}     # Vampire / Midas Mask change enhancements "before"
        self.lucky_hits = 0.0
        self.events: list[tuple] = []              # creations etc., applied only on a real play
        self.first_hand = sum(g.hand_played_round) == 0

    @property
    def real(self) -> bool:
        return self.rng is not None

    # joker state (copy-on-read so predictions don't mutate the game)
    def st(self, j) -> dict:
        s = self.jstate.get(j.uid)
        if s is None:
            s = dict(j.state)
            self.jstate[j.uid] = s
        return s

    def enh(self, c: Card) -> str:
        return self.enh_override.get(c.uid, c.enh)

    def scoring_cards(self) -> list[Card]:
        return [self.played[i] for i in self.scoring]

    def add_chips(self, x):
        self.chips += x

    def add_mult(self, x):
        self.mult += x

    def x_mult(self, x):
        self.mult *= x

    def x_chance(self, x, p):
        if self.rng is None:
            self.mult *= 1 + (x - 1) * p
        elif self.rng.random() < p:
            self.mult *= x

    def money_now(self, x):
        self.money += x

    def money_chance(self, x, p):
        if self.rng is None:
            self.money += x * p
        elif self.rng.random() < p:
            self.money += x

    def event(self, *e):
        if self.real:
            self.events.append(e)


def copies(g: "Game", j) -> str:
    """The key of the joker a Blueprint / Brainstorm copies ("" for other jokers or nothing to copy)."""
    if j.key not in ("blueprint", "brainstorm"):
        return ""
    from .jokers import _copy_target
    t = _copy_target(g, j, "right" if j.key == "blueprint" else "left")
    return t.key if t is not None else ""


def hook_variants(plan: Plan, held: list[Card]) -> Optional[list[list[Card]]]:
    """The Hook discards 2 random held cards before the hand scores. When that can change the score (a
    joker's held-card effect, a held Steel card, or Blackboard), the held cards left after each equally
    likely discard, in a fixed order; otherwise None. (The compiled scorer walks the same pairs.)"""
    if plan.boss != "hook" or not held:
        return None
    if not (plan.held or plan.blackboard or any(c.enh == "STEEL" for c in held)):
        return None
    n = len(held)
    if n <= 2:
        return [[]]
    return [[c for i, c in enumerate(held) if i != a and i != b] for a in range(n) for b in range(a + 1, n)]


def _hook_expected(g: "Game", played: list[Card], variants: list, plan: Plan, arm_applied: bool):
    """Expected score under The Hook: the mean over its possible discards of chips x mult (and of the
    chips, mult and money left in ctx)."""
    total = chips = mult = money = 0.0
    first = None
    for h in variants:
        _, ctx = score_hand(g, played, h, None, False, plan, arm_applied, hooked=True)
        total += ctx.chips * ctx.mult
        chips += ctx.chips
        mult += ctx.mult
        money += ctx.money
        if first is None:
            first = ctx
    n = len(variants)
    first.chips, first.mult, first.money = chips / n, mult / n, money / n
    return math.floor(total / n), first


def score_hand(g: "Game", played: list[Card], held: list[Card], rng: Optional[random.Random] = None,
               commit: bool = False, plan: Optional[Plan] = None, arm_applied: bool = False,
               hooked: bool = False) -> tuple[float, ScoreCtx]:
    """Return (score, ctx). rng=None -> expected value, no side effects.
    commit=True applies joker scaling / hiker / glass breaks / enhancement changes to the game.
    arm_applied: The Arm already lowered the hand's level in g.hand_levels (a real play does that first).
    hooked: The Hook already took its 2 cards out of `held` (a real play, or one case of the average)."""
    if plan is None:
        plan = Plan(g)
    if rng is None and not hooked:
        variants = hook_variants(plan, held)
        if variants is not None:
            return _hook_expected(g, played, variants, plan, arm_applied)
    ctx = ScoreCtx(g, played, held, rng, plan)
    boss = plan.boss

    # --- "before" effects (scaling jokers, Vampire, Midas Mask, To-Do money, Space Joker ...)
    for j, f in plan.before:
        f(ctx, j, ctx.st(j))

    level = g.hand_levels[ctx.hand]
    if boss == "arm" and not arm_applied and level > 1:     # The Arm first, then Space Joker's level-up
        level -= 1
    level += ctx.level_up
    chips, mult = hand_base(ctx.hand, level)
    if boss == "flint":
        chips, mult = max(0, int(chips / 2 + 0.5)), max(1, int(mult / 2 + 0.5))
    ctx.chips, ctx.mult = float(chips), float(mult)

    glass_broken: list[Card] = []

    # --- scored cards
    for pos, idx in enumerate(ctx.scoring):
        c = played[idx]
        if c.debuffed:
            continue
        e = ctx.enh(c)
        reps = 1 + (1 if c.seal == "RED" else 0)
        for j, f in plan.retrig:
            reps += f(ctx, j, ctx.st(j), c, pos) or 0
        for _ in range(reps):
            ctx.trigger_id += 1
            # the card itself: chips (base, then Bonus), mult (Mult / Lucky), money (Gold seal / Lucky),
            # Glass, then its edition
            ctx.add_chips((50 if e == "STONE" else c.chip_value()) + c.extra_chips)
            if e == "BONUS":
                ctx.add_chips(30)
            elif e == "MULT":
                ctx.add_mult(4)
            elif e == "LUCKY":
                p = g.prob(1, 5)
                if rng is None:
                    ctx.add_mult(20 * p)
                    ctx.lucky_hits += p
                elif rng.random() < p:
                    ctx.add_mult(20)
                    ctx.lucky_hits += 1
            if c.seal == "GOLD":
                ctx.money_now(3)
            if e == "LUCKY":
                p2 = g.prob(1, 15)
                if rng is None:
                    ctx.money += 20 * p2
                    ctx.lucky_hits += p2
                elif rng.random() < p2:
                    ctx.money += 20
                    ctx.lucky_hits += 1
            if e == "GLASS":
                ctx.x_mult(2)
            if c.edition == "FOIL":
                ctx.add_chips(50)
            elif c.edition == "HOLO":
                ctx.add_mult(10)
            elif c.edition == "POLYCHROME":
                ctx.x_mult(1.5)
            # then the jokers' per-card effects
            for j, f in plan.card:
                f(ctx, j, ctx.st(j), c)
        if e == "GLASS" and rng is not None and rng.random() < g.prob(1, 4):
            glass_broken.append(c)

    # --- held-in-hand effects; Red seals and Mime repeat a card only if it had an effect
    if plan.held or any(c.enh == "STEEL" for c in held):
        for c in held:
            if c.debuffed:
                continue
            before = (ctx.chips, ctx.mult, ctx.money, len(ctx.events))
            if c.enh == "STEEL":
                ctx.x_mult(1.5)
            for j, f in plan.held:
                f(ctx, j, ctx.st(j), c)
            if c.enh != "STEEL" and (ctx.chips, ctx.mult, ctx.money, len(ctx.events)) == before:
                continue
            for _ in range((1 if c.seal == "RED" else 0) + plan.mime):
                if c.enh == "STEEL":
                    ctx.x_mult(1.5)
                for j, f in plan.held:
                    f(ctx, j, ctx.st(j), c)

    # --- jokers: edition Foil / Holo, the joker's effect, Baseball Card, edition Polychrome
    for j, f, ed in plan.main:
        if ed == "FOIL":
            ctx.add_chips(50)
        elif ed == "HOLO":
            ctx.add_mult(10)
        if f:
            f(ctx, j, ctx.st(j))
        if plan.baseball and j.d.rarity == 2:
            for _ in range(plan.baseball):
                ctx.x_mult(1.5)
        if ed == "POLYCHROME":
            ctx.x_mult(1.5)

    # --- Observatory: held planets give X1.5 for their hand, after the last joker
    if "observatory" in g.vouchers:
        from .items import PLANETS
        for c in g.consumables:
            if c.kind == "planet" and PLANETS.get(c.name) == ctx.hand:
                ctx.x_mult(1.5)

    if g.deck_type == "PLASMA":
        avg = (ctx.chips + ctx.mult) / 2
        ctx.chips = ctx.mult = avg
    score = math.floor(ctx.chips * ctx.mult)

    if commit:
        for j in g.jokers:
            if j.uid in ctx.jstate:
                j.state = {k: v for k, v in ctx.jstate[j.uid].items() if not k.startswith("_")}
        if ctx.level_up:
            g.hand_levels[ctx.hand] += ctx.level_up
        for c in played:
            if c.uid in ctx.enh_override:
                c.enh = ctx.enh_override[c.uid]
        if glass_broken:
            g.destroy_cards(glass_broken)
    return score, ctx
