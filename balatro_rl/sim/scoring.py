"""Hand scoring. Follows Balatro's order: "before" effects -> base -> scored cards (with
retriggers) -> held cards -> jokers (editions wrap each joker)."""
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
        self.mime = sum(1 for j in self.jokers if j.key == "mime")
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


def score_hand(g: "Game", played: list[Card], held: list[Card], rng: Optional[random.Random] = None,
               commit: bool = False, plan: Optional[Plan] = None) -> tuple[float, ScoreCtx]:
    """Return (score, ctx). rng=None -> expected value, no side effects.
    commit=True applies joker scaling / hiker / glass breaks / enhancement changes to the game."""
    if plan is None:
        plan = Plan(g)
    ctx = ScoreCtx(g, played, held, rng, plan)
    boss = plan.boss

    # --- "before" effects (scaling jokers, Vampire, Midas Mask, To-Do money, Space Joker ...)
    for j, f in plan.before:
        f(ctx, j, ctx.st(j))

    level = g.hand_levels[ctx.hand] + ctx.level_up
    if boss == "arm":
        level = max(1, level - 1)
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
                p2 = g.prob(1, 15)
                if rng is None:
                    ctx.money += 20 * p2
                    ctx.lucky_hits += p2
                elif rng.random() < p2:
                    ctx.money += 20
                    ctx.lucky_hits += 1
            if c.seal == "GOLD":
                ctx.money_now(3)
            if c.edition == "FOIL":
                ctx.add_chips(50)
            elif c.edition == "HOLO":
                ctx.add_mult(10)
            for j, f in plan.card:
                f(ctx, j, ctx.st(j), c)
            if e == "GLASS":
                ctx.x_mult(2)
            if c.edition == "POLYCHROME":
                ctx.x_mult(1.5)
        if e == "GLASS" and rng is not None and rng.random() < g.prob(1, 4):
            glass_broken.append(c)

    # --- held-in-hand effects
    if plan.held or any(c.enh == "STEEL" for c in held):
        for c in held:
            if c.debuffed:
                continue
            reps = 1 + (1 if c.seal == "RED" else 0) + plan.mime
            for _ in range(reps):
                if c.enh == "STEEL":
                    ctx.x_mult(1.5)
                for j, f in plan.held:
                    f(ctx, j, ctx.st(j), c)

    # --- Observatory: planets held give X1.5 for their hand
    if "observatory" in g.vouchers:
        from .items import PLANETS
        for c in g.consumables:
            if c.kind == "planet" and PLANETS.get(c.name) == ctx.hand:
                ctx.x_mult(1.5)

    # --- jokers
    for j, f, ed in plan.main:
        if ed == "FOIL":
            ctx.add_chips(50)
        elif ed == "HOLO":
            ctx.add_mult(10)
        if f:
            f(ctx, j, ctx.st(j))
        if ed == "POLYCHROME":
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
