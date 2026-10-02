"""Stage 3, the shop module: a price for every option outside a round, used as a graded prior.

    price(option) = change in build strength + economy term - money cost x m(ante, money)

computed as the difference of a state value before and after the option is applied to a copy of the game:

    W(s) = S(s) + E(s) + U(money)
    S    build strength: the Stage 2 strength score (rewards/strength.py: clear chances of this ante's boss and
         the next three, with the current boss's real target) + log_weight x log(mean best-hand score)
         + deck_weight x mean card quality; or Phi itself (`value: phi`)
    E    economy: income per round of the economy jokers held (ECON_INCOME) x econ_rounds, and a one-off value
         for vouchers that do not show in the score (VOUCHER_DOLLARS), both in dollars
    U    money: dollar x a(ante) x (money + interest_rounds x interest steps held), so that
         m(ante, money) = (U(money) - U(money - cost)) / cost is high when spending drops money under the next
         interest threshold, and a(ante) falls linearly to m_floor at ante m_zero_ante

How each kind of option is priced (ShopPricer._price):
    buy joker        W after - W before; with the copier rule, the best slot for it
    sell joker       W after the sale, or, when the slots are full, the best "sell this one, then buy that
                     shop joker" pair if that is worth more
    planet           level that hand: bought for what using it would add; used for what it adds
    tarot, spectral  in a pack: applied to each valid target set the candidate list holds (actions.py); in the
                     shop: what using it adds if it needs no cards (Hermit pays money), else the best target
                     set over a few sampled hands, discounted (it can only be used in a round)
    pack             mean over a few sampled contents of the best pick (two picks for a Mega pack), minus cost
    voucher          measured where it shows in the score (hands, discards, hand size, ante), else a table
    reroll           running average of the best card price in fresh shops this game, weighted to this ante,
                     minus the cost
    leave, select    0 (the reference)
    skip blind       value of the tag (applied on a copy, or a table for tags that act later) minus what the
                     blind and its shop would have paid
    move joker       measured (copier rule), else the rule-based player's chips -> mult -> xmult order

Sampling. Every strength in one decision is measured on the same K hands and the same bootstrap draws
(common random numbers): hand h holds the cards with the lowest priority[h, card], where a card keeps its
priority across the copies of a decision (so a tarot's target is in the same sampled hands before and after),
and cards that only exist in a copy are numbered by their content, never by creation order. The priorities
are seeded from the deck's contents and the ante (potential._seed, a SHA-256: no process-dependent hashes).

No hidden information: prices only read what a player sees. Every option is applied to world.determinize
copies driven by the caller's random generator (the agent's own), sampled hands come from the deck's
contents in a canonical order, and the running reroll average only holds prices of shops the agent has seen
in this game (it is reset when a new game starts).

Rule switches (ShopConfig.rule_*), all off by default so each can be measured by itself:
    rule_skip     blind skips are priced by tag value (off: never skip, as today)
    rule_arcana   Arcana / Spectral packs and shop tarots / spectrals are priced by sampling targets
                  (off: never bought, as today; tarots met in a pack are still priced)
    rule_hold     Hermit / Temperance wait for the money they double, held consumables keep a value
    rule_scaling  scaling jokers are valued at their current state moved growth_rounds rounds forward
                  (growth_table: gain of state["val"] per round)
    rule_pace     behind pace (clear chance of this ante's boss < pace_threshold): money counts for less,
                  the interest reserve is dropped and rerolls are worth more
    rule_copier   Blueprint / Brainstorm: a joker is priced in its best slot and moves are priced by score
    rule_boss     the visible boss is modelled in this ante's clear chance: The Needle (one hand), The Water
                  (no discards), The Mouth (one hand type), The Eye (no repeated type), and the suit / face
                  debuffs of The Club, The Goad, The Head, The Window and The Plant
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..rewards.growth import discounted_rounds
from ..rewards.potential import Potential, _seed, best_play, fresh_hand_size, fresh_round
from ..rewards.strength import boss_targets, clear_chances, current_ante, strength_score
from ..sim import fastscore
from ..sim.cards import sort_hand
from ..sim.game import Consumable, Game, TARGETED
from ..sim.items import PACKS, PLANETS, SUIT_BOSS
from ..sim.jokers import Joker
from ..sim.scoring import Plan
from .features import N_PRICE       # candidate features: price, strength part, money part, "was priced"
from .world import Action, apply, determinize

PASSIVE = ("leave", "select", "pack_skip")
COPIERS = ("blueprint", "brainstorm")
MONEY_JOKERS = ("bull", "bootstraps")            # their score reads the money held

# dollars per round an economy joker brings in (own estimates from the rules; a callable reads the game)
ECON_INCOME = {
    "golden": 4.0, "rocket": 2.0, "delayed_grat": 1.0, "business": 1.0, "egg": 3.0, "mail": 2.0,
    "reserved_parking": 1.5, "rough_gem": 1.5, "faceless": 1.5, "matador": 0.5, "todo_list": 1.5, "trading": 1.0,
    "chaos": 3.0, "astronomer": 2.0, "hallucination": 1.0, "cartomancer": 2.0, "riff_raff": 2.0, "8_ball": 0.5,
    "vagabond": 1.0, "superposition": 0.3, "sixth_sense": 0.5, "seance": 0.2, "dna": 1.5, "certificate": 1.0,
    "burnt": 1.5, "space": 1.5, "hiker": 1.0, "perkeo": 3.0, "invisible": 2.0, "diet_cola": 2.0, "luchador": 1.0,
    "mr_bones": 2.0, "chicot": 3.0, "burglar": 2.0, "midas_mask": 1.5, "credit_card": 0.5,
    "to_the_moon": lambda g: float(min(max(g.money, 0) // 5, g.interest_cap())),
    "cloud_9": lambda g: float(sum(1 for c in g.full_deck if c.rank == 9 and not c.is_stone)),
    "gift": lambda g: float(len(g.jokers) + len(g.consumables)),
    "satellite": lambda g: float(len(g.planets_used)),
    "ticket": lambda g: 1.5 * sum(1 for c in g.full_deck if c.enh == "GOLD"),
}
# one-off value, in dollars, of vouchers whose effect the fresh-round score does not show
VOUCHER_DOLLARS = {
    "overstock_norm": 12.0, "overstock_plus": 12.0, "clearance_sale": 12.0, "liquidation": 18.0, "hone": 3.0,
    "glow_up": 4.0, "reroll_surplus": 6.0, "reroll_glut": 6.0, "crystal_ball": 4.0, "omen_globe": 3.0,
    "telescope": 8.0, "observatory": 8.0, "tarot_merchant": 4.0, "tarot_tycoon": 5.0, "planet_merchant": 5.0,
    "planet_tycoon": 6.0, "seed_money": 10.0, "money_tree": 12.0, "blank": 2.0, "antimatter": 25.0,
    "magic_trick": 2.0, "illusion": 2.0, "directors_cut": 2.0, "retcon": 3.0,
}
# tags that act in a later shop or round: value in dollars
TAG_DOLLARS = {"uncommon": 6.0, "rare": 9.0, "negative": 12.0, "polychrome": 8.0, "holo": 5.0, "foil": 4.0,
               "coupon": 8.0, "d_six": 4.0, "investment": 15.0, "voucher": 1.0, "juggle": 1.0, "double": 4.0}
RANDOM_TAGS = ("orbital", "top_up", "boss", "buffoon", "charm", "meteor", "standard", "ethereal")
# gain of state["val"] per round held (own estimates; Stage 2's logged games agree where they have data)
GROWTH = {"green_joker": 1.5, "ride_the_bus": 0.5, "runner": 5.0, "square": 3.0, "trousers": 0.7, "castle": 5.0,
          "wee": 6.0, "constellation": 0.03, "hologram": 0.05, "red_card": 1.0, "flash": 1.0, "vampire": 0.05,
          "lucky_cat": 0.1, "glass": 0.05, "madness": 0.3, "yorick": 0.2, "ceremonial": 2.0,
          "ice_cream": -15.0, "popcorn": -4.0, "ramen": -0.05}
RANDOM_USES = frozenset({"wheel_of_fortune", "aura", "familiar", "grim", "incantation", "immolate", "sigil",
                         "ouija", "ectoplasm", "hex", "ankh", "wraith", "judgement", "soul", "emperor",
                         "high_priestess"})


@dataclass
class ShopConfig:
    """Constants of the graded shop prior (module doc). Prices are in strength units: the strength score is
    in [0, 1], and `dollar` converts money."""
    # ---- build strength
    value: str = "strength"         # S: "strength" (Stage 2 score on shared sampled hands) or "phi" (Phi itself)
    samples: int = 32               # K sampled hands, shared by every option of a decision
    log_weight: float = 0.15        # + log_weight x ln(mean best-hand score): still moves when every clear
                                    #   chance is 0 or 1
    deck_weight: float = 0.0004     # + deck_weight x 52 x mean card quality (Game.card_value): one changed
                                    #   card is in few of the K hands, so its measured gain is mostly 0
    # ---- money: U(x) = dollar x a(ante) x (x + interest_rounds x min(x // 5, cap))
    dollar: float = 0.008           # strength units per dollar at ante 1
    m_zero_ante: int = 8            # a(ante) falls linearly from 1 at ante 1 to m_floor at this ante
    m_floor: float = 0.05
    interest_rounds: float = 2.0    # a $5 interest step held is worth this many extra dollars
    econ_rounds: float = 6.0        # an economy joker counts this many rounds of its income
    # ---- prices -> logits
    logit_scale: float = 100.0      # prior logit = (price - best price) x logit_scale, clipped at -30
    act_margin: float = 0.003       # an action must beat doing nothing (leave, select, skip the pack) by this
    use_bonus: float = 0.004        # using a consumable that costs nothing to use beats holding it
    # ---- sampling
    pack_samples: int = 2           # sampled contents per pack priced
    pack_targets: int = 4           # target sets tried per tarot / spectral inside a sampled pack
    use_samples: int = 2            # copies per option with a random outcome
    cons_samples: int = 2           # sampled hands for a tarot / spectral that needs cards (shop)
    inner_scale: float = 0.5        # inside a search or a rollout: this share of the sample counts (min 1)
    shop_cons_frac: float = 0.3     # a targeted consumable bought in the shop is worth this share of its
                                    #   sampled best use (it waits for a round, and for the agent to use it)
    created_dollars: float = 2.0    # value of a consumable an option creates (Emperor, High Priestess)
    # ---- reroll
    reroll_prior: float = 0.01      # expected best card price of a fresh shop before any was seen
    reroll_prior_n: float = 2.0     # weight of that prior, in shops
    reroll_min_money: int = 4       # after paying: with less than this nothing can be bought
    # ---- rule switches (module doc)
    rule_skip: bool = False
    rule_arcana: bool = False
    rule_hold: bool = False
    rule_scaling: bool = False
    rule_pace: bool = False
    rule_copier: bool = False
    rule_boss: bool = False
    # skip
    skip_shop_dollars: float = 6.0  # what the skipped blind's shop is worth
    skip_scaling: float = 0.004     # per owned joker that grows by being played
    skip_risk: float = 0.3          # x (1 - clear chance of the blind skipped): a skip cannot lose
    tag_dollars: dict = field(default_factory=lambda: dict(TAG_DOLLARS))
    # hold
    hold_discount: float = 0.8      # a payout that could be had later counts this much now
    # scaling
    growth_rounds: int = 3          # rounds a scaling joker is moved forward (one ante)
    growth_discount: float = 0.9
    growth_table: dict = field(default_factory=lambda: dict(GROWTH))
    # pace
    pace_threshold: float = 0.6     # behind pace: clear chance of this ante's boss below this
    pace_cost_scale: float = 0.5    # behind: money counts this much, and no interest reserve
    pace_reroll_scale: float = 2.0  # behind: a reroll is worth this much more
    # copier
    move_margin: float = 0.002      # a joker move must gain this much
    # tables
    voucher_dollars: dict = field(default_factory=lambda: dict(VOUCHER_DOLLARS))

    def __post_init__(self):
        if self.value not in ("strength", "phi"):
            raise ValueError(f"unknown shop value {self.value!r}")


def joker_category(j) -> int:
    """The rule-based player's order: chips (0), mult (1), xmult (2)."""
    from ..heuristic import CHIPS, XMULT
    if j.edition == "POLYCHROME" or j.key in XMULT:
        return 2
    return 0 if j.key in CHIPS else 1


def rule_move(g: Game) -> int:
    """Slot i whose joker the rule-based player would swap with slot i + 1 (heuristic._order_jokers); -1: none."""
    for i in range(len(g.jokers) - 1):
        a, b = g.jokers[i], g.jokers[i + 1]
        if a.key == "blueprint" or b.key == "blueprint":
            continue
        if joker_category(a) > joker_category(b):
            return i
    return -1


def _sig(c) -> tuple:
    return (c.rank, c.suit, c.enh, c.edition, c.seal, c.extra_chips)


class _Ctx:
    """One decision: the game it is about, and what every option's strength is measured with."""
    __slots__ = ("g", "seed", "index", "n_base", "prio", "w_dollar", "k_int", "behind", "p_boss", "w0", "inner",
                 "scores", "econ0")

    def __init__(self, g: Game):
        self.g = g
        base = sorted(g.full_deck, key=lambda c: (_sig(c), c.uid))
        self.index = {c.uid: i for i, c in enumerate(base)}
        self.n_base = len(base)
        self.seed = _seed(("shop", tuple(_sig(c) for c in base), g.ante))
        self.prio = None
        self.behind = False
        self.inner = False

    def indices(self, cards) -> list[int]:
        """Each card's number: its place in the decision's deck, or (cards a copy created) after them, in
        order of content."""
        new = sorted((c for c in cards if c.uid not in self.index), key=lambda c: (_sig(c), c.uid))
        extra = {c.uid: self.n_base + k for k, c in enumerate(new)}
        return [self.index.get(c.uid, extra.get(c.uid)) for c in cards]


class ShopPricer:
    def __init__(self, cfg: ShopConfig | None = None, potential: Potential | None = None, max_cache: int = 20_000):
        self.cfg = cfg or ShopConfig()
        self.potential = potential or Potential()
        self.cache: dict = {}
        self.max_cache = max_cache
        self.stats = Counter()
        self.last: dict = {}
        self.reset()

    # ------------------------------------------------------------------ per game
    def reset(self):
        """A new game: forget the shops seen (the reroll average) and the cached pack values."""
        self.fresh = {}                  # ante -> [sum of best prices, shops]
        self.seen = set()
        self.pack_cache = {}

    # ------------------------------------------------------------------ money
    def ante_factor(self, ante: int) -> float:
        c = self.cfg
        a = min(max(ante, 1), 8)
        return max(c.m_floor, (c.m_zero_ante - a) / max(1, c.m_zero_ante - 1))

    def money_value(self, g: Game, money: float, ante: int | None = None, behind: bool = False) -> float:
        """U(money) at this ante (module doc)."""
        c = self.cfg
        w = c.dollar * self.ante_factor(g.ante if ante is None else ante)
        k = c.interest_rounds * (1 + g.count("to_the_moon"))
        if behind and c.rule_pace:
            w, k = w * c.pace_cost_scale, 0.0
        x = max(money, 0)
        return w * (money + k * min(x // 5, g.interest_cap()))

    def m(self, g: Game, cost: float, behind: bool = False) -> float:
        """m(ante, money): strength units given up per dollar when `cost` is spent now."""
        if cost == 0:
            return 0.0
        return (self.money_value(g, g.money, behind=behind) - self.money_value(g, g.money - cost, behind=behind)) / cost

    def _u(self, ctx: _Ctx, g: Game) -> float:
        return self.money_value(ctx.g, g.money, ante=ctx.g.ante, behind=ctx.behind)

    # ------------------------------------------------------------------ economy
    def _econ(self, ctx: _Ctx, g: Game) -> float:
        c = self.cfg
        dollars = 0.0
        for j in g.jokers:
            inc = ECON_INCOME.get(j.key)
            if inc is not None and not j.debuffed:
                dollars += c.econ_rounds * (inc(g) if callable(inc) else inc)
        for v in g.vouchers:
            dollars += c.voucher_dollars.get(v, 0.0)
        return dollars * ctx.w_dollar

    # ------------------------------------------------------------------ strength on shared hands
    def _prio(self, ctx: _Ctx) -> np.ndarray:
        if ctx.prio is None:
            ctx.prio = np.random.default_rng(ctx.seed).random((self.cfg.samples, 128))
        return ctx.prio

    def _build_key(self, ctx: _Ctx, g: Game) -> tuple:
        idx = ctx.indices(g.full_deck)
        deck = tuple(sorted((i, _sig(c)) for i, c in zip(idx, g.full_deck)))
        jok = tuple((j.key, j.edition, j.debuffed, j.sell_value(),
                     tuple(sorted((k, repr(v)) for k, v in j.state.items() if not k.startswith("_"))))
                    for j in g.jokers)
        money = g.money if any(j.key in MONEY_JOKERS for j in g.jokers) else 0
        return (deck, jok, tuple(g.hand_levels), tuple(g.hand_played), tuple(sorted(g.vouchers)),
                tuple(sorted(c.key for c in g.consumables)), money, fresh_hand_size(g), g.round_hands(),
                g.round_discards(), g.tarots_used, g.blinds_skipped, g.joker_slots, g.starting_deck_size, g.deck_type)

    def _project(self, probe: Game):
        """rule_scaling: move the scaling jokers of `probe` (a copy) growth_rounds rounds forward."""
        c = self.cfg
        eff = discounted_rounds(c.growth_rounds, c.growth_discount)
        for j in probe.jokers:
            v = j.state.get("val")
            gain = c.growth_table.get(j.key)
            if gain is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
                new = v + gain * eff
                if gain < 0:
                    new = max(new, min(v, 0.0) if j.key != "ramen" else 1.0)
                j.state["val"] = new

    def _sample_scores(self, ctx: _Ctx, g: Game, debuff: str = "") -> tuple[np.ndarray, np.ndarray]:
        """(best-play score, its hand type) of each of the K shared hands, for the build of `g` at the first
        hand of a fresh round. debuff: a suit / face boss whose debuffs are applied to the cards first."""
        self.stats["scorings"] += 1
        probe = fresh_round(g)
        if self.cfg.rule_scaling:
            self._project(probe)
        if debuff:
            par, sm = probe.has("pareidolia"), probe.has("smeared")
            for c in probe.full_deck:
                c.debuffed = c.is_face(par) if debuff == "plant" else c.has_suit(SUIT_BOSS[debuff], sm)
        pool = sort_hand(probe.full_deck)
        n = min(fresh_hand_size(g), len(pool))
        k = self.cfg.samples
        if n == 0:
            return np.zeros(k), np.zeros(k, dtype=np.int64)
        idx = np.asarray(ctx.indices(pool), dtype=np.int64) % 128
        p = self._prio(ctx)[:, idx]
        hands = np.sort(np.argsort(p, axis=1, kind="stable")[:, :n], axis=1).tolist()
        plan = Plan(probe)
        fs = fastscore.pool_scorer(probe, plan, pool) if n <= 16 else None
        if fs is not None:
            best = fs.best_two_many(hands, probe.hands_left, probe.discards_left, len(pool) - n,
                                    fastscore.hand_types_mask(probe.round_hand_types), probe.mouth_hand)
            sc = [float(b[0][0]) if b else 0.0 for b in best]
            ty = [int(b[0][1]) if b else 0 for b in best]
        else:
            sc, ty = [], []
            for h in hands:
                hand = [pool[i] for i in h]
                ids = {id(c) for c in hand}
                probe.hand, probe.deck = hand, [c for c in pool if id(c) not in ids]
                s, sub = best_play(probe, plan, hand)
                sc.append(float(s))
                ty.append(int(probe.predict_many([sub], plan, hand)[0][1]) if sub else 0)
        return np.asarray(sc, dtype=float), np.asarray(ty, dtype=np.int64)

    def _boss_chance(self, ctx: _Ctx, g: Game, sc: np.ndarray, ty: np.ndarray, target: float, seed: int) -> float | None:
        """rule_boss: this ante's clear chance under the visible boss; None: no model for it."""
        boss = g.boss
        scfg = self.potential.cfg.strength
        hands, discards = g.round_hands(), g.round_discards()
        if boss == "needle":
            return clear_chances(sc, 1, discards, [target], scfg, seed)[0]
        if boss == "water":
            return clear_chances(sc, hands, 0, [target], scfg, seed)[0]
        if boss == "mouth":                               # every hand of the round is one hand type
            return max(clear_chances(np.where(ty == t, sc, 0.0), hands, discards, [target], scfg, seed)[0]
                       for t in set(ty.tolist()))
        if boss == "eye":                                 # no hand type twice
            return _eye_chance(sc, ty, hands, discards, target, scfg, seed)
        if boss in SUIT_BOSS or boss == "plant":
            sc2, _ = self._sample_scores(ctx, g, debuff=boss)
            return clear_chances(sc2, hands, discards, [target], scfg, seed)[0]
        return None

    def strength(self, ctx: _Ctx, g: Game) -> tuple[float, float, np.ndarray]:
        """(S, clear chance of this ante's boss, the K best-hand scores) of g's build."""
        c = self.cfg
        targets = boss_targets(g)
        key = (ctx.seed, self._build_key(ctx, g), tuple(targets), g.boss if c.rule_boss else "")
        self.stats["strength_calls"] += 1
        got = self.cache.get(key)
        if got is not None:
            self.stats["strength_hits"] += 1
            return got
        sc, ty = self._sample_scores(ctx, g)
        scfg = self.potential.cfg.strength
        bseed = _seed(("shop-bootstrap", ctx.seed))
        ch = clear_chances(sc, g.round_hands(), g.round_discards(), targets, scfg, bseed)
        if c.rule_boss:
            p = self._boss_chance(ctx, g, sc, ty, targets[0], bseed)
            if p is not None:
                ch[0] = p
        if c.value == "phi":
            s = float(self.potential(g))
        else:
            s = strength_score(current_ante(g), ch, scfg.score) + c.log_weight * math.log(max(float(sc.mean()), 1.0))
        s += c.deck_weight * 52.0 * _deck_quality(g)
        out = (s, float(ch[0]), sc)
        if len(self.cache) >= self.max_cache:
            self.cache.clear()
        self.cache[key] = out
        return out

    # ------------------------------------------------------------------ state value
    def _w(self, ctx: _Ctx, g: Game) -> tuple[float, float]:
        """(strength and economy part, money part) of W(g)."""
        return self.strength(ctx, g)[0] + self._econ(ctx, g), self._u(ctx, g)

    def _delta(self, ctx: _Ctx, g: Game) -> tuple[float, float]:
        a, b = self._w(ctx, g)
        return a - ctx.w0[0], b - ctx.w0[1]

    def context(self, g: Game, inner: bool = False) -> _Ctx:
        ctx = _Ctx(g)
        ctx.inner = inner
        ctx.w_dollar = self.cfg.dollar * self.ante_factor(g.ante)
        _, p, sc = self.strength(ctx, g)
        ctx.p_boss, ctx.scores = p, sc
        ctx.behind = bool(self.cfg.rule_pace and p < self.cfg.pace_threshold)
        ctx.w0 = self._w(ctx, g)
        return ctx

    def _n(self, ctx: _Ctx, n: int) -> int:
        return max(1, int(round(n * self.cfg.inner_scale))) if ctx.inner else n

    # ------------------------------------------------------------------ options
    def _applied(self, ctx: _Ctx, a: Action, rng: random.Random, samples: int = 1) -> list[Game]:
        """Copies of the decision's game with `a` carried out (hidden information redrawn from rng)."""
        out = []
        for _ in range(samples):
            g2 = determinize(ctx.g, rng)
            try:
                apply(g2, a)
            except (AssertionError, IndexError):
                continue
            out.append(g2)
        return out

    def _mean_delta(self, ctx: _Ctx, games: list[Game]) -> tuple[float, float]:
        if not games:
            return -1.0, 0.0
        d = [self._delta(ctx, g2) for g2 in games]
        return sum(x[0] for x in d) / len(d), sum(x[1] for x in d) / len(d)

    def _created(self, ctx: _Ctx, g2: Game) -> float:
        """Value of consumables an option left in the slots (beyond those the decision started with)."""
        new = Counter(c.key for c in g2.consumables) - Counter(c.key for c in ctx.g.consumables)
        return sum(new.values()) * self.cfg.created_dollars * ctx.w_dollar

    def _use_value(self, ctx: _Ctx, g: Game, i: int, rng: random.Random) -> float:
        """What using consumable i of `g` (a copy of the decision's game) right now adds, without its
        price; 0 when it cannot be used outside a round."""
        cons = g.consumables[i]
        if not g.consumable_usable(cons, []):
            return 0.0
        n = self._n(ctx, self.cfg.use_samples) if cons.name in RANDOM_USES else 1
        tot, k = 0.0, 0
        for _ in range(n):
            g2 = determinize(g, rng)
            try:
                apply(g2, Action("use", i))
            except (AssertionError, IndexError):
                continue
            a, b = self._w(ctx, g2)
            base = self._w(ctx, g)
            tot += (a - base[0]) + (b - base[1]) + self._created(ctx, g2)
            k += 1
        return tot / k if k else 0.0

    def _target_value(self, ctx: _Ctx, g: Game, cons: Consumable, rng: random.Random) -> float:
        """Best gain of a tarot / spectral that needs cards, over the valid target sets of a few sampled
        hands (the deck's contents in a canonical order, drawn with rng)."""
        from .actions import target_sets
        n = self._n(ctx, self.cfg.cons_samples)
        size = fresh_hand_size(g)
        base = self._w(ctx, g)[0]
        tot = 0.0
        for _ in range(n):
            g2 = determinize(g, rng)
            pool = sorted(g2.full_deck, key=lambda c: (_sig(c), c.uid))
            g2.rng.shuffle(pool)
            hand = sort_hand(pool[:size])
            if not g2.consumable_usable(cons, hand, from_slot=False):
                continue
            best = 0.0
            sets = target_sets(g2, cons, hand)[:self.cfg.pack_targets] if cons.name in TARGETED else [()]
            for s in sets:
                g3 = g2.clone()
                by_uid = {c.uid: c for c in g3.full_deck}
                cards = [by_uid[c.uid] for c in hand]
                try:
                    g3.apply_consumable(Consumable(cons.kind, cons.name), cards,
                                        targets=[cards[p] for p in sorted(s)] if s else None)
                except (AssertionError, IndexError, ValueError):
                    continue
                best = max(best, self._w(ctx, g3)[0] - base)
            tot += best
        return tot / n

    def hold_value(self, ctx: _Ctx, g: Game, i: int, rng: random.Random) -> float:
        """What consumable i, held in `g`, is worth beyond what holding it already shows in the score."""
        c = self.cfg
        cons = g.consumables[i]
        if cons.kind == "planet":
            return max(0.0, self._use_value(ctx, g, i, rng))
        if cons.name == "hermit":
            now = self._use_value(ctx, g, i, rng)
            later = c.hold_discount * 20 * ctx.w_dollar if c.rule_hold else 0.0
            return max(now, later, 0.0)
        if g.consumable_usable(cons, []):
            return max(0.0, self._use_value(ctx, g, i, rng))
        if c.rule_arcana and (cons.name in TARGETED or cons.kind != "planet"):
            return c.shop_cons_frac * self._target_value(ctx, g, cons, rng)
        return 0.0

    def _wait_penalty(self, ctx: _Ctx, g: Game, cons: Consumable, now: float) -> float:
        """rule_hold: what using this now gives up against using it when it pays most."""
        c = self.cfg
        if not c.rule_hold:
            return 0.0
        if cons.name == "hermit":
            later = c.hold_discount * 20 * ctx.w_dollar
        elif cons.name == "temperance":
            total = sum(j.sell_value() for j in g.jokers)
            room = g.joker_slots - len(g.jokers)
            later = c.hold_discount * min(50, total + 3 * max(room, 0)) * ctx.w_dollar
        else:
            return 0.0
        return max(0.0, later - now)

    def _best_slot(self, ctx: _Ctx, g2: Game) -> tuple[float, float]:
        """rule_copier: the delta with the newest joker of g2 in its best slot (g2 is left in that order)."""
        best = self._delta(ctx, g2)
        if not self.cfg.rule_copier or len(g2.jokers) < 2 or not any(j.key in COPIERS for j in g2.jokers):
            return best
        new = g2.jokers[-1]
        rest = g2.jokers[:-1]
        order = list(g2.jokers)
        for pos in range(len(rest)):
            g2.jokers = rest[:pos] + [new] + rest[pos:]
            d = self._delta(ctx, g2)
            if d[0] > best[0] + 1e-12:
                best, order = d, list(g2.jokers)
        g2.jokers = order
        return best

    def pack_value(self, ctx: _Ctx, g: Game, rng: random.Random, depth: int = 0) -> float:
        """`g` (a copy) is in an open pack: the best pick's gain, plus the best second pick of a Mega pack;
        0 for skipping."""
        from .actions import target_sets
        base = self._w(ctx, g)
        best = 0.0
        for i, x in enumerate(g.pack_cards):
            if not g.pack_pick_ok(i):
                continue
            sets = [()]
            if isinstance(x, Consumable) and x.name in TARGETED and g.pack_hand:
                sets = target_sets(g, x, g.pack_hand)[:self.cfg.pack_targets]
            for s in sets:
                g2 = g.clone()                      # g is already a redrawn copy: its generator is the agent's
                try:
                    apply(g2, Action("pick", i, cards=tuple(s)))
                except (AssertionError, IndexError):
                    continue
                if isinstance(x, Joker):
                    a, b = self._w(ctx, g2)
                    if self.cfg.rule_copier and any(j.key in COPIERS for j in g2.jokers):
                        a0, b0 = self._best_slot(ctx, g2)
                        a, b = a0 + ctx.w0[0], b0 + ctx.w0[1]
                else:
                    a, b = self._w(ctx, g2)
                v = (a - base[0]) + (b - base[1]) + self._created(ctx, g2)
                if g2.state == "PACK" and depth < 1:
                    v += self.pack_value(ctx, g2, rng, depth + 1)
                best = max(best, v)
        return best

    def _pack_price(self, ctx: _Ctx, w, idx: int, rng: random.Random) -> tuple[float, float]:
        g = ctx.g
        it = g.shop_packs[idx]
        money = self._u_after(ctx, g.money - it.cost)
        kind = it.pack[0]
        if kind in ("arcana", "spectral") and not self.cfg.rule_arcana:
            return -self.cfg.act_margin, money
        key = (ctx.seed, self._build_key(ctx, g), g.boss, it.key, ctx.inner)
        val = self.pack_cache.get(key)
        if val is None:
            tot, n = 0.0, self._n(ctx, self.cfg.pack_samples)
            for _ in range(n):
                g2 = determinize(g, rng)
                g2.money += it.cost                  # the cost is charged once, below
                try:
                    g2.buy_pack(idx)
                except (AssertionError, IndexError):
                    continue
                g2.money = g.money
                tot += self.pack_value(ctx, g2, rng)
            val = self.pack_cache[key] = tot / n
        return val, money

    def _u_after(self, ctx: _Ctx, money: float) -> float:
        g = ctx.g
        return self.money_value(g, money, behind=ctx.behind) - ctx.w0[1]

    def reroll_value(self, ctx: _Ctx) -> float:
        """Expected best card price of a fresh shop: this game's shops, weighted to this ante."""
        c = self.cfg
        tot = sum(s for s, _ in self.fresh.values())
        cnt = sum(n for _, n in self.fresh.values())
        overall = (c.reroll_prior * c.reroll_prior_n + tot) / (c.reroll_prior_n + cnt)
        s, n = self.fresh.get(ctx.g.ante, (0.0, 0))
        return (overall * c.reroll_prior_n + s) / (c.reroll_prior_n + n)

    def _skip_price(self, ctx: _Ctx, rng: random.Random) -> tuple[float, float]:
        c = self.cfg
        g = ctx.g
        if not c.rule_skip:
            return -1.0, 0.0
        tag = g.tags_offered[g.blind_idx]
        n = self._n(ctx, c.use_samples) if tag in RANDOM_TAGS else 1
        tot_a = tot_b = 0.0
        k = 0
        for _ in range(n):
            g2 = determinize(g, rng)
            try:
                g2.skip_blind()
            except (AssertionError, IndexError):
                continue
            a, b = self._delta(ctx, g2)
            if g2.state == "PACK":
                a += self.pack_value(ctx, g2, rng)
            tot_a, tot_b, k = tot_a + a, tot_b + b, k + 1
        if not k:
            return -1.0, 0.0
        a, b = tot_a / k, tot_b / k
        doubles = 1 + g.pending_tags.count("double")
        later = c.tag_dollars.get(tag, 0.0)
        if tag in ("uncommon", "rare") and len(g.jokers) >= g.joker_slots:
            later *= 0.5
        a += later * doubles * ctx.w_dollar
        # what the blind would have paid: its reward, about one unused hand, its interest, its shop
        reward = (0 if g.stake >= 1 else 3) if g.blind_idx == 0 else 4
        interest = (1 + g.count("to_the_moon")) * min(max(g.money, 0) // 5, g.interest_cap())
        cost = (reward + 1 + interest + c.skip_shop_dollars) * ctx.w_dollar
        cost += c.skip_scaling * sum(1 for j in g.jokers if c.growth_table.get(j.key, 0.0) > 0)
        scfg = self.potential.cfg.strength
        p = clear_chances(ctx.scores, g.round_hands(), g.round_discards(), [g.blind_target(g.blind_idx)], scfg,
                          _seed(("shop-bootstrap", ctx.seed)))[0]
        a += c.skip_risk * (1.0 - p)
        return a - cost - c.act_margin, b

    def _sell_price(self, ctx: _Ctx, idx: int, rng: random.Random) -> tuple[float, float]:
        g = ctx.g
        games = self._applied(ctx, Action("sell_joker", idx), rng)
        if not games:
            return -1.0, 0.0
        g2 = games[0]
        best = self._delta(ctx, g2)
        if g.state == "SHOP" and len(g.jokers) >= g.joker_slots:      # sell this one, then buy that one
            for k, it in enumerate(g2.shop):
                if it.kind != "joker" or not g2.can_buy(it):
                    continue
                g3 = g2.clone()
                try:
                    g3.buy_card(k)
                except (AssertionError, IndexError):
                    continue
                d = self._best_slot(ctx, g3)
                if d[0] + d[1] - self.cfg.act_margin > best[0] + best[1]:
                    best = (d[0] - self.cfg.act_margin, d[1])
        return best[0] - self.cfg.act_margin, best[1]

    def _price(self, ctx: _Ctx, w, cand, rng: random.Random) -> tuple[float, float]:
        """(strength / economy part, money part) of one option's price."""
        c = self.cfg
        g = ctx.g
        a = cand.action
        k = a.kind
        if k in ("leave", "select"):
            return 0.0, 0.0
        if k == "pack_skip":
            return self._mean_delta(ctx, self._applied(ctx, a, rng))
        if k == "buy":
            it = g.shop[a.idx]
            games = self._applied(ctx, a, rng)
            if not games:
                return -1.0, 0.0
            g2 = games[0]
            if it.kind == "joker":
                d = self._best_slot(ctx, g2)
                return d[0] - c.act_margin, d[1]
            d = self._delta(ctx, g2)
            if it.kind in ("planet", "tarot", "spectral"):
                if it.kind != "planet" and not c.rule_arcana:
                    return -c.act_margin, d[1]
                return d[0] + self.hold_value(ctx, g2, len(g2.consumables) - 1, rng) - c.act_margin, d[1]
            return d[0] - c.act_margin, d[1]
        if k == "buy_pack":
            v, money = self._pack_price(ctx, w, a.idx, rng)
            return v - c.act_margin, money
        if k == "voucher":
            d = self._mean_delta(ctx, self._applied(ctx, a, rng))
            return d[0] - c.act_margin, d[1]
        if k == "sell_joker":
            return self._sell_price(ctx, a.idx, rng)
        if k == "sell_cons":
            d = self._mean_delta(ctx, self._applied(ctx, a, rng))
            return d[0] - self.hold_value(ctx, g, a.idx, rng) - c.act_margin, d[1]
        if k == "use":
            cons = g.consumables[a.idx]
            n = self._n(ctx, c.use_samples) if cons.name in RANDOM_USES else 1
            games = self._applied(ctx, a, rng, n)
            d = self._mean_delta(ctx, games)
            extra = sum(self._created(ctx, g2) for g2 in games) / max(1, len(games))
            pen = self._wait_penalty(ctx, g, cons, d[0] + d[1])
            return d[0] + extra + c.use_bonus - pen - c.act_margin, d[1]
        if k == "move_joker":
            if c.rule_copier:
                d = self._mean_delta(ctx, self._applied(ctx, a, rng))
                return d[0] - c.move_margin - c.act_margin, 0.0
            i = rule_move(g)
            return (c.act_margin if (i >= 0 and a.idx == i and a.to == i + 1) else -1.0), 0.0
        if k == "pick":
            x = g.pack_cards[a.idx]
            n = self._n(ctx, c.use_samples) if (isinstance(x, Consumable) and x.name in RANDOM_USES) else 1
            games = self._applied(ctx, a, rng, n)
            if isinstance(x, Joker) and games:
                d = self._best_slot(ctx, games[0])
            else:
                d = self._mean_delta(ctx, games)
            extra = sum(self._created(ctx, g2) for g2 in games) / max(1, len(games))
            return d[0] + extra - 0.5 * c.act_margin, d[1]
        if k == "skip":
            return self._skip_price(ctx, rng)
        if k == "reroll_boss":
            d = self._mean_delta(ctx, self._applied(ctx, a, rng, self._n(ctx, c.use_samples)))
            return d[0] - c.act_margin, d[1]
        return -1.0, 0.0                                    # reroll is priced in prices(): it needs the others

    # ------------------------------------------------------------------ public
    def prices(self, w, choice, rng: random.Random, root: bool = True) -> np.ndarray:
        """[n, 2]: (strength / economy part, money part) of each candidate's price; their sum is the price."""
        g = w.g
        c = self.cfg
        ctx = self.context(g, inner=not root)
        out = np.zeros((len(choice.cands), 2))
        for i, cand in enumerate(choice.cands):
            if cand.kind != "reroll":
                out[i] = self._price(ctx, w, cand, rng)
        if g.state == "SHOP":
            cards = [out[i].sum() for i, cand in enumerate(choice.cands) if cand.kind in ("buy", "sell_joker")]
            best = max([0.0] + cards)
            fresh = (g.shops_seen, w.rerolls)
            if root and fresh not in self.seen:             # a shop the agent has not priced yet
                self.seen.add(fresh)
                s, n = self.fresh.get(g.ante, (0.0, 0))
                self.fresh[g.ante] = (s + best, n + 1)
            for i, cand in enumerate(choice.cands):
                if cand.kind != "reroll":
                    continue
                cost = 0 if g.free_rerolls > 0 else g.reroll_cost
                val = self.reroll_value(ctx)
                if ctx.behind:
                    val *= c.pace_reroll_scale
                if g.money - cost < c.reroll_min_money:
                    val = 0.0
                out[i] = (val - c.act_margin, self._u_after(ctx, g.money - cost))
        self.last = {"ctx": ctx, "prices": out}
        return out

    def prior(self, w, choice, rng: random.Random, root: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(prior logits, candidate price features [n, N_PRICE]) for a decision outside a round."""
        parts = self.prices(w, choice, rng, root)
        price = parts.sum(axis=1)
        logits = np.clip((price - price.max()) * self.cfg.logit_scale, -30.0, 0.0)
        feats = np.zeros((len(price), N_PRICE), np.float32)
        feats[:, 0] = np.clip(price * 10.0, -3.0, 3.0)
        feats[:, 1] = np.clip(parts[:, 0] * 10.0, -3.0, 3.0)
        feats[:, 2] = np.clip(parts[:, 1] * 10.0, -3.0, 3.0)
        feats[:, 3] = 1.0
        return logits, feats


def _deck_quality(g: Game) -> float:
    if not g.full_deck:
        return 0.0
    return sum(Game.card_value(c) for c in g.full_deck) / len(g.full_deck)


def _eye_chance(sc: np.ndarray, ty: np.ndarray, hands: int, discards: int, target: float, scfg, seed: int) -> float:
    """The Eye: a bootstrapped round counts its best hands of distinct hand types only."""
    if len(sc) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    extra = max(0, int(round(scfg.discard_weight * discards))) if scfg.discards == "extra_draws" else 0
    hands = max(1, int(hands))
    ix = rng.integers(0, len(sc), size=(scfg.bootstrap, hands + extra))
    s, t = sc[ix] * scfg.score_scale, ty[ix]
    order = np.argsort(-s, axis=1, kind="stable")
    s, t = np.take_along_axis(s, order, 1), np.take_along_axis(t, order, 1)
    used = np.zeros(len(s), dtype=np.int64)
    cnt = np.zeros(len(s), dtype=np.int64)
    tot = np.zeros(len(s))
    for col in range(s.shape[1]):
        bit = np.left_shift(1, t[:, col])
        ok = ((used & bit) == 0) & (cnt < hands)
        tot += np.where(ok, s[:, col], 0.0)
        used |= np.where(ok, bit, 0)
        cnt += ok
    return float((tot >= target).mean())
