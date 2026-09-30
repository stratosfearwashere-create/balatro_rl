"""A: candidate enumeration and the exact scorer.

`enumerate_candidates` lists every legal action at a decision point:
  in a round   plays (1-5 cards), discards (1-5 cards), consumable uses with every valid target set,
               selling jokers / consumables, moving a joker to another slot
  shop         buy card / pack / voucher, reroll, leave, sell, use (consumables that need no targets), move
  pack         pick a card (with every valid target set for targeted tarots / spectrals), skip
  blind select select, skip, reroll the boss

For plays it computes, with the simulator's own scoring code, the exact chips, mult and score, whether the
play clears the blind, the score when every chance effect fails (so "clears with certainty" is exact),
and what the play changes besides the score: each joker's runtime state (Green Joker's mult, Ride the
Bus's counter, Ice Cream's chips ...), money, hand levels, the deck (Hiker, Midas Mask, Vampire, glass
that may break) and consumables it creates. Discards get the same through the jokers' discard hooks and
purple seals; consumable uses and pack picks are carried out on a resampled copy of the game (world.py)
and compared with the original.

Card order: effects trigger left to right, so order can matter (a Mult card before a Glass card scores
more than after it, Hanging Chad retriggers the first card, Photograph doubles the first face card).
Each play is scored in hand order and in a few other orders; when any of them differs, every order of
its cards is scored and the best one becomes the play's order. Otherwise plays are canonical: one
per set of cards, and sets of identical cards (same rank, suit, enhancement, edition, seal) are merged.

The full enumeration can be thousands of actions (up to 6884 plays and as many discards with 16 cards,
~100 target sets per tarot). The network and search see a pruned set (`Config`): plays and discards are
grouped by what they change besides the score, and the best few of each group are kept, so a lower
scoring play that keeps Ride the Bus going or a discard that doesn't cost Green Joker its mult always
survives the pruning. `Choice.complete` says whether every legal play and discard was examined, which
the auto-play guard (agent.py) requires.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations, permutations

from ..env import _kept_feats_many
from ..sim import fastscore
from ..sim.game import Game, Consumable, TARGETED, MAX_SHOP, MAX_PACK, MAX_JOKERS
from ..sim.hands import N_HANDS
from ..sim.items import HAND_TO_PLANET
from ..sim.jokers import Joker
from ..sim.scoring import Plan, score_hand
from .world import (Action, World, MAX_MOVES_PER_PHASE, MAX_REROLLS_PER_SHOP, apply, card_sig, determinize,
                    deck_sig)

# jokers whose scoring side effects only happen on a real play (random, or creating things)
REAL_ONLY_JOKERS = frozenset({"8_ball", "dna", "sixth_sense", "superposition", "seance", "vagabond", "space",
                              "hiker"})
# consumables with random outcomes: their analysis averages a few resampled copies
RANDOM_CONSUMABLES = frozenset({"wheel_of_fortune", "aura", "familiar", "grim", "incantation", "immolate",
                                "sigil", "ouija", "ectoplasm", "hex", "ankh", "wraith", "judgement", "soul",
                                "emperor", "high_priestess"})


@dataclass
class Config:
    max_plays: int = 48              # plays kept for the network / search
    plays_per_group: int = 8         # best plays kept per side-effect group
    plays_per_hand_type: int = 2
    max_discards: int = 24
    discards_per_group: int = 8
    max_analyzed: int = 320          # plays / discards whose side effects are computed
    max_targets: int = 16            # target sets kept per consumable
    use_samples: int = 2             # resampled copies per random consumable use


@dataclass
class Cand:
    action: Action
    score: float = 0.0          # play: expected score (chance effects at their expected value)
    score_min: float = 0.0      # play: score when every chance effect fails
    chips: float = 0.0
    mult: float = 0.0
    hand: int = -1              # play: hand type
    clears: bool = False        # expected score reaches the chips still needed
    certain: bool = False       # the pessimistic score does too
    jdiff: tuple = ()           # ((joker slot, state key, old, new), ...) changes to jokers this action makes now
    effects: tuple = ()         # other lasting changes: (("money", x), ("levels", n), ("deck", n), ("create", n), ...)
    analyzed: bool = False      # jdiff / effects were computed
    rank_key: float = 0.0       # discard: cheap usefulness estimate used for pruning
    p_clear: float = -1.0       # solver (B): P(clear the blind) after this action; -1 = not evaluated
    e_chips: float = -1.0       # solver: expected chips at the end of the round / target
    best_after: float = -1.0    # use / pick: best immediate play afterwards / chips needed
    best_before: float = -1.0   # use / pick: the same before

    @property
    def kind(self) -> str:
        return self.action.kind

    def signature(self) -> tuple:
        return (self.jdiff, self.effects)


@dataclass
class Choice:
    """The candidates at one decision."""
    cands: list
    phase: str
    need: float = 0.0            # chips still needed (in a round)
    complete: bool = True        # every legal play and discard was examined (not only the kept ones)
    n_legal: int = 0             # size of the full enumeration
    after_jdiff: tuple = ()      # joker changes every play makes after scoring (Ice Cream, Seltzer)
    all_plays: list = field(default_factory=list)       # every analyzed play / discard, kept or not
    all_discards: list = field(default_factory=list)    # (the auto-play guard compares against all of them)


# ------------------------------------------------------------------ pessimistic / optimistic randomness
class _Fixed(random.Random):
    """A random generator that always returns the same draw: 1-eps makes every chance effect fail,
    0 makes every one succeed. randint returns the low (pessimist) or high (optimist) end."""

    def __init__(self, u: float):
        super().__init__(0)
        self.u = u

    def random(self):
        return self.u

    def randint(self, a, b):
        return a if self.u > 0.5 else b


PESSIMIST = _Fixed(1.0 - 1e-12)
OPTIMIST = _Fixed(0.0)


def _copies(cards):
    return [c.copy(new_uid=False) for c in cards]


def _state_diff(slot_of, j, new: dict) -> list:
    out = []
    old = j.state
    for k in set(old) | set(new):
        if k.startswith("_"):
            continue
        a, b = old.get(k), new.get(k)
        if a != b:
            out.append((slot_of[j.uid], k, a, b))
    return out


# ------------------------------------------------------------------ plays
def play_subsets(g: Game) -> list[tuple]:
    n = len(g.hand)
    forced = g.forced_pos()
    subs = [s for k in range(1, min(5, n) + 1) for s in combinations(range(n), k) if forced < 0 or forced in s]
    return _dedupe(g, subs)


def discard_subsets(g: Game) -> list[tuple]:
    if g.discards_left <= 0:
        return []
    return play_subsets(g)


def _sig(c, i: int) -> tuple:
    """A card's identity for merging identical cards; face-down cards are all distinct (by position)."""
    if c.hidden:
        return (-1, i, "", "", "", 0, False)
    return card_sig(c)


def _dedupe(g: Game, subs: list[tuple]) -> list[tuple]:
    """Merge card sets that differ only by swapping identical cards (face-down cards are all distinct)."""
    sigs = [_sig(c, i) for i, c in enumerate(g.hand)]
    if len(set(sigs)) == len(sigs):
        return subs
    seen, out = set(), []
    for s in subs:
        key = tuple(sorted(sigs[i] for i in s))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


class _HandScorer:
    """Scores plays of the current hand: all subsets in one compiled call (score_all), and any ordered
    subsets (card orders) on the same compiled scorer. Same values as Game.predict_many; falls back to it
    when the compiled scorer isn't built."""

    def __init__(self, g: Game, plan: Plan, view):
        self.g, self.plan, self.view = g, plan, view
        n = len(view)
        self.fs = fastscore.pool_scorer(g, plan, view) if n <= 16 else None
        self._all = None
        if self.fs is not None:
            self._all = self.fs.score_all(list(range(n)), g.hands_left, g.discards_left, len(g.deck),
                                          fastscore.hand_types_mask(g.round_hand_types), g.mouth_hand)
            self._index = {s: k for k, s in enumerate(fastscore.subset_patterns(n))}

    def subsets(self, subs: list[tuple]) -> list:
        """Scores of hand-order subsets (as produced by play_subsets)."""
        if self._all is None:
            return self.g.predict_many(subs, self.plan, self.view) if subs else []
        return [self._all[self._index[s]] for s in subs]

    def ordered(self, subs: list[tuple]) -> list:
        """Scores of subsets in any card order (the hand stays loaded after score_all)."""
        if self.fs is None:
            return self.g.predict_many(subs, self.plan, self.view)
        return self.fs.predict_many(subs)


def best_orders(g: Game, plan: Plan, view, subs: list[tuple], preds: list, scorer: _HandScorer | None = None
                ) -> tuple[list, list]:
    """Replace each play by its best-scoring card order, where order matters (see the module doc)."""
    predict = scorer.ordered if scorer is not None else (lambda s: g.predict_many(s, plan, view))
    multi = [i for i, s in enumerate(subs) if len(s) >= 2]
    alts = []
    for i in multi:
        s = subs[i]
        alts += [s[::-1], s[1:] + s[:1]]
    if not alts:
        return subs, preds
    res = predict(alts)
    subs, preds = list(subs), list(preds)
    for k, i in enumerate(multi):
        base = preds[i][0]
        if res[2 * k][0] == base and res[2 * k + 1][0] == base:
            continue
        perms = list(permutations(subs[i]))
        pr = predict(perms)
        b = max(range(len(perms)), key=lambda t: (pr[t][0], -t))
        if pr[b][0] > base:
            subs[i], preds[i] = perms[b], pr[b]
    return subs, preds


def analyze_play(g: Game, plan: Plan, view, pos: tuple, need: float, slot_of: dict, real_only: bool) -> Cand:
    """Exact score and every lasting change of one play (no side effects on g: scored on card copies)."""
    # the expected-value pass never changes cards (only a real play or a random generator does: Hiker, glass,
    # commit), so it can read the hand's own cards; the random passes below work on copies
    played = [view[i] for i in pos]
    held = [c for i, c in enumerate(view) if i not in pos]
    sc, ctx = score_hand(g, played, held, rng=None, commit=False, plan=plan)
    violated = g.violates_boss([view[i] for i in pos])
    if violated:
        sc = 0.0
    cand = Cand(Action("play", cards=tuple(pos)), score=float(sc), chips=ctx.chips, mult=ctx.mult, hand=ctx.hand,
                analyzed=True)
    jd = []
    for j in plan.jokers:
        if j.uid in ctx.jstate:
            jd += _state_diff(slot_of, j, ctx.jstate[j.uid])
    eff = Counter()
    if ctx.money:
        eff["money"] += ctx.money
    if ctx.level_up:
        eff["levels"] += ctx.level_up
    if ctx.enh_override:
        eff["deck"] += len(ctx.enh_override)
    glass = sum(1 for i in ctx.scoring if played[i].enh == "GLASS" and not played[i].debuffed)
    if glass:
        eff["glass"] += glass
    boss = plan.boss
    if boss == "tooth":
        eff["money"] -= len(pos)
    if boss == "arm" and g.hand_levels[ctx.hand] > 1:
        eff["levels"] -= 1
    if boss == "ox" and max(g.hand_played) > 0 and g.hand_played[ctx.hand] == max(g.hand_played):
        eff["ox"] += 1
    if real_only:                                        # effects that only a real (random) play produces
        p2, h2 = _copies([view[i] for i in pos]), _copies([c for i, c in enumerate(view) if i not in pos])
        _, octx = score_hand(g, p2, h2, rng=OPTIMIST, commit=False, plan=plan)
        for e in octx.events:
            eff[e[0]] += 1
        if octx.level_up:
            eff["levels_chance"] += octx.level_up
        hiked = sum(1 for a, b in zip(p2, played) if a.extra_chips != b.extra_chips)
        if hiked:
            eff["deck"] += hiked
    cand.clears = sc >= need
    if cand.clears:
        # the round ends: cards still held pay out (Gold cards, Blue seals)
        mime = sum(1 for j in plan.jokers if j.key == "mime")
        for c in held:
            if c.enh == "GOLD" and not c.debuffed:
                eff["money"] += 3 * (1 + mime + (c.seal == "RED"))
            if c.seal == "BLUE" and not c.debuffed:
                eff["create"] += 1
        p3, h3 = _copies([view[i] for i in pos]), _copies([c for i, c in enumerate(view) if i not in pos])
        smin, _ = score_hand(g, p3, h3, rng=PESSIMIST, commit=False, plan=plan)
        cand.score_min = 0.0 if violated else float(smin)
        cand.certain = cand.score_min >= need
    cand.jdiff = tuple(sorted(jd, key=repr))
    cand.effects = tuple(sorted((k, round(v, 6)) for k, v in eff.items() if v))
    return cand


def after_play_jdiff(g: Game, rng: random.Random) -> tuple:
    """Joker changes every play makes after scoring (hooks that don't depend on the cards)."""
    if not any(j.d.after and not j.debuffed for j in g.jokers):
        return ()
    probe = determinize(g, rng)
    slot_of = {j.uid: i for i, j in enumerate(g.jokers)}
    before = {j.uid: dict(j.state) for j in probe.jokers}
    for j in list(probe.jokers):
        if j.d.after and not j.debuffed:
            j.d.after(probe, j)
    out = []
    alive = {j.uid for j in probe.jokers}
    for j in g.jokers:
        if j.uid not in alive:
            out.append((slot_of[j.uid], "_removed", 0, 1))
    for j in probe.jokers:
        if j.uid in slot_of:
            old = before[j.uid]
            for k in set(old) | set(j.state):
                if not k.startswith("_") and old.get(k) != j.state.get(k):
                    out.append((slot_of[j.uid], k, old.get(k), j.state.get(k)))
    return tuple(sorted(out, key=repr))


# ------------------------------------------------------------------ discards
def analyze_discards(g: Game, subs: list[tuple], rng: random.Random) -> list[Cand]:
    """Discard hooks (Green Joker, Castle, Hit the Road, Faceless, Mail-In Rebate, Trading Card, Ramen,
    Burnt, Yorick) and purple seals, run on a resampled copy and restored after each discard."""
    out = [Cand(Action("discard", cards=s), analyzed=True) for s in subs]
    view = g.hand_view()
    for c in out:
        purple = sum(1 for i in c.action.cards if view[i].seal == "PURPLE" and not view[i].debuffed)
        if purple:
            c.effects = (("create", purple),)
    hooked = [j for j in g.jokers if j.d.discard and not j.debuffed]
    if not hooked:
        return out
    probe = determinize(g, rng)
    slot_of = {j.uid: i for i, j in enumerate(probe.jokers)}
    for c in out:
        saved = ([dict(j.state) for j in probe.jokers], list(probe.jokers), probe.money, list(probe.hand_levels),
                 dict(probe.flags), list(probe.consumables), probe.joker_slots)
        cards = [probe.hand[i] for i in c.action.cards]
        for j in list(probe.jokers):
            if j.d.discard and not j.debuffed and j in probe.jokers:
                j.d.discard(probe, j, cards)
        jd = []
        alive = {j.uid for j in probe.jokers}
        for j, st in zip(saved[1], saved[0]):
            if j.uid not in alive:
                jd.append((slot_of[j.uid], "_removed", 0, 1))
                continue
            for k in set(st) | set(j.state):
                if not k.startswith("_") and st.get(k) != j.state.get(k):
                    jd.append((slot_of[j.uid], k, st.get(k), j.state.get(k)))
        eff = Counter(dict(c.effects))
        if probe.money != saved[2]:
            eff["money"] += probe.money - saved[2]
        if sum(probe.hand_levels) != sum(saved[3]):
            eff["levels"] += sum(probe.hand_levels) - sum(saved[3])
        if "trading_destroy" in probe.flags:
            eff["deck"] += 1
        c.jdiff = tuple(sorted(jd, key=repr))
        c.effects = tuple(sorted((k, round(v, 6)) for k, v in eff.items() if v))
        for j, st in zip(saved[1], saved[0]):
            j.state = st
        (probe.jokers, probe.money, probe.hand_levels, probe.flags, probe.consumables,
         probe.joker_slots) = saved[1], saved[2], saved[3], saved[4], saved[5], saved[6]
    return out


def _discard_rank(g: Game, subs: list[tuple]) -> list[float]:
    """Cheap usefulness of each discard (what the kept cards could still make), as in env.candidates."""
    kfs = _kept_feats_many(g.hand, subs)
    return [3 * kf[0] + 2 * kf[1] + 2 * kf[2] + kf[3] - kf[4] + 0.1 * len(s) for s, kf in zip(subs, kfs)]


def _complement_discards(g: Game, plays: list[Cand], k: int = 8) -> list[tuple]:
    """For the best plays: discard up to 5 of the cards outside the play, least valuable first."""
    out = []
    forced = g.forced_pos()
    for p in sorted(plays, key=lambda c: -c.score)[:k]:
        rest = [i for i in range(len(g.hand)) if i not in p.action.cards and i != forced]
        rest.sort(key=lambda i: g.card_value(g.hand[i]) if not g.hand[i].hidden else 0.0)
        if rest:
            out.append(tuple(sorted(rest[:5])))
    return out


# ------------------------------------------------------------------ consumables and packs
def target_sets(g: Game, cons: Consumable, cards: list) -> list[tuple]:
    """Every valid target set for a consumable over `cards`, identical cards merged, the fixed rule's
    choice (Game.auto_targets) first, then by card value (lowest first for the destroying Hanged Man)."""
    lo, hi = g.target_range(cons)
    n = len(cards)
    if n == 0 or hi == 0:
        return []
    sigs = [_sig(c, i) for i, c in enumerate(cards)]
    subs, seen = [], set()
    for k in range(lo, min(hi, n) + 1):
        for s in combinations(range(n), k):
            key = tuple(sigs[i] for i in s) if cons.name == "death" else tuple(sorted(sigs[i] for i in s))
            if key not in seen:
                seen.add(key)
                subs.append(s)
    auto = {cards.index(c) for c in g.auto_targets(cons.name, cards)}
    sign = 1.0 if cons.name == "hanged_man" else -1.0
    val = [g.card_value(c) if not c.hidden else 0.0 for c in cards]
    subs.sort(key=lambda s: (set(s) != auto, -len(auto & set(s)), sign * sum(val[i] for i in s)))
    return subs


def _best_play_score(g: Game, cards: list) -> float:
    """Best expected score playable from `cards` right now (for use / pick analysis)."""
    if not cards:
        return 0.0
    plan = Plan(g)
    view = [c if not c.hidden else _hidden_placeholder(c) for c in cards]
    n = len(view)
    idx = range(n)
    if n > 10:                                          # the 10 most valuable cards are plenty here
        idx = sorted(sorted(range(n), key=lambda i: -g.card_value(view[i]))[:10])
    subs = [s for k in range(1, 6) for s in combinations(idx, k)]
    preds = g.predict_many(subs, plan, view) if subs else []
    return max((p[0] for p in preds), default=0.0)


def _hidden_placeholder(c):
    from ..sim.cards import Card
    return Card(2, 0, enh="HIDDEN", uid=c.uid)


def state_diff(g0: Game, g1: Game) -> tuple[tuple, tuple]:
    """(jdiff, effects) between two games, for consumable uses and pack picks."""
    slot_of = {j.uid: i for i, j in enumerate(g0.jokers)}
    now = {j.uid: j for j in g1.jokers}
    jd = []
    for j in g0.jokers:
        k = now.get(j.uid)
        if k is None:
            jd.append((slot_of[j.uid], "_removed", 0, 1))
            continue
        if k.edition != j.edition:
            jd.append((slot_of[j.uid], "_edition", j.edition, k.edition))
        for key in set(j.state) | set(k.state):
            if not key.startswith("_") and j.state.get(key) != k.state.get(key):
                jd.append((slot_of[j.uid], key, j.state.get(key), k.state.get(key)))
    eff = Counter()
    added = sum(1 for j in g1.jokers if j.uid not in slot_of)
    if added:
        eff["jokers"] += added
    if g1.money != g0.money:
        eff["money"] += g1.money - g0.money
    if g1.hand_levels != g0.hand_levels:
        eff["levels"] += sum(g1.hand_levels) - sum(g0.hand_levels)
    d0, d1 = Counter(deck_sig(g0)), Counter(deck_sig(g1))
    changed = sum(((d0 - d1) + (d1 - d0)).values())
    if changed:
        eff["deck"] += changed
    if g1.hand_size != g0.hand_size:
        eff["hand_size"] += g1.hand_size - g0.hand_size
    if g1.joker_slots != g0.joker_slots:
        eff["joker_slots"] += g1.joker_slots - g0.joker_slots
    cons0 = Counter(c.key for c in g0.consumables)
    cons1 = Counter(c.key for c in g1.consumables)
    created = sum((cons1 - cons0).values())
    if created:
        eff["create"] += created
    return tuple(sorted(jd, key=repr)), tuple(sorted((k, round(v, 6)) for k, v in eff.items() if v))


def analyze_transition(g: Game, a: Action, rng: random.Random, cards_attr: str, need: float,
                       samples: int = 1) -> Cand:
    """Carry out a use / pick on resampled copies; the best immediate play before and after (from the hand,
    or from the pack's hand for a pack), and what it changed."""
    c = Cand(a, analyzed=True)
    befores = _best_play_score(g, getattr(g, cards_attr)) if need > 0 else 0.0
    afters, sig = [], None
    for _ in range(max(1, samples)):
        w = determinize(g, rng)
        try:
            apply(w, a)
        except AssertionError:
            continue
        if need > 0:
            pool = w.hand if cards_attr == "hand" else w.pack_hand
            if cards_attr == "pack_hand" and a.kind == "pick" and w.state != "PACK":
                # the pack closed; score the (changed) cards that were in its hand
                uids = {x.uid for x in g.pack_hand}
                pool = [x for x in w.full_deck if x.uid in uids]
            afters.append(_best_play_score(w, pool))
        if sig is None:
            sig = state_diff(g, w)
    if sig is not None:
        c.jdiff, c.effects = sig
    if need > 0:
        c.best_before = min(5.0, befores / need)
        c.best_after = min(5.0, (sum(afters) / len(afters)) / need) if afters else c.best_before
    return c


# ------------------------------------------------------------------ enumeration
def _prune_plays(plays: list[Cand], cfg: Config) -> list[Cand]:
    order = sorted(plays, key=lambda c: (-c.score, len(c.action.cards)))
    keep, groups, types = [], Counter(), Counter()
    for c in order:
        grp = c.signature()
        if groups[grp] < cfg.plays_per_group or types[c.hand] < cfg.plays_per_hand_type:
            keep.append(c)
            groups[grp] += 1
            types[c.hand] += 1
    if len(keep) > cfg.max_plays:              # keep each group's best first, then fill by score
        firsts, rest, seen = [], [], set()
        for c in keep:
            (rest if c.signature() in seen else firsts).append(c)
            seen.add(c.signature())
        keep = (firsts + rest)[:cfg.max_plays]
    return keep


def _prune_discards(discs: list[Cand], cfg: Config, must: set) -> list[Cand]:
    order = sorted(discs, key=lambda c: (c.action.cards not in must, -c.rank_key))
    keep, groups = [], Counter()
    for c in order:
        grp = c.signature()
        if c.action.cards in must or groups[grp] < cfg.discards_per_group:
            keep.append(c)
            groups[grp] += 1
    firsts, rest, seen = [], [], set()
    for c in keep:
        (rest if c.signature() in seen else firsts).append(c)
        seen.add(c.signature())
    return (firsts + rest)[:cfg.max_discards]


def _shared(w: World, cands: list, in_round: bool, rng: random.Random, need: float, cfg: Config):
    """Selling, using (non-round: only consumables that need no cards) and moving jokers."""
    g = w.g
    for i, j in enumerate(g.jokers[:MAX_JOKERS]):
        if not j.eternal:
            c = Cand(Action("sell_joker", i), analyzed=True)
            c.jdiff = ((i, "_removed", 0, 1),)
            c.effects = (("money", j.sell_value()),)
            cands.append(c)
    for i, cons in enumerate(g.consumables):
        c = Cand(Action("sell_cons", i), analyzed=True)
        c.effects = (("money", cons.sell_value()),)
        cands.append(c)
        cards = g.hand if in_round else []
        if not g.consumable_usable(cons, cards):
            continue
        samples = cfg.use_samples if cons.name in RANDOM_CONSUMABLES else 1
        if cons.name in TARGETED and cards:
            for s in target_sets(g, cons, cards)[:cfg.max_targets]:
                cands.append(analyze_transition(g, Action("use", i, cards=s), rng, "hand", need, samples))
        else:
            cands.append(analyze_transition(g, Action("use", i), rng, "hand", need, samples))
    n = min(len(g.jokers), MAX_JOKERS)
    if w.moves < MAX_MOVES_PER_PHASE and n >= 2:
        seen = set()
        for i in range(n):
            for t in (0, n - 1, i - 1, i + 1):
                if 0 <= t < n and t != i and (i, t) not in seen:
                    seen.add((i, t))
                    cands.append(Cand(Action("move_joker", i, to=t), analyzed=True))


def enumerate_candidates(w: World, rng: random.Random, cfg: Config = Config()) -> Choice:
    g = w.g
    assert g.targeting is None, "the agent resolves targets inside its actions"
    st = g.state
    cands: list[Cand] = []
    if st == "SELECTING_HAND":
        return _round_candidates(w, rng, cfg)
    if st == "BLIND_SELECT":
        cands.append(Cand(Action("select"), analyzed=True))
        if g.blind_idx < 2:
            cands.append(Cand(Action("skip"), analyzed=True))
        if g.can_reroll_boss():
            cands.append(Cand(Action("reroll_boss"), analyzed=True, effects=(("money", -10),)))
    elif st == "SHOP":
        for i, it in enumerate(g.shop[:MAX_SHOP]):
            if g.can_buy(it):
                cands.append(Cand(Action("buy", i), analyzed=True, effects=(("money", -it.cost),)))
        for i, it in enumerate(g.shop_packs[:2]):
            if g.can_afford(it.cost):
                cands.append(Cand(Action("buy_pack", i), analyzed=True, effects=(("money", -it.cost),)))
        if g.shop_voucher is not None and g.can_afford(g.shop_voucher.cost):
            cands.append(Cand(Action("voucher"), analyzed=True, effects=(("money", -g.shop_voucher.cost),)))
        if w.rerolls < MAX_REROLLS_PER_SHOP and (g.free_rerolls > 0 or g.can_afford(g.reroll_cost)):
            cands.append(Cand(Action("reroll"), analyzed=True,
                              effects=(("money", 0 if g.free_rerolls else -g.reroll_cost),)))
        cands.append(Cand(Action("leave"), analyzed=True))
        _shared(w, cands, False, rng, 0.0, cfg)
    elif st == "PACK":
        need = float(g.blind_target(min(g.blind_idx, 2)))     # the next blind: the pack is a mini-round
        for i, x in enumerate(g.pack_cards[:MAX_PACK]):
            if not g.pack_pick_ok(i):
                continue
            if isinstance(x, Consumable) and x.name in TARGETED and g.pack_hand:
                samples = cfg.use_samples if x.name in RANDOM_CONSUMABLES else 1
                for s in target_sets(g, x, g.pack_hand)[:cfg.max_targets]:
                    cands.append(analyze_transition(g, Action("pick", i, cards=s), rng, "pack_hand", need, samples))
            elif isinstance(x, Consumable):
                samples = cfg.use_samples if x.name in RANDOM_CONSUMABLES else 1
                cands.append(analyze_transition(g, Action("pick", i), rng, "pack_hand",
                                                need if g.pack_hand else 0.0, samples))
            else:
                c = Cand(Action("pick", i), analyzed=True)
                if isinstance(x, Joker):
                    c.effects = (("jokers", 1),)
                else:
                    c.effects = (("deck", 1),)
                cands.append(c)
        cands.append(Cand(Action("pack_skip"), analyzed=True))
    return Choice(cands, st, n_legal=len(cands))


def _round_candidates(w: World, rng: random.Random, cfg: Config) -> Choice:
    g = w.g
    need = max(g.target - g.chips, 1)
    plan = Plan(g)
    view = g.hand_view()
    slot_of = {j.uid: i for i, j in enumerate(g.jokers)}
    subs = play_subsets(g)
    hs = _HandScorer(g, plan, view)
    preds = hs.subsets(subs)
    subs, preds = best_orders(g, plan, view, subs, preds, hs)
    order = sorted(range(len(subs)), key=lambda i: -preds[i][0])
    real_only = any(j.key in REAL_ONLY_JOKERS for j in plan.jokers)
    after = after_play_jdiff(g, rng)
    plays = []
    for rank, i in enumerate(order):
        if rank < cfg.max_analyzed:
            c = analyze_play(g, plan, view, subs[i], need, slot_of, real_only)
            c.jdiff = tuple(sorted(c.jdiff + after, key=repr))
        else:                                   # beyond the analysis budget: score only
            c = Cand(Action("play", cards=subs[i]), score=float(preds[i][0]), hand=preds[i][1],
                     clears=preds[i][0] >= need)
        plays.append(c)
    complete = len(order) <= cfg.max_analyzed
    kept_plays = _prune_plays([c for c in plays if c.analyzed], cfg)
    kept_discs = []
    n_disc = 0
    if g.discards_left > 0:
        dsubs = discard_subsets(g)
        n_disc = len(dsubs)
        ranks = _discard_rank(g, dsubs)
        comp = set(_complement_discards(g, kept_plays))
        top = sorted(range(len(dsubs)), key=lambda i: (dsubs[i] not in comp, -ranks[i]))[:cfg.max_analyzed]
        complete = complete and n_disc <= cfg.max_analyzed
        dc = analyze_discards(g, [dsubs[i] for i in top], rng)
        for c, i in zip(dc, top):
            c.rank_key = ranks[i]
        kept_discs = _prune_discards(dc, cfg, comp)
        kept_ids = {id(c) for c in kept_discs}
        extra = [c for c in dc if id(c) not in kept_ids]
    else:
        extra = []
    cands = kept_plays + kept_discs
    others = []
    _shared(w, others, True, rng, float(need), cfg)
    cands += others
    return Choice(cands, "SELECTING_HAND", need=float(need), complete=complete,
                  n_legal=len(subs) + n_disc + len(others), after_jdiff=after,
                  all_plays=[c for c in plays if c.analyzed], all_discards=kept_discs + extra)
