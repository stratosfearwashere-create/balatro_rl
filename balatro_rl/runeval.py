"""A hand-written static evaluation of a Balatro run state, in the style of Stockfish 1.0 (see RUNEVAL.md).

eval(states) scores a batch of run states as read from the engine's observation (the global vector g, the joker ids and
features jid / jf, the consumable ids cid) and returns, per state:
    score          the blended estimate (a probability of winning the run)
    early / mid / late   the three phase potentials: P(reach ante 4), P(reach ante 7), P(win)
    terms          the nine hand-written terms (TERMS), each computed once and weighted per phase
Everything is numpy over the whole batch: no search, no engine call, no learning at evaluation time. The weights come from
balatro_rl/data/runeval_weights.json (hand-set priors, replaced by tools/runeval_fit.py's fitted values).

The nine terms (higher is better for every one of them):
    power        log10 of hands x the best reliable play's expected score over the next blind's target
    scaling      log10 of how much the growing jokers' counters will have multiplied the score by the ante-8 boss, plus the xMult sources
    consistency  the chance of landing the main hand within a round's hands and discards (deck composition, hand size)
    deck         suit / rank concentration behind the main hand, thinning, enhanced cards
    economy      money, interest, income jokers, sell value (dollars per round, log scale)
    synergy      joker-joker and joker-deck fits (copies next to multipliers, hand / suit / face / enhancement jokers that match)
    risk         minus a capped quadratic penalty as the margin over the next blind shrinks (the king-safety table)
    flex         open slots, consumables, rerolls affordable, tag on offer, extra hands / discards / hand size
    boss         minus the known (or expected) boss's bite on this build

    python -m balatro_rl.runeval checkpoints/ppo_gold_v30_best.pt --games 16     # scores along a few games, with timing
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import numpy as np

from .marginal import MARG_G, POT_G
from .env import STATES, PACK_KINDS, VOUCHER_KEYS, global_size
from .sim.cards import ENHANCEMENTS, SEALS, EDITIONS
from .sim.hands import HAND_NAMES, HAND_BASE, N_HANDS, HC, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH, FIVE_KIND, FLUSH_HOUSE, FLUSH_FIVE
from .sim.items import BOSS_KEYS, BOSSES, FINISHERS, TAGS, VOCAB, BLIND_BASE, item_id

TERMS = ("power", "scaling", "consistency", "deck", "economy", "synergy", "risk", "flex", "boss")
PHASES = ("early", "mid", "late")
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "runeval_weights.json")


# ------------------------------------------------------------------ the observation's columns (env.encode_global, in order)
class Cols:
    ANTE = 0
    BLIND = 1                                   # 3 one-hot: small, big, boss
    STATE = 4                                   # one-hot over STATES
    LOG_TARGET = STATE + len(STATES)            # log10(target + 1) / 6 of the blind faced / about to be faced
    CHIPS_RATIO = LOG_TARGET + 1
    LOG_REMAIN = CHIPS_RATIO + 1
    HANDS_LEFT = LOG_REMAIN + 1                 # / 5
    DISC_LEFT = HANDS_LEFT + 1
    ROUND_HANDS = DISC_LEFT + 1                 # hands per round / 5
    ROUND_DISC = ROUND_HANDS + 1
    HAND_SIZE = ROUND_DISC + 1                  # effective hand size / 8
    IN_HAND = HAND_SIZE + 1
    MONEY = IN_HAND + 1                         # min(money, 100) / 50
    LOG_MONEY = MONEY + 1                       # log10(money + 1) / 3
    NEG = LOG_MONEY + 1
    DEBT = NEG + 1
    REROLL = DEBT + 1                           # reroll cost / 10
    FREE_REROLL = REROLL + 1
    REROLLS = FREE_REROLL + 1
    SWAPS = REROLLS + 1
    BOSS = SWAPS + 1                            # one-hot over BOSS_KEYS
    BOSS_ACTIVE = BOSS + len(BOSS_KEYS)
    BOSS_DISABLED = BOSS_ACTIVE + 1
    CAN_REROLL_BOSS = BOSS_DISABLED + 1
    TAG = CAN_REROLL_BOSS + 1                   # one-hot over TAGS: the tag a skip would take
    SKIPPED = TAG + len(TAGS)
    LEVELS = SKIPPED + 1                        # hand levels / 10
    PLAYED = LEVELS + N_HANDS                   # min(times played, 60) / 30
    PLAYED_ROUND = PLAYED + N_HANDS
    RANKS = PLAYED_ROUND + N_HANDS              # 13 rank counts / 4 (stone cards excluded)
    SUITS = RANKS + 13                          # 4 suit counts / 13
    DECK = SUITS + 4                            # draw pile / 52
    FULL = DECK + 1                             # full deck / 52
    ENH = FULL + 1                              # ENHANCEMENTS[1:] counts / 10
    SEAL = ENH + len(ENHANCEMENTS) - 1          # SEALS[1:] counts / 10
    JSLOTS = SEAL + len(SEALS) - 1              # / 7
    NJOKERS = JSLOTS + 1
    CSLOTS = NJOKERS + 1                        # / 3
    CUSED = CSLOTS + 1
    VOUCH = CUSED + 1                           # one per VOUCHER_KEYS
    TARGETS = VOUCH + len(VOUCHER_KEYS)         # log10(target + 1) / 6 of the ante's three blinds
    PACKS = TARGETS + 3
    PICKS = PACKS + len(PACK_KINDS)
    PLASMA = PICKS + 1
    TAROTS_USED = PLASMA + 1                    # / 20
    PLANETS_USED = TAROTS_USED + 1              # / 12
    STAKE = PLANETS_USED + 1                    # 9 one-hot
    END = STAKE + 9


assert Cols.END + MARG_G + POT_G == global_size(), (Cols.END, global_size())
I_BLIND_SELECT, I_SHOP, I_PACK = STATES.index("BLIND_SELECT"), STATES.index("SHOP"), STATES.index("PACK")
ENH_NAMES = ENHANCEMENTS[1:]                    # BONUS MULT WILD GLASS STEEL STONE GOLD LUCKY
E = {n: i for i, n in enumerate(ENH_NAMES)}
V = {k: i for i, k in enumerate(VOUCHER_KEYS)}
JF_FOIL, JF_HOLO, JF_POLY, JF_ETERNAL, JF_PERISH, JF_RENTAL, JF_DEBUFF, JF_SELL, JF_COUNTER = 1 + EDITIONS.index("FOIL"), 1 + EDITIONS.index("HOLO"), 1 + EDITIONS.index("POLYCHROME"), 6, 7, 8, 9, 10, 11

# cards scored by each hand, and which hands contain which (the game's "contains" rule: a joker that wants a Pair fires on a Full House)
N_CARDS = np.array([1, 2, 4, 3, 5, 5, 5, 4, 5, 5, 5, 5], dtype=np.float64)
_CONTAINS = {HC: (), PAIR: (PAIR,), TWO_PAIR: (PAIR, TWO_PAIR), TRIPS: (PAIR, TRIPS), STRAIGHT: (STRAIGHT,), FLUSH: (FLUSH,),
             FULL_HOUSE: (PAIR, TWO_PAIR, TRIPS, FULL_HOUSE), QUADS: (PAIR, TWO_PAIR, TRIPS, QUADS), STRAIGHT_FLUSH: (STRAIGHT, FLUSH, STRAIGHT_FLUSH),
             FIVE_KIND: (PAIR, TWO_PAIR, TRIPS, QUADS, FIVE_KIND), FLUSH_HOUSE: (PAIR, TWO_PAIR, TRIPS, FULL_HOUSE, FLUSH, FLUSH_HOUSE),
             FLUSH_FIVE: (PAIR, TWO_PAIR, TRIPS, QUADS, FIVE_KIND, FLUSH, FLUSH_FIVE)}
CONTAINS = np.zeros((N_HANDS, N_HANDS), dtype=np.float64)          # [hand h, requirement r] = 1 if h contains r
for _h, _rs in _CONTAINS.items():
    for _r in _rs:
        CONTAINS[_h, _r] = 1.0
IS_FLUSH = np.array([h in (FLUSH, STRAIGHT_FLUSH, FLUSH_HOUSE, FLUSH_FIVE) for h in range(N_HANDS)])
IS_SET = np.array([h in (PAIR, TWO_PAIR, TRIPS, FULL_HOUSE, QUADS, FIVE_KIND, FLUSH_HOUSE, FLUSH_FIVE) for h in range(N_HANDS)])
RANK_CHIPS = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11], dtype=np.float64)   # ranks 2 .. A
FACE_MASK = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], dtype=np.float64)
EVEN_MASK = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0], dtype=np.float64)
ODD_MASK = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1], dtype=np.float64)
ACE_MASK = np.array([0] * 12 + [1], dtype=np.float64)
HAND_CHIPS = np.array([HAND_BASE[h][0] for h in range(N_HANDS)], dtype=np.float64)
HAND_MULT = np.array([HAND_BASE[h][1] for h in range(N_HANDS)], dtype=np.float64)
HAND_DCHIPS = np.array([HAND_BASE[h][2] for h in range(N_HANDS)], dtype=np.float64)
HAND_DMULT = np.array([HAND_BASE[h][3] for h in range(N_HANDS)], dtype=np.float64)
# what a skip's tag is worth, in dollars (the Gold guide's view; investment / economy pay cash, the rest pay in kind)
TAG_VALUE = {"economy": 10, "investment": 25, "uncommon": 12, "rare": 20, "foil": 8, "holo": 10, "polychrome": 14, "negative": 20, "buffoon": 8, "meteor": 6,
             "charm": 8, "coupon": 10, "d_six": 6, "top_up": 8, "speed": 6, "handy": 5, "garbage": 4, "orbital": 6, "double": 10, "voucher": 8, "standard": 3,
             "ethereal": 10, "boss": 4, "juggle": 3}
TAG_VALUES = np.array([TAG_VALUE.get(t, 4) for t in TAGS], dtype=np.float64)
# bosses: the finishers, and the pool an unknown boss is drawn from
FINISHER_IDX = np.array([BOSS_KEYS.index(k) for k in FINISHERS], dtype=np.int64)
REGULAR_IDX = np.array([BOSS_KEYS.index(k) for k in BOSSES], dtype=np.int64)
BK = {k: i for i, k in enumerate(BOSS_KEYS)}
SUIT_BOSS = {"goad": 0, "head": 1, "club": 2, "window": 3}      # the engine's suit order S H C D


@lru_cache(maxsize=None)
def _spec():
    """The joker spec (joker_spec.SPEC, the engine's own numbers) as a raw [VOCAB, features] table plus a name -> column map. The numbers are
    unscaled (chips in chips, mult in mult, xmult = the multiplier minus one), unlike joker_spec.vectorise()'s network input."""
    from . import joker_spec as J
    names = (["sp_" + n for n in ("chips", "mult", "xmult", "chance", "unit", "retriggers", "max_cards", "exact_cards", "scales", "start_value", "grow_amount",
                                  "income_round", "last_hand", "first_hand", "first_discard", "discards_zero", "copies_joker", "interest", "d_hands", "d_discards",
                                  "d_hand_size", "d_joker_slots", "threshold", "needs_enh", "target_state")]
             + ["sp_hand_" + h.replace(" ", "_").lower() for h in HAND_NAMES] + ["sp_hand_none"]
             + [f"sp_suit_{c}" for c in "SHCD"] + ["sp_suit_none"]
             + [f"sp_rankclass_{r}" for r in J.RANK_CLASSES] + [f"sp_rank_{r}" for r in range(2, 15)]
             + [f"sp_trig_{t}" for t in J.TRIGGERS] + [f"sp_counter_{u}" for u in J.COUNTER_UNITS] + [f"sp_scales_{x}" for x in J.SCALES_WITH]
             + [f"sp_rule_{r}" for r in J.RULES])
    S = {n: i for i, n in enumerate(names)}
    out = np.zeros((len(VOCAB), len(names)), dtype=np.float64)
    for i, key in enumerate(VOCAB):
        if not (key.startswith("j_") and key[2:] in J.SPEC):
            continue
        f = J.full(key[2:])
        for n in ("chips", "mult", "xmult", "chance", "unit", "retriggers", "max_cards", "exact_cards", "scales", "start_value", "grow_amount", "income_round",
                  "last_hand", "first_hand", "first_discard", "discards_zero", "copies_joker", "interest", "d_hands", "d_discards", "d_hand_size", "d_joker_slots",
                  "threshold", "needs_enh", "target_state"):
            out[i, S["sp_" + n]] = float(f[n])
        out[i, S["sp_hand_" + HAND_NAMES[f["hand"]].replace(" ", "_").lower()] if f["hand"] >= 0 else S["sp_hand_none"]] = 1.0
        out[i, S[f"sp_suit_{'SHCD'[f['suit']]}"] if f["suit"] >= 0 else S["sp_suit_none"]] = 1.0
        out[i, S[f"sp_rankclass_{f['rank_class']}"]] = 1.0
        for r in f["rank_mask"]:
            out[i, S[f"sp_rank_{int(r)}"]] = 1.0
        for t in f["trigs"]:
            out[i, S[f"sp_trig_{t}"]] = 1.0
        out[i, S[f"sp_counter_{f['counter_unit']}"]] = 1.0
        out[i, S[f"sp_scales_{f['scales_with']}"]] = 1.0
        out[i, S[f"sp_rule_{f['rule']}"]] = 1.0
    return out, S


def _col(name):
    return _spec()[1][name]


@lru_cache(maxsize=None)
def _keysets():
    """Joker roles from the Gold guide, as boolean masks over VOCAB."""
    from .az import guide as G
    from .potential import GROW

    def mask(keys):
        m = np.zeros(len(VOCAB), dtype=bool)
        for k in keys:
            i = item_id(f"j_{k}")
            if i != item_id("j_unknown"):
                m[i] = True
        return m
    grow_rate = np.zeros(len(VOCAB))
    grow_cap = np.zeros(len(VOCAB))
    grow_unit = np.zeros(len(VOCAB), dtype=np.int64)        # 0 mult/chips (additive), 1 xmult (the counter is the multiplier)
    for k, (start, rate, lo, hi) in GROW.items():
        i = item_id(f"j_{k}")
        if i != item_id("j_unknown") and rate > 0:
            grow_rate[i], grow_cap[i], grow_unit[i] = rate, hi, int(start == 1)
    return dict(xmult=mask(G.XMULT), scaler=mask(G.SCALERS | G.SLOW_SCALERS), econ=mask(G.ECON), copy=mask({"blueprint", "brainstorm"}),
                retrig=mask({"hack", "sock_and_buskin", "dusk", "hanging_chad", "selzer", "mime"}), grow_rate=grow_rate, grow_cap=grow_cap, grow_unit=grow_unit,
                unknown=item_id("j_unknown"), blueprint=item_id("j_blueprint"), brainstorm=item_id("j_brainstorm"))


LOGFACT = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, 601, dtype=np.float64)))])     # log n! for n <= 600


def _lchoose(n, k):
    """log C(n, k) for integer arrays (0 <= k <= n <= 600)."""
    n = np.clip(np.rint(n), 0, 600).astype(np.int64)
    k = np.clip(np.rint(k), 0, 600).astype(np.int64)
    k = np.minimum(k, n)
    return LOGFACT[n] - LOGFACT[k] - LOGFACT[n - k]


def _hg_tail(n_good, n_total, draw, k):
    """P(X >= k) for X ~ Hypergeometric(population n_total, successes n_good, draws `draw`); integer arrays broadcast; k a scalar."""
    n_good = np.rint(np.asarray(n_good, dtype=np.float64))
    n_total = np.rint(np.asarray(n_total, dtype=np.float64))
    draw = np.rint(np.asarray(draw, dtype=np.float64))
    out = np.zeros(np.broadcast(n_good, n_total, draw).shape)
    for j in range(k, 14):
        ok = (j <= n_good) & (draw - j <= n_total - n_good) & (j <= draw)
        lp = _lchoose(n_good, j) + _lchoose(n_total - n_good, draw - j) - _lchoose(n_total, draw)
        out += np.where(ok, np.exp(np.where(ok, lp, 0.0)), 0.0)
    return np.clip(out, 0.0, 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ------------------------------------------------------------------ the state read
class State:
    """Scalars read off the observation batch, in game units."""

    def __init__(self, g, jid, jf, cid=None):
        g = np.asarray(g, dtype=np.float64)
        self.n = len(g)
        self.g = g
        self.jid = np.asarray(jid).astype(np.int64)
        self.jf = np.asarray(jf, dtype=np.float64)
        self.cid = None if cid is None else np.asarray(cid).astype(np.int64)
        self.ante = np.clip(np.rint(g[:, Cols.ANTE] * 8), 1, 8).astype(np.int64)
        self.blind = np.argmax(g[:, Cols.BLIND:Cols.BLIND + 3], axis=1)
        self.state = np.argmax(g[:, Cols.STATE:Cols.STATE + len(STATES)], axis=1)
        self.hands = np.maximum(np.rint(g[:, Cols.ROUND_HANDS] * 5), 1)
        self.discards = np.maximum(np.rint(g[:, Cols.ROUND_DISC] * 5), 0)
        self.hand_size = np.clip(np.rint(g[:, Cols.HAND_SIZE] * 8), 1, 16)
        self.money = np.round(np.where(g[:, Cols.MONEY] < 1.99, g[:, Cols.MONEY] * 50.0, 10.0 ** (3.0 * g[:, Cols.LOG_MONEY]) - 1.0), 2)
        self.reroll_cost = np.where(g[:, Cols.FREE_REROLL] > 0, 0.0, g[:, Cols.REROLL] * 10.0)
        self.levels = np.maximum(np.rint(g[:, Cols.LEVELS:Cols.LEVELS + N_HANDS] * 10), 1)
        self.played = g[:, Cols.PLAYED:Cols.PLAYED + N_HANDS] * 30.0
        self.ranks = np.maximum(np.rint(g[:, Cols.RANKS:Cols.RANKS + 13] * 4), 0)
        self.suits = np.maximum(np.rint(g[:, Cols.SUITS:Cols.SUITS + 4] * 13), 0)
        self.full = np.maximum(np.rint(g[:, Cols.FULL] * 52), 1)
        self.enh = np.maximum(np.rint(g[:, Cols.ENH:Cols.ENH + len(ENH_NAMES)] * 10), 0)
        self.seals = np.maximum(np.rint(g[:, Cols.SEAL:Cols.SEAL + len(SEALS) - 1] * 10), 0)
        self.jslots = np.rint(g[:, Cols.JSLOTS] * 7)
        self.njokers = np.rint(g[:, Cols.NJOKERS] * 7)
        self.cslots = np.rint(g[:, Cols.CSLOTS] * 3)
        self.cused = np.rint(g[:, Cols.CUSED] * 3)
        self.vouchers = g[:, Cols.VOUCH:Cols.VOUCH + len(VOUCHER_KEYS)] > 0.5
        self.targets = 10.0 ** (6.0 * g[:, Cols.TARGETS:Cols.TARGETS + 3]) - 1.0
        self.skipped = np.rint(g[:, Cols.SKIPPED] * 5)
        self.tag = g[:, Cols.TAG:Cols.TAG + len(TAGS)]
        self.boss = g[:, Cols.BOSS:Cols.BOSS + len(BOSS_KEYS)]
        self.boss_known = self.boss.max(1) > 0.5
        self.boss_idx = np.argmax(self.boss, axis=1)
        self.tarots_used = g[:, Cols.TAROTS_USED] * 20.0
        self.planets_used = g[:, Cols.PLANETS_USED] * 12.0
        self.stake = np.argmax(g[:, Cols.STAKE:Cols.STAKE + 9], axis=1)
        # the blind faced next: at a blind select the one about to be chosen; in a shop / pack the one after the blind just beaten
        nxt = np.where(self.state == I_BLIND_SELECT, self.blind, np.minimum(self.blind + 1, 2))
        self.next_is_boss = nxt == 2
        self.next_after_boss = (self.state != I_BLIND_SELECT) & (self.blind == 2)   # shop after a boss: the next ante's small blind, boss unknown
        self.next_target = np.take_along_axis(self.targets, nxt[:, None], 1)[:, 0]
        scaling = np.where(self.stake >= 5, 3, np.where(self.stake >= 2, 2, 1))
        base_next = np.array([BLIND_BASE[int(s)][min(int(a), 7)] for s, a in zip(scaling, np.minimum(self.ante + 1, 8))], dtype=np.float64)
        self.next_target = np.where(self.next_after_boss, base_next, self.next_target)
        self.ante8_target = np.array([BLIND_BASE[int(s)][7] for s in scaling], dtype=np.float64) * 2.0    # the final boss, x2 (Violet Vessel x6 aside)
        # rounds left before the ante-8 boss is played (its own round not counted), and until the next blind
        blinds_done = (self.ante - 1) * 3 + np.where(self.state == I_BLIND_SELECT, self.blind, self.blind + 1)
        self.rounds_left = np.maximum(23 - blinds_done, 0)
        # the deck's make-up as shares
        self.deck_n = np.maximum(self.ranks.sum(1), 1.0)                       # cards with a rank (stones apart)
        self.rank_share = self.ranks / self.deck_n[:, None]
        self.suit_share = self.suits / np.maximum(self.suits.sum(1), 1.0)[:, None]
        self.main_suit = np.argmax(self.suits, axis=1)
        # jokers: live (present, not debuffed), their counters
        self.live = (self.jid > 0) & (self.jf[:, :, 0] > 0.5) & (self.jf[:, :, JF_DEBUFF] < 0.5)
        self.counter = np.expm1(4.0 * self.jf[:, :, JF_COUNTER])


# ------------------------------------------------------------------ the terms
def hand_reach(s: State, rules, tries=None):
    """[n, 12]: the chance of landing each hand type within a round, from the deck's make-up, the hand size and the hands and discards.
    Each hand is hunted the way a player hunts it: the deal, then (hands - 1 + discards) redraws of the cards not kept, keeping the best
    partial (the most-held rank for sets, the best suit for flushes, the fullest 5-rank window for straights); hypergeometric deals, a
    gain-per-redraw chain for the chase. Combination hands are products of their parts with a discount."""
    n = s.n
    D = s.deck_n
    H = s.hand_size
    tries = np.maximum(s.hands - 1 + s.discards, 0) if tries is None else np.maximum(np.asarray(tries, dtype=np.float64), 0)
    T = int(min(tries.max(), 12)) if n else 0
    c = s.ranks
    c_eff = (c * c).sum(1) / D                                              # copies of the rank of a random card (4 in a fresh deck)
    out = np.zeros((n, N_HANDS))
    out[:, HC] = 1.0

    def chase(p_deal_ge, gain_of, kmax):
        """p_deal_ge [n, kmax + 1]: P(M0 >= k) after the deal; gain_of(m) -> P(+1) per redraw from m; -> P(M_final >= k) [n, kmax + 1]."""
        dist = np.zeros((n, kmax + 1))
        for m in range(kmax + 1):
            dist[:, m] = p_deal_ge[:, m] - (p_deal_ge[:, m + 1] if m + 1 <= kmax else 0.0)
        dist = np.clip(dist, 0.0, 1.0)
        for t in range(T):
            active = (tries > t)[:, None]
            nxt = dist.copy()
            for m in range(kmax):
                gm = np.clip(gain_of(m), 0.0, 1.0)
                move = dist[:, m] * gm
                nxt[:, m] -= move
                nxt[:, m + 1] += move
            dist = np.where(active, nxt, dist)
        ge = np.cumsum(dist[:, ::-1], axis=1)[:, ::-1]
        return np.clip(ge, 0.0, 1.0)

    # sets: the most-held rank
    p_ge = np.ones((n, 6))
    for k in range(2, 6):
        prod = np.ones(n)
        for r in range(13):
            prod *= 1.0 - _hg_tail(c[:, r], D, H, k)
        p_ge[:, k] = 1.0 - prod
    set_ge = chase(p_ge, lambda m: 1.0 - (1.0 - np.maximum(c_eff - m, 0.0) / D) ** np.maximum(H - m, 1.0), 5)
    out[:, PAIR] = set_ge[:, 2]
    out[:, TRIPS] = set_ge[:, 3]
    out[:, QUADS] = set_ge[:, 4]
    out[:, FIVE_KIND] = set_ge[:, 5]
    p2 = (c * (c - 1)).sum(1) / np.maximum(D * (D - 1), 1)                  # P(two given cards share a rank)
    r2 = np.maximum(H - 2, 0)
    q2 = 1.0 - np.exp(-r2 * (r2 - 1) / 2 * p2)                                # a second pair among the other cards, per redraw
    out[:, TWO_PAIR] = out[:, PAIR] * (1.0 - (1.0 - q2) ** (1 + tries)) * 0.9
    r3 = np.maximum(H - 3, 0)
    q3 = 1.0 - np.exp(-r3 * (r3 - 1) / 2 * p2)
    out[:, FULL_HOUSE] = out[:, TRIPS] * (1.0 - (1.0 - q3) ** (1 + tries / 2.0)) * 0.8
    # flushes: the deal's best suit is kept and the rest redrawn; each redraw adds Binomial(H - k, share of that suit left) cards of it.
    # Smeared Joker pairs the suits up; Four Fingers needs 4
    need = np.where(rules["four_fingers"], 4, 5)
    suits = s.suits.copy()
    if rules["smeared"].any():
        sm = rules["smeared"][:, None]
        paired = np.stack([suits[:, 0] + suits[:, 2], suits[:, 1] + suits[:, 3], suits[:, 0] + suits[:, 2], suits[:, 1] + suits[:, 3]], 1)
        suits = np.where(sm, paired, suits)
    p_ge_f = np.ones((n, 6))
    for k in range(1, 6):
        prod = np.ones(n)
        for i in range(4):
            prod *= 1.0 - _hg_tail(suits[:, i], D, H, k)
        p_ge_f[:, k] = 1.0 - prod
    n_s = suits.max(1)                                                        # the suit chased: the deck's richest
    dist = np.clip(p_ge_f - np.concatenate([p_ge_f[:, 1:], np.zeros((n, 1))], 1), 0.0, 1.0)      # P(k0 = k), k = 0..5 (5 = made)
    for t in range(T):
        active = (tries > t)[:, None]
        nxt = np.zeros_like(dist)
        nxt[:, 5] = dist[:, 5]
        for k in range(5):
            draw = np.maximum(H - k, 0.0)
            p_s = np.clip((n_s - k) / np.maximum(D - H, 1.0), 0.0, 1.0)
            stay = (1.0 - p_s) ** draw
            one = draw * p_s * (1.0 - p_s) ** np.maximum(draw - 1, 0.0)
            two = np.where(draw >= 2, draw * (draw - 1) / 2 * p_s ** 2 * (1.0 - p_s) ** np.maximum(draw - 2, 0.0), 0.0)
            more = np.clip(1.0 - stay - one - two, 0.0, 1.0)
            nxt[:, k] += dist[:, k] * stay
            nxt[:, min(k + 1, 5)] += dist[:, k] * one
            nxt[:, min(k + 2, 5)] += dist[:, k] * two
            nxt[:, min(k + 3, 5)] += dist[:, k] * more
        dist = np.where(active, nxt, dist)
    ge_f = np.cumsum(dist[:, ::-1], axis=1)[:, ::-1]
    pf = np.where(need == 4, ge_f[:, 4], ge_f[:, 5])
    out[:, FLUSH] = pf
    # straights: the fullest 5-rank window (7 with Shortcut), ranks present ~ independent at rate q per rank in the deal
    cbar = D / 13.0 * np.where(rules["shortcut"], 1.4, 1.0)
    q = 1.0 - (1.0 - cbar / D) ** H
    need_s = np.where(rules["four_fingers"], 4, 5)
    p_ge_s = np.ones((n, 6))
    for k in range(1, 6):
        pb = np.zeros(n)
        for j in range(k, 6):
            pb += np.exp(_lchoose(np.full(n, 5), j)) * q ** j * (1 - q) ** (5 - j)
        p_ge_s[:, k] = 1.0 - (1.0 - np.clip(pb, 0, 1)) ** 4                 # the best of the overlapping windows
    str_ge = chase(p_ge_s, lambda m: 1.0 - (1.0 - np.minimum((5 - m) * cbar / D, 1.0)) ** np.maximum(H - m, 1.0), 5)
    ps = np.where(need_s == 4, str_ge[:, 4], str_ge[:, 5])
    out[:, STRAIGHT] = ps
    # a flush is one card per rank unless the deck holds duplicates or wild cards: the flush-and-set hands need them, a straight flush is a lucky flush
    dups = np.clip((np.maximum(c - 4, 0).sum(1) + s.enh[:, E["WILD"]]) / 5.0, 0.0, 1.0)
    out[:, STRAIGHT_FLUSH] = pf * 0.03 * (1.0 + ps) * (1.0 + 2.0 * dups)
    out[:, FLUSH_HOUSE] = out[:, FULL_HOUSE] * pf * 0.5 * dups
    out[:, FLUSH_FIVE] = out[:, FIVE_KIND] * pf * 0.5 * dups
    return np.clip(out, 0.0, 1.0)


def _hg_tail_k(n_good, n_total, draw, k):
    """P(X >= k) with k an array (4 or 5)."""
    out = np.zeros(len(n_good))
    for kk in (4, 5):
        sel = k == kk
        if sel.any():
            out[sel] = _hg_tail(n_good[sel], n_total[sel], draw[sel], kk)
    return out


def joker_rules(s: State):
    """Rule jokers held (live): four_fingers, shortcut, smeared, splash, all_faces; and the copy jokers' targets."""
    spec, S = _spec()
    r = {}
    for name in ("four_fingers", "shortcut", "smeared", "splash", "all_faces"):
        col = S[f"sp_rule_{name}"]
        r[name] = ((spec[s.jid, col] > 0.5) & s.live).any(1)
    return r


def build_scores(s: State, rules):
    """[n, 12]: the expected score of a play of each hand type with the jokers held (editions, counters, conditions in expectation),
    and the main hand's level. Also returns per-slot multiplier contributions for the synergy / boss terms."""
    spec, S = _spec()
    K = _keysets()
    n = s.n
    jid, live = s.jid, s.live.astype(np.float64)
    nj = jid.shape[1]
    # deck-level expectations
    mean_chip = (s.rank_share * RANK_CHIPS).sum(1)
    enh_share = s.enh / np.maximum(s.full, 1)[:, None]                     # per card
    face_share = np.where(rules["all_faces"], 1.0, (s.rank_share * FACE_MASK).sum(1))
    even_share = (s.rank_share * EVEN_MASK).sum(1)
    odd_share = (s.rank_share * ODD_MASK).sum(1)
    ace_share = (s.rank_share * ACE_MASK).sum(1)
    ncards = np.where(rules["four_fingers"][:, None] & (IS_FLUSH | (np.arange(N_HANDS) == STRAIGHT))[None, :], np.minimum(N_CARDS, 4.0), N_CARDS[None, :])   # [n, 12]
    ncards = np.where(rules["splash"][:, None], np.maximum(ncards, 5.0), ncards)
    held = np.maximum(s.hand_size[:, None] - ncards, 0.0)                 # cards left in hand when scoring
    # card chips: ranks, bonus (+30), stone (+50, no rank), editions ignored; glass x2 per scoring glass card; steel x1.5 per held steel card
    card_chips = ncards * (mean_chip + 30.0 * enh_share[:, E["BONUS"]] + 50.0 * enh_share[:, E["STONE"]])[:, None]
    card_mult = ncards * (4.0 * enh_share[:, E["MULT"]] + 20.0 / 5.0 * enh_share[:, E["LUCKY"]])[:, None]
    card_x = (2.0 ** (ncards * enh_share[:, E["GLASS"]][:, None])) * (1.5 ** (held * enh_share[:, E["STEEL"]][:, None]))
    # per joker: flat chips / mult, xmult (minus one), conditions
    chips_j = spec[jid, S["sp_chips"]] * live
    mult_j = spec[jid, S["sp_mult"]] * live
    xm_j = spec[jid, S["sp_xmult"]] * live
    chance = spec[jid, S["sp_chance"]]
    unit = np.maximum(spec[jid, S["sp_unit"]], 1.0)
    # counters: a scaler's counter adds to chips / mult, or is the multiplier itself
    scales = spec[jid, S["sp_scales"]] > 0.5
    u_chips = scales & (spec[jid, S["sp_counter_chips"]] > 0.5)
    u_mult = scales & (spec[jid, S["sp_counter_mult"]] > 0.5)
    u_x = scales & (spec[jid, S["sp_counter_xmult"]] > 0.5)
    chips_j = chips_j + np.where(u_chips, s.counter, 0.0) * live
    mult_j = mult_j + np.where(u_mult, s.counter, 0.0) * live
    xm_j = np.where(u_x & (s.counter >= 1.0), s.counter - 1.0, xm_j) * live
    # scales_with: the effect per `unit` of a quantity of the state
    qty = np.zeros((n, nj))
    for name, q in (("money", s.money), ("jokers_held", s.njokers), ("deck_size", np.rint(s.g[:, Cols.DECK] * 52)), ("tarots_used", s.tarots_used),
                    ("planets_used", s.planets_used), ("deck_missing", np.maximum(52 - s.full, 0)), ("blinds_skipped", s.skipped),
                    ("empty_slots", np.maximum(s.jslots - s.njokers, 0)), ("discards_left", s.discards), ("hands_left", s.hands),
                    ("enhanced_cards", s.enh.sum(1)), ("cards_of_suit", s.suits.max(1))):
        col = S.get(f"sp_scales_{name}")
        if col is not None:
            qty = np.where(spec[jid, col] > 0.5, q[:, None], qty)
    sw_any = (qty > 0)
    for name, q in (("hand_level", None), ("times_hand_played", None), ("joker_sell_value", None)):
        col = S.get(f"sp_scales_{name}")
        if col is None:
            continue
        m = spec[jid, col] > 0.5
        if name == "hand_level":
            qv = s.levels.max(1)
        elif name == "times_hand_played":
            qv = s.played.max(1)
        else:
            qv = (s.jf[:, :, JF_SELL] * 10.0 * live).sum(1)
        qty = np.where(m, qv[:, None], qty)
        sw_any |= m
    steps = np.floor(qty / unit)
    chips_j = np.where(sw_any, chips_j * steps, chips_j)
    mult_j = np.where(sw_any, mult_j * steps, mult_j)
    xm_j = np.where(sw_any, xm_j * steps, xm_j)
    # chance: the expectation
    has_chance = chance > 0
    chips_j = np.where(has_chance, chips_j * chance, chips_j)
    mult_j = np.where(has_chance, mult_j * chance, mult_j)
    xm_j = np.where(has_chance, xm_j * chance, xm_j)
    # editions on the jokers themselves
    ed_chips = 50.0 * (s.jf[:, :, JF_FOIL] > 0.5) * live
    ed_mult = 10.0 * (s.jf[:, :, JF_HOLO] > 0.5) * live
    ed_x = np.where((s.jf[:, :, JF_POLY] > 0.5) & (live > 0), 1.5, 1.0)
    # triggers and conditions -> an expected number of firings per play of hand h: [n, nj, 12]
    trig_hand = spec[jid, S["sp_trig_hand"]] > 0.5
    trig_card = spec[jid, S["sp_trig_scored_card"]] > 0.5
    trig_held = spec[jid, S["sp_trig_held_card"]] > 0.5
    trig_other = ~(trig_hand | trig_card | trig_held)
    # the hand requirement: [n, nj, 12] = 1 if hand h contains the required hand
    req = np.full((n, nj), -1, dtype=np.int64)
    for h, name in enumerate(HAND_NAMES):
        col = S.get("sp_hand_" + name.replace(" ", "_").lower())
        if col is not None:
            req = np.where(spec[jid, col] > 0.5, h, req)
    ok_hand = np.where(req[:, :, None] >= 0, CONTAINS.T[np.clip(req, 0, N_HANDS - 1)], 1.0)      # [n, nj, 12]: hand h contains the requirement
    maxc = spec[jid, S["sp_max_cards"]]
    exactc = spec[jid, S["sp_exact_cards"]]
    ok_cards = np.where(maxc[:, :, None] > 0, ncards[:, None, :] <= maxc[:, :, None], True) & np.where(exactc[:, :, None] > 0, ncards[:, None, :] == exactc[:, :, None], True)
    # card matching share for scored_card / held_card triggers: suit, rank class
    share = np.ones((n, nj))
    for i, sname in enumerate("SHCD"):
        m = spec[jid, S[f"sp_suit_{sname}"]] > 0.5
        # a flush-type hand is of the main suit: share 1 when it matches (0 when not); other hands the suit's share of the deck
        share_i = s.suit_share[:, i]
        if rules["smeared"].any():
            share_i = np.where(rules["smeared"], s.suit_share[:, i] + s.suit_share[:, (i + 2) % 4], share_i)
        share = np.where(m, share_i[:, None], share)
    rank_share = np.ones((n, nj))
    for cls, msk in (("face", face_share), ("even", even_share), ("odd", odd_share), ("ace", ace_share)):
        m = spec[jid, S[f"sp_rankclass_{cls}"]] > 0.5
        rank_share = np.where(m, msk[:, None], rank_share)
    spec_mask = spec[jid, S["sp_rankclass_specific"]] > 0.5
    set_mask = spec[jid, S["sp_rankclass_set"]] > 0.5
    rank_bits = spec[jid][:, :, [S[f"sp_rank_{r}"] for r in range(2, 15)]]                    # [n, nj, 13]
    bit_share = (rank_bits * s.rank_share[:, None, :]).sum(2)
    rank_share = np.where(spec_mask | set_mask, bit_share, rank_share)
    match = share * rank_share                                                               # [n, nj]
    tgt = spec[jid, S["sp_target_state"]] > 0.5                                              # a random target each round: a suit (Ancient Joker) or a card (The Idol)
    match = np.where(tgt & (jid != item_id("j_idol")), 0.25, match)
    match = np.where(jid == item_id("j_idol"), 1.0 / 52.0, match)
    suit_flush = np.zeros((n, nj, N_HANDS))
    for i in range(4):
        m = spec[jid, S[f"sp_suit_{'SHCD'[i]}"]] > 0.5
        suit_flush += (m[:, :, None] * (s.main_suit == i)[:, None, None]) * IS_FLUSH[None, None, :]
    has_suit = (spec[jid][:, :, [S[f"sp_suit_{c}"] for c in "SHCD"]] > 0.5).any(2)
    # firings per play: hand triggers once; card triggers per matching scoring card (a flush of the suit: all of them); held: per matching held card
    fir_hand = ok_hand * ok_cards
    card_match = np.where(has_suit[:, :, None] & IS_FLUSH[None, None, :], suit_flush, match[:, :, None])
    fir_card = ok_hand * ok_cards * ncards[:, None, :] * card_match
    fir_held = ok_hand * ok_cards * held[:, None, :] * match[:, :, None]
    fir = np.where(trig_hand[:, :, None], fir_hand, 0.0) + np.where(trig_card[:, :, None], fir_card, 0.0) + np.where(trig_held[:, :, None], fir_held, 0.0)
    fir = np.where(trig_other[:, :, None], 0.0, fir)
    fir = np.where((jid == item_id("j_photograph"))[:, :, None], np.minimum(fir, 1.0), fir)      # the first face card only
    # situational flags: last hand ~ 1 / hands of the plays, zero discards ~ 0.35; a count threshold: every k-th hand (Loyalty Card) fires 1 / k of
    # the time, a deck threshold (Driver's License: 16 enhanced cards) fires when the deck has them
    thr = spec[jid, S["sp_threshold"]]
    needs_enh = spec[jid, S["sp_needs_enh"]] > 0.5
    situ = np.ones((n, nj))
    situ = np.where((thr > 0) & ~needs_enh, 1.0 / np.maximum(thr, 1.0), situ)
    situ = np.where((thr > 0) & needs_enh, (s.enh.sum(1)[:, None] >= thr).astype(np.float64), situ)
    situ = np.where(spec[jid, S["sp_last_hand"]] > 0.5, 1.0 / s.hands[:, None], situ)
    situ = np.where(spec[jid, S["sp_first_hand"]] > 0.5, 1.0 / s.hands[:, None], situ)
    situ = np.where(spec[jid, S["sp_discards_zero"]] > 0.5, 0.35, situ)
    fir = fir * situ[:, :, None]
    # copy jokers: Blueprint copies the joker to its right, Brainstorm the leftmost; each adds its target's contribution again (not another copy)
    copy_w = np.zeros((n, nj, nj))
    for a in range(nj):
        bp = (jid[:, a] == K["blueprint"]) & (live[:, a] > 0)
        if a + 1 < nj:
            copy_w[:, a, a + 1] += bp
        bs = (jid[:, a] == K["brainstorm"]) & (live[:, a] > 0)
        copy_w[:, a, 0] += bs * (a != 0)
    copyable = ~np.isin(jid, [K["blueprint"], K["brainstorm"]])
    copy_gain = np.einsum("nab,nb->nb", copy_w, copyable.astype(np.float64))       # [n, nj]: how many copies each slot receives
    fir = fir * (1.0 + copy_gain)[:, :, None]
    # total per hand h
    add_chips = (chips_j[:, :, None] * fir).sum(1) + (ed_chips[:, :, None] * (fir > 0).any(2, keepdims=True)).sum(1)
    add_mult = (mult_j[:, :, None] * fir).sum(1) + (ed_mult[:, :, None] * (fir > 0).any(2, keepdims=True)).sum(1)
    # a per-firing multiplier: (1 + x)^firings, polychrome once
    xm_tot = np.exp((np.log1p(np.maximum(xm_j, 0.0))[:, :, None] * fir).sum(1))
    xm_tot = xm_tot * np.exp((np.log(ed_x)[:, :, None] * (fir > 0).any(2, keepdims=True)).sum(1))
    # retriggers: card chips and per-card joker effects for matching cards fire again
    retrig = spec[jid, S["sp_retriggers"]] * live * match
    retrig_f = 1.0 + np.minimum(retrig.sum(1), 2.0)[:, None]
    level = s.levels                                                               # [n, 12]
    hand_chips = HAND_CHIPS[None] + HAND_DCHIPS[None] * (level - 1)
    hand_mult = HAND_MULT[None] + HAND_DMULT[None] * (level - 1)
    chips = hand_chips + card_chips * retrig_f + add_chips
    mult = hand_mult + card_mult * retrig_f + add_mult
    score = np.maximum(chips, 1.0) * np.maximum(mult, 1.0) * card_x * xm_tot
    parts = dict(xm_tot=xm_tot, add_mult=add_mult, add_chips=add_chips, chips_j=chips_j, mult_j=mult_j, xm_j=xm_j, fir=fir, copy_gain=copy_gain, match=match,
                 u_x=u_x, u_mult=u_mult, u_chips=u_chips, retrig=retrig)
    return score, parts


def compute_terms(s: State, want_deal=False):
    """-> dict of the nine terms [n], plus diagnostics (main hand, reach, scores; with want_deal the expected best play of the dealt hand alone,
    comparable to the engine's fast_power probe)."""
    spec, S = _spec()
    K = _keysets()
    rules = joker_rules(s)
    score, P = build_scores(s, rules)                                                     # [n, 12]
    n = s.n

    def best_play(reach):
        """The strongest hand landed: P(best = h) = reach_h x prod over the stronger hands of (1 - reach). -> E[score], the main hand (the
        biggest contributor), P(the main hand or a stronger one)."""
        order = np.argsort(-score, axis=1)
        sc_s = np.take_along_axis(score, order, 1)
        re_s = np.take_along_axis(reach, order, 1)
        not_better = np.cumprod(np.concatenate([np.ones((n, 1)), 1.0 - re_s[:, :-1]], 1), axis=1)
        p_best = re_s * not_better
        p_best[:, -1] += np.maximum(1.0 - p_best.sum(1), 0.0)                 # (the high card always lands: the remainder goes to the weakest)
        contrib = sc_s * p_best
        k = np.argmax(contrib, axis=1)
        p_hand = np.zeros_like(p_best)
        np.put_along_axis(p_hand, order, p_best, axis=1)                    # back to hand-type order: P(the best play is hand h)
        best_play.p_hand = p_hand
        return (sc_s * p_best).sum(1), order[np.arange(n), k], np.cumsum(p_best, axis=1)[np.arange(n), k]

    # power: hands x the expected best play over the next target, the better of two ways to play the round: score with every hand after the
    # discards' redraws, or hunt the hand with the discards and all but two hands and score with the last two
    reach_a = hand_reach(s, rules, s.discards)
    reach_b = hand_reach(s, rules, s.discards + np.maximum(s.hands - 2, 0))
    e_a, main_a, cons_a = best_play(reach_a)
    p_hand_a = best_play.p_hand
    e_b, main_b, cons_b = best_play(reach_b)
    p_hand_b = best_play.p_hand
    hands_b = np.minimum(s.hands, 2.0)
    hunt = hands_b * e_b > s.hands * e_a
    e_best = np.where(hunt, hands_b * e_b, s.hands * e_a)
    main = np.where(hunt, main_b, main_a)
    reach_main = np.where(hunt, cons_b, cons_a)
    reach = np.where(hunt[:, None], reach_b, reach_a)
    p_hand = np.where(hunt[:, None], p_hand_b, p_hand_a)
    per_hand = np.log10(np.maximum(s.hands[:, None] * score, 1.0) / np.maximum(s.next_target, 1.0)[:, None])
    power = np.clip(np.log10(np.maximum(e_best, 1.0) / np.maximum(s.next_target, 1.0)), -4.0, 4.0)
    # fail risk: the margin on the next blind in log10 units; quadratic penalty inside half an order of magnitude, capped at -1
    margin = power
    risk = -np.clip((0.5 - margin) / 0.5, 0.0, 1.0) ** 2
    # scaling: the growing jokers' counters moved by their rates to the ante-8 boss, as a log10 multiplier of the score; plus xMult sources
    jid, live = s.jid, s.live
    rate = K["grow_rate"][jid] * live
    cap = K["grow_cap"][jid]
    isx = K["grow_unit"][jid] > 0
    cur = s.counter
    proj = np.minimum(cur + rate * s.rounds_left[:, None], np.where(cap > 0, cap, np.inf))
    base_mult = (HAND_MULT[main] + HAND_DMULT[main] * (s.levels[np.arange(n), main] - 1)) + P["add_mult"][np.arange(n), main]
    base_chips = (HAND_CHIPS[main] + HAND_DCHIPS[main] * (s.levels[np.arange(n), main] - 1)) + P["add_chips"][np.arange(n), main]
    gain_x = np.where(isx & (rate > 0), np.log10(np.maximum(proj, 1.0) / np.maximum(cur, 1.0)), 0.0)
    gain_m = np.where(P["u_mult"] & (rate > 0), np.log10((base_mult[:, None] + proj - cur) / np.maximum(base_mult[:, None], 1.0)), 0.0)
    gain_c = np.where(P["u_chips"] & (rate > 0), np.log10((base_chips[:, None] + proj - cur) / np.maximum(base_chips[:, None], 1.0)), 0.0)
    n_x = ((K["xmult"][jid] | (P["xm_j"] > 0)) & live).sum(1) + (P["copy_gain"] * live).sum(1)
    scaling = (gain_x + gain_m + gain_c).sum(1) + 0.25 * n_x
    # consistency: the main hand's reach
    consistency = reach_main
    # deck: concentration behind the main hand, thinning, enhanced cards
    suit_conc = s.suit_share.max(1) * 4.0 - 1.0
    rank_conc = (s.rank_share ** 2).sum(1) * 13.0 - 1.0
    thinning = np.log2(52.0 / np.maximum(s.full, 8.0))
    enh_share = s.enh.sum(1) / np.maximum(s.full, 1)
    flushy = IS_FLUSH[main].astype(np.float64)
    setty = IS_SET[main].astype(np.float64)
    deck = suit_conc * (0.3 + 0.7 * flushy) + np.clip(rank_conc, 0, 3) * (0.3 + 0.7 * setty) + 0.5 * thinning + 2.0 * enh_share + 0.05 * s.seals.sum(1)
    # economy: dollars per round equivalent, log scale
    cap_i = np.where(s.vouchers[:, V["money_tree"]], 20, np.where(s.vouchers[:, V["seed_money"]], 10, 5))
    interest = np.minimum(np.floor(np.maximum(s.money, 0) / 5.0), cap_i)
    income = (spec[jid, S["sp_income_round"]] * live).sum(1)
    rentals = ((s.jf[:, :, JF_RENTAL] > 0.5) & live).sum(1)
    sell = (s.jf[:, :, JF_SELL] * 10.0 * live).sum(1)
    econ_jokers = (K["econ"][jid] & live).sum(1)
    economy = np.log2(1.0 + np.maximum(s.money + 3.0 * (interest + income - 3.0 * rentals) + 0.25 * sell + 4.0 * econ_jokers, 0.0) / 10.0)
    # synergy: copies next to multipliers; hand / suit / face / enhancement jokers that fit the build; retriggers with per-card jokers
    xm_slot = (P["xm_j"] > 0) | (K["xmult"][jid] & live) | K["scaler"][jid] & live
    copy_hit = (P["copy_gain"] * xm_slot).sum(1)
    hand_fit = ((P["fir"][np.arange(n), :, main] > 0) & (spec[jid][:, :, [S["sp_hand_" + h.replace(" ", "_").lower()] for h in HAND_NAMES]].max(2) > 0.5) & live).sum(1)
    suit_fit = (((spec[jid][:, :, [S[f"sp_suit_{c}"] for c in "SHCD"]] > 0.5).any(2)) & live).astype(np.float64) * (P["match"] * 4.0 - 1.0)
    suit_fit = np.clip(suit_fit, -1, 3).sum(1)
    face_fit = ((spec[jid, S["sp_rankclass_face"]] > 0.5) & live).sum(1) * np.clip((P["match"].max(1) * 13.0 / 3.0 - 1.0), -1, 3)
    steel = s.enh[:, E["STEEL"]]
    enh_fit = (((jid == item_id("j_steel_joker")) | (jid == item_id("j_baron"))) & live).sum(1) * np.minimum(steel / 4.0, 2.0) \
        + ((jid == item_id("j_glass")) & live).sum(1) * np.minimum(s.enh[:, E["GLASS"]] / 3.0, 2.0) \
        + ((jid == item_id("j_lucky_cat")) & live).sum(1) * np.minimum(s.enh[:, E["LUCKY"]] / 4.0, 2.0)
    retrig_fit = np.minimum(P["retrig"].sum(1), 2.0) * np.minimum(((P["fir"].max(2) > 0) & (spec[jid, S["sp_trig_scored_card"]] > 0.5) & live).sum(1), 3.0) * 0.5
    rule_fit = (rules["four_fingers"] | rules["shortcut"] | rules["smeared"]).astype(np.float64) * (IS_FLUSH[main] | (main == STRAIGHT))
    synergy = 1.0 * copy_hit + 0.5 * hand_fit + 0.5 * suit_fit + 0.5 * face_fit + 0.5 * enh_fit + retrig_fit + 0.7 * rule_fit
    # flex: room to grow and to pivot
    open_slots = np.maximum(s.jslots - s.njokers, 0)
    held_cons = s.cused
    rerolls_afford = np.minimum(np.maximum(s.money, 0) / np.maximum(s.reroll_cost, 1.0), 4.0)
    tag_val = (s.tag * TAG_VALUES[None]).sum(1) / 10.0
    flex = 0.35 * open_slots + 0.5 * held_cons + 0.3 * rerolls_afford + tag_val + 0.3 * (s.hands - 4) + 0.25 * (s.discards - 3) + 0.3 * (s.hand_size - 8)
    # boss: the known boss of this ante (until it is beaten), else the pool's mean
    expo = boss_exposure(s, main, reach, P)                                      # [n, len(BOSS_KEYS)]
    known = s.boss_known & ~s.next_after_boss
    pool = np.where(s.ante[:, None] >= 8, np.isin(np.arange(len(BOSS_KEYS)), FINISHER_IDX), np.isin(np.arange(len(BOSS_KEYS)), REGULAR_IDX)).astype(np.float64)
    mean_expo = (expo * pool).sum(1) / pool.sum(1)
    boss = -np.where(known, expo[np.arange(n), s.boss_idx], mean_expo) * np.where(s.next_is_boss, 1.0, 0.6)
    terms = dict(power=power, scaling=scaling, consistency=consistency, deck=deck, economy=economy, synergy=synergy, risk=risk, flex=flex, boss=boss)
    diag = dict(main=main, reach=reach, p_hand=p_hand, score=score, per_hand=per_hand, margin=margin, n_xmult=n_x, interest=interest, e_best=e_best / s.hands, hunt=hunt)
    if want_deal:
        diag["e_deal"] = best_play(hand_reach(s, rules, np.zeros(n)))[0]
    return terms, diag


def boss_exposure(s: State, main, reach, P):
    """[n, bosses]: how much each boss would cost this build, in rough log10-of-score units (0 = nothing)."""
    n = s.n
    nb = len(BOSS_KEYS)
    ex = np.zeros((n, nb))
    rm = reach[np.arange(n), main]
    one_hand = IS_SET[main] | (main == HC)
    lvl = s.levels[np.arange(n), main]
    ex[:, BK["needle"]] = 0.4 + 1.2 * (1.0 - rm)                                  # one hand only: unreliable builds die
    ex[:, BK["water"]] = 0.3 + 0.8 * (1.0 - rm) * (s.discards > 0)              # no discards
    ex[:, BK["wall"]] = 0.5                                                        # x4 target (x2 of a boss): margin
    ex[:, BK["violet_vessel"]] = 0.8
    ex[:, BK["eye"]] = 0.3 + 0.6 * one_hand                                        # no repeat hand types
    ex[:, BK["mouth"]] = 0.2 + 0.2 * (1.0 - one_hand)
    ex[:, BK["psychic"]] = 0.2 + 0.6 * one_hand * (N_CARDS[main] < 5)               # five cards
    ex[:, BK["arm"]] = 0.1 + 0.3 * np.clip((lvl - 1) / np.maximum(lvl, 1), 0, 1)
    ex[:, BK["flint"]] = 0.6
    ex[:, BK["hook"]] = 0.3
    ex[:, BK["manacle"]] = 0.25 + 0.2 * (s.hand_size <= 8)
    ex[:, BK["pillar"]] = 0.25
    ex[:, BK["serpent"]] = 0.1
    ex[:, BK["ox"]] = np.clip(s.money / 60.0, 0, 0.8)
    ex[:, BK["tooth"]] = 0.2
    for k in ("house", "wheel", "fish", "mark"):
        ex[:, BK[k]] = 0.3
    faces = (s.rank_share * FACE_MASK).sum(1)
    ex[:, BK["plant"]] = 0.1 + 1.5 * faces * (P["match"].max(1) > 0.9) + 0.3 * faces
    for k, i in SUIT_BOSS.items():
        share = s.suit_share[:, i]
        ex[:, BK[k]] = 0.1 + np.where(IS_FLUSH[main] & (s.main_suit == i), 1.2, 0.4 * share)
    ex[:, BK["verdant_leaf"]] = 0.5 + 0.3 * (s.njokers >= s.jslots)
    xm_share = np.where(P["xm_tot"][np.arange(n), main] > 1.0, np.log10(P["xm_tot"][np.arange(n), main]), 0.0)
    ex[:, BK["crimson_heart"]] = 0.3 + 0.4 * np.clip(xm_share / np.maximum(s.njokers, 1), 0, 1)
    ex[:, BK["amber_acorn"]] = 0.3 + 0.3 * (P["copy_gain"].sum(1) > 0)
    ex[:, BK["cerulean_bell"]] = 0.3
    return ex


# ------------------------------------------------------------------ weights, potentials, blend
PRIOR_W = {   # the spec's starting priors: Low 0.25, Medium 0.5, High 1.0, Very high 1.5 (x the term's scale)
    "early": dict(power=0.5, scaling=0.25, consistency=1.0, deck=0.25, economy=1.5, synergy=0.5, risk=1.0, flex=1.0, boss=0.25),
    "mid": dict(power=1.0, scaling=0.5, consistency=1.0, deck=0.5, economy=0.5, synergy=1.0, risk=1.0, flex=0.5, boss=0.5),
    "late": dict(power=1.0, scaling=1.5, consistency=0.5, deck=1.0, economy=0.25, synergy=1.0, risk=1.5, flex=0.25, boss=1.0),
}
PRIOR_B = {"early": -1.0, "mid": -2.0, "late": -4.0}
PRIOR_BLEND = dict(c_early=3.5, c_late=6.5, tau=0.7, conv_early=0.05, conv_mid=0.3)


def default_weights():
    return {"terms": list(TERMS), "phases": {ph: {"w": [PRIOR_W[ph][t] for t in TERMS], "b": PRIOR_B[ph], "mean": [0.0] * len(TERMS), "std": [1.0] * len(TERMS)} for ph in PHASES},
            "blend": dict(PRIOR_BLEND), "fitted": False}


def load_weights(path=WEIGHTS_PATH):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default_weights()


def blend_weights(ante, blend):
    """[n, 3]: the ante's smooth early / mid / late weights (sum to 1)."""
    a = np.asarray(ante, dtype=np.float64)
    ae = sigmoid((blend["c_early"] - a) / blend["tau"])
    al = sigmoid((a - blend["c_late"]) / blend["tau"])
    am = np.clip(1.0 - ae - al, 0.0, 1.0)
    w = np.stack([ae, am, al], 1)
    return w / w.sum(1, keepdims=True)


class RunEval:
    """The eval with a weight set."""

    def __init__(self, weights=None):
        self.w = weights if weights is not None else load_weights()
        self.W = np.array([[self.w["phases"][ph]["w"][k] for k in range(len(TERMS))] for ph in PHASES])          # [3, 9]
        self.B = np.array([self.w["phases"][ph]["b"] for ph in PHASES])
        self.MU = np.array([self.w["phases"][ph].get("mean", [0.0] * len(TERMS)) for ph in PHASES])
        self.SD = np.array([self.w["phases"][ph].get("std", [1.0] * len(TERMS)) for ph in PHASES])

    def terms(self, g, jid, jf, cid=None, want_deal=False):
        s = State(g, jid, jf, cid)
        terms, diag = compute_terms(s, want_deal)
        return np.stack([terms[t] for t in TERMS], 1), s, diag

    def potentials(self, T):
        """[n, 9] terms -> [n, 3] phase potentials (probabilities)."""
        z = (T[:, None, :] - self.MU[None]) / self.SD[None]
        logits = (z * self.W[None]).sum(2) + self.B[None]
        return sigmoid(logits)

    def __call__(self, g, jid, jf, cid=None, diag=False):
        T, s, dg = self.terms(g, jid, jf, cid)
        pot = self.potentials(T)
        bw = blend_weights(s.ante, self.w["blend"])
        conv = np.array([self.w["blend"]["conv_early"], self.w["blend"]["conv_mid"], 1.0])
        blended = (bw * pot).sum(1)                                   # the spec's E(A): the phase potentials blended
        win = (bw * pot * conv[None]).sum(1)                          # the same read as a win chance (each potential converted to a win rate)
        out = dict(score=blended, win=win, early=pot[:, 0], mid=pot[:, 1], late=pot[:, 2], terms=T)
        if diag:
            out["diag"] = dg
            out["state"] = s
        return out


def from_vecenv(v, idx=None):
    """The eval's inputs for the environments idx of a CppVecEnv (all by default)."""
    o = v.obs
    idx = slice(None) if idx is None else np.asarray(idx)
    return dict(g=np.asarray(o["g"])[idx], jid=np.asarray(o["jid"])[idx], jf=np.asarray(o["jf"])[idx], cid=np.asarray(o["cid"])[idx])


def main(argv=None):
    import argparse
    import time
    import torch
    from .engine_env import CppVecEnv
    from .env import A_SELECT
    from . import azc
    from .model import load_model
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", nargs="?", default="checkpoints/ppo_gold_v30_best.pt")
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--steps", type=int, default=400)
    a = p.parse_args(argv)
    ev = RunEval()
    v = CppVecEnv(a.games, "RED", "GOLD", seed0=123, threads=0)
    v.set_auto_round(True, 1.5, 0)
    model = load_model(a.ckpt, "cpu").eval()
    done = np.zeros(a.games, dtype=bool)
    t_eval = 0.0
    n_eval = 0
    for _ in range(a.steps):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        mask = np.asarray(v.obs["mask"])
        inp = from_vecenv(v, live)
        t0 = time.perf_counter()
        r = ev(**inp, diag=True)
        t_eval += time.perf_counter() - t0
        n_eval += len(live)
        at_select = r["state"].state == I_BLIND_SELECT
        for k in np.flatnonzero(at_select)[:2]:
            i = live[k]
            print(f"game {i} ante {r['state'].ante[k]} blind {r['state'].blind[k]} ${r['state'].money[k]:.0f} main {HAND_NAMES[r['diag']['main'][k]]:>15} "
                  f"score {r['score'][k]:.3f} win {r['win'][k]:.3f} e/m/l {r['early'][k]:.2f}/{r['mid'][k]:.2f}/{r['late'][k]:.2f} "
                  + " ".join(f"{t}={x:.2f}" for t, x in zip(TERMS, r["terms"][k])))
        ob = azc.obs_batch(v, list(live), "cpu")
        with torch.no_grad():
            lg, _ = model(ob)
        lg = lg.float().masked_fill(~torch.from_numpy(mask[live]), -1e9)
        act = np.full(a.games, -1, dtype=np.int64)
        act[live] = lg.argmax(1).numpy()
        _, d, info = v.step(act)
        done |= d
    print(f"eval: {1e6 * t_eval / max(n_eval, 1):.0f} us per state over {n_eval} states (batches of up to {a.games})")


if __name__ == "__main__":
    main()
