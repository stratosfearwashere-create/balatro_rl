"""RL environment: fixed action space + observation/action feature encoding.

The same `encode()` is used for the simulator and for the real game bridge, so a
policy trained in simulation sees identical inputs when playing the real game.

Card plays/discards use 218 candidate slots each. With 8 or fewer cards in hand the slots
are exactly every 1-5 card subset. With a bigger hand (Juggler, Turtle Bean, Paint Brush ...)
the slots hold the 218 highest-scoring plays and the 218 most promising discards; which cards
each slot holds is part of the observation (`subs`).
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Optional

import numpy as np

from .sim import fastscore
from .sim.cards import ENHANCEMENTS, EDITIONS, SEALS
from .sim.game import Game, Consumable, MAX_HAND, MAX_JOKERS, MAX_CONSUMABLES, MAX_SHOP, MAX_PACK, MAX_VOUCHERS
from .sim.hands import N_HANDS
from .sim.items import BOSS_KEYS, TAGS, VOUCHERS, PLANETS, item_id, VOCAB
from .sim.jokers import Joker
from .sim.scoring import Plan

# ------------------------------------------------------------------ action layout
BASE_HAND = 8
SUBSETS = [c for k in range(1, 6) for c in combinations(range(BASE_HAND), k)]
N_SUB = len(SUBSETS)                                   # 218 candidate slots
MAX_ENUM_CARDS = 10                                    # enumerate subsets over at most 10 cards
A_PLAY = 0
A_DISC = A_PLAY + N_SUB
A_SELECT = A_DISC + N_SUB
A_SKIP = A_SELECT + 1
A_REROLL_BOSS = A_SKIP + 1
A_BUY = A_REROLL_BOSS + 1
A_BUY_PACK = A_BUY + MAX_SHOP
A_VOUCHER = A_BUY_PACK + 2                             # voucher 1-2
A_REROLL = A_VOUCHER + MAX_VOUCHERS
A_LEAVE = A_REROLL + 1
A_SELL_J = A_LEAVE + 1
A_SELL_C = A_SELL_J + MAX_JOKERS
A_USE_C = A_SELL_C + MAX_CONSUMABLES
A_PICK = A_USE_C + MAX_CONSUMABLES
A_PSKIP = A_PICK + MAX_PACK
A_SWAP = A_PSKIP + 1                                   # swap joker i with joker i+1
N_ACTIONS = A_SWAP + MAX_JOKERS - 1

ACTION_TYPES = ["play", "discard", "select", "skip", "reroll_boss", "buy", "buy_pack", "voucher", "reroll",
                "leave", "sell_joker", "sell_cons", "use_cons", "pick", "pack_skip", "swap_joker"]
_BOUNDS = [A_PLAY, A_DISC, A_SELECT, A_SKIP, A_REROLL_BOSS, A_BUY, A_BUY_PACK, A_VOUCHER, A_REROLL, A_LEAVE,
           A_SELL_J, A_SELL_C, A_USE_C, A_PICK, A_PSKIP, A_SWAP, N_ACTIONS]
_TYPE_OF = np.zeros(N_ACTIONS, dtype=np.int64)
for _t, (_lo, _hi) in enumerate(zip(_BOUNDS[:-1], _BOUNDS[1:])):
    _TYPE_OF[_lo:_hi] = _t

MAX_REROLLS_PER_SHOP = 8
MAX_SWAPS_PER_PHASE = 4
MAX_STEPS = 2500

# ------------------------------------------------------------------ feature sizes
STATES = ["BLIND_SELECT", "SELECTING_HAND", "SHOP", "PACK"]
PACK_KINDS = ["arcana", "celestial", "standard", "buffoon", "spectral"]
VOUCHER_KEYS = list(VOUCHERS)
F_CARD = 1 + 13 + 4 + len(ENHANCEMENTS) + len(EDITIONS) + len(SEALS) + 3
F_JOKER = 1 + len(EDITIONS) + 7
F_ACT = len(ACTION_TYPES) + 1 + N_HANDS + 5 + 6 + 13 + 8 + 2
VOCAB_SIZE = len(VOCAB)
_OFF_PLAY = len(ACTION_TYPES)
RATIO_COL = _OFF_PLAY + 1 + N_HANDS                  # score / chips still needed


def _log(x, scale=6.0):
    return math.log10(max(x, 0) + 1) / scale


def global_size() -> int:
    return len(encode_global(Game(seed=0)))


class Counters:
    """Per-phase counters the env tracks to stop endless reroll/swap loops."""

    def __init__(self):
        self.rerolls = 0
        self.swaps = 0


def encode_global(g: Game, cnt: Optional[Counters] = None) -> np.ndarray:
    cnt = cnt or Counters()
    f = []
    f.append(g.ante / 8)
    f += [float(g.blind_idx == i) for i in range(3)]
    f += [float(g.state == s) for s in STATES]
    target = g.target if g.state == "SELECTING_HAND" else g.blind_target(min(g.blind_idx, 2))
    f.append(_log(target))
    f.append(min(2.0, g.chips / max(target, 1)) if g.state == "SELECTING_HAND" else 0.0)
    f.append(_log(max(target - g.chips, 0)))
    f.append(g.hands_left / 5)
    f.append(g.discards_left / 5)
    f.append(g.round_hands() / 5)
    f.append(g.round_discards() / 5)
    f.append(g.effective_hand_size() / 8)
    f.append(len(g.hand) / 8)
    f.append(min(g.money, 100) / 50)
    f.append(_log(max(g.money, 0), 3))
    f.append(float(g.money < 0))
    f.append(-g.debt_limit() / 20)
    f.append(g.reroll_cost / 10)
    f.append(g.free_rerolls / 2)
    f.append(cnt.rerolls / MAX_REROLLS_PER_SHOP)
    f.append(cnt.swaps / MAX_SWAPS_PER_PHASE)
    boss = [0.0] * len(BOSS_KEYS)
    if g.boss in BOSS_KEYS:
        boss[BOSS_KEYS.index(g.boss)] = 1.0
    f += boss
    f.append(float(g.boss_active() != ""))
    f.append(float(g.boss_disabled))
    f.append(float(g.can_reroll_boss()))
    tag = [0.0] * len(TAGS)
    if g.state == "BLIND_SELECT" and g.blind_idx < 2 and g.tags_offered[g.blind_idx] in TAGS:
        tag[TAGS.index(g.tags_offered[g.blind_idx])] = 1.0
    f += tag
    f.append(g.blinds_skipped / 5)
    f += [lv / 10 for lv in g.hand_levels]
    f += [min(p, 60) / 30 for p in g.hand_played]
    f += [min(p, 5) / 3 for p in g.hand_played_round]
    rc = [0] * 13
    sc = [0] * 4
    for c in g.deck if g.state == "SELECTING_HAND" else g.full_deck:
        if not c.is_stone:
            rc[c.rank - 2] += 1
            sc[c.suit] += 1
    f += [x / 4 for x in rc] + [x / 13 for x in sc]
    f.append(len(g.deck) / 52)
    f.append(len(g.full_deck) / 52)
    ec = [0] * len(ENHANCEMENTS)
    seals = [0] * len(SEALS)
    for c in g.full_deck:
        if c.enh in ENHANCEMENTS:
            ec[ENHANCEMENTS.index(c.enh)] += 1
        seals[SEALS.index(c.seal)] += 1
    f += [x / 10 for x in ec[1:]] + [x / 10 for x in seals[1:]]
    f.append(g.joker_slots / 7)
    f.append(len(g.jokers) / 7)
    f.append(g.consumable_slots / 3)
    f.append(g.cons_used() / 3)
    f += [float(v in g.vouchers) for v in VOUCHER_KEYS]
    f += [_log(g.blind_target(i)) for i in range(3)]
    f += [float(g.state == "PACK" and g.pack_kind == k) for k in PACK_KINDS]
    f.append(g.pack_picks / 2 if g.state == "PACK" else 0.0)
    f.append(float(g.deck_type == "PLASMA"))
    f.append(g.tarots_used / 20)
    f.append(len(g.planets_used) / 12)
    return np.asarray(f, dtype=np.float32)


def encode_card(c, forced=False) -> list[float]:
    v = [1.0]
    r = [0.0] * 13
    s = [0.0] * 4
    hidden = bool(getattr(c, "hidden", False))
    if not c.is_stone and not hidden:
        r[c.rank - 2] = 1.0
        s[c.suit] = 1.0
    v += r + s
    if hidden:
        v += [0.0] * (len(ENHANCEMENTS) + len(EDITIONS) + len(SEALS))
    else:
        v += [float(c.enh == e) for e in ENHANCEMENTS]
        v += [float(c.edition == e) for e in EDITIONS]
        v += [float(c.seal == e) for e in SEALS]
    v += [float(c.debuffed), float(forced), float(hidden)]
    return v


def encode_joker_feats(j: Joker) -> list[float]:
    if j.hidden:
        return [1.0] + [0.0] * (len(EDITIONS) + 6) + [1.0]
    val = j.state.get("val", 0)
    return [1.0] + [float(j.edition == e) for e in EDITIONS] + [
        float(j.eternal), (j.perishable or 0) / 5, float(j.rental), float(j.debuffed),
        j.sell_value() / 10, math.log1p(abs(float(val))) / 4, 0.0]


# ------------------------------------------------------------------ card-subset candidates
# most ranks present in any 5-rank window (A-5 .. T-A), indexed by a bitmask of ranks present
# (bit r for rank r, bit 1 = low ace)
_BEST_WINDOW = [max(((m >> s) & 31).bit_count() for s in range(1, 11)) for m in range(1 << 15)]


def _kept_feats(hand, sub) -> list[float]:
    # hot path (called for every discard candidate): one pass, rank presence as a bitmask
    suit = [0, 0, 0, 0]
    ranks = [0] * 15
    n_kept = 0
    for i, c in enumerate(hand):
        if i in sub or c.hidden:
            continue
        e = c.enh
        if e == "STONE" or e == "HIDDEN":
            continue
        if e == "WILD":
            suit[0] += 1; suit[1] += 1; suit[2] += 1; suit[3] += 1
        else:
            suit[c.suit] += 1
        ranks[c.rank] += 1
        n_kept += 1
    present = 0
    for r in range(2, 15):
        if ranks[r]:
            present |= 1 << r
    if present & (1 << 14):
        present |= 2                     # ace also counts low
    best_window = _BEST_WINDOW[present]
    n_disc = len(sub)
    return [max(suit) / 5, max(ranks) / 4, best_window / 5,
            sum(1 for r in ranks if r >= 2) / 3,
            sum(hand[i].chip_value() for i in sub) / (11 * max(1, n_disc)),
            n_kept / 8]


def _kept_feats_many(hand, subs) -> list[list[float]]:
    got = fastscore.kept_feats_many(hand, subs)
    return got if got is not None else [_kept_feats(hand, s) for s in subs]


def candidates(g: Game, plan: Optional[Plan] = None):
    """Returns (plays, discards): lists of N_SUB entries, each a tuple of hand positions or None,
    plus a dict of predicted (score, hand type) for the play candidates."""
    n = len(g.hand)
    forced = g.forced_pos()
    plan = plan or Plan(g)
    view = g.hand_view()
    preds = {}
    if n <= BASE_HAND:
        plays = [s if (s[-1] < n and (forced < 0 or forced in s)) else None for s in SUBSETS]
        valid = [s for s in plays if s is not None]
        preds = dict(zip(valid, g.predict_many(valid, plan, view)))
        discs = list(plays) if g.discards_left > 0 else [None] * N_SUB
        return plays, discs, preds
    # big hand: enumerate plays over the (at most) 10 most useful cards,
    # and discards over the 10 least useful ones
    suit_cnt = [sum(1 for c in g.hand if c.has_suit(st)) for st in range(4)]
    rank_cnt = {}
    for c in g.hand:
        rank_cnt[c.rank] = rank_cnt.get(c.rank, 0) + 1

    def usefulness(i):
        c = g.hand[i]
        v = g.card_value(c)
        if not c.is_stone:
            v += 3 * suit_cnt[c.suit] + 8 * rank_cnt.get(c.rank, 0)
        return v
    order = sorted(range(n), key=usefulness, reverse=True)
    pool = sorted(order[:MAX_ENUM_CARDS])
    low = sorted(order[-MAX_ENUM_CARDS:])
    if forced >= 0:
        if forced not in pool:
            pool = sorted(pool[:-1] + [forced])
        if forced not in low:
            low = sorted(low[1:] + [forced])
    allsubs = [s for k in range(1, 6) for s in combinations(pool, k) if forced < 0 or forced in s]
    scored = []
    for s, p in zip(allsubs, g.predict_many(allsubs, plan, view)):
        preds[s] = p
        scored.append((-p[0], len(s), s))
    scored.sort()
    plays = [x[2] for x in scored[:N_SUB]]
    plays += [None] * (N_SUB - len(plays))
    discs = [None] * N_SUB
    if g.discards_left > 0:
        dsubs = [s for k in range(1, 6) for s in combinations(low, k) if forced < 0 or forced in s]
        key = {s: -(3 * kf[0] + 2 * kf[1] + 2 * kf[2] + kf[3] - kf[4] + 0.1 * len(s))
               for s, kf in zip(dsubs, _kept_feats_many(g.hand, dsubs))}
        dsubs.sort(key=key.__getitem__)
        discs = dsubs[:N_SUB] + [None] * (N_SUB - min(N_SUB, len(dsubs)))
    return plays, discs, preds


def target_candidates(g: Game) -> list:
    """Card subsets a consumable waiting for targets may take, laid out on the play slots (None = unused).
    With up to 8 cards every subset keeps its usual slot; with more, 3-card consumables choose among 10
    cards that include the fixed rule's choice (auto_targets), so the slots can always express it."""
    cards = g.target_cards()
    lo, hi = g.target_range(g.targeting["cons"])
    n = len(cards)
    name = g.targeting["cons"].name
    aura = name == "aura"                           # Aura only takes a card without an edition

    def ok(s):
        return not aura or all(cards[i].edition == "" for i in s)
    if n <= BASE_HAND:
        return [s if s[-1] < n and lo <= len(s) <= hi and ok(s) else None for s in SUBSETS]
    auto = {c.uid for c in g.auto_targets(name, cards)}
    order = sorted(range(n), key=lambda i: (cards[i].uid not in auto, -g.card_value(cards[i])))
    pool = sorted(order[:n if hi <= 2 else MAX_ENUM_CARDS])
    subs = [s for r in range(lo, hi + 1) for s in combinations(pool, r) if ok(s)]
    return subs[:N_SUB] + [None] * (N_SUB - min(N_SUB, len(subs)))


# ------------------------------------------------------------------ legality
def legal_mask(g: Game, cnt: Counters, plays=None, discs=None) -> np.ndarray:
    m = np.zeros(N_ACTIONS, dtype=bool)
    if g.targeting is not None:                       # only the target choice, nothing else
        for i in range(N_SUB):
            m[A_PLAY + i] = plays[i] is not None
        return m
    st = g.state
    if st == "SELECTING_HAND":
        for i in range(N_SUB):
            m[A_PLAY + i] = plays[i] is not None
            m[A_DISC + i] = discs[i] is not None and g.discards_left > 0
    elif st == "BLIND_SELECT":
        m[A_SELECT] = True
        m[A_SKIP] = g.blind_idx < 2
        m[A_REROLL_BOSS] = g.can_reroll_boss()
    elif st == "SHOP":
        for i, it in enumerate(g.shop[:MAX_SHOP]):
            m[A_BUY + i] = g.can_buy(it)
        for i, it in enumerate(g.shop_packs[:2]):
            m[A_BUY_PACK + i] = g.can_afford(it.cost)
        for i, it in enumerate(g.shop_vouchers[:MAX_VOUCHERS]):
            m[A_VOUCHER + i] = g.can_afford(it.cost)
        m[A_REROLL] = cnt.rerolls < MAX_REROLLS_PER_SHOP and (g.free_rerolls > 0 or g.can_afford(g.reroll_cost))
        m[A_LEAVE] = True
    elif st == "PACK":
        for i in range(min(len(g.pack_cards), MAX_PACK)):
            m[A_PICK + i] = g.pack_pick_ok(i)
        m[A_PSKIP] = True
    if st in ("SHOP", "SELECTING_HAND"):
        for i, j in enumerate(g.jokers[:MAX_JOKERS]):
            m[A_SELL_J + i] = not j.eternal
        for i, c in enumerate(g.consumables[:MAX_CONSUMABLES]):
            m[A_SELL_C + i] = True
            cards = g.hand if st == "SELECTING_HAND" else []
            m[A_USE_C + i] = g.consumable_usable(c, cards)
        if cnt.swaps < MAX_SWAPS_PER_PHASE and len(g.jokers) >= 2:
            for i in range(min(len(g.jokers), MAX_JOKERS) - 1):
                m[A_SWAP + i] = True
    return m


# ------------------------------------------------------------------ action features
def _item_feats(g: Game, obj, cost: int) -> list[float]:
    """13 generic item features + 8 playing-card features + 2 planet features."""
    ed = ""
    eternal = perish = rental = 0.0
    rarity = 0.0
    slot_ok = 1.0
    sell = 0.0
    card = [0.0] * 8
    planet = [0.0, 0.0]
    if isinstance(obj, Joker):
        ed = obj.edition
        eternal, perish, rental = float(obj.eternal), (obj.perishable or 0) / 5, float(obj.rental)
        rarity = obj.d.rarity / 3
        slot_ok = float(ed == "NEGATIVE" or len(g.jokers) < g.joker_slots)
        sell = obj.sell_value() / 10
    elif isinstance(obj, Consumable):
        slot_ok = float(g.cons_room())
        if obj.kind == "planet":
            h = PLANETS[obj.name]
            planet = [g.hand_levels[h] / 10, min(g.hand_played[h], 60) / 30]
    elif obj is not None and hasattr(obj, "rank"):
        c = obj
        card = [0.0 if c.is_stone else c.rank / 14] + [float(not c.is_stone and c.suit == s) for s in range(4)] + [
            float(c.enh != ""), float(c.edition != ""), float(c.seal != "")]
        ed = c.edition
    base = [cost / 10, float(g.can_afford(cost))] + [float(ed == e) for e in EDITIONS] + [
        eternal, perish, rental, rarity, slot_ok, sell]
    return base + card + planet


def action_features(g: Game, mask: np.ndarray, plays, discs, preds):
    """Returns (item ids [N_ACTIONS], dense features [N_ACTIONS, F_ACT]). Only legal rows are filled."""
    ids = np.zeros(N_ACTIONS, dtype=np.int64)
    feats = np.zeros((N_ACTIONS, F_ACT), dtype=np.float32)
    feats[np.arange(N_ACTIONS), _TYPE_OF] = 1.0
    off_play = _OFF_PLAY
    off_disc = off_play + 1 + N_HANDS + 5
    off_item = off_disc + 6

    if g.targeting is not None:
        # each play slot is a set of target cards: tagged with the consumable being applied, so the
        # network knows what the cards are for (the model adds the chosen cards' embeddings itself)
        cons = g.targeting["cons"]
        cid, cf = item_id(cons.key), _item_feats(g, cons, 0)
        for i, sub in enumerate(plays):
            if sub is not None:
                ids[A_PLAY + i] = cid
                feats[A_PLAY + i, off_play] = len(sub) / 5
                feats[A_PLAY + i, off_item:off_item + 23] = cf
        return ids, feats

    if g.state == "SELECTING_HAND":
        remaining = max(g.target - g.chips, 1)
        best = 0.0
        for i, sub in enumerate(plays):
            if sub is None:
                continue
            sc, h = preds[sub]
            best = max(best, sc)
            row = feats[A_PLAY + i]
            row[off_play] = len(sub) / 5
            row[off_play + 1 + h] = 1.0
            b = RATIO_COL
            row[b] = min(5.0, sc / remaining)
            row[b + 1] = _log(sc, 7)
            row[b + 2] = float(sc >= remaining)
            row[b + 3] = float(g.hands_left == 1)
            row[b + 4] = min(5.0, sc * g.hands_left / remaining)
        live = [(i, sub) for i, sub in enumerate(discs) if sub is not None and mask[A_DISC + i]]
        for (i, sub), kf in zip(live, _kept_feats_many(g.hand, [s for _, s in live])):
            drow = feats[A_DISC + i]
            drow[off_play] = len(sub) / 5
            drow[RATIO_COL] = min(5.0, best / remaining)       # the best play given up
            drow[off_disc:off_disc + 6] = kf

    def put(a, key, obj, cost):
        ids[a] = item_id(key)
        feats[a, off_item:off_item + 23] = _item_feats(g, obj, cost)

    if g.state == "SHOP":
        for i, it in enumerate(g.shop[:MAX_SHOP]):
            obj = it.joker if it.kind == "joker" else (Consumable(it.kind, it.key[2:]) if it.kind != "card" else it.card)
            put(A_BUY + i, it.key, obj, it.cost)
        for i, it in enumerate(g.shop_packs[:2]):
            put(A_BUY_PACK + i, it.key, None, it.cost)
        for i, it in enumerate(g.shop_vouchers[:MAX_VOUCHERS]):
            put(A_VOUCHER + i, it.key, None, it.cost)
        feats[A_REROLL, off_item] = (0 if g.free_rerolls else g.reroll_cost) / 10
    if g.state == "BLIND_SELECT":
        feats[A_REROLL_BOSS, off_item] = 1.0
    if g.state == "PACK":
        for i, x in enumerate(g.pack_cards[:MAX_PACK]):
            if isinstance(x, Joker):
                put(A_PICK + i, f"j_{x.key}", x, 0)
            elif isinstance(x, Consumable):
                put(A_PICK + i, x.key, x, 0)
            else:
                put(A_PICK + i, "<playing_card>", x, 0)
    for i, j in enumerate(g.jokers[:MAX_JOKERS]):
        put(A_SELL_J + i, "j_unknown" if j.hidden else f"j_{j.key}", j, -j.sell_value())
        if i + 1 < min(len(g.jokers), MAX_JOKERS):
            ids[A_SWAP + i] = item_id("j_unknown" if j.hidden else f"j_{j.key}")
    for i, c in enumerate(g.consumables[:MAX_CONSUMABLES]):
        put(A_SELL_C + i, c.key, c, -c.sell_value())
        put(A_USE_C + i, c.key, c, 0)
    return ids, feats


def _subs_array(plays, discs) -> np.ndarray:
    arr = np.full((2 * N_SUB, 5), -1, dtype=np.int64)
    for i, s in enumerate(plays):
        if s is not None:
            arr[i, :len(s)] = s
    for i, s in enumerate(discs):
        if s is not None:
            arr[N_SUB + i, :len(s)] = s
    return arr


def encode(g: Game, cnt: Optional[Counters] = None) -> dict:
    cnt = cnt or Counters()
    if g.targeting is not None:
        plays, discs, preds = target_candidates(g), [None] * N_SUB, {}
    elif g.state == "SELECTING_HAND":
        plays, discs, preds = candidates(g)
    else:
        plays, discs, preds = [None] * N_SUB, [None] * N_SUB, {}
    mask = legal_mask(g, cnt, plays, discs)
    aid, af = action_features(g, mask, plays, discs, preds)
    cards = np.zeros((MAX_HAND, F_CARD), dtype=np.float32)
    src = g.hand if g.state == "SELECTING_HAND" else (g.pack_hand if g.state == "PACK" else [])
    forced = g.forced_pos()
    for i, c in enumerate(src[:MAX_HAND]):
        cards[i] = encode_card(c, i == forced)
    jid = np.zeros(MAX_JOKERS, dtype=np.int64)
    jf = np.zeros((MAX_JOKERS, F_JOKER), dtype=np.float32)
    for i, j in enumerate(g.jokers[:MAX_JOKERS]):
        jid[i] = item_id("j_unknown" if j.hidden else f"j_{j.key}")
        jf[i] = encode_joker_feats(j)
    cid = np.zeros(MAX_CONSUMABLES, dtype=np.int64)
    for i, c in enumerate(g.consumables[:MAX_CONSUMABLES]):
        cid[i] = item_id(c.key)
    return {"g": encode_global(g, cnt), "cards": cards, "jid": jid, "jf": jf, "cid": cid,
            "aid": aid, "af": af, "mask": mask, "subs": _subs_array(plays, discs)}


def subset_of(obs: dict, a: int) -> list[int]:
    row = obs["subs"][a - A_PLAY] if a < A_DISC else obs["subs"][N_SUB + a - A_DISC]
    return [int(x) for x in row if x >= 0]


# ------------------------------------------------------------------ executing actions
def apply_action(g: Game, a: int, obs: dict):
    if g.targeting is not None:
        assert A_PLAY <= a < A_DISC, "a consumable is waiting for its target cards"
        g.apply_targets(subset_of(obs, a))
        return
    if A_PLAY <= a < A_DISC:
        g.play(subset_of(obs, a))
    elif A_DISC <= a < A_SELECT:
        g.discard(subset_of(obs, a))
    elif a == A_SELECT:
        g.select_blind()
    elif a == A_SKIP:
        g.skip_blind()
    elif a == A_REROLL_BOSS:
        g.reroll_boss()
    elif A_BUY <= a < A_BUY_PACK:
        g.buy_card(a - A_BUY)
    elif A_BUY_PACK <= a < A_VOUCHER:
        g.buy_pack(a - A_BUY_PACK)
    elif A_VOUCHER <= a < A_REROLL:
        g.buy_voucher(a - A_VOUCHER)
    elif a == A_REROLL:
        g.reroll()
    elif a == A_LEAVE:
        g.leave_shop()
    elif A_SELL_J <= a < A_SELL_C:
        g.sell_joker(a - A_SELL_J)
    elif A_SELL_C <= a < A_USE_C:
        g.sell_consumable(a - A_SELL_C)
    elif A_USE_C <= a < A_PICK:
        g.use_consumable(a - A_USE_C, choose_targets=True)
    elif A_PICK <= a < A_PSKIP:
        g.pack_pick(a - A_PICK, choose_targets=True)
    elif a == A_PSKIP:
        g.pack_skip()
    elif A_SWAP <= a < N_ACTIONS:
        g.swap_jokers(a - A_SWAP)


def describe_action(g: Game, a: int, obs: Optional[dict] = None) -> str:
    if g.targeting is not None and A_PLAY <= a < A_DISC:
        cards = g.target_cards()
        pos = subset_of(obs, a) if obs is not None else []
        return f"target {' '.join(repr(cards[i]) for i in pos)} with {g.targeting['cons'].key}"
    if a < A_SELECT:
        verb = "play" if a < A_DISC else "discard"
        pos = subset_of(obs, a) if obs is not None else list(SUBSETS[(a - A_PLAY) if a < A_DISC else (a - A_DISC)])
        return verb + " " + " ".join(repr(g.hand[i]) for i in pos)
    names = {A_SELECT: "select blind", A_SKIP: "skip blind", A_REROLL_BOSS: "reroll boss",
             A_REROLL: "reroll", A_LEAVE: "leave shop", A_PSKIP: "skip pack"}
    if a in names:
        return names[a]
    if a < A_BUY_PACK:
        return f"buy {g.shop[a - A_BUY].key}"
    if A_VOUCHER <= a < A_REROLL:
        return f"buy {g.shop_vouchers[a - A_VOUCHER].key}"
    if a < A_VOUCHER:
        return f"buy pack {g.shop_packs[a - A_BUY_PACK].key}"
    if A_SELL_J <= a < A_SELL_C:
        return f"sell j_{g.jokers[a - A_SELL_J].key}"
    if A_SELL_C <= a < A_USE_C:
        return f"sell {g.consumables[a - A_SELL_C].key}"
    if A_USE_C <= a < A_PICK:
        return f"use {g.consumables[a - A_USE_C].key}"
    if A_PICK <= a < A_PSKIP:
        x = g.pack_cards[a - A_PICK]
        key = getattr(x, "key", None)
        return f"pick {key if isinstance(key, str) else repr(x)}"    # playing cards: e.g. Qs[STEEL][RED]
    i = a - A_SWAP
    return f"swap j_{g.jokers[i].key} <-> j_{g.jokers[i + 1].key}"


def blind_weights(ante_weight: float = 1.0) -> list[float]:
    """Reward for beating each of the 24 blinds (index 0 = ante 1 small blind). Rises linearly
    from ante 1 to ante 8, where it is `ante_weight` times ante 1's, scaled so the 24 add up to 24."""
    raw = [1 + (ante_weight - 1) * (b // 3) / 7 for b in range(24)]
    scale = 24 / sum(raw)
    return [w * scale for w in raw]


def potential(g: Game, weights: list[float], chips: float = 0.0, build=None, build_w: float = 0.0) -> float:
    """Shaping potential Phi(s); the reward gets gamma * Phi(s') - Phi(s) (potential-based shaping,
    Ng et al. 1999), which adds dense signal without changing which policy is optimal.
    `chips`: progress through the current blind (score/target, capped at 1) x that blind's weight;
    only on blinds not beaten before, so replays after -1 Ante still earn nothing.
    `build`, `build_w`: build_w x rewards.Potential (headroom of the build against the next boss, plus
    blinds beaten / 24). Finished runs are 0."""
    if g.done:
        return 0.0
    phi = 0.0
    if chips and g.state == "SELECTING_HAND" and g.target > 0:
        b = 3 * (g.ante - 1) + g.blind_idx
        if b + 1 > g.furthest_blind:
            phi += chips * weights[min(b, 23)] * min(1.0, g.chips / g.target)
    if build is not None and build_w:
        phi += build_w * build(g)
    return phi


def build_potential(reward_config: Optional[str] = None):
    """rewards.Potential from a reward config file (defaults if None)."""
    from .rewards.config import RewardConfig
    from .rewards.potential import Potential
    return Potential(RewardConfig.load(reward_config).potential)


class BalatroEnv:
    """Gym-style environment. Reward: for each blind beaten further than ever before in the run,
    that blind's weight (1 each by default; see blind_weights). Replaying blinds after
    Hieroglyph/Petroglyph's -1 Ante earns nothing. Plus `win_bonus` (default 10) for winning the
    run, and up to half a blind's weight on a loss at a new blind, for how close it was.
    Optional potential-based shaping (see potential): `shape_chips`, `shape_phi` (weight of
    rewards.Potential, configured by the `reward_config` file), with `shape_gamma` set to the
    discount PPO uses."""

    def __init__(self, deck: str = "RED", stake: str = "GOLD", ante_weight: float = 1.0, win_bonus: float = 10.0,
                 win_ante: int = 8, joker_pool=None,
                 shape_chips: float = 0.0, shape_phi: float = 0.0, reward_config: Optional[str] = None,
                 shape_gamma: float = 0.995):
        self.deck = deck
        self.stake = stake
        self.game_kw = {"win_ante": win_ante, "joker_pool": joker_pool}
        self.weights = blind_weights(ante_weight)
        self.win_bonus = win_bonus
        self.shape_chips, self.shape_phi, self.shape_gamma = shape_chips, shape_phi, shape_gamma
        self.build = build_potential(reward_config) if shape_phi else None
        self.g: Optional[Game] = None

    def potential(self) -> float:
        return potential(self.g, self.weights, self.shape_chips, self.build, self.shape_phi)

    def reset(self, seed: Optional[int] = None) -> dict:
        self.g = Game(seed=seed, deck_type=self.deck, stake=self.stake, **self.game_kw)
        self.steps = 0
        self.cnt = Counters()
        self.obs = encode(self.g, self.cnt)
        self.phi = self.potential()
        return self.obs

    def step(self, a: int):
        g = self.g
        before = g.furthest_blind
        prev_state = g.state
        apply_action(g, a, self.obs)
        self.steps += 1
        if a == A_REROLL:
            self.cnt.rerolls += 1
        if A_SWAP <= a < N_ACTIONS:
            self.cnt.swaps += 1
        if g.state != prev_state:
            if g.state == "SHOP" and prev_state != "PACK":
                self.cnt.rerolls = 0
            if {prev_state, g.state} != {"SHOP", "PACK"}:
                self.cnt.swaps = 0
        r = float(sum(self.weights[min(b, 23)] for b in range(before, g.furthest_blind)))
        if g.state == "WON":
            r += self.win_bonus
        elif g.state == "GAME_OVER" and g.target > 0:
            lost = 3 * (g.ante - 1) + g.blind_idx         # index of the blind that was lost
            if lost + 1 > g.furthest_blind:
                r += 0.5 * self.weights[min(lost, 23)] * min(1.0, g.chips / g.target)
        done = g.done or self.steps >= MAX_STEPS
        obs = None if done else encode(g, self.cnt)
        if obs is not None and not obs["mask"].any():
            g.state = "GAME_OVER"
            done, obs = True, None
        if self.shape_chips or self.shape_phi:
            phi = 0.0 if done else self.potential()          # runs cut off at MAX_STEPS end at 0 too
            r += self.shape_gamma * phi - self.phi
            self.phi = phi
        self.obs = obs
        return obs, r, done, {"ante": g.ante, "won": g.state == "WON", "blinds": g.furthest_blind}
