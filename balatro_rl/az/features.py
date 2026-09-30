"""C (inputs): the state as groups of tokens, and one feature row per candidate action.

Token groups (fixed maximum sizes, padded, with masks):
    global      1   scalars: phase, ante, blind, target, chips so far, hands / discards left, gold, boss,
                    tag, vouchers, slots, counters, deck and stake
    hand       16   cards in hand (face-down cards show only "hidden")
    deck       80   cards the player hasn't seen this round (draw pile + face-down cards in hand), or the
                    full deck outside a round; sorted by rank / suit, never in draw order
    pack_hand  16   cards dealt by an Arcana / Spectral pack for targeting
    jokers      8   id + edition, stickers, sell value, position and runtime state (value, rank, suit, hand)
    cons        6   consumables in slots
    levels     12   hand types: level, times played (run / round), base chips x mult at that level
    shop        7   shop cards 1-4, packs 1-2, voucher
    pack        5   the open pack's cards

A candidate row holds its kind, the exact score features (A), the solver's P(clear) / expected chips (B),
the immediate effects of consumable uses and picks, its side effects (money, levels, deck, creations),
a per-joker vector of runtime-state changes, which tokens it refers to (cards, joker, consumable, shop or
pack slot) and the prior logit (agent.py).
"""
from __future__ import annotations

import math

import numpy as np

from ..env import encode_card, encode_joker_feats, _item_feats, F_CARD, F_JOKER
from ..sim.game import Consumable, MAX_HAND, MAX_JOKERS
from ..sim.hands import N_HANDS, hand_base
from ..sim.items import BOSS_KEYS, TAGS, VOUCHERS, DECKS, STAKES, item_id, VOCAB
from ..sim.jokers import Joker
from .world import KINDS, KIND_INDEX, PHASES, World, MAX_REROLLS_PER_SHOP, MAX_MOVES_PER_PHASE

N_DECK = 80
N_CONS = 6
N_SHOP = 7
N_PACK = 5
GROUPS = [("glob", 1), ("hand", MAX_HAND), ("deck", N_DECK), ("phand", MAX_HAND), ("jok", MAX_JOKERS),
          ("cons", N_CONS), ("lev", N_HANDS), ("shop", N_SHOP), ("pack", N_PACK)]
OFFSET = {}
_o = 0
for _name, _n in GROUPS:
    OFFSET[_name] = _o
    _o += _n
N_TOKENS = _o
VOCAB_SIZE = len(VOCAB)
VOUCHER_KEYS = list(VOUCHERS)
PACK_KINDS = ["arcana", "celestial", "standard", "buffoon", "spectral"]
ITEM_KINDS = ["joker", "tarot", "planet", "spectral", "card", "pack", "voucher"]
EFFECT_KEYS = ["money", "levels", "levels_chance", "deck", "glass", "create", "jokers", "hand_size",
               "joker_slots", "ox", "destroy", "dna"]
N_REF = 6

F_HANDCARD = F_CARD + 1
F_JSTATE = 2 + 13 + 4 + N_HANDS + 1
F_JOK = F_JOKER + F_JSTATE + MAX_JOKERS
F_CONS = 1 + 3 + 1 + N_CONS
F_LEV = N_HANDS + 4
F_ITEM = 23 + len(ITEM_KINDS) + F_JOKER + F_JSTATE + F_CARD
F_CAND = (1 + N_HANDS + 7 + 3 + 4 + len(EFFECT_KEYS) + 3 + 2)


def _log(x, scale=6.0):
    return math.log10(max(x, 0) + 1) / scale


def _slog(x):
    return math.copysign(math.log1p(abs(x)), x) / 3.0


def global_feats(w: World) -> np.ndarray:
    g = w.g
    f = [g.ante / 8] + [float(g.ante == a) for a in range(1, 9)]
    f += [float(g.blind_idx == i) for i in range(3)]
    f += [float(g.state == s) for s in PHASES]
    in_round = g.state == "SELECTING_HAND"
    target = g.target if in_round else g.blind_target(min(g.blind_idx, 2))
    f += [_log(target), min(2.0, g.chips / max(target, 1)) if in_round else 0.0,
          _log(max(target - g.chips, 0)) if in_round else _log(target)]
    f += [g.hands_left / 5, g.discards_left / 5, g.round_hands() / 5, g.round_discards() / 5,
          g.effective_hand_size() / 8, len(g.hand) / 8]
    f += [min(g.money, 100) / 50, _log(max(g.money, 0), 3), float(g.money < 0), -g.debt_limit() / 20,
          min(max(g.money, 0), 5 * g.interest_cap()) / 25, g.interest_cap() / 20]
    f += [g.reroll_cost / 10, g.free_rerolls / 2, w.rerolls / MAX_REROLLS_PER_SHOP, w.moves / MAX_MOVES_PER_PHASE]
    boss = [0.0] * len(BOSS_KEYS)
    if g.boss in BOSS_KEYS:
        boss[BOSS_KEYS.index(g.boss)] = 1.0
    f += boss + [float(g.boss_active() != ""), float(g.boss_disabled), float(g.can_reroll_boss())]
    tag = [0.0] * len(TAGS)
    if g.state == "BLIND_SELECT" and g.blind_idx < 2 and g.tags_offered[g.blind_idx] in TAGS:
        tag[TAGS.index(g.tags_offered[g.blind_idx])] = 1.0
    f += tag
    pend = [0.0] * len(TAGS)
    for t in g.pending_tags:
        if t in TAGS:
            pend[TAGS.index(t)] += 1.0
    f += pend + [g.blinds_skipped / 5]
    f += [g.joker_slots / 7, len(g.jokers) / 7, g.consumable_slots / 3, g.cons_used() / 3]
    f += [float(v in g.vouchers) for v in VOUCHER_KEYS]
    f += [_log(g.blind_target(i)) for i in range(3)]
    f += [float(g.state == "PACK" and g.pack_kind == k) for k in PACK_KINDS]
    f.append(g.pack_picks / 2 if g.state == "PACK" else 0.0)
    unseen = len(g.deck) + sum(1 for c in g.hand if c.hidden) if in_round else len(g.full_deck)
    f += [unseen / 52, len(g.full_deck) / 52, g.tarots_used / 20, len(g.planets_used) / 12,
          g.furthest_blind / 24, g.total_hands_played / 100]
    f += [float(g.deck_type == d) for d in DECKS] + [float(g.stake == i) for i in range(len(STAKES))]
    f += [float(g.targeting is not None)]
    return np.asarray(f, dtype=np.float32)


def joker_state_feats(j: Joker) -> list[float]:
    f = [0.0] * F_JSTATE
    if j.hidden:
        return f
    st = j.state
    v = st.get("val")
    if isinstance(v, (int, float)):
        f[0] = _slog(float(v))
        f[1] = max(-2.0, min(2.0, float(v) / 50))
    r = st.get("rank")
    if isinstance(r, int) and 2 <= r <= 14:
        f[2 + r - 2] = 1.0
    s = st.get("suit")
    if isinstance(s, int) and 0 <= s < 4:
        f[15 + s] = 1.0
    h = st.get("hand")
    if isinstance(h, int) and 0 <= h < N_HANDS:
        f[19 + h] = 1.0
    left = st.get("left")
    if isinstance(left, (int, float)):
        f[19 + N_HANDS] = _slog(float(left))
    return f


def _joker_row(j: Joker, slot: int) -> list[float]:
    pos = [0.0] * MAX_JOKERS
    if slot >= 0:
        pos[slot] = 1.0
    return encode_joker_feats(j) + joker_state_feats(j) + pos


def _card_key(c):
    return (-(0 if c.is_stone else c.rank), c.suit, c.enh, c.edition, c.seal, c.extra_chips)


def _item_row(g, obj, key: str, kind: str, cost: int) -> list[float]:
    row = _item_feats(g, obj, cost) + [float(kind == k) for k in ITEM_KINDS]
    row += encode_joker_feats(obj) + joker_state_feats(obj) if isinstance(obj, Joker) else [0.0] * (F_JOKER + F_JSTATE)
    row += encode_card(obj) if (obj is not None and hasattr(obj, "rank")) else [0.0] * F_CARD
    return row


def encode_state(w: World) -> dict:
    g = w.g
    out = {}
    out["glob"] = global_feats(w)
    hand = np.zeros((MAX_HAND, F_HANDCARD), np.float32)
    forced = g.forced_pos()
    for i, c in enumerate(g.hand[:MAX_HAND]):
        hand[i, :F_CARD] = encode_card(c, i == forced)
        hand[i, F_CARD] = i / MAX_HAND
    out["hand"] = hand
    if g.state == "SELECTING_HAND":
        unseen = list(g.deck) + [c for c in g.hand if c.hidden]
    else:
        unseen = list(g.full_deck)
    unseen = sorted(unseen, key=_card_key)[:N_DECK]
    deck = np.zeros((N_DECK, F_HANDCARD), np.float32)
    for i, c in enumerate(unseen):
        deck[i, :F_CARD] = encode_card(_Visible(c))
    out["deck"] = deck
    ph = np.zeros((MAX_HAND, F_HANDCARD), np.float32)
    for i, c in enumerate(g.pack_hand[:MAX_HAND] if g.state == "PACK" else []):
        ph[i, :F_CARD] = encode_card(c)
        ph[i, F_CARD] = i / MAX_HAND
    out["phand"] = ph
    jok = np.zeros((MAX_JOKERS, F_JOK), np.float32)
    jid = np.zeros(MAX_JOKERS, np.int64)
    for i, j in enumerate(g.jokers[:MAX_JOKERS]):
        jok[i] = _joker_row(j, i)
        jid[i] = item_id("j_unknown" if j.hidden else f"j_{j.key}")
    out["jok"], out["jok_id"] = jok, jid
    cons = np.zeros((N_CONS, F_CONS), np.float32)
    cid = np.zeros(N_CONS, np.int64)
    for i, c in enumerate(g.consumables[:N_CONS]):
        cons[i, 0] = 1.0
        cons[i, 1 + ["tarot", "planet", "spectral"].index(c.kind)] = 1.0
        cons[i, 4] = float(c.negative)
        cons[i, 5 + i] = 1.0
        cid[i] = item_id(c.key)
    out["cons"], out["cons_id"] = cons, cid
    lev = np.zeros((N_HANDS, F_LEV), np.float32)
    for h in range(N_HANDS):
        lev[h, h] = 1.0
        ch, mu = hand_base(h, g.hand_levels[h])
        lev[h, N_HANDS:] = [g.hand_levels[h] / 10, min(g.hand_played[h], 60) / 30,
                            min(g.hand_played_round[h], 5) / 3, _log(ch * mu)]
    out["lev"] = lev
    shop = np.zeros((N_SHOP, F_ITEM), np.float32)
    sid = np.zeros(N_SHOP, np.int64)
    if g.state == "SHOP":
        for i, it in enumerate(g.shop[:4]):
            obj = it.joker if it.kind == "joker" else (it.card if it.kind == "card" else Consumable(it.kind, it.key[2:]))
            shop[i] = _item_row(g, obj, it.key, it.kind, it.cost)
            sid[i] = item_id(it.key)
        for i, it in enumerate(g.shop_packs[:2]):
            shop[4 + i] = _item_row(g, None, it.key, "pack", it.cost)
            sid[4 + i] = item_id(it.key)
        if g.shop_voucher is not None:
            shop[6] = _item_row(g, None, g.shop_voucher.key, "voucher", g.shop_voucher.cost)
            sid[6] = item_id(g.shop_voucher.key)
    out["shop"], out["shop_id"] = shop, sid
    pack = np.zeros((N_PACK, F_ITEM), np.float32)
    pid = np.zeros(N_PACK, np.int64)
    if g.state == "PACK":
        for i, x in enumerate(g.pack_cards[:N_PACK]):
            if isinstance(x, Joker):
                pack[i] = _item_row(g, x, f"j_{x.key}", "joker", 0)
                pid[i] = item_id(f"j_{x.key}")
            elif isinstance(x, Consumable):
                pack[i] = _item_row(g, x, x.key, x.kind, 0)
                pid[i] = item_id(x.key)
            else:
                pack[i] = _item_row(g, x, "<playing_card>", "card", 0)
                pid[i] = item_id("<playing_card>")
    out["pack"], out["pack_id"] = pack, pid
    out["mask"] = token_mask(g)
    return out


class _Visible:
    """A deck card shown face up (the unseen pool is a multiset the player knows)."""
    __slots__ = ("c",)

    def __init__(self, c):
        self.c = c

    def __getattr__(self, k):
        if k == "hidden":
            return False
        return getattr(self.c, k)


def token_mask(g) -> np.ndarray:
    m = np.zeros(N_TOKENS, dtype=bool)
    m[0] = True
    o = OFFSET
    m[o["hand"]:o["hand"] + min(len(g.hand), MAX_HAND)] = True
    n_unseen = (len(g.deck) + sum(1 for c in g.hand if c.hidden)) if g.state == "SELECTING_HAND" else len(g.full_deck)
    m[o["deck"]:o["deck"] + min(n_unseen, N_DECK)] = True
    if g.state == "PACK":
        m[o["phand"]:o["phand"] + min(len(g.pack_hand), MAX_HAND)] = True
        m[o["pack"]:o["pack"] + min(len(g.pack_cards), N_PACK)] = True
    m[o["jok"]:o["jok"] + min(len(g.jokers), MAX_JOKERS)] = True
    m[o["cons"]:o["cons"] + min(len(g.consumables), N_CONS)] = True
    m[o["lev"]:o["lev"] + N_HANDS] = True
    if g.state == "SHOP":
        m[o["shop"]:o["shop"] + min(len(g.shop), 4)] = True
        m[o["shop"] + 4:o["shop"] + 4 + min(len(g.shop_packs), 2)] = True
        m[o["shop"] + 6] = g.shop_voucher is not None
    return m


# ------------------------------------------------------------------ candidates
def _jd_vec(c) -> np.ndarray:
    v = np.zeros(MAX_JOKERS, np.float32)
    for slot, key, old, new in c.jdiff:
        if not 0 <= slot < MAX_JOKERS:
            continue
        if key == "val" and isinstance(old, (int, float)) and isinstance(new, (int, float)):
            v[slot] += _slog(float(new) - float(old))
        elif key == "_removed":
            v[slot] -= 2.0
        else:
            v[slot] += 0.5
    return v


def cand_row(g, c, need: float) -> list[float]:
    a = c.action
    f = [len(a.cards) / 5]
    ht = [0.0] * N_HANDS
    if a.kind == "play" and 0 <= c.hand < N_HANDS:
        ht[c.hand] = 1.0
    f += ht
    if a.kind == "play":
        f += [min(5.0, c.score / max(need, 1)), _log(c.score, 7), float(c.clears), float(c.certain),
              min(5.0, c.score_min / max(need, 1)), _log(c.chips, 5), _log(c.mult, 4)]
    else:
        f += [0.0] * 7
    ev = c.p_clear >= 0
    f += [c.p_clear if ev else 0.0, c.e_chips if ev else 0.0, float(ev)]
    has = c.best_after >= 0
    f += [c.best_before if has else 0.0, c.best_after if has else 0.0,
          (c.best_after - c.best_before) if has else 0.0, float(has)]
    eff = dict(c.effects)
    f += [_slog(float(eff.get(k, 0))) for k in EFFECT_KEYS]
    resets = sum(1 for _, k, old, new in c.jdiff if k == "val" and isinstance(old, (int, float))
                 and isinstance(new, (int, float)) and new < old)
    grows = sum(1 for _, k, old, new in c.jdiff if k == "val" and isinstance(old, (int, float))
                and isinstance(new, (int, float)) and new > old)
    f += [len(c.jdiff) / 4, resets / 2, grows / 2]
    f += [(a.idx / MAX_JOKERS) if a.kind == "move_joker" else 0.0, (a.to / MAX_JOKERS) if a.to >= 0 else 0.0]
    return f


def cand_refs(g, c) -> list[int]:
    a = c.action
    o = OFFSET
    refs = []
    if a.kind in ("play", "discard", "use"):
        refs = [o["hand"] + p for p in a.cards[:5]]
    elif a.kind == "pick":
        refs = [o["phand"] + p for p in a.cards[:5]]
    if a.kind in ("sell_joker", "move_joker"):
        refs.append(o["jok"] + a.idx)
    elif a.kind in ("use", "sell_cons"):
        refs.append(o["cons"] + a.idx)
    elif a.kind == "buy":
        refs.append(o["shop"] + a.idx)
    elif a.kind == "buy_pack":
        refs.append(o["shop"] + 4 + a.idx)
    elif a.kind == "voucher":
        refs.append(o["shop"] + 6)
    elif a.kind == "pick":
        refs.append(o["pack"] + a.idx)
    refs = [r for r in refs if r < N_TOKENS][:N_REF]
    return refs + [-1] * (N_REF - len(refs))


def encode_cands(w: World, choice, priors) -> dict:
    g = w.g
    n = len(choice.cands)
    kind = np.zeros(n, np.int64)
    feats = np.zeros((n, F_CAND), np.float32)
    refs = np.full((n, N_REF), -1, np.int64)
    jd = np.zeros((n, MAX_JOKERS), np.float32)
    for i, c in enumerate(choice.cands):
        kind[i] = KIND_INDEX[c.kind]
        feats[i] = cand_row(g, c, choice.need)
        refs[i] = cand_refs(g, c)
        jd[i] = _jd_vec(c)
    return {"c_kind": kind, "c_f": feats, "c_ref": refs, "c_jd": jd, "c_prior": np.asarray(priors, np.float32)}


def _global_size() -> int:
    from ..sim.game import Game
    return len(global_feats(World(Game(seed=0))))


F_GLOBAL = _global_size()
