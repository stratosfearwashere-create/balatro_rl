"""Compact build records: what the build-value model sees, and the outcome labels a finished game gives them.

A record is taken at every decision outside a round whose build differs from the last record's (a buy, sell,
use, pick, skip ...) and at the start of every blind. It holds
    glob      features.global_feats (ante, blind, money, boss, tags, vouchers, slots, stake, deck ...)
    deck      a summary of the full deck: counts by rank, suit and rank x suit, enhancements, editions, seals,
              extra chips, size (the per-card deck tokens of the search agent are far too big to store)
    jok       8 joker rows (features._joker_row) and their item ids
    cons      consumable rows and ids
    lev       12 hand-level rows
    phi       the potential (the hand-built build strength), so the model can learn a residual on it
and, once the game is over (GameRecorder.finish),
    win       1.0 if the run was won
    ante_reached   1..8, or 9 for a win  (the ordinal target)
    boss_now  this ante's boss was beaten (NaN when the game ended before it was faced is 0: it was lost)
    boss_next the next ante's boss was beaten
    blinds_after   blinds beaten after this record / 24
    money_next     money at the start of the next ante / 100 (NaN if there was none)
Everything is float16 / int16 and about 2 KB per record.
"""
from __future__ import annotations

import math

import numpy as np

from ..az.features import F_JOK, F_CONS, F_LEV, N_CONS, _joker_row, global_feats
from ..sim.cards import ENHANCEMENTS, EDITIONS, SEALS
from ..sim.game import MAX_JOKERS
from ..sim.hands import N_HANDS, hand_base
from ..sim.items import item_id

F_DECK = 13 + 4 + 52 + len(ENHANCEMENTS) + len(EDITIONS) + len(SEALS) + 4
ARRAY_KEYS = ("glob", "deck", "jok", "jok_id", "cons", "cons_id", "lev")
SCALAR_KEYS = ("phi", "ante", "blind_idx", "stake", "money", "step", "game",
               "win", "ante_reached", "boss_now", "boss_next", "blinds_after", "money_next")
LABEL_KEYS = ("win", "ante_reached", "boss_now", "boss_next", "blinds_after", "money_next")
N_ORDINAL = 8                      # P(reach ante k) for k = 2..9 (9: the run is won)


def deck_feats(g) -> np.ndarray:
    f = np.zeros(F_DECK, np.float32)
    n = 0
    extra = 0.0
    for c in g.full_deck:
        n += 1
        if not c.is_stone:
            f[c.rank - 2] += 1
            f[13 + c.suit] += 1
            f[17 + 13 * c.suit + (c.rank - 2)] += 1
        o = 69
        f[o + ENHANCEMENTS.index(c.enh) if c.enh in ENHANCEMENTS else o] += 1
        o += len(ENHANCEMENTS)
        f[o + (EDITIONS.index(c.edition) if c.edition in EDITIONS else 0)] += 1
        o += len(EDITIONS)
        f[o + (SEALS.index(c.seal) if c.seal in SEALS else 0)] += 1
        extra += c.extra_chips
    f[:13] /= 4
    f[13:17] /= 13
    f[17:69] /= 2
    f[69:F_DECK - 4] /= 8
    o = F_DECK - 4
    f[o] = n / 52
    f[o + 1] = extra / 100
    f[o + 2] = max((f[13 + s] for s in range(4)), default=0.0)
    f[o + 3] = g.starting_deck_size / 52
    return f


def build_record(w, potential) -> dict:
    g = w.g
    r = {"glob": global_feats(w), "deck": deck_feats(g)}
    jok = np.zeros((MAX_JOKERS, F_JOK), np.float32)
    jid = np.zeros(MAX_JOKERS, np.int16)
    for i, j in enumerate(g.jokers[:MAX_JOKERS]):
        jok[i] = _joker_row(j, i)
        jid[i] = item_id("j_unknown" if j.hidden else f"j_{j.key}")
    r["jok"], r["jok_id"] = jok, jid
    cons = np.zeros((N_CONS, F_CONS), np.float32)
    cid = np.zeros(N_CONS, np.int16)
    for i, c in enumerate(g.consumables[:N_CONS]):
        cons[i, 0] = 1.0
        cons[i, 1 + ["tarot", "planet", "spectral"].index(c.kind)] = 1.0
        cons[i, 4] = float(c.negative)
        cons[i, 5 + i] = 1.0
        cid[i] = item_id(c.key)
    r["cons"], r["cons_id"] = cons, cid
    lev = np.zeros((N_HANDS, F_LEV), np.float32)
    for h in range(N_HANDS):
        lev[h, h] = 1.0
        ch, mu = hand_base(h, g.hand_levels[h])
        lev[h, N_HANDS:] = [g.hand_levels[h] / 10, min(g.hand_played[h], 60) / 30,
                            min(g.hand_played_round[h], 5) / 3, math.log10(max(ch * mu, 1) + 1) / 6]
    r["lev"] = lev
    r["phi"] = float(potential(g))
    r["ante"], r["blind_idx"], r["stake"], r["money"], r["step"] = g.ante, min(g.blind_idx, 2), g.stake, g.money, w.steps
    return r


def build_signature(g) -> tuple:
    """What a record is about: when this is unchanged, a new decision adds nothing."""
    return (g.ante, min(g.blind_idx, 2), g.money, tuple((j.key, j.edition, tuple(sorted(j.state.items())))
            for j in g.jokers), tuple(g.hand_levels), tuple(sorted(c.key for c in g.consumables)),
            tuple(sorted(g.vouchers)), len(g.full_deck),
            tuple(sorted((c.rank, c.suit, c.enh, c.edition, c.seal, c.extra_chips) for c in g.full_deck)))


class GameRecorder:
    """Collects build records during one game and labels them at the end."""

    def __init__(self, potential, game_id: int):
        self.potential = potential
        self.game_id = game_id
        self.rows: list[dict] = []
        self.last_sig = None
        self.ante_money: dict[int, int] = {}       # money at the start of each ante
        self.boss_cleared: dict[int, bool] = {}    # ante -> its boss blind was beaten

    def maybe(self, w, blind_start: bool = False):
        g = w.g
        if g.state == "SELECTING_HAND" and not blind_start:
            return
        if blind_start and g.blind_idx == 0 and g.ante not in self.ante_money:
            self.ante_money[g.ante] = g.money
        sig = build_signature(g)
        if sig == self.last_sig and not blind_start:
            return
        self.last_sig = sig
        self.rows.append(build_record(w, self.potential))

    def boss(self, ante: int, cleared: bool):
        self.boss_cleared[ante] = cleared

    def finish(self, g) -> list[dict]:
        won = g.state == "WON"
        reached = 9 if won else min(max(g.ante, 1), 8)
        for r in self.rows:
            a = int(r["ante"])
            r["game"] = self.game_id
            r["win"] = float(won)
            r["ante_reached"] = reached
            r["boss_now"] = float(self.boss_cleared.get(a, False))
            r["boss_next"] = float(self.boss_cleared.get(a + 1, False)) if a < 8 else float(won)
            r["blinds_after"] = max(0, g.furthest_blind - (3 * (a - 1) + int(r["blind_idx"]))) / 24
            nxt = self.ante_money.get(a + 1)
            r["money_next"] = math.nan if nxt is None else nxt / 100
        return self.rows


def pack(rows: list[dict]) -> dict:
    """Rows -> one dict of stacked arrays (float16 for features)."""
    out = {}
    for k in ARRAY_KEYS:
        arr = np.stack([r[k] for r in rows])
        out[k] = arr.astype(np.float16) if arr.dtype == np.float32 else arr
    for k in SCALAR_KEYS:
        out[k] = np.asarray([r[k] for r in rows], dtype=np.float32 if k in LABEL_KEYS or k == "phi" else np.int32)
    return out


def concat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
