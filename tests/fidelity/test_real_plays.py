"""The scorer against the real game: 316 hands from nine recorded real-game runs, each with the state before
the play and the chips Balatro then awarded.

Data: tests/data/real_plays.json, derived from the trajectories published in Attol8/balatro-ai (directory
evidence/, CC BY 4.0, copyright (C) 2026 Attol8; see the file's "attribution" and real_plays_build.py).

Every hand without a chance effect must come out exactly, from the Python scorer and the compiled one."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from balatro_rl.sim.cards import Card  # noqa: E402
from balatro_rl.sim.game import Game, Consumable  # noqa: E402
from balatro_rl.sim.jokers import Joker  # noqa: E402
from balatro_rl.sim.scoring import Plan, hook_variants, score_hand  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "real_plays.json")

# Hands our scorer does not reproduce, and why (run, segment:event).
ROUNDING = {                      # one chip off: the recording shows Ramen / Yorick / Constellation rounded
    "astra-black-gold-PI4T2AH8 01:754", "astra-black-gold-PI4T2AH8 01:758", "astra-black-gold-PI4T2AH8 01:762",
    "astra-low-QD3F4XVW 01:46",
}
OPEN = {                          # the real game gave exactly +20 Mult more, after the last joker and the
    "astra-low-TAF7DNTX 04:875",  # held planets. These are the only six hands with Observatory owned
                                  # (Ante 13 of one run, with and without planets held): not understood
    "astra-low-TAF7DNTX 04:901", "astra-low-TAF7DNTX 04:948", "astra-low-TAF7DNTX 04:952",
    "astra-low-TAF7DNTX 04:956", "astra-low-TAF7DNTX 04:960",
}


def _card(row):
    rank, suit, enh, edition, seal, extra, debuffed = row
    c = Card(rank, suit, enh=enh, edition=edition, seal=seal)
    c.extra_chips, c.debuffed = extra, bool(debuffed)
    return c


def build(p):
    """(game, played cards, held cards) for one recorded play. Face-down held cards are left out."""
    g = Game(seed=1, deck_type=p["deck"], stake=p["stake"])
    g.state = "SELECTING_HAND"
    g.blind_idx, g.boss, g.boss_disabled = (2 if p["boss"] else 0), p["boss"], bool(p["boss_disabled"])
    g.money, g.hands_left, g.discards_left = p["money"], p["hands_left"], p["discards_left"]
    g.joker_slots = p["joker_slots"]
    g.vouchers = set(p["vouchers"])
    g.consumables = [Consumable("planet", name) for name in p["planets"]]
    g.hand_levels, g.hand_played, g.hand_played_round = list(p["levels"]), list(p["played"]), list(p["played_round"])
    g.round_hand_types = {h for h, n in enumerate(p["played_round"]) if n > 0}
    g.mouth_hand = p["mouth"]
    g.tarots_used, g.blinds_skipped = p["tarots_used"], p["skipped"]
    n, steel, stone, enhanced = p["full"]
    g.full_deck = ([Card(9, 1, enh="STEEL") for _ in range(steel)] + [Card(2, 0, enh="STONE") for _ in range(stone)]
                   + [Card(9, 1, enh="BONUS") for _ in range(enhanced - steel - stone)]
                   + [Card(14, 0) for _ in range(n - enhanced)])
    g.deck = [Card(2, 0) for _ in range(p["deck_len"])]
    g.jokers = []
    for key, edition, debuffed, sell, state in p["jokers"]:
        j = Joker(key, edition=edition)
        j.debuffed, j.cost = bool(debuffed), 2 * sell          # sells for what the game showed
        j.state.update(state)
        g.jokers.append(j)
    hand = [None if row is None else _card(row) for row in p["hand"]]
    played = [hand[i] for i in p["play"]]
    held = [c for i, c in enumerate(hand) if i not in p["play"] and c is not None]
    g.hand = played + held
    return g, played, held


def chance(plan, ctx, held):
    """A Lucky card scores, or The Hook first discards held cards that matter: the chips depend on a roll."""
    return (any(ctx.played[i].enh == "LUCKY" and not ctx.played[i].debuffed for i in ctx.scoring)
            or hook_variants(plan, held) is not None)


def test_scorer_matches_the_recorded_real_hands():
    with open(DATA, encoding="utf-8") as f:
        doc = json.load(f)
    assert "CC BY 4.0" in doc["attribution"] and len(doc["plays"]) == 316
    exact = rolled = 0
    for p in doc["plays"]:
        g, played, held = build(p)
        plan = Plan(g)
        py, ctx = score_hand(g, played, held, plan=plan)
        if g.violates_boss(played):
            py = 0
        fast = g.predict_many([tuple(range(len(played)))], plan, played + held)[0][0]
        assert fast == py, p["src"]                             # Python and compiled scorers agree
        real, src = p["chips"], p["src"]
        if chance(plan, ctx, held):
            rolled += 1
        elif src in ROUNDING:
            assert abs(py - real) == 1, src
        elif src in OPEN:
            assert 0 < real - py < 0.01 * real, src
        else:
            assert py == real, (src, py, real)
            exact += 1
    assert (exact, rolled) == (295, 11)
