"""Future value of every scaling joker: what each joker's counter will be after R more rounds, from the state, the economy projection
(balatro_rl/economy.py) and the shop's measured offer rates (tools/shop_rates.py -> balatro_rl/data/shop_rates.json).

Two kinds of growth:

  investment   the joker grows with things bought in the shop. Units per round = what the round's budget buys, cheapest source first, capped
               by what the shop offers: e.g. Constellation
                   constellation_future = constellation_now + rounds x planets_per_round x 0.1
                   planets_per_round    = budget buys: shop planet cards (0.29 a visit, $3), Celestial packs ($4 / $6 for 1, $8 for 2), then
                                          rerolls ($5, $6, ... a view, 0.29 planets each, + $3)
               Hologram (cards added: Standard packs), Fortune Teller (tarots), Flash Card (rerolls), Red Card (packs skipped), Campfire (cards
               sold, within an ante), Ceremonial Dagger (cheap jokers to eat), Lucky Cat / Glass Joker / Vampire / Steel / Stone (the tarots
               that make their cards: Magician, Justice, any enhancing tarot, Chariot, Tower), Erosion and Canio (Hanged Man).
  play         the joker grows with the cards played or discarded. Per round: plays and discards from the economy projection, the main hand
               and its reach from the run eval, the deck's make-up. E.g. Lucky Cat
                   lucky_scored_per_round = plays x cards scored per play x (lucky cards in the deck / deck size)
                   triggers               = lucky_scored x P(a Lucky card triggers) = 1 - (4/5)(14/15) = 0.253 (Oops: 1 - (3/5)(13/15) = 0.48)
               with the lucky cards in the deck rising by 2 per Magician the budget buys.
               Runner, Square, Spare Trousers, Ride the Bus, Green Joker, Wee Joker, Castle, Obelisk, Hit the Road, Yorick, Supernova, Hiker,
               Madness, Throwback, Abstract, Swashbuckler, Bull, Bootstraps; and the decaying ones (Ice Cream, Popcorn, Ramen, Seltzer, Turtle
               Bean, Gros Michel and Cavendish's survival).

    from balatro_rl.growth import project
    rows = project(g, jid, jf, cid, rounds=None)       # one list per state of dicts: key, unit, now, future, per_round, how
    python -m balatro_rl.growth --check 64             # play-driven jokers against the engine
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import numpy as np

from .economy import future_economy
from .runeval import RunEval, State, default_weights, Cols, E, V, N_CARDS, FACE_MASK, JF_SELL, IS_SET
from .sim.hands import HAND_NAMES, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH, FIVE_KIND, FLUSH_HOUSE, FLUSH_FIVE
from .sim.items import VOCAB, PACKS

RATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shop_rates.json")
P_LUCKY = 1.0 - (4.0 / 5.0) * (14.0 / 15.0)
P_LUCKY_OOPS = 1.0 - (3.0 / 5.0) * (13.0 / 15.0)
ENHANCING = ("magician", "empress", "heirophant", "lovers", "chariot", "justice", "devil", "tower")    # tarots that enhance cards (1-2 each)
ENH_PER_USE = {"magician": 2, "empress": 2, "heirophant": 2, "lovers": 1, "chariot": 1, "justice": 1, "devil": 1, "tower": 1}
CONTAINS_TWO_PAIR = {TWO_PAIR, FULL_HOUSE, FLUSH_HOUSE}
FOUR_CARD = {TWO_PAIR, QUADS}
STRAIGHTS = {STRAIGHT, STRAIGHT_FLUSH}
JACK, TWO = 9, 0                                                   # rank indices (2 .. A)


@lru_cache(maxsize=None)
def rates():
    d = json.load(open(RATES_PATH))
    return {int(a): v for a, v in d["antes"].items()}


def offers(ante):
    r = rates()
    return r[min(max(int(ante), 1), max(r))]


class Budget:
    """What a round's money buys for one resource: sources in order of cost per unit, each capped by what the shop offers per visit."""

    def __init__(self, ante, money_per_round, discount=0.0, reroll_base=5.0):
        self.o = offers(ante)
        self.b = max(0.0, float(money_per_round))
        self.disc = 1.0 - discount
        self.reroll_base = reroll_base

    def _packs(self, kind):
        out = []
        for (k, size), (choices, picks, cost, _w) in PACKS.items():
            if k == kind:
                out.append((self.o["packs"].get(f"{k}_{size}", 0.0), cost * self.disc, picks, choices))
        return out

    def buy(self, kind, name=None, unit_price=3.0, max_views=6):
        """-> expected units per round of `kind` (planet / tarot / tarot:name / card_added / pack / reroll / joker / sale) for the budget."""
        b = self.b
        units = 0.0
        srcs = []                                                  # (available per round, cost per purchase, units per purchase)
        if kind in ("planet", "tarot"):
            srcs.append((self.o["open"].get(kind, 0.0), unit_price * self.disc, 1.0))
            for avail, cost, picks, _c in self._packs("celestial" if kind == "planet" else "arcana"):
                srcs.append((avail, cost, picks))
        elif kind == "tarot_named":                                # one tarot by name (Magician, Justice, Chariot, Tower, Hanged Man ...)
            srcs.append((self.o["open"].get(f"tarot:{name}", 0.0), unit_price * self.disc, 1.0))
            for avail, cost, picks, choices in self._packs("arcana"):
                srcs.append((avail * (1.0 - (1.0 - 1.0 / 22.0) ** choices), cost, 1.0))
        elif kind == "card_added":
            for avail, cost, picks, _c in self._packs("standard"):
                srcs.append((avail, cost, picks))
        elif kind == "pack":                                       # any pack, bought to be skipped (Red Card)
            for k in ("arcana", "celestial", "standard", "buffoon", "spectral"):
                for avail, cost, picks, _c in self._packs(k):
                    srcs.append((avail, cost, 1.0))
        elif kind == "joker":                                      # the cheapest jokers (Ceremonial Dagger's food), about $5
            srcs.append((self.o["open"].get("joker", 0.0), 5.0 * self.disc, 1.0))
        elif kind == "sale":                                       # a planet or tarot bought for $3 and sold for $1 (Campfire)
            srcs.append((self.o["open"].get("planet", 0.0) + self.o["open"].get("tarot", 0.0), 2.0 * self.disc, 1.0))
        srcs.sort(key=lambda s: s[1] / max(s[2], 1e-9))
        for avail, cost, per in srcs:
            if b <= 0 or avail <= 0:
                continue
            k = min(avail, b / cost)
            units += k * per
            b -= k * cost
        # rerolls: each view shows the card slots again (0.29 planets / tarots, 1.4 jokers); the reroll cost rises by $1 a view
        if kind in ("planet", "tarot", "tarot_named", "joker", "sale", "reroll"):
            key = {"planet": "planet", "tarot": "tarot", "tarot_named": f"tarot:{name}", "joker": "joker", "sale": "planet", "reroll": None}[kind]
            per_view = self.o["reroll"].get(key, 0.0) if key else 1.0
            price = {"planet": unit_price, "tarot": unit_price, "tarot_named": unit_price, "joker": 5.0, "sale": 2.0, "reroll": 0.0}[kind] * self.disc
            for v in range(max_views):
                c = self.reroll_base + v
                cost_view = c + per_view * price
                if b <= 0 or per_view <= 0:
                    break
                f = min(1.0, b / cost_view)
                units += f * per_view
                b -= f * cost_view
        self.b = b
        return units


def _per_round(arr, R):
    """Mean of a per-round projection over the first R rounds the build survives (or its first round)."""
    arr = np.asarray(arr, dtype=np.float64)
    return float(arr[:max(1, R)].mean()) if len(arr) else 0.0


def project(g, jid, jf, cid=None, rounds=None, invest=1.0, econ=None):
    """For each state, each joker that changes over time: its counter now and after `rounds` rounds (default: the rounds left to the ante-8 boss).
    invest: the share of each round's projected income spent on that joker's feeder (each joker is projected as if it had the budget alone)."""
    s = State(g, jid, jf, cid)
    T, _s, dg = RunEval(default_weights()).terms(g, jid, jf, cid)
    if econ is None:
        econ = future_economy(g, jid, jf, cid, rounds=9)
    out = []
    for i in range(s.n):
        R = int(s.rounds_left[i]) if rounds is None else int(rounds)
        alive = econ["alive"][i]
        n_alive = int(alive.sum()) or 1
        plays = _per_round(econ["plays"][i][alive] if alive.any() else econ["plays"][i], n_alive)
        disc = max(_per_round(econ["discards"][i][alive] if alive.any() else econ["discards"][i], n_alive), 0.22 * s.discards[i])   # (the round player uses ~0.66 of 3)
        if alive.any():
            income = float(np.nanmean(econ["income"][i][alive]))
        else:                                                         # projected to fall at the next blind: the income of a scraped win
            income = 3.0 + 1.0 + min(math.floor(max(float(s.money[i]), 0.0) / 5.0), 5.0)
        budget = invest * max(income, 0.0) + max(0.0, s.money[i] - 25.0) / max(R, 1)
        main = int(dg["main"][i])
        reach = dg["reach"][i]
        ph = dg["p_hand"][i]                                          # P(a play is each hand type): the strongest hand that lands
        p_of = lambda hands: float(sum(ph[h] for h in hands))
        nc_mean = float((ph * N_CARDS).sum())
        nc = float(N_CARDS[main])
        deck = max(float(s.full[i]), 1.0)
        face = float((s.rank_share[i] * FACE_MASK).sum())
        ante = int(s.ante[i])
        disc_v = 0.5 if s.vouchers[i, V["liquidation"]] else (0.25 if s.vouchers[i, V["clearance_sale"]] else 0.0)
        rr = 5.0 - 2.0 * s.vouchers[i, V["reroll_surplus"]] - 2.0 * s.vouchers[i, V["reroll_glut"]]
        has = lambda k: bool(((s.jid[i] == VOCAB.index(f"j_{k}")) & s.live[i]).any())
        oops = has("oops")
        cards_scored = plays * nc_mean
        disc_cards = disc * 5.0                                       # cards per discard: the round player throws away the cards outside its play
        rows = []
        for slot in range(s.jid.shape[1]):
            if s.jid[i, slot] <= 0 or not s.live[i, slot]:
                continue
            key = VOCAB[int(s.jid[i, slot])][2:]
            now = float(s.counter[i, slot])
            bud = lambda: Budget(ante, budget, disc_v, rr)
            row = None
            # ---------------- investment
            if key == "constellation":
                per = bud().buy("planet") * 0.1
                row = ("xmult", max(now, 1.0), per, "planets bought per round x 0.1")
            elif key == "fortune_teller":
                t = bud().buy("tarot") + (1.0 if has("cartomancer") else 0.0) + (0.5 if has("vagabond") else 0.0)
                row = ("mult", float(s.tarots_used[i]), t, "tarots used per round x 1")
            elif key == "hologram":
                per = bud().buy("card_added") * 0.25
                row = ("xmult", max(now, 1.0), per, "cards added per round (Standard packs) x 0.25")
            elif key == "flash":
                per = bud().buy("reroll") * 2.0
                row = ("mult", now, per, "rerolls per round x 2")
            elif key == "red_card":
                per = bud().buy("pack") * 3.0
                row = ("mult", now, per, "packs skipped per round x 3")
            elif key == "ceremonial":
                per = min(bud().buy("joker"), 1.0) * 2.0 * 2.0
                row = ("mult", now, per, "a cheap joker eaten per round x 2 x its $2 sell value")
            elif key == "campfire":                                    # resets after each boss: its value in a round is the ante's sales so far
                sales = bud().buy("sale")
                row = ("xmult", 1.0, 0.0, "resets each boss"), sales
                fut = 1.0 + 0.25 * sales * 2.0                           # by the boss: two rounds of sales
                rows.append(dict(key=key, unit="xmult", now=max(now, 1.0), future=fut, per_round=0.25 * sales, how="cards sold per round x 0.25, reset each boss"))
                continue
            elif key == "lucky_cat":
                lucky0 = float(s.enh[i, E["LUCKY"]])
                conv = bud().buy("tarot_named", "magician") * 2.0         # lucky cards made per round
                p = P_LUCKY_OOPS if oops else P_LUCKY
                trig = 0.0
                L = lucky0
                for r in range(R):
                    trig += cards_scored * min(L / deck, 1.0) * p
                    L += conv
                rows.append(dict(key=key, unit="xmult", now=max(now, 1.0), future=max(now, 1.0) + 0.25 * trig, per_round=0.25 * trig / max(R, 1),
                                 how=f"lucky cards {lucky0:.0f} +{conv:.2f}/round (Magician); scored {cards_scored:.1f} cards/round x share x P(trigger) {p:.3f}"))
                continue
            elif key == "glass":
                g0 = float(s.enh[i, E["GLASS"]])
                conv = bud().buy("tarot_named", "justice")
                broke = 0.0
                G = g0
                for r in range(R):
                    b = cards_scored * min(G / deck, 1.0) * 0.25
                    broke += b
                    G = max(G - b, 0.0) + conv
                rows.append(dict(key=key, unit="xmult", now=max(now, 1.0), future=max(now, 1.0) + 0.75 * broke, per_round=0.75 * broke / max(R, 1),
                                 how=f"glass cards {g0:.0f} +{conv:.2f}/round (Justice), 1 in 4 breaks when scored"))
                continue
            elif key == "vampire":
                e0 = float(s.enh[i].sum())
                b_ = bud()
                conv = sum(b_.buy("tarot_named", t) * ENH_PER_USE[t] for t in ENHANCING)
                eaten = 0.0
                En = e0
                for r in range(R):
                    k = cards_scored * min(En / deck, 1.0)
                    eaten += k
                    En = max(En - k, 0.0) + conv
                rows.append(dict(key=key, unit="xmult", now=max(now, 1.0), future=max(now, 1.0) + 0.1 * eaten, per_round=0.1 * eaten / max(R, 1),
                                 how=f"enhanced cards {e0:.0f} +{conv:.2f}/round (enhancing tarots), each scored one eaten"))
                continue
            elif key in ("steel_joker", "stone"):
                enh, tarot, per_card, unit = ("STEEL", "chariot", 0.2, "xmult") if key == "steel_joker" else ("STONE", "tower", 25.0, "chips")
                c0 = float(s.enh[i, E[enh]])
                conv = bud().buy("tarot_named", tarot)
                base = 1.0 if unit == "xmult" else 0.0
                rows.append(dict(key=key, unit=unit, now=base + per_card * c0, future=base + per_card * (c0 + conv * R), per_round=per_card * conv,
                                 how=f"{enh.lower()} cards {c0:.0f} +{conv:.2f}/round ({tarot.title()}) x {per_card}"))
                continue
            elif key == "erosion":
                miss0 = max(52.0 - deck, 0.0)
                thin = bud().buy("tarot_named", "hanged_man") * 2.0 + (1.0 if has("trading") else 0.0)
                rows.append(dict(key=key, unit="mult", now=4.0 * miss0, future=4.0 * (miss0 + thin * R), per_round=4.0 * thin,
                                 how=f"cards below 52: {miss0:.0f} +{thin:.2f}/round (Hanged Man, Trading Card) x 4"))
                continue
            elif key == "caino":
                d_ = bud().buy("tarot_named", "hanged_man") * 2.0 * face
                row = ("xmult", max(now, 1.0), d_ * 1.0, "face cards destroyed per round (Hanged Man) x 1")
            # ---------------- play
            elif key == "runner":
                p = p_of(STRAIGHTS)
                row = ("chips", now, plays * p * 15.0, f"straights played per round ({plays:.1f} plays x P(a straight) {p:.2f}) x 15")
            elif key == "square":
                p = p_of(FOUR_CARD)
                row = ("chips", now, plays * p * 4.0, f"4-card hands per round ({plays:.1f} plays x {p:.2f}) x 4")
            elif key == "trousers":
                p = p_of(CONTAINS_TWO_PAIR)
                row = ("mult", now, plays * p * 2.0, f"hands containing Two Pair per round ({plays:.1f} plays x {p:.2f}) x 2")
            elif key == "ride_the_bus":                                # +1 a face-free play, back to 0 when a face card scores
                q = 1.0 - (1.0 - face) ** max(nc_mean - 1.0, 1.0)
                k = plays * R
                fut = now * (1.0 - q) ** k + (1.0 - q) * (1.0 - (1.0 - q) ** k) / max(q, 1e-3)
                rows.append(dict(key=key, unit="mult", now=now, future=fut, per_round=(fut - now) / max(R, 1),
                                 how=f"P(a face card scores) {q:.2f} a play over {k:.0f} plays: expected level {fut:.1f}"))
                continue
            elif key == "green_joker":
                row = ("mult", now, plays - disc, f"{plays:.1f} plays - {disc:.1f} discards per round")
            elif key == "wee":
                twos = float(s.rank_share[i, TWO])
                row = ("chips", now, cards_scored * twos * 8.0 * (2.0 if has("hack") else 1.0), "2s scored per round x 8")
            elif key == "castle":
                row = ("chips", now, disc_cards * 0.25 * 3.0, "cards of the round's suit discarded per round x 3")
            elif key == "obelisk":                                     # resets when the most played hand is played again
                rows.append(dict(key=key, unit="xmult", now=max(now, 1.0), future=1.0 + 0.2 * max(0.0, plays - 1.0) * 0.2, per_round=0.0,
                                 how="resets on the most played hand: the main hand keeps it near 1"))
                continue
            elif key == "hit_the_road":
                jacks = disc_cards * float(s.rank_share[i, JACK])
                rows.append(dict(key=key, unit="xmult", now=max(now, 1.0), future=1.0 + 0.5 * jacks, per_round=0.0, how="Jacks discarded in the round x 0.5, reset each round"))
                continue
            elif key == "yorick":
                row = ("xmult", max(now, 1.0), disc_cards / 23.0, "cards discarded per round / 23")
            elif key == "supernova":
                rows.append(dict(key=key, unit="mult", now=float(s.played[i, main]), future=float(s.played[i, main]) + plays * ph[main] * R, per_round=plays * ph[main],
                                 how=f"times {HAND_NAMES[main]} has been played, +{plays:.1f} per round"))
                continue
            elif key == "hiker":
                per_card = 5.0 * cards_scored / deck
                rows.append(dict(key=key, unit="chips", now=0.0, future=nc * per_card * R, per_round=nc * per_card,
                                 how=f"+5 chips on each card scored: {cards_scored:.1f} scored per round over {deck:.0f} cards"))
                continue
            elif key == "madness":
                row = ("xmult", max(now, 1.0), 0.5 * 2.0 / 3.0, "+0.5 per small / big blind (destroys a joker each)")
            elif key == "throwback":
                done_bl = max(1.0, 3.0 * (ante - 1) + float(s.blind[i]))
                rate = float(s.skipped[i]) / done_bl
                rows.append(dict(key=key, unit="xmult", now=1.0 + 0.25 * float(s.skipped[i]), future=1.0 + 0.25 * (float(s.skipped[i]) + rate * R),
                                 per_round=0.25 * rate, how=f"skips so far at {rate:.2f} a blind x 0.25"))
                continue
            elif key == "abstract":
                rows.append(dict(key=key, unit="mult", now=3.0 * float(s.njokers[i]), future=3.0 * float(max(s.jslots[i], s.njokers[i])), per_round=0.0,
                                 how="3 per joker, the slots filled"))
                continue
            elif key in ("bull", "bootstraps"):
                m_end = econ["money"][i]
                m_end = float(m_end[np.isfinite(m_end)][-1]) if np.isfinite(m_end).any() else float(s.money[i])
                f = (lambda m: 2.0 * max(m, 0.0)) if key == "bull" else (lambda m: 2.0 * math.floor(max(m, 0.0) / 5.0))
                rows.append(dict(key=key, unit="chips" if key == "bull" else "mult", now=f(float(s.money[i])), future=f(m_end), per_round=0.0,
                                 how="the money held: the economy projection, nothing spent"))
                continue
            elif key == "swashbuckler":
                sell = float((s.jf[i, :, JF_SELL] * 10.0 * s.live[i]).sum()) - float(s.jf[i, slot, JF_SELL] * 10.0)
                grow = (1.0 if has("gift") else 0.0) * max(float(s.njokers[i]) - 1.0, 0.0) + (3.0 if has("egg") else 0.0)
                rows.append(dict(key=key, unit="mult", now=sell, future=sell + grow * R, per_round=grow, how="the other jokers' sell value (Gift Card, Egg)"))
                continue
            # ---------------- decay and risk
            elif key == "ice_cream":
                row = ("chips", now if now > 0 else 100.0, -5.0 * plays, "-5 per hand played")
            elif key == "popcorn":
                row = ("mult", now if now > 0 else 20.0, -4.0, "-4 per round")
            elif key == "ramen":
                row = ("xmult", now if now > 0 else 2.0, -0.01 * disc_cards, "-0.01 per card discarded")
            elif key == "selzer":
                row = ("count", now if now > 0 else 10.0, -plays, "-1 per hand played (retriggers until 0)")
            elif key == "turtle_bean":
                row = ("hand_size", now if now > 0 else 5.0, -1.0, "-1 hand size per round")
            elif key in ("gros_michel", "cavendish"):
                p_die = 1.0 / 6.0 if key == "gros_michel" else 1.0 / 1000.0
                surv = (1.0 - p_die) ** R
                val = 15.0 if key == "gros_michel" else 3.0
                rows.append(dict(key=key, unit="mult" if key == "gros_michel" else "xmult", now=val, future=val * surv + (0.0 if key == "gros_michel" else 1.0 - surv),
                                 per_round=0.0, how=f"survives {R} rounds with chance {surv:.2f}"))
                continue
            if row is None:
                continue
            unit, base, per, how = row
            if unit in ("xmult",):
                lo = 1.05 if key == "ramen" else 1.0
                fut = max(lo, base + per * R)
            elif unit in ("chips", "mult", "count", "hand_size"):
                fut = max(0.0, base + per * R)
            else:
                fut = base + per * R
            rows.append(dict(key=key, unit=unit, now=base, future=fut, per_round=per, how=how))
        out.append(rows)
    return out


# ------------------------------------------------------------------ the check against the engine (play-driven jokers)
PLAY_JOKERS = ("runner", "square", "trousers", "green_joker", "wee", "castle", "yorick", "ice_cream", "ramen", "ride_the_bus", "obelisk")


def check(per_joker=48, rounds=6, seed0=600_000):
    """Each play-driven joker placed into a random ante-2..4 build (VecEnv.set_build), played on by the engine for `rounds` blinds (scripted
    rounds, every blind selected, every shop left); its counter after those rounds against the projection."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    from gen_builds import make_build, strength_weights, KEYS
    from runeval_data import Policy
    from .engine_env import CppVecEnv
    from .runeval import from_vecenv
    from .sim.jokers import JOKERS
    rng = np.random.default_rng(seed0)
    W = strength_weights()
    none = Policy("none")
    print(f"{'joker':>14} {'n':>3} {'actual change':>14} {'projected':>10} {'corr':>6}")
    for jk in PLAY_JOKERS:
        n = per_joker
        v = CppVecEnv(n, "RED", "GOLD", seed0=seed0 + hash(jk) % 1000, threads=0)
        v.set_auto_round(True, 1.5, 0)
        v.v.restart()
        for i in range(n):
            ante = int(rng.integers(2, 5))
            ks, cs, es, lv, su, sn, st, mo = make_build(rng, ante, W)
            ks = [k for k in ks if k != jk][:4] + [jk]
            cs = [float("nan")] * len(ks)
            v.set_build([i], ante, 0, [KEYS.index(k) for k in ks], cs, st, lv, mo, [0] * len(ks), su, sn)
        inp = from_vecenv(v)
        proj = project(**inp, rounds=rounds)
        jid0 = np.asarray(v.obs["jid"]).copy()
        slot = np.argmax(jid0 == VOCAB.index(f"j_{jk}"), axis=1)
        now = State(**inp).counter[np.arange(n), slot]
        done = np.zeros(n, dtype=bool)
        blinds_done = np.zeros(n, dtype=int)
        final = np.full(n, np.nan)
        for _ in range(2000):
            live = np.flatnonzero(~done)
            if not len(live):
                break
            g = np.asarray(v.obs["g"])
            was_sel = g[:, Cols.STATE] > 0.5
            _, d, _ = v.step(none.act(v, live))
            g = np.asarray(v.obs["g"])
            st_ = State(g, np.asarray(v.obs["jid"]), np.asarray(v.obs["jf"]))
            for i in live:
                if d[i]:
                    done[i] = True
                    continue
                if g[i, Cols.STATE] > 0.5 and not was_sel[i]:
                    blinds_done[i] += 1
                    if blinds_done[i] >= rounds:
                        jids = np.asarray(v.obs["jid"])[i]
                        hit = np.flatnonzero(jids == VOCAB.index(f"j_{jk}"))
                        final[i] = st_.counter[i, hit[0]] if len(hit) else np.nan
                        if jk == "supernova":
                            final[i] = np.nan
                        done[i] = True
        p_change = np.array([next((r["future"] - r["now"] for r in rows if r["key"] == jk), np.nan) for rows in proj])
        a_change = final - now
        ok = np.isfinite(a_change) & np.isfinite(p_change)
        c = np.corrcoef(a_change[ok], p_change[ok])[0, 1] if ok.sum() > 3 and np.std(a_change[ok]) > 0 and np.std(p_change[ok]) > 0 else float("nan")
        print(f"{jk:>14} {ok.sum():3d} {np.mean(a_change[ok]) if ok.any() else float('nan'):14.2f} {np.mean(p_change[ok]) if ok.any() else float('nan'):10.2f} {c:6.2f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", type=int, default=48)
    p.add_argument("--rounds", type=int, default=6)
    a = p.parse_args()
    check(a.check, a.rounds)
