"""Marginal-value inputs (the specification; the C++ engine's Env::marginal_fill must match it value for value).

The network sees what a joker is, and what the hand it scores now would score; it does not see what a joker is
WORTH to the build. These inputs say so, outside rounds (shop, packs, blind select) where the build is chosen:

* the build's expected best play: K_HANDS hands of the hand size are sampled from the deck (a fixed pseudo-random
  choice), the game is put in the state of a round's start (full hands and discards, nothing played), and the best of
  all plays of each hand is scored with the jokers held. The mean over the hands is the build's expected score;
* for each held joker: how much of that score it is (the score without it);
* for each joker and planet on offer (shop, pack picks, replacements): the score if bought / picked / replacing a
  held joker, against the current one;
* globals: the expected score, and how many of the ante's three blinds it clears with a round's hands.

All comparisons use the same sampled hands, so the differences carry the effect of the one change. The scoring
of a hand re-scores only the best play of each hand type found for the current build (a joker rarely makes an
unrelated play the best one).
"""
from __future__ import annotations

import math
import random
from itertools import combinations

from .sim.hands import N_HANDS
from .sim.items import PLANETS
from .sim.jokers import Joker
from .sim.scoring import Plan

K_HANDS = 4
MARG_G, MARG_J, MARG_A = 4, 2, 3          # columns added to the global vector, to a joker, to an action row
POT_G, POT_J, POT_A = 1, 1, 1             # the potential columns (level 2), after those

# POTENTIAL: what a growing joker would be worth after ROUNDS_AHEAD rounds if the events that grow it happen at the
# rate listed. key -> (growth of its counter, its counter before it has one). The growth is (increment per event, as
# in the game's rules) x (events per round a player who goes after it manages) x ROUNDS_AHEAD. It says what is possible,
# not what the policy does: the passive growers grow by playing hands, the others need deliberate actions.
ROUNDS_AHEAD = 3
POTENTIAL = {
    "green_joker": (1 * 4 * ROUNDS_AHEAD, 0.0),       # +1 mult per hand played (4 hands a round)
    "ride_the_bus": (1 * 3 * ROUNDS_AHEAD, 0.0),      # +1 mult per hand without a face card
    "red_card": (3 * 1.5 * ROUNDS_AHEAD, 0.0),        # +3 mult per booster pack skipped
    "flash": (2 * 2 * ROUNDS_AHEAD, 0.0),             # +2 mult per reroll
    "runner": (15 * 1 * ROUNDS_AHEAD, 0.0),           # +15 chips per hand containing a Straight
    "square": (4 * 2 * ROUNDS_AHEAD, 0.0),            # +4 chips per hand of exactly four cards
    "castle": (3 * 3 * ROUNDS_AHEAD, 0.0),            # +3 chips per discarded card of its suit
    "wee": (8 * 1 * ROUNDS_AHEAD, 0.0),               # +8 chips per 2 scored
    "trousers": (2 * 2 * ROUNDS_AHEAD, 0.0),          # +2 mult per Two Pair
    "constellation": (0.1 * 1 * ROUNDS_AHEAD, 1.0),   # +0.1 x per planet used
    "hologram": (0.25 * 2 * ROUNDS_AHEAD, 1.0),       # +0.25 x per playing card added to the deck
    "vampire": (0.1 * 2 * ROUNDS_AHEAD, 1.0),         # +0.1 x per enhanced card scored
    "obelisk": (0.2 * 2 * ROUNDS_AHEAD, 1.0),         # +0.2 x per hand that is not the most played
    "lucky_cat": (0.25 * 1 * ROUNDS_AHEAD, 1.0),      # +0.25 x per Lucky card trigger
    "glass": (0.75 * 0.5 * ROUNDS_AHEAD, 1.0),        # +0.75 x per Glass card broken
    "campfire": (0.25 * 1 * ROUNDS_AHEAD, 1.0),       # +0.25 x per card sold
    "hit_the_road": (0.5 * 2, 1.0),                   # +0.5 x per Jack discarded (resets every round: one round's worth)
}


def _pot_val(j) -> float:
    delta, v0 = POTENTIAL[j.key]
    return j.state.get("val", v0) + delta
BASE_HAND = 8
_SUBSETS = [c for k in range(1, 6) for c in combinations(range(BASE_HAND), k)]


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sample_hand(n_cards: int, k: int, n: int) -> list[int]:
    """Positions of the k-th sampled hand among the deck's cards (a partial shuffle driven by a fixed LCG)."""
    idx = list(range(n_cards))
    x = (0x2545F491 + 7919 * k) & 0x7FFFFFFF
    for i in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        j = i + x % (n_cards - i)
        idx[i], idx[j] = idx[j], idx[i]
    return idx[:n]


class _Scratch:
    """The game as it would be at the start of a round, over sampled hands."""

    def __init__(self, g):
        n = min(BASE_HAND, g.effective_hand_size(), len(g.full_deck))
        self.n = n
        s = g.clone()
        s.state = "SELECTING_HAND"
        s.blind_idx = 0
        s.hands_left = s.round_hands()
        s.discards_left = s.round_discards()
        s.discards_used_round = 0
        s.hand_played_round = [0] * N_HANDS
        s.round_hand_types = set()
        self.s = s
        self.jokers0 = list(s.jokers)
        self.levels0 = list(s.hand_levels)
        cards = s.full_deck
        self.hands, self.rests = [], []
        for k in range(K_HANDS):
            idx = sample_hand(len(cards), k, n)
            taken = set(idx)
            self.hands.append([cards[i] for i in idx])
            self.rests.append([c for i, c in enumerate(cards) if i not in taken])
        self.subs = [c for c in _SUBSETS if c[-1] < n]
        self.reps: list[list[tuple]] = []
        self.base = 0.0

    def _best(self, k: int, subs) -> tuple[float, list]:
        s = self.s
        s.hand = self.hands[k]
        s.deck = self.rests[k]
        plan = Plan(s)
        view = s.hand_view()
        preds = s.predict_many(subs, plan, view)
        best = {}
        for q, (sc, h) in enumerate(preds):
            if h not in best or sc > best[h][0]:
                best[h] = (sc, q)
        top = max(sc for sc, _ in best.values()) if best else 0.0
        return top, [subs[q] for _, q in best.values()]

    def prepare(self):
        total = 0.0
        for k in range(K_HANDS):
            top, reps = self._best(k, self.subs)
            total += top
            self.reps.append(reps)
        self.base = total / K_HANDS

    def mean(self) -> float:
        """Expected best play of the scratch game as it is now (the base plays of each hand type re-scored)."""
        s = self.s
        total = 0.0
        for k in range(K_HANDS):
            s.hand = self.hands[k]
            s.deck = self.rests[k]
            plan = Plan(s)
            view = s.hand_view()
            preds = s.predict_many(self.reps[k], plan, view)
            total += max(sc for sc, _ in preds)
        return total / K_HANDS

    def with_jokers(self, remove: int = -1, add=None, potential: bool = False) -> float:
        from .env import order_jokers
        s = self.s
        slots = s.joker_slots                    # (a Negative joker adds a slot; its init may draw random numbers)
        s.jokers = list(self.jokers0)
        if remove >= 0:
            del s.jokers[remove]
        if add is not None:
            j = Joker(add.key, add.edition, add.eternal, add.perishable, add.rental, add.base_cost, add.sell_bonus,
                      add.cost, add.debuffed, add.hidden, dict(add.state))
            s.rng = random.Random(12345)
            s.add_joker(j)
            order_jokers(s)
            if potential and j.key in POTENTIAL:
                j.state["val"] = _pot_val(j)
        m = self.mean()
        s.jokers = list(self.jokers0)
        s.joker_slots = slots
        return m

    def with_potential(self, idxs) -> float:
        """The expected score with the held jokers at these positions (those with a potential) grown to it."""
        s = self.s
        saved = []
        for i in idxs:
            j = s.jokers[i]
            if j.key in POTENTIAL:
                saved.append((j, j.state.get("val", None)))
                j.state["val"] = _pot_val(j)
        m = self.mean()
        for j, v in saved:
            if v is None:
                del j.state["val"]
            else:
                j.state["val"] = v
        return m

    def with_level(self, hand: int) -> float:
        s = self.s
        s.hand_levels[hand] += 1
        m = self.mean()
        s.hand_levels[hand] -= 1
        return m


LAST: dict = {}                      # the last report's raw numbers, for debugging a mismatch with the engine


def _gain_row(base: float, mean: float, cost: float) -> list[float]:
    lr = math.log10((mean + 1.0) / (base + 1.0))
    return [_clip(lr, -1.0, 3.0) / 3.0, _clip(mean / base - 1.0, -1.0, 5.0) / 5.0,
            _clip(lr / max(1.0, cost), -1.0, 1.0)]


def _pot_value(base: float, mean: float) -> float:
    return _clip(math.log10((mean + 1.0) / (base + 1.0)), -1.0, 3.0) / 3.0


def fill(g, mask, glob, jf, af, potential: bool = False):
    """Writes the marginal inputs into the observation arrays of env.encode (glob: the global vector, jf: joker
    features, af: action features; the columns are the MARG_* then the POT_* last of each). potential: also the
    potential columns."""
    from .env import A_BUY, A_REPLACE, A_PICK, MAX_SHOP, MAX_PACK, MAX_JOKERS
    F_GLOBAL_BASE = len(glob) - MARG_G - POT_G           # the columns are the last of each array
    F_JOKER_BASE = jf.shape[1] - MARG_J - POT_J
    F_ACT_BASE = af.shape[1] - MARG_A - POT_A
    G_POT, J_POT, A_POT = F_GLOBAL_BASE + MARG_G, F_JOKER_BASE + MARG_J, F_ACT_BASE + MARG_A
    sc = _Scratch(g)
    sc.prepare()
    base = sc.base
    LAST.clear()
    LAST["base"] = base
    LAST["items"] = []
    glob[F_GLOBAL_BASE] = math.log10(1.0 + base) / 7.0
    hands = g.round_hands()
    for i in range(3):
        glob[F_GLOBAL_BASE + 1 + i] = min(5.0, hands * base / max(1.0, float(g.blind_target(i))))
    for i, j in enumerate(g.jokers[:MAX_JOKERS]):
        m = sc.with_jokers(remove=i)
        jf[i, F_JOKER_BASE] = _clip(1.0 - m / base, -1.0, 1.0)
        jf[i, F_JOKER_BASE + 1] = _clip(math.log10((base + 1.0) / (m + 1.0)), -1.0, 3.0) / 3.0
    if potential:
        growers = [i for i, j in enumerate(g.jokers[:MAX_JOKERS]) if j.key in POTENTIAL]
        for i in growers:
            jf[i, J_POT] = _pot_value(base, sc.with_potential([i]))
        if growers:
            glob[G_POT] = _pot_value(base, sc.with_potential(growers))
    if g.state == "SHOP" and g.targeting is None:
        for i, it in enumerate(g.shop[:MAX_SHOP]):
            if it.kind == "joker":
                m = sc.with_jokers(add=it.joker)
                LAST["items"].append((A_BUY + i, m))
                af[A_BUY + i, F_ACT_BASE:F_ACT_BASE + MARG_A] = _gain_row(base, m, it.cost)
                grows = potential and it.joker.key in POTENTIAL
                if grows:
                    af[A_BUY + i, A_POT] = _pot_value(base, sc.with_jokers(add=it.joker, potential=True))
                for j, jk in enumerate(g.jokers[:MAX_JOKERS]):
                    if mask[A_REPLACE + i * MAX_JOKERS + j]:
                        m = sc.with_jokers(remove=j, add=it.joker)
                        row = A_REPLACE + i * MAX_JOKERS + j
                        af[row, F_ACT_BASE:F_ACT_BASE + MARG_A] = _gain_row(base, m, it.cost - jk.sell_value())
                        if grows:
                            af[row, A_POT] = _pot_value(base, sc.with_jokers(remove=j, add=it.joker, potential=True))
            elif it.kind == "planet":
                m = sc.with_level(PLANETS[it.key[2:]])
                af[A_BUY + i, F_ACT_BASE:F_ACT_BASE + MARG_A] = _gain_row(base, m, it.cost)
    if g.state == "PACK" and g.targeting is None:
        for i, x in enumerate(g.pack_cards[:MAX_PACK]):
            if isinstance(x, Joker):
                m = sc.with_jokers(add=x)
                af[A_PICK + i, F_ACT_BASE:F_ACT_BASE + MARG_A] = _gain_row(base, m, 0)
                if potential and x.key in POTENTIAL:
                    af[A_PICK + i, A_POT] = _pot_value(base, sc.with_jokers(add=x, potential=True))
            elif getattr(x, "kind", "") == "planet":
                m = sc.with_level(PLANETS[x.name])
                af[A_PICK + i, F_ACT_BASE:F_ACT_BASE + MARG_A] = _gain_row(base, m, 0)
