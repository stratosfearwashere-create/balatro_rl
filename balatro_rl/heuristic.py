"""Rule-based Balatro player. Used as a baseline and to generate behaviour-cloning data
that warm-starts the neural network before reinforcement learning."""
from __future__ import annotations

import numpy as np

from .env import (SUBSETS, N_SUB, A_PLAY, A_DISC, A_SELECT, A_SKIP, A_BUY, A_BUY_PACK, A_VOUCHER,
                  A_REROLL, A_LEAVE, A_SELL_J, A_SELL_C, A_USE_C, A_PICK, A_PSKIP, A_SWAP, N_ACTIONS,
                  RATIO_COL, subset_of)
from .sim.game import Game, Consumable
from .sim.hands import N_HANDS, PAIR, FLUSH, HAND_BASE
from .sim.items import PLANETS
from .sim.jokers import Joker

JOKER_TIER = {
    "blueprint": 9, "brainstorm": 8, "cavendish": 8, "duo": 7, "tribe": 7, "card_sharp": 7, "ancient": 7,
    "trio": 6, "order": 6, "baron": 6, "obelisk": 6, "loyalty_card": 6, "acrobat": 6, "idol": 6,
    "constellation": 6, "ramen": 6, "stencil": 6, "bloodstone": 6, "rocket": 6, "family": 5,
    "blackboard": 5, "flower_pot": 5, "seeing_double": 5, "photograph": 5, "golden": 5, "cloud_9": 5,
    "rough_gem": 5, "gros_michel": 5, "droll": 5, "half": 5, "abstract": 5, "supernova": 5,
    "ride_the_bus": 5, "green_joker": 5, "fibonacci": 5, "onyx_agate": 5, "trousers": 5, "bootstraps": 5,
    "space": 5, "selzer": 5, "to_the_moon": 4, "satellite": 4, "delayed_grat": 4, "business": 4,
    "reserved_parking": 4, "jolly": 4, "zany": 4, "mad": 4, "crazy": 4, "misprint": 4, "popcorn": 4,
    "smiley": 4, "even_steven": 4, "scholar": 4, "greedy_joker": 4, "lusty_joker": 4,
    "wrathful_joker": 4, "gluttenous_joker": 4, "swashbuckler": 4, "raised_fist": 4, "fortune_teller": 4,
    "red_card": 4, "flash": 4, "crafty": 4, "banner": 4, "blue_joker": 4, "ice_cream": 4, "runner": 4,
    "bull": 4, "arrowhead": 4, "hiker": 4, "four_fingers": 4, "smeared": 4, "hack": 4, "dusk": 4,
    "hanging_chad": 4, "sock_and_buskin": 4, "mime": 4, "joker": 3, "mystic_summit": 3, "walkie_talkie": 3,
    "shoot_the_moon": 3, "sly": 3, "wily": 3, "clever": 3, "devious": 3, "scary_face": 3, "odd_todd": 3,
    "square": 3, "shortcut": 3, "pareidolia": 3, "splash": 3, "chaos": 3, "drunkard": 3, "egg": 3,
    "faceless": 3, "mail": 3, "todo_list": 3,
    "credit_card": 1, "ceremonial": 3, "marble": 2, "8_ball": 2, "steel_joker": 3, "burglar": 3, "dna": 4,
    "sixth_sense": 2, "superposition": 2, "madness": 2, "seance": 2, "riff_raff": 3, "vampire": 4,
    "hologram": 4, "vagabond": 3, "midas_mask": 3, "luchador": 3, "gift": 2, "turtle_bean": 3, "erosion": 3,
    "hallucination": 2, "juggler": 3, "stone": 2, "lucky_cat": 3, "baseball": 6, "diet_cola": 2, "trading": 3,
    "castle": 4, "campfire": 4, "ticket": 2, "mr_bones": 4, "troubadour": 2, "certificate": 3, "throwback": 4,
    "glass": 3, "ring_master": 2, "wee": 4, "merry_andy": 3, "oops": 3, "matador": 3, "hit_the_road": 4,
    "stuntman": 5, "invisible": 4, "drivers_license": 3, "cartomancer": 3, "astronomer": 3, "burnt": 5,
    "caino": 6, "triboulet": 9, "yorick": 7, "chicot": 8, "perkeo": 7,
}
GOOD_VOUCHERS = ["grabber", "wasteful", "clearance_sale", "antimatter", "nacho_tong", "recyclomancy",
                 "overstock_norm", "liquidation", "seed_money", "crystal_ball", "telescope", "observatory",
                 "reroll_surplus", "planet_merchant", "hone", "glow_up", "money_tree"]
TAROT_PREF = ["judgement", "hermit", "justice", "chariot", "empress", "magician", "lovers", "temperance",
              "death", "strength", "star", "moon", "sun", "world", "heirophant", "devil", "wheel_of_fortune",
              "hanged_man", "high_priestess", "emperor", "fool"]
SPECTRAL_PREF = ["soul", "black_hole", "talisman", "deja_vu", "cryptid", "aura", "wraith", "trance", "familiar",
                 "grim", "incantation"]
_OFF_RATIO = RATIO_COL      # column in action features with score / chips still needed
# joker ordering: +chips first, then +mult, then xmult (so xmult multiplies everything)
XMULT = {"duo", "trio", "family", "order", "tribe", "baron", "ancient", "obelisk", "cavendish", "card_sharp",
         "loyalty_card", "acrobat", "blackboard", "flower_pot", "seeing_double", "idol", "constellation", "ramen",
         "stencil", "photograph", "bloodstone", "madness", "vampire", "hologram", "steel_joker", "lucky_cat",
         "baseball", "campfire", "throwback", "glass", "hit_the_road", "drivers_license", "caino", "triboulet",
         "yorick", "polychrome"}
CHIPS = {"sly", "wily", "clever", "devious", "crafty", "banner", "scary_face", "odd_todd", "blue_joker",
         "ice_cream", "runner", "square", "bull", "arrowhead", "stone", "castle", "wee", "stuntman", "hiker"}


def joker_value(g: Game, j: Joker) -> float:
    v = JOKER_TIER.get(j.key, 3)
    v += {"": 0, "FOIL": 0.5, "HOLO": 1.5, "POLYCHROME": 3, "NEGATIVE": 4}[j.edition]
    if j.rental:
        v -= 2.5
    if j.perishable is not None:
        v -= 1.5
    if j.eternal:
        v -= 0.3
    if j.debuffed:
        v -= 5
    return v


def main_hand(g: Game) -> int:
    if max(g.hand_played) == 0:
        return PAIR
    return max(range(N_HANDS), key=lambda h: g.hand_played[h] * 1.0 + g.hand_levels[h] * 2)


def reserve(g: Game) -> int:
    if len(g.jokers) < 4:
        return 0
    return min(25, 5 * g.ante)


class HeuristicPolicy:
    def __init__(self, rng=None, lookahead: bool = False):
        self.rng = rng or np.random.default_rng(0)
        self.lookahead = lookahead
        self.shop_rerolls = 0

    def act(self, g: Game, obs: dict) -> int:
        m = obs["mask"]
        st = g.state
        if st == "BLIND_SELECT":
            self.shop_rerolls = 0
            return A_SELECT
        if st == "SELECTING_HAND":
            return self._play(g, obs)
        if st == "SHOP":
            return self._shop(g, m)
        if st == "PACK":
            return self._pack(g, m)
        return int(np.flatnonzero(m)[0])

    # ---------------------------------------------------------------- hand
    def _play(self, g: Game, obs: dict) -> int:
        m = obs["mask"]
        # boss: verdant leaf -> sell the cheapest joker
        if g.boss_active() == "verdant_leaf" and g.flags.get("verdant"):
            cand = [i for i, j in enumerate(g.jokers) if m[A_SELL_J + i]]
            if cand:
                return A_SELL_J + min(cand, key=lambda i: joker_value(g, g.jokers[i]))
        # use consumables that help now
        for i, c in enumerate(g.consumables):
            if m[A_USE_C + i] and c.name not in ("hanged_man",):
                return A_USE_C + i
        ratio = obs["af"][A_PLAY:A_PLAY + N_SUB, _OFF_RATIO]
        legal_play = m[A_PLAY:A_PLAY + N_SUB]
        r = np.where(legal_play, ratio, -1.0)
        best = int(np.argmax(r))
        if r[best] >= 1.0:
            # among winning plays prefer fewer cards (keeps it tidy), tie -> higher score
            wins = np.flatnonzero(r >= 1.0)
            best = min(wins, key=lambda i: (len(subset_of(obs, A_PLAY + i)), -r[i]))
            return A_PLAY + best
        if g.discards_left > 0 and r[best] * g.hands_left < 1.15:
            if self.lookahead and len(g.hand) <= 8:
                d = self._discard_mc(g, m, r[best] * max(g.target - g.chips, 1))
            else:
                d = self._discard_choice(g, obs)
            if d is not None:
                return d
        return A_PLAY + best

    def _candidates(self, g: Game) -> list[tuple]:
        hand = g.hand
        sm = g.has("smeared")
        cands = set()
        for s in range(4):
            idx = [i for i, c in enumerate(hand) if c.has_suit(s, sm)]
            if len(idx) >= 3:
                out = sorted([i for i in range(len(hand)) if i not in idx], key=lambda i: g.card_value(hand[i]))[:5]
                if out:
                    cands.add(tuple(sorted(out)))
        rank_cnt = {}
        for c in hand:
            if not c.is_stone:
                rank_cnt[c.rank] = rank_cnt.get(c.rank, 0) + 1
        out = sorted([i for i, c in enumerate(hand) if c.is_stone or rank_cnt.get(c.rank, 0) < 2],
                     key=lambda i: g.card_value(hand[i]))
        if len(out) == len(hand):
            out = out[:-2]
        if out:
            cands.add(tuple(sorted(out[:5])))
        # straight windows
        for lo in range(1, 11):
            want = set(range(lo, lo + 5))
            keep, seen = [], set()
            for i, c in enumerate(hand):
                r = 1 if (not c.is_stone and c.rank == 14 and lo == 1) else c.rank
                if not c.is_stone and r in want and r not in seen:
                    keep.append(i)
                    seen.add(r)
            if len(keep) >= 3:
                out = [i for i in range(len(hand)) if i not in keep]
                out = sorted(out, key=lambda i: g.card_value(hand[i]))[:5]
                if out:
                    cands.add(tuple(sorted(out)))
        fp = g.forced_pos()
        res = []
        for c in cands:
            if fp >= 0 and fp not in c:
                c = tuple(sorted((list(c)[:4]) + [fp]))
            res.append(c)
        return res

    def _discard_mc(self, g: Game, m, cur_best: float, samples: int = 6):
        from .sim.scoring import Plan
        cands = self._candidates(g)
        if not cands:
            return None
        plan = Plan(g)
        hand0 = g.hand
        best_c, best_v = None, cur_best * 1.1
        rng = self.rng
        for cand in cands:
            idx = SUBSETS.index(cand) if cand in SUBSETS else -1
            if idx < 0 or not m[A_DISC + idx]:
                continue
            kept = [c for i, c in enumerate(hand0) if i not in cand]
            need = len(cand)
            tot = 0.0
            for _ in range(samples):
                if len(g.deck) >= need:
                    pick = rng.choice(len(g.deck), size=need, replace=False)
                    drawn = [g.deck[k] for k in pick]
                else:
                    drawn = list(g.deck)
                from .sim.cards import sort_hand
                g.hand = sort_hand(kept + drawn)
                bs = 0.0
                n = len(g.hand)
                for sc, _ in g.predict_many([s for s in SUBSETS if s[-1] < n], plan):
                    if sc > bs:
                        bs = sc
                tot += bs
            g.hand = hand0
            v = tot / samples
            if v > best_v:
                best_c, best_v = idx, v
        return None if best_c is None else A_DISC + best_c

    def _discard_choice(self, g: Game, obs) -> int | None:
        m = obs["mask"]
        hand = g.hand
        n = len(hand)
        suit_cnt = [0, 0, 0, 0]
        for c in hand:
            for s in range(4):
                if c.has_suit(s, g.has("smeared")):
                    suit_cnt[s] += 1
        s_best = max(range(4), key=lambda s: suit_cnt[s])
        rank_cnt = {}
        for c in hand:
            if not c.is_stone:
                rank_cnt[c.rank] = rank_cnt.get(c.rank, 0) + 1
        flush_bias = g.hand_levels[FLUSH] - g.hand_levels[PAIR] + (2 if g.has("droll") or g.has("tribe") or g.has("crafty") else 0)
        go_flush = suit_cnt[s_best] >= 4 or (suit_cnt[s_best] >= 3 and flush_bias >= 0 and max(rank_cnt.values(), default=1) < 3)
        if go_flush:
            out = [i for i, c in enumerate(hand) if not c.has_suit(s_best, g.has("smeared"))]
            out.sort(key=lambda i: g.card_value(hand[i]))
        else:
            out = [i for i, c in enumerate(hand) if c.is_stone or rank_cnt.get(c.rank, 0) < 2]
            out.sort(key=lambda i: g.card_value(hand[i]))
            # keep the two highest singletons if we have no pair at all
            if len(out) == n:
                out = out[:-2]
        out = sorted(out[:5])
        if g.forced_pos() >= 0 and g.forced_pos() not in out:
            if len(out) == 5:
                out = out[:4]
            out = sorted(out + [g.forced_pos()])
        if not out:
            return None
        return self._disc_slot(obs, out)

    @staticmethod
    def _disc_slot(obs, out) -> int | None:
        """Find the discard candidate slot closest to the wanted cards."""
        want = set(out)
        best, best_score = None, -1e9
        rows = obs["subs"][N_SUB:]
        m = obs["mask"][A_DISC:A_DISC + N_SUB]
        for i in np.flatnonzero(m):
            have = {int(x) for x in rows[i] if x >= 0}
            sc = len(have & want) * 2 - len(have - want) * 3 - len(want - have)
            if sc > best_score:
                best, best_score = i, sc
        if best is None or best_score <= 0:
            return None
        return A_DISC + int(best)

    # ---------------------------------------------------------------- shop
    def _order_jokers(self, g: Game, m) -> int | None:
        def cat(j):
            if j.edition == "POLYCHROME" or j.key in XMULT:
                return 2
            return 0 if j.key in CHIPS else 1
        for i in range(len(g.jokers) - 1):
            a, b = g.jokers[i], g.jokers[i + 1]
            if b.key == "blueprint" or a.key == "blueprint":
                continue
            if cat(a) > cat(b) and A_SWAP + i < N_ACTIONS and m[A_SWAP + i]:
                return A_SWAP + i
        return None

    def _shop(self, g: Game, m) -> int:
        res = reserve(g)
        mh = main_hand(g)
        sw = self._order_jokers(g, m)
        if sw is not None:
            return sw
        # planets: use immediately
        for i, c in enumerate(g.consumables):
            if m[A_USE_C + i] and c.kind == "planet":
                return A_USE_C + i
            if m[A_USE_C + i] and c.name in ("hermit", "temperance", "judgement", "black_hole"):
                return A_USE_C + i
        # voucher
        if m[A_VOUCHER]:
            v = g.shop_voucher.key[2:]
            if v in GOOD_VOUCHERS and len(g.jokers) >= 3 and g.money - g.shop_voucher.cost >= max(res, 5):
                return A_VOUCHER
        # jokers
        shop_j = [(i, it) for i, it in enumerate(g.shop) if it.kind == "joker"]
        free_slot = len(g.jokers) < g.joker_slots
        best_buy, best_v = None, -1e9
        for i, it in shop_j:
            v = joker_value(g, it.joker)
            if it.cost > g.money:
                continue
            need_res = res
            if g.money - it.cost < need_res and v < 6:
                continue
            if v > best_v:
                best_buy, best_v = i, v
        if best_buy is not None:
            if free_slot or g.shop[best_buy].joker.edition == "NEGATIVE":
                if best_v >= 3:
                    return A_BUY + best_buy
            else:
                sellable = [(joker_value(g, j), k) for k, j in enumerate(g.jokers) if not j.eternal]
                if sellable:
                    wv, wk = min(sellable)
                    if best_v >= wv + 2 and g.money + g.jokers[wk].sell_value() >= g.shop[best_buy].cost:
                        return A_SELL_J + wk
        # planets for main hand
        for i, it in enumerate(g.shop):
            if it.kind == "planet" and m[A_BUY + i] and PLANETS[it.key[2:]] == mh and g.money - it.cost >= min(res, 5):
                return A_BUY + i
        # packs
        for i, it in enumerate(g.shop_packs):
            if not m[A_BUY_PACK + i]:
                continue
            kind = it.pack[0]
            if kind == "buffoon" and free_slot and g.money - it.cost >= min(res, 5):
                return A_BUY_PACK + i
            if kind == "celestial" and g.money - it.cost >= res + 3:
                return A_BUY_PACK + i
            if kind == "spectral" and g.money - it.cost >= res + 6:
                return A_BUY_PACK + i
        # reroll to find jokers when we have spare money
        if m[A_REROLL] and free_slot and self.shop_rerolls < 3 and (
                g.free_rerolls > 0 or g.money - g.reroll_cost >= res + 8):
            self.shop_rerolls += 1
            return A_REROLL
        return A_LEAVE

    # ---------------------------------------------------------------- packs
    def _pack(self, g: Game, m) -> int:
        legal = [i for i in range(len(g.pack_cards)) if m[A_PICK + i]]
        if not legal:
            return A_PSKIP
        k = g.pack_kind
        if k == "buffoon":
            best = max(legal, key=lambda i: joker_value(g, g.pack_cards[i]))
            if joker_value(g, g.pack_cards[best]) >= 2:
                return A_PICK + best
            return A_PSKIP
        if k == "celestial":
            mh = main_hand(g)
            bh = [i for i in legal if g.pack_cards[i].name == "black_hole"]
            if bh:
                return A_PICK + bh[0]
            legal = [i for i in legal if g.pack_cards[i].name in PLANETS] or legal
            best = max(legal, key=lambda i: (PLANETS.get(g.pack_cards[i].name) == mh,
                                             g.hand_played[PLANETS.get(g.pack_cards[i].name, 0)],
                                             -PLANETS.get(g.pack_cards[i].name, 0)))
            return A_PICK + best
        if k == "arcana":
            pref = {n: r for r, n in enumerate(TAROT_PREF)}
            best = min(legal, key=lambda i: pref.get(g.pack_cards[i].name, 99))
            return A_PICK + best
        if k == "spectral":
            pref = {n: r for r, n in enumerate(SPECTRAL_PREF)}
            best = min(legal, key=lambda i: pref.get(g.pack_cards[i].name, 99))
            if pref.get(g.pack_cards[best].name, 99) < 99:
                return A_PICK + best
            return A_PSKIP
        if k == "standard":
            best = max(legal, key=lambda i: g.card_value(g.pack_cards[i]))
            c = g.pack_cards[best]
            if c.enh not in ("", "STONE") or c.seal or c.edition:
                return A_PICK + best
            return A_PSKIP
        return A_PSKIP
