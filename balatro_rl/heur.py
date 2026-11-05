"""A heuristic shop bot whose rules are a weighted score, tuned by CMA-ES (cma.py) on Gold games.

Every decision outside a round (shop, packs, blind select / skip) scores each legal action and takes the best; the exact round player plays the rounds
(the engine's auto round, as in evo.py). The score is a sum of readable terms, each with a named weight, in "dollars":

    item value      V[item] for what the action gains (a joker, a pack, a voucher, a consumable to use, a tag for a skip), minus what it gives up
                    (the joker sold or replaced); started from the Gold guide's tables (az/guide.py)
    joker roles     chips / +mult / xmult / scaler / economy / narrow: a bonus for the first of a role, one for each more of it, and a slope per ante
    planets         for the main hand (the most played), per level it already has, any planet
    stickers        edition (foil, holo, polychrome, negative), eternal, perishable, rental, rarity
    money           price, the interest the purchase gives up (dollars of interest per round), price x ante
    action biases   skip, reroll, leave, select, swap ...; leave and reroll also look at the money held, reroll at whether an xmult is held

    python -m balatro_rl.heur --pop 32 --games 32 --out checkpoints/heur1.pt          (tune)
    python -m balatro_rl.heur --show checkpoints/heur1_best.pt                        (the weights, biggest moves from the start first)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import cma
from .az import guide as G
from .env import (ENHANCEMENTS, SEALS, N_ACTIONS, A_LEAVE, A_PICK, A_PSKIP, A_REPLACE, A_REROLL, A_REROLL_BOSS, A_SELECT, A_SELL_C, A_SELL_J, A_SKIP, A_SWAP, A_SWAP_END, A_USE_C, A_BUY,
                  A_BUY_PACK, A_VOUCHER, ACTION_TYPES, EDITIONS, MAX_JOKERS, N_HANDS, TAGS, BOSS_KEYS)
from .sim.items import PLANETS, VOCAB

A0 = A_SELECT
SOLVER_OPTS = {}                                           # set by train(): smart (discards first where greedy cannot clear), lite (cheap solver steps)
NA = N_ACTIONS - A_SELECT                                  # the actions outside rounds (the 32+ replace rows included)
DECAY = {"gros_michel", "popcorn", "ice_cream", "ramen", "selzer", "turtle_bean", "cavendish"}
ROLES = ["chips", "mult", "xmult", "scaler", "econ", "narrow", "scoring", "slow scaler", "decaying"]
ROLE_SETS = [G.CHIPS, G.MULT, G.XMULT, G.SCALERS, G.ECON, G.NARROW, G.CHIPS | G.MULT | G.XMULT, G.SLOW_SCALERS, DECAY]
# columns of the observation (env.encode_global / env.action_features)
G_ANTE, G_MONEY = 0, 17
G_TAGS = 25 + len(BOSS_KEYS) + 3                          # the tag a skip would take (one-hot)
G_LEVELS = G_TAGS + len(TAGS) + 1
G_PLAYED = G_LEVELS + N_HANDS
OFF_ITEM = len(ACTION_TYPES) + 1 + N_HANDS + 5 + 6
C_COST, C_ED, C_ETERNAL, C_PERISH, C_RENTAL, C_RARITY, C_SELL = OFF_ITEM, OFF_ITEM + 2, OFF_ITEM + 7, OFF_ITEM + 8, OFF_ITEM + 9, OFF_ITEM + 10, OFF_ITEM + 12
# the actions outside rounds, by what they do
R = lambda lo, hi: list(range(lo - A0, hi - A0))
ROWS = {"select": R(A_SELECT, A_SELECT + 1), "skip": R(A_SKIP, A_SKIP + 1), "reroll_boss": R(A_REROLL_BOSS, A_REROLL_BOSS + 1), "buy": R(A_BUY, A_BUY_PACK),
        "pack": R(A_BUY_PACK, A_VOUCHER), "voucher": R(A_VOUCHER, A_REROLL), "reroll": R(A_REROLL, A_REROLL + 1), "leave": R(A_LEAVE, A_LEAVE + 1),
        "sell_joker": R(A_SELL_J, A_SELL_C), "sell_cons": R(A_SELL_C, A_USE_C), "use": R(A_USE_C, A_PICK), "pick": R(A_PICK, A_PSKIP),
        "pack_skip": R(A_PSKIP, A_SWAP), "swap": R(A_SWAP, A_SWAP_END), "replace": R(A_REPLACE, A0 + NA)}
BIASES = list(ROWS)
NEW_IN_V2 = {"power gap: margin (log)", "boss gap: margin (log)", "boss gap x econ joker", "sell an econ joker: slots full x boss gap",
             "replace a flat joker with an xmult joker: per ante from 3", "boss gap x skip", "boss gap x scoring joker", "boss gap x xmult joker", "boss gap x leave", "skip: no scoring joker held","power gap x scoring joker", "power gap x xmult joker", "power gap x econ joker", "power gap x pack", "power gap x reroll", "power gap x leave",
             "power gap x skip", "suit joker: same suit already held", "suit joker: another suit held", "hand joker: boosts the main hand",
             "hand joker: boosts another hand", "debt: per $ below zero after buying"}
SUIT_JOKERS = {"wrathful_joker": 0, "arrowhead": 0, "lusty_joker": 1, "bloodstone": 1, "gluttenous_joker": 2, "onyx_agate": 2, "greedy_joker": 3, "rough_gem": 3}   # the engine's suits: S H C D
# deck fit: the ranks (index rank - 2), suits or enhancements a joker wants in the deck
FACE = [9, 10, 11]
RANK_FIT = {"even_steven": [0, 2, 4, 6, 8], "odd_todd": [12, 1, 3, 5, 7], "walkie_talkie": [8, 2], "fibonacci": [12, 0, 1, 3, 6], "hack": [0, 1, 2, 3], "wee": [0],
            "scholar": [12], "baron": [11], "shoot_the_moon": [10], "triboulet": [10, 11], "8_ball": [6], "scary_face": FACE, "smiley": FACE, "photograph": FACE,
            "sock_and_buskin": FACE, "midas_mask": FACE, "business": FACE, "reserved_parking": FACE, "faceless": FACE}
SUIT_FIT = {**{k: [v] for k, v in SUIT_JOKERS.items()}, "blackboard": [0, 2]}
ENH_FIT = {"steel_joker": ["STEEL"], "glass": ["GLASS"], "lucky_cat": ["LUCKY"], "stone": ["STONE"], "ticket": ["GOLD"],
           "drivers_license": ["BONUS", "MULT", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY"]}
COPY = {"blueprint", "brainstorm"}
# build plans: core pieces, supporting pieces, the hand it levels (PLANETS' hand index; -1: none)
PLANS = {
    "flush":   ({"tribe", "droll", "crafty", "smeared", "four_fingers"}, set(SUIT_JOKERS) | {"ancient", "blackboard", "seeing_double", "flower_pot"}, 5),
    "pairs":   ({"duo", "jolly", "sly", "trousers", "mad", "clever"}, {"hanging_chad", "photograph", "card_sharp", "spare_trousers", "half"}, 1),
    "sets":    ({"trio", "family", "zany", "wily"}, {"duo", "jolly", "sly", "card_sharp", "hanging_chad"}, 3),
    "straight": ({"order", "crazy", "devious", "runner", "four_fingers", "shortcut"}, {"superposition", "smeared"}, 4),
    "faces":   ({"photograph", "sock_and_buskin", "smiley", "scary_face", "pareidolia", "triboulet"}, {"baron", "shoot_the_moon", "midas_mask", "business", "reserved_parking", "faceless"}, -1),
    "scaling": ({"hologram", "vampire", "lucky_cat", "madness", "constellation", "campfire", "throwback", "obelisk", "glass", "steel_joker"},
                {"ride_the_bus", "green_joker", "red_card", "square", "runner", "flash", "wee", "castle", "hiker"}, -1),
}
PLAN_NAMES = list(PLANS)
NPLAN = len(PLANS)
G_RC = G_PLAYED + 2 * N_HANDS                                # deck counts: 13 ranks (x/4), 4 suits (x/13), deck size, full deck size, 8 enhancements (x/10)
G_SC = G_RC + 13
G_FULL = G_SC + 5
G_ENH = G_SC + 6
G_CONS = G_ENH + len(ENHANCEMENTS) - 1 + len(SEALS) - 1 + 2  # consumable slots (x/3), consumables held (x/3)
HAND_JOKERS = {"jolly": 1, "sly": 1, "duo": 1, "mad": 2, "clever": 2, "zany": 3, "wily": 3, "trio": 3, "crazy": 4, "devious": 4, "order": 4,
               "droll": 5, "crafty": 5, "tribe": 5, "family": 7}     # the hand each boosts (PLANETS' hand index: 1 pair, 2 two pair, 3 three, 4 straight, 5 flush, 7 four)


def _item_tables():
    """Per vocabulary id: the starting value (the guide's), the joker roles, the hand a planet levels (-1: none)."""
    v0 = np.zeros(len(VOCAB), dtype=np.float32)
    roles = np.zeros((len(VOCAB), len(ROLES) + 1), dtype=np.float32)     # (the last column: it is a joker)
    planet = -np.ones(len(VOCAB), dtype=np.int64)
    suit = -np.ones(len(VOCAB), dtype=np.int64)
    hand = -np.ones(len(VOCAB), dtype=np.int64)
    rfit = np.zeros((len(VOCAB), 13), dtype=np.float32)
    sfit = np.zeros((len(VOCAB), 4), dtype=np.float32)
    efit = np.zeros((len(VOCAB), len(ENHANCEMENTS) - 1), dtype=np.float32)
    kind = np.zeros((len(VOCAB), 5), dtype=np.float32)        # copy joker, buffoon pack, celestial pack, consumable, (spare)
    core = np.zeros((len(VOCAB), NPLAN), dtype=np.float32)
    supp = np.zeros((len(VOCAB), NPLAN), dtype=np.float32)
    for i, name in enumerate(VOCAB):
        key = name.split("_", 1)[1] if "_" in name else name
        if name.startswith("j_"):
            v0[i] = G.VALUE.get(key, 3)
            roles[i, -1] = 1.0
            suit[i] = SUIT_JOKERS.get(key, -1)
            hand[i] = HAND_JOKERS.get(key, -1)
            for r in RANK_FIT.get(key, []):
                rfit[i, r] = 1.0
            for x in SUIT_FIT.get(key, []):
                sfit[i, x] = 1.0
            for e in ENH_FIT.get(key, []):
                efit[i, ENHANCEMENTS.index(e) - 1] = 1.0
            kind[i, 0] = float(key in COPY)
            for k, (c_, s_, _) in enumerate(PLANS.values()):
                core[i, k] = float(key in c_)
                supp[i, k] = float(key in s_ and key not in c_)
            for r, s in enumerate(ROLE_SETS):
                roles[i, r] = float(key in s)
        elif name.startswith("v_"):
            v0[i] = G.VOUCHERS.get(key, 3)
        elif name.startswith("c_"):
            kind[i, 3] = 1.0
            if key in PLANETS:
                planet[i] = PLANETS[key]
                v0[i] = 3
            else:
                v0[i] = G.TAROT_PICK.get(key, G.SPECTRAL_PICK.get(key, 2))
        elif name.startswith("p_"):
            kind[i, 1] = float("buffoon" in key)
            kind[i, 2] = float("celestial" in key)
            v0[i] = {"buffoon": 6, "celestial": 4, "arcana": 3, "spectral": 3, "standard": 1}.get(key.split("_")[0], 2) + (1 if "jumbo" in key else 2 if "mega" in key else 0)
        elif name.startswith("tag_"):
            v0[i] = {"investment": 6, "economy": 4, "rare": 6, "uncommon": 4, "polychrome": 5, "negative": 6, "holo": 3, "foil": 2, "double": 3, "voucher": 4,
                     "coupon": 4, "speed": 3, "handy": 2, "garbage": 2, "juggle": 1, "d_six": 2, "top_up": 2, "orbital": 3, "meteor": 3, "charm": 3, "buffoon": 4,
                     "ethereal": 2, "standard": 1, "boss": 1}.get(key, 2)
    return v0, roles, planet, suit, hand, (rfit, sfit, efit, kind, core, supp)


class HeurNet:
    """The weights in one flat vector: the item values first, then the named weights (NAMES, START)."""

    def __init__(self):
        self.v0, roles, planet, suit, hand, extra = _item_tables()
        self.extra_np = extra
        self.killer_np = np.array([float(b in G.KILLERS) for b in BOSS_KEYS], dtype=np.float32)
        self.NV = len(VOCAB)
        names, start = [], []

        def w(n, x):
            names.append(n); start.append(float(x))
        for r, nm in enumerate(ROLES):
            w(f"role {nm}: first of it", {"chips": 3, "mult": 2, "xmult": 0, "scaler": 3, "econ": 3, "narrow": 0, "scoring": 4, "slow scaler": 0, "decaying": 0}[nm])
            w(f"role {nm}: each more", {"chips": -2, "mult": 0, "xmult": -2, "scaler": 0, "econ": -1, "narrow": 0, "scoring": 1, "slow scaler": 0, "decaying": 0}[nm])
            w(f"role {nm}: per ante", {"chips": -0.3, "mult": 0, "xmult": 0.3, "scaler": -0.2, "econ": -0.5, "narrow": -0.3, "scoring": 0, "slow scaler": -0.5, "decaying": -0.5}[nm])
        w("econ joker while no scoring joker is held", -3)
        w("power gap x scoring joker", 2); w("power gap x xmult joker", 2); w("power gap x econ joker", -2); w("power gap x pack", -1)
        w("power gap x reroll", 0.5); w("power gap x leave", -1); w("power gap x skip", -3)
        w("suit joker: same suit already held", 3); w("suit joker: another suit held", -3)
        w("hand joker: boosts the main hand", 3); w("hand joker: boosts another hand", -2)
        w("debt: per $ below zero after buying", 0.5)
        w("money cushion: per $ below $5 after buying", 0.3)
        w("voucher: per ante", -0.3)
        w("copy joker: per scoring joker held", 1.5)
        w("xmult joker: per scoring joker held beyond one", 1.0)
        w("safety x skip", 0.0); w("safety x econ joker", 0.0)
        w("killer boss ahead x scoring joker", 1.0); w("killer boss ahead x reroll boss", 2.0)
        w("buffoon pack: a joker slot is free", 2); w("buffoon pack: no joker slot free", -3)
        w("celestial pack: main hand known", 1); w("pack: per ante", -0.2)
        w("consumable: no free consumable slot", -3)
        w("reroll: per point of the best buy on offer", 0.5)
        w("deck fit: rank joker (per 10% above a normal deck)", 1.0)
        w("deck fit: suit joker (per 10% above a normal deck)", 1.5)
        w("deck fit: enhancement joker (per 10% of the deck)", 1.0)
        # 1. long-term value: the change the action makes to the build's projected power at the ante-8 boss (env.projected_power), in log units
        w("projected power: per unit of log gain", 2.0)
        # 2. the power schedule: the log projected power a build should have by each ante (required = intercept + slope x ante), and how being behind it changes things
        w("schedule: required log power at ante 0", -8.0); w("schedule: required log power per ante", 1.0)
        w("schedule deficit x projected power gain", 1.0); w("schedule deficit x econ joker", -1.0); w("schedule deficit x leave", -0.5)
        w("schedule deficit x interest given up", -0.5)
        # 3. build plans: the committed plan's pieces, its hand, saving for a missing core piece, off-plan jokers
        w("plan: core piece of the committed plan", 4); w("plan: supporting piece of the committed plan", 2); w("plan: joker off the committed plan", -1)
        w("plan: planet for the plan's hand", 2); w("plan: no plan yet x core piece of any plan", 1.5)
        w("plan: save for a missing core piece (leave, per $10 held)", 0.5); w("plan: sell a joker off the committed plan", 1.0)
        w("joker: per empty joker slot", 1.5)
        # 4. the boss of this ante (fast_power against its target and hands, a 1.5x margin as the measure is optimistic): a skip gives up a shop before it
        w("boss gap x skip", -3.0); w("boss gap x scoring joker", 1.5); w("boss gap x xmult joker", 1.5); w("boss gap x leave", -1.0)
        w("skip: no scoring joker held", -4.0)
        # 5. fast_power is optimistic (the best play every hand): the gaps open below a margin, in log units (0.7: about 2x the target)
        w("power gap: margin (log)", 0.7); w("boss gap: margin (log)", 0.4)
        # 6. economy jokers in a full board that is behind the boss; flat jokers given up for a multiplier
        w("boss gap x econ joker", -1.5); w("sell an econ joker: slots full x boss gap", 1.0)
        w("replace a flat joker with an xmult joker: per ante from 3", 1.0)
        w("planet: main hand", 3); w("planet: per level of its hand", 0.2); w("planet: other hand", -1)
        for e in EDITIONS[1:]:
            w(f"edition {e.lower()}", {"FOIL": 2, "HOLO": 3, "POLYCHROME": 5, "NEGATIVE": 7}[e])
        w("eternal", -3); w("perishable", -3); w("rental", -6); w("rarity", 1)
        w("price", 0.7); w("interest given up", 1.5); w("price x ante", 0.0)
        w("value kept when selling (x item value)", 1.0)
        for b in BIASES:
            w(f"bias {b}", {"skip": -4, "reroll": -2, "reroll_boss": -2, "swap": -1, "sell_cons": -1, "use": 0, "pack_skip": 0}.get(b, 0))
        w("leave: per $10 held", 0.0); w("reroll: per $10 held", 0.3); w("reroll: no xmult held", 1.0); w("skip: per ante", -0.5)
        self.names, self.start = names, np.asarray(start, dtype=np.float32)
        self.n = self.NV + len(names)
        self.idx = {n: self.NV + i for i, n in enumerate(names)}
        self.roles_np, self.planet_np, self.suit_np, self.hand_np = roles, planet, suit, hand
        self.tag_ids_np = np.array([VOCAB.index(f"tag_{t}") for t in TAGS], dtype=np.int64)
        self._dev = None

    def init(self, seed=0):
        return torch.from_numpy(np.concatenate([self.v0, self.start]))

    def from_checkpoint(self, path):
        """A checkpoint's weights in this layout: the item values as they are, the named weights matched by name (a checkpoint without names: the
        layout before the power-gap / suit / hand / debt rules); weights the checkpoint lacks keep their starting values."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        old = ck["mean"].float()
        names = ck.get("names") or [n for n in self.names if n not in NEW_IN_V2]
        out = self.init().clone()
        out[:self.NV] = old[:self.NV]
        pos = {n: i for i, n in enumerate(names)}
        for n, k in self.idx.items():
            if n in pos:
                out[k] = old[self.NV + pos[n]]
        return out

    def _to(self, dev):
        if self._dev != dev:
            self.roles = torch.from_numpy(self.roles_np).to(dev)
            self.planet = torch.from_numpy(self.planet_np).to(dev)
            self.tag_ids = torch.from_numpy(self.tag_ids_np).to(dev)
            self.suit = torch.from_numpy(self.suit_np).to(dev)
            self.hand = torch.from_numpy(self.hand_np).to(dev)
            self.rfit, self.sfit, self.efit, self.kind, self.core, self.supp = (torch.from_numpy(t).to(dev) for t in self.extra_np)
            self.plan_hand = torch.tensor([h for _, _, h in PLANS.values()], device=dev)
            self.killer = torch.from_numpy(self.killer_np).to(dev)
            self._dev = dev

    def forward(self, W, g, jf, jid, cid, af, aid, mask, power=None, dpp=None, lpp=None, plan=None, missing=None, bpower=None):
        """power [P,M] fast_power share of the coming blind; dpp [P,M,NA] the log change in projected power each action makes (0: not measured);
        lpp [P,M] log projected power now; plan [P,M,NPLAN] the committed plan (one-hot, all zero: none); missing [P,M] 1 when the plan lacks a core piece."""
        """W [P, n]; observations batched [P, M, ...] (as evo.ShopNet.forward) -> scores [P, M, NA] (-1e9 where illegal)."""
        W = W.float()
        dev = W.device
        self._to(dev)
        P, M = g.shape[:2]
        x = lambda n: W[:, self.idx[n]][:, None, None]                     # one named weight as [P, 1, 1]
        V = W[:, :self.NV]
        aid = aid.long()
        jid = jid.long()
        ante = (g[..., G_ANTE] * 8)[..., None]                             # [P, M, 1]
        money = (g[..., G_MONEY] * 50)[..., None]
        held = self.roles[jid].sum(2)                                      # [P, M, 7] jokers held per role (the last: all jokers)
        levels = g[..., G_LEVELS:G_LEVELS + N_HANDS] * 10
        played = g[..., G_PLAYED:G_PLAYED + N_HANDS] * 30
        main = torch.argmax(played + 0.01 * levels, -1)                    # [P, M]
        known = (played.sum(-1) > 0).float()[..., None]                    # (no hand played yet: no main hand)
        # the power gap: how far the build is from beating the coming blind with all its hands (env.fast_power), as -log(share), 0 when it would clear
        gap = torch.zeros(P, M, 1, device=dev) if power is None else (-torch.log(power.clamp(min=0.05)).reshape(P, M, 1).float() + x("power gap: margin (log)")).clamp(min=0, max=3)
        bgap = torch.zeros(P, M, 1, device=dev) if bpower is None else (-torch.log((bpower / 1.5).clamp(min=0.05)).reshape(P, M, 1).float() + x("boss gap: margin (log)")).clamp(min=0, max=3)
        hs = self.suit[jid]                                                # [P, M, 8] the suit of each held suit joker (-1: none)
        safety = torch.zeros(P, M, 1, device=dev) if power is None else torch.log(power.clamp(min=0.05)).clamp(min=0, max=3).reshape(P, M, 1).float()
        killer = (g[..., 25:25 + len(BOSS_KEYS)] * self.killer).sum(-1, keepdim=True)          # the coming boss is one that wrecks builds
        rc = g[..., G_RC:G_RC + 13] * 4
        rshare = rc / rc.sum(-1, keepdim=True).clamp(min=1)                 # [P, M, 13]
        sc = g[..., G_SC:G_SC + 4] * 13
        sshare = sc / sc.sum(-1, keepdim=True).clamp(min=1)
        eshare = g[..., G_ENH:G_ENH + len(ENHANCEMENTS) - 1] * 10 / (g[..., G_FULL:G_FULL + 1] * 52).clamp(min=1)
        cons_free = ((g[..., G_CONS] - g[..., G_CONS + 1]) * 3 > 0.5).float()[..., None]
        zero = torch.zeros(P, M, 1, device=dev)
        lpp_ = zero if lpp is None else lpp.reshape(P, M, 1).float()
        required = x("schedule: required log power at ante 0") + x("schedule: required log power per ante") * ante
        deficit = zero if lpp is None else (required - lpp_).clamp(-3, 3)
        plan_ = torch.zeros(P, M, NPLAN, device=dev) if plan is None else plan.reshape(P, M, NPLAN).float()
        has_plan = plan_.sum(-1, keepdim=True)
        missing_ = zero if missing is None else missing.reshape(P, M, 1).float()

        def value(ids, counts):
            """The value of holding/using item ids [P, M, k] given role counts [P, M, 6] (held, not counting the item itself)."""
            v = torch.gather(V[:, None, :].expand(P, M, self.NV), 2, ids)
            rl = self.roles[ids]                                            # [P, M, k, 7]
            v = v + rl[..., -1] * x("joker: per empty joker slot") * (5 - counts[..., -1:]).clamp(min=0)
            no_scoring = (counts[..., ROLES.index("scoring")] < 0.5).float()[..., None]
            v = v + rl[..., ROLES.index("econ")] * no_scoring * x("econ joker while no scoring joker is held")
            v = v + gap * (rl[..., ROLES.index("scoring")] * x("power gap x scoring joker") + rl[..., ROLES.index("xmult")] * x("power gap x xmult joker")
                           + rl[..., ROLES.index("econ")] * x("power gap x econ joker"))
            v = v + bgap * (rl[..., ROLES.index("scoring")] * x("boss gap x scoring joker") + rl[..., ROLES.index("xmult")] * x("boss gap x xmult joker")
                            + rl[..., ROLES.index("econ")] * x("boss gap x econ joker"))
            su = self.suit[ids]                                             # [P, M, k]
            same = ((hs[:, :, None, :] == su[..., None]) & (su[..., None] >= 0)).float().sum(-1)
            other = ((hs[:, :, None, :] != su[..., None]) & (hs[:, :, None, :] >= 0) & (su[..., None] >= 0)).float().sum(-1)
            v = v + (same > 0.5).float() * x("suit joker: same suit already held") + (other > 0.5).float() * x("suit joker: another suit held")
            hd = self.hand[ids]
            ish = (hd >= 0).float() * known
            on = (hd == main[..., None]).float()
            v = v + ish * (on * x("hand joker: boosts the main hand") + (1 - on) * x("hand joker: boosts another hand"))
            n_sc = counts[..., ROLES.index("scoring"):ROLES.index("scoring") + 1]
            kd = self.kind[ids]                                             # [P, M, k, 5]
            v = v + kd[..., 0] * x("copy joker: per scoring joker held") * n_sc
            v = v + rl[..., ROLES.index("xmult")] * x("xmult joker: per scoring joker held beyond one") * (n_sc - 1).clamp(min=0)
            v = v + rl[..., ROLES.index("econ")] * x("safety x econ joker") * safety
            v = v + rl[..., ROLES.index("scoring")] * x("killer boss ahead x scoring joker") * killer
            free = (5 - counts[..., -1:] > 0.5).float()
            v = v + kd[..., 1] * (free * x("buffoon pack: a joker slot is free") + (1 - free) * x("buffoon pack: no joker slot free"))
            v = v + kd[..., 2] * known * x("celestial pack: main hand known")
            rf = self.rfit[ids]                                             # [P, M, k, 13]
            nr = rf.sum(-1)
            fit_r = ((rf * rshare[:, :, None, :]).sum(-1) - nr / 13) * 10
            v = v + (nr > 0).float() * fit_r * x("deck fit: rank joker (per 10% above a normal deck)")
            sf = self.sfit[ids]
            ns = sf.sum(-1)
            fit_s = ((sf * sshare[:, :, None, :]).sum(-1) - ns / 4) * 10
            v = v + (ns > 0).float() * fit_s * x("deck fit: suit joker (per 10% above a normal deck)")
            ef = self.efit[ids]
            v = v + (ef * eshare[:, :, None, :]).sum(-1) * 10 * x("deck fit: enhancement joker (per 10% of the deck)")
            v = v + rl[..., ROLES.index("econ")] * x("schedule deficit x econ joker") * deficit
            pc = (self.core[ids] * plan_[:, :, None, :]).sum(-1)            # a core piece of the committed plan
            ps = (self.supp[ids] * plan_[:, :, None, :]).sum(-1)
            isj = rl[..., -1]
            off = isj * has_plan * (1 - pc.clamp(max=1)) * (1 - ps.clamp(max=1))
            v = v + pc * x("plan: core piece of the committed plan") + ps * x("plan: supporting piece of the committed plan") + off * x("plan: joker off the committed plan")
            v = v + (1 - has_plan) * self.core[ids].amax(-1) * x("plan: no plan yet x core piece of any plan")
            for r, nm in enumerate(ROLES):
                first = (counts[..., r] < 0.5).float()[..., None]
                v = v + rl[..., r] * (x(f"role {nm}: first of it") * first + x(f"role {nm}: each more") * (1 - first) + x(f"role {nm}: per ante") * ante)
            ph = self.planet[ids]                                           # the hand a planet levels
            isp = (ph >= 0).float()
            phc = ph.clamp(min=0)
            lv = torch.gather(levels, 2, phc)
            on_main = (phc == main[..., None]).float()
            v = v + isp * (x("planet: main hand") * on_main + x("planet: other hand") * (1 - on_main) + x("planet: per level of its hand") * lv)
            ph_plan = (plan_ * self.plan_hand.clamp(min=0).float()).sum(-1, keepdim=True)          # the plan's hand (gated by plan_has_hand)
            plan_has_hand = (plan_ * (self.plan_hand >= 0).float()).sum(-1, keepdim=True)
            v = v + isp * plan_has_hand * (phc == ph_plan.long()).float() * x("plan: planet for the plan's hand")
            return v

        gain = value(aid, held)                                             # [P, M, NA]
        ed = af[..., C_ED + 1:C_ED + len(EDITIONS)]
        for k, e in enumerate(EDITIONS[1:]):
            gain = gain + ed[..., k] * x(f"edition {e.lower()}")
        gain = gain + af[..., C_ETERNAL] * x("eternal") + (af[..., C_PERISH] > 0).float() * x("perishable") + af[..., C_RENTAL] * x("rental") + af[..., C_RARITY] * 3 * x("rarity")
        cost = af[..., C_COST] * 10                                         # (negative for a sale)
        after = money - cost.clamp(min=0)
        interest = ((money.clamp(0, 25) // 5) - (after.clamp(0, 25) // 5)).clamp(min=0)
        score = -cost * (x("price") + x("price x ante") * ante) - interest * (x("interest given up") + x("schedule deficit x interest given up") * deficit)
        if dpp is not None:
            score = score + dpp.reshape(P, M, NA).float() * (x("projected power: per unit of log gain") + x("schedule deficit x projected power gain") * deficit)
        score = score - (-(money - cost)).clamp(min=0) * (cost > 0).float() * x("debt: per $ below zero after buying")
        score = score - (5 - (money - cost)).clamp(min=0, max=5) * (cost > 0).float() * x("money cushion: per $ below $5 after buying")
        is_cons = self.kind[aid][..., 3]
        buyrows = torch.zeros(NA, device=dev)
        buyrows[ROWS["buy"]] = 1.0
        score = score + is_cons * buyrows * (1 - cons_free) * x("consumable: no free consumable slot")
        type_gain = torch.zeros(NA, device=dev)
        for b in ("skip", "buy", "pack", "voucher", "use", "pick", "replace"):
            type_gain[ROWS[b]] = 1.0
        score = score + gain * type_gain
        sk = ROWS["skip"][0]                                                # a skip gains the tag offered (the state's one-hot; the action row has no item)
        score[..., sk] = score[..., sk] + (g[..., G_TAGS:G_TAGS + len(TAGS)] * V[:, self.tag_ids][:, None, :]).sum(-1)
        # what a sale or a replace gives up: the held joker's value (its role counts without itself)
        lost = torch.zeros(P, M, NA, device=dev)
        sj = ROWS["sell_joker"]
        own = jid
        own_v = torch.stack([value(own[..., j:j + 1], held - self.roles[own[..., j]])[..., 0] for j in range(MAX_JOKERS)], -1)   # [P, M, 8]
        lost[..., sj] = own_v
        rep = ROWS["replace"]
        lost[..., rep] = own_v.repeat(1, 1, len(rep) // MAX_JOKERS)        # replace row = shop slot i * 8 + joker j
        score = score - lost * x("value kept when selling (x item value)")       # (a sale's money is in the price term: its price is negative)
        bias = torch.zeros(P, M, NA, device=dev)
        for b in BIASES:
            bias[..., ROWS[b]] = x(f"bias {b}")
        has_x = (held[..., 2] > 0.5).float()[..., None]
        bias[..., ROWS["leave"]] += x("leave: per $10 held") * money / 10
        bias[..., ROWS["reroll"]] += x("reroll: per $10 held") * money / 10 + x("reroll: no xmult held") * (1 - has_x)
        bias[..., ROWS["skip"]] += x("skip: per ante") * ante + x("power gap x skip") * gap + x("boss gap x skip") * bgap
        no_sc = (held[..., ROLES.index("scoring")] < 0.5).float()[..., None]
        bias[..., ROWS["skip"]] += x("skip: no scoring joker held") * no_sc
        bias[..., ROWS["leave"]] += x("boss gap x leave") * bgap
        bias[..., ROWS["pack"]] += x("power gap x pack") * gap
        bias[..., ROWS["reroll"]] += x("power gap x reroll") * gap
        bias[..., ROWS["leave"]] += x("power gap x leave") * gap
        bias[..., ROWS["skip"]] += x("safety x skip") * safety
        bias[..., ROWS["reroll_boss"]] += x("killer boss ahead x reroll boss") * killer
        bias[..., ROWS["voucher"]] += x("voucher: per ante") * ante
        bias[..., ROWS["pack"]] += x("pack: per ante") * ante
        bias[..., ROWS["leave"]] += x("schedule deficit x leave") * deficit + x("plan: save for a missing core piece (leave, per $10 held)") * has_plan * missing_ * money / 10
        off_held = has_plan * (1 - ((self.core[jid] + self.supp[jid]) * plan_[:, :, None, :]).sum(-1).clamp(max=1)) * (jid > 0).float()     # [P, M, 8]
        bias[..., ROWS["sell_joker"]] += x("plan: sell a joker off the committed plan") * off_held
        full = (held[..., -1:] >= 5 - 0.5).float()                                                      # [P, M, 1] the board is full
        econ_held = self.roles[jid][..., ROLES.index("econ")]                                            # [P, M, 8]
        bias[..., ROWS["sell_joker"]] += x("sell an econ joker: slots full x boss gap") * econ_held * full * bgap
        # replace row = shop slot i * 8 + joker j: the bought item is a multiplier and the joker given up is flat (chips / +mult, not a multiplier or scaler)
        rep = ROWS["replace"]
        flat_held = ((self.roles[jid][..., ROLES.index("chips")] + self.roles[jid][..., ROLES.index("mult")]) > 0.5).float()             * (1 - self.roles[jid][..., ROLES.index("xmult")]) * (1 - self.roles[jid][..., ROLES.index("scaler")])                   # [P, M, 8]
        buys_x = self.roles[aid[..., rep]][..., ROLES.index("xmult")]                                    # [P, M, len(rep)]
        bias[..., rep] += x("replace a flat joker with an xmult joker: per ante from 3") * (ante - 2).clamp(min=0) * buys_x * flat_held.repeat(1, 1, len(rep) // MAX_JOKERS)
        score = score + bias
        offer = torch.zeros(NA, dtype=torch.bool, device=dev)
        for b in ("buy", "pack", "voucher"):
            offer[ROWS[b]] = True
        best_buy = score.masked_fill(~(mask & offer), -1e9).max(-1).values.clamp(min=0)          # the best purchase on offer (0: none worth it)
        rr = ROWS["reroll"][0]
        score[..., rr] = score[..., rr] - x("reroll: per point of the best buy on offer")[..., 0] * best_buy
        return score.masked_fill(~mask, -1e9)


class SolverEvaluator:
    """Like evo.Evaluator (P candidates x the same M exact-seed games in one vector environment), but the rounds are played by the exact round solver's
    best line, as azc plays them (evo's scripted round player is much weaker: every shop policy tuned with it stalled near 6 blinds on Gold)."""

    def __init__(self, net, P, M, deck="RED", stake="GOLD", threads=0, dev="cuda", win_bonus=10.0, node_cap=150, max_steps=4000,
                 project=True, pp_samples=2, max_cands=12, scratch=8192, escalate=(10, 100), honest=0):
        from .engine_env import CppVecEnv
        self.project, self.pp_samples, self.max_cands = project, pp_samples, max_cands
        self.escalate = tuple(int(m) for m in escalate)
        self.honest = int(honest)                        # > 0: the honest round player (no foresight): this many hidden-future copies vote on every move
        if self.honest:
            self.hs = CppVecEnv(scratch, deck, stake, seed0=2, threads=threads)
            self.hs.set_exact(True)
            self.hs.v.restart()
            self.hs.set_auto_round(False, 1.5, 0)
            self.salt = 0
        if project:
            self.scr = CppVecEnv(scratch, deck, stake, seed0=1, threads=threads)
            self.scr.set_exact(True)
            self.scr.v.restart()
            self.scr.set_auto_round(False, 1.5, 0)
        from .sim.jokers import JOKERS
        jk = list(JOKERS)
        core = net.extra_np[4]
        supp = net.extra_np[5]
        self.core_np, self.supp_np = core, supp
        # which rows' actions change the build (their projected power is measured): buying a joker or planet, replacing, selling a joker, picking, using a planet
        self.proj_rows = np.zeros(NA, dtype=bool)
        for b in ("buy", "replace", "sell_joker", "pick", "use"):
            self.proj_rows[ROWS[b]] = True
        self.build_item = np.zeros(len(VOCAB), dtype=bool)
        for i, n in enumerate(VOCAB):
            self.build_item[i] = n.startswith("j_") or (n.startswith("c_") and n[2:] in PLANETS)
        self.net, self.P, self.M, self.dev, self.win_bonus, self.node_cap, self.max_steps = net, P, M, dev, win_bonus, node_cap, max_steps
        self.deck, self.stake = deck, stake
        self.N = P * M
        self.v = CppVecEnv(self.N, deck, stake, seed0=1, threads=threads)
        self.v.set_exact(True)
        if SOLVER_OPTS.get("smart"):                     # (engine globals: the round solver's ordering and stepping)
            self.v.v.set_solver_order(True)
        if SOLVER_OPTS.get("lite"):
            self.v.v.set_solver_speed(True, False, 0.0)
        if SOLVER_OPTS.get("tt"):
            self.v.v.set_solver_tt(True)
        self.v.set_auto_round(False, 1.5, 0)
        self.src = [i % M for i in range(self.N)]
        self.dst = list(range(self.N))

    def _roots(self, seed0):
        from .engine_env import CppVecEnv
        r = CppVecEnv(self.M, self.deck, self.stake, seed0=seed0, threads=1)
        r.set_exact(True)
        r.v.restart()
        return r

    def _projection(self, v, idx, mask, aid):
        """For the shop games idx: log projected power now [len(idx)], and the log change each build-changing action makes [len(idx), NA]."""
        pp_now = v.projected_power(list(idx), self.pp_samples)
        l_now = np.log(np.maximum(pp_now, 1e-4))
        d = np.zeros((len(idx), NA), dtype=np.float32)
        jobs = []
        for r, i in enumerate(idx):
            rows = np.flatnonzero(mask[i, A0:] & self.proj_rows)
            rows = [x for x in rows if self.build_item[aid[i, A0 + x]] or x in ROWS["sell_joker"] or x >= ROWS["replace"][0]]
            for x in rows[:self.max_cands]:
                jobs.append((r, i, x))
        K = self.scr.n
        for c0 in range(0, len(jobs), K):
            ch = jobs[c0:c0 + K]
            self.scr.copy_many(v, [i for _, i, _ in ch], list(range(len(ch))))
            act = np.full(K, -1, dtype=np.int64)
            for k, (_, _, x) in enumerate(ch):
                act[k] = x + A0
            self.scr.step(act)
            pp = self.scr.projected_power(list(range(len(ch))), self.pp_samples)
            for k, (r, _, x) in enumerate(ch):
                d[r, x] = np.log(max(pp[k], 1e-4)) - l_now[r]
        return l_now, np.clip(d, -4, 4)

    def _honest_moves(self, v, ids, mask, af):
        """The next move of each in-round game ids, without foresight: K copies with the future hidden (Env::hide_future: the draw pile reshuffled, the
        rolls redrawn) are solved; the first moves of their clearing lines vote; no copy clears: the best predicted play."""
        from collections import Counter
        from .env import N_SUB, RATIO_COL
        K, H = self.honest, self.hs
        per = max(1, H.n // K)
        out = np.zeros(len(ids), dtype=np.int64)
        for c0 in range(0, len(ids), per):
            ch = ids[c0:c0 + per]
            src = [int(i) for i in ch for _ in range(K)]
            dst = list(range(len(src)))
            H.copy_many(v, src, dst)
            self.salt += 1
            H.v.hide_future(dst, [self.salt * 1_000_003 + 7919 * d + 11 for d in dst])
            sols = H.solve_rounds(dst, 1, self.node_cap)
            for r, i in enumerate(ch):
                votes = Counter(int(p[0][0]) for p in sols[r * K:(r + 1) * K] if p and p[0])
                a = votes.most_common(1)[0][0] if votes else -1
                if a < 0 or not mask[i, a]:
                    sc = np.where(mask[i, :N_SUB], af[i, :N_SUB, RATIO_COL + 1], -1e9)
                    a = int(np.argmax(sc)) if sc.max() > -1e8 else int(np.argmax(mask[i]))
                out[c0 + r] = a
        return out

    def _plans(self, jid, idx):
        """Commit each shop game to a build plan: progress = 2 x core pieces + supporting pieces held; commit at 2, switch when another plan leads by 2."""
        prog = 2 * self.core_np[jid[idx]].sum(1) + self.supp_np[jid[idx]].sum(1)            # [len(idx), NPLAN]
        for r, i in enumerate(idx):
            best = int(np.argmax(prog[r]))
            cur = self.plan[i]
            if cur < 0:
                if prog[r, best] >= 2:
                    self.plan[i] = best
            elif prog[r, best] >= prog[r, cur] + 2:
                self.plan[i] = best
        oh = np.zeros((len(jid), NPLAN), dtype=np.float32)
        has = self.plan >= 0
        oh[np.flatnonzero(has), self.plan[has]] = 1.0
        held_core = np.zeros(len(jid), dtype=np.float32)
        if has.any():
            hc = self.core_np[jid].sum(1)                                   # [N, NPLAN]
            held_core[has] = hc[np.flatnonzero(has), self.plan[has]]
        missing = has & (held_core < 2)                                      # the plan wants (at least) two core pieces
        return oh, missing.astype(np.float32)

    @torch.no_grad()
    def run(self, W, seed0, recorder=None):
        """W [P, n] -> (fitness [P], blinds [P, M], won [P, M]). recorder(v, act, scores) is called before each step (for traces)."""
        from .env import F_ACT_W, F_JOKER_W
        v, N, P, M, dev = self.v, self.N, self.P, self.M, self.dev
        v.copy_many(self._roots(seed0), self.src, self.dst)
        done = np.zeros(N, dtype=bool)
        blinds = np.zeros(N)
        won = np.zeros(N, dtype=bool)
        queue = {}
        self.plan = -np.ones(N, dtype=np.int64)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        for _ in range(self.max_steps):
            o = v.obs
            mask = np.asarray(o["mask"])
            g = np.asarray(o["g"])
            act = np.full(N, -1, dtype=np.int64)
            inround = mask[:, :A0].any(1) & ~done
            shop = ~inround & ~done
            if self.honest:                                                  # every move of a round decided afresh, without seeing the future
                ids = np.flatnonzero(inround)
                if len(ids):
                    act[ids] = self._honest_moves(v, ids, mask, np.asarray(o["af"]))
                ids = []
            for i in [i for i in queue if done[i] or not inround[i] or g[i, 5] < 0.5]:
                del queue[i]                                                     # the round is over (or it is targeting): drop the rest of the line
            fresh = [] if self.honest else [i for i in np.flatnonzero(inround) if i not in queue and g[i, 5] > 0.5 and g[i, 9] == 0 and abs(g[i, 11] - g[i, 13]) < 1e-6]
            if fresh:
                sols = dict(zip(fresh, v.solve_rounds(fresh, 1, self.node_cap)))
                for mult in self.escalate:                                 # no clearing line within the budget: the same round again with a bigger one
                    hard = [i for i in fresh if not sols[i]]
                    if not hard:
                        break
                    sols.update(zip(hard, v.solve_rounds(hard, 1, self.node_cap * mult)))
                for i in fresh:
                    queue[i] = list(sols[i][0]) if sols[i] else []
            for i in ([] if self.honest else np.flatnonzero(inround)):
                q = queue.get(i)
                a = q.pop(0) if q else -1
                act[i] = a if a >= 0 and mask[i, a] else int(np.argmax(mask[i]))   # (no line: the run is lost; targeting: the first legal choice)
            if shop.any():
                power = np.ones(N)
                bpower = np.ones(N)
                sidx = np.flatnonzero(shop)
                fp = v.fast_power(list(sidx), 4)
                power[sidx] = fp[:, 0]
                if fp.shape[1] > 4:
                    bpower[sidx] = fp[:, 4]
                aid_all = np.asarray(o["aid"])
                jid_all = np.asarray(o["jid"])
                lpp = np.zeros(N, dtype=np.float32)
                dpp = np.zeros((N, NA), dtype=np.float32)
                if self.project:
                    lpp[sidx], dpp[sidx] = self._projection(v, sidx, mask, aid_all)
                plan_oh, missing = self._plans(jid_all, sidx)
                scores = self.net.forward(W, t(g).float().reshape(P, M, -1), t(np.asarray(o["jf"])[:, :, :F_JOKER_W]).float().reshape(P, M, MAX_JOKERS, F_JOKER_W),
                                          t(o["jid"]).reshape(P, M, -1), t(o["cid"]).reshape(P, M, -1), t(np.asarray(o["af"])[:, A0:, :F_ACT_W]).float().reshape(P, M, NA, F_ACT_W),
                                          t(np.asarray(o["aid"])[:, A0:]).reshape(P, M, NA), t(mask[:, A0:]).reshape(P, M, NA), power=t(power),
                                          dpp=t(dpp) if self.project else None, lpp=t(lpp) if self.project else None, plan=t(plan_oh), missing=t(missing), bpower=t(bpower)).reshape(N, NA)
                best = (scores.argmax(1).cpu().numpy() + A0).astype(np.int64)
                act = np.where(shop, best, act)
                if recorder is not None:
                    recorder(v, act, scores)
            _, d, info = v.step(act)
            first = d & ~done
            blinds[first], won[first] = info["blinds"][first], info["won"][first]
            done |= d
            if done.all():
                break
        if not done.all():
            blinds[~done] = info["blinds"][~done]
        fit = (blinds + self.win_bonus * won).reshape(P, M).mean(1)
        return torch.from_numpy(fit), blinds.reshape(P, M), won.reshape(P, M)


def describe(a: int, aid_row) -> str:
    from .env import _TYPE_OF
    k = "replace" if a >= A_REPLACE else ACTION_TYPES[_TYPE_OF[a]]
    item = VOCAB[int(aid_row[a])] if 0 <= int(aid_row[a]) < len(VOCAB) else ""
    return f"{k} {item if item not in ('<none>', '<unknown>') else ''}".strip()


def trace(a, net, w, n_games=20, top=3, out=None):
    """Plays n_games held-out Gold games with weights w and writes every decision outside a round: the state, the choice and the best alternatives with their scores."""
    from .sim.jokers import JOKERS
    K = list(JOKERS)
    ev = SolverEvaluator(net, 1, n_games, a.deck, a.stake, a.threads, a.device, a.win_bonus, a.node_cap)
    log = {i: [] for i in range(n_games)}
    done = np.zeros(n_games, dtype=bool)
    BL = ["small", "big", "boss"]

    def rec(v, act, scores):
        o = v.obs
        aid = np.asarray(o["aid"])
        mask = np.asarray(o["mask"])
        sc = scores.float().cpu().numpy()
        idx = [i for i in range(n_games) if act[i] >= A0 and not mask[i, :A0].any()]
        if not idx:
            return
        for i, inf in zip(idx, v.cell_info(idx)):
            legal = np.flatnonzero(mask[i, A0:])
            order = legal[np.argsort(-sc[i, legal])][:top + 1]
            jok = [K[inf[8 + 3 * j]] for j in range(inf[7])]
            alts = ", ".join(f"{describe(int(r) + A0, aid[i])} {sc[i, r]:+.1f}" for r in order[1:])
            log[i].append(f"ante {inf[0]} {BL[inf[1]]:5s} ${inf[3]:<4d} [{', '.join(jok)}]\n      -> {describe(int(act[i]), aid[i])} {sc[i, int(act[i]) - A0]:+.1f}"
                          + (f"   (next: {alts})" if alts else ""))
    _, b, wn = ev.run(w[None].float().to(a.device), seed0=a.eval_seed0, recorder=rec)
    b, wn = b[0], wn[0]
    lines = [f"{n_games} held-out Gold games: {b.mean():.2f} blinds, {int(wn.sum())} wins"]
    for i in range(n_games):
        lines.append(f"\n=== game {i} (seed {a.eval_seed0 + i}): {int(b[i])} blinds{' WON' if wn[i] else ''} ===")
        lines += ["  " + x for x in log[i]]
    text = "\n".join(lines)
    if out:
        open(out, "w", encoding="utf-8").write(text)
    return text


def show(path, net=None):
    net = net or HeurNet()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    w = ck["mean"].float().numpy()
    w0 = net.init().numpy()
    print(f"{path}: generation {ck.get('gen')}, {ck.get('eval')}")
    print("\nnamed weights (start -> tuned):")
    for i, n in enumerate(net.names):
        k = net.NV + i
        print(f"  {n:45s} {w0[k]:7.2f} -> {w[k]:7.2f}")
    d = w[:net.NV] - w0[:net.NV]
    order = np.argsort(-np.abs(d))[:40]
    print("\nitem values that moved most (start -> tuned):")
    for i in order:
        print(f"  {VOCAB[i]:28s} {w0[i]:6.2f} -> {w[i]:6.2f}")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default="full", choices=["full", "sep", "lm"])
    p.add_argument("--pop", type=int, default=32)
    p.add_argument("--games", type=int, default=32, help="games per candidate per generation (fresh seeds every generation; every candidate meets the same ones)")
    p.add_argument("--sigma", type=float, default=0.5, help="the first step size, in dollars")
    p.add_argument("--gens", type=int, default=100000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--win-bonus", type=float, default=10.0)
    p.add_argument("--pace", type=float, default=1.5)
    p.add_argument("--rounds", default="solver", choices=["solver", "scripted"], help="who plays the rounds: the exact solver (as azc) or evo's scripted player")
    p.add_argument("--node-cap", type=int, default=150)
    p.add_argument("--smart-order", action="store_true", help="the round solver tries set-up discards first where the greedy line cannot clear (ordering only)")
    p.add_argument("--solver-tt", action="store_true", help="the round solver skips positions it already reached by another order of moves")
    p.add_argument("--honest", type=int, default=0, help="the honest round player: every move of a round voted by this many solves of copies whose future is hidden "
                   "(no foresight); 0: the solver plans the round seeing the draws")
    p.add_argument("--solver-lite", action="store_true", help="the round solver's cheap steps (no full observation per node)")
    p.add_argument("--escalate", default="10,100", help="when the solver finds no clearing line, it tries the round again with the node cap times these, in turn ('' : never)")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=400, help="the held-out seeds (the same ones every time)")
    p.add_argument("--eval-seed0", type=int, default=9_000_000)
    p.add_argument("--out", default="checkpoints/heur1.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", default="", help="start from a checkpoint's weights, matched by name (weights it lacks keep their starting values)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--show", default="", help="print a checkpoint's weights and exit")
    p.add_argument("--trace", default="", help="write every decision (with the best alternatives and their scores) of --trace-games held-out games of a checkpoint ('start': the starting weights)")
    p.add_argument("--trace-games", type=int, default=20)
    p.add_argument("--trace-out", default="traces/heur_trace.txt")
    p.add_argument("--eval-only", default="", help="evaluate a checkpoint (or 'start' for the guide's starting weights) on the held-out seeds and exit")
    return p.parse_args(argv)


def make_eval(a, net, P, M, dev):
    SOLVER_OPTS.update(smart=bool(getattr(a, "smart_order", False)), lite=bool(getattr(a, "solver_lite", False)), tt=bool(getattr(a, "solver_tt", False)))
    if a.rounds == "solver":
        return SolverEvaluator(net, P, M, a.deck, a.stake, a.threads, dev, a.win_bonus, a.node_cap,
                               escalate=tuple(int(x) for x in a.escalate.split(",") if x), honest=getattr(a, "honest", 0))
    from .evo import Evaluator
    return Evaluator(net, P, M, a.deck, a.stake, a.pace, a.threads, dev, a.win_bonus)


def train(a):
    net = HeurNet()
    if a.show:
        show(a.show, net)
        return
    dev = a.device
    if a.trace:
        w = net.init() if a.trace == "start" else net.from_checkpoint(a.trace)
        print(trace(a, net, w, a.trace_games, out=a.trace_out).splitlines()[0], "->", a.trace_out)
        return
    if a.eval_only:
        w = net.init() if a.eval_only == "start" else net.from_checkpoint(a.eval_only)
        ev = make_eval(a, net, 1, a.eval_games, dev)
        t = time.time()
        _, b, wn = ev.run(w[None].float().to(dev), seed0=a.eval_seed0)
        print(json.dumps({"checkpoint": a.eval_only, "games": a.eval_games, "blinds": round(float(b.mean()), 3), "wins": int(wn.sum()), "secs": round(time.time() - t, 1)}))
        return
    log_path = a.out.replace(".pt", "_log.jsonl")
    g0 = 0
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location="cpu", weights_only=False)
        es = cma.make(a.algo, ck["mean"], a.sigma, a.pop, device=dev, seed=a.seed + ck["gen"])
        es.load_state_dict(ck["es"])
        g0 = ck["gen"]
    else:
        x0 = net.from_checkpoint(a.init_from) if a.init_from else net.init()
        es = cma.make(a.algo, x0, a.sigma, a.pop, device=dev, seed=a.seed)
    print(json.dumps({"weights": net.n, "named": len(net.names), "pop": es.lam, "games": a.games, "device": str(dev)}), flush=True)
    train_ev = make_eval(a, net, es.lam, a.games, dev)
    eval_ev = make_eval(a, net, 1, a.eval_games, dev)
    best_eval, t0 = -1.0, time.time()
    if g0 == 0 and a.eval_every:
        _, b_, w_ = eval_ev.run(x0[None].float().to(dev), seed0=a.eval_seed0)
        row = {"gen": 0, "eval": {"games": a.eval_games, "blinds": round(float(b_.mean()), 3), "wins": int(w_.sum())}, "note": f"start: {a.init_from or 'the guide'}"}
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    for gen in range(g0 + 1, a.gens + 1):
        tg = time.time()
        X = es.ask()
        fit, blinds, won = train_ev.run(X, seed0=a.seed * 10_000_000 + gen * 1000)
        es.tell((-fit).double().cpu())
        row = {"gen": gen, "best": round(float(fit.max()), 3), "mean": round(float(fit.mean()), 3), "blinds_mean": round(float(blinds.mean()), 3),
               "wins": int(won.sum()), "sigma": round(es.sigma, 4), "secs": round(time.time() - tg, 1), "min": round((time.time() - t0) / 60, 1)}
        if a.eval_every and gen % a.eval_every == 0:
            _, b_, w_ = eval_ev.run(es.mean[None].float(), seed0=a.eval_seed0)
            row["eval"] = {"games": a.eval_games, "blinds": round(float(b_.mean()), 3), "wins": int(w_.sum())}
            if row["eval"]["blinds"] > best_eval:
                best_eval = row["eval"]["blinds"]
                torch.save({"mean": es.mean.cpu(), "gen": gen, "eval": row["eval"], "names": net.names}, a.out.replace(".pt", "_best.pt"))
            torch.save({"mean": es.mean.cpu(), "es": es.state_dict(), "gen": gen, "names": net.names}, a.out)
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    train(parse())
