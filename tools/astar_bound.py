"""An optimistic bound for the best-first shop search: how well could THIS state's run end, whatever it does in the shops from here?

For a node (the game at a blind select) the bound builds the best FANTASY run the seed still allows and plays its ante-8 boss with the exact round solver:

    jokers     the node's own jokers plus every joker the seed's shops (the first `pairs` refreshes of each ante from the node's ante on) and buffoon packs still offer, minus the perishable /
               rental ones. A build of up to `slots` of them (the node's slots + `extra_slots` for negative editions), chosen by search over the exact scorer: each candidate alone, then every
               subset of the best `top`.
    counters   every scaler at the counter it would have at the ante-8 boss if it had been held since ante 1 (the engine's growth table: the most it can reach)
    levels     the node's hand levels + `per_ante` per ante left on EVERY hand type (more planets than any run can buy)
    money      not modelled (a build costs nothing): the bound is optimistic in money

The result is the best share of the boss's chip requirement any play reaches (>= 1: the boss can be beaten). A node whose share times `slack` is below 1 cannot win and is pruned.

What makes it only approximately admissible, and what is done about it:
  * the shops show more than `pairs` refreshes if rerolled for long (a joker that only appears on reroll 9 is not in the pool): `pairs` is a parameter; the audit (astar_shop --audit) checks the bound
    along lines that did win, and `slack` pads the rest;
  * the build search is exhaustive only over the top `top` candidates: the padded slack covers what it misses;
  * the ante-8 boss is the one the engine draws when it builds the fantasy state (the exact seed's boss is the same stream, but its position can differ).
The bound is cached by (jokers, hand levels, ante): nodes that differ only in money or the stream position share it.
"""
import itertools
import os
import sys

import numpy as np

os.environ.setdefault("BALATRO_DEV_ENGINE", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import make                                                      # noqa: E402
from balatro_rl.env import A_SELECT, MAX_JOKERS, item_id                                   # noqa: E402
from balatro_rl.potential import GROW                                            # noqa: E402
from balatro_rl.sim.game import Game                                             # noqa: E402
from balatro_rl.sim.jokers import JOKERS                                         # noqa: E402
from balatro_rl.sim.seedrng import seed_string                                   # noqa: E402

KEYS = list(JOKERS)
ITEM2KEY = {item_id("j_" + k): k for k in KEYS}


def counter_max(key):
    """The counter a scaler can have at the ante-8 boss (held since ante 1)."""
    g = GROW.get(key)
    if g is None:
        return float("nan")
    start, rate, lo, hi = g
    return float(min(max(start + rate * (3 * 7 + 2), lo), hi))


def obtainable(seed, stake, deck="RED", antes=8, pairs=8):
    """{joker key: first ante it can be bought}: the shops' first `pairs` refreshes and the jokers in buffoon packs; perishable / rental ones left out."""
    g = Game(seed=1, deck_type=deck, stake=stake, exact_seed=seed_string(seed))
    g.money = 10_000
    out = {}
    for ante in range(1, antes + 1):
        if ante > 1:
            g.ante = ante
            g.new_ante()
        g.shop, g.shop_packs, g.shop_vouchers, g.ante_voucher = [], [], [], None
        g.open_shop()
        c = g.clone()
        c.money = 10_000
        for _ in range(len(c.shop_packs)):                                        # buffoon packs: a joker from a pack costs only the pack
            try:
                c.buy_pack(0)
            except Exception:
                break
            for x in getattr(c, "pack_cards", []):
                k = getattr(x, "key", None)
                name = str(k() if callable(k) else (k or getattr(x, "name", ""))).replace("j_", "")
                if name in JOKERS and name not in out:
                    out[name] = ante
            try:
                c.pack_skip()
            except Exception:
                break
        for k in range(pairs):
            if k:
                g.free_rerolls = 1
                g.reroll()
            for it in g.shop:
                if it.kind != "joker" or it.joker is None:
                    continue
                j = it.joker
                if getattr(j, "perishable", None) not in (None, False, 0) or getattr(j, "rental", False):
                    continue
                name = it.key[2:]
                if name in JOKERS and name not in out:
                    out[name] = ante
    return out


class Bound:
    def __init__(self, seed, stake="GOLD", deck="RED", pairs=8, top=8, extra_slots=1, per_ante=3, slack=1.25, cap=1024, node_cap=300):
        self.seed, self.stake, self.deck = seed, stake, deck
        self.top, self.extra_slots, self.per_ante, self.slack, self.node_cap = top, extra_slots, per_ante, slack, node_cap
        self.first = obtainable(seed, stake, deck, pairs=pairs)
        self.scratch = make(cap, deck, stake, seed, 0)
        self.cap = cap
        self.cache = {}
        self.stats = dict(calls=0, cache_hits=0, evals=0)

    def _shares(self, node_state, builds, levels):
        """builds: lists of joker keys. -> the best share of the ante-8 boss's requirement reached with each, in the node's deck."""
        out = []
        for c0 in range(0, len(builds), self.cap):
            ch = builds[c0:c0 + self.cap]
            self.scratch.copy_many(node_state[0], [node_state[1]] * len(ch), list(range(len(ch))))
            for i, b in enumerate(ch):
                self.scratch.set_build([i], 8, 2, [KEYS.index(k) for k in b], [counter_max(k) for k in b], 0, levels, 25)
            self.scratch.set_auto_round(False, 1.5, 0)
            act = np.full(self.scratch.n, -1, dtype=np.int64)
            act[:len(ch)] = A_SELECT
            self.scratch.step(act)                                                # the boss blind begins: hand dealt, the solver can play it
            r = self.scratch.solve_best(list(range(len(ch))), self.node_cap)
            out += [float(x) for x in r[:, 0]]
            self.stats["evals"] += len(ch)
        return out

    def share(self, env, slot, ante):
        """The optimistic best share (>= 1: possibly winnable) for the node at `slot` of VecEnv `env`, at the given ante (1..8)."""
        self.stats["calls"] += 1
        jid = np.asarray(env.obs["jid"])[slot]
        owned = sorted({ITEM2KEY[int(x)] for x in jid if int(x) in ITEM2KEY})
        gm = np.asarray(env.obs["g"])[slot]
        levels = self.levels_of(env, slot, ante)
        ck = (tuple(owned), ante, tuple(levels))
        if ck in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[ck]
        pool = sorted(set(owned) | {k for k, a in self.first.items() if a >= ante})
        slots = max(5, len(owned)) + self.extra_slots
        node = (env, slot)
        solo = self._shares(node, [[k] for k in pool], levels)
        order = [k for _, k in sorted(zip(solo, pool), reverse=True)]
        cand = list(dict.fromkeys(order[:self.top] + [k for k in owned if solo[pool.index(k)] >= 0.5 * max(solo)][:3]))
        full = list(cand[:slots])                                                  # first the best `slots` by solo score: when it already reaches the threshold the node is kept and
        best = max(self._shares(node, [full], levels) + [max(solo)])                # nothing more is needed (the value is then a floor of the bound, used only for ordering)
        if best * self.slack < 1.0:
            builds = [list(c) for r in range(1, min(slots, len(cand)) + 1) for c in itertools.combinations(cand, r)]
            best = max(self._shares(node, builds, levels) + [best])
        self.cache[ck] = best
        return best

    def levels_of(self, env, slot, ante):
        """The node's hand levels plus per_ante levels per ante left, on every hand type."""
        lv = env.hand_levels([slot])[0] if hasattr(env, "hand_levels") else [1] * 12
        return [int(x) + self.per_ante * (9 - ante) for x in lv]

    def __call__(self, env, slot, level):
        ante = min(8, 1 + level // 3)
        sh = self.share(env, slot, ante)
        return None if sh * self.slack < 1.0 else sh
