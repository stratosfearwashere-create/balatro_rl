"""Differential check of the C++ engine (balatro_rl.sim._engine) against the Python game.

Both play the same seeded game with the same actions; after every step the whole state is compared: every
field of the game, every card of every pile (with the relative order of card ids), jokers and their state,
the shop, the pack, and the random generator's state (so the two draw the same numbers in the same order).
With --obs the C++ environment's observation arrays are compared with env.encode()'s as well, and rewards.

    python -m tests.engine_diff --games 200                 # random + rule-based play, all stakes
    python -m tests.engine_diff --games 200 --spice         # plus random jokers, vouchers, consumables, bosses
    python -m tests.engine_diff --games 50 --obs --spice    # also the observations and rewards
    python -m tests.engine_diff --seed 1234 --spice -v      # one game, every action printed

Exit code 1 on the first mismatch, with the step, the action and the fields that differ.
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

from balatro_rl import env as E
from balatro_rl.heuristic import HeuristicPolicy
from balatro_rl.sim import _engine
from balatro_rl.sim.cards import ENHANCEMENTS, EDITIONS, SEALS
from balatro_rl.sim.game import Consumable, Game
from balatro_rl.sim.items import (ALL_BOSSES, BOSSES, DECKS, FINISHERS, PLANETS, SPECTRALS, STAKES, TAGS, TAROTS,
                                  TAG_MIN_ANTE, VOUCHERS)
from balatro_rl.sim.jokers import JOKERS, Joker

STATE_KEYS = ("val", "rank", "suit", "hand", "left")


# ------------------------------------------------------------------ the Python game's state, as the C++ dump
def _card(c):
    return (c.rank, c.suit, c.enh, c.edition, c.seal, c.extra_chips, bool(c.debuffed), bool(c.hidden), c.uid)


def _joker(j):
    st = {k: (float(v) if k == "val" else int(v)) for k, v in j.state.items() if k in STATE_KEYS}
    extra = [k for k in j.state if k not in STATE_KEYS]
    assert not extra, f"joker {j.key} has state keys the engine does not model: {extra}"
    return {"key": j.key, "edition": j.edition, "eternal": bool(j.eternal), "perishable": j.perishable,
            "rental": bool(j.rental), "base_cost": j.base_cost, "sell_bonus": j.sell_bonus, "cost": j.cost,
            "debuffed": bool(j.debuffed), "hidden": bool(j.hidden), "state": st}


def _item(it):
    d = {"kind": it.kind, "cost": it.cost, "key": it.key}
    if it.joker is not None:
        d["joker"] = _joker(it.joker)
    if it.card is not None:
        d["card"] = _card(it.card)
    return d


def py_dump(g: Game) -> dict:
    d = {k: getattr(g, k) for k in (
        "ante", "blind_idx", "state", "money", "hands_left", "discards_left", "discards_used_round", "joker_slots",
        "consumable_slots", "hand_size", "base_hands", "base_discards", "tarots_used", "blinds_skipped",
        "blinds_beaten", "furthest_blind", "total_hands_played", "last_hand", "total_unused_discards", "mouth_hand",
        "juggle", "boss_rerolled_ante", "ox_hand", "reroll_cost", "free_rerolls", "shops_seen", "pack_kind",
        "pack_picks", "pack_return", "boss", "starting_deck_size")}
    d["chips"], d["target"] = float(g.chips), float(g.target)
    d["boss_disabled"], d["first_draw_done"] = bool(g.boss_disabled), bool(g.first_draw_done)
    d["hand_levels"], d["hand_played"] = list(g.hand_levels), list(g.hand_played)
    d["hand_played_round"] = list(g.hand_played_round)
    d["vouchers"] = [v for v in VOUCHERS if v in g.vouchers]
    d["pending_tags"] = list(g.pending_tags)
    f = g.flags
    known = {"verdant", "gros_michel_extinct", "seen_soul", "seen_black_hole", "voucher_bought_ante", "ecto",
             "trading_destroy"}
    assert set(f) <= known, f"flags the engine does not model: {set(f) - known}"
    d["flags"] = {"verdant": bool(f.get("verdant")), "gros_michel_extinct": bool(f.get("gros_michel_extinct")),
                  "seen_soul": bool(f.get("seen_soul")), "seen_black_hole": bool(f.get("seen_black_hole")),
                  "voucher_bought_ante": f.get("voucher_bought_ante", -1), "ecto": f.get("ecto", 0),
                  "trading_destroy": "trading_destroy" in f}
    d["planets_used"] = [p for p in PLANETS if p in g.planets_used]
    d["last_consumable"] = g.last_consumable.name if g.last_consumable is not None else ""
    d["used_bosses"] = [b for b in ALL_BOSSES if b in g.used_bosses]
    d["crimson"] = next((i for i, j in enumerate(g.jokers) if j.uid == g.crimson_disabled), -1)
    d["forced"] = max([i for i, c in enumerate(g.hand) if c.uid == g.forced_uid], default=-1)
    d["round_hand_types"] = sorted(g.round_hand_types)
    d["targeting"] = None if g.targeting is None else (g.targeting["cons"].name, bool(g.targeting["from_pack"]))
    d["shop"] = [_item(it) for it in g.shop]
    d["shop_packs"] = [_item(it) for it in g.shop_packs]
    d["shop_vouchers"] = [_item(it) for it in g.shop_vouchers]
    d["ante_voucher"] = g.ante_voucher.key if g.ante_voucher is not None else ""
    d["pack_cards"] = [_joker(x) if isinstance(x, Joker) else (x.key if isinstance(x, Consumable) else _card(x))
                       for x in g.pack_cards]
    d["tags_offered"] = list(g.tags_offered)
    d["jokers"] = [_joker(j) for j in g.jokers]
    d["consumables"] = [(c.name, bool(c.negative), c.sell_bonus) for c in g.consumables]
    for pile in ("full_deck", "deck", "hand", "discard_pile", "pack_hand"):
        d[pile] = [_card(c) for c in getattr(g, pile)]
    d["ante_played"] = [c.uid in g.ante_played_uids for c in g.full_deck]
    return d


def canon(d: dict) -> dict:
    """Card ids replaced by their rank among the ids in the dump (only their order means anything)."""
    uids = set()

    def walk(x):
        if isinstance(x, tuple) and len(x) == 9 and isinstance(x[2], str):
            uids.add(x[8])
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(d)
    rank = {u: i for i, u in enumerate(sorted(uids))}

    def sub(x):
        if isinstance(x, tuple) and len(x) == 9 and isinstance(x[2], str):
            return tuple(x[:8]) + (rank[x[8]],)
        if isinstance(x, dict):
            return {k: sub(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [sub(v) for v in x]
        return x
    return sub(d)


CPP_ONLY = {"planet_uses", "planet_level_gains"}


def diff(a, b, path="") -> list[str]:
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                if k in CPP_ONLY and k not in a:                 # the sandbox's bookkeeping (planet cap), which the Python game does not have
                    continue
                out.append(f"{path}.{k}: only in {'python' if k in a else 'c++'}")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} (python) != {len(b)} (c++)")
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: {a!r} (python) != {b!r} (c++)")
    return out


def rng_equal(g: Game, cg) -> bool:
    if g.sg is not None:                                    # exact-seed mode: the named streams and the global state too
        st = {k: round(v, 15) for k, v in g.sg.streams.state.items()}
        cs = {k: round(v, 15) for k, v in cg.exact_streams().items()}
        if st != cs or list(g.sg.streams.rng.u) != list(cg.exact_rng()):
            return False
    return tuple(g.rng.getstate()[1]) == tuple(cg.rng_state())


# ------------------------------------------------------------------ driving both games
def apply_cpp(cg, a: int, obs: dict, targeting: bool):
    if targeting:
        cg.apply_targets(E.subset_of(obs, a))
    elif E.A_PLAY <= a < E.A_DISC:
        cg.play(E.subset_of(obs, a))
    elif a < E.A_SELECT:
        cg.discard(E.subset_of(obs, a))
    elif a == E.A_SELECT:
        cg.select_blind()
    elif a == E.A_SKIP:
        cg.skip_blind()
    elif a == E.A_REROLL_BOSS:
        cg.reroll_boss()
    elif a < E.A_BUY_PACK:
        cg.buy_card(a - E.A_BUY)
    elif a < E.A_VOUCHER:
        cg.buy_pack(a - E.A_BUY_PACK)
    elif a < E.A_REROLL:
        cg.buy_voucher(a - E.A_VOUCHER)
    elif a == E.A_REROLL:
        cg.reroll()
    elif a == E.A_LEAVE:
        cg.leave_shop()
    elif a < E.A_SELL_C:
        cg.sell_joker(a - E.A_SELL_J)
    elif a < E.A_USE_C:
        cg.sell_consumable(a - E.A_SELL_C)
    elif a < E.A_PICK:
        cg.use_consumable(a - E.A_USE_C, True)
    elif a < E.A_PSKIP:
        cg.pack_pick(a - E.A_PICK, True)
    elif a == E.A_PSKIP:
        cg.pack_skip()
    elif a < E.A_SWAP_END:
        cg.swap_jokers(a - E.A_SWAP)
    else:                                                  # replace: sell joker j, then buy shop joker i if it fits
        i, j = divmod(a - E.A_REPLACE, E.MAX_JOKERS)
        cg.sell_joker(j)
        try:
            cg.buy_card(i)
        except RuntimeError:
            pass
    cg.order_jokers()


def spice(g: Game, cg, rng: random.Random) -> str:
    """One random change applied to both games (only between decisions outside a round)."""
    kind = rng.choice(["joker", "joker", "joker", "cons", "voucher", "money", "card", "boss", "tag", "level", "ptag",
                       "jstate"])
    if kind == "jstate" and g.jokers:
        # a joker about to run out, fire or expire: the edges of the scaling / self-destroying jokers
        i = rng.randrange(len(g.jokers))
        j = g.jokers[i]
        edge = {"ramen": ("val", 1.01), "ice_cream": ("val", 5), "popcorn": ("val", 4), "turtle_bean": ("val", 1),
                "selzer": ("val", 1), "loyalty_card": ("val", rng.choice([0, 1])), "invisible": ("val", 2),
                "yorick": ("left", 2), "rocket": ("val", 7), "campfire": ("val", 2.5), "obelisk": ("val", 1.8),
                "green_joker": ("val", 0), "ride_the_bus": ("val", 9), "hologram": ("val", 2.25)}.get(j.key)
        if edge is None:
            return ""
        j.state[edge[0]] = edge[1]
        cg.debug_set_joker_state(i, edge[0], float(edge[1]))
        return f"joker state {j.key} {edge[0]}={edge[1]}"
    if kind == "joker" and len(g.jokers) < g.joker_slots + 1:
        k = rng.choice(list(JOKERS))
        ed = rng.choice(["", "", "", "FOIL", "HOLO", "POLYCHROME", "NEGATIVE"])
        eternal = rng.random() < 0.1
        rental = rng.random() < 0.1
        perish = 5 if rng.random() < 0.1 else None
        g.add_joker(Joker(k, edition=ed, base_cost=JOKERS[k].cost, eternal=eternal, rental=rental, perishable=perish))
        cg.debug_add_joker(k, ed, eternal, perish, rental, JOKERS[k].cost)
        return f"joker {k} {ed}"
    if kind == "cons" and len(g.consumables) < g.consumable_slots:
        pool = [("planet", p) for p in PLANETS] + [("tarot", t) for t in TAROTS] + [("spectral", s) for s in SPECTRALS]
        k, n = rng.choice(pool)
        g.consumables.append(Consumable(k, n))
        cg.debug_add_consumable(n, False)
        return f"consumable {n}"
    if kind == "voucher" and g.state in ("SHOP", "BLIND_SELECT"):
        v = rng.choice([v for v in VOUCHERS if v not in g.vouchers] or list(VOUCHERS))
        pre = VOUCHERS[v]
        out = []
        for x in ([pre] if pre and pre not in g.vouchers else []) + [v]:
            if x in ("hieroglyph", "petroglyph") or (g.state != "SHOP" and x.startswith("overstock")):
                continue
            g.redeem_voucher(x)
            cg.debug_redeem_voucher(x)
            out.append(x)
        return "voucher " + " ".join(out)
    if kind == "money":
        m = rng.choice([0, 3, 12, 30, 80, -5])
        g.money = m
        cg.money = m
        return f"money {m}"
    if kind == "card" and g.full_deck:
        i = rng.randrange(len(g.full_deck))
        enh, ed, seal = rng.choice(ENHANCEMENTS), rng.choice(EDITIONS[:4]), rng.choice(SEALS)
        c = g.full_deck[i]
        c.enh, c.edition, c.seal = enh, ed, seal
        cg.debug_set_card(i, enh, ed, seal)
        return f"card {i} {enh} {ed} {seal}"
    if kind == "boss" and g.state == "BLIND_SELECT":
        pool = list(FINISHERS) if g.ante >= 8 else [k for k, v in BOSSES.items() if v[1] <= g.ante]
        b = rng.choice(pool + list(FINISHERS)[:1])
        g.boss = b
        cg.debug_set_boss(b)
        return f"boss {b}"
    if kind == "tag" and g.state == "BLIND_SELECT" and g.blind_idx < 2:
        t = rng.choice([t for t in TAGS if TAG_MIN_ANTE.get(t, 1) <= max(g.ante, 2)])
        g.tags_offered[g.blind_idx] = t
        cg.debug_set_tag_offered(g.blind_idx, t)
        return f"tag {t}"
    if kind == "ptag":
        t = rng.choice(["uncommon", "rare", "foil", "holo", "polychrome", "negative", "coupon", "d_six", "investment",
                        "voucher", "double", "juggle"])
        g.pending_tags.append(t)
        cg.debug_add_tag(t)
        return f"pending tag {t}"
    if kind == "level":
        h, lv = rng.randrange(12), rng.randint(1, 9)
        g.hand_levels[h] = lv
        cg.debug_set_hand_level(h, lv)
        return f"level {h} {lv}"
    return ""


OBS_KEYS = ("g", "cards", "jid", "jf", "cid", "aid", "af", "mask", "subs", "aid2")


def obs_diff(po: dict, co: dict) -> list[str]:
    out = []
    for k in OBS_KEYS:
        a, b = np.asarray(po[k]), np.asarray(co[k])
        if a.shape != b.shape:
            out.append(f"obs.{k}: shape {a.shape} != {b.shape}")
            continue
        if a.dtype != b.dtype:
            out.append(f"obs.{k}: dtype {a.dtype} != {b.dtype}")
        bad = np.argwhere(a != b)
        if len(bad):
            i = tuple(bad[0])
            out.append(f"obs.{k}{list(i)}: {a[i]!r} (python) != {b[i]!r} (c++)  [{len(bad)} cells differ]")
    return out


def new_cov() -> dict:
    return {k: set() for k in ("joker", "joker_end", "cons", "boss", "tag", "ptag", "voucher", "pack", "deck", "stake",
                               "misc")}


def play_one(seed: int, deck: str, stake: str, do_spice: bool, check_obs: bool, verbose: bool = False,
             max_steps: int = 2500, p_random_round: float = 0.1, p_random_build: float = 0.5,
             win_ante: int = 8, deep: int = 0, p_spice: float = 0.15, marginal: int = 0,
             exact: bool = False) -> tuple[list[str], dict]:
    """One game on both engines. Returns (mismatches, info). deep: every hand starts at this level, so the
    rule-based play reaches the late antes (finisher bosses, many shops)."""
    rng = random.Random(seed * 7919 + 13)
    env = E.BalatroEnv(deck, stake, win_ante=win_ante, verdant_bonus=0.75, marginal=marginal)
    if exact:
        from balatro_rl.sim.seedrng import seed_string
        env.game_kw = dict(getattr(env, "game_kw", None) or {}, exact_seed=seed_string(seed))
    obs = env.reset(seed)
    g = env.g
    if check_obs:
        cenv = _engine.Env(deck, stake, win_ante=win_ante)
        cenv.verdant_bonus = 0.75
        cenv.marginal = marginal
        cenv.marg_verify = bool(marginal)                   # the cached build values are recomputed and compared on every hit
        cenv.exact = exact
        cobs = cenv.reset(seed)
        cg = cenv.game
    else:
        cenv = None
        if exact:
            from balatro_rl.sim.seedrng import seed_string
            cg = _engine.Game(seed, deck, stake, win_ante, exact_seed=seed_string(seed))
        else:
            cg = _engine.Game(seed, deck, stake, win_ante)
    pol = HeuristicPolicy(rng=np.random.default_rng(seed), lookahead=False)
    cov = new_cov()
    cov["deck"].add(deck)
    cov["stake"].add(stake)
    info = {"steps": 0, "spices": 0, "ante": 1, "cov": cov}
    if deep:
        for h in range(12):
            g.hand_levels[h] = deep
            cg.debug_set_hand_level(h, deep)
        env.obs = obs = E.encode(g, env.cnt, marginal)
        if check_obs:
            cobs = cenv.observe()

    def check(where: str) -> list[str]:
        d = diff(canon(py_dump(g)), canon(cg.dump()))
        if not rng_equal(g, cg):
            d.append("rng: the generators' states differ")
            if g.sg is not None:
                st = {k: round(v, 15) for k, v in g.sg.streams.state.items()}
                cs = {k: round(v, 15) for k, v in cg.exact_streams().items()}
                d.append("streams only in python: %s; only in c++: %s; differing: %s; global equal: %s" % (
                    sorted(set(st) - set(cs)), sorted(set(cs) - set(st)),
                    sorted(k for k in st if k in cs and st[k] != cs[k]), list(g.sg.streams.rng.u) == list(cg.exact_rng())))
        if check_obs and not d and not g.done:
            d = obs_diff(env.obs, cobs)
        return [f"{where}: {x}" for x in d[:25]]

    bad = check("start")
    if bad:
        return bad, info
    for step in range(max_steps):
        if do_spice and g.state in ("BLIND_SELECT", "SHOP") and g.targeting is None and rng.random() < p_spice:
            what = spice(g, cg, rng)
            if what:
                info["spices"] += 1
                if verbose:
                    print(f"  spice: {what}")
                env.obs = obs = E.encode(g, env.cnt, marginal)
                if check_obs:
                    cobs = cenv.observe()
                bad = check(f"step {step} after spice ({what})")
                if bad:
                    return bad, info
        legal = np.flatnonzero(obs["mask"])
        if len(legal) == 0:
            break
        in_round = g.state == "SELECTING_HAND"
        uses = [int(x) for x in legal if E.A_USE_C <= x < E.A_PICK]
        if uses and g.targeting is None and rng.random() < 0.3:
            a = rng.choice(uses)
        elif rng.random() < (p_random_round if in_round else p_random_build):
            a = int(rng.choice(list(legal)))
        else:
            a = int(pol.act(g, obs))
            if not obs["mask"][a]:
                a = int(rng.choice(list(legal)))
        desc = E.describe_action(g, a, obs)
        if verbose:
            print(f"  {step:4d} ante {g.ante} {g.state:14s} ${g.money:<4} {desc}")
        targeting = g.targeting is not None
        # what this step exercises
        cov["voucher"] |= g.vouchers
        cov["ptag"] |= set(g.pending_tags)
        if in_round and len(g.hand) > 8:
            cov["misc"].add("a hand of more than 8 cards")
        if targeting and len(g.target_cards()) > 8:
            cov["misc"].add("targets chosen among more than 8 cards")
        if targeting:
            cov["misc"].add("targets chosen in a pack" if g.targeting["from_pack"] else "targets chosen in a round")
        if in_round and g.boss_active() == "hook" and any(j.d.held for j in g.jokers):
            cov["misc"].add("The Hook with a held-card joker (averaged prediction)")
        if in_round and any(c.hidden for c in g.hand):
            cov["misc"].add("face-down cards in hand")
        if any(j.key in ("blueprint", "brainstorm") for j in g.jokers) and in_round:
            cov["misc"].add("a copier joker in a round")
        if not targeting and a >= E.A_REPLACE:
            cov["misc"].add("a joker replaced in one action")
        if in_round and not targeting and a < E.A_SELECT:
            cov["joker"] |= {j.key for j in g.jokers if not j.debuffed}
            if g.blind_idx == 2 and not g.boss_disabled:
                cov["boss"].add(g.boss)
        if not targeting:
            if E.A_USE_C <= a < E.A_PICK:
                cov["cons"].add(g.consumables[a - E.A_USE_C].name)
            elif E.A_PICK <= a < E.A_PSKIP and isinstance(g.pack_cards[a - E.A_PICK], Consumable):
                cov["cons"].add(g.pack_cards[a - E.A_PICK].name)
            elif a == E.A_SKIP:
                cov["tag"].add(g.tags_offered[g.blind_idx])
            elif E.A_BUY_PACK <= a < E.A_VOUCHER:
                cov["pack"].add(g.shop_packs[a - E.A_BUY_PACK].key)
        beaten = g.blinds_beaten
        keys_before = {j.key for j in g.jokers if not j.debuffed}
        try:
            if check_obs:
                cobs, cr, cdone, _ = cenv.step(a)
            else:
                apply_cpp(cg, a, obs, targeting)
        except Exception as ex:                                 # the C++ side refused the action
            return [f"step {step} ({desc}): c++ raised {ex!r}"], info
        obs, r, done, _ = env.step(a)
        if g.blinds_beaten > beaten:
            cov["joker_end"] |= keys_before
        info["steps"] = step + 1
        info["ante"] = g.ante
        where = f"step {step} ({desc})"
        if check_obs:
            if abs(r - cr) > 1e-12 or bool(done) != bool(cdone):
                return [f"{where}: reward {r!r} done {done} (python) != {cr!r} {cdone} (c++)"], info
            if done:                      # the Python env ends the game when no action is legal; compare the rest
                bad = [f"{where}: {x}" for x in diff(canon(py_dump(g)), canon(cg.dump()))[:25]]
                return bad, info
        if done and not check_obs and g.state == "GAME_OVER" and cg.state != "GAME_OVER":
            # the Python environment ends a game with no legal action; the bare C++ game has no such rule
            d = [x for x in diff(canon(py_dump(g)), canon(cg.dump())) if not x.startswith(".state:")]
            return [f"{where}: {x}" for x in d[:25]], info
        bad = check(where)
        if bad:
            return bad, info
        if done or g.done:
            break
    return [], info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--seed", type=int, default=None, help="one game with this seed")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--deck", default=None, help="default: cycles through all decks")
    p.add_argument("--stake", default=None, help="default: cycles through all stakes")
    p.add_argument("--spice", action="store_true")
    p.add_argument("--obs", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--deep", type=int, default=0, help="every hand starts at this level (reaches the late antes)")
    p.add_argument("--p-spice", type=float, default=0.15)
    p.add_argument("--marginal", type=int, nargs="?", const=1, default=0,
                   help="compare the marginal-value inputs too (needs --obs): 1 the marginal values, 2 also the potential")
    p.add_argument("--coverage", action="store_true", help="list the content the games never exercised")
    p.add_argument("--exact", action="store_true", help="exact-seed mode: the game's named random streams (both engines)")
    a = p.parse_args()
    seeds = [a.seed] if a.seed is not None else list(range(a.seed0, a.seed0 + a.games))
    t = time.time()
    fails = steps = spices = 0
    antes = []
    cov = new_cov()
    for s in seeds:
        deck = a.deck or DECKS[s % len(DECKS)]
        stake = a.stake or STAKES[(s // 3) % len(STAKES)]
        bad, info = play_one(s, deck, stake, a.spice, a.obs, a.verbose, deep=a.deep, p_spice=a.p_spice, marginal=a.marginal, exact=a.exact)
        for k, v in info["cov"].items():
            cov[k] |= v
        steps += info["steps"]
        spices += info["spices"]
        antes.append(info["ante"])
        if bad:
            fails += 1
            print(f"MISMATCH seed {s} deck {deck} stake {stake}:")
            for line in bad:
                print("   ", line)
            if not a.keep_going:
                sys.exit(1)
    print(f"{len(seeds)} games, {steps} steps, {spices} injected changes, mean ante {np.mean(antes):.2f}, "
          f"max ante {max(antes)}: {fails} mismatching games ({time.time() - t:.0f}s)")
    if a.coverage:
        from balatro_rl.sim.items import PACK_KEYS
        full = {"joker": set(JOKERS), "joker_end": set(JOKERS), "cons": set(PLANETS) | set(TAROTS) | set(SPECTRALS),
                "boss": set(ALL_BOSSES), "tag": set(TAGS), "voucher": set(VOUCHERS),
                "pack": {f"p_{k}_{sz}" for k, sz in PACK_KEYS}, "deck": set(DECKS), "stake": set(STAKES),
                "misc": {"a hand of more than 8 cards", "targets chosen among more than 8 cards",
                         "targets chosen in a pack", "targets chosen in a round",
                         "The Hook with a held-card joker (averaged prediction)", "face-down cards in hand",
                         "a copier joker in a round", "a joker replaced in one action"}}
        names = {"joker": "jokers held while a hand was played or discarded", "joker_end": "jokers held at a round's end",
                 "cons": "consumables used", "boss": "bosses played against", "tag": "tags taken by skipping",
                 "voucher": "vouchers owned", "pack": "packs bought", "deck": "decks", "stake": "stakes",
                 "misc": "special situations"}
        for k, allk in full.items():
            miss = sorted(allk - cov[k])
            print(f"  {names[k]}: {len(allk) - len(miss)}/{len(allk)}" + (f"   never: {'; '.join(miss)}" if miss else ""))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
