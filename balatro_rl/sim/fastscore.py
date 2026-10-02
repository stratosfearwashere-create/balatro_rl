"""Glue for the compiled scorer (_fastscore.pyx): flattens one game state so all candidate plays of
a decision can be scored in a single call. Same results as calling Game.predict for each play.

Build it with:  cythonize -i -3 balatro_rl/sim/_fastscore.pyx
Without the compiled module (or with BALATRO_PYSCORE=1) everything falls back to Python."""
from __future__ import annotations

import os

from .hands import HAND_BASE, N_HANDS, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, QUADS
from .items import PLANETS
from .jokers import JOKERS, _copy_target

try:
    from . import _fastscore as _fs
except ImportError:          # not built
    _fs = None

ENABLED = _fs is not None and os.environ.get("BALATRO_PYSCORE") != "1" and os.environ.get("BALATRO_PURE") != "1"

if _fs is not None:
    C = _fs.CODES
    assert [tuple(HAND_BASE[h]) for h in range(N_HANDS)] == [tuple(x) for x in _fs.HAND_BASE], \
        "hands.HAND_BASE changed: update _fastscore.pyx"

    ENH = {"": C["E_NONE"], "BONUS": C["E_BONUS"], "MULT": C["E_MULT"], "WILD": C["E_WILD"],
           "GLASS": C["E_GLASS"], "STEEL": C["E_STEEL"], "STONE": C["E_STONE"], "GOLD": C["E_GOLD"],
           "LUCKY": C["E_LUCKY"], "HIDDEN": C["E_HIDDEN"]}
    EDITION = {"": C["ED_NONE"], "FOIL": C["ED_FOIL"], "HOLO": C["ED_HOLO"], "POLYCHROME": C["ED_POLY"],
               "NEGATIVE": C["ED_NEG"]}
    BOSS = {"flint": C["B_FLINT"], "arm": C["B_ARM"], "psychic": C["B_PSYCHIC"], "eye": C["B_EYE"],
            "mouth": C["B_MOUTH"]}
    M, CH, X = C["K_MULT"], C["K_CHIPS"], C["K_X"]

    # key -> hook kinds and parameters (before, card, retrig, held, main, hand, kind, suit, amount)
    def _row(bk="BK_NONE", ck="CK_NONE", rk="RK_NONE", hk="HK_NONE", mk="MK_NONE", hand=0, kind=0, suit=0, amt=0.0):
        return (C[bk], C[ck], C[rk], C[hk], C[mk], hand, kind, suit, float(amt))

    ROWS = {
        # constant / "hand contains" jokers
        "joker": _row(mk="MK_CONST", kind=M, amt=4), "gros_michel": _row(mk="MK_CONST", kind=M, amt=15),
        "cavendish": _row(mk="MK_CONST", kind=X, amt=3), "stuntman": _row(mk="MK_CONST", kind=CH, amt=250),
        "misprint": _row(mk="MK_CONST", kind=M, amt=11.5),          # expected value of randint(0, 23)
        "jolly": _row(mk="MK_CONTAINS", hand=PAIR, kind=M, amt=8),
        "zany": _row(mk="MK_CONTAINS", hand=TRIPS, kind=M, amt=12),
        "mad": _row(mk="MK_CONTAINS", hand=TWO_PAIR, kind=M, amt=10),
        "crazy": _row(mk="MK_CONTAINS", hand=STRAIGHT, kind=M, amt=12),
        "droll": _row(mk="MK_CONTAINS", hand=FLUSH, kind=M, amt=10),
        "sly": _row(mk="MK_CONTAINS", hand=PAIR, kind=CH, amt=50),
        "wily": _row(mk="MK_CONTAINS", hand=TRIPS, kind=CH, amt=100),
        "clever": _row(mk="MK_CONTAINS", hand=TWO_PAIR, kind=CH, amt=80),
        "devious": _row(mk="MK_CONTAINS", hand=STRAIGHT, kind=CH, amt=100),
        "crafty": _row(mk="MK_CONTAINS", hand=FLUSH, kind=CH, amt=80),
        "duo": _row(mk="MK_CONTAINS", hand=PAIR, kind=X, amt=2),
        "trio": _row(mk="MK_CONTAINS", hand=TRIPS, kind=X, amt=3),
        "family": _row(mk="MK_CONTAINS", hand=QUADS, kind=X, amt=4),
        "order": _row(mk="MK_CONTAINS", hand=STRAIGHT, kind=X, amt=3),
        "tribe": _row(mk="MK_CONTAINS", hand=FLUSH, kind=X, amt=2),
        # per-card suit jokers
        "greedy_joker": _row(ck="CK_SUIT", suit=3, kind=M, amt=3),
        "lusty_joker": _row(ck="CK_SUIT", suit=1, kind=M, amt=3),
        "wrathful_joker": _row(ck="CK_SUIT", suit=0, kind=M, amt=3),
        "gluttenous_joker": _row(ck="CK_SUIT", suit=2, kind=M, amt=3),
        "arrowhead": _row(ck="CK_SUIT", suit=0, kind=CH, amt=50),
        "onyx_agate": _row(ck="CK_SUIT", suit=2, kind=M, amt=7),
        # other per-card jokers
        "scary_face": _row(ck="CK_SCARY"), "even_steven": _row(ck="CK_EVEN"), "odd_todd": _row(ck="CK_ODD"),
        "scholar": _row(ck="CK_SCHOLAR"), "walkie_talkie": _row(ck="CK_WALKIE"), "smiley": _row(ck="CK_SMILEY"),
        "photograph": _row(ck="CK_PHOTO"), "fibonacci": _row(ck="CK_FIB"), "bloodstone": _row(ck="CK_BLOOD"),
        "idol": _row(ck="CK_IDOL"), "ancient": _row(ck="CK_ANCIENT"), "triboulet": _row(ck="CK_TRIB"),
        "wee": _row(ck="CK_WEE", mk="MK_VAL_CHIPS"), "hiker": _row(ck="CK_HIKER"),
        # retriggers / held
        "hanging_chad": _row(rk="RK_CHAD"), "hack": _row(rk="RK_HACK"), "dusk": _row(rk="RK_DUSK"),
        "sock_and_buskin": _row(rk="RK_SOCK"), "selzer": _row(rk="RK_SELZER"),
        "shoot_the_moon": _row(hk="HK_SHOOT"), "baron": _row(hk="HK_BARON"),
        # scaling jokers (before hook grows "val", main uses it)
        "ride_the_bus": _row(bk="BK_BUS", mk="MK_VAL_MULT"), "green_joker": _row(bk="BK_GREEN", mk="MK_VAL_MULT"),
        "runner": _row(bk="BK_RUNNER", mk="MK_VAL_CHIPS"), "square": _row(bk="BK_SQUARE", mk="MK_VAL_CHIPS"),
        "trousers": _row(bk="BK_TROUSERS", mk="MK_VAL_MULT"), "loyalty_card": _row(bk="BK_LOYALTY", mk="MK_LOYALTY"),
        "obelisk": _row(bk="BK_OBELISK", mk="MK_VAL_X"), "vampire": _row(bk="BK_VAMPIRE", mk="MK_VAL_X"),
        "midas_mask": _row(bk="BK_MIDAS"), "dna": _row(bk="BK_DNA"),
        # jokers whose "val" is grown elsewhere in the game
        "popcorn": _row(mk="MK_VAL_MULT"), "red_card": _row(mk="MK_VAL_MULT"), "flash": _row(mk="MK_VAL_MULT"),
        "ceremonial": _row(mk="MK_VAL_MULT"),
        "ice_cream": _row(mk="MK_VAL_CHIPS"), "castle": _row(mk="MK_VAL_CHIPS"),
        "constellation": _row(mk="MK_VAL_X"), "ramen": _row(mk="MK_VAL_X"), "madness": _row(mk="MK_VAL_X"),
        "hologram": _row(mk="MK_VAL_X"), "campfire": _row(mk="MK_VAL_X"), "glass": _row(mk="MK_VAL_X"),
        "hit_the_road": _row(mk="MK_VAL_X"), "caino": _row(mk="MK_VAL_X"), "yorick": _row(mk="MK_VAL_X"),
        # main effects that read the game
        "half": _row(mk="MK_HALF"), "banner": _row(mk="MK_BANNER"), "mystic_summit": _row(mk="MK_MYSTIC"),
        "raised_fist": _row(hk="HK_RAISED_FIST"), "abstract": _row(mk="MK_ABSTRACT"),
        "supernova": _row(mk="MK_SUPERNOVA"), "blue_joker": _row(mk="MK_BLUE"), "swashbuckler": _row(mk="MK_SWASH"),
        "fortune_teller": _row(mk="MK_FORTUNE"), "card_sharp": _row(mk="MK_CARD_SHARP"), "bull": _row(mk="MK_BULL"),
        "bootstraps": _row(mk="MK_BOOTSTRAPS"), "acrobat": _row(mk="MK_ACROBAT"),
        "blackboard": _row(mk="MK_BLACKBOARD"), "flower_pot": _row(mk="MK_FLOWER"),
        "seeing_double": _row(mk="MK_SEEING"), "stencil": _row(mk="MK_STENCIL"), "steel_joker": _row(mk="MK_STEEL"),
        "erosion": _row(mk="MK_EROSION"), "stone": _row(mk="MK_STONE"),
        "lucky_cat": _row(ck="CK_LUCKY_CAT", mk="MK_LUCKY_CAT"),
        "throwback": _row(mk="MK_THROWBACK"),       # Baseball Card: Jkr.unc and the count below
        "drivers_license": _row(mk="MK_DRIVERS"),
        # copiers
        "blueprint": _row(bk="BK_COPY", ck="CK_COPY", rk="RK_COPY", hk="HK_COPY", mk="MK_COPY"),
        "brainstorm": _row(bk="BK_COPY", ck="CK_COPY", rk="RK_COPY", hk="HK_COPY", mk="MK_COPY"),
    }
    NO_ROW = _row()
    # hooks that only move money or queue events (never change the score)
    NO_SCORE_EFFECT = {("todo_list", "before"), ("space", "before"),
                       ("sixth_sense", "before"), ("superposition", "before"), ("seance", "before"),
                       ("vagabond", "before"), ("business", "card"), ("rough_gem", "card"),
                       ("8_ball", "card"), ("ticket", "card"), ("reserved_parking", "held")}

    def _check_rows():
        """Every scoring hook in jokers.py is ported, and nothing extra is."""
        for i, hook in enumerate(("before", "card", "retrig", "held", "main")):
            for key, d in JOKERS.items():
                has = getattr(d, hook) is not None
                ported = ROWS.get(key, NO_ROW)[i] != 0
                if has and not ported and (key, hook) not in NO_SCORE_EFFECT:
                    raise AssertionError(f"{key}.{hook} is not ported to _fastscore")
                if ported and not has:
                    raise AssertionError(f"{key}.{hook} is ported but jokers.py has no such hook")
        for key in ROWS:
            assert key in JOKERS, f"unknown joker {key} in fastscore.ROWS"
    _check_rows()


def kept_feats_many(hand, subsets):
    """Compiled env._kept_feats for many discards from one hand, or None when not built."""
    if not ENABLED:
        return None
    return _fs.kept_feats_many([(c.rank, c.suit, ENH.get(c.enh, C["E_OTHER"]), c.hidden) for c in hand], subsets)


def card_row(c) -> tuple:
    return (c.rank, c.suit, ENH.get(c.enh, C["E_OTHER"]), EDITION.get(c.edition, C["ED_OTHER"]),
            C["S_NONE"] if not c.seal else (C["S_RED"] if c.seal == "RED" else C["S_OTHER"]),
            c.extra_chips, c.debuffed)


def _context(g, plan) -> tuple:
    """Everything the scorer reads besides the cards: (jokers, lists, flags, boss, arrays, probs, scalars)."""
    table = g.jokers
    index = {id(j): i for i, j in enumerate(table)}
    jokers = []
    for j in table:
        row = ROWS.get(j.key, NO_ROW) if j.d is JOKERS.get(j.key) else NO_ROW
        target = -1
        if j.key in ("blueprint", "brainstorm"):
            t = _copy_target(g, j, "right" if j.key == "blueprint" else "left")
            target = index[id(t)] if t is not None else -1
        st = j.state
        v = st.get("val")
        jokers.append(row + (EDITION.get(j.edition, C["ED_OTHER"]), target, st.get("rank", 14), st.get("suit", 0),
                             v is not None, 0.0 if v is None else v, j.sell_value(), j.d.rarity == 2))
    lists = ([index[id(j)] for j, _ in plan.before], [index[id(j)] for j, _ in plan.card],
             [index[id(j)] for j, _ in plan.retrig], [index[id(j)] for j, _ in plan.held],
             [index[id(j)] for j, _, _ in plan.main],
             [index[id(j)] for j in plan.jokers if j.key == "hologram"])
    flags = (plan.four_fingers, plan.shortcut, plan.smeared, plan.pareidolia, plan.splash, plan.mime,
             g.has("four_fingers"), g.has("shortcut"), g.has("smeared"))
    boss = (BOSS.get(plan.boss, 0), BOSS.get(g.boss_active(), 0), hand_types_mask(g.round_hand_types), g.mouth_hand,
            plan.boss == "hook", plan.blackboard)
    obs = [0] * N_HANDS
    if "observatory" in g.vouchers:
        for c in g.consumables:
            if c.kind == "planet":
                h = PLANETS.get(c.name)
                if h is not None:
                    obs[h] += 1
    arrays = (g.hand_levels, g.hand_played, g.hand_played_round, obs)
    probs = (g.prob(1, 5), g.prob(1, 15), g.prob(1, 2))
    fd = g.full_deck
    scalars = (g.deck_type == "PLASMA", g.discards_left, g.hands_left, g.money, len(table), len(g.deck),
               g.tarots_used, g.blinds_skipped, g.joker_slots, sum(1 for o in table if o.key == "stencil"),
               sum(1 for c in fd if c.enh == "STEEL"), sum(1 for c in fd if c.enh == "STONE"),
               sum(1 for c in fd if c.enh), len(fd), g.starting_deck_size,
               sum(o.sell_value() for o in table), plan.baseball)       # (the "rare2" slot: Baseball count)
    return jokers, lists, flags, boss, arrays, probs, scalars


def hand_types_mask(types) -> int:
    m = 0
    for h in types:
        m |= 1 << h
    return m


def build(g, plan, view):
    """Compiled scorer for this state (plan and view as passed to Game.predict), or None."""
    if not ENABLED:
        return None
    cards, ctx = [card_row(c) for c in view], _context(g, plan)
    try:
        return _fs.Scorer(cards, *ctx)
    except (ValueError, OverflowError, TypeError):
        return None                       # unusual state (e.g. > 64 cards): use Python


def pool_scorer(g, plan, pool):
    """A compiled scorer for this state that scores hands drawn from `pool` (a list of cards), by index:
    best_two / best_two_many / score_all. Same results as build() on each hand. None if not built."""
    if not ENABLED or not hasattr(_fs.Scorer, "best_two"):
        return None
    rows, ctx = [card_row(c) for c in pool], _context(g, plan)
    try:
        sc = _fs.Scorer([], *ctx)
        sc.set_pool(rows)
        return sc
    except (ValueError, OverflowError, TypeError):
        return None


def subset_patterns(n: int):
    return _fs.subset_patterns(n)
