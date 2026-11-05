"""The joker school and the penalty for dying with money (engine training aids; see Env::school_*, death_pen)."""
import numpy as np
import pytest

from balatro_rl import env as E
from balatro_rl.engine_env import CppEnv, CppVecEnv
from balatro_rl.sim.jokers import JOKERS

try:
    from balatro_rl.sim import _engine
    HAVE = hasattr(_engine.Env, "school_p")
except ImportError:                                    # not built
    _engine = None
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="the C++ engine with the school is not built")
KEYS = list(JOKERS)


def school_env(student, seed=5):
    e = CppEnv("RED", "GOLD")
    e.e.school_p = 1.0
    e.e.school_ids = [KEYS.index(student)]
    e.reset(seed)
    return e


def to_shop(e, rounds=1):
    """Plays on (hand levels high so that the blinds fall) until the first shop."""
    for h in range(12):
        e.game.debug_set_hand_level(h, 30)
    for _ in range(400):
        if e.game.state == "SHOP":
            return True
        m = e.e.observe()["mask"]
        a = E.A_SELECT if m[E.A_SELECT] else int(np.flatnonzero(m)[0])
        e.step(a)
    return e.game.state == "SHOP"


def test_a_school_game_starts_with_an_eternal_student_and_its_kit():
    for student, kit in (("vampire", ["empress", "heirophant"]), ("lucky_cat", ["magician", "magician"]),
                         ("glass", ["justice", "justice"]), ("constellation", ["mars", "earth"]), ("hologram", [])):
        e = school_env(student)
        d = e.game.dump()
        assert [j["key"] for j in d["jokers"]] == [student] and d["jokers"][0]["eternal"]
        assert [c[0] for c in d["consumables"]] == kit
        assert e.e.school_student == KEYS.index(student)
        assert not e.e.observe()["mask"][E.A_SELL_J]            # an eternal cannot be sold


def test_no_school_when_off():
    e = CppEnv("RED", "GOLD")
    e.reset(5)
    assert e.e.school_student == -1 and not e.game.dump()["jokers"]


def test_every_shop_of_a_school_game_has_a_chain_item():
    for student, want in (("vampire", ("c_empress", "c_heirophant", "c_magician")), ("glass", ("c_justice",)),
                          ("lucky_cat", ("c_magician",)), ("constellation", ("c_mars", "c_earth", "c_jupiter", "c_saturn"))):
        e = school_env(student)
        assert to_shop(e), student
        keys = [it["key"] for it in e.game.dump()["shop"]]
        assert any(k in want for k in keys), (student, keys)
    e = school_env("hologram")
    assert to_shop(e)
    assert any(p["key"] == "p_standard_normal" for p in e.game.dump()["shop_packs"])


def test_the_school_is_off_for_games_that_do_not_draw_it():
    v = CppVecEnv(64, "RED", "GOLD", seed0=1, threads=1)
    v.set_school(0.25, [KEYS.index("vampire"), KEYS.index("glass")])
    rng = np.random.default_rng(0)
    seen = 0
    for _ in range(400):
        m = v.obs["mask"]
        r = rng.random(m.shape)
        r[~m] = -1
        v.step(r.argmax(1))
        seen += int((v.school_student() >= 0).sum())
    share = seen / (400 * 64)
    assert 0.1 < share < 0.4                                       # about a quarter of the games
    assert set(np.unique(v.school_student())) <= {-1, KEYS.index("vampire"), KEYS.index("glass")}


def test_dying_with_money_costs():
    def play(pen):
        e = CppEnv("RED", "GOLD")
        e.e.death_pen, e.e.death_cap = pen, 100.0
        e.reset(11)
        e.game.money = 80                                          # a rich start: the money is mostly still there at the end
        rng = np.random.default_rng(3)
        for _ in range(3000):
            m = e.e.observe()["mask"]
            legal = np.flatnonzero(m)
            a = E.A_LEAVE if (m[E.A_LEAVE] and rng.random() < 0.7) else int(rng.choice(legal))
            _, r, done, _ = e.step(a)
            if done:
                return r, int(e.game.money), e.game.state
        return None
    r0, money, state = play(0.0)
    r1, money1, _ = play(2.0)
    assert state == "GAME_OVER" and money == money1
    assert abs((r0 - r1) - 2.0 * min(max(money, 0), 100) / 100.0) < 1e-5


def test_xscale_start_hands_out_a_scaled_joker_and_set_stakes_moves_environments():
    v = CppVecEnv(64, "RED", "GOLD", seed0=1, threads=1)
    v.set_xscale(1.0, 3.0)
    v.set_stakes(["DIAMOND"] * 32 + ["GOLD"] * 32)
    assert v.stakes[:2] == ["DIAMOND", "DIAMOND"]
    names = set()
    for k in range(64):
        e = CppEnv("RED", "GOLD")
        e.e.start_xscale, e.e.start_xscale_val = 1.0, 3.0
        e.reset(100 + k)
        js = e.game.dump()["jokers"]
        assert len(js) == 1 and js[0]["key"] in ("constellation", "hologram", "vampire", "lucky_cat", "glass")
        assert abs(js[0]["state"]["val"] - 3.0) < 1e-9 and not js[0]["eternal"]
        names.add(js[0]["key"])
    assert len(names) >= 4
    e = CppEnv("RED", "GOLD")
    e.reset(5)
    assert not e.game.dump()["jokers"]                       # off by default


def test_level_cap_blocks_planets_after_two_uses_until_ante_4():
    def mask_use(e, slot):
        return bool(e.e.observe()["mask"][E.A_USE_C + slot])

    e = CppEnv("RED", "GOLD")
    e.e.level_cap_p, e.e.level_cap_n, e.e.level_cap_ante = 1.0, 2, 4
    e.reset(9)
    assert e.e.level_capped_game
    e.step(E.A_SELECT)
    for _ in range(2):
        e.game.debug_add_consumable("mercury", False)
        assert mask_use(e, 0)
        e.step(E.A_USE_C)
    e.game.debug_add_consumable("venus", False)
    assert not mask_use(e, 0)                                    # the third planet cannot be used (ante 1)
    off = CppEnv("RED", "GOLD")
    off.reset(9)
    assert not off.e.level_capped_game                           # off by default
    off.step(E.A_SELECT)
    for _ in range(3):
        off.game.debug_add_consumable("mercury", False)
        assert bool(off.e.observe()["mask"][E.A_USE_C])
        off.step(E.A_USE_C)


def test_death_penalty_counts_debt_too():
    def play(pen):
        e = CppEnv("RED", "GOLD")
        e.e.death_pen, e.e.death_cap = pen, 100.0
        e.reset(11)
        e.game.money = -60
        rng = np.random.default_rng(3)
        for _ in range(3000):
            m = e.e.observe()["mask"]
            legal = np.flatnonzero(m)
            a = E.A_LEAVE if (m[E.A_LEAVE] and rng.random() < 0.7) else int(rng.choice(legal))
            _, r, done, _ = e.step(a)
            if done:
                return r, int(e.game.money)
        raise AssertionError("the game did not end")
    r0, money = play(0.0)
    r1, _ = play(2.0)
    assert money < 0 and abs((r0 - r1) - 2.0 * min(abs(money), 100) / 100.0) < 1e-5
