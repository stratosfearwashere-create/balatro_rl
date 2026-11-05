"""The scenario generator (engine Env::apply_scenario, train.py --scenario-*): legal sampled starts, the three kinds, the cut-off, the coupling of jokers and deck."""
import numpy as np
import pytest

from balatro_rl import env as E
from balatro_rl.engine_env import CppEnv
from balatro_rl.sim.jokers import JOKERS

from balatro_rl.engine_env import _engine

pytestmark = pytest.mark.skipif(_engine is None or not hasattr(_engine.VecEnv, "set_scenario"), reason="engine not rebuilt with scenarios")


def scenario_env(kind_weights=(1, 1, 1), couple=0.5):
    e = CppEnv("RED", "GOLD")
    e.e.scn_p = 1.0
    e.e.set_scn_weights(*kind_weights)
    return e


def test_off_by_default():
    e = CppEnv("RED", "GOLD")
    e.reset(3)
    d = e.game.dump()
    assert e.e.scn_kind == -1 and d["ante"] == 1 and len(d["jokers"]) == 0


def test_same_seed_same_scenario():
    a, b = scenario_env(), scenario_env()
    for seed in range(1, 25):
        a.reset(seed); b.reset(seed)
        assert a.game.dump() == b.game.dump()
        assert a.e.scn_kind == b.e.scn_kind and a.e.scn_limit == b.e.scn_limit


def test_sampled_starts_are_legal():
    e = scenario_env()
    keys = list(JOKERS)
    kinds = set()
    for seed in range(1, 400):
        e.reset(seed)
        d = e.game.dump()
        kinds.add(e.e.scn_kind)
        assert 1 <= d["ante"] <= 7
        assert 2 <= len(d["jokers"]) <= 5 and len(d["jokers"]) <= d["joker_slots"]
        assert len({j["key"] for j in d["jokers"]}) == len(d["jokers"])
        assert all(JOKERS[j["key"]].rarity <= 3 for j in d["jokers"])           # no legendaries
        assert 20 <= len(d["full_deck"]) <= 90
        assert d["money"] >= 0
        assert d["state"] == ("BLIND_SELECT" if e.e.scn_kind == 0 else "SHOP")
        assert d["furthest_blind"] >= 3 * (d["ante"] - 1)
        if e.e.scn_kind == 2:
            assert e.e.scn_limit == 0
        else:
            assert e.e.scn_limit > d["furthest_blind"]
        assert e.obs["mask"].any()
    assert kinds == {0, 1, 2}


def test_one_blind_scenario_cannot_skip_and_is_cut_off_at_its_blind():
    e = scenario_env((1, 0, 0))
    cut = 0
    rng = np.random.default_rng(0)
    for seed in range(1, 300):
        e.reset(seed)
        assert e.e.scn_kind == 0
        start = e.game.dump()["furthest_blind"]
        for _ in range(300):
            m = np.asarray(e.obs["mask"])
            assert not m[E.A_SKIP]                                           # (a skipped blind would end nothing)
            _, _, done, info = e.step(int(rng.choice(np.flatnonzero(m))))
            if done:
                d = e.game.dump()
                if d["state"] not in ("GAME_OVER", "WON"):                    # cut off: the blind was beaten, nothing else happened
                    assert d["furthest_blind"] == start + 1
                    cut += 1
                break
    assert cut > 0


def test_few_blind_scenario_caps_rerolls():
    e = scenario_env((0, 1, 0))
    for seed in range(1, 40):
        e.reset(seed)
        assert e.e.scn_kind == 1
        d = e.game.dump()
        n = 0
        while e.game.dump()["state"] == "SHOP" and np.asarray(e.obs["mask"])[E.A_REROLL]:
            e.step(E.A_REROLL)
            n += 1
            if n > 12:
                break
        assert n <= 3


def test_counters_are_inside_their_bounds():
    e = scenario_env()
    seen = 0
    for seed in range(1, 800):
        e.reset(seed)
        for j in e.game.dump()["jokers"]:
            v = j["state"].get("val")
            if j["key"] in ("constellation", "hologram", "vampire", "lucky_cat", "glass", "campfire", "hit_the_road", "obelisk", "caino"):
                assert 1.0 <= v <= 8.0
                seen += 1
            if j["key"] in ("green_joker", "flash", "red_card", "trousers", "ride_the_bus"):
                assert 0 <= v <= 120
                seen += 1
            if j["key"] == "popcorn":
                assert 4 <= v <= 20
    assert seen > 50


def test_coupling_makes_decks_fit_jokers():
    FACE = {"scary_face", "smiley", "photograph", "sock_and_buskin", "pareidolia", "business", "midas_mask", "reserved_parking", "faceless", "baron",
            "triboulet", "shoot_the_moon"}
    e = scenario_env()
    with_joker, overall = [], []
    for seed in range(1, 1500):
        e.reset(seed)
        d = e.game.dump()
        heavy = sum(1 for c in d["full_deck"] if c[0] in (11, 12, 13)) > 14
        overall.append(heavy)
        if any(j["key"] in FACE for j in d["jokers"]):
            with_joker.append(heavy)
    assert len(with_joker) > 100
    assert np.mean(with_joker) > np.mean(overall) + 0.05          # decks of games holding a face joker are more often face-heavy than decks in general
