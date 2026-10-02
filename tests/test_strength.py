"""Tests for the build-strength module (balatro_rl/rewards/strength.py) and the options built on it."""
import copy
import math
import random

import numpy as np
import pytest
import torch

from balatro_rl.sim.game import Game
from balatro_rl.sim.items import BLIND_BASE
from balatro_rl.sim.jokers import Joker
from balatro_rl.rewards import growth
from balatro_rl.rewards.config import PotentialConfig, RewardConfig, StrengthConfig
from balatro_rl.rewards.potential import Headroom, Potential
from balatro_rl.rewards.strength import (HORIZON, N_CLEAR, N_SURVIVE, Strength, boss_targets, clear_chances,
                                         horizon_antes, round_totals, strength_score, survives_through)

torch.set_num_threads(1)
PLAIN = StrengthConfig(discards="none")


def white(seed=0, jokers=(), ante=1) -> Game:
    g = Game(seed=seed, stake="WHITE")
    if ante != 1:
        g.ante = ante
        g.new_ante()
    for k in jokers:
        g.add_joker(Joker(k, base_cost=4))
    return g


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_of_round_totals():
    rng = lambda: np.random.default_rng(0)
    tot = round_totals([100.0] * 32, 4, 3, PLAIN, rng())
    assert tot.shape == (PLAIN.bootstrap,) and (tot == 400.0).all()         # hands per round x the one score
    assert clear_chances([100.0] * 32, 4, 3, [400, 401], PLAIN, 0) == [1.0, 0.0]
    # one hand per round, half the hands score 100: the chance is the share of such hands
    p = clear_chances([0.0, 100.0] * 16, 1, 0, [100], StrengthConfig(discards="none", bootstrap=20_000), 1)[0]
    assert abs(p - 0.5) < 0.02
    # two hands, target 200: both must hit -> 1/4; target 100: at least one -> 3/4
    p2 = clear_chances([0.0, 100.0] * 16, 2, 0, [200, 100], StrengthConfig(discards="none", bootstrap=20_000), 1)
    assert abs(p2[0] - 0.25) < 0.02 and abs(p2[1] - 0.75) < 0.02
    # seeded: the same seed gives the same answer, and more demanding targets are never likelier
    scores = list(np.random.default_rng(3).gamma(2.0, 100.0, 32))
    a = clear_chances(scores, 4, 3, [300, 600, 900, 1500], PLAIN, 7)
    assert a == clear_chances(scores, 4, 3, [300, 600, 900, 1500], PLAIN, 7)
    assert a == sorted(a, reverse=True) and a != clear_chances(scores, 4, 3, [300, 600, 900, 1500], PLAIN, 8)
    assert clear_chances([], 4, 3, [1], PLAIN, 0) == [0.0]


def test_discard_and_scale_corrections():
    scores = list(np.random.default_rng(5).gamma(2.0, 100.0, 32))
    targets = [600, 900, 1200]
    none = clear_chances(scores, 4, 3, targets, PLAIN, 11)
    extra = clear_chances(scores, 4, 3, targets, StrengthConfig(discards="extra_draws", discard_weight=1.0), 11)
    more = clear_chances(scores, 4, 3, targets, StrengthConfig(discards="extra_draws", discard_weight=2.0), 11)
    assert all(a <= b <= c for a, b, c in zip(none, extra, more)) and sum(none) < sum(extra) < sum(more)
    # the best `hands` of hands + discards draws: with constant scores nothing changes
    assert (round_totals([50.0] * 8, 4, 3, StrengthConfig(), np.random.default_rng(0)) == 200.0).all()
    # no discards per round: the two models agree exactly
    assert clear_chances(scores, 4, 0, targets, StrengthConfig(), 11) == none
    scaled = clear_chances(scores, 4, 3, [2 * t for t in targets], StrengthConfig(discards="none", score_scale=2.0), 11)
    assert scaled == none
    with pytest.raises(ValueError):
        Strength(StrengthConfig(discards="magic"))
    with pytest.raises(ValueError):
        Strength(StrengthConfig(score="magic"))


# ------------------------------------------------------------------ targets
def test_targets_real_boss_now_standard_bosses_later():
    g = white(4, ante=2)
    g.boss = "hook"                                              # x2
    hook = boss_targets(g)
    assert hook == [2 * b for b in BLIND_BASE[1][1:]] and len(hook) == 7
    g.boss = "wall"                                              # x4 on this ante only
    wall = boss_targets(g)
    assert wall[0] == 2 * hook[0] == g.blind_target(2) and wall[1:] == hook[1:]
    g.boss = "needle"                                            # x1
    assert boss_targets(g)[0] == hook[0] // 2
    rep_hook = Strength().report(_with_boss(g, "hook"))
    rep_wall = Strength().report(_with_boss(g, "wall"))
    assert rep_wall["targets"][0] == 2 * rep_hook["targets"][0] and rep_wall["targets"][1:] == rep_hook["targets"][1:]
    assert rep_wall["clear"][0] <= rep_hook["clear"][0] and rep_wall["clear"][1:] == rep_hook["clear"][1:]
    assert rep_hook["antes"] == [2, 3, 4, 5, 8]
    # the last ante: only its own (real) boss is left, and the horizon repeats it
    g8 = white(4, ante=8)
    g8.boss = "violet_vessel"                                    # x6
    assert boss_targets(g8) == [6 * BLIND_BASE[1][7]]
    assert horizon_antes(8) == [8] * N_CLEAR and Strength().report(g8)["targets"] == [6 * BLIND_BASE[1][7]] * N_CLEAR
    # stake scaling and the Plasma deck (x2 on every blind)
    gold = Game(seed=1, stake="GOLD")
    assert boss_targets(gold)[1:] == [2 * b for b in BLIND_BASE[3][1:]]
    plasma = Game(seed=1, stake="WHITE", deck_type="PLASMA")
    assert boss_targets(plasma)[0] == plasma.blind_target(2) and boss_targets(plasma)[1] == 4 * BLIND_BASE[1][1]


def _with_boss(g, boss):
    g2 = copy.deepcopy(g)
    g2.boss = boss
    return g2


# ------------------------------------------------------------------ survives through ante, the score
def test_survives_through_ante():
    assert survives_through(2, [0.9, 0.6, 0.4, 0.8, 0.0, 0.0, 0.0]) == 3     # stops at the first ante below 50%
    assert survives_through(2, [0.49, 1.0]) == 1                           # not even this one
    assert survives_through(5, [0.5, 0.5, 0.5, 0.5]) == 8                  # 50% counts; capped at ante 8
    assert survives_through(1, [1.0] * 8) == 8
    g = white(1, ["joker"])
    rep = Strength().report(g)
    assert rep["survives"] == survives_through(1, rep["clear_all"]) and 0 <= rep["survives"] < N_SURVIVE
    assert len(rep["clear"]) == N_CLEAR == len(HORIZON) + 1 and len(rep["clear_all"]) == 8
    assert rep["clear"][:4] == rep["clear_all"][:4] and rep["clear"][4] == rep["clear_all"][7]


def test_strength_score_kinds_are_in_unit_range():
    p = [0.9, 0.5, 0.2, 0.0, 0.0, 0.0]                                     # ante 3 .. 8
    alive = [0.9, 0.45, 0.09, 0.0, 0.0, 0.0]
    assert abs(strength_score(3, p, "expected_antes") - sum(alive) / 6) < 1e-12
    assert abs(strength_score(3, p, "expected_total") - (2 + sum(alive)) / 8) < 1e-12
    assert abs(strength_score(3, p, "horizon") - sum(alive[:4]) / 4) < 1e-12
    assert strength_score(3, p, "current") == 0.9
    for kind in ("expected_antes", "expected_total", "horizon", "current"):
        assert strength_score(1, [1.0] * 8, kind) == 1.0
        assert strength_score(8, [1.0], kind) == 1.0
        lo = strength_score(1, [0.0] * 8, kind)
        assert lo == 0.0
        assert 0.0 <= Strength(StrengthConfig(score=kind)).value(white(2, ["joker"])) <= 1.0
    with pytest.raises(ValueError):
        strength_score(1, [1.0], "magic")


def test_a_stronger_build_is_stronger():
    weak = Strength().report(white(9))
    strong = Strength().report(white(9, ["cavendish", "joker", "gros_michel"]))
    assert strong["score"] > weak["score"] and strong["survives"] >= weak["survives"]
    assert all(a >= b for a, b in zip(strong["clear_all"], weak["clear_all"]))


# ------------------------------------------------------------------ no leakage, determinism, cache
def test_strength_ignores_the_unseen_deck_order_and_the_game_rng():
    g = white(1, ["green_joker", "joker"])
    g.jokers[0].state["val"] = 4
    g.select_blind()
    g.discard([0, 1, 2])
    base = Strength().report(g)
    assert 0.0 < base["score"] < 1.0                                # not a degenerate case
    for k in range(3):
        g2 = copy.deepcopy(g)
        random.Random(k).shuffle(g2.deck)
        g2.rng = random.Random(100 + k)
        assert Strength().report(g2) == base
        assert Strength().clear_chance(g2, 450) == Strength().clear_chance(g, 450)
    state = g.rng.getstate()
    order = [c.uid for c in g.deck]
    Strength().report(g)                                           # ... and it does not touch them either
    assert g.rng.getstate() == state and [c.uid for c in g.deck] == order


def test_strength_reads_the_build_as_a_fresh_round():
    g = white(3, ["mystic_summit", "acrobat", "banner"])
    g.select_blind()
    before = Strength().report(g)
    g2 = copy.deepcopy(g)
    g2.discards_left, g2.hands_left = 0, 1                       # Mystic Summit / Acrobat would trigger now
    g2.hand_played_round[1] = 2
    g2.chips = 123
    assert Strength().report(g2) == before


def test_same_build_same_strength_and_it_is_cached():
    g = white(2, ["joker"])
    s = Strength()
    a = s.report(g)
    assert Strength().report(copy.deepcopy(g)) == a              # seeded from the state: no sampling noise
    assert s.report(g) is a and s.stats["hits"] == 1 and s.stats["calls"] == 2
    g.hand_levels[1] += 3                                        # scoring-relevant change -> recomputed
    assert s.report(g) != a and s.stats["calls"] - s.stats["hits"] == 2
    g.boss = "wall" if g.boss != "wall" else "hook"              # the target is part of the cache key
    s.report(g)
    assert s.stats["calls"] - s.stats["hits"] == 3


def test_strength_samples_the_headrooms_hands():
    g = white(5, ["joker", "jolly"])
    assert Strength().best_scores(g) == Headroom().best_scores(g)


# ------------------------------------------------------------------ Phi flag
def test_default_phi_is_unchanged_and_never_computes_the_strength():
    g = white(6, ["joker"])
    p = Potential()
    assert p.cfg.head_term == "headroom"
    assert p(g) == p.cfg.w_head * Headroom().value(g) + p.cfg.w_prog * g.furthest_blind / 24.0
    assert p.components(g)["headroom"] == Headroom().value(g) and p._strength is None
    with pytest.raises(ValueError):
        Potential(PotentialConfig(head_term="magic"))


def test_phi_with_the_strength_term_range_and_terminal_zero():
    cfg = PotentialConfig(head_term="strength")
    p = Potential(cfg)
    g = white(6, ["joker"])
    g.furthest_blind = 5
    s = Strength(cfg.strength).value(g)
    assert p(g) == cfg.w_head * s + cfg.w_prog * 5 / 24.0
    assert 0.0 <= p(g) <= cfg.w_head + cfg.w_prog == 1.0
    comp = p.components(g)
    assert comp["headroom"] == s and comp["phi"] == p(g) and comp["prog"] == 5 / 24.0
    assert p.headroom.stats["calls"] == 0                         # the headroom is not computed for Phi
    for state in ("GAME_OVER", "WON"):
        g.state = state
        assert p(g) == 0.0 and p.components(g)["phi"] == 0.0


def test_shaping_with_the_strength_term_telescopes(tmp_path):
    """The PPO path with head_term: strength: shaped minus unshaped return is exactly -Phi(s0), gamma = 1."""
    from balatro_rl.env import BalatroEnv
    from balatro_rl.heuristic import HeuristicPolicy
    cfgfile = tmp_path / "rewards.yaml"
    cfgfile.write_text("rewards:\n  potential:\n    head_term: strength\n    strength:\n      score: horizon\n")
    pcfg = RewardConfig.load(str(cfgfile)).potential
    assert pcfg.head_term == "strength" and pcfg.strength.score == "horizon" and pcfg.strength.samples == 32

    def run(env, seed):
        pol = HeuristicPolicy(rng=np.random.default_rng(0))
        obs, done, tot, phis = env.reset(seed), False, 0.0, [env.phi]
        while not done:
            obs, r, done, _ = env.step(pol.act(env.g, obs))
            tot += r
            phis.append(env.phi)
        return tot, phis
    for seed in range(2):
        plain, _ = run(BalatroEnv("RED", "WHITE"), seed)
        env = BalatroEnv("RED", "WHITE", shape_phi=1.0, shape_gamma=1.0, reward_config=str(cfgfile))
        assert env.build.uses_strength
        shaped, phis = run(env, seed)
        phi0 = Potential(pcfg)(Game(seed=seed, stake="WHITE"))
        assert phi0 > 0 and phis[0] == phi0 and phis[-1] == 0.0
        assert abs(shaped - plain + phi0) < 1e-9
        assert all(0.0 <= x <= 1.0 for x in phis)


# ------------------------------------------------------------------ aux head flag
def test_the_strength_head_leaves_every_other_weight_alone():
    from balatro_rl.az.net import AZNet, N_HEADS, N_STRENGTH
    from balatro_rl.az.train import new_net
    torch.manual_seed(0)
    plain = AZNet()
    torch.manual_seed(0)
    default = new_net(PotentialConfig())                          # the flag is off by default
    torch.manual_seed(0)
    with_head = new_net(PotentialConfig(strength=StrengthConfig(aux_head=True)))
    a, b, c = plain.state_dict(), default.state_dict(), with_head.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert "strength_head" not in plain.config and default.strength_head is None
    assert set(c) - set(a) == {"strength_head.weight", "strength_head.bias"}
    assert all(torch.equal(a[k], c[k]) for k in a)                # same seed, same weights everywhere else
    assert with_head.config["strength_head"] is True
    assert tuple(c["strength_head.weight"].shape) == (N_STRENGTH, 128) and N_STRENGTH == N_SURVIVE + N_CLEAR
    out = torch.randn(3, N_HEADS + N_STRENGTH)
    phi = torch.zeros(3)
    h = with_head.split_heads(out, phi)
    assert h["survives_probs"].shape == (3, N_SURVIVE) and h["boss_clear"].shape == (3, N_CLEAR)
    assert bool(((h["survives"] >= 0) & (h["survives"] <= 8)).all())
    old = plain.split_heads(out[:, :N_HEADS], phi)                # the old heads read the same columns
    assert torch.equal(old["ante_probs"], h["ante_probs"]) and "survives" not in old


def test_strength_targets_are_recorded_and_trained_only_with_the_flag(tmp_path):
    from balatro_rl.az.agent import Agent, AgentConfig
    from balatro_rl.az.net import load_net, save_net
    from balatro_rl.az.train import new_net, play_game, train_on
    rcfg = RewardConfig.from_dict({"potential": {"strength": {"aux_head": True}}})
    assert rcfg.aux_loss_weights.strength_survive > 0 and rcfg.aux_loss_weights.strength_clear > 0
    torch.manual_seed(0)
    net = new_net(rcfg.potential).eval()
    agent = Agent(net, AgentConfig(search=False), seed=0, potential=rcfg.potential)
    rows, info = play_game(agent, 3)
    assert rows and all(0 <= r["str_survive"] < N_SURVIVE and len(r["str_clear"]) == N_CLEAR for r in rows)
    calc = Strength(rcfg.potential.strength)
    assert all(0.0 <= p <= 1.0 for r in rows for p in r["str_clear"])
    assert agent.potential.strength.stats["calls"] == len(rows)
    node_heads = agent.evaluate(_world(4), True, random.Random(0)).heads
    assert 0.0 <= node_heads["survives"] <= 8.0 and "boss_clear" not in node_heads      # scalars only
    before = net.strength_head.weight.detach().clone()
    logs, _ = train_on(net, rows, 2, 8, 1e-3, "cpu", rcfg, rcfg.schedule(0))
    assert logs["strength_survive"] > 0 and logs["strength_clear"] > 0
    assert not torch.equal(before, net.strength_head.weight)
    rows[0].pop("str_survive")                                    # rows recorded without the targets are skipped
    train_on(net, rows[:4], 1, 4, 1e-3, "cpu", rcfg, rcfg.schedule(0))
    path = str(tmp_path / "n.pt")
    save_net(net, path)
    again = load_net(path)
    assert again.strength_head is not None and torch.equal(again.strength_head.weight, net.strength_head.weight)
    # default: no targets recorded, no strength computed, no such loss
    torch.manual_seed(0)
    plain_cfg = RewardConfig()
    plain = Agent(new_net(plain_cfg.potential).eval(), AgentConfig(search=False), seed=0)
    prow, pinfo = play_game(plain, 3)
    assert pinfo["blinds"] == info["blinds"] and len(prow) == len(rows)       # the same game either way
    assert all("str_survive" not in r and "str_clear" not in r for r in prow) and plain.potential._strength is None
    plogs, _ = train_on(plain.net, prow, 1, 8, 1e-3, "cpu", plain_cfg, plain_cfg.schedule(0))
    assert "strength_survive" not in plogs
    assert calc.report(Game(seed=3, stake="WHITE"))["survives"] >= 0


def _world(seed):
    from balatro_rl.az.world import World
    g = Game(seed=seed, stake="WHITE")
    g.select_blind()
    return World(g)


# ------------------------------------------------------------------ projected growth
def test_growth_is_off_by_default_and_the_builtin_table_is_empty():
    g = white(7, ["green_joker", "joker"])
    base = Strength().report(g)
    assert growth.DEFAULT_TABLE == {} and StrengthConfig().growth is False
    assert Strength(StrengthConfig(growth=True)).report(g) == base           # empty table: nothing moves


def test_growth_projects_scaling_jokers_forward(tmp_path):
    import json
    table = tmp_path / "t.json"
    table.write_text(json.dumps({"green_joker": {"gain": 6.0, "n": 50, "lo": 0.0}, "ice_cream": -20.0}))
    assert growth.load_table(str(table)) == {"green_joker": (6.0, 0.0), "ice_cream": (-20.0, 0.0)}
    assert growth.discounted_rounds(3, 0.5) == 1.75 and growth.discounted_rounds(0, 0.9) == 0.0
    g = white(7, ["green_joker", "ice_cream", "joker"])
    v0, ice0 = g.jokers[0].state["val"], g.jokers[1].state["val"]
    p = growth.project(g, growth.load_table(str(table)), 3, 0.5)
    assert p.jokers[0].state["val"] == v0 + 6.0 * 1.75 and g.jokers[0].state["val"] == v0     # g untouched
    assert p.jokers[1].state["val"] == max(0.0, ice0 - 20.0 * 1.75)
    far = growth.project(g, growth.load_table(str(table)), 30, 1.0)
    assert far.jokers[1].state["val"] == 0.0                     # a falling value stops at its floor
    cfg = StrengthConfig(growth=True, growth_table=str(table), growth_discount=1.0)
    off, on = Strength().report(g), Strength(cfg).report(g)
    assert sum(on["clear_all"][1:]) > sum(off["clear_all"][1:])  # later antes: the joker has grown by then
    assert all(a >= b for a, b in zip(on["clear_all"][2:], off["clear_all"][2:]))
    # deterministic, and still blind to the deck order and the game's RNG
    g2 = copy.deepcopy(g)
    random.Random(1).shuffle(g2.deck)
    g2.rng = random.Random(5)
    assert Strength(cfg).report(g2) == on
    # rounds until this ante's boss count: in the boss round itself nothing is projected for it
    g3 = copy.deepcopy(g)
    g3.blind_idx = 2
    assert Strength(cfg).report(g3)["clear_all"][0] == off["clear_all"][0]


def test_growth_table_from_logged_games():
    lines = []
    for game in range(3):
        for rnd in range(12):
            jokers = [[100 + game, "green_joker", 2.0 * rnd], [200 + game, "ice_cream", 100.0 - 5 * rnd],
                      [400 + game, "loyalty_card", float(rnd % 6)]]
            if rnd >= 6:
                jokers.append([300 + game, "runner", 15.0 * (rnd - 6)])         # bought later: 5 gains per game
            lines.append({"game": game, "round": rnd, "jokers": jokers})
    random.Random(0).shuffle(lines)
    t = growth.build_table(lines, min_count=20)
    assert t["green_joker"] == {"gain": 2.0, "n": 33, "lo": 0.0}
    assert t["ice_cream"] == {"gain": -5.0, "n": 33, "lo": 45.0} and "runner" not in t      # too few rounds
    assert "loyalty_card" not in t                               # a cycling counter is not growth
    assert growth.build_table(lines, min_count=10)["runner"]["gain"] == 15.0
    g = white(1, ["green_joker", "joker"])
    assert growth.snapshot(g) == [[g.jokers[0].uid, "green_joker", float(g.jokers[0].state["val"])]]


# ------------------------------------------------------------------ the calibration tool
def test_solver_reference_and_calibration_tables():
    from balatro_rl.rewards import strength_check as sc
    g = white(11, ["joker"])
    assert sc.solver_fresh_round(g, 1, n=10) == 1.0 and sc.solver_fresh_round(g, 10 ** 9, n=10) == 0.0
    mid = sc.solver_fresh_round(g, 600, n=40, seed=1)
    assert mid == sc.solver_fresh_round(g, 600, n=40, seed=1) and 0.0 <= mid <= 1.0
    state = g.rng.getstate()
    sc.solver_fresh_round(g, 600, n=5)
    assert g.rng.getstate() == state and not g.hand                # the game itself is not touched
    assert Strength().clear_chance(g, 1) == 1.0 and Strength().clear_chance(g, 10 ** 9) == 0.0
    rows, starts, infos, logged = sc._gen_worker(([10_000], "RED", "WHITE", 20))
    assert infos[0]["seed"] == 10_000 and rows and starts and len(logged) == len(starts)
    assert all(set(s["solver"]) == {"blind", "boss", "next"} and s["cleared"] in (0.0, 1.0) for s in starts)
    assert sum(s["cleared"] for s in starts) == infos[0]["blinds"]
    assert all(r["blinds_after"] == infos[0]["blinds"] - r["furthest"] for r in rows)
    cal = sc.clear_calibration(starts, StrengthConfig())
    assert 0.0 <= cal["solver_mae"] <= 1.0 and sum(b["n"] for b in cal["vs_actual"]) == len(starts)
    phi = sc.phi_calibration(rows, StrengthConfig())
    assert phi["phi_old"]["win"]["n"] == len(rows) and 0.0 <= phi["phi_new"]["range"][0] <= phi["phi_new"]["range"][1] <= 1.0
    pcfg = PotentialConfig(head_term="strength")
    # the tool's offline Phi is the Phi the agent would have seen (the game's first decision: a new game)
    assert rows[0]["step"] == 0 and sc.new_phi(rows[0], pcfg.strength, pcfg)[0] == Potential(pcfg)(Game(seed=10_000, stake="WHITE"))
    assert rows[0]["phi_old"] == Potential()(Game(seed=10_000, stake="WHITE"))
    assert "Phi calibration" in sc.report({"rows": rows, "starts": starts, "infos": infos})


def test_training_run_with_the_strength_phi_and_head(tmp_path):
    """End to end through the worker processes: the nested config, the checkpoint and the log."""
    import json, subprocess, sys
    cfgfile = tmp_path / "rewards.yaml"
    cfgfile.write_text("rewards:\n  potential:\n    head_term: strength\n    value_bound: floor_sigmoid\n"
                       "    strength:\n      aux_head: true\n")
    out, data = tmp_path / "az.pt", tmp_path / "data"
    cmd = [sys.executable, "-m", "balatro_rl.az.train", "run", "--iters", "1", "--games", "2", "--workers", "1",
           "--warmup", "0", "--steps", "2", "--batch", "8", "--out", str(out), "--data", str(data),
           "--reward-config", str(cfgfile), "--eval-every", "1", "--eval-games", "1",
           "--cfg", json.dumps(dict(budget_round=4, budget_boss=4, budget_shop=4, budget_spectral=4))]
    subprocess.run(cmd, check=True, capture_output=True)
    ck = torch.load(out, weights_only=False)
    assert ck["config"]["strength_head"] is True and "strength_head.weight" in ck["state_dict"]
    saved = RewardConfig.from_dict(ck["extra"]["rewards"])
    assert saved.potential.head_term == "strength" and saved.potential.strength.aux_head is True
    row = json.loads(open(tmp_path / "az_log.jsonl").readline())
    assert row["loss_strength_survive"] > 0 and row["loss_strength_clear"] > 0
    assert row["strength_ms/decision"] > 0 and row["eval"]["strength_ms/decision"] > 0
    for cal in (row["calibration_phi"], row["eval"]["calibration_phi"]):          # Phi stays in [0, 1]
        assert all(0.0 <= b["lo"] <= b["hi"] <= 1.0 for b in cal["buckets"])
