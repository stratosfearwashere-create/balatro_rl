"""Tests for the reward shaping / value-target module (balatro_rl/rewards)."""
import copy
import math
import random

import numpy as np
import pytest

from balatro_rl.sim.cards import Card
from balatro_rl.sim.game import Game
from balatro_rl.sim.jokers import Joker
from balatro_rl.sim.scoring import Plan, score_hand
from balatro_rl.rewards.config import RewardConfig, PotentialConfig
from balatro_rl.rewards.potential import Headroom, Potential, best_play, fresh_round, scoring_key
from balatro_rl.rewards.novelty import NoveltyCounter, build_signature
from balatro_rl.rewards.targets import GameRecorder, z_target
from balatro_rl.rewards.diagnostics import HackAlarm, calibration, breakdown


def white(seed=0, jokers=()) -> Game:
    g = Game(seed=seed, stake="WHITE")
    for k in jokers:
        g.add_joker(Joker(k, base_cost=4))
    return g


# ------------------------------------------------------------------ telescoping
def _episode_phis(policy, seed: int):
    """Phi along one full episode of the base environment (Phi(terminal) = 0)."""
    from balatro_rl.env import BalatroEnv
    env = BalatroEnv("RED", "WHITE")
    pot = Potential()
    obs, done = env.reset(seed), False
    phis = [pot(env.g)]
    while not done:
        obs, _, done, _ = env.step(policy(env, obs))
        phis.append(0.0 if done else pot(env.g))
    return phis


@pytest.mark.parametrize("kind", ["random", "scripted"])
def test_shaping_telescopes_to_minus_phi0(kind):
    from balatro_rl.heuristic import HeuristicPolicy
    for seed in range(3):
        if kind == "random":
            rng = np.random.default_rng(seed)
            pol = lambda env, obs: int(rng.choice(np.flatnonzero(obs["mask"])))
        else:
            h = HeuristicPolicy(rng=np.random.default_rng(seed))
            pol = lambda env, obs: h.act(env.g, obs)
        phis = _episode_phis(pol, seed)
        F = [b - a for a, b in zip(phis, phis[1:])]              # gamma = 1
        assert phis[-1] == 0.0
        assert abs(sum(F) + phis[0]) < 1e-9


def test_env_shaping_uses_the_potential_and_telescopes():
    """The PPO path: shaped minus unshaped return over a run is exactly -Phi(s0) with gamma = 1."""
    from balatro_rl.env import BalatroEnv
    from balatro_rl.heuristic import HeuristicPolicy

    def run(env, seed):
        pol = HeuristicPolicy(rng=np.random.default_rng(0))
        obs, done, tot = env.reset(seed), False, 0.0
        while not done:
            obs, r, done, _ = env.step(pol.act(env.g, obs))
            tot += r
        return tot
    for seed in range(3):
        plain = run(BalatroEnv("RED", "WHITE"), seed)
        env = BalatroEnv("RED", "WHITE", shape_phi=1.0, shape_gamma=1.0)
        phi0 = Potential()(Game(seed=seed, stake="WHITE"))
        assert abs(run(env, seed) - plain + phi0) < 1e-9


# ------------------------------------------------------------------ invariance on a toy problem
def _toy():
    """A line of 7 states; the ends are terminal: left pays 0.7, right pays 1.0; gamma = 0.9.
    The optimal policy is: go left from state 1, right from states 2-5."""
    n, gamma = 7, 0.9
    reward = {0: 0.7, n - 1: 1.0}

    def step(s, a):
        s2 = s - 1 if a == 0 else s + 1
        return s2, reward.get(s2, 0.0), s2 in reward
    return n, gamma, step


def _phi_toy(s, n):
    return 0.0 if s in (0, n - 1) else 2.0 * (n - 1 - s) / n       # misleading: rewards moving left


def _value_iteration(shaped: bool):
    n, gamma, step = _toy()
    q = np.zeros((n, 2))
    for _ in range(500):
        for s in range(1, n - 1):
            for a in range(2):
                s2, r, term = step(s, a)
                if shaped:
                    r += gamma * _phi_toy(s2, n) - _phi_toy(s, n)
                q[s, a] = r + (0.0 if term else gamma * q[s2].max())
    return q


def _q_learning(shaped: bool, episodes=20_000, seed=0):
    n, gamma, step = _toy()
    rng = random.Random(seed)
    q = np.zeros((n, 2))
    for _ in range(episodes):
        s = rng.randrange(1, n - 1)
        for _ in range(50):
            a = rng.randrange(2) if rng.random() < 0.3 else int(q[s].argmax())
            s2, r, term = step(s, a)
            if shaped:
                r += gamma * _phi_toy(s2, n) - _phi_toy(s, n)
            target = r + (0.0 if term else gamma * q[s2].max())
            q[s, a] += 0.1 * (target - q[s, a])
            if term:
                break
            s = s2
    return q


def test_shaping_leaves_the_optimal_policy_unchanged():
    optimal = [0, 1, 1, 1, 1]                                    # states 1..5
    for solve in (_value_iteration, _q_learning):
        plain = solve(False)[1:-1].argmax(1).tolist()
        shaped = solve(True)[1:-1].argmax(1).tolist()
        assert plain == shaped == optimal, solve.__name__


# ------------------------------------------------------------------ no leakage, determinism
def test_headroom_ignores_the_unseen_deck_order():
    g = white(1, ["green_joker", "joker"])
    g.jokers[0].state["val"] = 4
    g.select_blind()
    g.discard([0, 1, 2])
    base = Headroom().raw(g)
    for k in range(3):
        g2 = copy.deepcopy(g)
        random.Random(k).shuffle(g2.deck)
        g2.rng = random.Random(100 + k)
        assert Headroom().raw(g2) == base


def test_same_build_same_phi_and_it_is_cached():
    g = white(2, ["joker"])
    h = Headroom()
    a = h.raw(g)
    assert Headroom().raw(copy.deepcopy(g)) == a                 # seeded from the state: no sampling noise
    assert h.raw(g) == a and h.stats["hits"] == 1
    g.hand_levels[1] += 3                                        # scoring-relevant change -> recomputed
    assert h.raw(g) != a and h.stats["calls"] - h.stats["hits"] == 2


def test_round_dependent_jokers_read_as_a_fresh_round():
    g = white(3, ["mystic_summit", "acrobat", "banner"])
    g.select_blind()
    before = Headroom().raw(g)
    g2 = copy.deepcopy(g)
    g2.discards_left, g2.hands_left = 0, 1                       # Mystic Summit / Acrobat would trigger now
    g2.hand_played_round[1] = 2
    assert Headroom().raw(g2) == before


def test_real_boss_target_counts():
    g = white(4)
    g.boss = "hook"                                              # x2
    hook = Headroom().raw(g)
    g.boss = "wall"                                              # x4: twice the target
    assert abs(hook - Headroom().raw(g) - math.log(2)) < 1e-9


def test_round_hands_option():
    g = white(5, ["joker"])
    one = Headroom(PotentialConfig()).raw(g)
    all_ = Headroom(PotentialConfig(round_hands=True)).raw(g)
    assert abs(all_ - one - math.log(g.round_hands())) < 1e-9


def test_phi_terminal_zero_and_bounded():
    g = white(6)
    p = Potential()
    v = p(g)
    assert -p.cfg.w_head <= v <= p.cfg.w_head + p.cfg.w_prog
    g.state = "GAME_OVER"
    assert p(g) == 0.0


# ------------------------------------------------------------------ scorer consistency
def test_headroom_uses_the_simulators_scorer():
    """On fixed hands: the headroom's best play scores what the python scorer and a real play award."""
    g = white(7, ["joker", "jolly", "blueprint", "green_joker"])
    g.jokers[3].state["val"] = 3
    probe = fresh_round(g)
    plan = Plan(probe)
    rng = random.Random(0)
    for _ in range(5):
        hand = rng.sample(probe.full_deck, 8)
        probe.hand = list(hand)
        probe.deck = [c for c in probe.full_deck if all(c is not h for h in hand)]
        score, pos = best_play(probe, plan, hand)
        played = [hand[i] for i in pos]
        held = [c for i, c in enumerate(hand) if i not in pos]
        py, _ = score_hand(probe, played, held, rng=None, commit=False, plan=plan)
        assert py == score
        real = copy.deepcopy(probe)
        real.target = 10 ** 12
        real.play(list(pos))                                     # the simulator's own play
        assert real.chips == score


# ------------------------------------------------------------------ annealing and targets
def test_annealing_reaches_exactly_zero():
    cfg = RewardConfig()
    for step in (cfg.novelty.end_step, cfg.novelty.end_step + 1, 10 ** 9):
        assert cfg.schedule(step).beta == 0.0
    for step in (cfg.solver_kl.end_step, 10 ** 9):
        assert cfg.schedule(step).kappa == 0.0
    s = cfg.schedule(cfg.lambda_end_step)
    assert s.lam == 0.0
    assert cfg.schedule(0).lam == cfg.lambda_start and 0 < cfg.schedule(cfg.lambda_end_step // 2).lam < cfg.lambda_start


def test_value_target_is_win_after_annealing():
    from balatro_rl.az.train import value_target
    cfg = RewardConfig()
    nov = NoveltyCounter()
    nov.add([("a",), ("a",), ("b",)])
    rows = [{"win": w, "progress": p, "sig": s} for w, p, s in ((0.0, 0.3, ("a",)), (1.0, 1.0, ("b",)), (0.0, 0.9, ("c",)))]
    late = cfg.schedule(max(cfg.lambda_end_step, cfg.novelty.end_step))
    for r in rows:
        assert value_target(r, late, nov) == (r["win"], 0.0)
    early = cfg.schedule(0)
    for r in rows:
        t, bonus = value_target(r, early, nov)
        assert 0.0 <= t <= 1.0 and bonus > 0.0
        assert t == min(1.0, z_target(r["win"], r["progress"], early.lam) + bonus)
    assert value_target(rows[0], early, nov)[1] < value_target(rows[2], early, nov)[1]   # rarer build, larger


def test_novelty_window_forgets():
    nov = NoveltyCounter(window=3)
    nov.add(["x", "x", "y", "z", "z"])
    assert nov.count("x") == 0 and nov.count("z") == 2
    g = white(8, ["joker", "blueprint"])
    assert build_signature(g) == (("blueprint", "joker"), -1)


def test_config_from_yaml(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("""rewards:
  lambda_start: 0.5
  lambda_end_step: 2_000_000
  potential:
    w_head: 0.7
    w_prog: 0.3
    headroom_samples: 32
    headroom_scale: 1.0
    value_residual: true
  novelty:
    beta: 0.02
    end_step: 3_000_000
  solver_kl:
    kappa: 0.1
    end_step: 1_500_000
  aux_loss_weights:
    p_clear_blind: 0.5
    ante_reached: 0.25
    blind_score_ratio: 0.25
    next_headroom: 0.1
""")
    got = RewardConfig.load(str(p))
    assert got.solver_kl.kappa == 0.1                            # the file's value, not the default
    got.solver_kl.kappa = RewardConfig().solver_kl.kappa
    assert got == RewardConfig()
    assert RewardConfig.load("balatro_rl/rewards/default.yaml") == RewardConfig()
    with pytest.raises(KeyError):
        RewardConfig.from_dict({"potential": {"w_heat": 1}})


def test_game_recorder_labels():
    from balatro_rl.az.agent import Agent, AgentConfig
    from balatro_rl.az.net import AZNet
    from balatro_rl.az.train import play_game
    agent = Agent(AZNet().eval(), AgentConfig(search=False), seed=0)
    rows, info = play_game(agent, 3, record=False)
    assert rows and all(r["win"] == float(info["won"]) for r in rows)
    assert all(0 <= r["ante_cls"] < 8 for r in rows)
    assert any(not math.isnan(r["ratio"]) for r in rows) and any(not math.isnan(r["next_head"]) for r in rows)
    for r in rows:
        assert r["clear"] == float(info["blinds"] > r["blind"])


# ------------------------------------------------------------------ diagnostics
def test_calibration_and_alarm():
    rng = np.random.default_rng(0)
    v = rng.random(2000)
    good = calibration(v, rng.random(2000) < v)
    assert good["status"] == "rises"
    bad = calibration(v, rng.random(2000) < 1 - v)
    assert bad["status"] == "FLAT_OR_FALLING"
    assert calibration(v, np.zeros(2000))["status"] == "insufficient wins"
    one_game = calibration(v, v > 0.5, games=[0] * 2000)        # many winning states, but one won game
    assert one_game["status"] == "insufficient wins" and one_game["won_games"] == 1
    alarm = HackAlarm(n=3)
    assert [alarm.update(r, 0.1) for r in (0.1, 0.2, 0.3)] == [None, None, None]
    assert alarm.update(0.4, 0.1) is not None                      # return up 3 times, win rate flat
    ok = HackAlarm(n=3)
    for r, w in ((0.1, 0.1), (0.2, 0.12), (0.3, 0.14)):
        ok.update(r, w)
    assert ok.update(0.4, 0.2) is None
    b = breakdown([{"won": False, "ante": 2, "bosses": [("wall", False)]}, {"won": True, "ante": 8, "bosses": []}])
    assert b["win_rate"] == 0.5 and b["by_boss"]["wall"]["cleared%"] == 0.0


# ------------------------------------------------------------------ lambda only fades while the agent wins
def test_lambda_holds_until_the_agent_wins():
    from balatro_rl.rewards.config import LambdaGate
    cfg = RewardConfig()
    gate = LambdaGate(0.10, 320)
    for _ in range(200):                                   # 200 iterations of 64 games without a win
        assert gate.update(0, 64, 10_000) is False
    assert gate.clock == 0
    s = cfg.schedule(2_000_000_000, gate.clock)
    assert s.lam == cfg.lambda_start                       # lambda untouched however long it trains
    assert s.beta == 0.0 and s.kappa == 0.0                # novelty and solver KL still follow the step count


def test_lambda_gate_opens_needs_enough_games_and_never_rewinds():
    from balatro_rl.rewards.config import LambdaGate
    gate = LambdaGate(0.10, 320)
    assert gate.update(64, 64, 5_000) is False and gate.rate() is None     # 100% but only 64 games: wait
    for _ in range(4):
        gate.update(7, 64, 5_000)
    assert gate.rate() == (64 + 28) / 320 and gate.clock == 5_000          # opened on the 5th iteration
    gate.update(0, 320, 5_000)                                             # the win rate collapses
    assert gate.clock == 5_000                                             # closed again, clock kept
    cfg = RewardConfig()
    lam = [cfg.schedule(0, c).lam for c in (0, 5_000, 5_000)]
    assert lam[0] > lam[1] == lam[2]                                       # lambda never rises again


def test_lambda_gate_zero_threshold_is_the_old_fixed_fade():
    from balatro_rl.rewards.config import LambdaGate
    gate = LambdaGate(0.0, 320)
    for _ in range(3):
        assert gate.update(0, 64, 1_000)
    cfg = RewardConfig()
    assert cfg.schedule(3_000, gate.clock).lam == cfg.schedule(3_000).lam


def test_training_saves_and_resumes_the_lambda_gate(tmp_path):
    import json, subprocess, sys, torch
    out, data = tmp_path / "az.pt", tmp_path / "data"
    base = [sys.executable, "-m", "balatro_rl.az.train", "run", "--games", "2", "--workers", "1", "--warmup", "5",
            "--steps", "2", "--batch", "8", "--out", str(out), "--data", str(data)]
    subprocess.run(base + ["--iters", "1"], check=True, capture_output=True)
    ck = torch.load(out, weights_only=False)["extra"]
    assert ck["lambda_clock"] == 0 and len(ck["lambda_gate"]["recent"]) == 1
    subprocess.run(base + ["--iters", "2", "--resume"], check=True, capture_output=True)
    ck = torch.load(out, weights_only=False)["extra"]
    assert len(ck["lambda_gate"]["recent"]) == 2                           # restored, then extended
    rows = [json.loads(l) for l in open(tmp_path / "az_log.jsonl")]
    assert [r["lam"] for r in rows] == [0.5, 0.5]                          # no wins yet: lambda held
    assert rows[-1]["lambda_gate"]["open"] is False
