"""The hierarchical trainer (balatro_rl/hier.py): the returns of the two levels, the worker's reward, the goal input, and a short run."""
import json
import numpy as np
import pytest
import torch

from balatro_rl import hier


def col(*xs):
    return np.array(xs, dtype=np.float64)[:, None]               # [T, 1]


def test_worker_gae_stops_at_the_end_of_a_round():
    # three worker steps (a round that is cleared on the third), then the manager acts, then two more worker steps of the next round
    rew = col(0.0, 0.0, 1.5, 0.0, 0.2, 0.7)
    val = col(1.0, 1.0, 1.0, 0.0, 0.5, 0.5)
    done = col(0, 0, 1, 0, 0, 1).astype(float)
    is_w = np.array([[1], [1], [1], [0], [1], [1]], dtype=bool)
    adv = hier.worker_gae(rew, val, done, is_w, last_val=np.zeros(1), gamma=1.0, lam=1.0)
    assert adv[3, 0] == 0.0                                       # the manager's step: nothing for the worker
    assert np.allclose(adv[:3, 0], [0.5, 0.5, 0.5])               # return 1.5 from every step of the first round, minus the value 1.0
    assert np.allclose(adv[4:, 0], [0.7 + 0.2 - 0.5, 0.7 - 0.5])  # the second round does not see the first


def test_worker_gae_bootstraps_a_round_still_going_at_the_end_of_the_rollout():
    rew = col(0.0, 0.0)
    val = col(1.0, 1.0)
    adv = hier.worker_gae(rew, val, np.zeros((2, 1)), np.ones((2, 1), dtype=bool), last_val=np.array([3.0]), gamma=1.0, lam=1.0)
    assert np.allclose(adv[:, 0], [2.0, 2.0])


def test_manager_window_rewards_include_the_rounds_in_between():
    # M W W M W(game ends): the first manager step is paid for the round that follows it, the second for the last round and the end
    rew = col(0.0, 0.0, 1.0, 0.0, 5.0)
    val = col(2.0, 0.0, 0.0, 3.0, 0.0)
    done = col(0, 0, 0, 0, 1).astype(bool)
    is_m = np.array([[1], [0], [0], [1], [0]], dtype=bool)
    adv, win = hier.manager_gae(rew, val, done, np.zeros((5, 1), dtype=bool), is_m, last_val=np.full(1, np.nan), gamma=1.0, lam=1.0)
    assert np.allclose(win[:, 0], [1.0, 0, 0, 5.0, 0])
    assert np.allclose(adv[3, 0], 5.0 - 3.0)                      # the game ended: no value after it
    assert np.allclose(adv[0, 0], 1.0 + 3.0 - 2.0 + (5.0 - 3.0))  # gamma = lam = 1: the whole return minus the value
    assert adv[1, 0] == 0.0 and adv[4, 0] == 0.0


def test_manager_return_does_not_cross_a_game_boundary():
    # M(game 1) W(dies) | M(game 2) W W: the second game's reward must not reach the first game's manager step
    rew = col(0.0, 0.5, 0.0, 0.0, 9.0)
    val = col(1.0, 0.0, 0.0, 0.0, 0.0)
    done = col(0, 1, 0, 0, 0).astype(bool)
    is_m = np.array([[1], [0], [1], [0], [0]], dtype=bool)
    adv, win = hier.manager_gae(rew, val, done, np.zeros((5, 1), dtype=bool), is_m, last_val=np.full(1, np.nan), gamma=1.0, lam=1.0)
    assert np.isclose(win[0, 0], 0.5) and np.isclose(adv[0, 0], 0.5 - 1.0)


def test_manager_truncation_bootstraps_from_its_own_value():
    rew = col(0.0, 1.0)
    val = col(4.0, 0.0)
    done = col(0, 1).astype(bool)
    trunc = col(0, 1).astype(bool)
    is_m = np.array([[1], [0]], dtype=bool)
    adv, _ = hier.manager_gae(rew, val, done, trunc, is_m, last_val=np.full(1, np.nan), gamma=1.0, lam=1.0)
    assert np.isclose(adv[0, 0], 1.0 + 4.0 - 4.0)


def test_worker_reward_is_progress_then_a_clear():
    # a play that takes the round from 0.25 to 0.75, then one that clears it with a leftover hand out of 4
    r = hier.worker_reward(np.array([0.25, 0.75]), np.array([0.75, 0.1]), np.array([False, True]), np.array([False, False]), np.zeros(2),
                           np.array([1.0, 1.0]), np.array([0, 0]), np.array([4.0, 2.0]), np.array([4.0, 4.0]), clear_bonus=0.5, bank_w=0.3, grow_w=0.3)
    assert np.allclose(r, [0.5, (1.0 - 0.75) + 0.5])


def test_worker_reward_pays_the_posture_only_on_a_clear():
    base = dict(prog_before=np.array([0.5, 0.5, 0.5]), prog_after_obs=np.array([0.5, 0.5, 0.5]), cleared=np.array([True, True, True]),
                done=np.array([False] * 3), engine_r=np.zeros(3), w_blind=np.ones(3), hands_left=np.array([3.0] * 3), round_hands=np.array([4.0] * 3),
                clear_bonus=0.5, bank_w=0.4, grow_w=0.4)
    r = hier.worker_reward(posture=np.array([0, 1, 2]), **base)
    left = 2.0 / 4.0                                              # 3 hands left before the clearing play: 2 after it
    assert np.allclose(r, [0.5 + 0.5, 0.5 + 0.5 + 0.4 * left, 0.5 + 0.5 + 0.4 * (1 - left)])
    r = hier.worker_reward(posture=np.array([1, 1, 1]), **{**base, "cleared": np.array([False] * 3)})
    assert np.allclose(r, 0.0)


def test_worker_reward_reads_the_progress_of_a_lost_round_from_the_part_credit():
    r = hier.worker_reward(np.array([0.2]), np.array([0.0]), np.array([False]), np.array([True]), np.array([0.5 * 2.0 * 0.6]), np.array([2.0]), np.array([0]),
                           np.array([1.0]), np.array([4.0]), 0.5, 0.3, 0.3)
    assert np.allclose(r, 0.6 - 0.2)


def test_goal_input_is_zero_at_first_and_the_head_follows_the_state():
    from balatro_rl.engine_env import CppVecEnv
    from balatro_rl.model import ActorCritic
    v = CppVecEnv(4, "RED", "GOLD", seed0=1, threads=1)
    ob = {k: torch.as_tensor(np.asarray(x)) for k, x in v.obs.items() if k != "priv"}
    w = ActorCritic(attn=2, goal_in=3)
    m = ActorCritic(attn=2, goal_out=3)
    l0, _ = w(ob)
    l1, _ = w({**ob, "goal": torch.tensor([0, 1, 2, 1])})
    assert torch.allclose(l0.float().clamp(min=-1e8), l1.float().clamp(min=-1e8))
    m(ob)
    assert tuple(m.goal_logits().shape) == (4, 3)
    ActorCritic(**w.config), ActorCritic(**m.config)


def test_a_short_run_trains_and_saves(tmp_path):
    a = hier.parse(["--out", str(tmp_path / "h.pt"), "--envs", "8", "--steps", "48", "--batch", "64", "--iters", "2", "--save-every", "1", "--eval-every", "2",
                    "--eval-games", "4", "--device", "cpu", "--threads", "1", "--scenario-phases", "1,1,1", "--record-wins"])
    hier.train(a)
    rows = [json.loads(l) for l in open(tmp_path / "h_log.jsonl")]
    assert [r["iter"] for r in rows] == [1, 2]
    assert rows[1]["eval"]["games"] == 4 and (tmp_path / "h.pt").exists()
    w, m = hier.load_hier(str(tmp_path / "h.pt"))
    assert w.goal_in == 3 and m.goal_out == 3
