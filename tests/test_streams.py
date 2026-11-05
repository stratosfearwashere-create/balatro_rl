"""The two-stream reward, the fade schedule and trigger logic, and truncation versus termination (balatro_rl/streams.py, engine step_split)."""
import numpy as np
import pytest

from balatro_rl.streams import (EntropyBump, MilestoneFade, N_ANTES, ReachTracker, ReturnNorm, RunningStat, StallDetector, by_ante, explained_variance, gae)


# ------------------------------------------------------------------------------------------------------------- truncation vs termination
def _one_step(reward, value, last_value, done, trunc, gamma=0.9, lam=0.95):
    rew = np.array([[[reward]]], dtype=np.float32)
    val = np.array([[[value]]], dtype=np.float32)
    return float(gae(rew, val, np.array([[last_value]], dtype=np.float32), np.array([[done]]), np.array([[trunc]]), gamma, lam)[0, 0, 0])


def test_termination_does_not_bootstrap():
    # a loss or a win at the last step of the rollout: delta = r - V, whatever the "next" value is
    assert _one_step(1.0, 0.4, 7.0, done=True, trunc=False) == pytest.approx(1.0 - 0.4)


def test_truncation_bootstraps_from_the_value_of_its_own_state():
    # a step limit: delta = r + gamma * V(s) - V(s); the next observation (another game's) is not used
    assert _one_step(1.0, 0.4, 7.0, done=True, trunc=True) == pytest.approx(1.0 + 0.9 * 0.4 - 0.4)


def test_rollout_end_bootstraps_from_the_last_value():
    assert _one_step(1.0, 0.4, 7.0, done=False, trunc=False) == pytest.approx(1.0 + 0.9 * 7.0 - 0.4)


def test_trace_is_cut_at_an_episode_end_either_kind():
    T = 3
    for trunc in (False, True):
        rew = np.zeros((T, 1, 1), dtype=np.float32)
        rew[2, 0, 0] = 100.0                                           # a big reward in the next game
        val = np.zeros((T, 1, 1), dtype=np.float32)
        done = np.array([[False], [True], [False]])
        tr = np.array([[False], [trunc], [False]])
        adv = gae(rew, val, np.zeros((1, 1), dtype=np.float32), done, tr, 0.99, 0.95)
        assert adv[0, 0, 0] == 0.0 and adv[1, 0, 0] == 0.0 and adv[2, 0, 0] == pytest.approx(100.0)


def test_gae_matches_the_textbook_discounted_sum_without_ends():
    rng = np.random.default_rng(0)
    T, g, lam = 6, 0.97, 1.0                                           # lam = 1: advantage = discounted return - value
    rew = rng.normal(size=(T, 2, 1)).astype(np.float32)
    val = rng.normal(size=(T, 2, 1)).astype(np.float32)
    last = rng.normal(size=(2, 1)).astype(np.float32)
    adv = gae(rew, val, last, np.zeros((T, 2), dtype=bool), np.zeros((T, 2), dtype=bool), g, lam)
    ret = last.copy()
    for t in reversed(range(T)):
        ret = rew[t] + g * ret
        assert np.allclose(adv[t], ret - val[t], atol=1e-4)


# ------------------------------------------------------------------------------------------------------------------------ reward split
def test_by_ante_puts_each_reward_in_its_antes_column():
    shaped = np.array([[1.0, 2.0], [3.0, 4.0]])
    ante = np.array([[1, 8], [3, 9]])                                  # (9 would be a bug: it is clipped to the last ante)
    out = by_ante(shaped, ante)
    assert out.shape == (2, 2, N_ANTES)
    assert out[0, 0, 0] == 1.0 and out[0, 1, 7] == 2.0 and out[1, 0, 2] == 3.0 and out[1, 1, 7] == 4.0
    assert np.allclose(out.sum(-1), shaped)


def test_split_streams_add_up_to_the_engine_reward():
    from balatro_rl.engine_env import CppVecEnv
    rng = np.random.default_rng(1)
    for from_blind in (24, 6, 0):
        v = CppVecEnv(8, "RED", "WHITE", seed0=100, threads=1, reward_kw=dict(ante_weight=2.5, win_bonus=50.0, win_ante=8, shape_chips=0.0, shape_gamma=0.995))
        v.set_sparse(from_blind)
        tot = sp = 0.0
        beaten = 0
        for _ in range(400):
            mask = np.asarray(v.obs["mask"])
            act = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
            r, d, info, sparse, trunc = v.step_split(act)
            assert r.shape == sparse.shape == trunc.shape == (8,)
            assert (sparse <= r + 1e-6).all() or (r < 0).any()          # (a death penalty could make r smaller; it is off here)
            if from_blind == 24:
                assert (sparse == 0).all()                              # no win in 400 random steps
            tot += float(r.sum()); sp += float(sparse.sum())
            beaten += int((r > 0).sum())
            assert not trunc.any()                                      # a loss is not a truncation
        if from_blind == 0:
            assert sp > 0 and sp <= tot + 1e-4                          # every blind's reward is sparse
    assert beaten > 0


# ------------------------------------------------------------------------------------------------------------------------ normalisation
def test_running_stat_matches_numpy():
    x = np.random.default_rng(2).normal(3.0, 2.0, size=1000)
    s = RunningStat()
    for part in np.array_split(x, 7):
        s.update(part)
    assert s.mean == pytest.approx(x.mean(), rel=1e-3) and s.var == pytest.approx(x.var(), rel=1e-3)


def test_return_norm_has_a_floor_and_resets_at_episode_ends():
    n = ReturnNorm(2, gamma=0.5, floor=0.5)
    n.update(np.zeros((5, 2)), np.zeros((5, 2), dtype=bool))
    assert n.std == 0.5                                                 # a stream that is all zeros: the floor, not a division by ~0
    n2 = ReturnNorm(1, gamma=0.5, floor=0.0)
    rew = np.array([[1.0], [0.0], [0.0]])
    n2.update(rew, np.array([[False], [True], [False]]))
    assert n2.G[0] == 0.0                                               # the return restarted after the episode end


# --------------------------------------------------------------------------------------------------------------------------- the fade
def test_no_fade_below_the_trigger():
    f = MilestoneFade(trigger=0.25, fade_iters=10)
    for _ in range(50):
        assert f.update(np.full(N_ANTES, 0.24)) == []
    assert (f.w == 1.0).all()


def test_fade_is_linear_per_milestone_and_never_a_step():
    f = MilestoneFade(trigger=0.25, fade_iters=10)
    reach = np.zeros(N_ANTES)
    reach[0] = 0.5                                                      # only ante 1 is mastered
    begins = []
    ws = []
    for _ in range(12):
        begins += f.update(reach)
        ws.append(f.w[0])
    assert begins == [0]                                                # one begin event
    assert np.allclose(np.diff(ws[:10]), -0.1)                          # linear: a tenth per update
    assert ws[-1] == 0.0 and (f.w[1:] == 1.0).all()                    # other milestones untouched
    assert max(abs(np.diff(ws))) <= 0.1 + 1e-9


def test_fade_holds_between_the_thresholds_and_reverses_below():
    f = MilestoneFade(trigger=0.4, fade_iters=10, reverse_ratio=0.5)
    reach = np.zeros(N_ANTES)
    reach[2] = 0.5
    for _ in range(5):
        f.update(reach)
    w_mid = f.w[2]
    assert w_mid == pytest.approx(0.5)
    reach[2] = 0.3                                                      # under the trigger, over trigger * ratio (0.2): hold
    for _ in range(5):
        f.update(reach)
    assert f.w[2] == pytest.approx(w_mid)
    reach[2] = 0.1                                                      # collapsed after the fade: reverse, at the same speed
    f.update(reach)
    assert f.w[2] == pytest.approx(w_mid + 0.1)
    for _ in range(20):
        f.update(reach)
    assert f.w[2] == 1.0                                                # (never above 1)


def test_a_milestone_that_never_started_is_not_reversed():
    f = MilestoneFade(trigger=0.4, fade_iters=10)
    for _ in range(5):
        f.update(np.zeros(N_ANTES))
    assert (f.w == 1.0).all()


def test_begin_fires_once_per_milestone_even_if_the_fade_pauses_and_resumes():
    f = MilestoneFade(trigger=0.4, fade_iters=10, reverse_ratio=0.5)
    r = np.zeros(N_ANTES)
    r[0] = 0.5
    assert f.update(r) == [0]
    assert f.update(r) == []
    r[0] = 0.3                                                          # paused
    assert f.update(r) == []
    r[0] = 0.5
    assert f.update(r) == []                                            # resumed: no second entropy bump
    r[1] = 0.5
    assert f.update(r) == [1]                                           # another milestone starting: its own


def test_reach_tracker_uses_shares_not_means():
    t = ReachTracker(window=10)
    for _ in range(5):
        t.add(1, False)                                                 # half die at once ...
    for _ in range(5):
        t.add(8, True)                                                  # ... half win
    r = t.rates()
    assert r[0] == pytest.approx(0.5) and r[6] == pytest.approx(0.5) and r[7] == pytest.approx(0.5)
    assert t.ready()


def test_entropy_bump_decays_to_zero():
    b = EntropyBump(peak=0.02, decay_iters=4)
    assert b.value() == 0.0
    b.trigger()
    vals = []
    for _ in range(5):
        vals.append(b.value())
        b.step()
    assert vals == pytest.approx([0.02, 0.015, 0.01, 0.005, 0.0])


def test_explained_variance():
    y = np.arange(10, dtype=float)
    assert explained_variance(y, y) == pytest.approx(1.0)
    assert explained_variance(np.zeros(10) + y.mean(), y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------------------------------------- stall detector
def _feed(d, values, bump_active=False):
    return [d.update(v, bump_active) for v in values]


def test_no_fire_until_two_full_windows_and_none_while_improving():
    d = StallDetector(window=5, delta=1.0)
    assert all(r == (None, False) for r in _feed(d, [10.0] * 9))        # fewer than two windows
    d = StallDetector(window=5, delta=1.0)
    assert all(r[0] is None for r in _feed(d, [10.0 + i for i in range(30)]))     # a steady climb: never a stall


def test_fires_on_a_flat_curve_and_doubles_up_to_the_cap():
    d = StallDetector(window=5, delta=1.0, base=0.003, cap=0.010)
    levels = [r[0] for r in _feed(d, [40.0] * 60) if r[0] is not None]
    assert levels[:4] == pytest.approx([0.003, 0.006, 0.010, 0.010])    # doubles, then the cap


def test_waits_a_whole_window_between_bumps():
    d = StallDetector(window=5, delta=1.0)
    res = _feed(d, [40.0] * 30)
    fired_at = [i for i, r in enumerate(res) if r[0] is not None]
    assert fired_at[0] == 9 and all(b - a >= 5 for a, b in zip(fired_at, fired_at[1:]))


def test_an_improving_window_starts_the_escalation_over():
    d = StallDetector(window=5, delta=1.0, base=0.003)
    _feed(d, [40.0] * 10)                                               # fires once (k = 1)
    assert d.k == 1
    _feed(d, [60.0] * 5)                                                # the next window is far better: no stall
    assert d.k == 0


def test_no_new_bump_while_one_is_running():
    d = StallDetector(window=5, delta=1.0)
    assert all(r[0] is None for r in _feed(d, [40.0] * 30, bump_active=True))


def test_guard_cuts_a_bump_when_the_win_rate_falls():
    d = StallDetector(window=5, delta=1.0, guard=3.0, guard_len=10)
    _feed(d, [40.0] * 10)                                               # a bump fires at 40
    assert d.guard_left == 10
    assert d.update(39.0, True) == (None, False)                        # a small dip: fine
    assert d.update(35.0, True) == (None, True)                         # more than 3 under: cut
    assert d.guard_left == 0


def test_reset_clears_the_windows():
    d = StallDetector(window=5, delta=1.0)
    _feed(d, [40.0] * 10)
    d.reset()
    assert d.hist == [] and d.k == 0 and d.guard_left == 0
