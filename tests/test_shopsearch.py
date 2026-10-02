"""Smoke test for shopsearch: search at strategic decisions and expert iteration.

Runs with untrained networks, so it needs no checkpoint files; the one test that uses the trained
checkpoints is skipped when they are absent."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from balatro_rl import shopsearch as S  # noqa: E402
from balatro_rl.env import N_ACTIONS  # noqa: E402
from balatro_rl.model import ActorCritic, save_model  # noqa: E402


@pytest.fixture(scope="module")
def nets():
    torch.manual_seed(0)
    model = ActorCritic().eval()                           # one untrained network plays both roles
    return S.Net(model), S.make_env(model, "WHITE")


def greedy(net):
    return lambda e, o: (int(net([o])[0][0].argmax()), None)


def test_boot_ci():
    mean, lo, hi = S.boot_ci([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5 and lo <= mean <= hi
    assert S.boot_ci([3.0] * 5) == (3.0, 3.0, 3.0)


def test_fork_redraws_only_the_unseen(nets):
    net, senv = nets
    obs = senv.reset(3)
    while senv.g.state != "SELECTING_HAND":                # up to the first blind start
        obs, _, done, _ = senv.step(int(net([obs])[0][0].argmax()))
        assert not done
    g = senv.g
    g.hand[0].hidden = True                                # as a boss that deals face down would
    seen = sorted(c.uid for c in g.hand[1:])
    unseen = sorted([c.uid for c in g.deck] + [g.hand[0].uid])
    order = [c.uid for c in g.deck]
    a, b, c = S.fork(senv, 1), S.fork(senv, 1), S.fork(senv, 2)
    assert [x.uid for x in g.deck] == order and g.hand[0].hidden      # the real game is untouched
    for f in (a, b, c):
        fg = f.env.g
        assert fg is not g and f.tactics is senv.tactics
        assert sorted(x.uid for x in fg.hand[1:]) == seen             # visible cards stay
        assert sorted([x.uid for x in fg.deck] + [fg.hand[0].uid]) == unseen
        assert fg.hand[0].hidden and not any(x.hidden for x in fg.deck)
    assert [x.uid for x in a.env.g.deck] == [x.uid for x in b.env.g.deck]   # same seed, same future
    assert [x.uid for x in a.env.g.deck] != [x.uid for x in c.env.g.deck]
    g.hand[0].hidden = False


def test_playout_matches_stepping_the_environment(nets):
    """playout() re-implements StrategicEnv.step in lockstep batches; both must give the same game."""
    net, senv = nets
    for seed in range(4):
        res, rewards, _ = S.play_game(senv, seed, greedy(net))
        assert res["decisions"] == len(rewards) >= 1
        obs = senv.reset(seed)
        first = int(net([obs])[0][0].argmax())
        copies = [S.copy.copy(senv) for _ in range(2)]
        for e in copies:
            e.env = S.copy.deepcopy(senv.env)
        ret, blinds, won = S.playout(net, copies, [first, first])
        assert ret[0] == pytest.approx(res["return"], abs=1e-6) and ret[1] == pytest.approx(ret[0])
        assert int(blinds[0]) == res["blinds"] and bool(won[0]) == res["won"]


def test_search_returns_a_legal_action(nets):
    net, senv = nets
    obs = senv.reset(11)
    search = S.StrategicSearch(net, samples=2, top_k=3, seed=0)
    a, rec = search.act(senv, obs)
    legal = np.flatnonzero(obs["mask"])
    assert a in legal
    assert rec is not None and len(rec["cands"]) == len(rec["q"]) == min(3, len(legal))
    assert set(rec["cands"]) <= set(legal) and int(rec["cands"][rec["picked"]]) == a
    assert rec["cands"][0] == legal[np.argmax(net([obs])[0][0][legal])]      # first candidate: the network's choice
    truncated = S.StrategicSearch(net, samples=2, top_k=2, horizon=1, objective="blinds")
    assert truncated.act(senv, obs)[0] in legal


def test_expert_iteration_round_trip(tmp_path, monkeypatch):
    """One searched game -> a data shard -> one epoch of training towards the search."""
    monkeypatch.setattr(S, "device", lambda: "cpu")
    torch.manual_seed(0)
    ck = str(tmp_path / "net.pt")
    save_model(ActorCritic(), ck)
    data = tmp_path / "data"
    data.mkdir()
    kw = {"samples": 2, "top_k": 2, "z": 2.0, "horizon": 0, "objective": "return"}
    games, n = S._gen_worker((ck, ck, "WHITE", [700_000], kw, str(data / "shard0.npz"), 0.995))
    assert len(games) == 1 and n >= 1
    d = np.load(data / "shard0.npz")
    assert d["cands"].shape == (n, 2) and d["q"].shape == (n, 2)
    assert d["logp"].shape == (n, N_ACTIONS) and d["obs_mask"].shape == (n, N_ACTIONS)
    out = str(tmp_path / "iter1.pt")
    S.distill(ck, str(data), out, epochs=1, batch=8, device="cpu")
    assert os.path.exists(out)


def test_with_trained_checkpoints():
    policy = "checkpoints/baseline_strategic.pt"
    if not (os.path.exists(policy) and os.path.exists(S.TACTICAL)):
        pytest.skip("needs checkpoints/baseline_strategic.pt and " + S.TACTICAL)
    net = S.Net(policy)
    senv = S.make_env(S.TACTICAL)
    res, _, _ = S.play_game(senv, S.EVAL_SEED0, greedy(net))
    assert 0 <= res["blinds"] <= 24
