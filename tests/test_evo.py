"""The evolved shop policy (evo.py): the batched forward pass equals the per-candidate one, candidates meet identical games, and a generation runs."""
import numpy as np
import pytest
import torch

from balatro_rl import evo
from balatro_rl.engine_env import CppVecEnv, _engine

pytestmark = pytest.mark.skipif(_engine is None or not hasattr(_engine.VecEnv, "set_auto_round"), reason="engine not built with the shop-only game")


def test_weight_count_and_layout():
    net = evo.ShopNet(64, 8)
    assert net.n == sum(int(np.prod(sh)) for _, sh in net.shapes)
    W = net.init(0)[None].repeat(3, 1)
    p = net.parts(W)
    assert p["W1"].shape == (3, net.S, 64) and p["emb"].shape[-1] == 8


def test_batched_forward_matches_one_by_one():
    net = evo.ShopNet(32, 8)
    P, M = 3, 2
    g = torch.Generator().manual_seed(1)
    W = torch.randn(P, net.n, generator=g) * 0.1
    args = dict(g=torch.rand(P, M, net.G, generator=g), jf=torch.rand(P, M, evo.MAX_JOKERS, evo.F_JOKER_W, generator=g),
                jid=torch.randint(0, 50, (P, M, evo.MAX_JOKERS), generator=g), cid=torch.randint(0, 50, (P, M, 3), generator=g),
                af=torch.rand(P, M, evo.NA, evo.F_ACT_W, generator=g), aid=torch.randint(0, 50, (P, M, evo.NA), generator=g),
                mask=torch.rand(P, M, evo.NA, generator=g) > 0.5)
    both = net.forward(W, **args)
    for p in range(P):
        one = net.forward(W[p:p + 1], **{k: v[p:p + 1] for k, v in args.items()})
        assert torch.allclose(both[p], one[0], atol=1e-5)


def test_candidates_meet_the_same_games_and_a_run_repeats():
    net = evo.ShopNet(32, 8)
    ev = evo.Evaluator(net, 4, 3, dev="cpu", threads=1)
    w = net.init(3)
    W = torch.stack([w, w, w + 0.05 * torch.randn(net.n), w])
    f1, b1, _ = ev.run(W, seed0=77)
    f2, b2, _ = ev.run(W, seed0=77)
    assert np.array_equal(b1, b2)                                         # the same weights and seeds: the same games
    assert np.array_equal(b1[0], b1[1]) and np.array_equal(b1[0], b1[3])  # identical candidates: identical outcomes (they met the same games)
    assert torch.isfinite(f1).all()
