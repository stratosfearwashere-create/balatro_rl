"""The three CMA-ES variants (cma.py) on small benchmarks: they must converge, and the ones that learn correlations must beat the diagonal one on a rotated ellipsoid."""
import math

import pytest
import torch

from balatro_rl import cma


def run(es, f, gens):
    for _ in range(gens):
        X = es.ask()
        es.tell(f(X).double())
    return float(f(es.mean[None])[0])


def sphere(X):
    return (X ** 2).sum(1)


def ellipsoid(cond):
    def f(X):
        n = X.shape[1]
        w = torch.tensor([cond ** (i / (n - 1)) for i in range(n)], dtype=X.dtype, device=X.device)
        return (w * X ** 2).sum(1)
    return f


def rotated(n, cond, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    base = ellipsoid(cond)
    return lambda X: base((X.double() @ Q).to(X.dtype))


@pytest.mark.parametrize("name", ["full", "sep", "lm"])
def test_sphere_converges(name):
    n = 40
    es = cma.make(name, torch.ones(n), 0.5, seed=1)
    assert run(es, sphere, 600) < 1e-6


def test_separable_ellipsoid_sep_and_full():
    n = 30
    f = ellipsoid(1e3)
    for name in ("sep", "full"):
        es = cma.make(name, torch.ones(n), 0.5, seed=2)
        assert run(es, f, 1500) < 1e-6


def test_full_beats_sep_on_a_rotated_ellipsoid():
    n = 20
    f = rotated(n, 1e4)
    full = run(cma.make("full", torch.ones(n), 0.5, seed=3), f, 1300)
    sep = run(cma.make("sep", torch.ones(n), 0.5, seed=3), f, 1300)
    assert full < 1e-6 and full < sep * 1e-3


def test_low_rank_beats_the_diagonal_on_a_rotated_ellipsoid():
    n = 100
    f = rotated(n, 1e3)
    lm = run(cma.make("lm", torch.ones(n), 0.5, seed=3), f, 2500)
    sep = run(cma.make("sep", torch.ones(n), 0.5, seed=3), f, 2500)
    assert lm < sep * 0.1


def test_low_rank_handles_a_larger_problem():
    n = 300
    es = cma.make("lm", torch.ones(n), 0.5, seed=4)
    assert run(es, sphere, 1500) < 1e-8


def test_state_roundtrip():
    es = cma.make("lm", torch.ones(100), 0.3, seed=5)
    run(es, sphere, 20)
    sd = es.state_dict()
    es2 = cma.make("lm", torch.zeros(100), 1.0, seed=6)
    es2.load_state_dict(sd)
    assert torch.allclose(es.mean, es2.mean) and es.sigma == es2.sigma and es.gen == es2.gen
