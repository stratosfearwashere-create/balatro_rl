"""Env::state_key: two exact-seed states at a blind select with the same key and the same money must have the same future. Random shop sequences from one shop state are made; the ones that land on the
same (key, money) by different action sequences are played on with the same continuation and must end alike."""
import numpy as np
import pytest

from balatro_rl.engine_env import CppVecEnv, _engine
from balatro_rl.env import A_LEAVE, A_SELECT

pytestmark = pytest.mark.skipif(_engine is None or not hasattr(_engine.VecEnv, "state_keys"), reason="engine not rebuilt with state_keys")


def to_a_shop(seed, ante_target, rng):
    """An exact-seed game played (mostly leaving the shops, sometimes buying a card) to a shop of at least this ante; tries neighbouring seeds when a game ends first."""
    for s in range(seed, seed + 40):
        v = CppVecEnv(1, "RED", "GOLD", seed0=s, threads=1)
        v.set_exact(True)
        v.v.restart()
        v.set_auto_round(True, 1.5, 2)
        for _ in range(400):
            m = np.asarray(v.obs["mask"])[0]
            g = np.asarray(v.obs["g"])[0]
            if m[A_LEAVE] and int(round(g[0] * 8)) >= ante_target:
                return v
            if m[A_SELECT]:
                a = A_SELECT
            elif m[A_LEAVE]:
                buys = np.flatnonzero(m[439:443])
                a = int(439 + buys[0]) if (len(buys) and rng.random() < 0.3) else A_LEAVE
            else:
                a = int(rng.choice(np.flatnonzero(m)))
            _, d, _ = v.step(np.array([a]))
            if d[0]:
                break
    return None


def continue_both(env, ids, n_blinds=3):
    """The same continuation for the games `ids`: leave nothing to chance: select (the scripted player plays the round), then in the shop buy the first legal card, then leave."""
    dead = set()                                                    # a game that ends restarts on its slot's own seed: it is out of the comparison
    for _ in range(n_blinds):
        act = np.full(env.n, -1, dtype=np.int64)
        m = np.asarray(env.obs["mask"])
        for i in ids:
            act[i] = A_SELECT if m[i, A_SELECT] else A_LEAVE
        _, d, _ = env.step(act)
        dead |= {i for i in ids if d[i]}
        m = np.asarray(env.obs["mask"])
        act = np.full(env.n, -1, dtype=np.int64)
        for i in ids:
            if m[i, A_LEAVE] and i not in dead:
                buys = np.flatnonzero(m[i, 439:443])
                act[i] = 439 + buys[0] if len(buys) else A_LEAVE
        _, d, _ = env.step(act)
        dead |= {i for i in ids if d[i]}
        m = np.asarray(env.obs["mask"])
        act = np.full(env.n, -1, dtype=np.int64)
        for i in ids:
            if m[i, A_LEAVE] and i not in dead:
                act[i] = A_LEAVE
        _, d, _ = env.step(act)
        dead |= {i for i in ids if d[i]}
    return dead


@pytest.mark.parametrize("seed,ante", [(7101, 2), (7102, 3), (7103, 2)])
def test_equal_key_and_money_means_equal_future(seed, ante):
    rng = np.random.default_rng(seed)
    v = to_a_shop(seed, ante, rng)
    if v is None:
        pytest.skip("the game ended before a shop of that ante")
    K = 1500
    pool = CppVecEnv(K, "RED", "GOLD", seed0=1, threads=1)
    pool.set_exact(True)
    pool.copy_many(v, [0] * K, list(range(K)))
    pool.set_auto_round(True, 1.5, 2)
    paths = [[] for _ in range(K)]
    for step in range(10):
        m = np.asarray(pool.obs["mask"])
        act = np.full(K, -1, dtype=np.int64)
        for i in range(K):
            if m[i, A_LEAVE]:
                legal = np.flatnonzero(m[i, 439:])
                a = A_LEAVE if (step >= 5 or rng.random() < 0.25) else int(439 + rng.choice(legal))
                act[i] = a
                paths[i].append(a)
        pool.step(act)
    at_select = [i for i in range(K) if np.asarray(pool.obs["mask"])[i, A_SELECT]]
    keys, money = pool.state_keys(at_select)
    groups = {}
    for i, k, mo in zip(at_select, keys, money):
        groups.setdefault((int(k), int(mo)), []).append(i)
    dup = [g for g in groups.values() if len(g) > 1 and len({tuple(paths[i]) for i in g}) > 1]
    assert len(dup) >= 3, "the test needs several different routes to the same state"
    members = [i for g in dup for i in g]
    dead = continue_both(pool, members)
    bad = 0
    for g in dup:
        outs = set()
        for i in g:
            if i in dead:
                outs.add(("over",))
            elif np.asarray(pool.obs["mask"])[i].any():
                ks, mo = pool.state_keys([i])
                outs.add((int(ks[0]), int(mo[0])))
            else:
                outs.add(("over",))
        if len(outs) > 1:
            bad += 1
    assert bad == 0, f"{bad} of {len(dup)} groups with the same key and money ended differently: the key is missing something"
