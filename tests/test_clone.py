"""Game.clone must be a drop-in for copy.deepcopy(game): same contents, same sharing, no links back."""
import copy
import random

import numpy as np

from balatro_rl.env import BalatroEnv
from balatro_rl.heuristic import HeuristicPolicy
from balatro_rl.sim.cards import Card
from balatro_rl.sim.game import Consumable
from balatro_rl.sim.jokers import JOKERS, Joker

_ATOMS = (int, float, str, bool, type(None), frozenset)


UID_KEYS = {"uid", "ante_played_uids", "forced_uid", "crimson_disabled"}


def _mirror(a, b, pairs: dict, ignore_uids: bool = False):
    """a and b hold equal values with the same sharing pattern, and b shares no mutable object with a.
    ignore_uids: skip card / joker ids (new cards take ids from a global counter, so two copies played on
    one after the other number their new cards differently)."""
    if type(a) in _ATOMS or (type(a) is tuple and all(type(x) in _ATOMS for x in a)):   # immutable
        assert a == b
        return
    assert type(a) is type(b)
    if id(a) in pairs:
        assert pairs[id(a)] is b, "sharing differs"
        return
    pairs[id(a)] = b
    if isinstance(a, random.Random):
        assert a is not b and a.getstate() == b.getstate()
        return
    if hasattr(a, "__deepcopy__"):                 # fixed definitions (JokerDef) are shared on purpose
        assert a is b
        return
    assert a is not b, f"{type(a).__name__} shared with the original"
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _mirror(x, y, pairs, ignore_uids)
    elif isinstance(a, set):
        assert a == b
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            if not (ignore_uids and k in UID_KEYS):
                _mirror(a[k], b[k], pairs, ignore_uids)
    elif hasattr(type(a), "FIELDS"):               # the compiled Card keeps its fields outside __dict__:
        assert a.FIELDS == b.FIELDS                # compared one by one (a temporary dict's id gets reused)
        for k in a.FIELDS:
            if not (ignore_uids and k in UID_KEYS):
                _mirror(getattr(a, k), getattr(b, k), pairs, ignore_uids)
        ea, eb = getattr(a, "__dict__", None) or {}, getattr(b, "__dict__", None) or {}   # ad-hoc attributes
        assert ea.keys() == eb.keys()
        for k in ea:
            _mirror(ea[k], eb[k], pairs, ignore_uids)
    else:
        _mirror(vars(a), vars(b), pairs, ignore_uids)


def _states(n_games=12):
    """Game states from heuristic play, spiced with random jokers, consumables and odd flags."""
    rng = random.Random(0)
    keys = [k for k, d in JOKERS.items() if d.rarity < 4]
    for seed in range(n_games):
        env = BalatroEnv("RED", "WHITE")
        obs = env.reset(seed)
        g = env.g
        for k in rng.sample(keys, 3):
            g.add_joker(Joker(k, base_cost=4, edition=rng.choice(["", "FOIL", "NEGATIVE"])))
        g.consumables.append(Consumable("tarot", "death"))
        pol = HeuristicPolicy(rng=np.random.default_rng(seed))
        done, k = False, 0
        while not done:
            if k % 7 == 0:
                if g.state == "SELECTING_HAND" and g.hand:
                    g.flags["trading_destroy"] = g.hand[0]      # a card referenced from two places
                yield g
            obs, _, done, _ = env.step(pol.act(g, obs))
            k += 1


def test_clone_mirrors_the_game():
    n = 0
    for g in _states():
        c = g.clone()
        _mirror(vars(g), vars(c), {})
        if g.state == "SELECTING_HAND":
            full = {id(x) for x in c.full_deck}
            assert all(id(x) in full for x in c.hand if any(x.uid == y.uid for y in g.full_deck))
        n += 1
    assert n > 50


def test_clone_and_deepcopy_play_the_same_future():
    for g in _states(4):
        a, b = copy.deepcopy(g), g.clone()
        before = copy.deepcopy(g)
        env_a, env_b = BalatroEnv("RED", "WHITE"), BalatroEnv("RED", "WHITE")
        for env, x in ((env_a, a), (env_b, b)):
            env.g, env.steps = x, 0
            env.cnt = __import__("balatro_rl.env", fromlist=["Counters"]).Counters()
            env.obs = __import__("balatro_rl.env", fromlist=["encode"]).encode(x, env.cnt)
        pol = HeuristicPolicy(rng=np.random.default_rng(1))
        for _ in range(40):
            if a.done:
                break
            act = pol.act(a, env_a.obs)
            env_a.step(act)
            env_b.step(act)
            _mirror(vars(a), vars(b), {}, ignore_uids=True)
        _mirror(vars(before), vars(g), {})           # the original was never touched


def test_clone_speed():
    import time
    g = next(iter(_states(1)))
    t = time.perf_counter()
    for _ in range(200):
        copy.deepcopy(g)
    slow = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(200):
        g.clone()
    fast = time.perf_counter() - t
    assert fast < slow / 2
