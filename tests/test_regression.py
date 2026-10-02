"""Regression: speed work must not change anything the agent computes.

The golden file (tests/data/golden.json) was recorded from the agent before the scorer / cache / pruning
work. On fixed seeds with fixed random generators, every decision, value, search policy, candidate score,
side effect, solver estimate and headroom must come out identical (floats compared exactly, via repr; only the network's own float32 outputs, its value and
the search policy, get a 1e-6 tolerance: they vary between torch builds).

Regenerate only on purpose, when behaviour is meant to change:
    BALATRO_REGEN_GOLDEN=1 python -m pytest tests/test_regression.py
"""
import json
import os
import random

import pytest
import torch

from balatro_rl.sim.game import Game, Consumable
from balatro_rl.sim.jokers import JOKERS, Joker
from balatro_rl.az.world import World
from balatro_rl.az.actions import enumerate_candidates
from balatro_rl.az.solver import RoundSolver
from balatro_rl.az.agent import Agent, AgentConfig
from balatro_rl.az.net import AZNet
from balatro_rl.rewards.potential import Headroom

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "golden.json")
BOSSES = ["hook", "serpent", "eye", "mouth", "psychic", "house", "wheel", "fish", "mark", "flint", "arm", "plant",
          "club", "needle", "water", "manacle", "pillar", "tooth", "ox", "wall", "cerulean_bell", "crimson_heart"]


def _r(x):
    return repr(x)


def states():
    """Fixed in-round states: every listed boss, random jokers (with state), consumables, a discard or not."""
    rng = random.Random(1234)
    keys = sorted(k for k, d in JOKERS.items() if d.rarity < 4)
    out = []
    for i, boss in enumerate(BOSSES + [None] * 8):
        g = Game(seed=5000 + i, stake="WHITE")
        g.ante = rng.randint(1, 5)
        g.new_ante()
        for k in rng.sample(keys, rng.randint(0, 5)):
            j = Joker(k, base_cost=4, edition=rng.choice(["", "", "FOIL", "HOLO", "POLYCHROME"]))
            g.add_joker(j)
            if isinstance(j.state.get("val"), (int, float)):
                j.state["val"] += rng.randint(0, 4)
        for c in rng.sample(g.full_deck, 8):
            c.enh = rng.choice(["", "", "BONUS", "MULT", "WILD", "GLASS", "STEEL", "GOLD", "LUCKY", "STONE"])
            c.seal = rng.choice(["", "", "", "RED", "GOLD", "BLUE", "PURPLE"])
            c.edition = rng.choice(["", "", "", "FOIL", "HOLO", "POLYCHROME"])
        g.hand_levels = [lv + rng.randint(0, 2) for lv in g.hand_levels]
        if rng.random() < 0.5:
            g.consumables = [Consumable("tarot", rng.choice(["empress", "death", "strength", "star"]))]
        if boss is not None:
            g.boss, g.blind_idx = boss, 2
        g.select_blind()
        if rng.random() < 0.5 and g.discards_left > 0 and g.hand:
            g.discard([0, 1])
        out.append(g)
    return out


def _cands(ch):
    return [[str(c.action), _r(c.score), _r(c.score_min), _r(c.chips), _r(c.mult), c.hand, c.clears, c.certain,
             _r(c.jdiff), _r(c.effects), _r(c.p_clear), _r(c.e_chips), _r(c.best_before), _r(c.best_after)]
            for c in ch.cands]


def record() -> dict:
    torch.set_num_threads(1)
    out = {"states": [], "trace": []}
    for i, g in enumerate(states()):
        entry = {}
        for depth in (1, 2):
            ch = enumerate_candidates(World(g), random.Random(i))
            RoundSolver(samples=6).solve(g, ch, random.Random(100 + i), depth=depth)
            entry[f"depth{depth}"] = _cands(ch)
            if depth == 1:
                entry["all_plays"] = [[str(c.action), _r(c.score), c.hand] for c in ch.all_plays]
                entry["n_legal"] = ch.n_legal
        entry["headroom"] = _r(Headroom().raw(g))
        out["states"].append(entry)
    torch.manual_seed(0)
    net = AZNet().eval()
    cfg = AgentConfig(budget_round=4, budget_boss=6, budget_shop=6, budget_spectral=8)
    for seed in (900001, 900002):
        agent = Agent(net, cfg, seed=seed)
        w = World(Game(seed=seed, stake="WHITE"))
        while not w.done:
            d = agent.decide(w, explore=True)
            if d.action is None:
                break
            out["trace"].append([seed, str(d.action), d.reason, _r(d.value), [_r(float(x)) for x in d.policy],
                                 d.sims])
            w.step(d.action)
        out["trace"].append([seed, "END", w.g.state, w.g.furthest_blind])
    return out


@pytest.fixture(scope="module")
def current():
    return record()


def test_golden(current):
    if os.environ.get("BALATRO_REGEN_GOLDEN") == "1" or not os.path.exists(GOLDEN):
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump(current, f)
        pytest.skip("golden file written")
    with open(GOLDEN) as f:
        gold = json.load(f)
    assert len(current["states"]) == len(gold["states"])
    for i, (a, b) in enumerate(zip(current["states"], gold["states"])):
        for key in b:
            assert a[key] == b[key], f"state {i}: {key} changed"
    for k, (a, b) in enumerate(zip(current["trace"], gold["trace"])):
        assert _same_decision(a, b), f"decision {k} changed: {a[:4]} vs {b[:4]}"
    assert len(current["trace"]) == len(gold["trace"])


# The network's value and search policy are float32 torch outputs: they differ in the 7th digit between
# torch builds / CPUs (BLAS kernels). They are compared with this tolerance; everything else (the decision,
# its reason, the simulation count, every solver / scorer / headroom value) must be identical.
NET_TOL = 1e-6


def _close(a: str, b: str) -> bool:
    return a == b or abs(float(a) - float(b)) <= NET_TOL


def _same_decision(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    if len(a) == 4:                                   # [seed, "END", state, furthest blind]
        return a == b
    seed, act, reason, value, policy, sims = a
    seed2, act2, reason2, value2, policy2, sims2 = b
    return ((seed, act, reason, sims) == (seed2, act2, reason2, sims2) and _close(value, value2)
            and len(policy) == len(policy2) and all(_close(x, y) for x, y in zip(policy, policy2)))
