"""Tarot and spectral targets are chosen by the policy: a targeting step after "use" or "pick"."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from balatro_rl.env import (A_DISC, A_PICK, A_SELECT, A_USE_C, BalatroEnv, N_SUB, encode,  # noqa: E402
                            subset_of)
from balatro_rl.heuristic import target_action  # noqa: E402
from balatro_rl.sim.cards import Card  # noqa: E402
from balatro_rl.sim.game import Consumable, Game, TARGETED  # noqa: E402


def _in_blind(seed=3, hand=None, consumable="lovers"):
    env = BalatroEnv("RED", "WHITE")
    env.reset(seed)
    env.step(A_SELECT)
    g = env.g
    if hand is not None:
        g.hand = hand
    g.consumables = [Consumable("tarot" if consumable not in ("aura", "cryptid") else "spectral", consumable)]
    env.obs = encode(g, env.cnt)
    return env, g


def _target_slots(obs):
    return [a for a in range(A_DISC) if obs["mask"][a]]


def test_use_enters_targeting_and_applies_to_chosen_cards():
    env, g = _in_blind(consumable="lovers")                 # lovers: exactly 1 card becomes Wild
    assert env.obs["mask"][A_USE_C]
    env.step(A_USE_C)
    assert g.targeting is not None and g.targeting["cons"].name == "lovers"
    obs = env.obs
    assert obs["mask"].sum() == len(_target_slots(obs))      # nothing but target slots
    assert all(len(subset_of(obs, a)) == 1 for a in _target_slots(obs))
    a = _target_slots(obs)[-1]
    card = g.hand[subset_of(obs, a)[0]]
    env.step(a)
    assert g.targeting is None and card.enh == "WILD" and not g.consumables


def test_death_copies_right_card_onto_left():
    hand = [Card(14, 0), Card(9, 1, enh="GLASS"), Card(5, 2), Card(3, 3)]
    env, g = _in_blind(hand=hand, consumable="death")
    env.step(A_USE_C)
    obs = env.obs
    assert all(len(subset_of(obs, a)) == 2 for a in _target_slots(obs))
    a = next(a for a in _target_slots(obs) if subset_of(obs, a) == [0, 1])
    left, right = g.hand[0], g.hand[1]
    env.step(a)
    assert (left.rank, left.suit, left.enh) == (right.rank, right.suit, right.enh) == (9, 1, "GLASS")


def test_death_needs_two_cards():
    g = Game(seed=1, stake="WHITE")
    assert not g.consumable_usable(Consumable("tarot", "death"), [Card(5, 0)])
    assert g.consumable_usable(Consumable("tarot", "death"), [Card(5, 0), Card(6, 1)])


def test_pack_pick_targeting_finishes_the_pick():
    env = BalatroEnv("RED", "WHITE")
    env.reset(4)
    g = env.g
    g.open_pack("arcana", "normal")
    g.pack_cards = [Consumable("tarot", "star"), Consumable("tarot", "hermit"), Consumable("tarot", "moon")]
    env.obs = encode(g, env.cnt)
    env.step(A_PICK)
    assert g.targeting is not None and g.targeting["from_pack"] and g.state == "PACK"
    sizes = {len(subset_of(env.obs, a)) for a in _target_slots(env.obs)}
    assert sizes == {1, 2, 3}                                 # the Star: up to 3 cards
    a = next(a for a in _target_slots(env.obs) if len(subset_of(env.obs, a)) == 3)
    chosen = [g.pack_hand[i] for i in subset_of(env.obs, a)]
    env.step(a)
    assert g.targeting is None and all(c.suit == 3 for c in chosen)
    assert g.state != "PACK"                                  # a normal arcana pack has one pick


def test_big_hand_slots_include_the_rules_choice():
    rng = np.random.default_rng(0)
    hand = [Card(int(rng.integers(2, 15)), int(rng.integers(4))) for _ in range(14)]
    env, g = _in_blind(hand=hand, consumable="sun")
    env.step(A_USE_C)
    obs = env.obs
    slots = _target_slots(obs)
    assert 0 < len(slots) <= N_SUB
    cards = g.target_cards()
    want = sorted(cards.index(c) for c in g.auto_targets("sun", cards))
    assert any(subset_of(obs, a) == want for a in slots)
    assert subset_of(obs, target_action(g, obs)) == want


def test_teacher_matches_the_fixed_rule_for_every_targeted_consumable():
    for name in sorted(TARGETED):
        env, g = _in_blind(seed=5, consumable=name)
        if not env.obs["mask"][A_USE_C]:
            continue
        env.step(A_USE_C)
        cards = g.target_cards()
        want = sorted(cards.index(c) for c in g.auto_targets(name, cards))
        assert subset_of(env.obs, target_action(g, env.obs)) == want, name


def test_strategic_env_lets_ppo_choose_targets(tmp_path):
    from balatro_rl.model import ActorCritic, save_model
    from balatro_rl.strategic import StrategicEnv
    ck = str(tmp_path / "net.pt")
    save_model(ActorCritic(), ck)
    env = StrategicEnv("RED", "WHITE", tactical=ck)
    env.reset(3)
    env.step(A_SELECT)
    g = env.g
    g.consumables = [Consumable("tarot", "lovers")]
    obs = env._decision_obs()
    env.step(A_USE_C)
    assert g.targeting is not None
    obs = env.obs
    assert obs["mask"].sum() == len(_target_slots(obs)) > 1   # the full choice, not a tactical suggestion
    env.step(_target_slots(obs)[0])
    assert g.targeting is None and not env.handed_over       # choosing targets doesn't start the blind
