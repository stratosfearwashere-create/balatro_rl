"""Random games with random jokers/vouchers/consumables, to catch crashes in rarely used code."""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from balatro_rl.env import (BalatroEnv, A_PLAY, A_DISC, A_SELECT, A_LEAVE, A_PSKIP, N_ACTIONS,  # noqa: E402
                            encode, MAX_HAND, RATIO_COL)
from balatro_rl.sim.game import Consumable  # noqa: E402
from balatro_rl.sim.items import VOUCHERS, TAROTS, SPECTRALS, PLANETS, DECKS  # noqa: E402
from balatro_rl.sim.jokers import JOKERS, Joker  # noqa: E402


def _spice(g, rng):
    """Give the game random jokers, vouchers and consumables."""
    keys = list(JOKERS)
    for _ in range(rng.randint(1, 6)):
        if len(g.jokers) < g.joker_slots:
            k = rng.choice(keys)
            g.add_joker(Joker(k, edition=rng.choice(["", "", "FOIL", "HOLO", "POLYCHROME", "NEGATIVE"]),
                              base_cost=JOKERS[k].cost, eternal=rng.random() < 0.1,
                              rental=rng.random() < 0.1, perishable=5 if rng.random() < 0.1 else None))
    for v in rng.sample(list(VOUCHERS), 4):
        pre = VOUCHERS[v]
        if pre:
            g.redeem_voucher(pre)
        g.redeem_voucher(v)
    pool = [("tarot", t) for t in TAROTS] + [("spectral", s) for s in SPECTRALS] + [("planet", p) for p in PLANETS]
    for _ in range(3):
        kind, name = rng.choice(pool)
        g.consumables.append(Consumable(kind, name))
    g.money += rng.randint(0, 60)


def test_fuzz_all_content():
    rng = random.Random(123)
    seen_jokers = set()
    big_hands = 0
    for game in range(150):
        env = BalatroEnv(deck=rng.choice(DECKS), stake=rng.choice(["WHITE", "GOLD"]))
        obs = env.reset(seed=game)
        _spice(env.g, rng)
        obs = env.obs = encode(env.g, env.cnt)
        seen_jokers.update(j.key for j in env.g.jokers)
        done = False
        steps = 0
        while not done and steps < 600:
            m = obs["mask"]
            legal = np.flatnonzero(m)
            assert len(legal) > 0
            st = env.g.state
            if st == "SELECTING_HAND":
                big_hands += len(env.g.hand) > 8
                assert len(env.g.hand) <= MAX_HAND
                # mostly play the highest-scoring candidate so games progress
                if rng.random() < 0.7:
                    ratio = np.where(m[A_PLAY:A_DISC], obs["af"][A_PLAY:A_DISC, RATIO_COL], -1)
                    a = int(np.argmax(ratio)) + A_PLAY
                else:
                    a = int(rng.choice(legal))
            elif st == "SHOP" and rng.random() < 0.4:
                a = A_LEAVE
            elif st == "BLIND_SELECT" and rng.random() < 0.8:
                a = A_SELECT
            else:
                a = int(rng.choice(legal))
            assert 0 <= a < N_ACTIONS and m[a]
            obs, r, done, info = env.step(a)
            steps += 1
            if not done and rng.random() < 0.05:
                _spice(env.g, rng)
                obs = env.obs = encode(env.g, env.cnt)
                seen_jokers.update(j.key for j in env.g.jokers)
    assert len(seen_jokers) > 120
    assert big_hands > 0
