"""Plays random games and writes the whole game state after every step, for tests/test_differential.py.

    python -m tests.diffplay --games 10 --out /tmp/a.jsonl                 # whichever backend is built
    BALATRO_PURE=1 python -m tests.diffplay --games 10 --out /tmp/b.jsonl  # the pure-Python reference

The backend is chosen at import time (BALATRO_PURE=1 selects the Python versions of every compiled
extension), so each one runs in its own process; the test compares the two files line by line. The policy
is test_fuzz's: random jokers, vouchers and consumables are injected, most plays are the highest-scoring
candidate, the rest of the actions are uniform over the legal ones. Everything random comes from one
random.Random seeded per game, so both backends make the same moves as long as their states agree.

Each line is one JSON object: {"game", "step", "action", "state", "obs"}. "state" is snapshot(game): every
attribute of the Game, with cards, jokers, consumables, shop items and sets in a canonical form, and the
game's random generator as a digest of its state (so RNG draws must happen in the same order). "obs" is
a digest of the encoded observation (scores and masks: the scorer's results)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from balatro_rl.env import (BalatroEnv, A_PLAY, A_DISC, A_SELECT, A_LEAVE, N_ACTIONS, encode,  # noqa: E402
                            RATIO_COL)
from balatro_rl.sim.game import Consumable, Game, ShopItem  # noqa: E402
from balatro_rl.sim.cards import Card  # noqa: E402
from balatro_rl.sim.items import VOUCHERS, TAROTS, SPECTRALS, PLANETS, DECKS  # noqa: E402
from balatro_rl.sim.jokers import JOKERS, Joker  # noqa: E402

CARD_FIELDS = ("rank", "suit", "enh", "edition", "seal", "extra_chips", "debuffed", "hidden", "uid")
JOKER_FIELDS = ("key", "edition", "eternal", "perishable", "rental", "base_cost", "sell_bonus", "cost", "debuffed",
                "hidden", "uid")
CONS_FIELDS = ("kind", "name", "negative", "sell_bonus")
SHOP_FIELDS = ("kind", "key", "cost", "joker", "card", "pack")


def _num(v):
    """Numbers keep their type: an int stays an int (3), a float a float (3.0); nan / inf become strings."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return repr(v)
        return v
    return v


def canon(v):
    """A JSON-able canonical form of a value of the game (cards, jokers ... by their fields)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return _num(v)
    if isinstance(v, Card):
        return {"card": [canon(getattr(v, f)) for f in CARD_FIELDS]}
    if isinstance(v, Joker):
        return {"joker": [canon(getattr(v, f)) for f in JOKER_FIELDS], "state": canon(v.state)}
    if isinstance(v, Consumable):
        return {"cons": [canon(getattr(v, f)) for f in CONS_FIELDS]}
    if isinstance(v, ShopItem):
        return {"item": [canon(getattr(v, f)) for f in SHOP_FIELDS]}
    if isinstance(v, random.Random):
        return {"rng": hashlib.sha256(repr(v.getstate()).encode()).hexdigest()[:24]}
    if isinstance(v, dict):
        return {str(k): canon(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (set, frozenset)):
        return {"set": sorted((canon(x) for x in v), key=repr)}
    if isinstance(v, tuple):
        return {"tuple": [canon(x) for x in v]}
    if isinstance(v, list):
        return [canon(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return _num(float(v))
    if isinstance(v, np.ndarray):
        return {"array": v.tolist()}
    if hasattr(v, "__dict__"):
        return {"obj": type(v).__name__, "fields": canon(vars(v))}
    return {"repr": repr(v)}


def snapshot(g: Game) -> dict:
    """Every attribute of the game in canonical form (see canon)."""
    return canon(vars(g))


def obs_digest(obs) -> str:
    if obs is None:
        return "none"
    h = hashlib.sha256()
    for k in sorted(obs):
        a = np.asarray(obs[k])
        h.update(k.encode())
        h.update(a.astype(np.float64 if a.dtype.kind == "f" else np.int64).tobytes())
    return h.hexdigest()[:24]


def spice(g, rng):
    """Random jokers, vouchers and consumables (as tests/test_fuzz.py)."""
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


def play(seed: int, max_steps: int, write, spice_rate: float = 0.05):
    """One game with the fuzz policy; write(dict) is called after reset and after every step."""
    rng = random.Random(seed * 1000003 + 7)
    env = BalatroEnv(deck=rng.choice(DECKS), stake=rng.choice(["WHITE", "GOLD"]))
    obs = env.reset(seed=seed)
    spice(env.g, rng)
    obs = env.obs = encode(env.g, env.cnt)
    write({"game": seed, "step": 0, "action": None, "state": snapshot(env.g), "obs": obs_digest(obs)})
    done = False
    steps = 0
    while not done and steps < max_steps:
        m = obs["mask"]
        legal = np.flatnonzero(m)
        st = env.g.state
        if st == "SELECTING_HAND":
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
        obs, r, done, info = env.step(a)
        steps += 1
        if not done and rng.random() < spice_rate:
            spice(env.g, rng)
            obs = env.obs = encode(env.g, env.cnt)
        write({"game": seed, "step": steps, "action": a, "reward": _num(r), "done": done,
               "state": snapshot(env.g), "obs": obs_digest(obs)})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.out, "w") as f:
        def write(row):
            f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        for seed in range(a.seed0, a.seed0 + a.games):
            play(seed, a.steps, write)


if __name__ == "__main__":
    main()
