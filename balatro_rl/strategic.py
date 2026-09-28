"""Strategic layer: the environment PPO sees when a tactical player handles the cards.

PPO decides at blind select, in the shop and in packs, and once at the start of each blind, where
it may use consumables (choosing their target cards), sell or reorder jokers, or start the blind. Starting the blind means taking
the tactical player's suggested first play/discard, the only card action left enabled. From then
on the tactical player plays the blind out inside step(). A run is ~40-60 strategic decisions
instead of ~250, so what a purchase leads to is far fewer steps away.

Reward: the base environment's reward earned during the step (blind weights, win bonus, near-miss
on a loss) plus `margin` x how comfortably a cleared blind was beaten (score/target - 1, capped at 1).
Observations and actions are the base environment's, so any checkpoint can drive it.
"""
from __future__ import annotations

from typing import Optional

from .env import A_SELECT, BalatroEnv
from .tactical import NetTactics, in_blind


class StrategicEnv:
    def __init__(self, deck: str = "RED", stake: str = "GOLD", tactical: str = "", margin: float = 0.2,
                 ante_weight: float = 1.0, win_bonus: float = 10.0, device: str = "cpu",
                 win_ante: int = 8, joker_pool=None):
        self.env = BalatroEnv(deck, stake, ante_weight=ante_weight, win_bonus=win_bonus,
                              win_ante=win_ante, joker_pool=joker_pool)
        self.tactics = NetTactics(tactical, device)
        self.margin = margin
        self.blind_key = None
        self.handed_over = False
        self.obs: Optional[dict] = None

    @property
    def g(self):
        return self.env.g

    def reset(self, seed: Optional[int] = None) -> dict:
        self.env.reset(seed)
        self.blind_key, self.handed_over = None, False
        return self._decision_obs()

    def step(self, a: int):
        g = self.env.g
        beaten = g.blinds_beaten
        was_targeting = g.targeting is not None
        obs, r, done, info = self.env.step(a)
        self._track_blind()
        if a < A_SELECT and not was_targeting:             # started the blind: hand over the cards
            self.handed_over = True
        while not done and self.handed_over and in_blind(self.env):
            obs, rr, done, info = self.env.step(self.tactics.act(self.env))
            r += rr
        if g.blinds_beaten > beaten and g.target > 0:
            r += self.margin * min(1.0, max(0.0, g.chips / g.target - 1.0))
        return (None if done else self._decision_obs()), r, done, info

    def _track_blind(self):
        """A new blind starts with PPO's blind-start decision, not with the tactical player."""
        g = self.env.g
        if in_blind(self.env):
            key = (g.ante, g.blind_idx, g.blinds_beaten, g.blinds_skipped)
            if key != self.blind_key:
                self.blind_key, self.handed_over = key, False

    def _decision_obs(self) -> dict:
        """The base observation, with card actions reduced to the tactical suggestion at a blind start."""
        obs = self.env.obs
        if not in_blind(self.env) or self.env.g.targeting is not None:   # PPO picks tarot targets itself
            self.obs = obs
            return obs
        self._track_blind()
        obs = dict(obs)
        mask = obs["mask"].copy()
        suggestion = self.tactics.act(self.env)
        mask[:A_SELECT] = False
        mask[suggestion] = True
        obs["mask"] = mask
        self.obs = obs
        return obs
