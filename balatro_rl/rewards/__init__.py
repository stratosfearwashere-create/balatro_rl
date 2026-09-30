"""Reward shaping and value targets.

The objective is P(win the run). Shaping terms either leave the optimal policy unchanged (the potential,
potential.py) or decay to exactly 0 (lambda in the value target, novelty, solver KL; config.py). Nothing
optimised rewards raw score, overkill, gold held or leftover hands / discards; those only appear as
auxiliary predictions (targets.py). Held-out win rate is the measure of success (diagnostics.py).

    config.py       RewardConfig (YAML / JSON) and annealing schedules
    potential.py    Phi(s) = w_head * tanh(headroom / scale) + w_prog * blinds / 24, cached, leak-free
    targets.py      z = (1 - lam) * win + lam * progress, auxiliary targets from a finished game
    novelty.py      build signatures and the rolling visit counts for the novelty bonus
    diagnostics.py  calibration, results by ante / boss, reward-hacking alarm
"""
from .config import RewardConfig
from .potential import Potential
