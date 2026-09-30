"""Reward / value-target configuration and annealing schedules.

    cfg = RewardConfig.load("rewards.yaml")      # or RewardConfig() for the defaults below
    s = cfg.schedule(step)                        # lambda, beta (novelty), kappa (solver KL) at this step

A YAML (or JSON) file may hold the settings at the top level or under a `rewards:` key; any setting left
out keeps its default. Every temporary coefficient decays linearly and is exactly 0 from its end step on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass, asdict


@dataclass
class PotentialConfig:
    w_head: float = 0.7
    w_prog: float = 0.3
    headroom_samples: int = 32
    headroom_scale: float = 1.0
    aggregate: str = "mean"           # "mean": log of the mean best-hand score; "geomean": mean of the logs
    round_hands: bool = False         # compare E[best hand] x hands per round with the target
    value_residual: bool = True       # V(s) = Phi(s) + R(s); False: V(s) = R(s) (ablation)


@dataclass
class NoveltyConfig:
    beta: float = 0.02
    end_step: int = 3_000_000
    window: int = 200_000             # visits counted over the most recent this many decisions


@dataclass
class SolverKLConfig:
    kappa: float = 0.1
    end_step: int = 1_500_000


@dataclass
class AuxWeights:
    p_clear_blind: float = 0.5
    ante_reached: float = 0.25
    blind_score_ratio: float = 0.25
    next_headroom: float = 0.1


@dataclass
class Schedule:
    step: int
    lam: float        # z = (1 - lam) * win + lam * progress
    beta: float       # novelty bonus
    kappa: float      # solver-closeness KL weight


@dataclass
class RewardConfig:
    lambda_start: float = 0.5
    lambda_end_step: int = 2_000_000
    potential: PotentialConfig = field(default_factory=PotentialConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    solver_kl: SolverKLConfig = field(default_factory=SolverKLConfig)
    aux_loss_weights: AuxWeights = field(default_factory=AuxWeights)

    def schedule(self, step: int) -> Schedule:
        return Schedule(step, _decay(self.lambda_start, self.lambda_end_step, step),
                        _decay(self.novelty.beta, self.novelty.end_step, step),
                        _decay(self.solver_kl.kappa, self.solver_kl.end_step, step))

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, d: dict | None) -> "RewardConfig":
        d = dict(d or {})
        if "rewards" in d and isinstance(d["rewards"], dict):
            d = d["rewards"]
        return _build(cls, d)

    @classmethod
    def load(cls, path: str | None) -> "RewardConfig":
        if not path:
            return cls()
        with open(path) as f:
            text = f.read()
        if path.endswith((".yaml", ".yml")):
            import yaml
            return cls.from_dict(yaml.safe_load(text))
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict:
        return asdict(self)


def _decay(start: float, end_step: int, step: int) -> float:
    """Linear from `start` at step 0 to exactly 0 at `end_step` and after."""
    if step >= end_step or end_step <= 0:
        return 0.0
    return start * (1.0 - step / end_step)


def _build(cls, d: dict):
    kw = {}
    known = {f.name: f for f in fields(cls)}
    for k, v in d.items():
        if k not in known:
            raise KeyError(f"unknown reward setting {cls.__name__}.{k}")
        f = known[k]
        default = f.default_factory() if callable(f.default_factory) else f.default
        if is_dataclass(default) and isinstance(v, dict):
            v = _build(type(default), v)
        elif isinstance(default, bool):
            v = bool(v)
        elif isinstance(default, int) and not isinstance(default, bool):
            v = int(v)
        elif isinstance(default, float):
            v = float(v)
        kw[k] = v
    return cls(**kw)
