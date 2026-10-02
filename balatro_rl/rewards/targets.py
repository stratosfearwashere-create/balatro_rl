"""Value and auxiliary targets from a finished game.

Value target: z = (1 - lam) * win + lam * progress, progress = furthest blind beaten / 24, lam annealed to 0
(config.schedule), after which z == win. The novelty bonus (novelty.py) is added to the training target
only, clipped to [0, 1]; never to evaluation or to the auxiliary targets.

Auxiliary targets (predicted, never rewarded):
    clear       the blind this decision belongs to (the one being played, or the next one) was beaten
    ante_cls    ante reached, 8 classes: ante 1..7, and "8+" (reached ante 8 or won)
    ratio       log(final chips / required) of that blind; missing if it was skipped
    next_head   Phi_headroom at the start of the next blind that begins after this decision; missing if none
    str_survive, str_clear   (only with potential.strength.aux_head) the calculated strength of the build
                at this decision (rewards/strength.py): "survives through ante" as a class 0..8, and the
                clear chances of this ante's boss, the next three antes' and ante 8's
Missing targets are NaN and masked out of the loss.
"""
from __future__ import annotations

import math

from .novelty import build_signature
from .potential import Potential

N_ANTE_CLASSES = 8


def z_target(win: float, progress: float, lam: float) -> float:
    return (1.0 - lam) * win + lam * progress


def ante_class(g) -> int:
    if g.state == "WON":
        return N_ANTE_CLASSES - 1
    return min(max(g.ante, 1), N_ANTE_CLASSES) - 1


def blind_index(g) -> int:
    return 3 * (g.ante - 1) + min(g.blind_idx, 2)


class GameRecorder:
    """Collects, during one game, what the targets need: per-decision rows, blind starts and blind ends."""

    def __init__(self, potential: Potential):
        self.potential = potential
        self.rows: list[dict] = []
        self.starts: list[tuple] = []           # (step, blind index, headroom)
        self.ends: list[tuple] = []             # (step, blind index, chips / target)
        self.bosses: list[tuple] = []           # (boss key, cleared)

    def decision(self, w, **extra) -> dict:
        g = w.g
        comp = self.potential.components(g)
        row = {"step": w.steps, "blind": blind_index(g), "in_round": g.state == "SELECTING_HAND",
               "sig": build_signature(g), "phi": comp["phi"], "headroom": comp["headroom"],
               "prog": comp["prog"], **extra}
        if self.potential.cfg.strength.aux_head:
            rep = self.potential.strength.report(g)
            row["str_survive"] = rep["survives"]
            row["str_clear"] = list(rep["clear"])
        self.rows.append(row)
        return row

    def transition(self, prev_state: str, prev_blind: int, prev_boss: str, w):
        """Call after every step with the state before it."""
        g = w.g
        if prev_state != "SELECTING_HAND" and g.state == "SELECTING_HAND":
            self.starts.append((w.steps, blind_index(g), self.potential.headroom.value(g)))
        if prev_state == "SELECTING_HAND" and g.state != "SELECTING_HAND":
            ratio = g.chips / g.target if g.target > 0 else 0.0
            self.ends.append((w.steps, prev_blind, ratio))
            if prev_blind % 3 == 2:
                self.bosses.append((prev_boss, g.state in ("SHOP", "WON", "PACK", "BLIND_SELECT")))

    def finish(self, g) -> list[dict]:
        won = g.state == "WON"
        progress = g.furthest_blind / 24.0
        acls = ante_class(g)
        for r in self.rows:
            r["win"] = float(won)
            r["progress"] = progress
            r["ante_cls"] = acls
            r["clear"] = float(g.furthest_blind > r["blind"])
            r["ratio"] = math.nan
            for step, b, ratio in self.ends:
                if step > r["step"] and b == r["blind"]:
                    r["ratio"] = math.log(min(max(ratio, 1e-3), 1e3))
                    break
            r["next_head"] = math.nan
            for step, b, head in self.starts:
                if step > r["step"] and not (r["in_round"] and b == r["blind"]):
                    r["next_head"] = head
                    break
        return self.rows
