"""Novelty bonus beta / sqrt(N(sig)) for rarely seen builds (temporary: beta anneals to exactly 0).

sig = (sorted joker ids, most played hand type). N counts decisions with that signature among the most
recent `window` training decisions. The bonus goes into the value training target only (targets.py).
"""
from __future__ import annotations

import math
from collections import Counter, deque


def build_signature(g) -> tuple:
    most = max(range(len(g.hand_played)), key=lambda h: (g.hand_played[h], -h)) if any(g.hand_played) else -1
    return (tuple(sorted(j.key for j in g.jokers)), most)


class NoveltyCounter:
    def __init__(self, window: int = 200_000):
        self.window = window
        self.recent: deque = deque()
        self.counts: Counter = Counter()

    def add(self, sigs):
        for s in sigs:
            self.recent.append(s)
            self.counts[s] += 1
            if len(self.recent) > self.window:
                old = self.recent.popleft()
                self.counts[old] -= 1
                if self.counts[old] <= 0:
                    del self.counts[old]

    def count(self, sig) -> int:
        return self.counts.get(sig, 0)

    def bonus(self, sig, beta: float) -> float:
        if beta == 0.0:
            return 0.0
        return beta / math.sqrt(max(1, self.count(sig)))
