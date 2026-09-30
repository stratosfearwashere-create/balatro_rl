"""D: Gumbel tree search on the real simulator, with chance nodes.

Every simulation starts from a fresh resampling of the root's hidden information (world.determinize: the
draw order, face-down cards and all future randomness), so no simulation sees the true future. Taking
an action leads to a chance node; its outcomes are the distinct information sets (world.infoset_key)
the action produced: different draws, shop or pack contents, or results of a random consumable. Chance
nodes use progressive widening: a new outcome is sampled while the node has fewer than
ceil(pw_c * visits^pw_alpha) outcomes; after that an existing outcome is revisited (chosen in proportion
to how often it came up), continuing from a fresh resampling of that outcome's hidden information.
Deterministic actions keep a single outcome, and frequent outcomes get the visits.

Root: Gumbel-Top-k with Sequential Halving (Danihelka et al. 2022). Interior nodes: the deterministic
Gumbel MuZero rule argmax(pi'(a) - N(a) / (1 + sum N)), with pi' = softmax(logits + sigma(completed Q)).
The improved policy softmax(logits + sigma(completed Q)) at the root is the policy training target.

Leaf value: the network's V(s) = Phi(s) + R(s), trained towards z = (1 - lam) * win + lam * progress
(rewards/). Finished games: z itself, with lam from the reward schedule (0 once it has annealed, so
the search then backs up P(win) alone).
"""
from __future__ import annotations

import math
import random

import numpy as np

from .world import World, infoset_key


class Node:
    __slots__ = ("world", "choice", "enc", "logits", "value", "n", "w", "children", "terminal", "heads")

    def __init__(self, world, choice=None, enc=None, logits=None, value=0.0, terminal=False, heads=None):
        self.world, self.choice, self.enc, self.logits, self.value = world, choice, enc, logits, value
        self.terminal, self.heads = terminal, heads
        k = 0 if logits is None else len(logits)
        self.n = np.zeros(k)
        self.w = np.zeros(k)
        self.children = [None] * k

    def q(self) -> np.ndarray:
        return np.where(self.n > 0, self.w / np.maximum(self.n, 1), 0.0)


class Chance:
    __slots__ = ("outcomes", "counts", "n")

    def __init__(self):
        self.outcomes = {}
        self.counts = {}
        self.n = 0


class GumbelSearch:
    def __init__(self, expand, c_visit: float = 50.0, c_scale: float = 0.1, pw_c: float = 1.0,
                 pw_alpha: float = 0.5, m_root: int = 8, max_depth: int = 60):
        """expand(world, root: bool) -> Node (evaluated by the network, or terminal)."""
        self.expand = expand
        self.c_visit, self.c_scale = c_visit, c_scale
        self.pw_c, self.pw_alpha = pw_c, pw_alpha
        self.m_root = m_root
        self.max_depth = max_depth
        self.vmin, self.vmax = math.inf, -math.inf

    # ------------------------------------------------------------------ value transforms
    def _norm(self, q):
        lo, hi = self.vmin, self.vmax
        if not hi > lo:
            return np.full_like(q, 0.5, dtype=float)
        return np.clip((q - lo) / (hi - lo), 0.0, 1.0)

    def _observe(self, v: float):
        self.vmin, self.vmax = min(self.vmin, v), max(self.vmax, v)

    def completed_q(self, node: Node) -> np.ndarray:
        p = _softmax(node.logits)
        visited = node.n > 0
        q = node.q()
        if visited.any():
            wq = (p[visited] * q[visited]).sum() / max(p[visited].sum(), 1e-12)
            total = node.n.sum()
            v_mix = (node.value + total * wq) / (1.0 + total)
        else:
            v_mix = node.value
        return self._norm(np.where(visited, q, v_mix))

    def sigma(self, node: Node, cq: np.ndarray) -> np.ndarray:
        return (self.c_visit + node.n.max(initial=0)) * self.c_scale * cq

    def improved_policy(self, node: Node) -> np.ndarray:
        return _softmax(node.logits + self.sigma(node, self.completed_q(node)))

    # ------------------------------------------------------------------ simulations
    def _select_interior(self, node: Node) -> int:
        pi = self.improved_policy(node)
        return int(np.argmax(pi - node.n / (1.0 + node.n.sum())))

    def _visit(self, node: Node, a: int, w: World, rng: random.Random, depth: int) -> float:
        ch = node.children[a]
        if ch is None:
            ch = node.children[a] = Chance()
        allowed = max(1, math.ceil(self.pw_c * (ch.n + 1) ** self.pw_alpha))
        if len(ch.outcomes) < allowed or not ch.outcomes:
            w.step(node.choice.cands[a].action)
            key = infoset_key(w)
            child = ch.outcomes.get(key)
            if child is None:
                child = self.expand(w, False)
                ch.outcomes[key] = child
                ch.counts[key] = 1
                v = child.value
            else:
                ch.counts[key] += 1
                v = self._descend(child, w, rng, depth + 1)
        else:
            keys = list(ch.outcomes)
            k = rng.choices(keys, weights=[ch.counts[x] for x in keys])[0]
            child = ch.outcomes[k]
            ch.counts[k] += 1
            v = self._descend(child, child.world.determinize(rng) if not child.terminal else child.world, rng, depth + 1)
        ch.n += 1
        node.n[a] += 1
        node.w[a] += v
        self._observe(v)
        return v

    def _descend(self, node: Node, w: World, rng, depth: int) -> float:
        if node.terminal or depth >= self.max_depth or len(node.logits) == 0:
            return node.value
        return self._visit(node, self._select_interior(node), w, rng, depth)

    def run(self, root: Node, budget: int, rng: random.Random, explore: bool = True) -> tuple[int, np.ndarray]:
        """Returns (index of the chosen candidate, improved policy over the root's candidates)."""
        k = len(root.logits)
        self._observe(root.value)
        g = np.array([-math.log(-math.log(max(rng.random(), 1e-12))) for _ in range(k)]) if explore else np.zeros(k)
        if budget <= 0 or k == 1:
            return int(np.argmax(g + root.logits)), _softmax(root.logits)
        m = min(self.m_root, k, budget)
        live = list(np.argsort(-(g + root.logits))[:m])
        phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        used = 0
        for ph in range(phases):
            per = max(1, budget // (phases * len(live)))
            for _ in range(per):
                for a in live:
                    if used >= budget and ph > 0:
                        break
                    self._visit(root, int(a), root.world.determinize(rng), rng, 0)
                    used += 1
            if len(live) <= 1:
                break
            score = g + root.logits + self.sigma(root, self.completed_q(root))
            live = sorted(live, key=lambda a: -score[a])[:max(1, len(live) // 2)]
        score = g + root.logits + self.sigma(root, self.completed_q(root))
        best = max(live, key=lambda a: score[a])
        self.used = used
        return int(best), self.improved_policy(root)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    e = np.exp(x - x.max())
    return e / e.sum()
