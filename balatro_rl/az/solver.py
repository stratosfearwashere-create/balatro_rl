"""B: the round solver. P(clear this blind) and the expected final score for each candidate action.

Each candidate play / discard is tried on the same N sampled futures (common random numbers). A future is
an order of the cards the player hasn't seen: the draw pile plus face-down cards in hand, shuffled from a
canonical order with the agent's own random generator (never the game's). After the candidate, the rest
of the round is played by a fixed policy that uses exact scores: play a clearing hand when there is one;
otherwise discard the cards outside the best play while repeating the best play wouldn't clear in the
hands left, else play the best hand. With depth=2 the first decision after the candidate is a max node
instead: a few plays and discards are each tried on `inner` re-shuffles and the best is taken
(expectimax: sampled chance nodes, max over actions, policy playout at the leaves).

This is a prior: it optimises "clear this blind", nothing beyond it (see agent.py for how the network and
search can overrule it). Inside its futures: jokers don't scale within the round, cards drawn later are
face up, glass doesn't break; The Serpent's and The Hook's draws, The Eye, The Mouth and The Psychic are
modelled. Scores come from the compiled scorer when it is built and are cached per hand within a call.

    python -m balatro_rl.az.solver bench --games 20
"""
from __future__ import annotations

import argparse
import random
import time
from itertools import combinations

from ..sim import fastscore
from ..sim.cards import sort_hand
from ..sim.scoring import Plan
from .actions import Choice

_SUBS = {}


def _subsets(n: int) -> list[tuple]:
    s = _SUBS.get(n)
    if s is None:
        s = _SUBS[n] = [c for k in range(1, min(5, n) + 1) for c in combinations(range(n), k)]
    return s


class _Round:
    __slots__ = ("hand", "draw", "hl", "dl", "chips", "types", "mouth")

    def __init__(self, hand, draw, hl, dl, chips, types, mouth):
        self.hand, self.draw, self.hl, self.dl, self.chips = hand, draw, hl, dl, chips
        self.types, self.mouth = types, mouth

    def copy(self):
        return _Round(list(self.hand), list(self.draw), self.hl, self.dl, self.chips, self.types, self.mouth)


class RoundSolver:
    def __init__(self, samples: int = 12, depth: int = 1, inner: int = 2, top_plays: int = 10, max_steps: int = 40):
        self.samples, self.depth, self.inner, self.top_plays, self.max_steps = samples, depth, inner, top_plays, max_steps
        self.calls = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------ scoring with a cache
    def _setup(self, g):
        sg = g.clone()
        sg.rng = random.Random(0)                    # never used: nothing random happens in here
        self.sg, self.plan = sg, Plan(sg)
        self.target = g.target
        self.hand_size = g.effective_hand_size()
        self.boss = g.boss_active()
        self.cache = {}
        # one compiled scorer for the whole solve; every hand of every future is drawn from these cards
        pool = list(sg.hand) + list(sg.deck)
        self.fs = fastscore.pool_scorer(sg, self.plan, pool)
        self.pidx = {id(c): i for i, c in enumerate(pool)}
        self.base_types = (fastscore.hand_types_mask(sg.round_hand_types), sg.mouth_hand)
        return sg

    def _preds(self, r: _Round) -> list:
        """[(score, hand type, subset)] for the two best plays from r.hand, best first (all the rollouts
        use). Ties go to the earlier subset, as a stable sort by score would."""
        eye = self.boss in ("eye", "mouth")
        key = (tuple(c.uid for c in r.hand), r.hl, r.dl, len(r.draw), (r.types, r.mouth) if eye else None)
        got = self.cache.get(key)
        if got is not None:
            self.cache_hits += 1
            return got
        self.calls += 1
        if self.fs is not None and len(r.hand) <= 16:     # (The Serpent can push a simulated hand past 16)
            if eye:                             # as the uncompiled path leaves them (read by _evaluate)
                self.sg.round_hand_types, self.sg.mouth_hand = set(r.types), r.mouth
            types, mouth = (fastscore.hand_types_mask(r.types), r.mouth) if eye else self.base_types
            pidx = self.pidx
            best = self.fs.best_two([pidx[id(c)] for c in r.hand], r.hl, r.dl, len(r.draw), types, mouth)
            subs = _subsets(len(r.hand))
            out = [(sc, h, subs[k]) for sc, h, k in best]
            self.cache[key] = out
            return out
        sg = self.sg
        sg.hand, sg.hands_left, sg.discards_left, sg.deck = r.hand, r.hl, r.dl, r.draw
        if eye:
            sg.round_hand_types, sg.mouth_hand = set(r.types), r.mouth
        subs = _subsets(len(r.hand))
        preds = sg.predict_many(subs, self.plan, r.hand) if subs else []
        b1 = b2 = -1
        s1 = s2 = float("-inf")
        for i, p in enumerate(preds):
            sc = p[0]
            if sc > s1:
                b2, s2 = b1, s1
                b1, s1 = i, sc
            elif sc > s2:
                b2, s2 = i, sc
        out = [(preds[b][0], preds[b][1], subs[b]) for b in (b1, b2) if b >= 0]
        self.cache[key] = out
        return out

    # ------------------------------------------------------------------ one simulated round
    def _play(self, r: _Round, pos, score, htype, rng) -> bool:
        """Play `pos` (score already known). Returns True when the round is over."""
        r.chips += score
        r.hl -= 1
        if self.boss in ("eye", "mouth"):
            r.types = r.types | frozenset([htype])
            if r.mouth < 0:
                r.mouth = htype
        if r.chips >= self.target or r.hl <= 0:
            return True
        self._redraw(r, pos, rng)
        return not r.hand

    def _discard(self, r: _Round, pos, rng):
        r.dl -= 1
        self._redraw(r, pos, rng, after_play=False)

    def _redraw(self, r: _Round, pos, rng, after_play: bool = True):
        keep = [c for i, c in enumerate(r.hand) if i not in pos]
        if after_play and self.boss == "hook" and keep:
            for c in rng.sample(keep, min(2, len(keep))):
                keep.remove(c)
        n = 3 if self.boss == "serpent" else self.hand_size - len(keep)
        new = [r.draw.pop() for _ in range(min(max(0, n), len(r.draw)))]
        r.hand = sort_hand(keep + new)

    def _policy_step(self, r: _Round, rng) -> bool:
        preds = self._preds(r)
        if not preds:
            return True
        sc, h, pos = preds[0]
        need = self.target - r.chips
        if sc < need and r.dl > 0 and sc * r.hl < need:
            rest = [i for i in range(len(r.hand)) if i not in pos]
            if rest:
                rest.sort(key=lambda i: r.hand[i].chip_value())
                self._discard(r, tuple(sorted(rest[:5])), rng)
                return False
        return self._play(r, pos, sc, h, rng)

    def _playout(self, r: _Round, rng) -> _Round:
        for _ in range(self.max_steps):
            if self._policy_step(r, rng):
                break
        return r

    def _max_node(self, r: _Round, rng) -> tuple[float, float]:
        """depth 2: the best of a few plays and discards, each on `inner` re-shuffles of the draw pile."""
        preds = self._preds(r)
        acts = [("p", p) for p in preds[:2]]
        if r.dl > 0:
            seen = set()
            for sc, h, pos in preds[:2]:
                rest = sorted([i for i in range(len(r.hand)) if i not in pos], key=lambda i: r.hand[i].chip_value())
                d = tuple(sorted(rest[:5]))
                if d and d not in seen:
                    seen.add(d)
                    acts.append(("d", d))
        best = (-1.0, 0.0)
        for kind, x in acts:
            tot_p = tot_c = 0.0
            for _ in range(self.inner):
                q = r.copy()
                rng.shuffle(q.draw)
                if kind == "p":
                    over = self._play(q, x[2], x[0], x[1], rng)
                else:
                    self._discard(q, x, rng)
                    over = False
                if not over:
                    self._playout(q, rng)
                tot_p += q.chips >= self.target
                tot_c += q.chips
            v = (tot_p / self.inner, tot_c / self.inner)
            if v > best:
                best = v
        return best

    # ------------------------------------------------------------------ public
    def solve(self, g, choice: Choice, rng: random.Random, samples: int | None = None, depth: int | None = None):
        """Fill p_clear and e_chips on the play, discard and use candidates of `choice`."""
        n_s = samples or self.samples
        depth = depth or self.depth
        sg = self._setup(g)
        target = max(1.0, float(g.target))
        hidden = [i for i, c in enumerate(sg.hand) if c.hidden]
        pool = sorted(list(sg.deck) + [sg.hand[i] for i in hidden], key=lambda c: c.uid)
        futures = []
        for _ in range(n_s):
            order = list(pool)
            rng.shuffle(order)
            hand = list(sg.hand)
            for i in hidden:
                hand[i] = order.pop()
            futures.append((hand, order))
        start = _Round(None, None, sg.hands_left, sg.discards_left, sg.chips, frozenset(sg.round_hand_types),
                       sg.mouth_hand)
        plays = [c for c in choice.cands if c.kind == "play"]
        discs = [c for c in choice.cands if c.kind == "discard"]
        best_by_group = {}
        for c in sorted(plays, key=lambda c: -c.score):
            best_by_group.setdefault(c.signature(), c)
        chosen = set(id(c) for c in sorted(plays, key=lambda c: -c.score)[:self.top_plays])
        chosen |= {id(c) for c in best_by_group.values()}
        for c in plays:
            if c.clears:
                c.p_clear = 1.0 if c.certain or not hidden else c.p_clear
                c.e_chips = min(5.0, (g.chips + c.score) / target)
                if c.p_clear == 1.0:
                    continue
            if id(c) not in chosen and not c.clears:
                continue
            self._evaluate(c, futures, start, rng, depth, target, is_play=True, hidden=bool(hidden))
        for c in discs:
            self._evaluate(c, futures, start, rng, depth, target, is_play=False, hidden=bool(hidden))
        base = max([c.p_clear for c in plays + discs if c.p_clear >= 0], default=0.0)
        base_e = max([c.e_chips for c in plays + discs if c.e_chips >= 0], default=0.0)
        for c in choice.cands:
            if c.kind == "use":
                c.p_clear = 1.0 if c.best_after >= 1.0 else base
                c.e_chips = max(base_e, (g.chips / target) + c.best_after * (choice.need / target))

    def _evaluate(self, c, futures, start: _Round, rng, depth, target, is_play: bool, hidden: bool):
        pos = c.action.cards
        tot_p = tot_c = 0.0
        for hand, order in futures:
            r = _Round(list(hand), list(order), start.hl, start.dl, start.chips, start.types, start.mouth)
            if is_play:
                if hidden:                          # the play's cards may be face down: score this future's
                    sg = self.sg
                    sg.hand, sg.deck, sg.hands_left, sg.discards_left = r.hand, r.draw, r.hl, r.dl
                    sc, h = sg.predict_many([pos], self.plan, r.hand)[0]
                else:
                    sc, h = c.score, c.hand
                over = self._play(r, pos, sc, h, rng)
            else:
                self._discard(r, pos, rng)
                over = False
            if not over:
                if depth >= 2:
                    p, ch = self._max_node(r, rng)
                    tot_p += p
                    tot_c += ch
                    continue
                self._playout(r, rng)
            tot_p += r.chips >= self.target
            tot_c += r.chips
        n = len(futures)
        c.p_clear = tot_p / n
        c.e_chips = min(5.0, tot_c / n / target)


PyRoundSolver, _PyRound = RoundSolver, _Round          # the reference implementation, always available
try:                                   # compiled solver (python setup_cython.py build): same results
    import os as _os
    if _os.environ.get("BALATRO_PURE") == "1":
        raise ImportError
    from ._solver import RoundSolver, _Round, _subsets      # noqa: F811
    COMPILED = True
except ImportError:
    COMPILED = False


# ------------------------------------------------------------------ benchmark
def bench(games: int = 20, samples: int = 12, depth: int = 1, stake: str = "GOLD", seed0: int = 30_000):
    """Time the solver on the in-round decisions of games played by the rule-based player, and check it:
    how often the round is cleared when the solver says P(clear) is high / low."""
    import numpy as np
    from ..env import BalatroEnv
    from ..heuristic import HeuristicPolicy
    from .actions import enumerate_candidates
    from .world import World
    solver = RoundSolver(samples=samples, depth=depth)
    rng = random.Random(0)
    t_enum = t_solve = 0.0
    n = 0
    calib = []
    for k in range(games):
        env = BalatroEnv("RED", stake)
        obs = env.reset(seed0 + k)
        pol = HeuristicPolicy(rng=np.random.default_rng(k))
        done = False
        pending = []
        while not done:
            g = env.g
            if g.state == "SELECTING_HAND" and g.targeting is None:
                t = time.perf_counter()
                ch = enumerate_candidates(World(g), rng)
                t_enum += time.perf_counter() - t
                t = time.perf_counter()
                solver.solve(g, ch, rng)
                t_solve += time.perf_counter() - t
                n += 1
                best = max((c.p_clear for c in ch.cands if c.kind in ("play", "discard")), default=0.0)
                pending.append((best, g.blinds_beaten))
            before = g.blinds_beaten
            obs, _, done, _ = env.step(pol.act(g, obs))
            if g.blinds_beaten > before or done:
                cleared = g.blinds_beaten > before
                calib += [(p, cleared) for p, _ in pending]
                pending = []
    print(f"{n} in-round decisions: enumeration {1e3 * t_enum / max(n, 1):.1f} ms, solver {1e3 * t_solve / max(n, 1):.1f} ms "
          f"per decision ({samples} samples, depth {depth}); {solver.calls} hands scored, {solver.cache_hits} cache hits")
    arr = np.array(calib, dtype=float)
    if len(arr):
        for lo, hi in ((0, .2), (.2, .5), (.5, .8), (.8, 1.01)):
            m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            if m.any():
                print(f"  solver's best P(clear) in [{lo:.1f}, {min(hi, 1):.1f}): {int(m.sum()):5d} decisions, "
                      f"blind cleared (heuristic playing) {100 * arr[m, 1].mean():5.1f}%")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--games", type=int, default=20)
    b.add_argument("--samples", type=int, default=12)
    b.add_argument("--depth", type=int, default=1)
    b.add_argument("--stake", default="GOLD")
    a = p.parse_args()
    bench(a.games, a.samples, a.depth, a.stake)


if __name__ == "__main__":
    main()
