# cython: language_level=3, boundscheck=False, wraparound=False
"""Compiled round solver: solver.RoundSolver with C types (used when built; solver.py keeps the reference
implementation and the benchmark). The random generator is called in exactly the same order, so P(clear)
and expected chips are bit for bit the same."""
cimport cython
import random
from itertools import combinations

from ..sim import fastscore
from ..sim.cards import sort_hand
from ..sim.scoring import Plan
from ..sim._cards cimport Card

cdef dict _SUBS = {}


def _subsets(int n):
    s = _SUBS.get(n)
    if s is None:
        s = _SUBS[n] = [c for k in range(1, min(5, n) + 1) for c in combinations(range(n), k)]
    return s


cdef class _Round:
    cdef public list hand
    cdef public list draw
    cdef public int hl
    cdef public int dl
    cdef public object chips          # a Python number: scores can exceed 2^63
    cdef public object types          # frozenset of hand types played this round
    cdef public int mouth

    def __init__(self, hand, draw, int hl, int dl, chips, types, int mouth):
        self.hand, self.draw, self.hl, self.dl, self.chips = hand, draw, hl, dl, chips
        self.types, self.mouth = types, mouth

    cdef _Round dup(self):
        cdef _Round r = _Round.__new__(_Round)
        r.hand, r.draw, r.hl, r.dl, r.chips = list(self.hand), list(self.draw), self.hl, self.dl, self.chips
        r.types, r.mouth = self.types, self.mouth
        return r

    def copy(self):
        return self.dup()


cdef inline tuple _uids(list hand):
    cdef Py_ssize_t i, n = len(hand)
    cdef list tmp = [0] * n
    for i in range(n):
        tmp[i] = (<Card>hand[i]).uid
    return tuple(tmp)


def _chip_key(c):
    return (<Card>c).chip_value()


cdef class RoundSolver:
    cdef public int samples, depth, inner, top_plays, max_steps
    cdef public long long calls, cache_hits
    cdef public object sg, plan, fs, pidx, base_types, boss
    cdef public dict cache
    cdef public double target
    cdef public int hand_size

    def __init__(self, int samples=12, int depth=1, int inner=2, int top_plays=10, int max_steps=40):
        self.samples, self.depth, self.inner, self.top_plays, self.max_steps = samples, depth, inner, top_plays, max_steps
        self.calls = 0
        self.cache_hits = 0
        self.sg = self.plan = self.fs = self.pidx = self.base_types = None
        self.boss = ""
        self.cache = {}
        self.target = 0.0
        self.hand_size = 0

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

    cpdef list _preds(self, _Round r):
        """[(score, hand type, subset)] for the two best plays from r.hand, best first (all the rollouts
        use). Ties go to the earlier subset, as a stable sort by score would."""
        cdef bint eye = self.boss == "eye" or self.boss == "mouth"
        cdef list hand = r.hand
        cdef Py_ssize_t i
        key = (_uids(hand), r.hl, r.dl, len(r.draw), (r.types, r.mouth) if eye else None)
        got = self.cache.get(key)
        if got is not None:
            self.cache_hits += 1
            return got
        self.calls += 1
        if self.fs is not None and len(hand) <= 16:     # (The Serpent can push a simulated hand past 16)
            if eye:                             # as the uncompiled path leaves them (read by _evaluate)
                self.sg.round_hand_types, self.sg.mouth_hand = set(r.types), r.mouth
            types, mouth = (fastscore.hand_types_mask(r.types), r.mouth) if eye else self.base_types
            pidx = self.pidx
            idx = [pidx[id(c)] for c in hand]
            best = self.fs.best_two(idx, r.hl, r.dl, len(r.draw), types, mouth)
            subs = _subsets(len(hand))
            out = [(sc, h, subs[k]) for sc, h, k in best]
            self.cache[key] = out
            return out
        sg = self.sg
        sg.hand, sg.hands_left, sg.discards_left, sg.deck = hand, r.hl, r.dl, r.draw
        if eye:
            sg.round_hand_types, sg.mouth_hand = set(r.types), r.mouth
        subs = _subsets(len(hand))
        preds = sg.predict_many(subs, self.plan, hand) if subs else []
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
    cdef int _play(self, _Round r, tuple pos, score, int htype, rng) except -1:
        """Play `pos` (score already known). Returns True when the round is over."""
        r.chips = r.chips + score
        r.hl -= 1
        if self.boss == "eye" or self.boss == "mouth":
            r.types = r.types | frozenset([htype])
            if r.mouth < 0:
                r.mouth = htype
        if r.chips >= self.target or r.hl <= 0:
            return True
        self._redraw(r, pos, rng, True)
        return not r.hand

    cdef void _discard(self, _Round r, tuple pos, rng) except *:
        r.dl -= 1
        self._redraw(r, pos, rng, False)

    cdef void _redraw(self, _Round r, tuple pos, rng, bint after_play) except *:
        cdef list keep = []
        cdef list hand = r.hand
        cdef Py_ssize_t i, n_draw
        cdef int n, k
        for i in range(len(hand)):
            if i not in pos:
                keep.append(hand[i])
        if after_play and self.boss == "hook" and keep:
            for c in rng.sample(keep, min(2, len(keep))):
                keep.remove(c)
        n = 3 if self.boss == "serpent" else self.hand_size - len(keep)
        n_draw = min(max(0, n), len(r.draw))
        for k in range(n_draw):
            keep.append(r.draw.pop())
        r.hand = sort_hand(keep)

    cdef int _policy_step(self, _Round r, rng) except -1:
        cdef list preds = self._preds(r)
        cdef list rest
        cdef Py_ssize_t i
        if not preds:
            return True
        sc, h, pos = preds[0]
        need = self.target - r.chips
        if sc < need and r.dl > 0 and sc * r.hl < need:
            rest = [i for i in range(len(r.hand)) if i not in pos]
            if rest:
                rest.sort(key=_chip_key_at(r.hand))
                self._discard(r, tuple(sorted(rest[:5])), rng)
                return False
        return self._play(r, pos, sc, h, rng)

    cpdef _Round _playout(self, _Round r, rng):
        cdef int k
        for k in range(self.max_steps):
            if self._policy_step(r, rng):
                break
        return r

    cdef tuple _max_node(self, _Round r, rng):
        """depth 2: the best of a few plays and discards, each on `inner` re-shuffles of the draw pile."""
        cdef list preds = self._preds(r)
        cdef list acts = [("p", p) for p in preds[:2]]
        cdef _Round q
        cdef int k
        cdef bint over
        if r.dl > 0:
            seen = set()
            for sc, h, pos in preds[:2]:
                rest = sorted([i for i in range(len(r.hand)) if i not in pos], key=_chip_key_at(r.hand))
                d = tuple(sorted(rest[:5]))
                if d and d not in seen:
                    seen.add(d)
                    acts.append(("d", d))
        best = (-1.0, 0.0)
        for kind, x in acts:
            tot_p = tot_c = 0.0
            for k in range(self.inner):
                q = r.dup()
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
    def solve(self, g, choice, rng, samples=None, depth=None):
        """Fill p_clear and e_chips on the play, discard and use candidates of `choice`."""
        cdef int n_s = self.samples if samples is None else samples
        cdef int dp = self.depth if depth is None else depth
        cdef int k
        sg = self._setup(g)
        target = max(1.0, float(g.target))
        hidden = [i for i, c in enumerate(sg.hand) if (<Card>c).hidden]
        pool = sorted(list(sg.deck) + [sg.hand[i] for i in hidden], key=_uid_key)
        futures = []
        for k in range(n_s):
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
        for c in sorted(plays, key=_neg_score):
            best_by_group.setdefault(c.signature(), c)
        chosen = set(id(c) for c in sorted(plays, key=_neg_score)[:self.top_plays])
        chosen |= {id(c) for c in best_by_group.values()}
        for c in plays:
            if c.clears:
                c.p_clear = 1.0 if c.certain or not hidden else c.p_clear
                c.e_chips = min(5.0, (g.chips + c.score) / target)
                if c.p_clear == 1.0:
                    continue
            if id(c) not in chosen and not c.clears:
                continue
            self._evaluate(c, futures, start, rng, dp, target, True, bool(hidden))
        for c in discs:
            self._evaluate(c, futures, start, rng, dp, target, False, bool(hidden))
        base = max([c.p_clear for c in plays + discs if c.p_clear >= 0], default=0.0)
        base_e = max([c.e_chips for c in plays + discs if c.e_chips >= 0], default=0.0)
        for c in choice.cands:
            if c.kind == "use":
                c.p_clear = 1.0 if c.best_after >= 1.0 else base
                c.e_chips = max(base_e, (g.chips / target) + c.best_after * (choice.need / target))

    cdef void _evaluate(self, c, list futures, _Round start, rng, int depth, double target, bint is_play,
                        bint hidden) except *:
        cdef tuple pos = c.action.cards
        cdef _Round r
        cdef bint over
        cdef Py_ssize_t n
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


def _uid_key(c):
    return (<Card>c).uid


def _neg_score(c):
    return -c.score


class _chip_key_at:
    """key= for sorting hand positions by the card's chip value (as the lambda in solver.py)."""
    __slots__ = ("hand",)

    def __init__(self, hand):
        self.hand = hand

    def __call__(self, i):
        return (<Card>self.hand[i]).chip_value()
