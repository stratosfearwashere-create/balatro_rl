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
search can overrule it). Inside its futures (with joker_state, the default): the jokers' runtime values
and the hands-played counts are carried from hand to hand, so Green Joker, Ride the Bus, Runner, Square,
Spare Trousers, Wee Joker, Obelisk, Loyalty Card, Vampire, Lucky Cat, Supernova and Card Sharp score as
they would over the round, and the discard hooks of Green Joker, Castle, Hit the Road and Ramen and Ice
Cream's melt are applied (JOKER_HOOKS; Yorick's counter, Burnt Joker's level-up and the money hooks are
not). Cards drawn later are face up, glass doesn't break; The Serpent's and The Hook's draws, The Eye, The
Mouth and The Psychic are modelled. Scores come from the compiled scorer when it is built and are cached
per hand (and state) within a call.

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
from .actions import Choice, RANDOM_CONSUMABLES as _RANDOM_USES

_SUBS = {}


def _subsets(n: int) -> list[tuple]:
    s = _SUBS.get(n)
    if s is None:
        s = _SUBS[n] = [c for k in range(1, min(5, n) + 1) for c in combinations(range(n), k)]
    return s


# discard / after-play hooks the playouts apply to a joker's "val" (the scorer handles the scoring hooks)
DK_NONE, DK_GREEN, DK_CASTLE, DK_ROAD, DK_RAMEN = 0, 1, 2, 3, 4
AK_NONE, AK_ICE = 0, 1
JOKER_HOOKS = {"green_joker": (DK_GREEN, AK_NONE), "castle": (DK_CASTLE, AK_NONE), "hit_the_road": (DK_ROAD, AK_NONE),
               "ramen": (DK_RAMEN, AK_NONE), "ice_cream": (AK_NONE, AK_ICE)}


def joker_hooks(g) -> list:
    """(discard kind, after kind, Castle's suit, active) per joker, in the scorer's table order (g.jokers)."""
    out = []
    for j in g.jokers:
        dk, ak = JOKER_HOOKS.get(j.key, (DK_NONE, AK_NONE))
        suit = j.state.get("suit", 0) if j.key == "castle" else 0
        out.append((dk, ak, int(suit) if isinstance(suit, int) else 0, not j.debuffed))
    return out


def base_state(g) -> list:
    """[vals, has_vals, hands played per type this run, this round] as the scorer's tables hold them."""
    vals, hvs = [], []
    for j in g.jokers:
        v = j.state.get("val")
        hvs.append(1 if v is not None else 0)
        vals.append(0.0 if v is None else float(v))
    return [vals, hvs, list(g.hand_played), list(g.hand_played_round)]


class _Round:
    __slots__ = ("hand", "draw", "hl", "dl", "chips", "types", "mouth", "st")

    def __init__(self, hand, draw, hl, dl, chips, types, mouth, st=None):
        self.hand, self.draw, self.hl, self.dl, self.chips = hand, draw, hl, dl, chips
        self.types, self.mouth = types, mouth
        self.st = st                      # [vals, has_vals, hands played, hands played this round] or None

    def copy(self):
        st = None if self.st is None else [list(x) for x in self.st]
        return _Round(list(self.hand), list(self.draw), self.hl, self.dl, self.chips, self.types, self.mouth, st)


USE_TARGETS = 3            # target sets per consumable whose use is played out (the best by immediate effect)
USE_SAMPLES = 8            # futures per use playout


def use_candidates(choice, limit: int = USE_TARGETS) -> list:
    """The use candidates to play out: per consumable, the `limit` target sets with the best immediate
    effect; a use that already clears the blind outright keeps P(clear) = 1."""
    by_cons = {}
    for c in choice.cands:
        if c.kind == "use" and c.best_after < 1.0:
            by_cons.setdefault(c.action.idx, []).append(c)
    out = []
    for cands in by_cons.values():
        out += sorted(cands, key=lambda c: -c.best_after)[:limit]
    return out


def apply_use(g, action, rng):
    """The use carried out on a copy with the hidden information redrawn; None if the round is over."""
    from .world import apply, determinize
    g2 = determinize(g, rng)
    try:
        apply(g2, action)
    except (AssertionError, IndexError, ValueError):
        return None
    if g2.state != "SELECTING_HAND" or not g2.hand or g2.targeting is not None:
        return None
    return g2


class RoundSolver:
    def __init__(self, samples: int = 12, depth: int = 1, inner: int = 2, top_plays: int = 10, max_steps: int = 40,
                 joker_state: bool = True, use_playout: bool = True):
        self.samples, self.depth, self.inner, self.top_plays, self.max_steps = samples, depth, inner, top_plays, max_steps
        self.joker_state = joker_state
        self.use_playout = use_playout
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
        self.hooks = joker_hooks(sg)
        self.base = base_state(sg)
        return sg

    # ------------------------------------------------------------------ round state
    def _apply_state(self, st):
        """Put a round state into whatever scores next: the pooled scorer's tables, or the game copy."""
        if st is None:
            return
        vals, hvs, hp, hpr = st
        if self.fs is not None:
            self.fs.set_state(vals, hvs, hp, hpr)
        sg = self.sg
        for j, joker in enumerate(sg.jokers):
            if hvs[j]:
                joker.state["val"] = vals[j]
            else:
                joker.state.pop("val", None)
        sg.hand_played, sg.hand_played_round = list(hp), list(hpr)

    def _advance_play(self, r: _Round, pos, htype: int):
        """The state after `pos` is played from r: the scorer's before / card hooks (a forbidden hand changes
        nothing), the after-play hooks, and the hands-played counts."""
        if r.st is None:
            return
        self._apply_state(r.st)
        eye = self.boss in ("eye", "mouth")
        sg = self.sg
        if self.fs is not None and len(r.hand) <= 16:
            types, mouth = (fastscore.hand_types_mask(r.types), r.mouth) if eye else self.base_types
            pidx = self.pidx
            _, _, viol, vals, hvs = self.fs.score_state([pidx[id(c)] for c in r.hand], list(pos), r.hl, r.dl,
                                                        len(r.draw), types, mouth)
        else:
            from ..sim.scoring import score_hand
            sg.hand, sg.hands_left, sg.discards_left, sg.deck = r.hand, r.hl, r.dl, r.draw
            if eye:
                sg.round_hand_types, sg.mouth_hand = set(r.types), r.mouth
            played = [r.hand[i] for i in pos]
            held = [c for i, c in enumerate(r.hand) if i not in pos]
            viol = sg.violates_boss(played)
            _, ctx = score_hand(sg, played, held, None, False, self.plan)
            vals, hvs = [], []
            for joker in sg.jokers:
                v = ctx.jstate.get(joker.uid, joker.state).get("val")
                hvs.append(1 if v is not None else 0)
                vals.append(0.0 if v is None else float(v))
        if not viol:
            r.st[0], r.st[1] = list(vals), list(hvs)
            for j, (dk, ak, suit, active) in enumerate(self.hooks):
                if active and ak == AK_ICE:                      # Ice Cream: -5 chips per hand; gone at 0
                    v = (r.st[0][j] if r.st[1][j] else 100.0) - 5.0
                    r.st[0][j], r.st[1][j] = (v if v > 0 else 0.0), 1
        if 0 <= htype < 12:
            r.st[2][htype] += 1
            r.st[3][htype] += 1

    def _advance_discard(self, r: _Round, pos):
        if r.st is None:
            return
        vals, hvs = r.st[0], r.st[1]
        for j, (dk, ak, suit, active) in enumerate(self.hooks):
            if not active or dk == DK_NONE:
                continue
            v = vals[j] if hvs[j] else 0.0
            if dk == DK_GREEN:                                   # -1 Mult per discard, not below 0
                v = max(0.0, v - 1.0)
            elif dk == DK_CASTLE:                                # +3 chips per discarded card of its suit
                v += 3.0 * sum(1 for i in pos if not r.hand[i].is_stone
                               and (r.hand[i].enh == "WILD" or r.hand[i].suit == suit))
            elif dk == DK_ROAD:                                  # +0.5 X per discarded Jack
                if not hvs[j]:
                    v = 1.0
                v += 0.5 * sum(1 for i in pos if not r.hand[i].is_stone and r.hand[i].rank == 11)
            elif dk == DK_RAMEN:                                 # -0.01 X per discarded card; gone at 1
                if not hvs[j]:
                    v = 2.0
                v -= 0.01 * len(pos)
                if v <= 1.0:
                    v = 1.0
            vals[j], hvs[j] = v, 1

    def _preds(self, r: _Round) -> list:
        """[(score, hand type, subset)] for the two best plays from r.hand, best first (all the rollouts
        use). Ties go to the earlier subset, as a stable sort by score would."""
        eye = self.boss in ("eye", "mouth")
        key = (tuple(c.uid for c in r.hand), r.hl, r.dl, len(r.draw), (r.types, r.mouth) if eye else None,
               None if r.st is None else tuple(tuple(x) for x in r.st))
        got = self.cache.get(key)
        if got is not None:
            self.cache_hits += 1
            return got
        self.calls += 1
        self._apply_state(r.st)
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
        self._advance_play(r, pos, htype)
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
        self._advance_discard(r, pos)
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
                       sg.mouth_hand, self.base if self.joker_state else None)
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
        self._uses(g, choice, rng, plays + discs, target)

    def _uses(self, g, choice, rng, evaluated, target):
        """P(clear) / expected chips of the use candidates: a use that clears outright is 1; with use_playout
        the others are the playout value of the round after the use (USE_SAMPLES futures, averaged over
        use_samples copies for random consumables); otherwise the round's best."""
        base = max([c.p_clear for c in evaluated if c.p_clear >= 0], default=0.0)
        base_e = max([c.e_chips for c in evaluated if c.e_chips >= 0], default=0.0)
        for c in choice.cands:
            if c.kind == "use":
                c.p_clear = 1.0 if c.best_after >= 1.0 else base
                c.e_chips = max(base_e, (g.chips / target) + c.best_after * (choice.need / target))
        if not self.use_playout:
            return
        if g.boss_active() == "verdant_leaf" and g.flags.get("verdant"):
            # every card is debuffed until a joker is sold: the sale is the move that makes scoring possible
            from .world import determinize
            for c in choice.cands:
                if c.kind != "sell_joker":
                    continue
                g2 = determinize(g, rng)
                try:
                    g2.sell_joker(c.action.idx)
                except (AssertionError, IndexError):
                    continue
                if g2.state == "SELECTING_HAND" and g2.hand:
                    c.p_clear, c.e_chips = self.null_value(g2, rng, USE_SAMPLES)
        for c in use_candidates(choice):
            tot_p = tot_e = 0.0
            n = 0
            for _ in range(2 if g.consumables[c.action.idx].name in _RANDOM_USES else 1):
                g2 = apply_use(g, c.action, rng)
                if g2 is None:
                    continue
                p, e = self.null_value(g2, rng, USE_SAMPLES)
                tot_p, tot_e, n = tot_p + p, tot_e + e, n + 1
            if n:
                c.p_clear, c.e_chips = tot_p / n, tot_e / n

    def null_value(self, g, rng, samples: int) -> tuple[float, float]:
        """(P(clear), expected chips / target) of playing the round out from `g` by the playout policy."""
        sg = self._setup(g)
        target = max(1.0, float(g.target))
        hidden = [i for i, c in enumerate(sg.hand) if c.hidden]
        pool = sorted(list(sg.deck) + [sg.hand[i] for i in hidden], key=lambda c: c.uid)
        types0, mouth0 = frozenset(sg.round_hand_types), sg.mouth_hand    # before any playout writes them back
        futures = []                                  # every future is drawn before any is played out (as
        for _ in range(samples):                      # solve does: The Hook's draws come after the shuffles)
            order = list(pool)
            rng.shuffle(order)
            hand = list(sg.hand)
            for i in hidden:
                hand[i] = order.pop()
            futures.append((hand, order))
        tot_p = tot_c = 0.0
        for hand, order in futures:
            r = _Round(hand, order, sg.hands_left, sg.discards_left, sg.chips, types0, mouth0,
                       [list(x) for x in self.base] if self.joker_state else None)
            self._playout(r, rng)
            tot_p += r.chips >= self.target
            tot_c += r.chips
        return tot_p / samples, min(5.0, tot_c / samples / target)

    def _evaluate(self, c, futures, start: _Round, rng, depth, target, is_play: bool, hidden: bool):
        pos = c.action.cards
        tot_p = tot_c = 0.0
        for hand, order in futures:
            r = _Round(list(hand), list(order), start.hl, start.dl, start.chips, start.types, start.mouth,
                       None if start.st is None else [list(x) for x in start.st])
            if is_play:
                if hidden:                          # the play's cards may be face down: score this future's
                    sg = self.sg
                    self._apply_state(r.st)
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
    from ._solver import RoundSolver, _subsets                # noqa: F811  (_Round stays the Python one)
    COMPILED = True
except ImportError:
    COMPILED = False


class CoreRoundSolver:
    """RoundSolver on the C++ core (sim/cpp/solver.hpp): the futures, playouts and the random draws run in
    C++ with the agent's generator replicated, and the compiled scorer is called through its C API. Same
    results as RoundSolver (bit for bit below 2^53 chips); falls back to it when a hand gets too large for
    the compiled scorer (The Serpent) or the state is unusual."""

    def __init__(self, samples: int = 12, depth: int = 1, inner: int = 2, top_plays: int = 10, max_steps: int = 40,
                 joker_state: bool = True, use_playout: bool = True):
        self.samples, self.depth, self.inner, self.top_plays, self.max_steps = samples, depth, inner, top_plays, max_steps
        self.joker_state = joker_state
        self.use_playout = use_playout
        self.calls = 0
        self.cache_hits = 0
        # the Cython solver is stateless: with joker_state the Python reference is the fallback
        self.fallback = (PyRoundSolver(samples, depth, inner, top_plays, max_steps, joker_state=True, use_playout=use_playout)
                         if joker_state else CompiledRoundSolver(samples, depth, inner, top_plays, max_steps))

    def _core(self, g, fs, pool):
        return _core.SolverCore(pool, fs, g.effective_hand_size(), g.boss_active(), float(g.target), g.hands_left,
                                g.discards_left, float(g.chips), fastscore.hand_types_mask(g.round_hand_types),
                                g.mouth_hand, self.inner, self.max_steps, self.joker_state,
                                joker_hooks(g) if self.joker_state else [])

    def null_value(self, g, rng, samples: int) -> tuple[float, float]:
        plan = Plan(g)
        pool = list(g.hand) + list(g.deck)
        fs = fastscore.pool_scorer(g, plan, pool)
        if fs is None or len(g.hand) > 16:
            return self.fallback.null_value(g, rng, samples)
        state = rng.getstate()
        hidden = [i for i, c in enumerate(g.hand) if c.hidden]
        pidx = {id(c): i for i, c in enumerate(pool)}
        order = [pidx[id(c)] for c in sorted(list(g.deck) + [g.hand[i] for i in hidden], key=lambda c: c.uid)]
        core = self._core(g, fs, pool)
        core.set_rng_state(state)
        core.make_futures(list(range(len(g.hand))), order, hidden, samples)
        try:
            p, e = core.evaluate([], 0.0, 0, False, 1)
        except ValueError:
            rng.setstate(state)
            return self.fallback.null_value(g, rng, samples)
        rng.setstate(core.rng_state())
        self.calls += core.calls
        self.cache_hits += core.cache_hits
        return p, e

    # the fallback's internals (_setup, _preds, _playout ...), for callers that drive one solve by hand
    def __getattr__(self, name):
        if name == "fallback":
            raise AttributeError(name)
        return getattr(self.fallback, name)

    @property
    def sg(self):
        return self.fallback.sg

    @property
    def plan(self):
        return self.fallback.plan

    @property
    def fs(self):
        return self.fallback.fs

    @fs.setter
    def fs(self, v):
        self.fallback.fs = v

    def solve(self, g, choice: Choice, rng: random.Random, samples: int | None = None, depth: int | None = None):
        n_s = samples or self.samples
        depth = depth or self.depth
        plan = Plan(g)
        pool = list(g.hand) + list(g.deck)
        fs = fastscore.pool_scorer(g, plan, pool)
        if fs is None or len(g.hand) > 16:
            return self.fallback.solve(g, choice, rng, samples, depth)
        state = rng.getstate()
        target = max(1.0, float(g.target))
        hidden = [i for i, c in enumerate(g.hand) if c.hidden]
        pidx = {id(c): i for i, c in enumerate(pool)}
        order = [pidx[id(c)] for c in sorted(list(g.deck) + [g.hand[i] for i in hidden], key=lambda c: c.uid)]
        core = self._core(g, fs, pool)
        core.set_rng_state(state)
        core.make_futures(list(range(len(g.hand))), order, hidden, n_s)
        plays = [c for c in choice.cands if c.kind == "play"]
        discs = [c for c in choice.cands if c.kind == "discard"]
        best_by_group = {}
        for c in sorted(plays, key=lambda c: -c.score):
            best_by_group.setdefault(c.signature(), c)
        chosen = set(id(c) for c in sorted(plays, key=lambda c: -c.score)[:self.top_plays])
        chosen |= {id(c) for c in best_by_group.values()}
        try:
            for c in plays:
                if c.clears:
                    c.p_clear = 1.0 if c.certain or not hidden else c.p_clear
                    c.e_chips = min(5.0, (g.chips + c.score) / target)
                    if c.p_clear == 1.0:
                        continue
                if id(c) not in chosen and not c.clears:
                    continue
                c.p_clear, c.e_chips = core.evaluate(list(c.action.cards), float(c.score), int(c.hand), True, depth)
            for c in discs:
                c.p_clear, c.e_chips = core.evaluate(list(c.action.cards), 0.0, 0, False, depth)
        except ValueError:                           # a simulated hand grew past the compiled scorer's size
            rng.setstate(state)
            return self.fallback.solve(g, choice, rng, samples, depth)
        rng.setstate(core.rng_state())
        self.calls += core.calls
        self.cache_hits += core.cache_hits
        PyRoundSolver._uses(self, g, choice, rng, plays + discs, target)


try:                                   # the C++ core (python setup_cython.py build); BALATRO_PURE=1 disables
    if _os.environ.get("BALATRO_PURE") == "1" or not COMPILED:
        raise ImportError
    from ..sim import _core
    if not hasattr(_core, "SolverCore"):
        raise ImportError
    CORE = True
except ImportError:
    CORE = False
CompiledRoundSolver = RoundSolver      # the Cython solver (stateless; CoreRoundSolver's fallback without joker_state)
if CORE:
    assert (_core.DK_GREEN, _core.DK_CASTLE, _core.DK_ROAD, _core.DK_RAMEN, _core.AK_ICE) == \
        (DK_GREEN, DK_CASTLE, DK_ROAD, DK_RAMEN, AK_ICE), "hook kinds differ between solver.py and the core"
    RoundSolver = CoreRoundSolver      # noqa: F811
elif COMPILED:
    RoundSolver = PyRoundSolver        # noqa: F811  (the Cython solver has no joker state)


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
