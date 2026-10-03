# Speed notes

CPU seconds per game on the agreed bench (`python bench_speed.py`: 20 no-search graded games, seeds
70001-70020, `shop_prior: graded`, `rule_arcana: true`, Red Deck, White Stake). The absolute numbers are
from the development container (4 slow cores; the reference machine ran the baseline commit cd7dfc3 at
about 3.4 s/game, 2.3x faster than this container); the ratio is what matters.

| step | commit | CPU s/game | vs baseline |
|---|---|---|---|
| baseline (cd7dfc3) | - | 7.76 | 1.00 |
| stage 1: differential harness, bench | 33bed6d | 7.76 | 1.00 |
| stage 2: scorer: hand-detection cache + pruning bound (exact) | 20aa8d9 | 6.38 | 0.82 |
| stage 4, steps 1-2: C++ core: solver playouts, hands.evaluate; scorer prunes in bound order (exact) | e59d79b | 4.77 | 0.61 |
| stage 3: pricer: 16 hands for cheap options, lazy pack sampling (measured, decisions change) | 25d4af4 | 3.82 | 0.49 |
| bootstrap memo, headroom hands from the C++ replica (exact) | f8a83d2 | 3.75 | 0.48 |
| stage 3b: inner solver samples 4 -> 2 (search only; no effect on this bench) | 6818341 | 3.75 | 0.48 |

Every commit was benched the same way, each built in its own worktree with the machine otherwise idle
(the mean furthest blind over the 20 games is 15.80 for every exact step and 14.35 after stage 3, whose
decisions differ; the 300-seed paired comparison below is the measurement of that change). On the
reference machine, 3.4 s/game at the baseline, the same ratio gives about 1.6 s/game.

## Stage 2 (exact: the golden test is unchanged)

Micro-benchmark of the pooled scorer (32 hands of 8 cards, every 1-5 card subset of each):

| | ns per subset |
|---|---|
| before | 141 (no jokers) - 172 (5 jokers) |
| after | 60 - 62 |

(a) hand detection is cached per loaded hand (module-level, keyed on the cards' rank / suit / enhancement /
debuff and the flags), so the pricer's 32 sampled hands are evaluated once per decision and only the jokers'
pass runs per option; (b) `_best_two` visits the largest subsets first and skips any subset whose upper bound
on chips x mult is below the second-best score so far (81% of the subsets of random states are skipped; the
tie rule is applied explicitly so the result is identical to scoring every subset in order). The bound is
sound for every joker but DNA (then it is off); `tests/fidelity/test_fastscore.py` checks bound >= score on
random states.

## Stage 4, steps 1-2 (exact: golden, real plays and the differential test unchanged)

The round solver was 27% of a no-search game (46 ms per solve, ~720 scorer calls each): its futures and
playouts now run in C++ on card indices with the agent's random generator replicated (CPP_PORT_NOTES.md),
and the compiled scorer is reached through a C API. 46 -> 18 ms per solve. The scorer's cold path (every
subset of a hand the hand cache has not seen) was the rest of a solve: `_best_two` now scores the two
largest-bound subsets first and skips every subset whose bound is below the second best, the straight
search only visits the masks a straight can use, and the bound skips its empty terms: 25 -> 14 us per
cold 8-card hand, 18 -> 11 us warm. A search game went from 40 to 33 CPU s on this container.

## Stage 3 (changes decisions: measured)

`python -m balatro_rl.az.compare grid` (grids in eval_grids/) on 300 paired seeds (10000-10299, Red Deck, White Stake, no search,
graded prior with `rule_arcana`, untrained network), against the unchanged graded prior. Rule: a change
is kept only if blinds do not fall by more than one paired SE. (cpu s/game here was measured while other
work ran on the machine; the bench row above is the clean number.)

| variant | option | blinds +- SE | vs base (paired) | wins | cpu s/game |
|---|---|---|---|---|---|
| base | - | 17.02 +- 0.35 | | 41/300 | 6.9 |
| small16 | `samples_small: 16` | 16.83 +- 0.35 | -0.19 +- 0.30 | 27/300 | 6.5 |
| lazy | `pack_lazy: 0.05` | 17.87 +- 0.33 | +0.85 +- 0.39 | 36/300 | 6.8 |
| reuse | `reuse_behind: 0.05` | 15.89 +- 0.32 | -1.13 +- 0.37 | 19/300 | 4.7 |
| all3 | all three | 15.67 +- 0.29 | -1.35 +- 0.40 | 13/300 | 4.6 |
| lazy16 | `samples_small: 16`, `pack_lazy: 0.05` | 17.50 +- 0.33 | +0.48 +- 0.41 | 31/300 | 5.8 |
| lazy10 | `pack_lazy: 0.10` | 17.34 +- 0.33 | +0.32 +- 0.40 | 31/300 | 6.0 |

Kept (the new defaults): `samples_small = 16` (sells, boss rerolls and single playing cards bought in
the shop are measured on the first 16 of the 32 shared hands, against the decision's state measured on
the same 16; a playing card picked from a pack keeps 32: card_margin is set for that) and
`pack_lazy = 0.05` (packs are priced after the other options; a pack whose first sampled content prices
more than 5 logits behind the best option so far gets no second sample). Rejected: `reuse_behind` (at the
next decision in the same shop, options that priced more than 5 logits behind were not priced again;
the state changes between decisions, a joker bought fills the slots and sells become the way to the next
one, so it loses over a blind). The option stays in ShopConfig, off.

Stage 3 was expected to cut more. Per-option profile of the pricer (5 games, before stage 3): packs 44% of
pricing time (15 ms each: 2 sampled contents x every pick x up to 4 target sets), sells 30% (2.7 ms each;
89% of them more than 5 logits behind), consumable picks 12%, tarots 7%. The hands are not where the
time is: a strength evaluation costs about 1 ms, of which 0.4 ms is compiled scoring and the rest is
Python (fresh_round's clone, the scorer's tables, numpy's bootstrap, the build key), so halving K saves
a fifth of those calls, and lazy packs rarely stop after the first sample (a pack is seldom 5 logits
behind on its first content). The gains that remain are in the Python glue (CPP_PORT_NOTES.md).

Solver (3b): `_preds` hit rate within a solve is 3.3% without search and 11.5% with it (the futures draw
different hands; identical hands are already reused), so nothing to change there. `solver_samples_inner`
only applies inside the search (the no-search bench and grid never run it), so it was measured with a
search grid (`eval_grids/stage3_search.json`, 300 paired seeds, search on, the stage-3 pricer defaults):

| variant | blinds +- SE | vs search base (paired) | wins | cpu s/game |
|---|---|---|---|---|
| search_base (inner 4) | 16.30 +- 0.37 | | 24/300 | 53.3 |
| search_inner2 (inner 2) | 16.95 +- 0.35 | +0.65 +- 0.49 | 26/300 | 50.6 |

Kept: `solver_samples_inner = 2` (the golden file was regenerated for it: the search's inner evaluations
change, the solver / scorer section of the golden file does not).
