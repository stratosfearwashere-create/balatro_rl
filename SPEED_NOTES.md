# Speed notes

CPU seconds per game on the agreed bench (`python bench_speed.py`: 20 no-search graded games, seeds
70001-70020, `shop_prior: graded`, `rule_arcana: true`, Red Deck, White Stake). The absolute numbers are
from the development container (4 slow cores; the reference machine ran the baseline commit cd7dfc3 at
about 3.4 s/game, 2.3x faster than this container); the ratio is what matters.

| step | commit | CPU s/game | vs baseline |
|---|---|---|---|
| baseline (cd7dfc3) | - | 7.87 | 1.00 |
| stage 1: differential harness, bench | 33bed6d | 7.87 | 1.00 |
| stage 2: scorer: hand-detection cache + pruning bound | 20aa8d9 | 6.47 | 0.82 |
| stage 4 (steps 1-2): C++ core: solver playouts, hands.evaluate; scorer prunes in bound order | e59d79b | 4.95 | 0.63 |

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
