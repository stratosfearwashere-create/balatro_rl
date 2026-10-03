# Speed notes

CPU seconds per game on the agreed bench (`python bench_speed.py`: 20 no-search graded games, seeds
70001-70020, `shop_prior: graded`, `rule_arcana: true`, Red Deck, White Stake). The absolute numbers are
from the development container (4 slow cores; the reference machine ran the baseline commit cd7dfc3 at
about 3.4 s/game, 2.3x faster than this container); the ratio is what matters.

| step | commit | CPU s/game | vs baseline |
|---|---|---|---|
| baseline (cd7dfc3) | - | 7.87 | 1.00 |
| stage 1: differential harness, bench | 33bed6d | 7.87 | 1.00 |
| stage 2: scorer: hand-detection cache + pruning bound | (this) | 6.47 | 0.82 |

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
