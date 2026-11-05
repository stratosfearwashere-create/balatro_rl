# Run evaluation function

A hand-written static evaluation of a Balatro run state, built to the spec "Balatro Run Evaluation Function: Scope": a weighted sum of nine
terms, scored three times with phase weights (early, mid, late) and blended by ante, the way Stockfish 1.0 keeps a middlegame and an endgame
score. The simulator is this repository's exact engine; the weights are fitted against its outcomes. Everything lives in:

| piece | where |
|---|---|
| the eval | `balatro_rl/runeval.py` (`RunEval`, `compute_terms`, `State`, `Cols`) |
| the weights | `balatro_rl/data/runeval_weights.json` (hand-set priors until `runeval_fit.py` writes fitted ones) |
| labelled states | `tools/runeval_data.py` -> `data/runeval/*.npz` |
| the fit, metrics, ablations, report | `tools/runeval_fit.py` -> `traces/runeval_report.md` |
| shop decision agreement | `tools/runeval_shop.py` -> `traces/runeval_shop.json` |
| tests | `tests/test_runeval.py` |

```bash
python -m balatro_rl.runeval checkpoints/ppo_gold_v30_best.pt --games 16        # scores along a few games, with timing
python tools/runeval_data.py --games 2048 --builds 6000 --rollouts 16 --out data/runeval/states_v1.npz
python tools/runeval_fit.py data/runeval/states_v1.npz data/runeval/states_v1_late.npz
python tools/runeval_shop.py --shops 256 --rollouts 32
python -m pytest -q tests/test_runeval.py
```

## Answers to the spec's open items

- **Game version, stake, deck.** Balatro 1.0.1o (the user's copy; `Balatro/version.jkr`), Gold stake, Red deck, no mods. The engine's content
  tables (`balatro_rl/sim/*.py`, `pools.json`, the C++ `tables_gen.hpp`) are the version-pinned data set; `tests/fidelity`, `tests/test_shopgen.py`
  and `tests/engine_diff.py` are the "matches known scoring examples" gate (shops, bosses, tags and deck order were checked against the game's
  own Lua with zero mismatches, see `project-exact-seed-engine` in the project notes).
- **Live or offline.** Offline planning first. The eval reads the engine's observation, so the bridge to the real game can feed it too; at a batch
  of one it costs about 50 ms (fixed overhead), in a batch about 0.15 ms per state.
- **Late potential.** Winning at ante 8 (the project's goal), not endless depth.
- **Ante split.** 1-3 / 4-6 / 7-8 as specified; the labelled data below shows where runs die under the simulator's policy.

## What the eval sees

The engine's observation at a decision outside a round: the global vector `g` (ante, blind, state, money, hands, discards, hand size, hand
levels, times each hand was played, the deck's 13 rank and 4 suit counts, its size, enhancement and seal counts, joker and consumable slots,
vouchers, the ante's three targets, the boss, the tag a skip would take, the stake), the joker ids with their features (edition, eternal,
perishable, rental, debuffed, sell value, counter) and the consumable ids. `runeval.Cols` names the columns; `State` turns them back into game
units. Scoring order, boss effects and the exact joker definitions come in through `joker_spec.py`'s numeric description of every joker and the
engine's growth table (`potential.GROW`).

## The nine terms

Each term is one number per state, higher is better, computed once and weighted per phase.

| term | what it computes | Stockfish 1.0 analogue |
|---|---|---|
| power | log10 of (hands x the expected best play) over the next blind's target. The best play is the strongest hand that lands: P(best = h) = reach_h x prod over stronger hands of (1 - reach); each hand's score is the hand's chips and mult at its level, the cards' chips and enhancements, and every live joker's expected contribution (flat chips / mult / xMult per trigger, conditions as expected matches, counters, editions, copies, retriggers). Two ways to play the round are compared: score with every hand after the discards' redraws, or hunt with the discards and all but two hands and score with the last two | material and piece-square tables |
| scaling | log10 of how much the growing jokers' counters, moved by their engine rates over the rounds left, will have multiplied the score at the ante-8 boss, plus 0.25 per xMult source (xMult jokers, copies of them) | passed pawns, endgame terms |
| consistency | the chance of landing the main hand (or a stronger one) within the round: a hypergeometric deal, then a redraw chain per hunted partial (the most-held rank, the richest suit, the fullest straight window) over the discards and spare hands | mobility |
| deck | suit and rank concentration behind the main hand, thinning (log2 of 52 / deck size), enhanced and sealed cards | pawn structure |
| economy | log2 of 1 + (money + 3 x (interest + income jokers - rental upkeep) + a quarter of the jokers' sell value + 4 per economy joker) / 10 | tempo |
| synergy | copy jokers next to multipliers, hand / suit / face / enhancement jokers that fit the main hand and deck, retriggers with per-card jokers, Four Fingers / Shortcut / Smeared with a flush or straight build | bishop pair, imbalances |
| risk | minus a quadratic penalty as the power margin on the next blind drops under half an order of magnitude, capped at one | king safety |
| flex | open joker slots, consumables held, rerolls affordable, the tag on offer, extra hands, discards and hand size | mobility, outposts |
| boss | minus the known boss's bite on this build (The Needle against unreliable hands, suit bosses against flushes of that suit, The Plant against face builds, ...), the pool's mean when the boss is unknown; 0.6 of it when the boss is not the next blind | king-safety attack units |

## Phase potentials and the blend

`E_phase = sigmoid(b_phase + sum_k w_phase,k x (term_k - mean_k) / std_k)` with early = P(reach ante 4 with $10 or more), mid = P(reach ante 7),
late = P(win). The blend `E(A) = a_e(A) E_early + a_m(A) E_mid + a_l(A) E_late` uses `a_e = sigmoid((c_early - A) / tau)`,
`a_l = sigmoid((A - c_late) / tau)`, `a_m = 1 - a_e - a_l`, normalised. `RunEval` also reports `win`, the blend with the early and mid potentials
converted to win chances (fitted conversions), which is the number the shop tool ranks options by.

## Simulator as ground truth

`tools/runeval_data.py` labels states by playing them forward in the engine: the scripted round player for rounds (the spec's greedy v1: it sees
only its hand, plays the best predicted play if it clears the blind or is on pace, else discards the cards outside it) and a network's greedy
choices outside rounds (`checkpoints/ppo_gold_v30_best.pt`, the strongest shop policy in the repository; `--ckpt none` = select and leave). Every
state gets K rollouts with reseeded draws; states come from the policy's own games (blind selects and a third as many shops) and from random
builds placed at antes 2-8 with `VecEnv.set_build`. The engine's probes `fast_power` (hands x mean best play / target on redrawn hands),
`projected_power` and `final_boss_clear` are stored beside the labels to check the power and scaling terms against the engine's own scorer.

## Tuning and validation

`tools/runeval_fit.py`: a held-out set split by game, per-phase logistic regressions ridge-regularised toward the spec's priors (Low 0.25,
Medium 0.5, High 1.0, Very high 1.5), the blend fitted last against the win label, then the spec's metrics on the held-out states and one
ablation per term. The results are in `traces/runeval_report.md`; a summary follows once the first fit has run.

## Results (first fit, 2026-10-23)

Data: 21,757 states along 2,048 games plus 6,000 random builds (16 rollouts each) and 4,000 builds at antes 5-8 (32 rollouts each); 20% of the
games held out. Full tables: `traces/runeval_report.md`.

**Gate 1, simulator correctness.** The engine's existing fidelity tests stand in for it (see above). The eval's own scoring model was checked
against the engine: its expected best play of the dealt hand ranks states like the engine's measured best play (Spearman 0.89 held-out,
0.68-0.82 within each ante, median ratio 1.01), and the power term against the engine's `fast_power` probe at 0.80.

**Gate 2, speed.** 160 us per state in batches of 4,096 (numpy, one core); about 50 ms for a batch of one, which is fixed overhead.

**Gate 3, accuracy on held-out states.** Each potential against its own label:

| potential | label | Spearman | calibration error |
|---|---|---|---|
| early | reach ante 4 with $10+ (rollouts) | 0.62 | 0.024 |
| mid | reach ante 7 (rollouts) | 0.27 | 0.003 |
| late | the engine's ante-8 probe: the build with its counters projected to the final boss, played by the exact round solver | 0.71 | 0.005 |

Within one ante the blended score orders states by the mean ante they go on to reach at Spearman 0.27 (ante 1) to 0.56 (antes 3-4), falling
to 0.17 at ante 8. The spec's 0.80 is not reached. Two reasons, in order: the labels are noisy (16 rollouts of a 38% event give a label
noise that caps the attainable correlation near 0.9, and the mid label happens in 1.7% of states), and the simulator's play policy is weak:
the greedy scripted rounds plus the v30 network win 0.01% of fresh Gold runs, so a rollout-based late label does not exist, which is why the
late potential is fitted on the engine's exact ante-8 probe instead. Calibration is well inside 0.05 everywhere.

**Fitted weights** (standardised units; the ablation column is the held-out Spearman of that phase with the term removed and the rest refitted):

| term | early | mid | late | ablation (early / mid / late) |
|---|---|---|---|---|
| power | +0.74 | +0.40 | +0.30 | 0.56 / 0.26 / 0.69 |
| scaling | +0.07 | +0.43 | +0.67 | 0.62 / 0.26 / 0.51 |
| consistency | +0.01 | -0.06 | -0.02 | unchanged |
| deck | -0.05 | +0.05 | +0.01 | unchanged |
| economy | +1.03 | +0.69 | +0.54 | 0.26 / 0.21 / 0.68 |
| synergy | -0.06 | -0.24 | -0.11 | unchanged |
| risk | -0.22 | +0.19 | -0.10 | 0.62 / 0.27 / 0.70 |
| flex | +0.04 | -0.53 | -0.32 | 0.62 / 0.26 / 0.68 |
| boss | +0.02 | +0.13 | +0.05 | unchanged |

Reading: money is the early game (removing the economy term halves the early correlation), scaling is the late game (0.71 to 0.51 without it),
power matters throughout. Consistency, deck, synergy and boss earn nothing yet in this data: either the simulator's greedy play does not
exploit what they measure, or their hand-set forms are wrong; they are the first terms to rework. The eval picks a flush as the main hand in
45% of states (it hunts flushes with the discards), two pair or a straight in most of the rest.

**Gate 4, shop decision agreement.** `tools/runeval_shop.py`, 320 shops along 2,048 games (at most 64 per ante), every option
(each buy, voucher, pack, reroll, leave) played out 32 times with reseeded draws; the simulator's verdict is the mean blinds the game goes on
to beat, and an option agrees when it is within half a blind of the best.

| measure | eval | always leave | the v30 policy's own choice | spec target |
|---|---|---|---|---|
| agreement with the simulator's best option | 0.59 | 0.62 | 0.72 | >= 0.70 |
| agreement on clear-cut shops (best option ahead by 2 standard errors, 102 shops) | 0.32 | | | |
| regret, blinds lost against the best option | 0.87 | 0.92 | 0.47 | |

| ante | shops | agreement eval | leave | policy | regret eval | leave | policy |
|---|---|---|---|---|---|---|---|
| 1 | 64 | 0.34 | 0.33 | 0.78 | 2.07 | 2.07 | 0.43 |
| 2 | 64 | 0.48 | 0.20 | 0.63 | 0.95 | 1.24 | 0.69 |
| 3 | 64 | 0.58 | 0.71 | 0.64 | 0.66 | 0.43 | 0.56 |
| 4 | 64 | 0.70 | 0.75 | 0.69 | 0.41 | 0.42 | 0.42 |
| 5 | 64 | 0.84 | 0.83 | 0.83 | 0.26 | 0.29 | 0.27 |

Not met. The eval picks "leave" in 60% of shops, so it behaves like the money-keeping baseline: the fitted economy weight is the largest
early term, and spending $5-8 lowers the economy term more than a joker raises power or scaling in the eval's own arithmetic, while the
simulator says the purchase was right about half the time. The network's own choices are better than both. This is the first thing to fix in
the next iteration: fit on decision pairs (the option taken against the options not taken, from this tool's play-outs) rather than on states
alone, and let the fit rebalance the economy against the power and scaling gained per dollar.

## Where this stands against the spec

| gate | target | result |
|---|---|---|
| 1 simulator correctness | matches known scoring examples | the engine's fidelity tests; the eval's scoring model ranks states like the engine's measured best play (Spearman 0.89) |
| 2 speed | under 1 ms per state | 0.16 ms per state in a batch; 50 ms fixed cost for a single state |
| 3 accuracy | Spearman >= 0.80, calibration < 0.05 | early 0.62, mid 0.27, late 0.71 against their labels; calibration 0.003-0.024; not met on correlation |
| 4 shop decisions | agreement >= 70% | 59%, no better than always leaving; not met |

What would move it: a stronger play policy for the labels (the spec's own first risk; the greedy v1 wins 0.01% of Gold runs, so the win label
is empty), more rollouts per state for the early and mid labels (the label noise alone caps the attainable correlation near 0.9), a decision-pair
fit for the shop gate, and reworking the four terms that earn nothing in the ablations (consistency, deck, synergy, boss).

