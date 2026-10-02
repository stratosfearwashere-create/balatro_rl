# Stage 2: build-strength module

Branch `az-stage2` (from `az-stage1`). Everything new is behind config flags that default to today's
behaviour; `tests/test_regression.py` (golden, bit-identical) passes unchanged and the golden file was not
regenerated. Test suite: 142 passed (122 before + 20 new in `tests/test_strength.py`).

## What was built

| File | What |
|---|---|
| `balatro_rl/rewards/strength.py` | `Strength`: clear chance of each upcoming boss, "survives through ante", scalar strength score |
| `balatro_rl/rewards/growth.py` | projected growth of scaling jokers: table format, projection, and the script that logs games and builds the table (the built-in table is empty) |
| `balatro_rl/rewards/strength_check.py` | calibration and cost tool: `gen`, `report`, `cost` |
| `rewards/config.py`, `default.yaml` | `StrengthConfig` (under `potential.strength`), `potential.head_term`, two aux weights |
| `rewards/potential.py` | `Potential` picks its head term from `head_term`; the strength calculator is only built when used |
| `rewards/targets.py`, `az/net.py`, `az/train.py`, `az/agent.py` | optional strength head: targets, separate module, loss, logging |
| `tests/test_strength.py` | 20 tests |

How the strength is computed (`Strength.report(g)`):

1. Probe = `potential.fresh_round(g)`; K = 32 hands from the full deck, best play of each, one batched
   compiled-scorer call. It reuses `Headroom.best_scores` with the same seed, so these are exactly the
   headroom's hands and scores (tested).
2. B = 1000 whole rounds are bootstrapped from the K scores; P(clear) = share of totals >= target.
3. Targets: `g.blind_target(2)` for the current ante (The Wall x4, The Needle x1, Violet Vessel x6), then
   `2 x BLIND_BASE[g.scaling()][ante-1]` (x2 again on Plasma) for later antes.
4. Outputs: `clear` for antes (current, +1, +2, +3, 8), capped at 8; `clear_all` for every ante left;
   `survives` = last ante, counting on from the current one, with clear chance >= 50% (ante - 1 if the
   current one is below); `score` in [0, 1].
5. Hands and bootstrap are both seeded from a hash of `scoring_key(g)`, results are cached on
   (scoring key, targets), and only the deck's contents are read in a canonical order.

## Config flags (all defaults leave current behaviour unchanged)

| Flag | Default | Effect |
|---|---|---|
| `potential.head_term` | `headroom` | `strength`: Phi = w_head x strength score + w_prog x progress |
| `potential.strength.samples` / `bootstrap` | 32 / 1000 | K hands, B resampled rounds |
| `potential.strength.score` | `horizon` | `horizon`, `expected_antes`, `expected_total`, `current` |
| `potential.strength.discards` | `extra_draws` | `none`: hands-per-round independent best hands; `extra_draws`: best `hands` of `hands + weight x discards` draws |
| `potential.strength.discard_weight` | 1.0 | see calibration below |
| `potential.strength.score_scale` | 1.0 | multiplier on every best-hand score (off) |
| `potential.strength.growth` / `growth_discount` / `growth_table` | false / 0.9 / "" | projected growth of scaling jokers |
| `potential.strength.aux_head` | false | network gets a strength head (9 survive classes + 5 clear chances) |
| `aux_loss_weights.strength_survive` / `strength_clear` | 0.25 / 0.25 | only used with `aux_head` |

**Range of Phi.** Old: `0.7 x tanh(headroom) + 0.3 x progress`, in [-0.7, 1.0]; on the games below its mean
is -0.43 and 93% of states are negative. New: in [0, 1]; observed 0.04 to 0.93, mean 0.26. With
`value_bound: floor_sigmoid` and the current `value_init_bias: -1.9`, the untrained value averages 0.090
with the old Phi and 0.163 with the new one; `value_init_bias: -2.6` brings the new one back to 0.088.
I did not change those defaults.

## Data for the checks

150 greedy White Stake games, seeds 10000-10149, untrained agent, no search, 2 workers, 4.8 minutes:
12,656 decisions, 1,712 blind starts (533 bosses). **0 games won**, mean 10.4 blinds. Reproduce with
`python -m balatro_rl.rewards.strength_check gen --games 150 --workers 2 --solver-rounds 200 --out x.pkl`,
then `report --data x.pkl`.

## Calibration of the clear chance (the known weakness)

At each blind start, three references:
- (a) the round solver's playout policy on 200 fresh rounds of the same build (exact scores, discards, a
  deck that runs down, no boss effect), against three targets: the blind's own, this ante's boss, the next
  ante's boss;
- (b) whether that blind was then really cleared;
- the solver's own estimate at the real first hand of that blind (max over candidates).

**Before (no correction: independent hands, discards ignored): badly off.**

| predicted | n | mean predicted | (a) solver |
|---|---|---|---|
| 0.0-0.1 | 1625 | 0.014 | 0.130 |
| 0.1-0.3 | 417 | 0.192 | 0.660 |
| 0.3-0.5 | 333 | 0.401 | 0.854 |
| 0.5-0.7 | 492 | 0.595 | 0.894 |
| 0.7-0.9 | 424 | 0.811 | 0.972 |
| 0.9-1.0 | 1845 | 0.982 | 0.998 |

MAE vs (a) 0.155, bias -0.152. Against (b): Brier 0.135, bias -0.163.

**After (`discards: extra_draws`, weight 1.0: each discard is one more draw, keep the best `hands`).**
No fitted parameter.

| predicted | n | mean predicted | (a) solver |
|---|---|---|---|
| 0.0-0.1 | 1081 | 0.009 | 0.057 |
| 0.1-0.3 | 404 | 0.219 | 0.209 |
| 0.3-0.5 | 222 | 0.407 | 0.482 |
| 0.5-0.7 | 224 | 0.608 | 0.665 |
| 0.7-0.9 | 417 | 0.814 | 0.823 |
| 0.9-1.0 | 2788 | 0.988 | 0.975 |

MAE vs (a) 0.064, bias -0.009.

Against (b), the blind's own target:

| predicted | n | mean predicted | really cleared | (a) solver | solver at real first hand |
|---|---|---|---|---|---|
| 0.0-0.1 | 84 | 0.018 | 0.429 | 0.150 | 0.207 |
| 0.1-0.3 | 48 | 0.202 | 0.625 | 0.408 | 0.606 |
| 0.3-0.5 | 48 | 0.404 | 0.729 | 0.541 | 0.747 |
| 0.5-0.7 | 63 | 0.613 | 0.825 | 0.705 | 0.860 |
| 0.7-0.9 | 148 | 0.819 | 0.831 | 0.854 | 0.962 |
| 0.9-1.0 | 1321 | 0.993 | 0.974 | 0.989 | 0.991 |

Brier 0.079 (was 0.135), bias -0.035. Boss blinds only: Brier 0.112; top bucket predicted 0.993, cleared
0.941 (boss effects are not modelled).

**Still wrong: the low end against real outcomes.** Blinds predicted under 10% were cleared 43% of the
time. The solver's own playout policy has the same error (its under-10% bucket: 40% cleared; its own Brier
against the outcome is 0.066), and so does the solver at the real first hand (under 10%: 32% cleared). So
this is not the bootstrap: it is what none of the three model (jokers growing within the round, consumables
used in the round, a smarter discard choice than the playout's). I did not find a cheap correction for it.

**Grid of corrections** (weight x score scale; "held-out" = Brier on odd seeds):

| weight | scale | MAE vs (a) | bias vs (a) | MAE vs real first hand | Brier vs (b) | bias vs (b) | Brier held-out |
|---|---|---|---|---|---|---|---|
| 0 | 1.0 | 0.155 | -0.152 | 0.187 | 0.135 | -0.163 | - |
| **1.0** | **1.0** | **0.064** | -0.009 | 0.069 | 0.079 | -0.035 | 0.093 |
| 1.5 | 1.0 | 0.068 | +0.025 | 0.051 | 0.074 | -0.011 | 0.084 |
| 2.0 | 1.0 | 0.077 | +0.048 | 0.043 | 0.072 | +0.004 | 0.078 |
| 3.0 | 1.0 | 0.098 | +0.081 | 0.038 | 0.070 | +0.022 | 0.074 |
| 1.0 | 1.25 | 0.091 | +0.069 | 0.048 | 0.071 | +0.013 | 0.075 |

The two references disagree. Weight 1.0 is the best match to the solver's fresh rounds and is calibrated in
every bucket there. Weight 2.0 is unbiased against real outcomes, but against (a) its middle buckets are too
high (predicted 0.42 -> solver 0.20; 0.61 -> 0.45; 0.81 -> 0.64). I left the default at 1.0 because it is
parameter-free and matches the like-for-like reference; 2.0 is a defensible alternative and is one config
value away. This choice is a judgement, not a clear win, and the real-outcome fit is specific to this weak
agent: refit it on the trained agent's games.

**K matters more than B.** With K = 128 hands instead of 32: MAE vs (a) 0.046 (from 0.064), Brier vs (b)
0.074. K = 32 against K = 128 on the same states differs by 0.043 on average and by 0.12 in the mid-range
(0.1-0.9). That sampling noise is fixed per build (seeded), but it is the size of the step Phi takes when
a build changes slightly.

**"Survives through ante" against the last boss really cleared** (per decision): exact 41%, within one
ante 78%; the game got further than predicted in 45% of states and less far in 15% (mean +0.6 antes),
because the build keeps improving in shops and nothing here models that.

## Check 1: Phi against the game's outcome, old and new, same games

No game was won, so calibration against the win has no signal ("insufficient wins" for every variant). **I
could not test the original complaint (top Phi bucket has the lowest win rate).** Substitute outcomes,
8 quantile buckets, `diagnostics.calibration`, new Phi with `score: horizon`:

Final blinds of the game (what the value target is while nothing is won):

| bucket | old Phi | outcome | new Phi | outcome |
|---|---|---|---|---|
| 1 | -0.630 | 9.14 | 0.098 | 9.94 |
| 2 | -0.585 | 9.63 | 0.145 | 11.07 |
| 3 | -0.556 | 11.32 | 0.178 | 11.68 |
| 4 | -0.527 | 12.62 | 0.217 | 11.75 |
| 5 | -0.492 | 13.31 | 0.260 | 12.41 |
| 6 | -0.438 | 14.32 | 0.304 | 12.58 |
| 7 | -0.320 | 13.54 | 0.372 | 12.96 |
| 8 | 0.083 | 13.18 | 0.500 | 14.97 |
| rank correlation | | 0.83 (top two buckets fall) | | 1.00 |

This ante's boss cleared:

| bucket | old Phi | cleared | new Phi | cleared |
|---|---|---|---|---|
| 1 | -0.630 | 0.796 | 0.098 | 0.602 |
| 2 | -0.585 | 0.603 | 0.145 | 0.535 |
| 3 | -0.556 | 0.592 | 0.178 | 0.505 |
| 4 | -0.527 | 0.555 | 0.217 | 0.746 |
| 5 | -0.492 | 0.648 | 0.260 | 0.779 |
| 6 | -0.438 | 0.701 | 0.304 | 0.794 |
| 7 | -0.320 | 0.909 | 0.372 | 0.882 |
| 8 | 0.083 | 0.959 | 0.500 | 0.919 |
| rank correlation | | 0.55 | | 0.90 |

Blinds cleared after the state: old 0.14, new 0.60 (old is U-shaped: its lowest bucket has 6.9 blinds
ahead, its fourth 2.9).

**Read this with care.** The head terms alone are about equally good: against "boss cleared" both the
tanh-headroom term and the strength term have bucket rank correlation 1.00; against blinds after, 0.98
(headroom) and 0.93 (strength). Within one ante at a time (which removes "late states have little game
left"), row-level rank correlation with blinds cleared afterwards:

| ante | n | old head term | strength term | old Phi | new Phi |
|---|---|---|---|---|---|
| 1 | 2594 | 0.099 | 0.106 | 0.075 | 0.084 |
| 2 | 2602 | 0.370 | 0.343 | 0.355 | 0.317 |
| 3 | 2686 | 0.367 | 0.365 | 0.337 | 0.331 |
| 4 | 2623 | 0.346 | 0.315 | 0.259 | 0.266 |
| 5 | 1367 | 0.412 | 0.364 | 0.323 | 0.269 |
| 6 | 518 | 0.238 | 0.395 | 0.117 | 0.262 |
| 7 | 223 | 0.254 | 0.298 | 0.092 | 0.144 |

So the strength score is not a better predictor than the headroom at ranking builds inside an ante (slightly
worse in antes 2-5, better in 6-7). The whole-game bucket tables improve because of how the head term
combines with the progress term, not because the head term carries more information. I would not expect
the Phi swap by itself to change results much.

Other score kinds, same data: `expected_total` is worse than the old Phi (rank correlation -0.93 against
blinds after and -0.83 against boss cleared: it is mostly "antes already behind"). `expected_antes` gives
FLAT_OR_FALLING on both. `current` rises on both (0.48, 0.76) but saturates: its top three buckets of the
strength term are a single bucket at 0.998 holding 38% of states. `horizon` is the default for that reason.

How far ahead the build sees (share of decisions with a clear chance above 1%): this ante 86%, +1 47%,
+2 13%, +3 2%, ante 8 0.7%. The "ante 8" output is almost always 0 for this agent.

## Check 2: cost

| | ms per uncached call | ms per decision with the cache | ms per cache hit |
|---|---|---|---|
| headroom, no search (399 decisions, 6 games) | 1.42 | 0.83 | 0.036 |
| strength, same states | 1.46 | 0.84 | 0.023 |
| strength with growth, every joker in the table (worst case) | 4.05 | 2.28 | 0.023 |
| headroom, with search, same nodes (466 decisions, 7,134 simulations) | 1.36 | 7.14 | - |
| strength, with search, same nodes | 1.50 | 7.85 | - |

The strength costs about 10% more than the headroom per call (the bootstrap); the cache hit rate is the same
(45% without search, 68-69% with). The machine was shared, so treat these as +/- 10%.

## Check 3: tests (`tests/test_strength.py`, 20 tests)

Bootstrap (exact cases, seeding, monotone in the target); discard and scale corrections; targets (The Wall
x4 on the current ante only, The Needle, Violet Vessel, Gold Stake scaling, Plasma); survives-through-ante;
the four score kinds stay in [0, 1]; no leakage (unchanged under reshuffled deck order and game RNG, and
the call does not touch either); fresh-round reading; determinism and caching; same hands as the headroom;
default Phi unchanged and strength never constructed; Phi with strength: range, terminal 0, PPO shaping
telescopes to -Phi(s0) through `BalatroEnv`; strength head leaves every other weight identical after
`torch.manual_seed(0)`; targets recorded and trained only with the flag; growth off by default and empty
table is a no-op; projection; table from logs; the calibration tool; an end-to-end `train run` in worker
processes with the strength Phi and head.

## Projected growth: what exists and what it needs

`growth.py log` plays games and writes one line per blind start: `{"game", "round", "jokers": [[uid, key,
val]]}`. `growth.py build` averages, per joker key, the change in `val` between consecutive blind starts
for jokers held at both (minimum 20 rounds; `loyalty_card` excluded: its `val` is a cycling counter).
The built-in table is empty, so `growth: true` alone changes nothing (tested).

From the 150 games above (untrained agent, 1,712 rounds) the script gives, for example: `runner` +7.0
chips/round (n=32), `castle` +6.7 (23), `square` +3.2 (20), `ceremonial` +2.8 (41), `green_joker` +1.1
(109), `ride_the_bus` +0.25 (64), `constellation` +0.022 (36), `ice_cream` -11.8 (57), `popcorn` -4.0
(64), `ramen` -0.056 (53). I did not ship this table: 20-100 rounds per joker from a player that never
wins is too thin, and the gains depend on who plays. **I did not measure calibration with
growth on.** That needs a table from the agent it is for (a few thousand games) and an outcome for later
antes ("boss of ante +k cleared"), which the tool does not log yet.

## Keep / do not keep

Keep:
- `strength.py` with `extra_draws`: a probability on a known scale, 10% dearer than the headroom, calibrated
  against the solver.
- "Survives through ante" and the per-ante chances as diagnostics and as the aux head's targets.
- `strength_check` as the standing calibration tool (it also shows the solver's own calibration).
- `head_term: strength` with `score: horizon` as an option to try in Stage 3, with `value_init_bias` refit.

Do not keep, or do not rely on:
- `score: expected_total` and `expected_antes` (worse than the old Phi on this data); `current` (saturates).
- `score_scale`: no setting beat the discard model alone on the solver reference; it is left at 1.0.
- `growth` until there is a real table and a calibration of it.
- The ante-8 output as a training signal: nearly constant zero.

## Where I disagree with the spec, or could not do it

1. **Win calibration could not be run**: 0 wins in 150 games. All Phi conclusions rest on substitute
   outcomes from a weak, no-search agent.
2. **The strength term does not out-predict the headroom term** within an ante. Both are monotone
   functions of the same 32 scores against the same target. The case for the strength is its scale and its
   look-ahead, not accuracy.
3. **The cumulative product treats antes as independent** and uses today's build for all of them. Later
   antes are therefore near 0 for an early build, and the score is dominated by the next one or two bosses.
4. **Discard weight is ambiguous** (1.0 fits the solver, 2.0 fits real outcomes); I chose 1.0.
5. **The low-probability end is wrong against real outcomes** for every model including the solver; not fixed.
6. **Boss effects are ignored** by design; on boss blinds the top bucket (predicted 0.99) cleared 0.94.
7. **The aux targets are the calculated strength of the current state** (the network learns to compute
   it), not of the next blind's start and not a realised outcome. With `head_term: strength` the head is
   partly redundant with the Phi input.
8. `rows["headroom"]` and the log's `phi_headroom_term` hold Phi's head term, which is the strength score
   when `head_term: strength`. The `next_head` aux target stays the tanh headroom in both cases.
9. Seeding the bootstrap per state (as specified) means near-identical builds get independent sampling
   noise; a fixed resampling matrix would be smoother. The larger noise source is K = 32 anyway.
