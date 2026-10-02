# Stage 3: shop module

Branch `az-stage3` (from `az-stage2`). Everything new is behind config flags whose defaults leave today's
behaviour unchanged: `tests/test_regression.py` (golden, bit-identical) passes and the golden file was not
regenerated. Test suite: **167 passed** (142 before + 25 new in `tests/test_stage3.py`).

## Result in one table

60 White Stake games, seeds 10000-10059, untrained network, bounded value (`value_bound: floor_sigmoid`),
2 workers. "Shop decided by" rows have in-round search off (`budget_round` 0, `budget_boss` 0).

| variant | blinds +- SE | vs priors today | vs graded prior | wins | CPU s/game | games/h (2 workers) | override % (strong) |
|---|---|---|---|---|---|---|---|
| `rule`: priors today, no search | 10.77 +- 0.59 | | -5.17 +- 0.94 | 0/60 | 2.3 | 3136 | 0.0 |
| `graded`: graded prior, no search | 15.93 +- 0.78 | **+5.17 +- 0.94** | | 3/60 | 3.8 | 1890 | 0.0 |
| `shop_gumbel`: (a) Gumbel search, `c_scale_build` 10, `heur_bonus` 1, `crn` | 12.63 +- 0.71 | +1.87 +- 0.75 | -3.30 +- 1.04 | 1/60 | 11.3 | 639 | 18.1 (0.0) |
| `shop_onestep`: (b) graded prior + one-step V | 16.53 +- 0.66 | +5.77 +- 0.97 | +0.60 +- 1.02 | 4/60 | 8.3 | 871 | 9.2 (2.7) |
| `shop_onestep_close`: (c) = (b) + paired rollouts on close calls | 15.88 +- 0.76 | +5.12 +- 1.02 | -0.05 +- 1.02 | 3/60 | 11.6 | 618 | 9.8 (3.0) |

What is and is not distinguishable at 60 games (paired SE about 0.75-1.0 blinds):

- **The graded prior alone beats today's priors by 5.2 blinds (5.5 SE) and beats the Gumbel shop search by
  3.3 blinds (3.2 SE)**, at a third of the search's cost per game. This is the Stage 3 result.
- (b) and (c) are both clearly better than (a) (+3.90 +- 0.94 and +3.25 +- 1.01).
- **One-step V and the close-call rollouts are not distinguishable from the graded prior alone** (+0.60 and
  -0.05, both within one SE). With an untrained network they cost 2.2x and 3x the CPU for nothing measurable.
- The Gumbel search's gain over the priors (+1.87 +- 0.75) is smaller here than the 300-seed figure quoted to
  me (13.42 against 10.46); at 60 games that is within noise of it.

Cost per decision (CPU ms, all decisions; the graded player makes about twice as many decisions outside
rounds, because it sells, reorders and buys more, and its games last longer):

| variant | decisions/game | of which outside rounds | CPU ms/decision | evaluations/decision |
|---|---|---|---|---|
| rule | 94.5 | 41.2 | 24.3 | 0 |
| graded | 135.6 | 80.5 | 28.1 | 0 |
| shop_gumbel | 100.6 | 53.0 | 112.1 | 11.3 |
| shop_onestep | 137.7 | 79.6 | 60.0 | 9.2 |
| shop_onestep_close | 135.9 | 78.8 | 85.7 | 11.7 |

games/hour is 3600 x workers / CPU seconds per game (the machine was shared, so wall-clock is not quoted).
For equal wall-clock: one `shop_gumbel` game costs as much as 3.0 `graded` games.

Override rates by phase (share of decisions whose final choice is not the prior's top; in brackets, at least
1 logit below it): `shop_gumbel` shop 42.1% (0.0), pack 30.1%, blind 9.4%; `shop_onestep` shop 19.6% (5.6),
pack 13.8% (4.5), blind 0.0%; `shop_onestep_close` shop 20.6% (6.3), pack 15.2% (4.6), blind 0.0%. In (c),
708 of 4,380 decisions outside rounds (16%) were close calls and the rollouts reversed 222 of them.

## Rule switches, one at a time, on the graded prior (search off)

| variant | blinds +- SE | vs graded | wins | CPU s/game | same switch on the tuning seeds (40 games) |
|---|---|---|---|---|---|
| graded | 15.93 +- 0.78 | | 3/60 | 3.8 | 16.38 |
| + `rule_skip` | 15.37 +- 0.89 | -0.57 +- 1.03 | 7/60 | 3.6 | 15.53 |
| + `rule_arcana` | 17.88 +- 0.75 | **+1.95 +- 0.81** | 11/60 | 6.0 | 17.85 |
| + `rule_hold` | 15.93 +- 0.78 | +0.00 +- 0.00 | 3/60 | 3.5 | 16.38 |
| + `rule_scaling` | 14.68 +- 0.79 | -1.25 +- 0.74 | 2/60 | 3.3 | 16.12 |
| + `rule_pace` | 15.15 +- 0.76 | -0.78 +- 0.43 | 2/60 | 3.4 | 16.32 |
| + `rule_copier` | 15.80 +- 0.82 | -0.13 +- 0.75 | 5/60 | 4.6 | 17.10 |
| + `rule_boss` | 15.12 +- 0.77 | -0.82 +- 0.50 | 2/60 | 3.7 | 16.62 |

- **Keep `rule_arcana`**: +1.95 +- 0.81 (2.4 SE) here and +1.5 on the 40 tuning seeds, wins 3 -> 11 of 60.
  It costs 1.6x per game (target sets are sampled for every pack). Confirm at 300 seeds.
- `rule_hold` changed no game at all: without `rule_arcana` no tarot is ever bought, so Hermit and
  Temperance never turn up. It can only be measured together with `rule_arcana`; the grid has
  `graded+arcana+hold` for that, which I did not run.
- `rule_skip`: blinds not distinguishable (-0.57 +- 1.03), but wins 7 against 3 here and 5 against 2 on the
  tuning seeds. It seems to trade early deaths for wins; needs 300 seeds and a look at wins, not blinds.
- `rule_scaling`, `rule_pace`, `rule_boss`: each leans negative by 1.6-1.8 SE here and is flat on the tuning
  seeds. Not distinguishable from zero; I would not keep any of the three without a 300-seed result.
  Their constants (growth table, pace threshold, boss models) are my estimates and were not tuned.
- `rule_copier`: no measurable effect (-0.13 +- 0.75; +0.7 on the tuning seeds), 1.2x the cost.

## Commands

Grid file: `stage3_grid.json` (repo root). Every variant plays seeds 10000 onwards.

    # what I ran (60 games, 2 workers; results in checkpoints/eval/stage3, which git ignores)
    python -m balatro_rl.az.compare grid  --grid stage3_grid.json --out checkpoints/eval/stage3 --games 60 --workers 2
    python -m balatro_rl.az.compare table --grid stage3_grid.json --out checkpoints/eval/stage3 --base rule --detail
    python -m balatro_rl.az.compare table --grid stage3_grid.json --out checkpoints/eval/stage3 --base graded

    # the same at 300 seeds: everything, or one variant per command with --only
    python -m balatro_rl.az.compare grid  --grid stage3_grid.json --out checkpoints/eval/stage3_300 --games 300 --workers 7
    python -m balatro_rl.az.compare grid  --grid stage3_grid.json --out checkpoints/eval/stage3_300 --games 300 --workers 7 --only rule
    python -m balatro_rl.az.compare grid  --grid stage3_grid.json --out checkpoints/eval/stage3_300 --games 300 --workers 7 --only graded
    python -m balatro_rl.az.compare grid  --grid stage3_grid.json --out checkpoints/eval/stage3_300 --games 300 --workers 7 --only shop_gumbel
    python -m balatro_rl.az.compare grid  --grid stage3_grid.json --out checkpoints/eval/stage3_300 --games 300 --workers 7 --only shop_onestep
    python -m balatro_rl.az.compare grid  --grid stage3_grid.json --out checkpoints/eval/stage3_300 --games 300 --workers 7 --only shop_onestep_close
    python -m balatro_rl.az.compare grid  --grid stage3_grid.json --out checkpoints/eval/stage3_300 --games 300 --workers 7 --only graded+skip,graded+arcana,graded+hold,graded+scaling,graded+pace,graded+copier,graded+boss,graded+arcana+hold
    python -m balatro_rl.az.compare table --grid stage3_grid.json --out checkpoints/eval/stage3_300 --base rule --detail

Variants already written to `--out` are skipped, so an interrupted run can be restarted. Expected CPU at 300
games, from the costs above: rule 12 min, graded 19, each switch 17-30, shop_onestep 42, shop_gumbel 57,
shop_onestep_close 58 (CPU minutes; divide by the workers). New in the tool: `--seed0`, a CPU s/game column,
and `agent_stats` (budget and close-call counts) in each result file.

## What was built

| File | What |
|---|---|
| `balatro_rl/az/shop.py` (new) | `ShopConfig`, `ShopPricer`: a price for every option outside a round, the seven rule switches |
| `balatro_rl/az/agent.py` | `AgentConfig.shop_prior / shop / price_feature / shop_eval / onestep_* / close_*` (documented in its docstring); `Agent.values`, `_decide_build`, `_paired_rollouts`, `_rollout`, `policy_action` |
| `balatro_rl/az/features.py`, `net.py`, `train.py` | optional price inputs (`N_PRICE` = 4 per candidate) in a separate layer created last |
| `balatro_rl/az/compare.py` | `--seed0`, cost column |
| `stage3_grid.json`, `tests/test_stage3.py` | the checks, 25 tests |

**1. Graded prior** (`shop_prior: graded`). price = W(after) - W(before), W = S + E + U:
- S, build strength: the Stage 2 score (clear chances of this ante's boss with its real target and of the
  next three) + `log_weight` x ln(mean best-hand score) + `deck_weight` x mean card quality; or Phi
  (`shop.value: phi`).
- E, economy: income per round of economy jokers x `econ_rounds`, a dollar table for vouchers the score does
  not show.
- U, money: `dollar` x a(ante) x (money + `interest_rounds` x interest steps held). So
  m(ante, money) = (U(money) - U(money - cost)) / cost is higher when the purchase drops under an interest
  threshold, and a(ante) falls linearly to `m_floor` (0.05) at ante 8.
- Jokers: measured with the joker added; with full slots the sell option carries the best "sell this, then
  buy that" pair. Planets: measured level-up. Tarots / spectrals: every target set the candidate list holds
  (packs), or sampled hands (shop). Packs: mean over `pack_samples` sampled contents of the best pick, two
  picks for a Mega pack. Reroll: running average of the best card price of the fresh shops seen this game,
  weighted to this ante, minus the cost. Vouchers: measured (hands, discards, hand size, ante) or the table.
- The prior for all options is (price - best price) x `logit_scale` (100), clipped at -30.
- `price_feature: true` builds the network with the price inputs; default networks are unchanged (tested:
  every existing weight identical after `torch.manual_seed(0)`).

All strengths of one decision are measured on the same 32 sampled hands and the same bootstrap draws. A
card keeps its place in the sampling across the copies of a decision, so a tarot's target sits in the same
hands before and after. Seeds come from a SHA-256 of the deck's contents and the ante.

**2. Rule switches**: `rule_skip`, `rule_arcana`, `rule_hold`, `rule_scaling`, `rule_pace`, `rule_copier`,
`rule_boss` (`shop.py` module doc lists what each does). All written from the game's rules and our own
heuristic's gaps; nothing was copied from the reviewed project.

**3. One-step evaluation** (`shop_eval: onestep`, needs `search: true`): each option is applied to a redrawn
copy, V is read there in one batched network call (two copies for packs, rerolls, blind select and random
consumables), score = logits + `onestep_weight` x V. Replaces the Gumbel search outside rounds only.

**4. Close calls** (`close_calls: true`): top two within `close_margin` logits -> `close_rollouts` paired
rollouts with the no-search policy, both options on the same redrawn copy and generator, to the end of the
next blind or boss; outcome V or blinds; the pair's scores move by `close_weight` x (outcome - pair mean).

Both record `Decision.policy` = softmax(scores) over `choice.cands` with `searched=True`, and the override
statistics count them (tested, and visible in the tables above).

## Tuning record

Constants were checked on separate seeds (20000-20039, 40 games, search off) so the table above is held out.
The defaults were set before these runs and none was changed because of them.

| setting | blinds | | setting | blinds |
|---|---|---|---|---|
| rule prior | 10.25 | | `interest_rounds` 0 / 4 | 16.60 / 17.57 |
| graded, defaults | 16.60 | | `value: phi` | 13.78 |
| `dollar` 0.004 / 0.016 | 14.70 / 14.18 | | `samples` 64 | 16.07 |
| `log_weight` 0.05 / 0.4 | 15.20 / 15.38 | | `card_margin` 0 / 0.01 (added after) | 16.60 / 16.38 |
| one-step, weight 300 / 100 | 16.25 / 17.10 | | one-step + close calls | 17.70 |
| Gumbel, `c_scale_build` 6 | 12.20 | | | |

SE is about 1 blind at 40 games: only the rule prior, `value: phi` and the Gumbel search are clearly
different from the rest. `value: phi` (prices from Phi itself) is 2.8 blinds worse than the strength score.

## Tests (`tests/test_stage3.py`)

Defaults unchanged (config, candidate feature size, network weights, no price ever built by a default
agent); price = strength change - cost x m; m higher under an interest threshold and falling with ante;
sell-then-buy pairs; planets, vouchers, packs, plain playing cards; the reroll average and its reset per
game; one test per rule switch; **no leakage**: two games differing only in draw order and future random
numbers get exactly equal prices (shop, full slots with consumables, Arcana pack, Standard pack, blind
select, all switches on) and identical decisions and policies for the graded prior, one-step V and paired
rollouts; pricing leaves the game and its generator untouched; one-step policy equals logits + weight x V;
close calls share each future between the two options and do not touch the reroll average; price inputs
reach the network and train; `compare` runs a Stage 3 grid.

## Where I disagree with the spec, or could not do it

1. **One-step V adds nothing yet.** The untrained V is a function of Phi (the headroom), a coarser measure
   than the price it is added to, and it gives packs and rerolls no credit at all (the build is unchanged
   until a card is picked). I expect it to matter only once V is trained. Until then the graded prior with
   search off is the better and cheaper choice.
2. **Close calls were run with a cut-down budget**: 2 rollouts, to the end of the next blind, margin 0.5
   logits, to keep a 60-game run near 10 minutes. The spec's example (to the end of the next boss) costs
   roughly 3x more per rollout and was not run. With the short horizon the outcome is mostly the same V the
   one-step evaluation already read, so this check says little about longer rollouts.
3. **The price does not use `Strength.report` directly.** Stage 2 seeds its hands per build, so two builds
   differing by one joker get independent sampling noise (0.04-0.12 in the score), which is as large as the
   differences a price must resolve. The pricer uses the same formulas on hands shared across the options.
4. **m(ante, money) is a difference of a money utility**, not a separate multiplier, so a purchase crossing
   two interest thresholds pays for both and selling is priced by the same function.
5. **The reroll average has little data**: two or three fresh shops per ante. It is a prior (0.01) that this
   game's shops pull on.
6. **Single-card changes are barely measurable**: one changed card is in about 5 of 32 sampled hands.
   Hence `deck_weight` (card quality) and `card_margin` (a plain card must gain 0.01). Tarot prices are the
   noisiest part of the module, although `rule_arcana` is the one switch that clearly helps.
7. **The switches were measured one at a time only**; no combination was run, and their constants are
   untuned.
8. The price inputs to the network are tested for plumbing and one training step only; no training run.
9. `compare changed` (share of decisions changed in the same position) was not run.
10. In all three "shop decided by" variants a spectral card usable in a round still gets `budget_spectral`
    simulations; that is the same for the three.
11. The graded prior also works inside the Gumbel search (it is the prior at inner nodes, with halved
    sample counts), but that combination was not measured.
12. The simulator's `open_pack` and `select_blind` shuffle from `full_deck` in its stored order, so a copy's
    future depends on that order. It is public information (not the draw order), and the leakage tests leave
    it alone; the pricer's own sampling sorts the deck by content first.
