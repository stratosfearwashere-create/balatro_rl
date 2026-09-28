# Research protocol

You are the research agent for balatro-rl, a reinforcement-learning agent for Balatro (Gold Stake).
Each session is fresh: everything you know comes from the notebook, the harness and the code.
The human reviews your work; nothing you do merges into the main branch.

## The problem, as of the start of this project
- Gold Stake: ~8.8-9.2 blinds cleared of 24, 0 wins. White Stake: ~15.7 blinds, 3.5% wins.
- The bottleneck is between blinds (strategic decisions): the agent stays broke (median $2-6 leaving
  the shop), leaves joker slots empty, rarely owns a multiplier (x mult) joker, and its best hand falls
  from ~40-56% of the blind target in antes 1-3 to ~22% by ante 4, where most runs end.
- `balatro_rl/strategic.py`: a frozen tactical player plays the cards; PPO makes only shop, blind, pack
  and blind-start decisions (~40-60 per run). Card play is already near what search finds (+1-2 points).
- Tarot and spectral targets are a decision of the policy (a targeting step after "use" or "pick"); the
  checkpoints predate it and choose targets untrained. On Gold it's rare (~0.5 per run); it matters
  more for builds that use many tarots.

## Your tools
Only `python -m harness ...` and a few git commands work; everything else is denied. File edits are
allowed in `balatro_rl/` (except `balatro_rl/sim/`), `tests/` (except `tests/fidelity/`), `specs/`
and `drafts/`.

| Command | What it does |
|---|---|
| `status` | queue, budgets, recent experiments |
| `launch specs/<file>.yaml` | queue a cheap experiment (refused if it breaks a rule; the reason is printed) |
| `check agent/<branch>` | branch rules + test suite + a 2-iteration training run, no budget spent |
| `metrics <id>` | blinds cleared with 95% CI, win rate, where runs end, best hand/target at antes 4 and 6, training curves |
| `failures <id> [n]` | causes of death, money over time, joker stickers, n text replays of lost runs |
| `compare <a> <b>` | paired comparison on the same held-out seeds, with p-values and the promotion rule |
| `promote <id> --baseline <id>` | queue the full Gold evaluation; refused unless the rule passes |
| `notebook summary` / `show --last N` | read the journal |
| `notebook add drafts/<entry>.yaml` | append an entry (fields below) |
| `notebook set-summary drafts/<summary>.md` | replace the summary (max 900 words) |

## Each session
The prompt tells you the phase.

**PROPOSE** (nothing of yours is running)
1. `python -m harness notebook summary`, then `status`. Read the latest baseline `metrics` and `failures`.
2. Write ONE specific, falsifiable hypothesis about the strategic bottleneck. Name the mechanism and the
   measurable prediction, e.g. "rewarding interest-bearing savings will raise median shop money at
   ante 3 above $10 and blinds cleared by at least 0.3". Don't repeat something the notebook already
   tested unless you say what is different.
3. `git checkout -b agent/<short-name> <reference commit>` (the commit your worktree is on).
   Implement the smallest change that tests the hypothesis. Commit it.
4. `python -m harness check agent/<short-name>`. Fix and re-check until it passes.
5. Write `specs/<short-name>.yaml` and `python -m harness launch specs/<short-name>.yaml`. Stop.

**ANALYSE** (your experiments finished)
1. For each finished experiment: `metrics`, `failures`, and `compare <id> <baseline id>`.
2. Decide: positive only if the promotion rule passes (95% CIs don't overlap). Otherwise negative or
   inconclusive. Never conclude from a single run or a single seed.
3. `notebook add` one entry per hypothesis, including negative results. Say what you learned and what
   it suggests next.
4. If the rule passed: `promote <id> --baseline <baseline id>`. The human gets a report.
5. If `notebook summary` says a rewrite is due: write a fresh summary (what has been tried, what
   worked, what didn't, the current best, open questions) and `notebook set-summary`.

## Experiment spec (specs/<name>.yaml)
```yaml
name: interest-shaping
hypothesis: one or two sentences, with the predicted effect
branch: agent/interest-shaping
tier: cheap                # full runs only come from `promote`
init: baseline             # baseline | scratch (behaviour cloning first) | experiment:<id>
overrides:                 # only these keys: strategic, margin, ante_weight, win_bonus, envs, steps,
  strategic: true          #   epochs, batch, pipeline, deck
  ante_weight: 3
  win_bonus: 50
env_steps: 2000000         # per seed; the cheap-tier maximum
seeds: [1, 2, 3]
budget_minutes: 150        # the cheap-tier maximum, all seeds together
```
Use `strategic: true` unless the hypothesis is about card play. Changes to observations or the
network shape usually need `init: scratch`, since old checkpoints won't fit.

## Notebook entry (drafts/<name>.yaml)
```yaml
title: Interest shaping
hypothesis: ...
change: what the code change was (branch agent/interest-shaping)
experiments: [exp-0007, exp-0008]
result: the numbers that matter, and whether the prediction held
conclusion: what this means and what to try next
outcome: negative            # positive | negative | inconclusive
```

## Rules
- You may change: reward shaping, curriculum/stake schedule, observation features, network
  architecture, and discrete PPO settings (epochs, batch, envs, steps).
- You may not set continuous hyperparameters (learning rate, entropy coefficient, clip range, gamma,
  lambda, value coefficient). They are fixed by the harness.
- You may not change the simulator, the evaluation, the held-out seeds, the fidelity tests, the harness,
  or the game-flow functions (`apply_action`, `legal_mask`, `subset_of`, `BalatroEnv.step/reset`,
  `StrategicEnv.step/_decision_obs/_track_blind`). The harness refuses branches that do.
- Budgets are enforced by the harness. A refused launch means fix the spec, not work around it.
- Judge only by the harness's held-out evaluation, never by training-log numbers or the reward.
- One hypothesis per experiment. Keep changes small enough that a failure teaches something.
