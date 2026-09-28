# balatro-rl — a neural net that learns to play Balatro (Gold Stake)

This project has three parts:

1. **A fast Balatro simulator** (`balatro_rl/sim/`) with all of the game's content, and Gold Stake rules on by default.
2. **A neural network plus a training pipeline** (`model.py`, `train.py`). The net first imitates a rule-based player (behaviour cloning with DAgger), then improves with reinforcement learning (PPO).
3. **A bridge to the real game** (`bridge.py`). It reads the real game's state through the [BalatroBot](https://github.com/coder/balatrobot) mod and sends back the model's moves.

> **Expectations.** Nothing here beats Gold Stake out of the box. Balatro is very hard for RL:
> runs last about 250 decisions, there are 475 possible actions, and the long-term shop decisions
> only pay off antes later. The code is a complete, working base. How strong it gets depends on how
> long you train it, and on how faithful you make the simulator (see *Limitations*).

---

## Install

**Windows with an NVIDIA GPU:** install Python 3.11 or newer from python.org and tick "Add python.exe to PATH". Then, in a terminal opened in this folder:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy pytest
python -c "import torch; print(torch.cuda.is_available())"     # should print True
```

**Any other setup:**

```bash
python -m pip install -r requirements.txt      # Python 3.10+, numpy, torch
python -m pytest -q tests                        # rules, content fuzzing, bridge consistency (~2 min)
```

**Optional, about 5x faster simulator:** score predictions have a compiled version in `balatro_rl/sim/_fastscore.pyx`. Building it needs a C compiler (on Windows, the Microsoft C++ Build Tools: `winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`):

```bash
python -m pip install cython
cythonize -i -3 balatro_rl/sim/_fastscore.pyx
```

It is used automatically once built, and gives exactly the same results as the Python code (`tests/test_fastscore.py` checks this). Rebuild it after changing the `.pyx` file. Set `BALATRO_PYSCORE=1` to force the pure-Python scorer.

## Quick start

```bash
# Baselines on the simulator (Red Deck, Gold Stake)
python -m balatro_rl.evaluate --policy random    --games 50
python -m balatro_rl.evaluate --policy heuristic --games 200

# 1) Behaviour cloning: copy the rule-based player (DAgger).  ~30-60 min on a laptop
python -m balatro_rl.train bc  --iters 150 --envs 8  --out checkpoints/bc.pt

# 2) PPO reinforcement learning, starting from the BC model. Leave this running for hours or days.
python -m balatro_rl.train ppo --init checkpoints/bc.pt --iters 20000 --envs 16 --steps 128 --out checkpoints/ppo.pt

# 3) Evaluate a checkpoint on 200 unseen seeds
python -m balatro_rl.evaluate --policy checkpoints/ppo_best.pt --games 200

# Watch one simulated game move by move
python -m balatro_rl.evaluate --policy checkpoints/ppo_best.pt --games 1 --verbose
```

Training writes a JSON line per iteration to `checkpoints/<name>_log.jsonl`. `blinds` is the mean
furthest blind beaten over the last 200 runs (24 means a win; blinds replayed after Hieroglyph/Petroglyph don't count), `ante` is the mean ante reached,
and `win%` is the win rate. `ppo_best.pt` is saved whenever `blinds` reaches a new high.

**Tuning tips**
- Set `--envs` to your CPU core count. The simulator runs in parallel worker processes, and the network uses the GPU if one is available.
- With a GPU, PPO is faster with about twice your core count for `--envs` (the GPU handles a bigger batch almost as fast). Halve `--steps` to keep the same number of samples per update (e.g. 8 cores: `--envs 16 --steps 64`).
- `--pipeline` makes two groups of environments take turns, so one group simulates while the GPU picks actions for the other. It only helps when the simulator is the bottleneck, i.e. without the compiled scorer; with it built, leave `--pipeline` off.
- Train on any of the 15 decks with `--deck RED|BLUE|YELLOW|GREEN|BLACK|MAGIC|NEBULA|GHOST|ABANDONED|CHECKERED|ZODIAC|PAINTED|ANAGLYPH|PLASMA|ERRATIC`, and on easier stakes with `--stake WHITE` etc. A curriculum usually helps: train on WHITE first, then fine-tune on GOLD with `--init`.
- If PPO gets worse right after BC, lower `--lr` (for example `5e-5`) or raise `--ent`.

## Included checkpoints and results so far

`checkpoints/bc.pt` and `checkpoints/ppo.pt` come from a short demo run on 2 CPU cores:
about 27 minutes of behaviour cloning, then 22 minutes of PPO (the best point of a longer run).
They were evaluated on the same 100 unseen seeds (Red Deck, Gold Stake, greedy actions):

| Policy | Blinds beaten (of 24) | Mean ante reached | Wins |
|---|---|---|---|
| Random legal moves | 0.02 | 1.00 | 0 |
| Rule-based heuristic | 5.52 | 2.39 | 0 |
| Behaviour cloning (`bc.pt`) | 5.47 | 2.41 | 0 |
| BC + PPO (`ppo.pt`) | **5.94** | **2.56** | 0 |

These numbers are lower than an earlier version of this project that left out ~50 jokers.
With the full joker pool the shops are more diluted, as in the real game.
PPO beats its teacher after very little training, but it is nowhere near winning Gold Stake.
That will need far more training (see the time estimates you were given).

In the demo run, PPO started getting worse after about 125 iterations, because its exploration
bonus was too strong for this action space. The default `--ent` is now 0.002 to prevent that.
Always evaluate `ppo_best.pt`, not the last checkpoint.

## Playing the real game

1. Install **BalatroBot** by following its docs: <https://coder.github.io/balatrobot/latest/installation/>.
   It needs Steamodded and Lovely, and it serves a JSON-RPC API on `http://127.0.0.1:12346`.
   Check it responds:
   ```bash
   curl -X POST http://127.0.0.1:12346 -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"health","id":1}'
   ```
2. Run the bridge. It starts runs itself, from the main menu:
   ```bash
   python -m balatro_rl.bridge --model checkpoints/ppo_best.pt --deck RED --stake GOLD --runs 5 --verbose --delay 0.5
   # or run the rule-based player instead:
   python -m balatro_rl.bridge --model heuristic --runs 1 --verbose
   ```
   You need to have unlocked the stake and deck on your save for `start` to accept them.
   BalatroBot has no call for Director's Cut / Retcon boss rerolls, so the bridge never picks that action.

If the game rejects the model's move, the bridge tries the model's next-best legal move (up to 6),
then falls back to a safe action. You can test the whole path without the game, using a mock server
that speaks the same API and is backed by the simulator:

```bash
python tests/mock_balatrobot.py --port 12346 &
python -m balatro_rl.bridge --model heuristic --runs 3
```

## How it works

**Action space (475).** Plays and discards each have 218 candidate slots. With 8 or fewer cards in hand, those are exactly every subset of 1–5 cards. With a bigger hand (up to 16 cards), the play slots hold the 218 highest-scoring plays and the discard slots the 218 most promising discards; the observation says which cards each slot holds. The rest are: select or skip the blind, reroll the boss (Director's Cut / Retcon), buy shop card 1–4, buy pack 1–2, buy the voucher, reroll, leave the shop, sell joker 1–8, sell or use consumable 1–3, pick pack card 1–5 or skip the pack, and swap two neighbouring jokers (to set joker order). Illegal actions are masked out, and rerolls and swaps are capped per shop so the agent can't loop forever.
Tarot and spectral targets are chosen by a fixed rule (`Game.auto_targets`), so the net only decides *whether* and *which* consumable to use.

**Observation.** The observation has these parts:
- **Global features** (196): ante, blind, target, chips, hands, discards, hand size, money and debt limit, boss, tag, hand levels and play counts, cards left in the deck by rank and suit, deck enhancements and seals, and vouchers owned.
- **Hand cards:** 16 slots × 40 features. Face-down cards show only a "hidden" flag.
- **Jokers:** 8 slots, each with a learned embedding plus edition, sticker and scaling-value features. Face-down jokers (Amber Acorn) show as unknown.
- **Consumables:** one embedding each.
- **One feature row per action.** For plays, this row holds the *simulated expected score* of that exact play (all joker effects included), the hand type, and whether it wins the blind. For discards, it describes what the kept cards could still make. For shop and pack items, it holds the item's embedding, cost, edition and stickers.

**Network** (`model.py`). This is a "pointer" policy: one shared network scores each action from
`(state embedding, action embedding)`. Card-subset actions also get the average of the chosen cards'
embeddings. A separate value head feeds PPO. It has about 0.4M parameters.

**Rewards.** +1 per blind beaten further than ever before in the run, +10 for beating ante 8, and on a loss at a new blind up to +0.5 for how close you got. Blinds replayed after Hieroglyph or Petroglyph (−1 Ante) earn nothing, so the agent can't farm reward by setting itself back.

## What the simulator models

- **Scoring:** Balatro's scoring order, with retriggers, editions, enhancements, seals and held-in-hand effects.
- **Hand rules:** Four Fingers, Shortcut, Smeared, Splash and Pareidolia.
- **Jokers:** all 150 (61 common, 64 uncommon, 20 rare, 5 legendary), with the real rarities and prices. Legendaries come only from The Soul.
- **Consumables:** all 12 planets, 22 tarots and 18 spectrals. The Soul and Black Hole appear at the real 0.3% rate. Perkeo's Negative copies don't use slots.
- **Vouchers, tags, decks, stakes:** all 32 vouchers, 24 tags, 15 decks and 8 stakes.
- **Bosses:** all 23 boss blinds and 5 finishers. The House, Wheel, Fish and Mark deal cards face down, and the agent can't see them. Amber Acorn flips jokers face down.
- **Hand size:** up to 16 cards (Juggler, Turtle Bean, Troubadour, Paint Brush, Palette, Juggle Tag, Painted Deck), and down again for Stuntman, Merry Andy, Ouija, Ectoplasm and The Manacle.
- **Gold Stake rules:** no Small Blind reward, faster blind scaling, −1 discard, and Eternal, Perishable and Rental stickers.
- **Shop:** weights, rarities, edition odds, prices (including Balatro's discount rounding) and reroll costs match the wiki. The first shop guarantees a Buffoon pack. Magic Trick and Illusion put playing cards in the shop.
- **Checks:** unit tests cover scoring and the new jokers, and a fuzz test plays 150 random games stuffed with random jokers, vouchers and consumables.

## Limitations (where to improve)

- **Randomness differs.** The simulator uses Python's RNG, not Balatro's seeded one, so a given seed deals different cards and shops than the real game.
- **Small rule approximations.** A few details are simplified: the order Baseball Card applies its ×1.5s, which boss effects count as "triggered" for Matador, the odds inside Standard packs, and Luchador on bosses that change hands or discards at the start of the round.
- **Consumable targets.** Tarots and spectrals that need cards pick them with a fixed rule (`Game.auto_targets`), not the net.
- **Bridge reading.** Jokers with counters (Ride the Bus, Castle, Idol, To Do List …) are read from their description text. Hiker's bonus chips are read the same way. This is best-effort, because it depends on how BalatroBot words the text.
- **Teacher quality.** The rule-based player used for warm-starting is only moderately good.

## Layout

```
balatro_rl/
  sim/cards.py hands.py scoring.py jokers.py items.py game.py   # the simulator
  sim/_fastscore.pyx fastscore.py   # optional compiled score prediction (same results, ~60x faster per play)
  env.py        # action space, observation + action features, reward
  heuristic.py  # rule-based player (baseline + teacher)
  model.py      # actor-critic network
  vec_env.py    # parallel environments
  train.py      # bc (DAgger) and ppo
  evaluate.py   # benchmark policies
  bridge.py     # play the real game through BalatroBot
tests/          # scoring tests, bridge-vs-simulator consistency test, mock BalatroBot server
```
