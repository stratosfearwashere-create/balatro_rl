# Cloud report: Claude teacher games on Yellow Deck / Gold stake

Branch `lm-teacher`. Teacher: Claude Opus 5.5 via `claude -p` (effort high); honest Monte-Carlo round player (`MCRounds`); lessons loop on (`--lessons-every 5`).

## Result

60 games (4 batches of 15, seeds 7100000-7100059), 1 win.

- Mean blinds beaten: **11.85** (median 12; batch means 11.2, 10.5, 12.7, 13.0).
- Wins: 1 of 60 (24 blinds, in batch 4). Next best: 23, 23, 22, then 19, 18, 18.
- Distribution (blinds: games): 0: 2, 1: 2, 2: 1, 4: 1, 5: 2, 6: 1, 7: 1, 8: 1, 9: 5, 10: 3, 11: 6, 12: 8, 13: 7, 14: 6, 15: 2, 16: 2, 17: 3, 18: 2, 19: 1, 22: 1, 23: 2, 24: 1.
- Earlier local teacher games reached 12, 13, 18 and 18 blinds; the old best PPO agent was 7.6 honest.
- Per-batch cost (Claude Code `total_cost_usd` equivalent, not necessarily billed): $33.9, $34.9, $38.6 and $44.6 (about $152 in total). About 860-1000 calls per batch.
- All four batches started before 07:59 GMT, so all ran with `--parallel 16` (batches ran 28-35 minutes each). No usage-limit waits were seen in the logs (`.err` output empty; no `lessons_error`, no `target_fallback` rows).

## Per game

| batch | game | seed | blinds | ante | won |
|---|---|---|---|---|---|
| 1 | 8 | 7100008 | 1 | 1 | no |
| 1 | 3 | 7100003 | 5 | 2 | no |
| 1 | 13 | 7100013 | 10 | 4 | no |
| 1 | 4 | 7100004 | 10 | 4 | no |
| 1 | 9 | 7100009 | 13 | 5 | no |
| 1 | 0 | 7100000 | 11 | 4 | no |
| 1 | 5 | 7100005 | 12 | 5 | no |
| 1 | 7 | 7100007 | 9 | 4 | no |
| 1 | 1 | 7100001 | 14 | 5 | no |
| 1 | 2 | 7100002 | 12 | 5 | no |
| 1 | 10 | 7100010 | 11 | 4 | no |
| 1 | 12 | 7100012 | 14 | 5 | no |
| 1 | 6 | 7100006 | 13 | 5 | no |
| 1 | 14 | 7100014 | 17 | 6 | no |
| 1 | 11 | 7100011 | 16 | 6 | no |
| 2 | 7 | 7100022 | 0 | 1 | no |
| 2 | 12 | 7100027 | 0 | 1 | no |
| 2 | 0 | 7100015 | 1 | 1 | no |
| 2 | 9 | 7100024 | 2 | 1 | no |
| 2 | 13 | 7100028 | 11 | 4 | no |
| 2 | 5 | 7100020 | 10 | 4 | no |
| 2 | 6 | 7100021 | 9 | 4 | no |
| 2 | 10 | 7100025 | 12 | 5 | no |
| 2 | 2 | 7100017 | 12 | 5 | no |
| 2 | 3 | 7100018 | 14 | 5 | no |
| 2 | 14 | 7100029 | 13 | 5 | no |
| 2 | 4 | 7100019 | 14 | 5 | no |
| 2 | 8 | 7100023 | 19 | 7 | no |
| 2 | 1 | 7100016 | 23 | 8 | no |
| 2 | 11 | 7100026 | 17 | 6 | no |
| 3 | 14 | 7100044 | 4 | 2 | no |
| 3 | 11 | 7100041 | 5 | 2 | no |
| 3 | 3 | 7100033 | 7 | 3 | no |
| 3 | 10 | 7100040 | 9 | 4 | no |
| 3 | 2 | 7100032 | 9 | 4 | no |
| 3 | 8 | 7100038 | 13 | 5 | no |
| 3 | 7 | 7100037 | 11 | 4 | no |
| 3 | 12 | 7100042 | 15 | 6 | no |
| 3 | 4 | 7100034 | 17 | 6 | no |
| 3 | 6 | 7100036 | 11 | 4 | no |
| 3 | 1 | 7100031 | 13 | 5 | no |
| 3 | 9 | 7100039 | 18 | 7 | no |
| 3 | 13 | 7100043 | 14 | 5 | no |
| 3 | 0 | 7100030 | 22 | 8 | no |
| 3 | 5 | 7100035 | 23 | 8 | no |
| 4 | 3 | 7100048 | 6 | 3 | no |
| 4 | 9 | 7100054 | 8 | 3 | no |
| 4 | 5 | 7100050 | 13 | 5 | no |
| 4 | 2 | 7100047 | 11 | 4 | no |
| 4 | 13 | 7100058 | 9 | 3 | no |
| 4 | 12 | 7100057 | 12 | 5 | no |
| 4 | 10 | 7100055 | 12 | 5 | no |
| 4 | 7 | 7100052 | 15 | 6 | no |
| 4 | 6 | 7100051 | 14 | 5 | no |
| 4 | 4 | 7100049 | 13 | 5 | no |
| 4 | 0 | 7100045 | 12 | 5 | no |
| 4 | 8 | 7100053 | 16 | 6 | no |
| 4 | 11 | 7100056 | 12 | 5 | no |
| 4 | 14 | 7100059 | 24 | 8 | WIN |
| 4 | 1 | 7100046 | 18 | 7 | no |

## What broke and how it was handled

- `python` here is 3.11 and `pip` pointed at another interpreter: installed with `python3 -m pip`. Built the engine with `python3 setup_cython.py build` (needed `cython`, `pybind11`, `numpy`; compiles with warnings only).
- **torch could not be installed**: the proxy refused the PyTorch wheel index (403) and the PyPI download of torch never completed. `balatro_rl/llm_strategist.py`, `planner.py` and `azc.py` import torch at module level, but teacher games only use the MCRounds player and the survival table. I ran the games with a small torch stand-in module kept outside the repo (scratchpad `sitecustomize.py` on `PYTHONPATH`), which makes `import torch` succeed and is never called. No repo code was changed for this. Anyone running this should install real torch, or make those imports lazy.
- **`claude -p`**: works in this cloud session both with the inherited `CLAUDE*` variables and with them stripped (as `play.py` and `lessons.py` do; the sign-in comes from the session proxy). The exact flags used by the teacher (`--tools "" --safe-mode --json-schema ...`) returned structured output. No code change was needed.
- Game logs contain no `error` text rows. `.err` files empty.

## Student training

Skipped: `nvidia-smi` is not installed (no GPU in this container). `sft.py` and `grpo.py` were not run, so there are no student results and no checkpoint.

## Lessons (end state: version 13, `balatro_rl/lm/lessons.md`, `traces/lm/lessons.json`)

The loop rewrote the lessons after every 5 games (v4 to v13). Many entries are new or sharpened: planet targets for the main hand, hard reroll floors, rental/eternal rules, a perishable cap (at most 2), "convert flat to a multiplier when the cliff is near", and a chips-vs-mult balance rule. The full text is in the file.

### Corrections I made by hand (checked against `balatro_rl/data/item_text_filled.json`)

1. Blackboard listed among "unconditional +Mult or chips" jokers; it is a conditional X3 Mult (held cards all Spades/Clubs). Removed.
2. Half Joker listed as unconditional; it needs 3 or fewer cards played. Reworded (the loop later restated it as fine for a 2-card Pair).
3. Jolly Joker listed as unconditional; it needs a Pair. Reworded to "when Pair is the main hand".
4. Supernova called "limited-use"; it adds the times a hand was played this run (a lasting scaler). Replaced by Seltzer (10 hands).
5. Riff-Raff and Supernova named as "flat or perishable fillers"; neither is. Reduced the list to Swashbuckler.
6. Perishable called an attribute of Jolly/Raised Fist; it is a sticker. Reworded the stopgap rule.
7. "How often you have played the hand does not matter" ignored Telescope (its planet is for the most played hand). Added the exception (the loop kept it).
8. Seltzer called a flat +Mult rental; it is a retrigger. Fixed.
9. Ramen listed with Castle as wanting many discards; Ramen loses xMult per discarded card, so it wants zero. Fixed.

Items I checked and found right: Gros Michel (+15 Mult, destroyable), Cavendish, Card Sharp, Acrobat, Vampire, The Idol, Ancient, Baseball Card, The Order, Hologram, Obelisk, Loyalty Card (X4 every 6 hands), Flower Pot, Seeing Double, Duo, Ice Cream, Popcorn, Turtle Bean, Stuntman, Bull/Bootstraps.

## Reading the games

Every batch failed the same way as the earlier games: by ante 5-6 the build has too little lasting xMult and the main hand is under-levelled. The lessons loop mostly adds spending rules (reroll caps, interest floors); the one win and the 22-24 blind games should be compared against the rest for what they bought before ante 4.
