# Cloud brief: Claude teacher games, lessons, student training

Branch: `lm-teacher` (the cloud session starts from `main`: run `git fetch origin lm-teacher && git checkout lm-teacher` first).

## Background

A Balatro agent aiming to win Gold stake. After about 57 PPO / AlphaZero runs plateaued (best about 11.5 blinds of the 24 needed), the user
switched to a language-model route:

- A teacher (Claude Opus 5.5, called through the `claude -p` command line) plays Gold stake / Yellow Deck runs. It makes every decision
  outside a round (shop, packs, blind select), and inside rounds it decides consumable use (with target cards) and a style of play.
- An honest Monte-Carlo round player (`balatro_rl/lm/rounds.py`, `MCRounds`) plays the cards: each play or discard by its clear rate over
  64 redrawn draw orders, then the style, then money. It never sees the real draws.
- A small open model (`Qwen/Qwen3-4B-Instruct-2507`) is then fine-tuned on the teacher's decisions (`balatro_rl/lm/sft.py`) and refined with
  GRPO (`balatro_rl/lm/grpo.py`, never run yet).

Results so far (honest rounds, Yellow Gold): full teacher games reached 12, 13, 18 and 18 blinds (the old best PPO agent: 7.6 honest).
Every game failed the same way: by ante 6-7 the build is about 5x short of the ante-8 boss (too few lasting xMult jokers, main hand
under-levelled, perishable or unreliable scorers).

A lessons loop exists (`balatro_rl/lm/lessons.py`; lessons in `balatro_rl/lm/lessons.md`, version 4, appended to the system prompt). Three
lessons were corrected by hand: an outdated "do not trust the outlook" lesson, Gros Michel wrongly called xMult, and a lesson about which hand
to play (the round player decides that).

Rules from the user: run long jobs in the background; do not touch `main`; only `lm-teacher` may be committed and pushed (the user approved
pushing results there); never put API keys or credentials in files or commits; keep teacher parallelism modest (16 parallel calls filled the
user's 5-hour plan window in 12 minutes on their PC). This runs on the user's cloud session credit.

## Tasks

1. **Setup.** Check out `lm-teacher`. Build the C++ engine: `python setup_cython.py build` (Linux; needs a C++17 compiler and pybind11; read
   `setup_cython.py` and install what it needs). Check with `python -m balatro_rl.lm.play --dry-run --games 4` (prints one shop prompt, no
   model call).
2. **Make the teacher's Claude calls work.** The teacher runs `claude -p` (`balatro_rl/lm/play.py`, `ClaudeCLI._ask_once`; also
   `balatro_rl/lm/lessons.py`, function `claude`). On the user's PC the child process had to have every `CLAUDE*` environment variable
   removed to use the saved login; in the cloud the sign-in may come from those variables instead. Test a one-line
   `claude -p --output-format json` call with and without the stripping, and adapt both places so it works. If `claude -p` cannot be made to
   work, stop and write that up in the report (task 5); do not spend credit another way.
3. **Teacher games with lessons**, in the background, in batches:
   `python -u -m balatro_rl.lm.play --games 10 --parallel 4 --seed0 7100000 --lessons-every 5 --log traces/lm/teacher_cloud_b1.jsonl`
   (next batches: seed0 7100010, 7100020, ...; one log per batch). Target 40 games. Watch each log for errors (rows with `error`,
   `lessons_error`, `target_fallback`; the `.err` output; the line `stopped:`). The driver waits and retries on usage limits by itself.
   After each batch, check the lessons (`python -m balatro_rl.lm.lessons show`) and correct factual errors in `balatro_rl/lm/lessons.md` and
   `traces/lm/lessons.json` against `balatro_rl/data/item_text_filled.json` (for example a joker called xMult that is +Mult). Note each
   correction in the report.
4. **Student training, only if `nvidia-smi` shows a GPU with 16 GB or more.** `pip install "transformers>=4.51" peft trl datasets accelerate`;
   `python -m balatro_rl.lm.sft --data "traces/lm/*.jsonl" --out checkpoints/lm/sft_v1`; evaluate the student on 20 games with seeds the
   teacher did not play (`python -m balatro_rl.lm.play --backend student --adapter checkpoints/lm/sft_v1 --games 20 --seed0 7500000
   --log traces/lm/student_sft_v1.jsonl`); if that works, smoke-test `balatro_rl/lm/grpo.py` for a few updates (it has never been run; fix
   what breaks). With no suitable GPU, skip this and say so.
5. **Done** means: the 40 games (or as many as completed); a report at `traces/lm/cloud_report.md` (per game: blinds and ante reached; mean
   and distribution; the lessons at the end, with the corrections made; the student results if task 4 ran; anything that broke and how it
   was fixed); and everything except model weights committed to `lm-teacher` and pushed (game logs, lessons, report, code fixes). Do not push
   to `main`, do not commit adapters or other weights over 50 MB, and never write credentials into files.
