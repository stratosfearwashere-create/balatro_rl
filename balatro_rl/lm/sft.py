"""Warm-up: fine-tune the student (a small open model, LoRA) on the teacher's decisions (lm/play.py logs).

Each example: the system prompt, the decision's message (the run's state, CHECKPOINTS, ESTIMATE, legal moves) -> the teacher's JSON answer
(reasoning, plan, move). Loss on the answer only.

--system short (default): the student gets a short system prompt (role + answer format) instead of the teacher's full playbook; it learns
the playbook's decisions from the teacher's answers and still reads the CHECKPOINTS block (the playbook's targets computed for each state),
so each decision costs about a third of the tokens (important for GRPO). --system full: the teacher's own system prompt.

Games can be filtered or weighted by how far they got (--min-blinds): a teacher decision from a game that died early is still mostly a
sound decision, so the default keeps everything.

Run in the fine-tuning environment:
    C:/venvs/balatro-lm/Scripts/python -m balatro_rl.lm.sft --data traces/lm/teacher.jsonl --out checkpoints/lm/sft_v1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

SHORT_SYSTEM = """You are the strategist in a game of Balatro on GOLD STAKE with the Yellow Deck. Goal: win the run (beat the ante-8 Boss Blind).
You make every decision outside a round (shop, packs, consumables, blind select / skip); an exact solver plays the rounds for the most
money. Play by the Gold Stake playbook: survive first; a multiplier (xMult) by ante 4-5 is worth more than anything else; level one main
hand with planets; hold $25 from ante 3 (+$3 per rental); no rentals before ante 5; skip blinds only for the Investment tag; thin the deck.
Inside a round you are asked at its start and when a consumable could be used: use it now (naming target cards) or let the round
player go on, and set the style of play for the rest of the round (a joker to grow, a hand to play, a rank or suit to discard).
Answer in JSON: {"reasoning": "2-6 short sentences", "plan": "your updated plan note (outside rounds)", "move": <number of the move>,
"targets": [card numbers for a tarot that acts on selected cards, else nothing], "style": {"grow_joker": n, "hand": "", "discard": ""}
(in rounds)}."""


def load_examples(paths, system, min_blinds):
    from .render import SYSTEM
    sys_text = SYSTEM if system == "full" else SHORT_SYSTEM
    rows, results = [], {}
    for p in paths:
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l)
            key = (p, d.get("seed"))
            if d.get("kind") == "result":
                results[key] = d
            elif d.get("kind") == "decision" and d.get("answer") and d.get("move_index") is not None and not d.get("error"):
                rows.append((key, d))
    out = []
    for key, d in rows:
        r = results.get(key)
        if r is None or r["blinds"] < min_blinds:
            continue
        ans = d["answer"]
        tgt = {"reasoning": ans.get("reasoning", "")}
        if d.get("state") != "ROUND":
            tgt["plan"] = ans.get("plan", "")
        tgt["move"] = int(ans["move"])
        tgt["targets"] = [int(x) for x in (ans.get("targets") or []) if isinstance(x, (int, float))]
        if d.get("state") == "ROUND":
            tgt["style"] = ans.get("style") or {"grow_joker": 0, "hand": "", "discard": ""}
        target = json.dumps(tgt, ensure_ascii=False)
        out.append({"prompt": [{"role": "system", "content": sys_text}, {"role": "user", "content": d["user"]}],
                    "completion": [{"role": "assistant", "content": target}]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=["traces/lm/teacher*.jsonl"])
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--out", default="checkpoints/lm/sft_v1")
    p.add_argument("--system", default="short", choices=["short", "full"])
    p.add_argument("--min-blinds", type=float, default=0.0)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--max-len", type=int, default=6144)
    p.add_argument("--val", type=float, default=0.05, help="share of games held out")
    a = p.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    paths = sorted({f for pat in a.data for f in glob.glob(pat)})
    ex = load_examples(paths, a.system, a.min_blinds)
    random.Random(0).shuffle(ex)
    n_val = max(1, int(a.val * len(ex))) if len(ex) > 20 else 0
    train, val = ex[n_val:], ex[:n_val]
    print(json.dumps({"files": paths, "examples": len(ex), "train": len(train), "val": len(val)}), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    lora = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
                    per_device_train_batch_size=1, gradient_accumulation_steps=a.accum, gradient_checkpointing=True, bf16=True,
                    max_length=a.max_len, logging_steps=5, save_strategy="epoch", eval_strategy="epoch" if val else "no",
                    per_device_eval_batch_size=1, report_to=[], completion_only_loss=True)
    tr = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(train), eval_dataset=Dataset.from_list(val) if val else None,
                    processing_class=tok, peft_config=lora)
    tr.train()
    tr.save_model(a.out)
    tok.save_pretrained(a.out)
    json.dump({"base": a.model, "system": a.system, "data": paths, "examples": len(ex)}, open(os.path.join(a.out, "lm_meta.json"), "w"))
    print("->", a.out, flush=True)


if __name__ == "__main__":
    main()
