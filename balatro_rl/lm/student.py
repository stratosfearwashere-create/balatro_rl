"""The student as a player for lm/play.py (--backend student): a base model plus a LoRA adapter (lm/sft.py, lm/grpo.py), with the requests
of the games in flight batched into one generate call.

ask(user) is called from play.py's worker threads; a single GPU thread collects up to --batch waiting requests, generates, and answers each.
"""
from __future__ import annotations

import json
import os
import queue
import threading
from concurrent.futures import Future

from .render import parse_answer
from .sft import SHORT_SYSTEM


class Student:
    def __init__(self, adapter, base=None, batch=8, max_new=384, temperature=0.0, system="short"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        meta = {}
        if adapter and os.path.exists(os.path.join(adapter, "lm_meta.json")):
            meta = json.load(open(os.path.join(adapter, "lm_meta.json")))
        base = base or meta.get("base", "Qwen/Qwen3-4B-Instruct-2507")
        self.system = SHORT_SYSTEM if meta.get("system", system) == "short" else __import__("balatro_rl.lm.render", fromlist=["SYSTEM"]).SYSTEM
        self.tok = AutoTokenizer.from_pretrained(adapter or base)
        self.tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
        self.model = model.eval()
        self.batch, self.max_new, self.temperature = batch, max_new, temperature
        self.q = queue.Queue()
        self.usage = {"calls": 0, "batches": 0, "output": 0}
        threading.Thread(target=self._worker, daemon=True).start()

    def prompt(self, user):
        return self.tok.apply_chat_template([{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True)

    def _worker(self):
        import torch
        while True:
            items = [self.q.get()]
            while len(items) < self.batch:
                try:
                    items.append(self.q.get(timeout=0.05))
                except queue.Empty:
                    break
            texts = [self.prompt(u) for u, _ in items]
            try:
                enc = self.tok(texts, return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    kw = dict(max_new_tokens=self.max_new, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
                    if self.temperature > 0:
                        kw.update(do_sample=True, temperature=self.temperature, top_p=1.0)
                    else:
                        kw.update(do_sample=False)
                    out = self.model.generate(**enc, **kw)
                gen = out[:, enc["input_ids"].shape[1]:]
                for (u, fut), g in zip(items, gen):
                    fut.set_result(self.tok.decode(g, skip_special_tokens=True))
                self.usage["batches"] += 1
                self.usage["calls"] += len(items)
                self.usage["output"] += int((gen != (self.tok.pad_token_id or -1)).sum())
            except Exception as ex:
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(ex)

    def generate(self, user) -> str:
        fut = Future()
        self.q.put((user, fut))
        return fut.result()

    def ask(self, user, schema=None):
        """play.py's interface: -> (answer dict or None, error)"""
        try:
            text = self.generate(user)
        except Exception as ex:
            return None, f"generate failed: {ex.__class__.__name__}: {str(ex)[:200]}"
        m = None
        try:
            import re
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)), None
        except Exception:
            return None, f"bad answer: {text[:200]}"
