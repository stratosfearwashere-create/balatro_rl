"""Talk to a GPT model from the terminal.

    # OpenAI-compatible API (needs OPENAI_API_KEY; no extra packages)
    python -m balatro_rl.chat --model gpt-4o

    # Local GPT-2 (needs `pip install transformers`; downloads ~500 MB on first use)
    python -m balatro_rl.chat --local gpt2

Commands inside the chat: /reset clears the history, /quit exits.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

SYSTEM = "You are a helpful assistant."


def api_reply(messages, model: str, base_url: str, key: str) -> str:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "messages": messages}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def make_local(name: str):
    from transformers import pipeline          # GPT-2 is a plain text continuer, not a chat model
    gen = pipeline("text-generation", model=name)

    def reply(messages) -> str:
        prompt = "".join(f"{'You' if m['role'] == 'user' else 'GPT'}: {m['content']}\n"
                         for m in messages if m["role"] != "system") + "GPT:"
        out = gen(prompt, max_new_tokens=80, do_sample=True, top_p=0.9,
                  return_full_text=False, pad_token_id=gen.tokenizer.eos_token_id)[0]["generated_text"]
        return out.split("\nYou:")[0].strip()
    return reply


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-4o", help="API model name")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--local", metavar="NAME", help="use a local Hugging Face model instead, e.g. gpt2")
    args = ap.parse_args()

    if args.local:
        reply = make_local(args.local)
    else:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Set OPENAI_API_KEY, or use --local gpt2.")
        reply = lambda msgs: api_reply(msgs, args.model, args.base_url, key)

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            history = history[:1]
            continue
        if not text:
            continue
        history.append({"role": "user", "content": text})
        answer = reply(history)
        history.append({"role": "assistant", "content": answer})
        print(f"gpt> {answer}\n")


if __name__ == "__main__":
    main()
