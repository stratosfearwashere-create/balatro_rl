"""traces/joker_strength.jsonl (tools/joker_strength.py) -> balatro_rl/joker_strength.json, the strength columns of the item descriptors (joker_desc.item_descriptors(strength=True)).

Per joker: its measured gain (log10 of the share of the boss's chip target, with the joker against without) at the bosses of antes 2, 4, 6 and 8, then a flag that says the measure applies.
  * a gain that is not above twice its standard error is set to 0 (it is not measured to be anything);
  * the gains are clipped to [-0.5, 1.5];
  * economy jokers and copy jokers (Blueprint, Brainstorm) get the flag 0 and gains 0: the test pays in chips and mult, not money, and copies act on a neighbour a random context rarely has.

    python tools/export_strength.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("BALATRO_DEV_ENGINE", "")
from seed_report import role_of                                                  # noqa: E402

ANTES = (2, 4, 6, 8)


def main(src="traces/joker_strength.jsonl", dst="balatro_rl/joker_strength.json"):
    rows = [json.loads(x) for x in open(src)]
    out = {}
    for r in rows:
        k = r["joker"]
        e = out.setdefault(k, {"gains": [0.0] * len(ANTES), "measured": 0.0 if role_of(k) in ("econ", "copy") else 1.0})
        if e["measured"] == 0.0 or r["ante"] not in ANTES:
            continue
        g = r["gain"] if abs(r["gain"]) > 2 * r["gain_se"] else 0.0
        e["gains"][ANTES.index(r["ante"])] = round(max(-0.5, min(1.5, g)), 4)
    json.dump({"antes": list(ANTES), "jokers": out}, open(dst, "w"), indent=0)
    print(f"{len(out)} jokers -> {dst}; measured {int(sum(v['measured'] for v in out.values()))}")


if __name__ == "__main__":
    main()
