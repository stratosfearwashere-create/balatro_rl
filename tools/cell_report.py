"""The build of the deepest entry of a cell-explore archive: replay its action log (the log is the truth) and read the game.

    python tools/cell_report.py --seed 42010 --archive traces/cell2_42010.jsonl [--top 3]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("BALATRO_DEV_ENGINE", "")
from cell_explore import Sim, apply_actions, parse_info                          # noqa: E402
from astar_bound import KEYS                                                     # noqa: E402

HANDS = ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"]
EDITIONS = {0: "", 1: "foil", 2: "holo", 3: "polychrome", 4: "negative"}


def report(seed, stake, actions):
    s = Sim(2, "RED", stake, seed)
    s.clone(Sim(1, "RED", stake, seed), [0], [0])
    apply_actions(s, [0], [actions])
    inf = parse_info(s.info([0])[0])
    lines = [f"ante {inf['ante']} blind {inf['blind']}, ${inf['money']}, {len(actions)} actions, vouchers mask {inf['vouchers']:b}, joker slots {inf['jslots']}, consumables held {inf['ncons']}"]
    lines.append("jokers (left to right): " + ", ".join(f"{KEYS[k]}" + (f" [{EDITIONS.get(e, e)}]" if e else "") + (" eternal" if f & 1 else "") + (" rental" if f & 2 else "") + (" perishable" if f & 4 else "")
                                                      for k, e, f in inf["jokers"]))
    lines.append("hand levels: " + ", ".join(f"{h} {l}" for h, l in zip(HANDS, inf["levels"]) if l > 1) or "all level 1")
    d = inf["deck"]
    lines.append(f"deck: {d[0]} cards; enhanced {sum(d[2:13])}, sealed {sum(d[14:21])}, with edition {sum(d[22:29])}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42010)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--archive", required=True)
    p.add_argument("--top", type=int, default=3)
    a = p.parse_args()
    rows = [json.loads(x) for x in open(a.archive)]
    rows.sort(key=lambda r: (-r["depth"], -len(r["actions"])))
    for r in rows[:a.top]:
        print(f"--- depth {r['depth']} (blind index), visits {r['visits']}")
        print(report(a.seed, a.stake, r["actions"]))


if __name__ == "__main__":
    main()
