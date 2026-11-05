"""Replay the recorded winning games of a run (<out>_wins.jsonl written with --record-wins) and print how each one was won.

    python tools/replay_wins.py --wins checkpoints/ppo_ante4_v1_wins.jsonl --start-ante 4 --money 500 --ban-file <comma list>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppEnv     # noqa: E402
from balatro_rl.sim.jokers import JOKERS     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wins", required=True)
    p.add_argument("--deck", default="RED")
    p.add_argument("--start-ante", type=int, default=4)
    p.add_argument("--money", type=int, default=500)
    p.add_argument("--ban", default="none", help="comma list of banned joker keys, or a path to a file holding it")
    a = p.parse_args()
    ban = open(a.ban).read().strip() if os.path.exists(a.ban) else a.ban
    keys = list(JOKERS)
    for line in open(a.wins):
        w = json.loads(line)
        if "actions" not in w:
            continue
        e = CppEnv(a.deck, w["stake"])
        e.e.start_ante, e.e.start_money_p, e.e.start_money = a.start_ante, 1.0, a.money
        if ban.lower() != "none":
            e.e.banned = [keys.index(k) for k in ban.split(",")]
            e.e.ban_p = 1.0
        e.reset(w["seed"])
        antes = []
        for i, x in enumerate(w["actions"]):
            before = e.game.dump()["ante"]
            _, _, done, _ = e.step(int(x))
            d = e.game.dump()
            if d["ante"] != before:
                antes.append((i, before, d["ante"]))
            if done:
                break
        d = e.game.dump()
        print(f"seed {w['seed']} iter {w['iter']}: {d['state']} ante {d['ante']} | ante changes (step, from, to) {antes} | "
              f"vouchers {d.get('vouchers')} | jokers {[j['key'] for j in d['jokers']]} | money {d['money']} | boss {d['boss']}")


if __name__ == "__main__":
    main()
