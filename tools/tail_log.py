"""The last N updates of a PPO run's log, one line each.

    python tools/tail_log.py checkpoints/ppo_gold_rel1_log.jsonl 15
"""
import json
import sys

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
rows = []
for line in open(path, encoding="utf-8", errors="replace"):
    line = line.strip()
    if line.startswith("{"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "iter" in r and "games" in r:
            rows.append(r)
print(f"{'iter':>5} {'games':>9} {'blinds':>6} {'ante':>5} {'win%':>5} {'wins':>5} {'ret':>6} {'ent':>5} {'v':>6} {'sps':>6}  eval (blinds / wins)")
for r in rows[-n:]:
    ev = r.get("eval")
    print(f"{r['iter']:>5} {r['games']:>9} {r.get('blinds', 0):>6.2f} {r.get('ante', 0):>5.2f} {r.get('win%', 0):>5} {r.get('wins', 0):>5} {r.get('ret', 0):>6.2f} {r.get('ent', 0):>5.2f} "
          f"{r.get('v', 0):>6.3f} {r.get('sps', 0):>6}  " + (f"{ev['blinds']:.2f} / {100 * ev['win_rate']:.1f}%" if isinstance(ev, dict) else ""))
