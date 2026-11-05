"""The last few updates of a PPO run's log.

    python tools/last_iters.py [log] [n]
"""
import json
import sys

log = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ppo_gold_v34_log.jsonl"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
rows = [json.loads(x) for x in open(log) if x.strip()][-n:]
if rows and "train_blinds" in rows[-1]:                     # an AlphaZero (azc) log
    print("iter  beta  train_blinds  wins  policy_loss  value_loss  eval search / net blinds")
    for r in rows:
        ev = f"{r['eval_search_blinds']} / {r['eval_net_blinds']}" if "eval_search_blinds" in r else ""
        print(f"{r['iter']:4d} {r['beta']:5.2f} {r['train_blinds']:12.2f} {r['train_wins']:5d} {r.get('policy_loss', 0):12.4f} {r.get('value_loss', 0):11.4f}  {ev}")
    raise SystemExit
extra = [k for k in ("blinds", "proj8_log", "vloop_corr") if any(k in r for r in rows)]   # (shown when the run logs them; vloop_corr only on fit updates)
print("iter    games  wins    ent      v      kl  mean_ante" + "".join(f"{k:>11s}" for k in extra))
for r in rows:
    a = r["antes"]
    mean_ante = sum((i + 1) * c for i, c in enumerate(a)) / max(sum(a), 1)
    ex = "".join(f"{r[k]:11.3f}" if isinstance(r.get(k), (int, float)) else f"{'':>11s}" for k in extra)
    print(f"{r['iter']:5d} {r['games']:8d} {r['wins']:5d} {r['ent']:6.3f} {r['v']:6.3f} {r['kl']:7.4f} {mean_ante:9.2f}" + ex)
