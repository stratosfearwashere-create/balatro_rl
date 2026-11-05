"""How much strength can a seed's shops support when money is not the constraint? Extra starting money (decreasing), the plain strength ranking, beam 32.

    python tools/solve_money.py --stake GOLD --seeds 40001-40008
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import solve_fast as S

p = argparse.ArgumentParser()
p.add_argument("--stake", default="GOLD")
p.add_argument("--seeds", default="40001-40008")
p.add_argument("--money", default="100,50,25,0")
p.add_argument("--beam", type=int, default=32)
a = p.parse_args()
lo, hi = [int(x) for x in a.seeds.split("-")]
os.makedirs("traces/money", exist_ok=True)
levels = [int(x) for x in a.money.split(",")]
tab = {m: [] for m in levels}
for sd in range(lo, hi + 1):
    row = []
    for m in levels:
        S.START_MONEY = m
        best, won = 0, False
        for mw in (0.0, 0.01):
            path, info = S.solve("RED", a.stake, sd, a.beam, 8, 4, 4, 1000, mw, 0.0)
            if path:
                won = True
                json.dump({"seed": sd, "deck": "RED", "stake": a.stake, "start_money": m, "solved": True, "actions": path},
                          open(f"traces/money/{a.stake}_{sd}_m{m}.json", "w"))
                break
            best = max(best, info["level"])
        tab[m].append("WIN" if won else best)
        row.append(f"+${m}: {'WIN' if won else best}")
    print(f"seed {sd}: " + "   ".join(row), flush=True)
print()
for m in levels:
    w = sum(1 for x in tab[m] if x == "WIN")
    d = [x for x in tab[m] if x != "WIN"]
    print(f"+${m:>3}: {w}/{len(tab[m])} won; mean death blind of the rest {sum(d) / max(1, len(d)):.1f}", flush=True)
