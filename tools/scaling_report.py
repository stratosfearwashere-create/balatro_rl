"""Reads traces/scaling_sweep.jsonl (tools/scaling_sweep.py) and writes what each scaling joker is worth if you play for it.

Per joker and starting money: the counter at the start of each ante (median over the seeds, with the 10th and 90th percentile at the end), the growth per round it really got against the engine's
natural-rate table (potential.GROW), and what the counter is worth in score (decades of the exact scorer's expected best play over the counter at its start value).

    python tools/scaling_report.py --in traces/scaling_sweep.jsonl --out traces/scaling_value.html
"""
import argparse
import html
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.potential import GROW                    # noqa: E402

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="20"><title>Scaling jokers</title><style>
:root{--bg:#fafaf7;--ink:#1d1d1b;--muted:#6b6b66;--grid:#e4e3dc;--card:#fff;--c1:#2a6fb0;--good:#3c9a5f;--bad:#b3402f}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--ink:#ecebe6;--muted:#9a9990;--grid:#2c2d30;--card:#202225;--c1:#6aa8e6;--good:#5cc384;--bad:#e5806e}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}main{max-width:1100px;margin:0 auto;padding:18px 16px 40px}
h1{font-size:20px;margin:0 0 2px}h2{font-size:14px;margin:14px 0 8px}.sub{color:var(--muted);margin:0 0 12px;font-size:13px}.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px 14px;margin-bottom:14px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:3px 8px;text-align:right;border-top:1px solid var(--grid);white-space:nowrap}th{color:var(--muted);font-weight:500}td:first-child,th:first-child{text-align:left}
.bar{height:9px;border-radius:3px;background:var(--c1);display:inline-block;vertical-align:middle}.mut{color:var(--muted)}</style></head><body><main>
<h1>What a scaling joker is worth when you play for it</h1><p class="sub">__SUB__</p>__BODY__</main></body></html>"""


def load(path):
    runs = defaultdict(list)
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r["levels"]:
            runs[(r["joker"], r["money"])].append(r["levels"])
    return runs


def stats(joker, lv_list):
    init, rate, lo, hi = GROW[joker]
    final = np.array([lv[-1][2] for lv in lv_list])
    nb = np.array([len(lv) for lv in lv_list])
    at = {}
    for ante in (2, 4, 6, 8):
        v = [next((x[2] for x in lv if x[0] == ante), np.nan) for lv in lv_list]
        d = [next((x[3] for x in lv if x[0] == ante), np.nan) for lv in lv_list]
        at[ante] = (np.nanmedian(v), np.nanmedian(d))
    per_round = (final - init) / np.maximum(nb, 1)
    return {"n": len(lv_list), "final_med": float(np.median(final)), "p10": float(np.percentile(final, 10)), "p90": float(np.percentile(final, 90)),
            "per_round": float(np.median(per_round)), "natural": rate, "real_share": float(np.median(per_round) / rate), "at": at, "dec_final": float(np.median([lv[-1][3] for lv in lv_list])),
            "dec_a4": at[4][1], "init": init}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="traces/scaling_sweep.jsonl")
    p.add_argument("--out", default="traces/scaling_value.html")
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    while True:
        runs = load(a.inp) if os.path.exists(a.inp) else {}
        jokers = sorted({j for j, _ in runs}, key=lambda j: -np.median([lv[-1][3] for m in sorted({m for jj, m in runs if jj == j}) for lv in runs[(j, m)]]) if runs else 0)
        moneys = sorted({m for _, m in runs})
        body, text = "", []
        for m in moneys:
            rows = []
            for j in jokers:
                if (j, m) not in runs:
                    continue
                s = stats(j, runs[(j, m)])
                a_ = s["at"]
                rows.append((j, s["n"], f"{s['init']:g}", f"{a_[2][0]:.2f}", f"{a_[4][0]:.2f}", f"{a_[6][0]:.2f}", f"{s['final_med']:.2f}", f"{s['p10']:.1f} to {s['p90']:.1f}",
                             f"{s['per_round']:.2f}", f"{s['natural']:g}", f"{100 * s['real_share']:.0f}%", f"{a_[4][1]:+.2f}", f"{s['dec_final']:+.2f}", f'<span class="bar" style="width:{max(0, min(160, s["dec_final"] * 120)):.0f}px"></span>'))
                text.append((m,) + rows[-1][:-1])
            head = ["joker", "seeds", "start", "counter ante 2", "ante 4", "ante 6", "end (ante 8)", "10th to 90th pct", "growth / round", "table rate", "share of table", "worth at ante 4 (decades)", "worth at the end", ""]
            body += f'<h2>Starting with ${m}</h2><div class="card"><table><tr>' + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>"
        n_runs = sum(len(v) for v in runs.values())
        sub = f"{n_runs} runs ({len({m for _, m in runs})} money levels). Counter = the joker's own counter (chips, mult or xMult as the joker uses it). 'Worth' = decades of expected best-play score the counter adds over its starting value, by the exact scorer. Updated {time.strftime('%H:%M:%S')}."
        open(a.out + ".tmp", "w", encoding="utf-8").write(PAGE.replace("__SUB__", html.escape(sub)).replace("__BODY__", body))
        os.replace(a.out + ".tmp", a.out)
        if a.once:
            print(f"{n_runs} runs")
            print(f"{'$':>3s} {'joker':14s} {'n':>3s} {'start':>5s} {'a2':>7s} {'a4':>7s} {'a6':>7s} {'end':>7s} {'p10-p90':>14s} {'per rd':>7s} {'table':>6s} {'share':>6s} {'a4 dec':>7s} {'end dec':>8s}")
            for r in text:
                print(f"{r[0]:3d} {r[1]:14s} {r[2]:3d} {r[3]:>5s} {r[4]:>7s} {r[5]:>7s} {r[6]:>7s} {r[7]:>7s} {r[8]:>14s} {r[9]:>7s} {r[10]:>6s} {r[11]:>6s} {r[12]:>7s} {r[13]:>8s}")
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
