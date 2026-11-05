"""A live page for balatro_rl.shopq: the labelling progress, the Q head's held-out fit and the improved policy's blinds against the base network's, round by round.

    python tools/shopq_dashboard.py --log checkpoints/shopq1.out --out traces/shopq1.html
"""
import argparse
import html
import json
import os
import re
import time

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5"><title>Shop policy improvement</title><style>
:root{--bg:#fafaf7;--ink:#1d1d1b;--muted:#6b6b66;--grid:#e4e3dc;--card:#fff;--c1:#2a6fb0;--c2:#d9822b;--c3:#3c9a5f;--bad:#b3402f}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--ink:#ecebe6;--muted:#9a9990;--grid:#2c2d30;--card:#202225;--c1:#6aa8e6;--c2:#f0a95a;--c3:#5cc384;--bad:#e5806e}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}main{max-width:980px;margin:0 auto;padding:18px 16px 40px}
h1{font-size:20px;margin:0 0 2px}h2{font-size:14px;margin:0 0 8px}.sub{color:var(--muted);margin:0 0 14px;font-size:13px}
.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:14px}.stat{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px 12px}
.stat b{font-size:20px}.stat span{display:block;color:var(--muted);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:4px 8px;text-align:right;border-top:1px solid var(--grid)}th{color:var(--muted);font-weight:500}td:first-child,th:first-child{text-align:left}
.bar{height:10px;border-radius:3px;background:var(--c1);display:inline-block;vertical-align:middle}.good{color:var(--c3);font-weight:600}.bad{color:var(--bad)}
svg{width:100%;height:auto}text{fill:var(--muted);font-size:11px}</style></head><body><main>
<h1>Shop policy improved from rollouts</h1><p class="sub">__LOG__ . Updated __NOW__ (refreshes every 5 s). __STAGE__</p>__BODY__</main></body></html>"""


def parse(path):
    d = {"base": None, "label": None, "rounds": {}, "cur": 0, "epochs": [], "stage": "starting", "done": False}
    if not os.path.exists(path):
        return d
    for line in open(path, errors="replace"):
        line = line.strip()
        m = re.match(r"round 0 .*blinds ([\d.]+), wins (\d+)/(\d+)", line)
        if m:
            d["base"] = (float(m[1]), int(m[2]), int(m[3])); d["stage"] = "labelling shop states (round 1)"; d["cur"] = 1
        m = re.match(r"(\d+)/(\d+) labelled states, (\d+)s", line)
        if m:
            d["label"] = (int(m[1]), int(m[2]), int(m[3])); d["stage"] = f"labelling shop states (round {d['cur']}): {m[1]} of {m[2]}"
        m = re.match(r"round (\d+): (\d+) states labelled in (\d+)s", line)
        if m:
            d["rounds"].setdefault(int(m[1]), {})["labelled"] = (int(m[2]), int(m[3])); d["stage"] = f"fitting the Q head (round {m[1]})"; d["epochs"] = []
        m = re.match(r"epoch (\d+): train loss ([\d.]+); held out (\{.*\})", line)
        if m:
            d["epochs"].append((int(m[1]), float(m[2]), json.loads(m[3])))
        m = re.match(r"round (\d+): held-out (\{.*\})", line)
        if m:
            d["rounds"].setdefault(int(m[1]), {})["held"] = json.loads(m[2]); d["pending"] = int(m[1])
        m = re.match(r"new policy: blinds ([\d.]+) \(base ([\d.]+); paired difference ([+-][\d.]+) \+- ([\d.]+)\), wins (\d+)/(\d+), Q overrode the network at (\d+)/(\d+)", line)
        if m:
            r = d.get("pending", d["cur"])
            d["rounds"].setdefault(r, {})["new"] = (float(m[1]), float(m[3]), float(m[4]), int(m[5]), int(m[7]), int(m[8]))
            d["cur"] = r + 1; d["stage"] = f"labelling shop states (round {r + 1})"
    return d


def render(d, log):
    body = ""
    base = d["base"]
    stats = [("base network", f"{base[0]:.2f} blinds" if base else "-", f"{base[1]}/{base[2]} wins on exact-seed games" if base else "")]
    last = max((r for r in d["rounds"] if "new" in d["rounds"][r]), default=None)
    if last:
        n = d["rounds"][last]["new"]
        stats.append((f"policy after round {last}", f"{n[0]:.2f} blinds", f"{n[1]:+.2f} +- {n[2]:.2f} vs the base (paired), {n[3]} wins"))
        stats.append(("Q overrides the network", f"{n[4]}/{n[5]}", "shop decisions where the Q head's best action was taken"))
    if d["label"]:
        stats.append(("labelling", f"{d['label'][0]}/{d['label'][1]}", f"{d['label'][2]} s; paired rollouts per state"))
    body += '<div class="stats">' + "".join(f'<div class="stat"><b>{html.escape(a)}</b><span>{html.escape(b)}</span><span>{html.escape(c)}</span></div>' for a, b, c in stats[:4]) + "</div>"
    if d["label"] and not any("labelled" in d["rounds"].get(d["cur"], {}) for _ in [0]):
        w = 100.0 * d["label"][0] / max(1, d["label"][1])
        body += f'<div class="card"><h2>Labelling round {d["cur"]}</h2><span class="bar" style="width:{w * 4:.0f}px"></span> {d["label"][0]} of {d["label"][1]} states</div>'
    rows = []
    for r in sorted(d["rounds"]):
        x = d["rounds"][r]
        h, nw = x.get("held", {}), x.get("new")
        rows.append((r, x["labelled"][0] if "labelled" in x else "-", h.get("corr", "-"), h.get("sign_acc", "-"), h.get("gain_if_follow_q", "-"), h.get("gain_of_best_possible", "-"),
                     f"{nw[0]:.2f}" if nw else "-", (f'<span class="{"good" if nw[1] > 2 * nw[2] else "bad" if nw[1] < -2 * nw[2] else ""}">{nw[1]:+.2f} +- {nw[2]:.2f}</span>') if nw else "-", f"{nw[4]}/{nw[5]}" if nw else "-"))
    if rows:
        head = ["round", "states", "held-out corr", "sign acc", "gain if Q followed", "best possible gain", "policy blinds", "vs base (paired)", "Q overrides"]
        body += '<div class="card"><h2>Rounds</h2><table><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>"
    if d["epochs"]:
        body += '<div class="card"><h2>Q-head fit (this round)</h2><table><tr><th>epoch</th><th>train loss</th><th>held-out corr</th><th>sign acc</th><th>gain if Q followed</th></tr>' + "".join(
            f"<tr><td>{e}</td><td>{l:.3f}</td><td>{h['corr']}</td><td>{h['sign_acc']}</td><td>{h['gain_if_follow_q']}</td></tr>" for e, l, h in d["epochs"][-8:]) + "</table></div>"
    return PAGE.replace("__LOG__", html.escape(log)).replace("__NOW__", time.strftime("%H:%M:%S")).replace("__STAGE__", html.escape(d["stage"])).replace("__BODY__", body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="checkpoints/shopq1.out")
    p.add_argument("--out", default="traces/shopq1.html")
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    while True:
        open(a.out + ".tmp", "w", encoding="utf-8").write(render(parse(a.log), a.log))
        os.replace(a.out + ".tmp", a.out)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
