"""A live page for a teacher-generation run: reads the solver's log every few seconds and rewrites a self-refreshing HTML file.

    python tools/gold_teacher_dashboard.py --log traces/gold_teacher.log --total 2500 --out traces/gold_teacher.html
"""
import argparse
import json
import os
import re
import time

LINE = re.compile(r"seed (\d+): (SOLVED|died at blind\s+(\d+))\s+(\d+)s")

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="__REFRESH__">
<title>Gold teacher run</title>
<style>
:root{--bg:#fafaf7;--ink:#1d1d1b;--muted:#6b6b66;--grid:#e4e3dc;--a:#2a6fb0;--b:#d9822b;--card:#fff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--ink:#ecebe6;--muted:#9a9990;--grid:#2c2d30;--a:#6aa8e6;--b:#f0a95a;--card:#1f2023}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 18px;font-size:13px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:22px}
.stat{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px 12px}
.stat b{display:block;font-size:22px}.stat span{color:var(--muted);font-size:12px}
.chart{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px 12px 6px;margin-bottom:16px}
.chart h2{font-size:14px;margin:0 0 6px}.legend{display:flex;gap:16px;color:var(--muted);font-size:12px;margin-bottom:4px}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}
svg{width:100%;height:auto;display:block}text{fill:var(--muted);font-size:11px}
</style></head><body><main>
<h1>Gold teacher run</h1>
<p class="sub">Beam 128, scaling-aware ranking. Refreshes every __REFRESH__ s. Updated __UPDATED__.</p>
<div class="stats" id="stats"></div>
<div class="chart"><h2>Average blind reached (a win counts as 24)</h2>
<div class="legend"><span><i style="background:var(--a)"></i>all seeds so far</span><span><i style="background:var(--b)"></i>last 50 seeds</span></div>
<svg id="c1" viewBox="0 0 900 260"></svg></div>
<div class="chart"><h2>Win rate (%)</h2>
<div class="legend"><span><i style="background:var(--a)"></i>all seeds so far</span><span><i style="background:var(--b)"></i>last 50 seeds</span></div>
<svg id="c2" viewBox="0 0 900 260"></svg></div>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function stat(label, value){ return '<div class="stat"><b>'+value+'</b><span>'+label+'</span></div>'; }
const s = D.summary;
document.getElementById('stats').innerHTML =
  stat('seeds searched', s.n + ' / ' + s.total) + stat('winning lines', s.wins) + stat('win rate', s.win_rate.toFixed(1) + '%') +
  stat('average blind', s.avg.toFixed(2)) + stat('seconds per seed', s.secs.toFixed(1)) + stat('wins per hour', s.wins_per_hour.toFixed(0)) +
  stat('time left (est.)', s.eta);
function chart(id, a, b, fmt, ymax){
  const svg = document.getElementById(id), W = 900, H = 260, L = 44, R = 14, T = 10, B = 26;
  const n = a.length; if (!n) { svg.innerHTML = '<text x="20" y="40">waiting for the first seeds</text>'; return; }
  const top = ymax || Math.max(1, Math.max(...a, ...b) * 1.15);
  const X = i => L + (W - L - R) * (n === 1 ? 0 : i / (n - 1)), Y = v => T + (H - T - B) * (1 - v / top);
  let out = '';
  for (let k = 0; k <= 4; k++) { const v = top * k / 4, y = Y(v);
    out += '<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+fmt(v)+'</text>'; }
  out += '<text x="'+L+'" y="'+(H-6)+'">seed 1</text><text x="'+(W-R)+'" y="'+(H-6)+'" text-anchor="end">seed '+n+'</text>';
  const path = arr => arr.map((v, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1)).join(' ');
  out += '<path d="'+path(b)+'" fill="none" stroke="'+css('--b')+'" stroke-width="1.8" stroke-linejoin="round"/>';
  out += '<path d="'+path(a)+'" fill="none" stroke="'+css('--a')+'" stroke-width="2.2" stroke-linejoin="round"/>';
  out += '<text x="'+(W-R)+'" y="'+(Y(a[n-1])-8)+'" text-anchor="end" style="fill:'+css('--a')+'">'+fmt(a[n-1])+'</text>';
  svg.innerHTML = out;
}
chart('c1', D.avg_cum, D.avg_roll, v => v.toFixed(1), 24);
chart('c2', D.win_cum, D.win_roll, v => v.toFixed(0) + '%', null);
</script></main></body></html>
"""


def parse(log):
    rows = []
    if not os.path.exists(log):
        return rows
    for line in open(log, encoding="utf-8", errors="replace"):
        m = LINE.search(line)
        if m:
            won = m.group(2) == "SOLVED"
            rows.append((int(m.group(1)), 24 if won else int(m.group(3)), won, int(m.group(4))))
    return rows


def build(rows, total, window=50):
    n = len(rows)
    avg_cum, avg_roll, win_cum, win_roll = [], [], [], []
    for i in range(n):
        lo = max(0, i + 1 - window)
        seg = rows[lo:i + 1]
        allr = rows[:i + 1]
        avg_cum.append(sum(r[1] for r in allr) / len(allr))
        avg_roll.append(sum(r[1] for r in seg) / len(seg))
        win_cum.append(100.0 * sum(r[2] for r in allr) / len(allr))
        win_roll.append(100.0 * sum(r[2] for r in seg) / len(seg))
    wins = sum(r[2] for r in rows)
    secs = sum(r[3] for r in rows) / n if n else 0.0
    left = max(0, total - n) * secs
    eta = "%dh %02dm" % (left // 3600, (left % 3600) // 60) if n else "n/a"
    summary = dict(n=n, total=total, wins=wins, win_rate=100.0 * wins / n if n else 0.0, avg=sum(r[1] for r in rows) / n if n else 0.0,
                   secs=secs, wins_per_hour=(3600.0 / secs) * (wins / n) if n and secs else 0.0, eta=eta)
    return dict(summary=summary, avg_cum=avg_cum, avg_roll=avg_roll, win_cum=win_cum, win_roll=win_roll)


def write(args):
    data = build(parse(args.log), args.total)
    html = (PAGE.replace("__DATA__", json.dumps(data)).replace("__REFRESH__", str(args.every))
            .replace("__UPDATED__", time.strftime("%H:%M:%S")))
    tmp = args.out + ".tmp"
    open(tmp, "w", encoding="utf-8").write(html)
    os.replace(tmp, args.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="traces/gold_teacher.log")
    p.add_argument("--total", type=int, default=2500)
    p.add_argument("--out", default="traces/gold_teacher.html")
    p.add_argument("--every", type=int, default=30)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    while True:
        write(a)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
