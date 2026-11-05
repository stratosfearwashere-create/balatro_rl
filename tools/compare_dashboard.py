"""A live page comparing PPO runs by update: blinds per game in training (window mean), the fixed greedy evaluation, wins, and the potential.

    python tools/compare_dashboard.py --runs "A=checkpoints/shopA_log.jsonl,B=checkpoints/shopB_log.jsonl,C=checkpoints/shopC_log.jsonl" --out traces/run_compare.html

Rewrites a self-refreshing HTML file every few seconds.
"""
import argparse
import json
import os
import time

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="__REFRESH__">
<title>__TITLE__</title>
<style>
:root{--bg:#fafaf7;--ink:#1d1d1b;--muted:#6b6b66;--grid:#e4e3dc;--card:#fff;--c0:#8a8a85;--c1:#2a6fb0;--c2:#d9822b;--c3:#3c9a5f;--c4:#b0457a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--ink:#ecebe6;--muted:#9a9990;--grid:#2c2d30;--card:#202225;--c0:#a5a59f;--c1:#6aa8e6;--c2:#f0a95a;--c3:#5cc384;--c4:#e07fb0}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 16px;font-size:13px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:18px}
.stat{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px 12px}
.stat b{font-size:20px}.stat span{display:block;color:var(--muted);font-size:12px}
.chart{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px 12px 6px;margin-bottom:16px}
.chart h2{font-size:14px;margin:0 0 6px}.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-bottom:4px}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}
svg{width:100%;height:auto;display:block}text{fill:var(--muted);font-size:11px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:4px 8px;text-align:right;border-top:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}
</style></head><body><main>
<h1>__TITLE__</h1>
<p class="sub">Refreshes every __REFRESH__ s. Updated __UPDATED__. Same game for every run: Gold stake, the scripted card player, the shop network decides outside rounds.</p>
<div class="stats" id="stats"></div>
<div class="chart"><h2>Blinds beaten per game in training (mean of the log's window), by update</h2><div class="legend" id="lg1"></div><svg id="c1" viewBox="0 0 900 290"></svg></div>
<div class="chart"><h2>Greedy evaluation on fixed Gold seeds (100 games), blinds beaten, by update</h2><div class="legend" id="lg2"></div><svg id="c2" viewBox="0 0 900 260"></svg></div>
<div class="chart"><h2>Runs</h2><table id="tbl"></table></div>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = ['--c0', '--c1', '--c2', '--c3', '--c4'];
document.getElementById('stats').innerHTML = D.runs.map((r, i) => '<div class="stat"><b style="color:' + css(COL[i % 5]) + '">' + r.name + '</b><span>update ' + r.iter + ' · ' + r.games.toLocaleString() + ' games · wins ' + r.wins + '</span><span>training blinds ' + r.blinds.toFixed(2) + (r.eval_last != null ? ' · last eval ' + r.eval_last.toFixed(2) : '') + '</span></div>').join('');
for (const id of ['lg1', 'lg2']) document.getElementById(id).innerHTML = D.runs.map((r, i) => '<span><i style="background:' + css(COL[i % 5]) + '"></i>' + r.name + ' — ' + r.label + '</span>').join('');
function chart(id, H, key, ymin, ymax, dots) {
  const svg = document.getElementById(id), W = 900, L = 44, R = 70, T = 10, B = 28;
  const runs = D.runs.map(r => ({...r, pts: r[key]}));
  const xs = runs.flatMap(r => r.pts.map(p => p[0]));
  if (!xs.length) { svg.innerHTML = '<text x="20" y="30">waiting for rows</text>'; return; }
  const x1 = Math.max(...xs, 1), ys = runs.flatMap(r => r.pts.map(p => p[1]));
  const lo = ymin != null ? ymin : 0, hi = ymax != null ? ymax : Math.max(1, ...ys) * 1.1;
  const X = x => L + (W - L - R) * x / x1, Y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  let out = '';
  for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4, y = Y(v);
    out += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + y + '" y2="' + y + '" stroke="' + css('--grid') + '"/><text x="' + (L - 6) + '" y="' + (y + 4) + '" text-anchor="end">' + v.toFixed(1) + '</text>'; }
  for (let k = 0; k <= 4; k++) { const x = x1 * k / 4; out += '<text x="' + X(x) + '" y="' + (H - 8) + '" text-anchor="' + (k == 0 ? 'start' : k == 4 ? 'end' : 'middle') + '">' + D.xname + ' ' + Math.round(x) + '</text>'; }
  runs.forEach((r, i) => {
    if (!r.pts.length) return;
    const c = css(COL[i % 5]);
    if (!dots) out += '<path d="' + r.pts.map((p, j) => (j ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(' ') + '" fill="none" stroke="' + c + '" stroke-width="2.2" stroke-linejoin="round"/>';
    else { out += '<path d="' + r.pts.map((p, j) => (j ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(' ') + '" fill="none" stroke="' + c + '" stroke-width="1.4" stroke-dasharray="4 3"/>';
      out += r.pts.map(p => '<circle cx="' + X(p[0]) + '" cy="' + Y(p[1]) + '" r="3.5" fill="' + c + '"/>').join(''); }
    const last = r.pts[r.pts.length - 1];
    out += '<text x="' + (X(last[0]) + 6) + '" y="' + (Y(last[1]) + 4) + '" style="fill:' + c + '">' + r.name + ' ' + last[1].toFixed(2) + '</text>';
  });
  svg.innerHTML = out;
}
chart('c1', 290, 'train', 0, null, false);
chart('c2', 260, 'evals', 0, null, true);
document.getElementById('tbl').innerHTML = '<tr><th>run</th><th>update</th><th>games</th><th>training blinds</th><th>last eval</th><th>best eval</th><th>wins</th><th>manager entropy</th><th>potential</th></tr>' +
  D.runs.map(r => '<tr><td>' + r.name + '</td><td>' + r.iter + '</td><td>' + r.games.toLocaleString() + '</td><td>' + r.blinds.toFixed(2) + '</td><td>' + (r.eval_last != null ? r.eval_last.toFixed(2) : '-') + '</td><td>' + (r.best != null ? r.best.toFixed(2) : '-') + '</td><td>' + r.wins + '</td><td>' + (r.ent != null ? r.ent.toFixed(2) : '-') + '</td><td>' + (r.phi != null ? r.phi.toFixed(1) : '-') + '</td></tr>').join('');
</script></main></body></html>
"""


def read(path):
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "iter" in r and "games" in r:
                    rows.append(r)
    return rows


def build(spec, labels, xkey="iter"):
    runs = []
    for name, path in spec:
        rows = read(path)
        if not rows:
            continue
        step = max(1, len(rows) // 600)
        keep = list(dict.fromkeys(list(range(0, len(rows), step)) + [len(rows) - 1]))
        evs = [(r[xkey], r["eval"]["blinds"]) for r in rows if isinstance(r.get("eval"), dict)]
        last = rows[-1]
        runs.append({"name": name, "label": labels.get(name, ""), "iter": last["iter"], "games": last["games"], "wins": last.get("wins", 0), "blinds": last.get("blinds", 0.0),
                     "train": [(rows[k][xkey], rows[k]["blinds"]) for k in keep if rows[k].get("blinds") is not None], "evals": evs,
                     "eval_last": evs[-1][1] if evs else None, "best": max(e[1] for e in evs) if evs else None,
                     "ent": (last.get("manager") or {}).get("ent"), "phi": last.get("phi_mean")})
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help='"A=path,B=path,..."')
    p.add_argument("--labels", default="A=blinds + win,B=blinds + win + judge potential,C=judge potential + blinds x0.1")
    p.add_argument("--out", default="traces/run_compare.html")
    p.add_argument("--title", default="Shop network: baseline vs judge potential")
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--x", default="iter", choices=["iter", "games"], help="the x axis: the update number or the games played")
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    spec = [tuple(x.split("=", 1)) for x in a.runs.split(",")]
    labels = dict(x.split("=", 1) for x in a.labels.split(",") if "=" in x)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    while True:
        html = (PAGE.replace("__DATA__", json.dumps({"runs": build(spec, labels, a.x), "xname": "update" if a.x == "iter" else "games"})).replace("__REFRESH__", str(a.every)).replace("__TITLE__", a.title)
                .replace("__UPDATED__", time.strftime("%H:%M:%S")))
        tmp = a.out + ".tmp"
        open(tmp, "w", encoding="utf-8").write(html)
        os.replace(tmp, a.out)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
