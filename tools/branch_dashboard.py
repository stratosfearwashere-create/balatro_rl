"""A live page for tools/branch_train.py: python tools/branch_dashboard.py --log checkpoints/branch1_log.jsonl --out traces/run_branch1.html

Headroom (how much better the best alternative is than the policy's own choice), how often they differ, the returns, the evaluation against the starting policy's, the training losses.
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
:root{--bg:#fafaf7;--ink:#1d1d1b;--muted:#6b6b66;--grid:#e4e3dc;--a:#2a6fb0;--b:#d9822b;--c:#3c9a5f;--card:#fff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--ink:#ecebe6;--muted:#9a9990;--grid:#2c2d30;--a:#6aa8e6;--b:#f0a95a;--c:#5cc384;--card:#1f2023}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 18px;font-size:13px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:22px}
.stat{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px 12px}
.stat b{display:block;font-size:22px}.stat span{color:var(--muted);font-size:12px}
.chart{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px 12px 6px;margin-bottom:16px}
.chart h2{font-size:14px;margin:0 0 6px}.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-bottom:4px}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}
svg{width:100%;height:auto;display:block}text{fill:var(--muted);font-size:11px}
</style></head><body><main>
<h1>__TITLE__</h1>
<p class="sub">Refreshes every __REFRESH__ s. Updated __UPDATED__. Each iteration: roots from the policy's own games, each alternative played out to the end under shared futures, the policy trained toward the better ones.</p>
<div class="stats" id="stats"></div>
<div class="chart"><h2>Headroom: how much better the best alternative is than the policy's own choice (return units)</h2>
<div class="legend"><span><i style="background:var(--a)"></i>headroom (smoothed over 5 iterations)</span></div><svg id="c1" viewBox="0 0 900 230"></svg></div>
<div class="chart"><h2>Share of roots where the best alternative is not the policy's greedy choice</h2>
<div class="legend"><span><i style="background:var(--b)"></i>differ</span></div><svg id="c2" viewBox="0 0 900 200"></svg></div>
<div class="chart"><h2>Headroom by the ante of the root (last 10 iterations): where a better choice exists</h2><div id="byante"></div></div>
<div class="chart"><h2>Mean return of a game from a root</h2>
<div class="legend"><span><i style="background:var(--a)"></i>under the policy's greedy choice</span><span><i style="background:var(--c)"></i>under the best alternative</span></div><svg id="c3" viewBox="0 0 900 230"></svg></div>
<div class="chart"><h2>Greedy evaluation on normal Gold (blinds beaten)</h2>
<div class="legend"><span><i style="background:var(--c)"></i>this run</span><span><i style="background:var(--b)"></i>the starting policy</span></div><svg id="c4" viewBox="0 0 900 230"></svg></div>
<div class="chart"><h2>Training: the loss among the alternatives, and the KL to the starting policy</h2>
<div class="legend"><span><i style="background:var(--a)"></i>alternative loss</span><span><i style="background:var(--b)"></i>KL (x100)</span></div><svg id="c5" viewBox="0 0 900 200"></svg></div>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const stat = (l, v) => '<div class="stat"><b>' + v + '</b><span>' + l + '</span></div>';
const last = D.rows[D.rows.length - 1] || {};
const ev = D.rows.filter(r => r.eval);
document.getElementById('stats').innerHTML = stat('iterations', D.rows.length) + stat('minutes', last.min != null ? last.min : 0) + stat('labelled roots (all)', (D.rows.length * (last.roots || 0)).toLocaleString()) +
  stat('headroom, last 10', D.head10 == null ? '-' : D.head10.toFixed(3)) + stat('best differs, last 10', D.diff10 == null ? '-' : (100 * D.diff10).toFixed(0) + '%') +
  stat('start eval (blinds)', D.base == null ? '-' : D.base.toFixed(2)) + stat('latest eval (blinds)', ev.length ? ev[ev.length - 1].eval.blinds.toFixed(2) : '-') +
  stat('resets', D.rows.filter(r => r.reset).length);
function chart(id, H, series, fmt, xs, points, hline) {
  const svg = document.getElementById(id), W = 900, L = 48, R = 14, T = 10, B = 26;
  if (!xs.length) { svg.innerHTML = '<text x="20" y="40">waiting for the first iterations</text>'; return; }
  const all = series.flatMap(q => q.v.filter(v => v != null)).concat((points || []).flatMap(p => p.v.filter(v => v != null))).concat(hline != null ? [hline] : []);
  let lo = Math.min(0, ...all), hi = Math.max(1e-9, ...all); if (hi === lo) hi = lo + 1; hi += (hi - lo) * 0.08;
  const x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
  const X = g => L + (W - L - R) * (g - x0) / (x1 - x0), Y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  let out = '';
  for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4, y = Y(v);
    out += '<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+fmt(v)+'</text>'; }
  for (let k = 0; k <= 4; k++) { const g = x0 + (x1 - x0) * k / 4;
    out += '<text x="'+X(g)+'" y="'+(H-6)+'" text-anchor="'+(k==0?'start':k==4?'end':'middle')+'">'+Math.round(g)+'</text>'; }
  if (hline != null) out += '<line x1="'+L+'" x2="'+(W-R)+'" y1="'+Y(hline)+'" y2="'+Y(hline)+'" stroke="'+css('--b')+'" stroke-dasharray="5 4" stroke-width="2"/>';
  for (const q of series) {
    let d = '', pen = false;
    q.v.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + Y(v).toFixed(1) + ' '; pen = true; });
    out += '<path d="'+d+'" fill="none" stroke="'+css(q.color)+'" stroke-width="2" stroke-linejoin="round"/>';
  }
  for (const p of (points || [])) p.v.forEach((v, i) => { if (v != null) out += '<circle cx="'+X(xs[i])+'" cy="'+Y(v)+'" r="3.5" fill="'+css(p.color)+'"/>'; });
  svg.innerHTML = out;
}
(function () {
  const rows = D.rows.slice(-10).filter(r => r.headroom_by_ante); if (!rows.length) { document.getElementById('byante').innerHTML = '<span style="color:var(--muted);font-size:12px">not logged by this version of the run</span>'; return; }
  const tab = a => { const xs = rows.map(r => r[a]).filter(x => x); const out = []; for (let k = 1; k <= 8; k++) { const v = xs.map(x => x[k]).filter(y => y != null); out.push(v.length ? v.reduce((p, q) => p + q, 0) / v.length : null); } return out; };
  const h = tab('headroom_by_ante'), q = tab('q_by_ante');
  document.getElementById('byante').innerHTML = '<table style="border-collapse:collapse;font-size:12px;width:100%"><tr style="color:var(--muted)"><th style="text-align:left">ante of the root</th>' + h.map((_, i) => '<th>' + (i + 1) + '</th>').join('') + '</tr>' +
    '<tr><td>headroom</td>' + h.map(v => '<td style="text-align:center">' + (v == null ? '-' : v.toFixed(3)) + '</td>').join('') + '</tr>' +
    '<tr><td>mean return (greedy choice)</td>' + q.map(v => '<td style="text-align:center">' + (v == null ? '-' : v.toFixed(2)) + '</td>').join('') + '</tr></table>';
})();
const it = D.rows.map(r => r.iter);
const smooth = (a, w) => a.map((_, i) => { const s = a.slice(Math.max(0, i - w + 1), i + 1); return s.reduce((x, y) => x + y, 0) / s.length; });
chart('c1', 230, [{v: smooth(D.rows.map(r => r.headroom), 5), color: '--a'}], v => v.toFixed(3), it);
chart('c2', 200, [{v: smooth(D.rows.map(r => r.differ), 5), color: '--b'}], v => (100 * v).toFixed(0) + '%', it);
chart('c3', 230, [{v: D.rows.map(r => r.q_greedy), color: '--a'}, {v: D.rows.map(r => r.q_best), color: '--c'}], v => v.toFixed(2), it);
chart('c4', 230, [], v => v.toFixed(1), it, [{v: D.rows.map(r => r.eval ? r.eval.blinds : null), color: '--c'}], D.base);
chart('c5', 200, [{v: D.rows.map(r => r.alt_loss), color: '--a'}, {v: D.rows.map(r => r.kl == null ? null : 100 * r.kl), color: '--b'}], v => v.toFixed(2), it);
</script></main></body></html>
"""


def read(path):
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def base_eval(out_path):
    if out_path and os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8", errors="replace"):
            if '"initial_eval"' in line:
                try:
                    return json.loads(line)["initial_eval"]["blinds"]
                except (ValueError, KeyError):
                    pass
    return None


def write(a):
    rows = read(a.log)
    tail = rows[-10:]
    data = {"rows": rows[-1500:], "base": base_eval(a.events), "head10": sum(r["headroom"] for r in tail) / len(tail) if tail else None,
            "diff10": sum(r["differ"] for r in tail) / len(tail) if tail else None}
    html = (PAGE.replace("__DATA__", json.dumps(data)).replace("__REFRESH__", str(a.every)).replace("__TITLE__", a.title).replace("__UPDATED__", time.strftime("%H:%M:%S")))
    tmp = a.out + ".tmp"
    open(tmp, "w", encoding="utf-8").write(html)
    os.replace(tmp, a.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out", default="traces/run_branch.html")
    p.add_argument("--title", default="Branching rollouts")
    p.add_argument("--events", default="", help="the trainer's output file (it prints the starting evaluation there)")
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    while True:
        write(a)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
