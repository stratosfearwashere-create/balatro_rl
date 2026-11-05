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
<p class="sub">Refreshes every __REFRESH__ s. Updated __UPDATED__. AlphaZero-style search (MCTS over levels, a value network that sees the known future) on a list of fixed seeds; every batch is 16 games searched at once.</p>
<div class="stats" id="stats"></div>
<div class="chart"><h2>Seeds won, per pass over the 64-seed list (a pass = 4 batches of 16 games)</h2>
<div class="legend"><span><i style="background:var(--c)"></i>share of the pass's games that found a winning line</span><span><i style="background:var(--b)"></i>seeds solved at least once (of the list)</span></div><svg id="c1" viewBox="0 0 900 230"></svg></div>
<div class="chart"><h2>Blinds beaten per game (the blind a line died at; 24 = won)</h2>
<div class="legend"><span><i style="background:var(--a)"></i>this batch</span><span><i style="background:var(--b)"></i>smoothed over 4 batches</span></div><svg id="c2" viewBox="0 0 900 230"></svg></div>
<div class="chart"><h2>Value network: loss on the replay buffer</h2>
<div class="legend"><span><i style="background:var(--a)"></i>mean squared error (targets in units of 1 / 54)</span></div><svg id="c3" viewBox="0 0 900 200"></svg></div>
<div class="chart"><h2>Time per batch of 16 games (seconds)</h2>
<div class="legend"><span><i style="background:var(--a)"></i>search</span><span><i style="background:var(--b)"></i>training</span></div><svg id="c4" viewBox="0 0 900 200"></svg></div>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const stat = (l, v) => '<div class="stat"><b>' + v + '</b><span>' + l + '</span></div>';
const rows = D.rows, last = rows[rows.length - 1] || {};
const sm = (a, w) => a.map((_, i) => { const s = a.slice(Math.max(0, i - w + 1), i + 1); return s.reduce((x, y) => x + y, 0) / s.length; });
const passes = []; for (let i = 3; i < rows.length; i += 4) { const g = rows.slice(i - 3, i + 1); passes.push({batch: rows[i].batch, win: g.reduce((x, r) => x + r.wins, 0) / 64, blinds: g.reduce((x, r) => x + r.blinds, 0) / 4}); }
const lp = passes[passes.length - 1];
document.getElementById('stats').innerHTML = stat('batches', rows.length) + stat('games', (last.games || 0).toLocaleString()) + stat('minutes', last.min != null ? last.min : 0) +
  stat('games won, total', rows.reduce((x, r) => x + r.wins, 0)) + stat('seeds solved at least once', (last.seeds_solved_ever || 0) + ' / ' + (last.seeds_seen || 0)) +
  stat('last pass: win share', lp ? (100 * lp.win).toFixed(0) + '%' : '-') + stat('last pass: blinds / game', lp ? lp.blinds.toFixed(2) : '-') + stat('value loss', last.loss != null ? last.loss.toExponential(1) : '-');
function chart(id, H, series, fmt, xs, ymin, ymax) {
  const svg = document.getElementById(id), W = 900, L = 52, R = 14, T = 10, B = 26;
  if (!xs.length) { svg.innerHTML = '<text x="20" y="40">waiting for the first batches</text>'; return; }
  const all = series.flatMap(q => q.v.filter(v => v != null));
  let lo = ymin != null ? ymin : Math.min(0, ...all), hi = ymax != null ? ymax : Math.max(1e-12, ...all) * 1.1; if (hi <= lo) hi = lo + 1;
  const x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
  const X = g => L + (W - L - R) * (g - x0) / (x1 - x0), Y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  let out = '';
  for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4, y = Y(v);
    out += '<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+fmt(v)+'</text>'; }
  for (let k = 0; k <= 4; k++) { const g = x0 + (x1 - x0) * k / 4;
    out += '<text x="'+X(g)+'" y="'+(H-6)+'" text-anchor="'+(k==0?'start':k==4?'end':'middle')+'">batch '+Math.round(g)+'</text>'; }
  for (const q of series) {
    let d = '', pen = false;
    q.v.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? 'L' : 'M') + X(q.x ? q.x[i] : xs[i]).toFixed(1) + ' ' + Y(v).toFixed(1) + ' '; pen = true; });
    out += '<path d="'+d+'" fill="none" stroke="'+css(q.color)+'" stroke-width="2" stroke-linejoin="round"/>';
  }
  svg.innerHTML = out;
}
const bx = rows.map(r => r.batch);
chart('c1', 230, [{v: passes.map(p => p.win), x: passes.map(p => p.batch), color: '--c'}, {v: rows.map(r => r.seeds_solved_ever / 64), color: '--b'}], v => (100 * v).toFixed(0) + '%', bx, 0, null);
chart('c2', 230, [{v: rows.map(r => r.blinds), color: '--a'}, {v: sm(rows.map(r => r.blinds), 4), color: '--b'}], v => v.toFixed(1), bx, 0, 25);
chart('c3', 200, [{v: rows.map(r => r.loss), color: '--a'}], v => v.toExponential(0), bx, 0, null);
chart('c4', 200, [{v: rows.map(r => r.secs.search), color: '--a'}, {v: rows.map(r => r.secs.train), color: '--b'}], v => v.toFixed(0), bx, 0, null);
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
    data = {"rows": rows[-2000:]}
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
