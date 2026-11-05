"""A live page for a population of policies trained side by side (train.py --partners): python tools/pop_dashboard.py --members pop0,pop1,pop2 --out traces/run_pop.html"""
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
table{border-collapse:collapse;font-size:13px;width:100%;margin-bottom:18px}th{text-align:left;color:var(--muted);font-weight:500;padding:4px 6px}td{padding:5px 6px;border-top:1px solid var(--grid)}
.chart{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px 12px 6px;margin-bottom:16px}
.chart h2{font-size:14px;margin:0 0 6px}.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-bottom:4px}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}
svg{width:100%;height:auto;display:block}text{fill:var(--muted);font-size:11px}
</style></head><body><main>
<h1>__TITLE__</h1>
<p class="sub">Refreshes every __REFRESH__ s. Updated __UPDATED__. Each member is its own network and value function; control of a game passes between members at blind selects, and each learns only from its own decisions.</p>
<table id="tbl"></table>
<div class="chart"><h2>Blinds beaten per game (training)</h2><div class="legend" id="lg1"></div><svg id="c1" viewBox="0 0 900 250"></svg></div>
<div class="chart"><h2>Greedy evaluation on normal Gold (each member alone)</h2><div class="legend" id="lg2"></div><svg id="c2" viewBox="0 0 900 230"></svg></div>
<div class="chart"><h2>Policy entropy</h2><div class="legend" id="lg3"></div><svg id="c3" viewBox="0 0 900 200"></svg></div>
<div class="chart"><h2>Share of a member's steps it played itself (the rest were played by the others)</h2><div class="legend" id="lg4"></div><svg id="c4" viewBox="0 0 900 180"></svg></div>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = ['--a', '--b', '--c', '--muted'];
const names = Object.keys(D.members);
const rowsOf = m => D.members[m].rows;
const lastOf = m => rowsOf(m)[rowsOf(m).length - 1] || {};
document.getElementById('tbl').innerHTML = '<tr><th>member</th><th>update</th><th>games</th><th>blinds</th><th>best</th><th>greedy eval</th><th>wins</th><th>entropy</th><th>partners loaded</th><th>own share</th><th>steps/s</th></tr>' +
  names.map((m, i) => { const r = lastOf(m), ev = rowsOf(m).filter(x => x.eval).slice(-1)[0];
    return '<tr><td><span style="color:' + css(COL[i]) + '">&#9632;</span> ' + m + '</td><td>' + (r.iter || '-') + '</td><td>' + (r.games || 0).toLocaleString() + '</td><td>' + (r.blinds != null ? r.blinds.toFixed(2) : '-') + '</td><td>' +
      Math.max(0, ...rowsOf(m).map(x => x.blinds || 0)).toFixed(2) + '</td><td>' + (ev ? ev.eval.blinds.toFixed(2) + ' (u' + ev.iter + ')' : '-') + '</td><td>' + (r.wins || 0) + '</td><td>' +
      (r.ent != null ? r.ent.toFixed(2) : '-') + '</td><td>' + (r.pop_partners != null ? r.pop_partners : '-') + '</td><td>' + (r.pop_own_share != null ? (100 * r.pop_own_share).toFixed(0) + '%' : '-') + '</td><td>' + (r.sps || '-') + '</td></tr>'; }).join('');
function chart(id, lg, H, key, fmt, evOnly) {
  const svg = document.getElementById(id), W = 900, L = 48, R = 14, T = 10, B = 26;
  const ser = names.map((m, i) => ({m, i, pts: rowsOf(m).filter(r => evOnly ? r.eval : r[key] != null).map(r => [r.iter, evOnly ? r.eval.blinds : r[key]])}));
  const all = ser.flatMap(s => s.pts.map(p => p[1]));
  if (!all.length) { svg.innerHTML = '<text x="20" y="40">waiting for the first rows</text>'; return; }
  const hi = Math.max(1e-9, ...all) * 1.1, lo = 0, x1 = Math.max(1, ...ser.flatMap(s => s.pts.map(p => p[0])));
  const X = g => L + (W - L - R) * g / x1, Y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  let out = '';
  for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4, y = Y(v); out += '<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+fmt(v)+'</text>'; }
  for (let k = 0; k <= 4; k++) { const g = x1 * k / 4; out += '<text x="'+X(g)+'" y="'+(H-6)+'" text-anchor="'+(k==0?'start':k==4?'end':'middle')+'">update '+Math.round(g)+'</text>'; }
  for (const s of ser) {
    const d = s.pts.map((p, j) => (j ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(' ');
    out += '<path d="'+d+'" fill="none" stroke="'+css(COL[s.i])+'" stroke-width="2" stroke-linejoin="round"/>';
    if (evOnly) s.pts.forEach(p => { out += '<circle cx="'+X(p[0])+'" cy="'+Y(p[1])+'" r="3.5" fill="'+css(COL[s.i])+'"/>'; });
  }
  svg.innerHTML = out;
  document.getElementById(lg).innerHTML = ser.map(s => '<span><i style="background:' + css(COL[s.i]) + '"></i>' + s.m + '</span>').join('');
}
chart('c1', 'lg1', 250, 'blinds', v => v.toFixed(1), false);
chart('c2', 'lg2', 230, null, v => v.toFixed(1), true);
chart('c3', 'lg3', 200, 'ent', v => v.toFixed(1), false);
chart('c4', 'lg4', 180, 'pop_own_share', v => (100 * v).toFixed(0) + '%', false);
</script></main></body></html>
"""

KEYS = ("iter", "games", "blinds", "wins", "ent", "sps", "eval", "pop_partners", "pop_own_share")


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
                    rows.append({k: r.get(k) for k in KEYS})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--members", default="pop0,pop1,pop2")
    p.add_argument("--dir", default="checkpoints")
    p.add_argument("--out", default="traces/run_pop.html")
    p.add_argument("--title", default="Population of v7 policies")
    p.add_argument("--every", type=int, default=10)
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    names = a.members.split(",")
    while True:
        data = {"members": {m: {"rows": read(os.path.join(a.dir, m + "_log.jsonl"))} for m in names}}
        html = PAGE.replace("__DATA__", json.dumps(data)).replace("__REFRESH__", str(a.every)).replace("__TITLE__", a.title).replace("__UPDATED__", time.strftime("%H:%M:%S"))
        tmp = a.out + ".tmp"
        open(tmp, "w", encoding="utf-8").write(html)
        os.replace(tmp, a.out)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
