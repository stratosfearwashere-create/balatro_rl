"""A live page for the cloud azc run: fetches the cloud machine's log over ssh every minute and rewrites a self-refreshing HTML page,
with the local azc_v5 history (before the move) in front of it.

    python tools/cloud/live.py --host ssh2.vast.ai --port 34974 --out traces/azc_cloud.html

Open the page in a browser; it reloads itself.
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="__REFRESH__">
<title>azc cloud run</title>
<style>
:root{--bg:#fafaf7;--ink:#1d1d1b;--muted:#6b6b66;--grid:#e4e3dc;--a:#2a6fb0;--b:#d9822b;--c:#3c9a5f;--d:#8a5cb8;--card:#fff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--ink:#ecebe6;--muted:#9a9990;--grid:#2c2d30;--a:#6aa8e6;--b:#f0a95a;--c:#5cc384;--d:#b08ae0;--card:#1f2023}}
:root[data-theme="dark"]{--bg:#17181a;--ink:#ecebe6;--muted:#9a9990;--grid:#2c2d30;--a:#6aa8e6;--b:#f0a95a;--c:#5cc384;--d:#b08ae0;--card:#1f2023}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 18px;font-size:13px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:22px}
.stat{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px 12px}
.stat b{display:block;font-size:22px}.stat span{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
@media (max-width:480px){.grid{grid-template-columns:1fr}}
.chart{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px 12px 6px;position:relative}
.chart h2{font-size:14px;margin:0 0 6px}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-bottom:4px}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}
svg{width:100%;height:auto;display:block}text{fill:var(--muted);font-size:11px}
.tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--grid);border-radius:6px;padding:4px 8px;font-size:12px;display:none;white-space:nowrap}
</style></head><body><main>
<h1>azc: AlphaZero on Gold stake</h1>
<p class="sub">__STAGES__ Refreshes every __REFRESH__ s. Updated __UPDATED__.__ERR__</p>
<div class="stats" id="stats"></div>
<div class="grid">
<div class="chart"><h2>Blinds beaten per game (of 24)</h2><div class="legend"><span><i style="background:var(--a)"></i>training games (5-iteration mean)</span><span><i style="background:var(--b)"></i>evaluation, with search</span><span><i style="background:var(--c)"></i>evaluation, network alone</span></div><svg id="c1" viewBox="0 0 460 240"></svg><div class="tip"></div></div>
<div class="chart"><h2>Gold wins (training games)</h2><div class="legend"><span><i style="background:var(--c)"></i>wins per 1,000 games from the start (5-iteration window)</span><span><i style="background:var(--b)"></i>wins per 1,000 mid-run starts (ante 5+)</span></div><svg id="c2" viewBox="0 0 460 240"></svg><div class="tip"></div></div>
<div class="chart"><h2>Policy loss</h2><div class="legend"><span><i style="background:var(--d)"></i>cross-entropy to the search's choices</span></div><svg id="c3" viewBox="0 0 460 240"></svg><div class="tip"></div></div>
<div class="chart"><h2>Value loss</h2><div class="legend"><span><i style="background:var(--a)"></i>squared error to the game outcome</span></div><svg id="c4" viewBox="0 0 460 240"></svg><div class="tip"></div></div>
<div class="chart"><h2>Minutes per iteration</h2><div class="legend"><span><i style="background:var(--b)"></i>wall-clock</span></div><svg id="c5" viewBox="0 0 460 240"></svg><div class="tip"></div></div>
<div class="chart"><h2>Beta (share of the exact value in the search)</h2><div class="legend"><span><i style="background:var(--muted)"></i>falls to 0 at iteration 150</span></div><svg id="c6" viewBox="0 0 460 240"></svg><div class="tip"></div></div>
</div>
<script>
const D = __DATA__;
const SW = __SWITCH__, MARKS = __MARKS__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const it = D.map(r => r.iter);
function roll(xs, k, w) {  // mean over the last k entries, weighted by w if given
  return xs.map((_, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - k + 1); j <= i; j++) { if (xs[j] == null) continue; const ww = w ? w[j] : 1; s += xs[j] * ww; n += ww; } return n ? s / n : null; });
}
const games = D.map(r => r.games);
// the log's "min" is minutes since that trainer process started: per iteration is the difference (none across a restart)
const per = D.map((r, i) => i && r.min != null && D[i - 1].min != null && r.min > D[i - 1].min ? r.min - D[i - 1].min : null);
const tb = roll(D.map(r => r.train_blinds), 5, games);
const wr = D.map((_, i) => { let w = 0, g = 0; for (let j = Math.max(0, i - 4); j <= i; j++) { w += D[j].train_wins; g += D[j].games; } return 1000 * w / g; });
const swr = D.map((_, i) => { let w = 0, g = 0; for (let j = Math.max(0, i - 4); j <= i; j++) { w += D[j].snap_wins; g += D[j].snap_games; } return g ? 1000 * w / g : null; });
function chart(id, series, opts = {}) {
  const svg = document.getElementById(id), W = 460, H = 240, L = 40, R = 10, T = 10, B = 26;
  let ys = []; series.forEach(s => s.y.forEach(v => { if (v != null) ys.push(v); }));
  if (!ys.length) return;
  let y0 = opts.y0 != null ? opts.y0 : Math.min(...ys), y1 = Math.max(...ys);
  if (y1 - y0 < 1e-9) { y1 += 1; y0 -= opts.y0 != null ? 0 : 1; }
  const pad = (y1 - y0) * 0.06; if (opts.y0 == null) y0 -= pad; y1 += pad;
  const x0 = it[0], x1 = it[it.length - 1] || 1;
  const X = x => L + (W - L - R) * (x - x0) / Math.max(1, x1 - x0), Y = y => T + (H - T - B) * (1 - (y - y0) / (y1 - y0));
  let g = '';
  for (let k = 0; k <= 4; k++) { const v = y0 + (y1 - y0) * k / 4, yy = Y(v); g += `<line x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}" stroke="${css('--grid')}"/><text x="${L - 5}" y="${yy + 4}" text-anchor="end">${opts.fmt ? opts.fmt(v) : v.toFixed(2)}</text>`; }
  const step = Math.max(10, Math.ceil((x1 - x0) / 6 / 10) * 10);
  for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) g += `<text x="${X(x)}" y="${H - 8}" text-anchor="middle">${x}</text>`;
  MARKS.forEach(([m, lab]) => { if (m >= x0 && m < x1) g += `<line x1="${X(m + 0.5)}" x2="${X(m + 0.5)}" y1="${T}" y2="${H - B}" stroke="${css('--muted')}" stroke-dasharray="3 3"/><text x="${X(m + 0.5) + 4}" y="${T + 10}">${lab} →</text>`; });
  series.forEach(s => {
    if (s.dots) { s.y.forEach((v, i) => { if (v != null) g += `<circle cx="${X(it[i])}" cy="${Y(v)}" r="4" fill="${s.c}" stroke="${css('--card')}" stroke-width="2"/>`; }); return; }
    let d = '', pen = false; s.y.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? 'L' : 'M') + X(it[i]).toFixed(1) + ',' + Y(v).toFixed(1); pen = true; });
    g += `<path d="${d}" fill="none" stroke="${s.c}" stroke-width="2" stroke-linejoin="round"/>`;
  });
  g += `<line class="hx" y1="${T}" y2="${H - B}" stroke="${css('--muted')}" style="display:none"/>`;
  svg.innerHTML = g;
  const tip = svg.parentElement.querySelector('.tip'), hx = svg.querySelector('.hx');
  svg.onmousemove = e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
    let best = 0; it.forEach((x, i) => { if (Math.abs(X(x) - px) < Math.abs(X(it[best]) - px)) best = i; });
    hx.setAttribute('x1', X(it[best])); hx.setAttribute('x2', X(it[best])); hx.style.display = '';
    tip.innerHTML = `<b>iteration ${it[best]}</b>` + series.map(s => s.y[best] == null ? '' : `<br>${s.n}: ${opts.fmt ? opts.fmt(s.y[best]) : s.y[best].toFixed(3)}`).join('');
    tip.style.display = 'block'; tip.style.left = Math.min(r.width - 170, (e.clientX - r.left) + 12) + 'px'; tip.style.top = '40px';
  };
  svg.onmouseleave = () => { tip.style.display = 'none'; hx.style.display = 'none'; };
}
function draw() {
  chart('c1', [{ y: tb, c: css('--a'), n: 'training' }, { y: D.map(r => r.eval_search), c: css('--b'), n: 'eval search', dots: true }, { y: D.map(r => r.eval_net), c: css('--c'), n: 'eval network', dots: true }], { fmt: v => v.toFixed(2) });
  chart('c2', [{ y: wr, c: css('--c'), n: 'wins / 1000 from start' }, { y: swr, c: css('--b'), n: 'wins / 1000 mid-run' }], { y0: 0, fmt: v => v.toFixed(1) });
  chart('c3', [{ y: D.map(r => r.policy_loss), c: css('--d'), n: 'policy loss' }]);
  chart('c4', [{ y: D.map(r => r.value_loss), c: css('--a'), n: 'value loss' }], { fmt: v => v.toFixed(4) });
  chart('c5', [{ y: per, c: css('--b'), n: 'minutes' }], { y0: 0, fmt: v => v.toFixed(1) });
  chart('c6', [{ y: D.map(r => r.beta), c: css('--muted'), n: 'beta' }], { y0: 0 });
  const cl = D.filter(r => r.iter > SW), last = D[D.length - 1], ev = [...D].reverse().find(r => r.eval_search != null);
  const cw = cl.reduce((s, r) => s + r.train_wins, 0), cg = cl.reduce((s, r) => s + r.games, 0);
  const cp = per.filter((v, i) => v != null && D[i].iter > SW).slice(-5);
  const mins = cp.length ? cp.reduce((s, v) => s + v, 0) / cp.length : null;
  const tiles = [
    [last.iter, 'latest iteration'],
    [tb[tb.length - 1].toFixed(2), 'training blinds (5-iteration mean)'],
    [ev ? `${ev.eval_search.toFixed(2)} / ${ev.eval_net.toFixed(2)}` : '-', ev ? `evaluation at ${ev.iter}: search / network` : 'evaluation'],
    [`${cw} / ${cg.toLocaleString()}`, 'Gold wins / games on the cloud'],
    [mins ? mins.toFixed(1) + ' min' : '-', 'per iteration (cloud, last 5)'],
    [mins && last.iter < 150 ? Math.round((150 - last.iter) * mins) + ' min' : '-', 'until iteration 150'],
    [__COST__, 'spent on the cloud so far (est.)'],
  ];
  document.getElementById('stats').innerHTML = tiles.map(t => `<div class="stat"><b>${t[0]}</b><span>${t[1]}</span></div>`).join('');
}
draw();
</script></main></body></html>"""


def read(path):
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def norm(r, games):
    return {"iter": r["iter"], "beta": r.get("beta"), "train_blinds": r.get("train_blinds"), "train_wins": r.get("train_wins", 0), "games": r.get("games", games), "snap_games": r.get("snap_games", 0), "snap_wins": r.get("snap_wins", 0), "snap_blinds": r.get("snap_blinds"),
            "policy_loss": r.get("policy_loss"), "value_loss": r.get("value_loss"), "min": r.get("min"),
            "eval_search": r.get("eval_search_blinds"), "eval_net": r.get("eval_net_blinds")}


def backup(a):
    """Copy the cloud checkpoint here, and keep it only if it loads (the trainer may be writing it at that moment)."""
    dl = a.backup + ".dl"
    try:
        r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-P", str(a.port), f"root@{a.host}:{a.remote_ckpt}", dl],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(time.strftime("%H:%M:%S"), "backup: copy failed", r.stderr.strip()[-200:], flush=True)
            return
        import torch
        torch.load(dl, map_location="cpu", weights_only=False)
        os.replace(dl, a.backup)
        print(time.strftime("%H:%M:%S"), "backup: saved", a.backup, flush=True)
    except Exception as e:
        print(time.strftime("%H:%M:%S"), "backup: skipped", repr(e)[:200], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="ssh2.vast.ai")
    p.add_argument("--port", default="34974")
    p.add_argument("--run", default="azc_cloud", help="the cloud run's name (checkpoints/<run>.pt and <run>_log.jsonl on the machine)")
    p.add_argument("--history", default="checkpoints/azc_v5_log.jsonl:118:128:local PC",
                   help="earlier stages in front of the run, comma separated path:last iteration:games per iteration:label (a path on this PC)")
    p.add_argument("--label", default="cloud", help="the run's own label")
    p.add_argument("--cloud-games", type=int, default=1536)
    p.add_argument("--price", type=float, default=1.89, help="$ per hour")
    p.add_argument("--started", default="2026-10-19T22:16:09", help="when the rental started (local time)")
    p.add_argument("--out", default="traces/azc_cloud.html")
    p.add_argument("--cache", default="", help="(default: checkpoints/<run>_log.jsonl)")
    p.add_argument("--every", type=int, default=60)
    p.add_argument("--backup-every", type=int, default=600, help="seconds between copies of the cloud checkpoint to this PC (0: never)")
    p.add_argument("--backup", default="", help="(default: checkpoints/<run>.pt)")
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    a.remote_log = f"balatro_rl/checkpoints/{a.run}_log.jsonl"
    a.remote_ckpt = f"balatro_rl/checkpoints/{a.run}.pt"
    a.cache = a.cache or f"checkpoints/{a.run}_log.jsonl"
    a.backup = a.backup or f"checkpoints/{a.run}.pt"
    stages = []
    for h in a.history.split(","):
        path, last, games, label = h.split(":", 3)
        stages.append((path, int(last), int(games), label))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    last_backup = 0.0
    while True:
        if a.backup_every and time.time() - last_backup >= a.backup_every:
            last_backup = time.time()
            backup(a)
        err = ""
        try:
            out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-p", str(a.port), f"root@{a.host}", f"cat {a.remote_log}"],
                                 capture_output=True, text=True, timeout=60)
            if out.returncode == 0:
                open(a.cache, "w", encoding="utf-8").write(out.stdout)
            else:
                err = " Could not reach the cloud machine; showing the last copy."
        except Exception:
            err = " Could not reach the cloud machine; showing the last copy."
        data, lo, marks, text = [], 0, [], []
        for path, last, games, label in stages:
            data += [norm(r, games) for r in read(path) if lo < r.get("iter", 0) <= last]
            text.append(f"{label}: iterations {lo + 1}-{last}.")
            lo = last
            marks.append(last)
        data += [norm(r, a.cloud_games) for r in read(a.cache) if r.get("iter", 0) > lo]
        labels = [st[3] for st in stages[1:]] + [a.label]
        text.append(f"{a.label}: from iteration {lo + 1}.")
        hours = max(0.0, (datetime.now() - datetime.fromisoformat(a.started)).total_seconds() / 3600)
        page = (PAGE.replace("__DATA__", json.dumps(data)).replace("__SWITCH__", str(lo)).replace("__MARKS__", json.dumps([[m, lab] for m, lab in zip(marks, labels)])).replace("__STAGES__", " ".join(text)).replace("__REFRESH__", str(a.every))
                .replace("__UPDATED__", time.strftime("%H:%M:%S")).replace("__ERR__", err).replace("__COST__", json.dumps(f"${hours * a.price:.2f}")))
        tmp = a.out + ".tmp"
        open(tmp, "w", encoding="utf-8").write(page)
        os.replace(tmp, a.out)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
