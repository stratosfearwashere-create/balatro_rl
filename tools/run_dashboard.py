"""A live page for a PPO run: blinds per game (the log's window mean, and the best window so far) and wins, against games played.

    python tools/run_dashboard.py --log checkpoints/ppo_gold_v28_log.jsonl --out traces/run_v28.html --title "v28"

Reads the trainer's log every few seconds and rewrites a self-refreshing HTML file (open it in a browser).
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
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:22px}
.stat{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px 12px}
.stat b{display:block;font-size:22px}.stat span{color:var(--muted);font-size:12px}
.chart{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px 12px 6px;margin-bottom:16px}
.chart h2{font-size:14px;margin:0 0 6px}.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-bottom:4px}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;border-radius:2px}
svg{width:100%;height:auto;display:block}text{fill:var(--muted);font-size:11px}
</style></head><body><main>
<h1>__TITLE__</h1>
<p class="sub">Refreshes every __REFRESH__ s. Updated __UPDATED__.</p>
<div class="stats" id="stats"></div>
<div class="chart" id="revbox" style="display:none"><h2>Reverse curriculum: where the games start</h2>
<div class="sub" id="rev_note"></div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px">
<div><div class="legend"><span><i style="background:var(--a)"></i>earliest ante games start from (the band is [this, 8])</span></div><svg id="r1" viewBox="0 0 440 220"></svg></div>
<div><div class="legend"><span><i style="background:var(--c)"></i>win rate of the games started in that ante</span><span><i style="background:var(--b)"></i>needed to extend (__ADV__%)</span></div><svg id="r2" viewBox="0 0 440 220"></svg></div>
</div>
<h2 style="margin-top:14px">Win rate by start ante (games started from saved positions; last 25 updates, and all updates)</h2>
<div id="rev_table" style="overflow-x:auto"></div>
<h2 style="margin-top:14px">Seeding and band moves</h2>
<div id="rev_events" style="font-size:12px;color:var(--muted)"></div></div>
<div class="chart" id="autobox" style="display:none"><h2>Round solver: how much of the play the network hands over (--autoplay)</h2>
<div class="sub" id="auto_note"></div>
<div class="legend"><span><i style="background:var(--b)"></i>share of blind selects handed to the solver</span><span><i style="background:var(--a)"></i>blinds beaten in training (with the solver)</span><span><i style="background:var(--c)"></i>greedy evaluation WITHOUT the solver (the network plays every hand)</span></div>
<svg id="au1" viewBox="0 0 900 250"></svg></div>
<div class="chart" id="orbox" style="display:none"><h2>Ante-8 oracle and money-option curve (means over each update's shop states; train.py --oracle-*)</h2>
<div class="sub" id="or_note"></div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px">
<div><div class="legend"><span><i style="background:var(--a)"></i>oracle: the build's best share of the ante-8 boss, log scale (0 = 1/10,000 of it, 1 = clears it)</span></div><svg id="or1" viewBox="0 0 440 220"></svg></div>
<div><div class="legend"><span><i style="background:var(--c)"></i>money-option: chance of a strong multiplier within one ante with the money held</span></div><svg id="or2" viewBox="0 0 440 220"></svg></div>
</div></div>
<div class="chart" id="immbox" style="display:none"><h2>Immortal run: how often each blind is beaten</h2>
<div class="sub" id="imm_note"></div>
<div id="imm_table" style="overflow-x:auto"></div>
<div class="legend" style="margin-top:8px"><span><i style="background:var(--a)"></i>share of games that beat the first blind of each ante, over the updates</span></div>
<svg id="m1" viewBox="0 0 900 220"></svg></div>
<div class="chart"><h2>Blinds beaten per game</h2>
<div class="legend"><span><i style="background:var(--a)"></i>recent games (the log's window)</span><span><i style="background:var(--b)"></i>best window so far</span><span><i style="background:var(--c)"></i>fixed greedy evaluation</span></div>
<svg id="c1" viewBox="0 0 900 270"></svg></div>
<div class="chart"><h2>Win rate (%)</h2>
<div class="legend"><span><i style="background:var(--a)"></i>recent games (the log's window)</span><span><i style="background:var(--c)"></i>fixed greedy evaluation</span></div>
<svg id="c5" viewBox="0 0 900 250"></svg></div>
<div class="chart"><h2>Wins (total)</h2>
<div class="legend"><span><i style="background:var(--a)"></i>wins so far</span></div>
<svg id="c2" viewBox="0 0 900 230"></svg></div>
<div class="chart"><h2>Shaping weight per ante (1 = full, 0 = faded out)</h2>
<div class="legend" id="lg3"></div>
<svg id="c3" viewBox="0 0 900 250"></svg></div>
<div class="chart"><h2>Reward on offer now, per ante, against the win reward</h2>
<div class="legend"><span><i style="background:var(--a)"></i>shaped: the ante's three blinds x their weight x the ante's fade</span><span><i style="background:var(--b)"></i>sparse stream (not faded)</span><span><i style="background:var(--c)"></i>win bonus</span></div>
<svg id="c4" viewBox="0 0 900 270"></svg></div>
<div class="chart" id="currbox" style="display:none"><h2>Curriculum: how often a scenario is cleared (one blind beaten / the few blinds got through), by update</h2>
<div class="legend"><span><i style="background:var(--a)"></i>one-blind scenarios cleared</span><span><i style="background:var(--b)"></i>few-blind scenarios cut off at their end (survived)</span><span><i style="background:var(--c)"></i>share of games that are real runs</span></div>
<svg id="c6" viewBox="0 0 900 250"></svg>
<h2 style="margin-top:14px">Auxiliary heads (loss; down = it can tell what plays and jokers are worth)</h2>
<div class="legend"><span><i style="background:var(--a)"></i>score of the chosen play (log)</span><span><i style="background:var(--b)"></i>leave-one-out joker contribution</span></div>
<svg id="c7" viewBox="0 0 900 220"></svg>
<h2 style="margin-top:14px">Jokers bought, by ante (real games, last 30 updates): share that are +mult / xMult / scaling / economy / chips</h2>
<div id="buytable" style="overflow-x:auto"></div></div>
<div class="chart" id="buildsbox" style="display:none"><h2>What it wins with</h2>
<div class="sub" id="builds_note"></div>
<div id="builds_counts"></div>
<div id="builds_table" style="overflow-x:auto"></div></div>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const stat = (label, value) => '<div class="stat"><b>' + value + '</b><span>' + label + '</span></div>';
const s = D.summary;
document.getElementById('stats').innerHTML = stat('games', s.games.toLocaleString()) + stat('update', s.iter) + stat('blinds, recent', s.blinds.toFixed(2)) +
  stat('best window', s.best.toFixed(2)) + stat('wins', s.wins) + stat('win rate, recent', (D.winrate[D.winrate.length - 1] || 0).toFixed(1) + '%') + stat('steps / s', s.sps.toLocaleString()) +
  (s.eval ? stat('greedy eval blinds', s.eval.blinds.toFixed(2) + ' (' + (100 * s.eval.win_rate).toFixed(1) + '% wins)') : '') +
  (s.ladder ? stat('training stake', s.ladder.stake + ' (' + s.ladder.since + ' updates)') : '') +
  (s.ladder && s.ladder.next ? stat('next: ' + s.ladder.next + ' preview', s.ladder.preview == null ? 'n/a' : (100 * s.ladder.preview).toFixed(1) + '% (need ' + (100 * s.ladder.dense).toFixed(0) + '%)') : '');
function chart(id, H, series, ymax, fmt, points) {
  const svg = document.getElementById(id), W = 900, L = 48, R = 14, T = 10, B = 26, xs = D.games;
  if (!xs.length) { svg.innerHTML = '<text x="20" y="40">waiting for the first rows</text>'; return; }
  const x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
  const top = ymax || Math.max(1, ...series.flatMap(q => q.v), ...(points || []).map(p => p[1])) * 1.15;
  const X = g => L + (W - L - R) * (g - x0) / (x1 - x0), Y = v => T + (H - T - B) * (1 - v / top);
  let out = '';
  for (let k = 0; k <= 4; k++) { const v = top * k / 4, y = Y(v);
    out += '<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+fmt(v)+'</text>'; }
  for (let k = 0; k <= 4; k++) { const g = x0 + (x1 - x0) * k / 4;
    out += '<text x="'+X(g)+'" y="'+(H-6)+'" text-anchor="'+(k==0?'start':k==4?'end':'middle')+'">'+Math.round(g).toLocaleString()+' games</text>'; }
  for (const q of series) {
    const d = q.v.map((v, i) => (i ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + Y(v).toFixed(1)).join(' ');
    out += '<path d="'+d+'" fill="none" stroke="'+css(q.color)+'" stroke-width="'+(q.w||2)+'" stroke-linejoin="round"'+(q.dash?' stroke-dasharray="5 4"':'')+'/>';
    const n = q.v.length - 1;
    out += '<text x="'+(W-R)+'" y="'+(Y(q.v[n])-7)+'" text-anchor="end" style="fill:'+css(q.color)+'">'+fmt(q.v[n])+'</text>';
  }
  for (const p of (points || [])) out += '<circle cx="'+X(p[0])+'" cy="'+Y(p[1])+'" r="3.5" fill="'+css('--c')+'"/>';
  svg.innerHTML = out;
}
chart('c1', 270, [{v: D.best, color: '--b', dash: true, w: 1.8}, {v: D.blinds, color: '--a', w: 2.2}], null, v => v.toFixed(1), D.eval_points);
chart('c2', 230, [{v: D.wins, color: '--a'}], null, v => Math.round(v).toString());
chart('c5', 250, [{v: D.winrate, color: '--a', w: 2.2}], null, v => v.toFixed(1) + '%', D.eval_wins);
const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.dataset.theme !== 'light';
const rampColor = k => 'hsl(210,70%,' + (dark ? 78 - k * 6 : 74 - k * 6) + '%)';
(function () {
  const svg = document.getElementById('c3'), W = 900, H = 250, L = 48, R = 56, T = 10, B = 26;
  if (!D.fade || !D.fade.length) { svg.innerHTML = '<text x="20" y="40">not a two-stream run (no fade weights in the log)</text>'; return; }
  const xs = D.games, x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
  const X = g => L + (W - L - R) * (g - x0) / (x1 - x0), Y = v => T + (H - T - B) * (1 - v);
  let out = '';
  for (let k = 0; k <= 4; k++) { const v = k / 4, y = Y(v);
    out += '<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+v.toFixed(2)+'</text>'; }
  for (let k = 0; k <= 4; k++) { const g = x0 + (x1 - x0) * k / 4;
    out += '<text x="'+X(g)+'" y="'+(H-6)+'" text-anchor="'+(k==0?'start':k==4?'end':'middle')+'">'+Math.round(g).toLocaleString()+' games</text>'; }
  for (let a = 0; a < D.fade[0].length; a++) {
    const d = D.fade.map((row, i) => (i ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + Y(row[a]).toFixed(1)).join(' ');
    out += '<path d="'+d+'" fill="none" stroke="'+rampColor(a)+'" stroke-width="2" stroke-linejoin="round"/>';
    const last = D.fade[D.fade.length - 1][a];
    out += '<text x="'+(W-R+6)+'" y="'+(Y(last)+4+(a%2?0:0))+'" style="fill:'+rampColor(a)+'">ante '+(a+1)+'</text>';
  }
  svg.innerHTML = out;
})();
(function () {
  const svg = document.getElementById('c4'), W = 900, H = 270, L = 48, R = 14, T = 22, B = 30;
  if (!D.fade || !D.fade.length) { svg.innerHTML = '<text x="20" y="40">not a two-stream run</text>'; return; }
  const w = D.fade[D.fade.length - 1];
  const items = D.base_ante.map((b, k) => ({label: 'ante ' + (k + 1), v: k + 1 >= D.sparse_from ? b : b * w[k], color: k + 1 >= D.sparse_from ? '--b' : '--a'}));
  items.push({label: 'win', v: D.win_bonus, color: '--c'});
  const top = Math.max(...items.map(i => i.v)) * 1.12;
  const bw = (W - L - R) / items.length, Y = v => T + (H - T - B) * (1 - v / top);
  let out = '';
  for (let k = 0; k <= 4; k++) { const v = top * k / 4, y = Y(v);
    out += '<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+v.toFixed(0)+'</text>'; }
  items.forEach((it, i) => {
    const x = L + i * bw + bw * 0.18, ww = bw * 0.64, y = Y(it.v);
    out += '<rect x="'+x+'" y="'+y+'" width="'+ww+'" height="'+Math.max(0, Y(0) - y)+'" rx="4" fill="'+css(it.color)+'"/>';
    out += '<text x="'+(x + ww / 2)+'" y="'+(y - 5)+'" text-anchor="middle">'+it.v.toFixed(it.v >= 10 ? 0 : 2)+'</text>';
    out += '<text x="'+(x + ww / 2)+'" y="'+(H - 10)+'" text-anchor="middle">'+it.label+'</text>';
  });
  svg.innerHTML = out;
})();
(function () {
  const B = D.builds; if (!B) return;
  document.getElementById('buildsbox').style.display = 'block';
  const esc = t => String(t).replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));
  const last = B.summaries.length ? B.summaries[B.summaries.length - 1] : null;
  document.getElementById('builds_note').innerHTML = B.n_wins + ' wins recorded' + (last ? ' · latest snapshot (update ' + last.iter + '): ' + last.wins + ' of ' + last.games + ' sampled games won' : '') +
    (B.n_wins ? '' : ' · nothing to show yet: the recorder samples games every 50 updates');
  if (!B.n_wins) return;
  const mx = Math.max(...B.counts.map(c => c[1]));
  document.getElementById('builds_counts').innerHTML = '<div style="font-size:12px;color:var(--muted);margin:6px 0">jokers held at the end of a win (all wins so far)</div>' +
    B.counts.map(c => '<div style="display:flex;align-items:center;gap:8px;margin:2px 0;font-size:12px"><span style="width:130px;text-align:right">' + esc(c[0]) + '</span><span style="display:inline-block;height:10px;border-radius:3px;background:var(--c);width:' + (260 * c[1] / mx) + 'px"></span><span>' + c[1] + '</span></div>').join('');
  document.getElementById('builds_table').innerHTML = '<table style="border-collapse:collapse;font-size:12px;margin-top:10px;width:100%"><tr style="text-align:left;color:var(--muted)">' +
    '<th>update</th><th>jokers</th><th>hand levels</th><th>money left</th><th>planets</th><th>skips</th><th>boss</th><th>flag</th></tr>' +
    B.recent.map(w => '<tr style="border-top:1px solid var(--grid)"><td>' + w.iter + '</td><td>' + esc(w.jokers.map(j => j[0] + (j[1] ? ' (' + j[1].toLowerCase() + ')' : '')).join(', ')) + '</td><td>' + esc(Object.entries(w.levels).map(e => e[0] + ' ' + e[1]).join(', ') || 'all 1') +
    '</td><td>$' + w.money + '</td><td>' + w.planets + '</td><td>' + w.skips + '</td><td>' + esc(w.boss) + '</td><td style="color:var(--b)">' + esc(w.flag) + '</td></tr>').join('') + '</table>';
})();
(function () {
  const O = D.oracle; if (!O || !O.iters.length) return;
  document.getElementById('orbox').style.display = 'block';
  const Lo = O.iters.length - 1;
  document.getElementById('or_note').innerHTML = 'Update ' + O.iters[Lo] + ': oracle log-share ' + O.logshare[Lo].toFixed(3) + ', money-option ' + (100 * O.pfind[Lo]).toFixed(0) +
    '%; potential weights ' + O.w_build[Lo].toFixed(2) + ' (build) / ' + O.w_money[Lo].toFixed(2) + ' (money); ' + O.calls[Lo] + ' oracle calls in the update.';
  function oplot(id, xs, ys, color, ymax, fmt) {
    const svg = document.getElementById(id), W = 440, H = 220, L = 40, Rr = 10, T = 10, B = 24;
    const x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
    const X = g => L + (W - L - Rr) * (g - x0) / (x1 - x0), Y = v => T + (H - T - B) * (1 - v / ymax);
    let out = '';
    for (let k = 0; k <= 4; k++) { const v = ymax * k / 4, y = Y(v);
      out += '<line x1="'+L+'" x2="'+(W-Rr)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+fmt(v)+'</text>'; }
    for (let k = 0; k <= 4; k++) { const g = x0 + (x1 - x0) * k / 4;
      out += '<text x="'+X(g)+'" y="'+(H-6)+'" text-anchor="'+(k==0?'start':k==4?'end':'middle')+'">'+Math.round(g)+'</text>'; }
    let dstr = '', pen = false;
    ys.forEach((v, i) => { if (v == null) { pen = false; return; } dstr += (pen ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + Y(v).toFixed(1) + ' '; pen = true; });
    out += '<path d="'+dstr+'" fill="none" stroke="'+css(color)+'" stroke-width="2" stroke-linejoin="round"/>';
    svg.innerHTML = out;
  }
  oplot('or1', O.iters, O.logshare, '--a', 1, v => v.toFixed(2));
  oplot('or2', O.iters, O.pfind, '--c', 1, v => (100 * v).toFixed(0) + '%');
})();
(function () {
  const A = D.auto; if (!A) return;
  document.getElementById('autobox').style.display = 'block';
  const L = A.iters.length - 1;
  document.getElementById('auto_note').innerHTML = 'Update ' + A.iters[L] + ': ' + (100 * A.share[L]).toFixed(0) + '% of blind selects are handed to the solver (random choice would be ~37%). Training blinds ' + A.blinds[L].toFixed(2) +
    (A.ev.length ? '; evaluation without the solver ' + A.ev[A.ev.length - 1][1].toFixed(2) + ' (update ' + A.ev[A.ev.length - 1][0] + ')' : '; no evaluation yet') + '.';
  const svg = document.getElementById('au1'), W = 900, H = 250, Lm = 48, R = 14, T = 10, B = 26, xs = A.iters;
  const top = Math.max(1, ...A.blinds, ...A.ev.map(e => e[1])) * 1.1;
  const x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
  const X = g => Lm + (W - Lm - R) * (g - x0) / (x1 - x0), Y = v => T + (H - T - B) * (1 - v / top), Ys = v => T + (H - T - B) * (1 - v);
  let out = '';
  for (let k = 0; k <= 4; k++) { const v = top * k / 4, y = Y(v);
    out += '<line x1="'+Lm+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(Lm-6)+'" y="'+(y+4)+'" text-anchor="end">'+v.toFixed(1)+'</text><text x="'+(W-R+2)+'" y="'+(y+4)+'">'+(100*k/4).toFixed(0)+'%</text>'; }
  for (let k = 0; k <= 4; k++) { const g = x0 + (x1 - x0) * k / 4;
    out += '<text x="'+X(g)+'" y="'+(H-6)+'" text-anchor="'+(k==0?'start':k==4?'end':'middle')+'">update '+Math.round(g)+'</text>'; }
  const path = (vs, f, col) => '<path d="'+vs.map((v, i) => (i ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + f(v).toFixed(1)).join(' ')+'" fill="none" stroke="'+css(col)+'" stroke-width="2"/>';
  out += path(A.blinds, Y, '--a') + path(A.share, Ys, '--b');
  A.ev.forEach(e => { out += '<circle cx="'+X(e[0])+'" cy="'+Y(e[1])+'" r="4" fill="'+css('--c')+'"/>'; });
  svg.innerHTML = out;
})();
(function () {
  const I = D.imm; if (!I) return;
  document.getElementById('immbox').style.display = 'block';
  document.getElementById('imm_note').innerHTML = 'Update ' + I.iter + '. Share of games (last ' + I.window + ' updates, ' + I.games + ' games) that beat each blind. A failed blind does not end the run, so every game meets every ante.';
  const cell = v => { const a = Math.max(0, Math.min(1, v)); return '<td style="text-align:center;background:rgba(42,111,176,' + (0.08 + 0.75 * a).toFixed(2) + ');color:' + (a > 0.55 ? '#fff' : 'inherit') + '">' + (100 * v).toFixed(0) + '%</td>'; };
  document.getElementById('imm_table').innerHTML = '<table style="border-collapse:collapse;font-size:12px;width:100%"><tr style="color:var(--muted)"><th style="text-align:left">ante</th><th>small</th><th>big</th><th>boss</th><th>weight per blind</th>' + (I.mult ? '<th>pays now (x multiplier)</th>' : '') + '</tr>' +
    I.rate.map((r, a) => '<tr style="border-top:1px solid var(--grid)"><td>' + (a + 1) + '</td>' + r.map(cell).join('') + '<td style="text-align:center;color:var(--muted)">' + I.weights[a].toFixed(2) + '</td>' + (I.mult ? '<td style="text-align:center">' + (I.weights[a] * I.mult[a]).toFixed(2) + ' (x' + I.mult[a].toFixed(2) + ')</td>' : '') + '</tr>').join('') + '</table>';
  const svg = document.getElementById('m1'), W = 900, H = 220, L = 44, Rr = 14, T = 10, B = 26, xs = I.iters;
  const x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
  const X = g => L + (W - L - Rr) * (g - x0) / (x1 - x0), Y = v => T + (H - T - B) * (1 - v);
  let out = '';
  for (let k = 0; k <= 4; k++) { const v = k / 4, y = Y(v);
    out += '<line x1="'+L+'" x2="'+(W-Rr)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+(100*v).toFixed(0)+'%</text>'; }
  for (let k = 0; k <= 4; k++) { const g = x0 + (x1 - x0) * k / 4;
    out += '<text x="'+X(g)+'" y="'+(H-6)+'" text-anchor="'+(k==0?'start':k==4?'end':'middle')+'">update '+Math.round(g)+'</text>'; }
  I.series.forEach((s, a) => {
    const dstr = s.map((v, i) => (i ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + Y(v).toFixed(1)).join(' ');
    const col = 'hsl(210,70%,' + (dark ? 78 - a * 7 : 74 - a * 7) + '%)';
    out += '<path d="'+dstr+'" fill="none" stroke="'+col+'" stroke-width="2"/><text x="'+(W-Rr+2)+'" y="'+(Y(s[s.length-1])+4)+'" style="fill:'+col+'">'+(a+1)+'</text>';
  });
  svg.innerHTML = out;
})();
(function () {
  const R = D.rev; if (!R || !R.iters.length) return;
  document.getElementById('revbox').style.display = 'block';
  const esc = t => String(t).replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));
  const last = R.iters.length - 1;
  document.getElementById('rev_note').innerHTML = 'Update ' + R.iters[last] + ': games start from antes ' + R.lo[last] + '-8 (' + (100 * R.share).toFixed(0) + '% of games, the rest from ante 1). ' +
    (R.band_wr[last] == null ? 'Win rate in the earliest ante of the band: not enough games yet.' : 'Win rate in ante ' + R.lo[last] + ': ' + (100 * R.band_wr[last]).toFixed(0) + '% (' + R.band_ends + ' games so far; needs ' + (100 * R.adv).toFixed(0) + '% over ' + R.adv_games + ').') +
    (R.stalled_for >= 1 ? ' <b>Held at ante ' + R.lo[last] + ' for ' + R.stalled_for + ' updates.</b>' : '');
  function plot(id, xs, series, ymax, fmt, hline, step) {
    const svg = document.getElementById(id), W = 440, H = 220, L = 40, Rr = 10, T = 10, B = 24;
    const x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
    const X = g => L + (W - L - Rr) * (g - x0) / (x1 - x0), Y = v => T + (H - T - B) * (1 - v / ymax);
    let out = '';
    for (let k = 0; k <= 4; k++) { const v = ymax * k / 4, y = Y(v);
      out += '<line x1="'+L+'" x2="'+(W-Rr)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+fmt(v)+'</text>'; }
    for (let k = 0; k <= 4; k++) { const g = x0 + (x1 - x0) * k / 4;
      out += '<text x="'+X(g)+'" y="'+(H-6)+'" text-anchor="'+(k==0?'start':k==4?'end':'middle')+'">'+Math.round(g)+'</text>'; }
    if (hline != null) out += '<line x1="'+L+'" x2="'+(W-Rr)+'" y1="'+Y(hline)+'" y2="'+Y(hline)+'" stroke="'+css('--b')+'" stroke-dasharray="5 4"/>';
    for (const q of series) {
      let dstr = '', pen = false, prev = null;
      q.v.forEach((v, i) => { if (v == null) { pen = false; return; }
        if (step && pen) dstr += 'L' + X(xs[i]).toFixed(1) + ' ' + Y(prev).toFixed(1) + ' ';
        dstr += (pen ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + Y(v).toFixed(1) + ' '; pen = true; prev = v; });
      out += '<path d="'+dstr+'" fill="none" stroke="'+css(q.color)+'" stroke-width="2" stroke-linejoin="round"/>';
    }
    svg.innerHTML = out;
  }
  plot('r1', R.iters, [{v: R.lo, color: '--a'}], 8, v => v.toFixed(0), null, true);
  plot('r2', R.iters, [{v: R.band_wr, color: '--c'}], 1, v => (100 * v).toFixed(0) + '%', R.adv, false);
  document.getElementById('rev_table').innerHTML = '<table style="border-collapse:collapse;font-size:12px;width:100%"><tr style="text-align:left;color:var(--muted)">' +
    '<th>start ante</th><th>in band</th><th>snapshots held</th><th>games started</th><th>games ended</th><th>won</th><th>win rate</th><th>all updates: ended</th><th>won</th><th>win rate</th></tr>' +
    R.table.map(r => '<tr style="border-top:1px solid var(--grid)' + (r.inband ? '' : ';color:var(--muted)') + '"><td>' + r.ante + '</td><td>' + (r.inband ? 'yes' : '-') + '</td><td>' + r.snaps + '</td><td>' + r.starts +
      '</td><td>' + r.ends + '</td><td>' + r.wins + '</td><td>' + (r.ends ? (100 * r.wins / r.ends).toFixed(1) + '%' : '-') +
      '</td><td>' + r.ends_all + '</td><td>' + r.wins_all + '</td><td>' + (r.ends_all ? (100 * r.wins_all / r.ends_all).toFixed(2) + '%' : '-') + '</td></tr>').join('') + '</table>';
  document.getElementById('rev_events').innerHTML = R.events.map(e => esc(e)).join('<br>');
})();
(function () {
  const C = D.curr; if (!C || !C.iters.length) return;
  document.getElementById('currbox').style.display = 'block';
  function line(id, H, series, fmt, ymax) {
    const svg = document.getElementById(id), W = 900, L = 48, R = 14, T = 10, B = 26, xs = C.iters;
    const x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
    const top = ymax || Math.max(1e-9, ...series.flatMap(q => q.v.filter(v => v != null))) * 1.15;
    const X = g => L + (W - L - R) * (g - x0) / (x1 - x0), Y = v => T + (H - T - B) * (1 - v / top);
    let out = '';
    for (let k = 0; k <= 4; k++) { const v = top * k / 4, y = Y(v);
      out += '<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+fmt(v)+'</text>'; }
    for (let k = 0; k <= 4; k++) { const g = x0 + (x1 - x0) * k / 4;
      out += '<text x="'+X(g)+'" y="'+(H-6)+'" text-anchor="'+(k==0?'start':k==4?'end':'middle')+'">update '+Math.round(g)+'</text>'; }
    for (const q of series) {
      let d = '', pen = false;
      q.v.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + Y(v).toFixed(1) + ' '; pen = true; });
      out += '<path d="'+d+'" fill="none" stroke="'+css(q.color)+'" stroke-width="2" stroke-linejoin="round"/>';
    }
    svg.innerHTML = out;
  }
  line('c6', 250, [{v: C.one, color: '--a'}, {v: C.few, color: '--b'}, {v: C.real, color: '--c'}], v => (100 * v).toFixed(0) + '%', 1);
  line('c7', 220, [{v: C.play, color: '--a'}, {v: C.loo, color: '--b'}], v => v.toFixed(3));
  const names = ['+mult', 'xMult', 'scaling', 'economy', 'chips'];
  document.getElementById('buytable').innerHTML = '<table style="border-collapse:collapse;font-size:12px;width:100%"><tr style="text-align:left;color:var(--muted)"><th>ante</th><th>jokers bought</th>' +
    names.map(n => '<th>' + n + '</th>').join('') + '</tr>' + C.buy.map(r => '<tr style="border-top:1px solid var(--grid)"><td>' + r[0] + '</td><td>' + r[6] + '</td>' +
    r.slice(1, 6).map(v => '<td>' + (r[6] ? (100 * v).toFixed(0) + '%' : '-') + '</td>').join('') + '</tr>').join('') + '</table>';
})();
document.getElementById('lg3').innerHTML = (D.fade && D.fade.length) ? D.fade[0].map((_, a) => '<span><i style="background:' + rampColor(a) + '"></i>ante ' + (a + 1) + '</span>').join('') : '';
</script></main></body></html>
"""


REWARD = {}


def ante_rewards(ante_weight, power, win_bonus, sparse_from, geom=0.0):
    """The engine's per-blind rewards (Env::set_weights: 1 at ante 1 up to ante_weight at ante 8, scaled to add up to 24), summed per ante."""
    raw = ([geom ** (b // 3) for b in range(24)] if geom > 0 else
           [1 + (ante_weight - 1) * (((b // 3) / 7.0) ** power if power != 1.0 else (b // 3) / 7.0) for b in range(24)])
    sc = 24.0 / sum(raw)
    return dict(base_ante=[sum(raw[3 * k + j] * sc for j in range(3)) for k in range(8)], win_bonus=win_bonus, sparse_from=sparse_from)


def read(log):
    rows = []
    if os.path.exists(log):
        for line in open(log, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "games" in r and r.get("blinds") is not None:
                    rows.append(r)
    return rows


def read_builds(path, recent=15):
    """The recorder's file (tools/record_wins.py): the wins (jokers, hand levels, money left ...) and the per-snapshot summaries."""
    if not path or not os.path.exists(path):
        return None
    wins, sums = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        (wins if r.get("type") == "win" else sums if r.get("type") == "summary" else []).append(r)
    counts = {}
    for w in wins:
        for j in w.get("jokers", []):
            counts[j["key"]] = counts.get(j["key"], 0) + 1
    rec = []
    for w in wins[-recent:][::-1]:
        flag = []
        if w.get("money", 0) >= 500:
            flag.append("money left $%d: a money route?" % w["money"])
        if w.get("planet_uses", 0) >= 3:
            flag.append("planets")
        rec.append(dict(iter=w.get("iter"), jokers=[[j["key"], j.get("edition", "")] for j in w.get("jokers", [])], levels=w.get("hand_levels_named", {}),
                        money=w.get("money", 0), planets=w.get("planet_uses", 0), skips=w.get("blinds_skipped", 0), boss=w.get("boss", ""), flag="; ".join(flag)))
    return dict(n_wins=len(wins), counts=sorted(counts.items(), key=lambda kv: -kv[1])[:15], recent=rec,
                summaries=[dict(iter=x["iter"], wins=x["wins"], games=x["games"]) for x in sums[-30:]])


def build(rows, max_points=1500, stake=""):
    wins0 = 0
    if stake:                                          # only the rows trained on this stake (a --ladder run); wins counted from its first row
        rows = [r for r in rows if r.get("ladder_stake") == stake]
        if rows:
            wins0 = rows[0].get("wins", 0)
    if not rows:
        return dict(games=[], blinds=[], best=[], wins=[], eval_points=[], summary=dict(games=0, iter=0, blinds=0.0, best=0.0, wins=0, sps=0, eval=None))
    best, b = [], -1.0
    for r in rows:
        b = max(b, r["blinds"])
        best.append(b)
    step = max(1, len(rows) // max_points)
    keep = list(range(0, len(rows), step))
    if keep[-1] != len(rows) - 1:
        keep.append(len(rows) - 1)
    ev = [(r["games"], r["eval"]["blinds"]) for r in rows if isinstance(r.get("eval"), dict)]
    last_eval = next((r["eval"] for r in reversed(rows) if isinstance(r.get("eval"), dict)), None)
    last = rows[-1]
    fade = [rows[i]["fade_w"] for i in keep] if all("fade_w" in rows[i] for i in keep) else []
    ev_w = [(r["games"], 100.0 * r["eval"]["win_rate"]) for r in rows if isinstance(r.get("eval"), dict)]
    return dict(games=[rows[i]["games"] for i in keep], blinds=[rows[i]["blinds"] for i in keep], best=[best[i] for i in keep],
                winrate=[float(rows[i].get("win%", 0) or 0) for i in keep], eval_wins=ev_w,
                wins=[rows[i].get("wins", 0) - wins0 for i in keep], eval_points=ev, fade=fade, **REWARD,
                summary=dict(games=last["games"], iter=last.get("iter", 0), blinds=last["blinds"], best=best[-1], wins=last.get("wins", 0) - wins0,
                             sps=last.get("sps", 0), eval=last_eval,
                             ladder=({"stake": last["ladder_stake"], "next": last.get("ladder_next"), "preview": last.get("ladder_preview"),
                                      "since": last.get("ladder_since", 0), "dense": last.get("ladder_dense", 0.2)} if "ladder_stake" in last else None)))


def curriculum(rows):
    """The curriculum run's own columns (train.py --scenario-*): clear rates of the scenarios by update, the aux losses, the jokers bought by ante."""
    rows = [r for r in rows if "scn" in r or "buy_share" in r]
    if not rows:
        return None
    step = max(1, len(rows) // 400)
    ks = list(range(0, len(rows), step))

    def rate(r, kind, key):
        k = (r.get("scn") or {}).get(kind) or {}
        return k.get(key, 0) / k["games"] if k.get("games", 0) >= 20 else None
    buy = [[0.0] * 6 for _ in range(8)]
    for r in rows[-30:]:
        for a_, row in enumerate(r.get("buy_share") or []):
            n = row[5]
            for c in range(5):
                buy[a_][c] += row[c] * n
            buy[a_][5] += n
    return dict(iters=[rows[i]["iter"] for i in ks], one=[rate(rows[i], "one_blind", "cut") for i in ks], few=[rate(rows[i], "few_blinds", "cut") for i in ks],
                real=[1.0 - rows[i].get("scn_p", 0.0) for i in ks], play=[rows[i].get("aux_play") for i in ks], loo=[rows[i].get("aux_loo") for i in ks],
                buy=[[a_ + 1] + [(b[c] / b[5]) if b[5] else 0.0 for c in range(5)] + [int(b[5])] for a_, b in enumerate(buy)])


def read_events(path):
    """The seeding results and band moves the trainer prints (they are in its output file, not in the log)."""
    ev = []
    if not path or not os.path.exists(path):
        return ev
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"archive_seed"' in line or '"archive_band"' in line:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "archive_seed" in r:
                    s = r["archive_seed"]
                    ev.append("seeded: %d games from +$%d (greedy), %.0fs" % (s["games"], s["money"], s["secs"]))
                else:
                    s = r["archive_band"]
                    ev.append("update %d: band extended back to ante %d (win rate %.0f%% over %d games)" % (s["iter"], s["extended_to"], 100 * s["win_rate"], s["games"]))
    return ev


def reverse(rows, events, adv, adv_games, share):
    """The reverse-curriculum run's columns (train.py --archive-ante reverse): the band's lower edge and win rate over the updates, and a table by start ante."""
    rows = [r for r in rows if "archive_lo" in r]
    if not rows:
        return None
    step = max(1, len(rows) // 500)
    ks = list(range(0, len(rows), step))
    if ks[-1] != len(rows) - 1:
        ks.append(len(rows) - 1)
    last = rows[-1]
    tail = rows[-25:]
    ends = [0] * 24
    wins = [0] * 24
    starts = [0] * 24
    for r in tail:
        o = r.get("archive_out")
        if o:
            for b in range(24):
                ends[b] += o[0][b]; wins[b] += o[1][b]
        for b, x in enumerate(r.get("archive_starts") or []):
            starts[b] += x
    ends_all, wins_all = [0] * 24, [0] * 24                      # every update since outcomes were logged
    for r in rows:
        o = r.get("archive_out")
        if o:
            for b in range(24):
                ends_all[b] += o[0][b]; wins_all[b] += o[1][b]
    fill = last.get("archive_fill") or [0] * 24
    lo = last["archive_lo"]
    table = []
    for a_ in range(1, 9):
        sl = slice(3 * (a_ - 1), 3 * a_)
        table.append(dict(ante=a_, inband=a_ >= lo, snaps=sum(fill[sl]), starts=sum(starts[sl]), ends=sum(ends[sl]), wins=sum(wins[sl]),
                          ends_all=sum(ends_all[sl]), wins_all=sum(wins_all[sl])))
    stalled = 0
    for r in reversed(rows):
        if r["archive_lo"] == lo:
            stalled += 1
        else:
            break
    b0 = 3 * (lo - 1)
    band_ends = 0
    for r in reversed(rows):
        if r["archive_lo"] != lo:
            break
        o = r.get("archive_out")
        if o:
            band_ends += sum(o[0][b0:b0 + 3])
    return dict(iters=[rows[i]["iter"] for i in ks], lo=[rows[i]["archive_lo"] for i in ks], band_wr=[rows[i].get("archive_band_wr") for i in ks], table=table,
                events=events[-25:], adv=adv, adv_games=adv_games, share=share, stalled_for=stalled if len(rows) > 3 else 0, band_ends=band_ends)


def autoplay(rows):
    """--autoplay runs: the share of blind selects handed to the solver, the training blinds, and the evaluation (which never has the solver)."""
    rows = [r for r in rows if "auto_share" in r]
    if not rows:
        return None
    step = max(1, len(rows) // 400)
    ks = list(range(0, len(rows), step))
    if ks[-1] != len(rows) - 1:
        ks.append(len(rows) - 1)
    return dict(iters=[rows[i]["iter"] for i in ks], share=[rows[i]["auto_share"] for i in ks], blinds=[rows[i]["blinds"] for i in ks],
                ev=[[r["iter"], r["eval"]["blinds"]] for r in rows if isinstance(r.get("eval"), dict) and r["iter"] > 1])


def oracle(rows):
    """The ante-8 oracle and money-option means per update (train.py --oracle-*), sampled to ~400 points."""
    rows = [r for r in rows if isinstance(r.get("oracle"), dict)]
    if not rows:
        return None
    step = max(1, len(rows) // 400)
    ks = list(range(0, len(rows), step))
    if ks[-1] != len(rows) - 1:
        ks.append(len(rows) - 1)
    pick = lambda key: [rows[i]["oracle"].get(key) for i in ks]
    return dict(iters=[rows[i]["iter"] for i in ks], logshare=pick("logshare"), pfind=pick("pfind"), w_build=pick("w_build"), w_money=pick("w_money"), calls=pick("calls"))


def immortal(rows, window=10):
    """The immortal run's per-blind clear rates (train.py --immortal): the table over the last `window` updates and, for each ante's first blind, the rate over the updates."""
    rows = [r for r in rows if "clear_rate" in r]
    if not rows:
        return None
    tail = rows[-window:]
    g = sum(r["clear_games"] for r in tail)
    rate = [[sum(r["clear_rate"][3 * a + j] * r["clear_games"] for r in tail) / max(1, g) for j in range(3)] for a in range(8)]
    step = max(1, len(rows) // 300)
    ks = list(range(0, len(rows), step))
    mult = rows[-1].get("ante_mult")
    return dict(iter=rows[-1]["iter"], window=len(tail), games=g, rate=rate, weights=ANTE_W, mult=mult, ema=rows[-1].get("ante_ema"), iters=[rows[i]["iter"] for i in ks],
                series=[[rows[i]["clear_rate"][3 * a] for i in ks] for a in range(8)])


ANTE_W = []


def write(a):
    html = (PAGE.replace("__DATA__", json.dumps({**build(read(a.log), stake=a.stake), "builds": read_builds(a.builds), "curr": curriculum(read(a.log)),
                                                          "imm": immortal(read(a.log)), "auto": autoplay(read(a.log)), "oracle": oracle(read(a.log)), "rev": reverse(read(a.log), read_events(a.events), a.advance, a.advance_games, a.archive_share)})).replace("__REFRESH__", str(a.every))
            .replace("__ADV__", str(int(round(100 * a.advance)))).replace("__TITLE__", a.title).replace("__UPDATED__", time.strftime("%H:%M:%S")))
    tmp = a.out + ".tmp"
    open(tmp, "w", encoding="utf-8").write(html)
    os.replace(tmp, a.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out", default="traces/run.html")
    p.add_argument("--title", default="PPO run")
    p.add_argument("--every", type=int, default=15)
    p.add_argument("--builds", default="", help="a tools/record_wins.py file: adds a panel of what the wins were made of")
    p.add_argument("--stake", default="", help="show only the updates trained on this stake (a --ladder run's ladder_stake), e.g. RED")
    p.add_argument("--win-bonus", type=float, default=50.0, help="the run's terminal reward (--stream-win-bonus)")
    p.add_argument("--ante-weight", type=float, default=3.0, help="per-blind reward at ante 8 against ante 1 (--stream-ante-weight)")
    p.add_argument("--weight-power", type=float, default=1.0)
    p.add_argument("--events", default="", help="the trainer's output file (the seeding and band moves it prints; reverse-curriculum runs)")
    p.add_argument("--advance", type=float, default=0.3, help="the run's --archive-advance")
    p.add_argument("--advance-games", type=int, default=200, help="the run's --archive-advance-games")
    p.add_argument("--archive-share", type=float, default=0.5, help="the run's --archive-p")
    p.add_argument("--weight-geom", type=float, default=0.0)
    p.add_argument("--sparse-from-ante", type=int, default=7)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    REWARD.update(ante_rewards(a.ante_weight, a.weight_power, a.win_bonus, a.sparse_from_ante, a.weight_geom))
    ANTE_W[:] = [x / 3.0 for x in REWARD["base_ante"]]
    while True:
        write(a)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
