"""A live page for a Go-Explore run: deepest blind reached, cells in the archive and new cells per iteration, from the run's log.

    python tools/ge_dashboard.py --log traces/ge_42003.log --out traces/go_explore.html
"""
import argparse
import json
import os
import re
import time

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="__R__"><title>Go-Explore run</title><style>
:root{--bg:#fafaf7;--ink:#1d1d1b;--muted:#6b6b66;--grid:#e4e3dc;--card:#fff;--c1:#2a6fb0;--c2:#d9822b;--c3:#3c9a5f}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--ink:#ecebe6;--muted:#9a9990;--grid:#2c2d30;--card:#202225;--c1:#6aa8e6;--c2:#f0a95a;--c3:#5cc384}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}main{max-width:980px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 16px;font-size:13px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:18px}.stat{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px 12px}
.stat b{font-size:20px}.stat span{display:block;color:var(--muted);font-size:12px}.chart{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px 12px 6px;margin-bottom:16px}
.chart h2{font-size:14px;margin:0 0 6px}svg{width:100%;height:auto;display:block}text{fill:var(--muted);font-size:11px}</style></head><body><main>
<h1>Go-Explore: __TITLE__</h1><p class="sub">Refreshes every __R__ s. Updated __U__. __STATUS__</p><div class="stats" id="st"></div>
<div class="chart"><h2>Deepest blind reached, by seconds</h2><svg id="a" viewBox="0 0 900 230"></svg></div>
<div class="chart"><h2>Cells in the archive, by seconds</h2><svg id="b" viewBox="0 0 900 230"></svg></div>
<div class="chart"><h2>New cells found per iteration (the search's progress: near 0 means it has stopped finding anything new)</h2><svg id="c" viewBox="0 0 900 230"></svg></div>
<script>const D=__DATA__;const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const last=D[D.length-1]||{};document.getElementById('st').innerHTML=[['deepest blind',last.deep],['archive cells',last.cells],['expansions',last.exp],['seconds',last.t],['iterations',last.it]].map(x=>'<div class="stat"><b>'+(x[1]??'-')+'</b><span>'+x[0]+'</span></div>').join('');
function ch(id,key,col){const s=document.getElementById(id),W=900,H=230,L=44,R=20,T=10,B=24;if(!D.length){s.innerHTML='<text x="20" y="30">waiting</text>';return}
const x1=Math.max(...D.map(d=>d.t),1),hi=Math.max(1,...D.map(d=>d[key]))*1.1,X=x=>L+(W-L-R)*x/x1,Y=v=>T+(H-T-B)*(1-v/hi);let o='';
for(let k=0;k<=4;k++){const v=hi*k/4,y=Y(v);o+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'" stroke="'+css('--grid')+'"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end">'+v.toFixed(hi<10?1:0)+'</text>'}
o+='<path d="'+D.map((d,i)=>(i?'L':'M')+X(d.t).toFixed(1)+' '+Y(d[key]).toFixed(1)).join(' ')+'" fill="none" stroke="'+css(col)+'" stroke-width="2.2"/>';s.innerHTML=o}
ch('a','deep','--c1');ch('b','cells','--c2');ch('c','new','--c3');</script></main></body></html>"""
LINE = re.compile(r"it\s+(\d+)\s+(\d+)s\s+expansions (\d+)\s+cells (\d+)\s+deepest blind (\d+).*new this it (\d+)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out", default="traces/go_explore.html")
    p.add_argument("--every", type=int, default=10)
    a = p.parse_args()
    while True:
        rows, status, title = [], "running", os.path.basename(a.log)
        if os.path.exists(a.log):
            for l in open(a.log, errors="replace"):
                m = LINE.search(l)
                if m:
                    rows.append({"it": int(m[1]), "t": int(m[2]), "exp": int(m[3]), "cells": int(m[4]), "deep": int(m[5]), "new": int(m[6])})
                elif l.startswith("SOLVED"):
                    status = "SOLVED"
                elif l.startswith("no win"):
                    status = "finished: no win"
                elif l.startswith("go-explore on"):
                    title = l.strip()[len("go-explore on "):]
        html = PAGE.replace("__DATA__", json.dumps(rows)).replace("__R__", str(a.every)).replace("__U__", time.strftime("%H:%M:%S")).replace("__STATUS__", status).replace("__TITLE__", title)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out + ".tmp", "w", encoding="utf-8").write(html)
        os.replace(a.out + ".tmp", a.out)
        if status != "running":
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
