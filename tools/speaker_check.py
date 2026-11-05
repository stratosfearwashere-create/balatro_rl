"""Check left/right speaker placement with a phone as the microphone.

Run this on the PC that drives the speakers:

    python tools/speaker_check.py

It prints an https address. Open it on a phone on the same Wi-Fi (accept the
certificate warning, the certificate is generated locally), put the phone where
your head goes with the mic facing the speakers, and tap Start. The PC plays
a test signal through the speakers, the phone records it, and the PC analyzes
the recording:

  * arrival-time difference between left and right  -> distance mismatch in cm
  * level difference, overall and per octave band   -> balance / toe-in / room
  * polarity of each speaker and mono-sum gain      -> swapped +/- wiring

`--selftest` runs the analysis on a simulated recording, no audio hardware.
"""
import argparse
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

C_SOUND = 343.0          # m/s at about 20 C
PLAY_FS = 48000
BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]

CHIRP_DUR, CHIRP_F0, CHIRP_F1, CHIRP_AMP = 0.25, 200.0, 8000.0, 0.5
NOISE_DUR, NOISE_RMS = 1.2, 0.12
ROUNDS = 3
LEAD, ROUND_LEN, TAIL = 0.5, 6.0, 0.8
OFF_CHIRP_R, OFF_NOISE_L, OFF_NOISE_R, OFF_NOISE_LR = 0.6, 1.4, 2.9, 4.4
NOISE_SKIP, NOISE_USE = 0.1, 1.0   # analyze the middle second of each burst


def make_chirp(fs):
    n = int(CHIRP_DUR * fs)
    t = np.arange(n) / fs
    k = np.log(CHIRP_F1 / CHIRP_F0) / CHIRP_DUR
    x = np.sin(2 * np.pi * CHIRP_F0 * (np.exp(k * t) - 1) / k)
    fade = int(0.005 * fs)
    w = np.ones(n)
    w[:fade] = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    w[-fade:] = w[:fade][::-1]
    return x * w


def make_noise(fs):
    """Pink noise, 100 Hz to 12 kHz, fixed seed so every run is identical."""
    rng = np.random.default_rng(7)
    n = int(NOISE_DUR * fs)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    g = np.zeros_like(f)
    m = (f >= 100) & (f <= 12000)
    g[m] = 1 / np.sqrt(f[m])
    x = np.fft.irfft(X * g, n)
    x *= NOISE_RMS / np.sqrt(np.mean(x ** 2))
    fade = int(0.02 * fs)
    w = np.ones(n)
    w[:fade] = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    w[-fade:] = w[:fade][::-1]
    return x * w


def total_duration():
    return LEAD + ROUNDS * ROUND_LEN + TAIL


def make_signal(fs):
    """Stereo test signal. Per round: left chirp, right chirp, left noise,
    right noise, the same noise on both channels."""
    sig = np.zeros((int(total_duration() * fs), 2))
    chirp = CHIRP_AMP * make_chirp(fs)
    noise = make_noise(fs)

    def put(chans, start, x):
        i = int(round(start * fs))
        for c in chans:
            sig[i:i + len(x), c] += x

    for k in range(ROUNDS):
        T = LEAD + k * ROUND_LEN
        put([0], T, chirp)
        put([1], T + OFF_CHIRP_R, chirp)
        put([0], T + OFF_NOISE_L, noise)
        put([1], T + OFF_NOISE_R, noise)
        put([0, 1], T + OFF_NOISE_LR, noise)
    return sig.astype(np.float32)


def write_wav(path, sig, fs):
    pcm = np.clip(sig, -1, 1)
    pcm = (pcm * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())


# ----------------------------------------------------------------- analysis

def xcorr(rec, tpl):
    n = 1 << int(np.ceil(np.log2(len(rec) + len(tpl))))
    c = np.fft.irfft(np.fft.rfft(rec, n) * np.conj(np.fft.rfft(tpl, n)), n)
    return c[:len(rec)]


def envelope(c):
    n = len(c)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    return np.abs(np.fft.ifft(np.fft.fft(c) * h))


def parabolic(y0, y1, y2):
    d = y0 - 2 * y1 + y2
    return 0.0 if d == 0 else 0.5 * (y0 - y2) / d


def peak_near(env, c, center_s, fs, half=0.05):
    a = max(0, int((center_s - half) * fs))
    b = min(len(env), int((center_s + half) * fs))
    if b - a < 5:
        return None
    i = a + int(np.argmax(env[a:b]))
    d = parabolic(env[i - 1], env[i], env[i + 1]) if 0 < i < len(env) - 1 else 0.0
    rho = float(c[i] / env[i]) if env[i] > 0 else 0.0
    return (i + d) / fs, float(env[i]), rho


def band_power(x, fs):
    """Power in each octave band (arbitrary but consistent units)."""
    w = np.hanning(len(x))
    P = np.abs(np.fft.rfft(x * w)) ** 2 / (fs * np.sum(w ** 2))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    df = fs / len(x)
    out = []
    for fc in BANDS:
        m = (f >= fc / np.sqrt(2)) & (f < fc * np.sqrt(2))
        out.append(P[m].sum() * df)
    return np.array(out)


def db(p):
    return 10 * np.log10(np.maximum(p, 1e-18))


def analyze(rec, fs):
    rec = np.asarray(rec, dtype=np.float64)
    rec = rec - rec.mean()
    res = {'ok': False, 'fs': fs, 'warnings': [], 'advice': []}
    res['seconds'] = round(len(rec) / fs, 2)
    res['peak'] = round(float(np.max(np.abs(rec))), 3)
    if res['peak'] >= 0.98:
        res['warnings'].append('The recording clipped. Lower the speaker volume or move the phone back.')

    chirp = make_chirp(fs)
    c = xcorr(rec, chirp)
    env = envelope(c) / np.sum(chirp ** 2)
    if env.max() <= 0:
        res['error'] = 'Nothing recorded.'
        return res
    first = int(np.argmax(env > 0.25 * env.max()))
    win = int(0.05 * fs)
    t0 = (first + int(np.argmax(env[first:first + win]))) / fs
    ref = float(env.max())

    # ---- timing and polarity from the chirps
    rows = []
    for k in range(ROUNDS):
        TL = t0 + k * ROUND_LEN
        if TL + OFF_NOISE_LR + NOISE_DUR + 0.1 > len(rec) / fs:
            break
        pl = peak_near(env, c, TL, fs)
        pr = peak_near(env, c, TL + OFF_CHIRP_R, fs)
        if pl is None or pr is None:
            break
        rows.append((TL, pl, pr))
    if not rows:
        res['error'] = ('Could not find the test chirps. Raise the speaker volume, '
                        'check that both speakers play, and keep the room quiet.')
        return res

    weak = [min(r[1][1], r[2][1]) / ref for r in rows]
    if np.median(weak) < 0.2:
        res['warnings'].append('One channel is much quieter than the other or silent. '
                               'Check cables and that both speakers are on.')
    dts = [(r[2][0] - r[1][0]) - OFF_CHIRP_R for r in rows]
    dt = float(np.median(dts))
    res['rounds'] = len(rows)
    res['dt_ms'] = round(dt * 1000, 3)
    res['dist_cm'] = round(dt * C_SOUND * 100, 1)      # positive = right farther
    res['dt_spread_ms'] = round(float(np.ptp(dts)) * 1000, 3)
    rl = float(np.median([r[1][2] for r in rows]))
    rr = float(np.median([r[2][2] for r in rows]))
    res['rho'] = [round(rl, 2), round(rr, 2)]
    if rl * rr < 0 and min(abs(rl), abs(rr)) > 0.4:
        res['polarity'] = 'inverted'
    elif rl * rr > 0 and min(abs(rl), abs(rr)) > 0.4:
        res['polarity'] = 'same'
    else:
        res['polarity'] = 'unclear'

    # ---- levels from the noise bursts
    amb_end = min(t0 - 0.3, 1.5)
    if amb_end > 0.4:
        amb = band_power(rec[int(0.1 * fs):int(amb_end * fs)], fs)
    else:
        amb = np.full(len(BANDS), 1e-12)
        res['warnings'].append('Not enough silence before the test to measure room noise.')

    acc = {'L': [], 'R': [], 'LR': []}
    snr = []
    for TL, _, _ in rows:
        for key, off in (('L', OFF_NOISE_L), ('R', OFF_NOISE_R), ('LR', OFF_NOISE_LR)):
            a = int((TL + off + NOISE_SKIP) * fs)
            b = a + int(NOISE_USE * fs)
            p = band_power(rec[a:b], fs)
            acc[key].append(p)
            snr.append(db(p) - db(amb))
    pw = {k: np.mean(v, axis=0) for k, v in acc.items()}
    net = {k: np.maximum(v - amb, 1e-15) for k, v in pw.items()}
    snr_db = float(np.median(np.array(snr)[:, 1:6]))
    res['snr_db'] = round(snr_db, 1)
    if snr_db < 15:
        res['warnings'].append('The test is only %.0f dB above the room noise. '
                               'Turn the speakers up or quiet the room.' % snr_db)

    Ld, Rd, LRd = db(net['L']), db(net['R']), db(net['LR'])
    tot = {k: float(db(v.sum())) for k, v in net.items()}
    res['level_diff_db'] = round(tot['L'] - tot['R'], 2)       # positive = left louder
    res['bands'] = [{'hz': fc, 'l': round(float(Ld[i]), 1), 'r': round(float(Rd[i]), 1),
                     'diff': round(float(Ld[i] - Rd[i]), 1),
                     'sum': round(float(LRd[i] - (Ld[i] + Rd[i]) / 2), 1)}
                    for i, fc in enumerate(BANDS)]
    low = [b['sum'] for b in res['bands'][:3]]
    res['mono_sum_db'] = round(float(np.mean(low)), 1)
    hi_diff = float(np.mean([b['diff'] for b in res['bands'][4:6]]))
    res['hf_level_diff_db'] = round(hi_diff, 1)
    res['ok'] = True
    res['advice'] = make_advice(res)
    return res


def make_advice(r):
    out = []
    d = r['dist_cm']
    if abs(d) < 1.5:
        out.append('Distances match (within %.1f cm).' % abs(d))
    else:
        far, near = ('right', 'left') if d > 0 else ('left', 'right')
        out.append('The %s speaker is %.0f cm farther from the phone than the %s one. '
                   'Move it %.0f cm closer, or the %s one back. The image pulls toward the %s.'
                   % (far, abs(d), near, abs(d), near, near))
    ld = r['level_diff_db']
    if abs(ld) <= 0.7:
        out.append('Levels match (within %.1f dB).' % abs(ld))
    else:
        loud = 'left' if ld > 0 else 'right'
        out.append('The %s speaker is %.1f dB louder. The image pulls toward the %s. '
                   'Fix it with the balance control, then check toe-in and walls if it persists.'
                   % (loud, abs(ld), loud))
    if abs(r['hf_level_diff_db'] - ld) > 2.0:
        loud = 'left' if r['hf_level_diff_db'] > ld else 'right'
        out.append('Highs differ from the overall balance by %.1f dB, favoring the %s. '
                   'That usually means the toe-in angle differs between the two speakers.'
                   % (abs(r['hf_level_diff_db'] - ld), loud))
    if r['polarity'] == 'inverted' or r['mono_sum_db'] < 0.5:
        out.append('Polarity looks wrong: one speaker moves out while the other moves in. '
                   'Swap the + and - wires on one speaker.')
    elif r['polarity'] == 'same':
        out.append('Both speakers are wired in the same polarity.')
    return out


# --------------------------------------------------------------- live mode
# A looping pair of chirps, left then right 0.4 s later, repeating every 1.0 s.
# The uneven gaps (0.4 s, then 0.6 s) tell the phone which chirp is which, so
# the readout keeps updating while you move a speaker.
LIVE_PERIOD, LIVE_GAP = 1.0, 0.4


def make_live_signal(fs):
    sig = np.zeros((int(LIVE_PERIOD * fs), 2))
    chirp = CHIRP_AMP * make_chirp(fs)
    sig[:len(chirp), 0] = chirp
    a = int(LIVE_GAP * fs)
    sig[a:a + len(chirp), 1] = chirp
    return sig.astype(np.float32)


def live_analyze(rec, fs):
    rec = np.asarray(rec, dtype=np.float64)
    rec = rec - rec.mean()
    chirp = make_chirp(fs)
    c = xcorr(rec, chirp)
    env = envelope(c) / np.sum(chirp ** 2)
    mx = float(env.max()) if len(env) else 0.0
    res = {'ok': False, 'peak': round(float(np.max(np.abs(rec))), 3) if len(rec) else 0}
    if mx <= 1e-6:
        return res
    work = env.copy()
    excl = int(0.2 * fs)
    peaks = []
    while True:
        i = int(np.argmax(work))
        if work[i] < 0.3 * mx:
            break
        work[max(0, i - excl):i + excl] = 0
        d = parabolic(env[i - 1], env[i], env[i + 1]) if 0 < i < len(env) - 1 else 0.0
        t = (i + d) / fs
        if t < len(rec) / fs - 0.3:
            peaks.append((t, float(env[i]), float(c[i] / env[i])))
    peaks.sort()
    dts, lvls, rl, rr = [], [], [], []
    for a, b in zip(peaks, peaks[1:]):
        gap = b[0] - a[0]
        if abs(gap - LIVE_GAP) < 0.05:
            dts.append(gap - LIVE_GAP)
            lvls.append(20 * np.log10(a[1] / b[1]))
            rl.append(a[2])
            rr.append(b[2])
    res['pairs'] = len(dts)
    if not dts:
        return res
    dt = float(np.median(dts))
    res.update(ok=True, dt_ms=round(dt * 1000, 3), dist_cm=round(dt * C_SOUND * 100, 1),
               level_diff_db=round(float(np.median(lvls)), 2))
    pl, pr = float(np.median(rl)), float(np.median(rr))
    res['polarity'] = ('inverted' if pl * pr < 0 and min(abs(pl), abs(pr)) > 0.4 else
                       'same' if pl * pr > 0 and min(abs(pl), abs(pr)) > 0.4 else 'unclear')
    res['balanced'] = abs(res['dist_cm']) < 1.5 and abs(res['level_diff_db']) <= 0.7
    return res


# --------------------------------------------------------------- self test

def simulate(fs, d_left=480, d_right=495, g_left=1.0, g_right=0.8, invert_right=False, noise=0.002):
    sig = make_signal(fs).astype(np.float64)
    pre = int(1.6 * fs)
    n = pre + len(sig) + int(1.0 * fs)
    rng = np.random.default_rng(1)
    rec = rng.standard_normal(n) * noise

    for ch, (g, delay) in enumerate(((g_left, d_left), (g_right, d_right))):
        x = sig[:, ch] * g * (-1 if (ch == 1 and invert_right) else 1)
        # direct sound plus a few weaker reflections
        for dd, gg in ((delay, 1.0), (delay + 190, 0.35), (delay + 420, 0.25), (delay + 900, 0.15)):
            a = pre + dd
            rec[a:a + len(x)] += x * gg * 0.05
    return rec


def selftest():
    fs = 48000
    ok = True
    for kw, expect in (
        ({}, dict(dist=10.7, lvl=1.94, pol='same')),
        (dict(d_right=480, g_right=1.0), dict(dist=0.0, lvl=0.0, pol='same')),
        (dict(invert_right=True), dict(dist=10.7, lvl=1.94, pol='inverted')),
    ):
        r = analyze(simulate(fs, **kw), fs)
        print(json.dumps({k: r.get(k) for k in
                          ('dist_cm', 'level_diff_db', 'polarity', 'mono_sum_db', 'snr_db', 'warnings')}))
        for line in r.get('advice', []):
            print('   ', line)
        good = (r['ok'] and abs(r['dist_cm'] - expect['dist']) < 1.0
                and abs(r['level_diff_db'] - expect['lvl']) < 0.5 and r['polarity'] == expect['pol'])
        print('   ->', 'PASS' if good else 'FAIL')
        ok &= good
    # live mode: 4 s of the looping chirp pair, right speaker 15 samples later and quieter
    loop = make_live_signal(fs).astype(np.float64)
    sig = np.tile(loop, (5, 1))
    rng = np.random.default_rng(2)
    rec = rng.standard_normal(len(sig) + 2000) * 0.002
    rec[300:300 + len(sig)] += sig[:, 0] * 0.05
    rec[315:315 + len(sig)] += sig[:, 1] * 0.04
    r = live_analyze(rec[int(0.37 * fs):int(4.37 * fs)], fs)
    good = r['ok'] and abs(r['dist_cm'] - 10.7) < 1.0 and abs(r['level_diff_db'] - 1.94) < 0.3
    print(json.dumps(r), '-> LIVE', 'PASS' if good else 'FAIL')
    ok &= bool(good)
    return 0 if ok else 1


# ------------------------------------------------------------------ server

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Speaker Check</title>
<style>
:root{--bg:#f4f5f7;--fg:#16181d;--mut:#5d6573;--card:#fff;--line:#d9dde4;--acc:#1f5eff;--on:#fff;--ok:#12805c;--warn:#b25e00;--bad:#c4262e}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e8eaee;--mut:#9aa3b2;--card:#181b22;--line:#2b303b;--acc:#6b9bff;--on:#0f1115;--ok:#4cc79a;--warn:#f0a548;--bad:#ff6b73;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;padding:16px;padding-block:20px 40px}
main{max-width:34rem;margin:0 auto;display:flex;flex-direction:column;gap:16px}
h1{font-size:1.4rem;margin:0}
h2{font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:0 0 8px}
p,ul{margin:0}ul{padding-left:1.1rem}li+li{margin-top:4px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
button{font:inherit;font-weight:600;width:100%;padding:14px;border:0;border-radius:10px;background:var(--acc);color:var(--on)}
button.alt{background:transparent;color:var(--acc);border:1px solid var(--acc)}
button:disabled{opacity:.5}
.banner{font-weight:700;text-align:center;padding:12px;border-radius:10px;border:1px solid var(--line)}
.banner.good{color:var(--ok);border-color:var(--ok)}
.mut{color:var(--mut)}
.meter{height:8px;border-radius:4px;background:var(--line);overflow:hidden;margin-top:10px}
.meter i{display:block;height:100%;width:0;background:var(--acc);transition:width .08s}
.gauge{position:relative;height:10px;border-radius:5px;background:var(--line);margin:22px 0 4px}
.gauge::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:2px;background:var(--mut)}
.gauge b{position:absolute;top:-5px;width:6px;height:20px;border-radius:3px;margin-left:-3px;background:var(--acc)}
.ends{display:flex;justify-content:space-between;font-size:.8rem;color:var(--mut)}
.big{font-size:1.6rem;font-weight:700;font-variant-numeric:tabular-nums}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.9rem}
th,td{text-align:right;padding:5px 4px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:500}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
</style></head><body><main>
<h1>Speaker Check</h1>
<div class="card">
<h2>Before you start</h2>
<ul>
<li>Hold or prop the phone where your head sits, at ear height, mic toward the speakers.</li>
<li>Set the speakers to a normal listening volume. The room should be quiet.</li>
<li>The test takes about 20 seconds and plays loud chirps and noise bursts.</li>
</ul></div>
<div style="display:flex;flex-direction:column;gap:10px">
<button id="go">Start test</button>
<button id="live" class="alt">Live mode: adjust while it plays</button>
</div>
<div class="card" id="statusCard" hidden><p id="status"></p><div class="meter"><i id="lvl"></i></div></div>
<div id="out"></div>
</main>
<script>
const $=id=>document.getElementById(id);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const f1=x=>(x>0?'+':'')+x.toFixed(1);
function status(t){$('statusCard').hidden=false;$('status').textContent=t}
async function run(){
  $('go').disabled=true;$('live').disabled=true;$('out').innerHTML='';
  let stream,ctx;
  try{
    if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error('This page needs https. Open the https address printed on the PC.');
    status('Asking for the microphone...');
    stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false,channelCount:1}});
    ctx=new (window.AudioContext||window.webkitAudioContext)();await ctx.resume();
    const fs=ctx.sampleRate,src=ctx.createMediaStreamSource(stream),proc=ctx.createScriptProcessor(4096,1,1),mute=ctx.createGain();
    mute.gain.value=0;const chunks=[];
    proc.onaudioprocess=e=>{const d=e.inputBuffer.getChannelData(0);chunks.push(new Float32Array(d));
      let s=0;for(let i=0;i<d.length;i+=8)s+=d[i]*d[i];const rms=Math.sqrt(s/(d.length/8));
      $('lvl').style.width=clamp((20*Math.log10(rms+1e-9)+80)/80*100,0,100)+'%'};
    src.connect(proc);proc.connect(mute);mute.connect(ctx.destination);
    status('Measuring room noise. Stay quiet.');await sleep(1800);
    const info=await (await fetch('/play',{method:'POST'})).json();
    status('Playing the test signal ('+Math.round(info.duration)+' s). Stay still and quiet.');
    await sleep((info.duration+1.2)*1000);
    proc.disconnect();src.disconnect();stream.getTracks().forEach(t=>t.stop());
    let n=0;chunks.forEach(c=>n+=c.length);const all=new Float32Array(n);let o=0;chunks.forEach(c=>{all.set(c,o);o+=c.length});
    status('Analyzing...');
    const r=await fetch('/analyze?fs='+fs,{method:'POST',body:all.buffer});
    show(await r.json());$('statusCard').hidden=true;
  }catch(e){status('Problem: '+e.message)}
  finally{try{ctx&&ctx.close()}catch(_){}$('go').disabled=false;$('live').disabled=false;$('go').textContent='Run again'}
}
function gauge(title,val,range,unit,left,right,text){
  const pct=50-clamp(val/range,-1,1)*50;
  return '<div class="card"><h2>'+title+'</h2><div class="big">'+text+'</div><div class="gauge"><b style="left:'+pct+'%"></b></div>'
   +'<div class="ends"><span>'+left+'</span><span>centered</span><span>'+right+'</span></div></div>';
}
function show(r){
  const out=$('out');
  if(!r.ok){out.innerHTML='<div class="card bad">'+(r.error||'Analysis failed.')+'</div>'+warn(r);return}
  let h='<div class="card"><h2>What to do</h2><ul>'+r.advice.map(a=>'<li>'+a+'</li>').join('')+'</ul></div>';
  h+=warn(r);
  // image pulls toward the earlier / louder speaker: negative distance diff (left farther) pulls right
  h+=gauge('Arrival time',r.dist_cm,30,'cm','image pulls left','image pulls right',
    (Math.abs(r.dist_cm)<1.5?'Matched':Math.abs(r.dist_cm).toFixed(0)+' cm, '+(r.dist_cm>0?'right':'left')+' farther')
    +' <span class="mut" style="font-size:.9rem">('+f1(r.dt_ms)+' ms)</span>');
  h+=gauge('Level',r.level_diff_db,6,'dB','image pulls left','image pulls right',
    (Math.abs(r.level_diff_db)<=0.7?'Matched':Math.abs(r.level_diff_db).toFixed(1)+' dB, '+(r.level_diff_db>0?'left':'right')+' louder'));
  const pol=r.polarity==='inverted'?'<span class="bad">Inverted</span>':r.polarity==='same'?'<span class="ok">In phase</span>':'<span class="warn">Unclear</span>';
  h+='<div class="card"><h2>Polarity</h2><div class="row"><span class="big">'+pol+'</span><span class="mut">mono sum '+f1(r.mono_sum_db)+' dB at 125-500 Hz (about +6 expected)</span></div></div>';
  h+='<div class="card"><h2>By octave band</h2><div style="overflow-x:auto"><table><tr><th>Hz</th><th>L dB</th><th>R dB</th><th>L-R</th><th>Mono sum</th></tr>'
   +r.bands.map(b=>'<tr><td>'+b.hz+'</td><td>'+b.l.toFixed(1)+'</td><td>'+b.r.toFixed(1)+'</td><td>'+f1(b.diff)+'</td><td>'+f1(b.sum)+'</td></tr>').join('')
   +'</table></div><p class="mut" style="margin-top:8px;font-size:.85rem">Levels are relative, not calibrated SPL. '+r.rounds+' rounds, timing spread '+r.dt_spread_ms+' ms, signal '+r.snr_db+' dB over room noise.</p></div>';
  out.innerHTML=h;
}
function warn(r){return (r.warnings&&r.warnings.length)?'<div class="card warn"><h2>Check</h2><ul>'+r.warnings.map(w=>'<li>'+w+'</li>').join('')+'</ul></div>':''}
let liveStop=null;
async function live(){
  if(liveStop){liveStop();return}
  $('go').disabled=true;$('live').textContent='Stop live mode';$('out').innerHTML='';
  let stream,ctx,proc,src,go=true;const chunks=[];let total=0;
  const cleanup=async()=>{
    if(!go&&!liveStop)return;go=false;liveStop=null;
    try{await fetch('/live/stop',{method:'POST'})}catch(_){}
    try{proc&&proc.disconnect();src&&src.disconnect();stream&&stream.getTracks().forEach(t=>t.stop());ctx&&ctx.close()}catch(_){}
    $('go').disabled=false;$('live').textContent='Live mode: adjust while it plays';$('statusCard').hidden=true;
  };
  liveStop=cleanup;
  try{
    if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error('This page needs https. Open the https address printed on the PC.');
    status('Asking for the microphone...');
    stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false,channelCount:1}});
    ctx=new (window.AudioContext||window.webkitAudioContext)();await ctx.resume();
    const fs=ctx.sampleRate,need=Math.round(4*fs);
    src=ctx.createMediaStreamSource(stream);proc=ctx.createScriptProcessor(4096,1,1);const mute=ctx.createGain();mute.gain.value=0;
    proc.onaudioprocess=e=>{const d=e.inputBuffer.getChannelData(0);chunks.push(new Float32Array(d));total+=d.length;
      while(chunks.length>1&&total-chunks[0].length>=need)total-=chunks.shift().length;
      let s=0;for(let i=0;i<d.length;i+=8)s+=d[i]*d[i];
      $('lvl').style.width=clamp((20*Math.log10(Math.sqrt(s/(d.length/8))+1e-9)+80)/80*100,0,100)+'%'};
    src.connect(proc);proc.connect(mute);mute.connect(ctx.destination);
    await fetch('/live/start',{method:'POST'});
    status('Live: listening for the test sound...');
    while(go){
      await sleep(1000);
      if(!go||total<need*0.75)continue;
      const all=new Float32Array(total);let o=0;chunks.forEach(c=>{all.set(c,o);o+=c.length});
      const buf=all.slice(Math.max(0,total-need));
      const r=await (await fetch('/live?fs='+fs,{method:'POST',body:buf.buffer})).json();
      if(go)showLive(r);
    }
  }catch(e){status('Problem: '+e.message)}
  finally{await cleanup()}
}
function showLive(r){
  const out=$('out');
  if(!r.ok){
    status(r.peak<0.01?'Hearing almost nothing. Raise the speaker volume or move the phone closer.':'Hearing the room but not the test sound yet...');
    return;
  }
  status('Live: move a speaker and watch the markers. Updates every second.');
  let h='<div class="banner'+(r.balanced?' good':'')+'">'+(r.balanced?'Balanced':'Not balanced yet')+'</div>';
  h+=gauge('Arrival time',r.dist_cm,30,'cm','image pulls left','image pulls right',
    (Math.abs(r.dist_cm)<1.5?'Matched':Math.abs(r.dist_cm).toFixed(1)+' cm, '+(r.dist_cm>0?'right':'left')+' farther'));
  h+=gauge('Level',r.level_diff_db,6,'dB','image pulls left','image pulls right',
    (Math.abs(r.level_diff_db)<=0.7?'Matched':Math.abs(r.level_diff_db).toFixed(1)+' dB, '+(r.level_diff_db>0?'left':'right')+' louder'));
  if(r.polarity==='inverted')h+='<div class="card bad">One speaker looks wired out of phase. Swap its + and - wires.</div>';
  out.innerHTML=h;
}
$('live').addEventListener('click',live);
window.addEventListener('pagehide',()=>{try{navigator.sendBeacon('/live/stop')}catch(_){}});
$('go').addEventListener('click',run);
</script></body></html>
'''


class State:
    wav_path = None
    live_path = None
    duration = total_duration()
    lock = threading.Lock()
    live_on = False
    live_seen = 0.0


def start_playback():
    path = str(State.wav_path)
    if sys.platform == 'win32':
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    elif sys.platform == 'darwin':
        subprocess.Popen(['afplay', path])
    else:
        subprocess.Popen(['aplay', '-q', path])


def _live_loop_posix():
    cmd = ['afplay'] if sys.platform == 'darwin' else ['aplay', '-q']
    while State.live_on:
        subprocess.run(cmd + [str(State.live_path)])


def start_live():
    State.live_on = True
    State.live_seen = time.time()
    if sys.platform == 'win32':
        import winsound
        winsound.PlaySound(str(State.live_path),
                           winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
    else:
        threading.Thread(target=_live_loop_posix, daemon=True).start()


def stop_live():
    State.live_on = False
    if sys.platform == 'win32':
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)


def live_watchdog():
    """Stop the loop if the phone stops polling (page closed, phone asleep)."""
    while True:
        time.sleep(2)
        if State.live_on and time.time() - State.live_seen > 8:
            print('  live mode: phone stopped polling, sound off')
            stop_live()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), 'application/json')

    def do_GET(self):
        if urlparse(self.path).path == '/':
            self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
        else:
            self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == '/play':
            with State.lock:
                start_playback()
            print('  playing test signal')
            self._json({'duration': State.duration})
        elif u.path == '/live/start':
            with State.lock:
                start_live()
            print('  live mode on')
            self._json({'ok': True})
        elif u.path == '/live/stop':
            with State.lock:
                stop_live()
            print('  live mode off')
            self._json({'ok': True})
        elif u.path == '/live':
            n = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(n)
            State.live_seen = time.time()
            fs = int(parse_qs(u.query).get('fs', ['48000'])[0])
            try:
                res = live_analyze(np.frombuffer(body, dtype='<f4'), fs)
            except Exception as e:
                res = {'ok': False, 'error': str(e)}
            self._json(res)
        elif u.path == '/analyze':
            n = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(n)
            fs = int(parse_qs(u.query).get('fs', ['48000'])[0])
            rec = np.frombuffer(body, dtype='<f4')
            print('  analyzing %.1f s at %d Hz' % (len(rec) / fs, fs))
            try:
                res = analyze(rec, fs)
            except Exception as e:  # report to the phone instead of dropping the connection
                res = {'ok': False, 'error': 'Analysis failed: %s' % e, 'warnings': []}
            print('  ' + ' | '.join(res.get('advice', [res.get('error', '')])))
            self._json(res)
        else:
            self._send(404, b'not found', 'text/plain')


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except OSError:
        return '127.0.0.1'
    finally:
        s.close()


def find_openssl():
    exe = shutil.which('openssl')
    if exe:
        return exe
    for p in (r'C:\Program Files\Git\mingw64\bin\openssl.exe', r'C:\Program Files\Git\usr\bin\openssl.exe'):
        if os.path.exists(p):
            return p
    sys.exit('openssl was not found. Install it (it ships with Git for Windows) and retry.')


def ensure_cert(ip):
    d = Path.home() / '.speaker_check'
    d.mkdir(exist_ok=True)
    cert, key = d / ('cert_%s.pem' % ip), d / ('key_%s.pem' % ip)
    if not (cert.exists() and key.exists()):
        subprocess.run([find_openssl(), 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                        '-keyout', str(key), '-out', str(cert), '-days', '365',
                        '-subj', '/CN=speaker-check', '-addext', 'subjectAltName=IP:%s' % ip],
                       check=True, capture_output=True)
    return str(cert), str(key)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8443)
    ap.add_argument('--selftest', action='store_true', help='analyze a simulated recording and exit')
    ap.add_argument('--save-wav', metavar='PATH', help='write the test signal as a WAV file and exit')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    sig = make_signal(PLAY_FS)
    if a.save_wav:
        write_wav(a.save_wav, sig, PLAY_FS)
        print('wrote', a.save_wav)
        return
    State.wav_path = Path(tempfile.gettempdir()) / 'speaker_check_test.wav'
    write_wav(State.wav_path, sig, PLAY_FS)
    State.live_path = Path(tempfile.gettempdir()) / 'speaker_check_live.wav'
    write_wav(State.live_path, make_live_signal(PLAY_FS), PLAY_FS)
    threading.Thread(target=live_watchdog, daemon=True).start()
    ip = lan_ip()
    cert, key = ensure_cert(ip)
    srv = ThreadingHTTPServer(('0.0.0.0', a.port), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print('Open this on your phone (same Wi-Fi), accept the certificate warning:')
    print('\n    https://%s:%d\n' % (ip, a.port))
    print('If it does not load, allow Python through Windows Firewall on private networks. Ctrl+C to stop.')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
