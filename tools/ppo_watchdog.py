"""Keeps an unattended PPO run (train.py --engine cpp) alive and sane.

    python tools/ppo_watchdog.py checkpoints/ppo_cpp_gold2.json

The config file holds the trainer's arguments (a list, without --resume) and the thresholds. Every `every`
minutes the watchdog reads the run's log and acts on:

  dead / stalled   the log has not grown for `stall_minutes`            -> kill what is left, resume
  crawling         fewer than `min_sps` steps/s over `slow_checks` checks -> kill, resume
                   (a full GPU makes Windows page its memory: training carries on at a twentieth of the speed)
  collapsed        mean policy entropy of the last `window` rows below `min_entropy`, or NaN in the log
                   -> kill, put the best greedy checkpoint back (<out>_best_greedy.pt, kept by eval_watch.py),
                      resume with the learning rate halved

A resume restarts the trainer with --resume: same checkpoint file, same log, iteration count and schedules
carried on. At most `max_restarts` restarts; everything it does is written to <out>_watchdog.log.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULTS = {"every": 2.0, "stall_minutes": 6.0, "min_sps": 4000, "slow_checks": 3, "min_entropy": 0.35,
            "window": 150, "max_restarts": 8}


def say(path: str, msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def tail_rows(log: str, n: int) -> list[dict]:
    try:
        with open(log, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 700 * n))
            lines = f.read().decode(errors="ignore").splitlines()[1:]
    except OSError:
        return []
    rows = []
    for l in lines[-n:]:
        try:
            r = json.loads(l)
            if "iter" in r:
                rows.append(r)
        except ValueError:
            pass
    return rows


def kill_trainer(out: str):
    """Stop every trainer process writing to this run's checkpoint."""
    name = os.path.basename(out)
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match "
          f"'balatro_rl.train' -and $_.CommandLine -match '{name}' }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
    time.sleep(3)


def launch(args: list[str], out: str, lr_scale: float):
    cmd = [sys.executable, "-u", "-m", "balatro_rl.train"] + args + ["--resume", "--lr-scale", str(lr_scale)]
    flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0            # detached, own process group
    so = open(out.replace(".pt", ".out"), "a")
    se = open(out.replace(".pt", ".err"), "a")
    subprocess.Popen(cmd, cwd=ROOT, stdout=so, stderr=se, stdin=subprocess.DEVNULL, creationflags=flags)


def progress(log: str, window: int, lag: int):
    """(games so far, gain): the mean of the training blinds over the last `window` games minus the mean over
    the `window` games that ended `lag` games earlier; gain is None until the run is that long."""
    games, blinds = [], []
    try:
        with open(log, "rb") as f:
            for l in f:
                try:
                    r = json.loads(l)
                    games.append(r["games"])
                    blinds.append(r["blinds"])
                except (ValueError, KeyError):
                    pass
    except OSError:
        return 0, None
    if not games or games[-1] < lag + 2 * window:
        return (games[-1] if games else 0), None
    g = games[-1]

    def mean(lo, hi):
        v = [b for x, b in zip(games, blinds) if lo < x <= hi]
        return sum(v) / len(v) if v else None
    now, then = mean(g - window, g), mean(g - lag - window, g - lag)
    return g, (now - then if now is not None and then is not None else None)


def apply_stage(cfg: dict, stage: dict, cfg_path: str):
    """Change the trainer's arguments as the stage says ("set": replace or add option values) and write the
    config back, so every later restart keeps the change."""
    args = cfg["args"]
    for k, v in stage.get("set", {}).items():
        if k in args:
            args[args.index(k) + 1] = str(v)
        else:
            i = args.index("--iters")
            args[i:i] = [k, str(v)]
    cfg.setdefault("stages_applied", []).append({"name": stage["name"], "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(cfg_path, "w") as f:
        json.dump({k: v for k, v in cfg.items() if k not in DEFAULTS or cfg[k] != DEFAULTS[k] or k in ("every",)}, f, indent=1)


def wait_for_save(out: str, limit: float = 120.0):
    """Until the trainer writes its checkpoint again (so stopping it loses nothing)."""
    try:
        m = os.path.getmtime(out)
    except OSError:
        return
    t = time.time()
    while time.time() - t < limit:
        try:
            if os.path.getmtime(out) != m:
                time.sleep(2.0)
                return
        except OSError:
            pass
        time.sleep(0.2)


def main():
    cfg = dict(DEFAULTS)
    cfg.update(json.load(open(sys.argv[1])))
    args = cfg["args"]
    out = args[args.index("--out") + 1]
    log = out.replace(".pt", "_log.jsonl")
    wlog = out.replace(".pt", "_watchdog.log")
    lr_scale = float(cfg.get("lr_scale", 1.0))
    restarts = 0
    slow = 0
    last = None                                  # (time, env_steps) at the previous check
    launched = time.time()                       # when the trainer was last (re)started, as far as this knows
    say(wlog, f"watching {out}: every {cfg['every']} min, restart if stalled {cfg['stall_minutes']} min, "
              f"< {cfg['min_sps']} steps/s x{cfg['slow_checks']}, entropy < {cfg['min_entropy']}")
    while True:
        time.sleep(cfg["every"] * 60)
        rows = tail_rows(log, cfg["window"])
        now = time.time()
        # (a run that has not written its first row yet is as old as its launch, not infinitely old)
        age = (now - (os.path.getmtime(log) if os.path.exists(log) else launched)) / 60
        reason, restore = "", False
        if rows and rows[-1]["iter"] >= int(args[args.index("--iters") + 1]):
            say(wlog, "the run has finished")
            return
        if age > cfg["stall_minutes"]:
            reason = f"the log has not grown for {age:.1f} min (the trainer died or hung)"
        elif rows:
            steps = rows[-1]["env_steps"]
            if last is not None and steps >= last[1]:
                rate = (steps - last[1]) / max(now - last[0], 1e-9)
                slow = slow + 1 if rate < cfg["min_sps"] else 0
                if slow >= cfg["slow_checks"]:
                    reason = f"only {rate:,.0f} steps/s for {slow} checks"
            last = (now, steps)
            ents = [r["ent"] for r in rows if "ent" in r]
            bad = any(isinstance(r.get(k), float) and math.isnan(r[k]) for r in rows[-5:] for k in ("pi", "v", "ent", "kl"))
            if bad:
                reason, restore = "NaN in the losses", True
            elif len(ents) >= cfg["window"] // 2 and sum(ents) / len(ents) < cfg["min_entropy"]:
                reason, restore = f"policy entropy {sum(ents) / len(ents):.2f} (the policy has collapsed)", True
        if not reason:
            # staged changes when progress flattens: each stage once, in order (config "stages": name, min_games,
            # window, lag, min_gain, set); see progress()
            done_names = {s["name"] for s in cfg.get("stages_applied", [])}
            todo = [s for s in cfg.get("stages", []) if s["name"] not in done_names]
            if todo and rows:
                st = todo[0]
                g, gain = progress(log, int(st.get("window", 100000)), int(st.get("lag", 300000)))
                if g >= int(st.get("min_games", 0)) and gain is not None and gain < float(st["min_gain"]):
                    say(wlog, f"progress has flattened ({gain:+.2f} blinds over the last {st.get('lag', 300000):,} games, at "
                              f"{g:,} games): applying stage '{st['name']}': {st.get('set', {})}")
                    wait_for_save(out)
                    kill_trainer(out)
                    apply_stage(cfg, st, sys.argv[1])
                    launch(args, out, lr_scale)
                    slow, last = 0, None
                    time.sleep(180)
            continue
        if restarts >= cfg["max_restarts"]:
            say(wlog, f"{reason}; not restarting again ({restarts} restarts made)")
            return
        say(wlog, f"{reason}: restarting")
        kill_trainer(out)
        if restore:
            best = out.replace(".pt", "_best_greedy.pt")
            if os.path.exists(best):
                shutil.copy(out, out.replace(".pt", f"_collapsed{restarts + 1}.pt"))
                shutil.copy(best, out)
                say(wlog, f"restored {os.path.basename(best)}")
            lr_scale *= 0.5
            say(wlog, f"learning rate scale now {lr_scale}")
        launch(args, out, lr_scale)
        launched = time.time()
        restarts += 1
        slow, last = 0, None
        time.sleep(180)                           # let it load and write a few rows before judging it


if __name__ == "__main__":
    main()
