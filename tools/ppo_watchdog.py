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
    say(wlog, f"watching {out}: every {cfg['every']} min, restart if stalled {cfg['stall_minutes']} min, "
              f"< {cfg['min_sps']} steps/s x{cfg['slow_checks']}, entropy < {cfg['min_entropy']}")
    while True:
        time.sleep(cfg["every"] * 60)
        rows = tail_rows(log, cfg["window"])
        now = time.time()
        age = (now - os.path.getmtime(log)) / 60 if os.path.exists(log) else 1e9
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
        restarts += 1
        slow, last = 0, None
        time.sleep(180)                           # let it load and write a few rows before judging it


if __name__ == "__main__":
    main()
