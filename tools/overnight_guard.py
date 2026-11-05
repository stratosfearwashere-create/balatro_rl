"""Keeps an unattended --ladder run going (python tools/overnight_guard.py --config checkpoints/ppo_white_win1.json).

  * restarts the trainer from its last checkpoint (--resume) if it died or its log has not grown for --stall-minutes (at most --max-restarts);
  * copies the checkpoint aside every time the training stake advances (<save-dir>/stage_<STAKE>_u<iter>.pt) and keeps rolling snapshots every
    --snap-every updates (the last --keep-snaps), each with the training win rate it came from;
  * stops the trainer (and does not restart it) if the log shows a NaN / inf;
  * a collapse (the training win rate on the current stake --collapse-points under its best of the last --collapse-window updates, for
    --collapse-hold updates in a row, not counting the first --grace updates after a stake change) puts the snapshot with the best win rate
    back as the checkpoint and restarts from it (at most --max-rollbacks).
Everything it does is written to <save-dir>/guard.log. It never touches anything but the run's own files.
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time


def log(path, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60).stdout


def trainer_pids(out):
    name = os.path.basename(out)
    txt = ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*balatro_rl.train*' -and "
             "$_.CommandLine -like '*%s*' } | ForEach-Object { $_.ProcessId }" % name)          # (python.exe only: not this query itself, not the cmd wrapper)
    return [int(x) for x in txt.split() if x.strip().isdigit()]


def kill(pids):
    for p in pids:
        ps("Stop-Process -Id %d -Force -ErrorAction SilentlyContinue" % p)


def launch(args, root, out):
    cmd = ["python", "-u", "-m", "balatro_rl.train"] + args
    base = os.path.splitext(out)[0]
    o, e = open(base + ".out", "a"), open(base + ".err", "a")
    subprocess.Popen(cmd, cwd=root, stdout=o, stderr=e, creationflags=0x00000008 | 0x00000200)      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def read_rows(path, last=2500):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()[-last:]
    for l in lines:
        l = l.strip()
        if l.startswith("{"):
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--save-dir", default="checkpoints/saved/white_ladder")
    p.add_argument("--every", type=int, default=90, help="seconds between checks")
    p.add_argument("--stall-minutes", type=float, default=10.0)
    p.add_argument("--max-restarts", type=int, default=6)
    p.add_argument("--snap-every", type=int, default=100)
    p.add_argument("--keep-snaps", type=int, default=6)
    p.add_argument("--collapse-points", type=float, default=10.0)
    p.add_argument("--collapse-window", type=int, default=400)
    p.add_argument("--collapse-hold", type=int, default=25)
    p.add_argument("--grace", type=int, default=60)
    p.add_argument("--max-rollbacks", type=int, default=3)
    p.add_argument("--gentler", type=float, default=0.7, help="after a collapse the learning rate and the stall bump (and its cap) are multiplied by this")
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry", action="store_true", help="decide and log, do not start or stop anything")
    a = p.parse_args()
    root = os.getcwd()
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    args = list(cfg["args"])
    out = args[args.index("--out") + 1]
    logp = out.replace(".pt", "_log.jsonl")
    os.makedirs(a.save_dir, exist_ok=True)
    glog = os.path.join(a.save_dir, "guard.log")
    if "--resume" not in args:
        args.append("--resume")
    restarts = rollbacks = 0
    floor_iter = 0                                              # after a rollback only the updates after it count towards a collapse
    last_stake, last_snap_iter, snaps = None, -1, []          # snaps: (iter, win%, path)
    log(glog, f"guard on {out}: restarts<= {a.max_restarts}, stall {a.stall_minutes} min" + (" (dry)" if a.dry else ""))
    while True:
        rows = read_rows(logp)
        pids = trainer_pids(out)
        age = (time.time() - os.path.getmtime(logp)) / 60.0 if os.path.exists(logp) else 1e9
        last = rows[-1] if rows else None
        # --- NaN / inf: stop for good
        if last and any(isinstance(last.get(k), float) and not math.isfinite(last[k]) for k in ("pi", "v", "ent", "kl")):
            log(glog, f"NaN/inf in the log at update {last.get('iter')}: stopping the trainer and the guard")
            if not a.dry:
                kill(pids)
            return
        # --- a stake advance: keep the checkpoint
        if last and last.get("ladder_stake") and last["ladder_stake"] != last_stake:
            if last_stake is not None and os.path.exists(out):
                dst = os.path.join(a.save_dir, f"stage_{last_stake}_to_{last['ladder_stake']}_u{last.get('iter')}.pt")
                shutil.copy2(out, dst)
                log(glog, f"stake {last_stake} -> {last['ladder_stake']} at update {last.get('iter')}: checkpoint kept as {os.path.basename(dst)}")
            last_stake = last["ladder_stake"]
        # --- rolling snapshots
        if last and last.get("iter", 0) // a.snap_every > last_snap_iter // a.snap_every and os.path.exists(out) and a.snap_every > 0:
            recent = [r["win%"] for r in rows[-20:] if r.get("win%") is not None]
            wr = sum(recent) / len(recent) if recent else 0.0
            dst = os.path.join(a.save_dir, f"snap_u{last['iter']}.pt")
            shutil.copy2(out, dst)
            snaps.append((last["iter"], wr, dst))
            while len(snaps) > a.keep_snaps:
                old = snaps.pop(0)
                if os.path.exists(old[2]):
                    os.remove(old[2])
            last_snap_iter = last["iter"]
            log(glog, f"snapshot at update {last['iter']} (win rate {wr:.1f}% on {last.get('ladder_stake')})")
        # --- collapse: the win rate on the current stake far under its recent best
        collapsed = False
        if last and last.get("ladder_stake"):
            st = last["ladder_stake"]
            cur_rows = [r for r in rows if r.get("ladder_stake") == st and r.get("win%") is not None]
            cur_rows = [r for r in cur_rows[a.grace:] if r.get("iter", 0) > floor_iter]
            win = cur_rows[-a.collapse_window:]
            if len(win) > a.collapse_hold + 20:
                best_i = max(range(len(win)), key=lambda i: sum(x["win%"] for x in win[max(0, i - 19):i + 1]) / len(win[max(0, i - 19):i + 1]))
                best = sum(x["win%"] for x in win[max(0, best_i - 19):best_i + 1]) / len(win[max(0, best_i - 19):best_i + 1])
                tail = win[-a.collapse_hold:]
                collapsed = all(x["win%"] < best - a.collapse_points for x in tail) and len(win) - best_i > a.collapse_hold
        # --- decide
        reason = None
        if collapsed and rollbacks < a.max_rollbacks and snaps:
            reason = "collapse"
        elif not pids:
            reason = "dead"
        elif age > a.stall_minutes:
            reason = f"stalled ({age:.0f} min without a log row)"
        if reason and not a.dry:
            if reason == "collapse":
                good = max(snaps, key=lambda s: s[1])
                log(glog, f"COLLAPSE at update {last.get('iter')} (win rate under its best by > {a.collapse_points} points): rolling back to the snapshot of update "
                          f"{good[0]} ({good[1]:.1f}%)")
                kill(pids)
                time.sleep(4)
                shutil.copy2(good[2], out)
                rollbacks += 1
                floor_iter = last.get("iter", 0)
                for flag in ("--lr", "--stall-ent-bump", "--stall-ent-cap"):          # a gentler run from the snapshot
                    if flag in args:
                        i = args.index(flag)
                        args[i + 1] = "%g" % (float(args[i + 1]) * a.gentler)
                cfg["args"] = [x for x in args if x != "--stall-now"]
                json.dump(cfg, open(a.config, "w", encoding="utf-8"), indent=1)
                log(glog, "settings now: " + " ".join(f"{f} {args[args.index(f) + 1]}" for f in ("--lr", "--stall-ent-bump", "--stall-ent-cap") if f in args))
                snaps.clear()
                launch(args + ["--stall-now"] if "--stall-ent" in args else args, root, out)
            elif restarts < a.max_restarts:
                log(glog, f"trainer {reason} at update {last.get('iter') if last else '?'}: restarting from the checkpoint (restart {restarts + 1}/{a.max_restarts})")
                kill(pids)
                time.sleep(4)
                restarts += 1
                launch(args, root, out)
            else:
                log(glog, f"trainer {reason} and the restart limit is reached: the guard stops")
                return
            time.sleep(120)                                    # let it get going before the next look
        elif reason:
            log(glog, f"(dry) would act: {reason}")
        if a.once:
            log(glog, f"status: update {last.get('iter') if last else None}, stake {last.get('ladder_stake') if last else None}, trainer pids {pids}, "
                      f"log age {age:.1f} min, snaps {[(s[0], round(s[1], 1)) for s in snaps]}")
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
