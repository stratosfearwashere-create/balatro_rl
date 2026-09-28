"""Runs queued experiments one at a time (the machine fits one training job), with hard budget kills."""
from __future__ import annotations

import datetime as dt
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from .core import (ROOT, STATE, add_minutes, config, exp_dir, load_registry, locked, minutes_used, now, repo_path,
                   update)
from .scheduler import Scheduler
from .workspace import create_worktree, python_env

SCHED: Scheduler | None = None          # set by run(); wake-ups are checked while the daemon waits


def _tick():
    if SCHED is not None:
        try:
            SCHED.tick()
        except Exception as e:                             # noqa: BLE001  never let the agent stop the queue
            SCHED.log(f"scheduler error: {e!r}")


def _emit(kind: str, **info):
    if SCHED is not None:
        SCHED.emit(kind, **info)


def kill_tree(pid: int):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def train_commands(exp: dict, seed: int, out: Path) -> list[list[str]]:
    """The train.py invocations for one seed (behaviour cloning first when training from scratch)."""
    cfg = config()
    spec, tier = exp["spec"], cfg["tiers"][exp["tier"]]
    ov = dict(spec.get("overrides") or {})
    envs, steps = int(ov.get("envs", 16)), int(ov.get("steps", 64))
    iters = max(1, math.ceil(int(exp["env_steps"]) / (envs * steps)))
    common = ["--stake", tier["stake"], "--win-ante", str(tier["win_ante"]), "--seed", str(seed)]
    if tier.get("joker_pool"):
        common += ["--joker-pool-file", str(repo_path(tier["joker_pool"]))]
    ppo = [sys.executable, "-m", "balatro_rl.train", "ppo", *common, "--iters", str(iters),
           "--envs", str(envs), "--steps", str(steps), "--out", str(out / "model.pt")]
    for k, v in cfg["locked_hyperparameters"].items():
        if v is not None:
            ppo += [f"--{k}", str(v)]
    for k, v in ov.items():
        if k in ("envs", "steps"):
            continue
        flag = "--" + k.replace("_", "-")
        if k == "strategic":
            if v:
                ppo += ["--strategic", "--tactical", str(repo_path(cfg["tactical_checkpoint"]))]
        elif k == "pipeline":
            if v:
                ppo.append("--pipeline")
        else:
            ppo += [flag, str(v)]
    cmds = []
    init = spec.get("init", "baseline")
    if init == "scratch":
        bc_out = out / "bc.pt"
        cmds.append([sys.executable, "-m", "balatro_rl.train", "bc", *common, "--iters", "150", "--envs", "8",
                     "--out", str(bc_out)])
        ppo += ["--init", str(bc_out)]
    elif init == "baseline":
        ppo += ["--init", str(repo_path(cfg["baseline_checkpoint"]))]
    else:                                                  # experiment:<id>  (its seed-1 checkpoint)
        src = init.split(":", 1)[1]
        ppo += ["--init", str(exp_dir(src) / "seed1" / "model.pt")]
    cmds.append(ppo)
    return cmds


def _wait_for_day_budget(exp_id: str):
    """Sleep until tomorrow if today's training budget is spent."""
    cfg = config()
    while True:
        with locked():
            reg = load_registry()
            left = cfg["budgets"]["day_minutes"] - minutes_used(reg)
            if reg["experiments"][exp_id].get("kill_requested"):
                return 0.0
        if left > 1:
            return left
        update(lambda r: r["experiments"][exp_id].update(status="waiting (daily budget)"))
        tomorrow = dt.datetime.combine(dt.date.today() + dt.timedelta(days=1), dt.time(0, 1))
        time.sleep(max(60, (tomorrow - dt.datetime.now()).total_seconds()))


def _run(cmd, cwd, env, log_path: Path, cap_minutes: float, exp_id: str, seed: int) -> tuple[str, float]:
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, **kw)
        update(lambda r: r["experiments"][exp_id]["runs"][-1].update(pid=proc.pid))
        status = "done"
        while proc.poll() is None:
            time.sleep(5)
            _tick()
            mins = (time.time() - t0) / 60
            reg = load_registry()
            if reg["experiments"][exp_id].get("kill_requested"):
                kill_tree(proc.pid)
                status = "killed (requested)"
            elif mins > cap_minutes:
                kill_tree(proc.pid)
                status = "stopped (budget)"
        proc.wait()
        if status == "done" and proc.returncode != 0:
            status = f"failed (exit {proc.returncode})"
    return status, (time.time() - t0) / 60


def evaluate(exp: dict, seed: int, ckpt: Path, out_json: Path, code: Path) -> tuple[bool, str]:
    cfg = config()
    tier = cfg["tiers"][exp["tier"]]
    cmd = [sys.executable, str(ROOT / "eval" / "run_eval.py"), "--code", str(code), "--checkpoint", str(ckpt),
           "--seeds", str(repo_path(tier["eval_seeds"])), "--games", str(tier["eval_games"]),
           "--stake", tier["stake"], "--win-ante", str(tier["win_ante"]),
           "--max-minutes", str(cfg["budgets"]["eval_minutes"]), "--out", str(out_json)]
    if tier.get("joker_pool"):
        cmd += ["--joker-pool", str(repo_path(tier["joker_pool"]))]
    if (exp["spec"].get("overrides") or {}).get("strategic"):
        cmd += ["--tactical", str(repo_path(cfg["tactical_checkpoint"]))]
    r = subprocess.run(cmd, cwd=ROOT, env=python_env(ROOT), capture_output=True, text=True,
                       timeout=60 * (cfg["budgets"]["eval_minutes"] + 5))
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else ""


def run_experiment(exp_id: str):
    reg = load_registry()
    exp = reg["experiments"][exp_id]
    update(lambda r: r["experiments"][exp_id].update(status="running", started=now()))
    if not any(e["status"] == "queued" for k, e in reg["experiments"].items() if k != exp_id):
        _emit("queue_low", id=exp_id)                      # the GPU goes idle when this one ends
    try:
        code = create_worktree(exp_id, exp["commit"])
    except Exception as e:                                 # noqa: BLE001
        update(lambda r: r["experiments"][exp_id].update(status="failed", error=f"worktree: {e}", ended=now()))
        return
    seeds = exp["seeds"]
    used = 0.0
    for i, seed in enumerate(seeds):
        day_left = _wait_for_day_budget(exp_id)
        if load_registry()["experiments"][exp_id].get("kill_requested"):
            break
        cap = min((exp["budget_minutes"] - used) / (len(seeds) - i), day_left)
        out = exp_dir(exp_id) / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        def start(r):
            r["experiments"][exp_id]["status"] = "running"
            r["experiments"][exp_id]["runs"].append({"seed": seed, "status": "running", "started": now()})
        update(start)
        status, mins = "done", 0.0
        for cmd in train_commands(exp, seed, out):
            st, m = _run(cmd, code, python_env(code), out / "train.log", cap - mins, exp_id, seed)
            mins += m
            if st != "done":
                status = st
                break
        used += mins

        def record(r):
            add_minutes(r, mins)
            r["experiments"][exp_id]["runs"][-1].update(status=status, minutes=round(mins, 2), ended=now())
        update(record)
        if status != "done" and not status.startswith("killed"):
            log_lines = (out / "train.log").read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(log_lines[-20:])
            _emit("run_problem", id=exp_id, seed=seed, status=status, log_tail=tail)
        ckpt = out / "model.pt"
        if ckpt.exists():
            ok, msg = evaluate(exp, seed, ckpt, out / "eval.json", code)
            update(lambda r: r["experiments"][exp_id]["runs"][-1].update(eval=("ok: " if ok else "failed: ") + msg))
    reg = load_registry()
    runs = reg["experiments"][exp_id]["runs"]
    evaluated = sum(1 for r in runs if str(r.get("eval", "")).startswith("ok"))
    final = "killed" if reg["experiments"][exp_id].get("kill_requested") else ("done" if evaluated else "failed")
    update(lambda r: r["experiments"][exp_id].update(status=final, ended=now()))
    if final != "killed" and not (exp.get("full_of") and not exp.get("promoted_from")):
        _emit("experiment_finished", id=exp_id, status=final)
    if final == "done" and exp.get("promoted_from"):
        from .report import write_report
        write_report(exp_id)


def next_queued() -> str | None:
    reg = load_registry()
    queued = [(e["created"], k) for k, e in reg["experiments"].items() if e["status"] == "queued"]
    return min(queued)[1] if queued else None


def run(once: bool = False, poll: float = 15, agent: bool = True):
    """Run the queue. With agent=True the daemon also wakes the research agent (see scheduler.py)."""
    global SCHED
    SCHED = Scheduler() if agent else None
    print(f"harness daemon started {now()} (state in {STATE}; agent wake-ups {'on' if agent else 'off'})", flush=True)
    idle_reported = False
    while True:
        exp_id = next_queued()
        if exp_id:
            idle_reported = False
            print(f"{now()} running {exp_id}", flush=True)
            run_experiment(exp_id)
            print(f"{now()} finished {exp_id}: {load_registry()['experiments'][exp_id]['status']}", flush=True)
            if once:
                return
        elif once:
            return
        else:
            if not idle_reported:
                _emit("queue_empty")
                idle_reported = True
            _tick()
            time.sleep(poll)
