"""Drives the research agent with headless Claude Code. Started by the human, never by the agent:

    python -m harness agent --dry-run     # show the next command without running it
    python -m harness agent --once        # one iteration (propose or analyse), then stop
    python -m harness agent               # keep going within the daily caps

Each wake-up is a fresh `claude -p` session (state lives in the notebook and the registry), run in the
agent's own worktree with harness/claude_settings.json (dontAsk: anything not allowed is denied)."""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import time

from .core import CONFIG_PATH, ROOT, STATE, config, load_registry, now, reference, today
from .workspace import git

WORKSPACE = STATE / "agent_workspace"
LOG = STATE / "agent_log.jsonl"
FINISHED = ("done", "failed", "killed")
ACTIVE = ("queued", "running", "waiting (daily budget)")


def logged_ids() -> set[str]:
    from .notebook import JOURNAL as j
    return set(re.findall(r"exp-\d{4}", j.read_text(encoding="utf-8"))) if j.exists() else set()


def next_phase() -> tuple[str, list[str]]:
    reg = load_registry()
    logged = logged_ids()
    todo = [k for k, e in sorted(reg["experiments"].items())
            if e["status"] in FINISHED and k not in logged and not (e.get("full_of") and not e.get("promoted_from"))]
    if todo:
        return "analyse", todo
    if any(e["status"] in ACTIVE for e in reg["experiments"].values()):
        return "wait", []
    return "propose", []


def wakeups_today() -> int:
    if not LOG.exists():
        return 0
    return sum(1 for l in LOG.read_text().splitlines() if l.startswith("{") and json.loads(l)["time"][:10] == today())


def prepare_workspace(ref: str):
    """The agent's worktree, reset to the pinned reference commit (detached: it can't commit to main)."""
    if not WORKSPACE.exists():
        git("worktree", "add", "--detach", str(WORKSPACE), ref)
        for f in glob.glob(str(ROOT / "balatro_rl" / "sim" / "_fastscore*.pyd")):
            shutil.copy2(f, WORKSPACE / "balatro_rl" / "sim")
    if git("status", "--porcelain", cwd=WORKSPACE).strip():
        git("stash", "push", "--include-untracked", "-m", f"agent leftovers {now()}", cwd=WORKSPACE)
    git("checkout", "--detach", ref, cwd=WORKSPACE)


def command(phase: str, ids: list[str]) -> list[str]:
    cfg = config()
    a = cfg["agent"]
    if phase == "propose":
        task = ("Phase: PROPOSE. None of your experiments is queued or running. Follow harness/PROTOCOL.md: read the "
                "notebook summary and status, choose ONE hypothesis, implement it on a new agent/ branch, run "
                "`python -m harness check`, launch the cheap run, then stop.")
        model = a["model"]
    else:
        task = (f"Phase: ANALYSE. These experiments finished and have no notebook entry yet: {', '.join(ids)}. "
                "Follow harness/PROTOCOL.md: read metrics, failures and the paired comparison, write one notebook entry "
                "per hypothesis, promote only if the rule passes, then stop.")
        model = a["analyse_model"]
    return ["claude", "-p", task, "--model", model, "--permission-mode", "dontAsk",
            "--settings", str(ROOT / "harness" / "claude_settings.json"),
            "--append-system-prompt-file", str(ROOT / "harness" / "PROTOCOL.md"),
            "--max-turns", str(a["max_turns"]), "--max-budget-usd", str(cfg["budgets"]["agent_usd_per_wakeup"]),
            "--output-format", "json"]


def run(once: bool = False, dry_run: bool = False):
    cfg = config()
    ref = reference()
    if ref is None:
        print("no reference commit pinned: review the main branch, then run `python -m harness pin`")
        return
    while True:
        if wakeups_today() >= cfg["budgets"]["agent_wakeups_per_day"]:
            print(f"{now()} daily wake-up cap reached")
            if once or dry_run:
                return
            tomorrow = dt.datetime.combine(dt.date.today() + dt.timedelta(days=1), dt.time(0, 5))
            time.sleep(max(60, (tomorrow - dt.datetime.now()).total_seconds()))
            continue
        phase, ids = next_phase()
        if phase == "wait":
            if once or dry_run:
                print(f"{now()} nothing to do: an experiment is queued or running")
                return
            time.sleep(300)
            continue
        cmd = command(phase, ids)
        env = dict(os.environ, HARNESS_MAIN=str(ROOT), HARNESS_STATE=str(STATE), HARNESS_CONFIG=str(CONFIG_PATH))
        if dry_run:
            print(f"next phase: {phase} {ids}\nworking directory: {WORKSPACE}\n" + " ".join(
                f'"{c}"' if " " in c else c for c in cmd))
            return
        prepare_workspace(ref)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=WORKSPACE, env=env, capture_output=True, text=True, timeout=4 * 3600)
        try:
            res = json.loads(r.stdout)
        except json.JSONDecodeError:
            res = {"result": (r.stdout + r.stderr)[-2000:], "is_error": True}
        entry = {"time": now(), "phase": phase, "ids": ids, "minutes": round((time.time() - t0) / 60, 1),
                 "cost_usd": res.get("total_cost_usd"), "turns": res.get("num_turns"),
                 "is_error": res.get("is_error", r.returncode != 0), "result": str(res.get("result", ""))[-1500:]}
        STATE.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"{now()} {phase} finished: cost ${entry['cost_usd']}, {entry['turns']} turns, error {entry['is_error']}")
        if re.search(r"hit your (session|weekly|opus|sonnet) limit|usage limit", entry["result"], re.I):
            print(f"{now()} usage limit reached; waiting an hour")
            if once:
                return
            time.sleep(3600)
        if once:
            return
