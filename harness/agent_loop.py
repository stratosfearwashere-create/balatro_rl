"""The research agent's workspace and two manual tools. The daemon wakes the agent itself (scheduler.py);
these are for the human:

    python -m harness agent --dry-run     # show the brief the next wake-up would get (or that none is due)
    python -m harness agent --once        # wake the agent now for pending events (daily caps still apply)
"""
from __future__ import annotations

import glob
import shutil
import time

from .core import ROOT, STATE, now, reference
from .workspace import git

WORKSPACE = STATE / "agent_workspace"


def prepare_workspace(ref: str):
    """The agent's worktree, reset to the pinned reference commit (detached: it can't commit to main)."""
    if not WORKSPACE.exists():
        git("worktree", "add", "--detach", str(WORKSPACE), ref)
        for f in glob.glob(str(ROOT / "balatro_rl" / "sim" / "_fastscore*.pyd")):
            shutil.copy2(f, WORKSPACE / "balatro_rl" / "sim")
    if git("status", "--porcelain", cwd=WORKSPACE).strip():
        git("stash", "push", "--include-untracked", "-m", f"agent leftovers {now()}", cwd=WORKSPACE)
    git("checkout", "--detach", ref, cwd=WORKSPACE)


def run(once: bool = False, dry_run: bool = False):
    from .scheduler import Scheduler, build_brief
    if reference() is None:
        print("no reference commit pinned: review the main branch, then run `python -m harness pin`")
        return
    sched = Scheduler()
    pending = sched._load()["pending"]
    if dry_run or not once:
        print(build_brief(pending) if pending else "no pending events: no wake-up would start")
        return
    if not pending:
        print("no pending events: nothing to wake the agent for")
        return
    if not sched.tick(ignore_gap=True):
        print(f"no wake-up started: see {sched.log_file}")
        return
    print(f"{now()} wake-up started; waiting for it to finish")
    while sched._load()["session"] is not None:
        time.sleep(10)
        sched.tick()
    print(f"{now()} finished; see {sched.runs_file}")
