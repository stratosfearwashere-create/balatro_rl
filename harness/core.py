"""Shared state: configuration, the experiment registry and the budget ledger.

State lives in experiments/ (gitignored): registry.json, one folder per experiment, and the
agent's log. HARNESS_STATE and HARNESS_CONFIG override the locations (used by the harness's own
tests so they never touch real state)."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import time
from pathlib import Path

import yaml

# The main repository. The agent runs from its own worktree; the driver sets HARNESS_MAIN so state,
# config, notebook, checkpoints and the evaluation always come from the main repository.
ROOT = Path(os.environ.get("HARNESS_MAIN", Path(__file__).resolve().parents[1]))
STATE = Path(os.environ.get("HARNESS_STATE", ROOT / "experiments"))
CONFIG_PATH = Path(os.environ.get("HARNESS_CONFIG", ROOT / "harness" / "config.yaml"))


def config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def today() -> str:
    return dt.date.today().isoformat()


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


@contextlib.contextmanager
def locked():
    """Exclusive lock on the state folder (the daemon and CLI calls may run at the same time)."""
    STATE.mkdir(parents=True, exist_ok=True)
    lock = STATE / ".lock"
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > 120:      # stale lock from a crashed process
                lock.unlink(missing_ok=True)
            time.sleep(0.1)
    else:
        raise RuntimeError(f"state is locked: {lock}")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _registry_file() -> Path:
    return STATE / "registry.json"


def load_registry() -> dict:
    f = _registry_file()
    if not f.exists():
        return {"experiments": {}, "ledger": {}, "next_id": 1}
    return json.loads(f.read_text())


def save_registry(reg: dict):
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = _registry_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1))
    os.replace(tmp, _registry_file())


def update(fn):
    """Apply fn(registry) under the lock and save. Returns fn's result."""
    with locked():
        reg = load_registry()
        out = fn(reg)
        save_registry(reg)
        return out


def exp_dir(exp_id: str) -> Path:
    return STATE / exp_id


def reference() -> str | None:
    """The commit every branch is checked against, pinned by the human (`python -m harness pin`).
    Pinned rather than read from the live main branch, so a commit to main can't move the goalposts."""
    f = STATE / "reference.json"
    return json.loads(f.read_text())["commit"] if f.exists() else None


def pin_reference(commit: str):
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "reference.json").write_text(json.dumps({"commit": commit, "pinned": now()}))


def minutes_used(reg: dict, day: str | None = None) -> float:
    return float(reg["ledger"].get(day or today(), 0.0))


def add_minutes(reg: dict, minutes: float, day: str | None = None):
    day = day or today()
    reg["ledger"][day] = round(reg["ledger"].get(day, 0.0) + minutes, 2)


def minutes_committed(reg: dict) -> float:
    """Budget already promised to queued or running experiments (not yet spent)."""
    out = 0.0
    for e in reg["experiments"].values():
        if e["status"] in ("queued", "running"):
            out += e["budget_minutes"] - sum(r.get("minutes", 0.0) for r in e.get("runs", []))
    return out
