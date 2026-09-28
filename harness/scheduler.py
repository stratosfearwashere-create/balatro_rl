"""When to wake the research agent. Runs inside the daemon: all status checking happens here, and the
LLM is only started when an event is waiting and every limit allows it.

Events: an experiment finished; the queue is about to run dry (its last experiment started) or is
empty; a run crashed or was stopped by its budget; the daily summary time passed.
Limits (config `agent:`): wake-ups per day, minimum minutes between wake-ups (events arriving inside
that window are batched into the next wake-up), dollars per wake-up (claude --max-budget-usd) and per
day. A limit that blocks a wake-up is logged; the queue keeps running and the events wait.

Plan only: claude runs on the subscription login (API-key variables are removed from its environment),
so the dollar figures are list-price estimates of plan usage, used as brakes. When Claude reports that a
plan limit is reached, wake-ups pause for `limit_pause_minutes` instead of retrying.

State: experiments/scheduler.json (pending events, last wake-up, running session) and
experiments/scheduler.log. Each wake-up is also a line in experiments/agent_log.jsonl."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import core

WAKE_LIMIT_TEXT = re.compile(r"hit your (session|weekly|opus|sonnet) limit|usage limit", re.I)


class Scheduler:
    def __init__(self, state_dir: Path | None = None, clock=time.time, launcher=None, brief_builder=None,
                 settings: dict | None = None):
        self.state_dir = Path(state_dir or core.STATE)
        self.clock = clock
        self.launcher = launcher or self._launch_claude          # (brief, model) -> handle
        self.brief_builder = brief_builder or build_brief        # (events) -> str
        self._settings = settings
        self.file = self.state_dir / "scheduler.json"
        self.log_file = self.state_dir / "scheduler.log"
        self.runs_file = self.state_dir / "agent_log.jsonl"
        self.handle = None

    # ------------------------------------------------------------------ settings and state
    @property
    def s(self) -> dict:
        return self._settings if self._settings is not None else core.config()["agent"]

    def _load(self) -> dict:
        if self.file.exists():
            return json.loads(self.file.read_text())
        return {"pending": [], "last_wake": 0.0, "last_daily": "", "session": None, "blocked_logged": "",
                "paused_until": 0.0}

    def _save(self, st: dict):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        os.replace(tmp, self.file)

    def _today(self) -> str:
        return dt.datetime.fromtimestamp(self.clock()).date().isoformat()

    def log(self, msg: str):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.fromtimestamp(self.clock()).isoformat(timespec="seconds")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{stamp} {msg}\n")

    def _runs_today(self) -> list[dict]:
        if not self.runs_file.exists():
            return []
        day = self._today()
        out = []
        for line in self.runs_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                if r.get("day") == day:
                    out.append(r)
        return out

    # ------------------------------------------------------------------ events
    def emit(self, kind: str, **info):
        """Record an event. Identical events (same kind and experiment) already pending are not repeated."""
        st = self._load()
        ev = {"kind": kind, "time": self.clock(), **info}
        key = (kind, info.get("id"), info.get("seed"))
        if not any((e["kind"], e.get("id"), e.get("seed")) == key for e in st["pending"]):
            st["pending"].append(ev)
            self._save(st)
            self.log(f"event {kind} {info.get('id', '')}".rstrip())

    # ------------------------------------------------------------------ the tick
    def tick(self, ignore_gap: bool = False) -> bool:
        """Called often by the daemon. Starts a wake-up only if something is pending and the limits allow.
        Returns True if a wake-up started. ignore_gap (manual `agent --once`) skips only the minimum gap."""
        st = self._load()
        if st["session"] is not None:
            if not self._harvest(st):
                return False                                      # a session is still running
            st = self._load()
        now = self.clock()
        daily_at = self.s.get("daily_summary_time")
        if daily_at and st["last_daily"] != self._today():
            h, m = (int(x) for x in daily_at.split(":"))
            due = dt.datetime.fromtimestamp(now).replace(hour=h, minute=m, second=0, microsecond=0)
            if now >= due.timestamp():
                st["last_daily"] = self._today()
                self._save(st)
                self.emit("daily_summary")
                st = self._load()
        if not st["pending"]:
            return False                                          # nothing has changed: never wake
        if now < st.get("paused_until", 0.0):
            return False                                          # plan limit reached: wait it out
        if not ignore_gap and now - st["last_wake"] < 60 * self.s["min_minutes_between_wakeups"]:
            return False                                          # batch into the next wake-up
        runs = self._runs_today()
        reason = None
        if len(runs) >= self.s["wakeups_per_day"]:
            reason = f"daily wake-up limit ({self.s['wakeups_per_day']}) reached"
        elif sum(r.get("cost_usd") or 0.0 for r in runs) >= self.s["usd_per_day"]:
            reason = f"daily spend limit (${self.s['usd_per_day']}) reached"
        if reason:
            if st["blocked_logged"] != self._today() + reason:
                self.log(f"limit: {reason}; {len(st['pending'])} event(s) wait, the queue keeps running")
                st["blocked_logged"] = self._today() + reason
                self._save(st)
            return False
        events = st["pending"]
        brief = self.brief_builder(events)
        model = self.s["analyse_model"] if all(e["kind"] == "daily_summary" for e in events) else self.s["model"]
        self.handle = self.launcher(brief, model)
        st.update(pending=[], last_wake=now, session={"started": now, "events": events, "model": model,
                                                      "pid": getattr(self.handle, "pid", None)})
        self._save(st)
        self.log(f"wake-up started ({model}): " + ", ".join(f"{e['kind']} {e.get('id', '')}".strip() for e in events))
        return True

    def _harvest(self, st: dict) -> bool:
        """Record a finished session. False while it is still running."""
        h = self.handle
        if h is None:                                             # daemon restarted during a session
            pid = st["session"].get("pid")
            if pid and _alive(pid):
                return False
            result = {"is_error": True, "result": "session ended while the daemon was restarting"}
        else:
            if h.poll() is None:
                return False
            result = h.result()
        sess = st["session"]
        entry = {"day": self._today(), "time": core.now(), "events": [e["kind"] for e in sess["events"]],
                 "ids": sorted({e["id"] for e in sess["events"] if e.get("id")}), "model": sess["model"],
                 "minutes": round((self.clock() - sess["started"]) / 60, 1), "cost_usd": result.get("total_cost_usd"),
                 "turns": result.get("num_turns"), "is_error": result.get("is_error", False),
                 "result": str(result.get("result", ""))[-1500:]}
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.runs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.log(f"wake-up finished: ${entry['cost_usd']}, {entry['turns']} turns, error {entry['is_error']}")
        if WAKE_LIMIT_TEXT.search(entry["result"]):
            pause = self.s.get("limit_pause_minutes", 60)
            st["paused_until"] = self.clock() + 60 * pause
            self.log(f"plan usage limit reached: wake-ups paused for {pause} min; the queue keeps running")
        st["session"] = None
        self.handle = None
        self._save(st)
        return True

    # ------------------------------------------------------------------ starting claude
    def _launch_claude(self, brief: str, model: str):
        from .agent_loop import WORKSPACE, prepare_workspace
        ref = core.reference()
        prepare_workspace(ref)
        brief_file = self.state_dir / "last_brief.md"
        brief_file.write_text(brief, encoding="utf-8")
        cmd = [shutil.which("claude") or "claude", "-p",
               "Your wake-up brief is on standard input. Follow harness/PROTOCOL.md for the events it lists.",
               "--model", model, "--permission-mode", "dontAsk",
               "--settings", str(core.ROOT / "harness" / "claude_settings.json"),
               "--append-system-prompt-file", str(core.ROOT / "harness" / "PROTOCOL.md"),
               "--max-turns", str(self.s["max_turns"]), "--max-budget-usd", str(self.s["usd_per_wakeup"]),
               "--output-format", "json"]
        env = dict(os.environ, HARNESS_MAIN=str(core.ROOT), HARNESS_STATE=str(core.STATE),
                   HARNESS_CONFIG=str(core.CONFIG_PATH))
        for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                  "CLAUDE_CODE_USE_FOUNDRY"):
            env.pop(k, None)                                      # plan login only: never bill an API key
        return _ClaudeHandle(cmd, WORKSPACE, env, brief_file, self.state_dir / "last_session.json")


class _ClaudeHandle:
    def __init__(self, cmd, cwd, env, brief_file: Path, out_file: Path):
        self.out_file = out_file
        self._stdin = open(brief_file, encoding="utf-8")
        self._stdout = open(out_file, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=self._stdin, stdout=self._stdout,
                                     stderr=subprocess.STDOUT, text=True)
        self.pid = self.proc.pid

    def poll(self):
        return self.proc.poll()

    def result(self) -> dict:
        self._stdin.close()
        self._stdout.close()
        text = self.out_file.read_text(encoding="utf-8", errors="replace")
        try:
            return json.loads(text[text.find("{"):])
        except (json.JSONDecodeError, ValueError):
            return {"is_error": True, "result": text[-2000:]}


def _alive(pid: int) -> bool:
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ------------------------------------------------------------------ the wake-up brief
def build_brief(events: list[dict]) -> str:
    """Everything the agent needs to start: the notebook summary, what happened, and the results."""
    from . import failures, metrics, notebook
    lines = ["# Wake-up brief", "", "## Notebook summary", notebook.show_summary(), "", "## Events since your last wake-up"]
    for e in events:
        when = dt.datetime.fromtimestamp(e["time"]).isoformat(timespec="minutes")
        detail = {k: v for k, v in e.items() if k not in ("kind", "time", "log_tail")}
        lines.append(f"- {when} {e['kind']} {json.dumps(detail) if detail else ''}".rstrip())
    finished = sorted({e["id"] for e in events if e["kind"] in ("experiment_finished", "run_problem") and e.get("id")})
    for exp_id in finished:
        lines += ["", f"## {exp_id}: metrics", "```", metrics.text(exp_id), "```",
                  "", f"## {exp_id}: failures", "```", failures.text(exp_id, 3), "```"]
    for e in events:
        if e["kind"] == "run_problem" and e.get("log_tail"):
            lines += ["", f"## {e['id']} seed {e.get('seed')}: end of the training log ({e.get('status')})",
                      "```", e["log_tail"], "```"]
    lines += ["", "## Queue", "```", _status_text(), "```"]
    return "\n".join(lines)


def _status_text() -> str:
    from .__main__ import status
    return status()
