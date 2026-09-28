"""The daemon's agent wake-ups: batching window, daily caps, and no wake-up while nothing changed.
Fake clock and fake Claude sessions: nothing is started, nothing is spent."""
import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.scheduler import Scheduler  # noqa: E402

START = dt.datetime(2026, 9, 29, 6, 0).timestamp()
MIN = 60.0
SETTINGS = {"model": "big", "analyse_model": "small", "max_turns": 5, "wakeups_per_day": 20,
            "min_minutes_between_wakeups": 20, "usd_per_wakeup": 5.0, "usd_per_day": 25.0, "daily_summary_time": None}


class Clock:
    def __init__(self):
        self.t = START

    def __call__(self):
        return self.t


class Session:
    def __init__(self, cost):
        self.cost, self.done, self.pid = cost, False, None

    def poll(self):
        return 0 if self.done else None

    def result(self):
        return {"total_cost_usd": self.cost, "num_turns": 3, "is_error": False, "result": "ok"}


class Harness:
    def __init__(self, tmp_path, cost=1.0, **settings):
        self.clock = Clock()
        self.launched = []                                 # (events in the brief, model, session)
        self.cost = cost

        def launcher(brief, model):
            s = Session(self.cost)
            self.launched.append((brief, model, s))
            return s
        self.sched = Scheduler(state_dir=tmp_path, clock=self.clock, launcher=launcher,
                               brief_builder=lambda events: [e["kind"] + ":" + str(e.get("id")) for e in events],
                               settings={**SETTINGS, **settings})

    def advance(self, minutes, step=1.0):
        """Tick every `step` minutes, like the daemon does."""
        end = self.clock.t + minutes * MIN
        while self.clock.t < end:
            self.clock.t = min(end, self.clock.t + step * MIN)
            self.sched.tick()

    def finish_running(self):
        for _, _, s in self.launched:
            s.done = True
        self.sched.tick()


def test_no_wake_up_while_nothing_changed(tmp_path):
    h = Harness(tmp_path)
    h.advance(3 * 24 * 60, step=5)                         # three days of ticks, no events
    assert h.launched == []


def test_events_inside_the_window_are_batched(tmp_path):
    h = Harness(tmp_path)
    h.sched.emit("experiment_finished", id="exp-0001")
    h.sched.tick()
    assert len(h.launched) == 1 and h.launched[0][0] == ["experiment_finished:exp-0001"]
    h.finish_running()
    h.advance(5)
    h.sched.emit("queue_low", id="exp-0002")
    h.advance(5)
    h.sched.emit("run_problem", id="exp-0002")
    h.advance(9)                                           # 19 minutes after the first wake-up
    assert len(h.launched) == 1
    h.advance(1)                                           # 20 minutes: one wake-up with both events
    assert len(h.launched) == 2
    assert h.launched[1][0] == ["queue_low:exp-0002", "run_problem:exp-0002"]


def test_no_second_wake_up_while_a_session_runs(tmp_path):
    h = Harness(tmp_path)
    h.sched.emit("queue_empty")
    h.sched.tick()
    h.sched.emit("experiment_finished", id="exp-0003")
    h.advance(90)                                          # session still running
    assert len(h.launched) == 1
    h.finish_running()
    assert len(h.launched) == 2 and h.launched[1][0] == ["experiment_finished:exp-0003"]


def test_daily_wake_up_cap(tmp_path):
    h = Harness(tmp_path, wakeups_per_day=3)
    for i in range(3):
        h.sched.emit("experiment_finished", id=f"exp-{i}")
        h.advance(21)
        h.finish_running()
    assert len(h.launched) == 3
    h.sched.emit("experiment_finished", id="exp-9")
    h.advance(120)
    assert len(h.launched) == 3                            # capped; the event waits
    assert "daily wake-up limit (3) reached" in (tmp_path / "scheduler.log").read_text()
    h.advance(24 * 60)                                     # next day
    assert len(h.launched) == 4 and h.launched[3][0] == ["experiment_finished:exp-9"]


def test_daily_spend_cap(tmp_path):
    h = Harness(tmp_path, cost=4.0, usd_per_day=10.0)
    for i in range(4):
        h.sched.emit("experiment_finished", id=f"exp-{i}")
        h.advance(21)
        h.finish_running()
    assert len(h.launched) == 3                            # $4 + $4 + $4: the cap stops the fourth
    assert "daily spend limit ($10.0) reached" in (tmp_path / "scheduler.log").read_text()
    h.advance(24 * 60)
    assert len(h.launched) == 4


def test_daily_summary_once_a_day(tmp_path):
    h = Harness(tmp_path, daily_summary_time="08:00")
    h.advance(119)                                         # 06:00 -> 07:59
    assert h.launched == []
    h.advance(2)
    assert len(h.launched) == 1 and h.launched[0][0] == ["daily_summary:None"]
    assert h.launched[0][1] == "small"                     # summary-only wake-ups use the smaller model
    h.finish_running()
    h.advance(12 * 60)
    assert len(h.launched) == 1                            # once per day
    h.advance(12 * 60)
    assert len(h.launched) == 2


def test_repeated_events_are_not_duplicated(tmp_path):
    h = Harness(tmp_path)
    h.sched.emit("experiment_finished", id="exp-0001")
    h.sched.tick()
    h.finish_running()
    h.sched.emit("queue_low", id="exp-0002")
    h.sched.emit("queue_low", id="exp-0002")
    h.advance(20)
    assert h.launched[1][0] == ["queue_low:exp-0002"]


def test_plan_limit_pauses_wake_ups(tmp_path):
    h = Harness(tmp_path)
    h.sched.emit("experiment_finished", id="exp-0001")
    h.sched.tick()
    h.launched[0][2].result = lambda: {"total_cost_usd": 0.1, "num_turns": 1, "is_error": True,
                                       "result": "You've hit your session limit"}
    h.finish_running()
    h.sched.emit("queue_low", id="exp-0002")
    h.advance(59)
    assert len(h.launched) == 1                            # paused, not retrying
    assert "plan usage limit reached" in (tmp_path / "scheduler.log").read_text()
    h.advance(2)
    assert len(h.launched) == 2


def test_claude_never_gets_api_key_variables(tmp_path, monkeypatch):
    import harness.scheduler as S
    captured = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(S, "_ClaudeHandle", lambda cmd, cwd, env, *a: captured.update(env=env))
    import harness.agent_loop as A
    monkeypatch.setattr(A, "prepare_workspace", lambda ref: None)
    monkeypatch.setattr(S.core, "reference", lambda: "0" * 40)
    sched = S.Scheduler(state_dir=tmp_path, settings=SETTINGS)
    sched._launch_claude("brief", "big")
    assert "ANTHROPIC_API_KEY" not in captured["env"]
