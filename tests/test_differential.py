"""Differential test: two backends must produce identical game states after every step of random games.

Today the backends are the compiled extensions (cards, clone, scorer, solver) against BALATRO_PURE=1 (the
pure-Python versions); the C++ game core is compared against the Python game the same way. Each backend
plays the games of tests/diffplay.py in its own process (the backend is chosen when the package is
imported) and the two logs are compared line by line: the first differing step is reported with the
attributes that differ.

    python -m pytest -q tests/test_differential.py
    BALATRO_DIFF_GAMES=60 python -m pytest -q tests/test_differential.py      # a longer fuzz run
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GAMES = int(os.environ.get("BALATRO_DIFF_GAMES", "8"))
STEPS = int(os.environ.get("BALATRO_DIFF_STEPS", "300"))


def _run(out: str, env_over: dict, games: int = GAMES, seed0: int = 0, steps: int = STEPS):
    env = dict(os.environ)
    env.pop("BALATRO_PURE", None)
    env.pop("BALATRO_PYSCORE", None)
    env.update(env_over)
    cmd = [sys.executable, "-m", "tests.diffplay", "--games", str(games), "--seed0", str(seed0),
           "--steps", str(steps), "--out", out]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"backend {env_over or 'compiled'} crashed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
    with open(out) as f:
        return [json.loads(line) for line in f]


def _diff(a: dict, b: dict, path: str = "") -> list[str]:
    """Paths (dotted) where two canonical values differ, with both values."""
    out = []
    if type(a) is not type(b):
        return [f"{path}: {a!r} vs {b!r}"]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: {'missing' if k not in a else a[k]!r} vs {'missing' if k not in b else b[k]!r}")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f"{path}[{i}]")
    elif a != b or repr(a) != repr(b):            # 3 vs 3.0 differ (int / float distinctions are kept)
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def compare_logs(ref: list, other: list, ref_name: str, other_name: str):
    for i, (x, y) in enumerate(zip(ref, other)):
        d = _diff(x, y)
        if d:
            head = f"game {x['game']} step {x['step']} (action {x.get('action')}): {ref_name} vs {other_name}"
            raise AssertionError(head + "\n  " + "\n  ".join(d[:40]))
    assert len(ref) == len(other), f"{ref_name} logged {len(ref)} steps, {other_name} {len(other)}"


def _available(env_over: dict) -> bool:
    """Whether the backend differs from the reference at all (compiled extensions built / C++ core built)."""
    code = ("import balatro_rl.sim.cards as c, balatro_rl.sim.fastscore as f, balatro_rl.sim.game as g;"
            "print(int(c.COMPILED or f.ENABLED or g._clone_game is not None or getattr(g, 'CPP', False)))")
    env = dict(os.environ)
    env.pop("BALATRO_PURE", None)
    env.update(env_over)
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "1"


def test_compiled_matches_pure(tmp_path):
    if not _available({}):
        pytest.skip("no compiled extension is built (python setup_cython.py build)")
    ref = _run(str(tmp_path / "pure.jsonl"), {"BALATRO_PURE": "1"})
    cur = _run(str(tmp_path / "compiled.jsonl"), {})
    assert len(ref) > GAMES * 20, "the fuzz games are too short to test anything"
    compare_logs(ref, cur, "pure", "compiled")


def test_snapshot_sees_every_field():
    """snapshot keeps the attribute list of the game: a new field cannot go unchecked."""
    from balatro_rl.sim.game import Game
    from tests.diffplay import snapshot
    g = Game(seed=1, stake="WHITE")
    snap = snapshot(g)
    assert set(snap) == set(vars(g))
    assert snap["rng"]["rng"] and snap["money"] == g.money
