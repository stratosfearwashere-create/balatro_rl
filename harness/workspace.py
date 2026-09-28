"""Code for experiments: branch checks, one git worktree per experiment, smoke tests."""
from __future__ import annotations

import ast
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .core import ROOT, config, exp_dir, reference


def git(*args, cwd=ROOT, check=True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def resolve(ref: str) -> str:
    return git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()


def changed_files(commit: str, base: str) -> list[str]:
    return [f for f in git("diff", "--name-only", f"{base}...{commit}").splitlines() if f]


def _function_source(rev: str, path: str, qualname: str) -> str | None:
    """Normalised source of a function or Class.method at a revision (None if missing)."""
    try:
        src = git("show", f"{rev}:{path}")
    except RuntimeError:
        return None
    tree = ast.parse(src)
    parts = qualname.split(".")
    nodes = tree.body
    node = None
    for i, name in enumerate(parts):
        node = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                     and n.name == name), None)
        if node is None:
            return None
        nodes = node.body
    return ast.unparse(node)


def _under(f: str, p: str) -> bool:
    p = p.rstrip("/")
    return f == p or f.startswith(p + "/")


def check_branch(branch: str) -> dict:
    """Everything launch needs to know about a branch. problems == [] means it may run."""
    cfg = config()
    problems = []
    prefix = cfg["agent"]["allowed_branch_prefix"]
    if not branch.startswith(prefix) and branch != cfg["main_branch"]:
        problems.append(f"experiment branches must be named {prefix}<something>")
    ref = reference()
    if ref is None:
        return {"commit": None, "files": [], "sim_changed": False,
                "problems": ["no reference commit pinned: the human must run `python -m harness pin`"]}
    try:
        commit = resolve(branch)
    except RuntimeError:
        return {"commit": None, "files": [], "problems": [f"branch {branch!r} not found"], "sim_changed": False}
    files = changed_files(commit, ref)
    sim_changed = any(f.startswith(tuple(cfg["simulator_paths"])) for f in files)
    for f in files:
        for p in cfg["protected_paths"]:
            if _under(f, p) and not (p in cfg["simulator_paths"] and cfg["allow_simulator_changes"]):
                problems.append(f"touches protected path {p}: {f}")
    for path, funcs in cfg["protected_functions"].items():
        for q in funcs:
            if _function_source(ref, path, q) != _function_source(commit, path, q):
                problems.append(f"changes protected function {path}::{q}")
    return {"commit": commit, "files": files, "problems": problems, "sim_changed": sim_changed}


def create_worktree(exp_id: str, commit: str) -> Path:
    path = exp_dir(exp_id) / "code"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "--detach", str(path), commit)
    # the compiled scorer is not in git; the simulator is protected, so main's build matches
    for f in glob.glob(str(ROOT / "balatro_rl" / "sim" / "_fastscore*.pyd")) + \
            glob.glob(str(ROOT / "balatro_rl" / "sim" / "_fastscore*.so")):
        shutil.copy2(f, path / "balatro_rl" / "sim")
    return path


def remove_worktree(exp_id: str):
    path = exp_dir(exp_id) / "code"
    if path.exists():
        git("worktree", "remove", "--force", str(path), check=False)


def python_env(code: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(code)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_tests(code: Path, which: str = "tests") -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", which], cwd=code, env=python_env(code),
                       capture_output=True, text=True, timeout=1800)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
    return r.returncode == 0, tail


def smoke_train(code: Path, out_dir: Path, strategic: bool = True) -> tuple[bool, str]:
    """Two tiny PPO iterations with the branch's code: catches crashes before real budget is spent."""
    cfg = config()
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-m", "balatro_rl.train", "ppo", "--init", str(ROOT / cfg["baseline_checkpoint"]),
            "--stake", "GOLD", "--iters", "2", "--envs", "2", "--steps", "16", "--save-every", "1000",
            "--out", str(out_dir / "smoke.pt")]
    if strategic:
        args += ["--strategic", "--tactical", str(ROOT / cfg["tactical_checkpoint"])]
    r = subprocess.run(args, cwd=code, env=python_env(code), capture_output=True, text=True, timeout=900)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-8:])
    return r.returncode == 0, tail
