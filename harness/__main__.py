"""python -m harness <command>. The agent uses this instead of a shell. See harness/PROTOCOL.md."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

from .core import ROOT, STATE, config, load_registry, minutes_committed, minutes_used, now, pin_reference, update
from . import compare, failures, metrics, notebook


def launch(spec_file: str) -> str:
    from .workspace import check_branch
    cfg = config()
    spec = yaml.safe_load(Path(spec_file).read_text()) or {}
    errors = []
    for k in ("name", "hypothesis", "branch"):
        if not spec.get(k):
            errors.append(f"missing '{k}'")
    tier_name = spec.get("tier", "cheap")
    if tier_name not in cfg["tiers"]:
        return f"refused: unknown tier {tier_name!r}"
    if tier_name == "full":
        errors.append("full-tier runs are started by `promote`, not by launch")
    tier = cfg["tiers"][tier_name]
    ov = spec.get("overrides") or {}
    locked = [k for k in ov if k in cfg["locked_hyperparameters"]]
    if locked:
        errors.append(f"continuous hyperparameters are fixed by the harness, not set by experiments: {locked}")
    bad = [k for k in ov if k not in cfg["allowed_overrides"] and k not in locked]
    if bad:
        errors.append(f"overrides not allowed: {bad} (allowed: {cfg['allowed_overrides']})")
    init = spec.get("init", "baseline")
    reg = load_registry()
    if init not in ("baseline", "scratch"):
        src = init.split(":", 1)[1] if init.startswith("experiment:") else None
        if not src or reg["experiments"].get(src, {}).get("status") != "done":
            errors.append(f"init must be baseline, scratch or experiment:<id of a finished experiment>, not {init!r}")
    steps = int(spec.get("env_steps", tier["env_steps"]))
    if not 0 < steps <= tier["env_steps"]:
        errors.append(f"env_steps must be between 1 and {tier['env_steps']} for the {tier_name} tier")
    seeds = spec.get("seeds", tier["seeds"])
    if not seeds or any(not isinstance(s, int) or not 1 <= s <= 100 for s in seeds):
        errors.append("seeds must be a non-empty list of integers from 1 to 100")
    cap = cfg["budgets"]["experiment_minutes"][tier_name]
    budget = float(spec.get("budget_minutes", cap))
    if not 0 < budget <= cap:
        errors.append(f"budget_minutes must be between 1 and {cap}")
    br = check_branch(spec.get("branch", ""))
    errors += br["problems"]
    if br["sim_changed"] and cfg["allow_simulator_changes"]:
        errors.append("simulator changes need `python -m harness check <branch>` to pass the fidelity tests first")
    if errors:
        return "refused:\n  - " + "\n  - ".join(errors)

    def register(r):
        exp_id = f"exp-{r['next_id']:04d}"
        r["next_id"] += 1
        r["experiments"][exp_id] = {
            "id": exp_id, "created": now(), "status": "queued", "tier": tier_name, "spec": spec,
            "branch": spec["branch"], "commit": br["commit"], "files_changed": br["files"],
            "env_steps": steps, "seeds": seeds, "budget_minutes": budget, "runs": []}
        return exp_id
    exp_id = update(register)
    reg = load_registry()
    ahead = minutes_used(reg) + minutes_committed(reg)
    note = "" if ahead <= cfg["budgets"]["day_minutes"] else \
        " (beyond today's training budget: it will wait for tomorrow's)"
    return f"queued {exp_id}: {tier_name}, {len(seeds)} seed(s) x {steps:,} env steps, budget {budget:.0f} min{note}"


def status(show_all: bool = False) -> str:
    cfg = config()
    reg = load_registry()
    exps = sorted(reg["experiments"].values(), key=lambda e: e["created"])
    if not show_all:
        exps = exps[-15:]
    lines = [f"training budget today: {minutes_used(reg):.0f} used + {minutes_committed(reg):.0f} committed "
             f"of {cfg['budgets']['day_minutes']} min"]
    for e in exps:
        used = sum(r.get("minutes", 0) for r in e.get("runs", []))
        lines.append(f"{e['id']}  {e['status']:24s} {e['tier']:5s} {used:4.0f}/{e['budget_minutes']:.0f} min  "
                     f"{e['branch']}  {e['spec'].get('name', '')}")
    return "\n".join(lines) if exps else lines[0] + "\n(no experiments yet)"


def kill(exp_id: str) -> str:
    def f(r):
        e = r["experiments"].get(exp_id)
        if e is None:
            return f"unknown experiment {exp_id}"
        if e["status"] in ("done", "failed", "killed"):
            return f"{exp_id} already {e['status']}"
        e["kill_requested"] = True
        if e["status"] == "queued":
            e["status"] = "killed"
        return f"{exp_id}: kill requested (the daemon stops it within seconds and still evaluates saved checkpoints)"
    return update(f)


def check(branch: str) -> str:
    """Branch checks, the test suite and a two-iteration training run, without spending budget."""
    from .workspace import check_branch, git, run_tests, smoke_train
    br = check_branch(branch)
    out = [f"{branch} @ {str(br['commit'])[:8]}: {len(br['files'])} file(s) changed"]
    out += [f"  problem: {p}" for p in br["problems"]] or ["  protected paths and functions: ok"]
    if br["commit"] is None:
        return "\n".join(out)
    tmp = STATE / "_checks" / now().replace(":", "")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "--detach", str(tmp), br["commit"])
    try:
        import glob
        for f in glob.glob(str(ROOT / "balatro_rl" / "sim" / "_fastscore*.pyd")):
            shutil.copy2(f, tmp / "balatro_rl" / "sim")
        ok, tail = run_tests(tmp, "tests")
        out.append(f"tests (including fidelity): {'pass' if ok else 'FAIL'}")
        if not ok:
            out.append(tail)
        ok2, tail2 = smoke_train(tmp, tmp / "_smoke")
        out.append(f"two-iteration training run: {'ok' if ok2 else 'FAIL'}")
        if not ok2:
            out.append(tail2)
    finally:
        git("worktree", "remove", "--force", str(tmp), check=False)
    return "\n".join(out)


def promote(exp_id: str, baseline: str) -> str:
    cfg = config()
    reg = load_registry()
    for e in (exp_id, baseline):
        x = reg["experiments"].get(e)
        if x is None or x["status"] != "done" or x["tier"] != "cheap":
            return f"refused: {e} must be a finished cheap-tier experiment"
    r = compare.result(exp_id, baseline)
    if "error" in r:
        return f"refused: {r['error']}"
    if not r["non_overlapping"]:
        return (f"refused by the promotion rule: {exp_id}'s 95% CI {r['a']['blinds'][1]:.2f}-{r['a']['blinds'][2]:.2f} "
                f"does not lie above {baseline}'s {r['b']['blinds'][1]:.2f}-{r['b']['blinds'][2]:.2f}")
    full = cfg["tiers"]["full"]

    def queue(r_, src_id, extra):
        src = r_["experiments"][src_id]
        new = f"exp-{r_['next_id']:04d}"
        r_["next_id"] += 1
        r_["experiments"][new] = {**{k: src[k] for k in ("spec", "branch", "commit", "files_changed")},
                                  "id": new, "created": now(), "status": "queued", "tier": "full",
                                  "env_steps": full["env_steps"], "seeds": full["seeds"],
                                  "budget_minutes": cfg["budgets"]["experiment_minutes"]["full"], "runs": [], **extra}
        return new

    def f(r_):
        base_full = next((k for k, e in r_["experiments"].items() if e.get("full_of") == baseline
                          and e["status"] not in ("failed", "killed")), None)
        if base_full is None:
            base_full = queue(r_, baseline, {"full_of": baseline})
        new = queue(r_, exp_id, {"full_of": exp_id, "promoted_from": exp_id, "baseline": base_full})
        return new, base_full
    new, base_full = update(f)
    return (f"promoted: queued full-tier {new} (from {exp_id}) and compared against baseline {base_full}. "
            f"When it finishes, reports/{new}.md is written for the human. Nothing is merged.")


def main():
    p = argparse.ArgumentParser(prog="python -m harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("launch").add_argument("spec")
    st = sub.add_parser("status")
    st.add_argument("--all", action="store_true")
    sub.add_parser("kill").add_argument("id")
    sub.add_parser("check").add_argument("branch")
    sub.add_parser("metrics").add_argument("id")
    fl = sub.add_parser("failures")
    fl.add_argument("id")
    fl.add_argument("n", type=int, nargs="?", default=5)
    cp = sub.add_parser("compare")
    cp.add_argument("a")
    cp.add_argument("b")
    nb = sub.add_parser("notebook")
    nbs = nb.add_subparsers(dest="nbcmd", required=True)
    nbs.add_parser("add").add_argument("entry_file")
    nbs.add_parser("summary")
    sh = nbs.add_parser("show")
    sh.add_argument("--last", type=int, default=3)
    nbs.add_parser("set-summary").add_argument("draft_file")
    pr = sub.add_parser("promote")
    pr.add_argument("id")
    pr.add_argument("--baseline", required=True)
    sub.add_parser("report").add_argument("id")
    pn = sub.add_parser("pin", help="(human only) pin the reference commit branches are checked against")
    pn.add_argument("ref", nargs="?", default=None)
    dm = sub.add_parser("daemon", help="run the queue (for the human to start)")
    dm.add_argument("--once", action="store_true")
    ag = sub.add_parser("agent", help="the research loop driver (for the human to start)")
    ag.add_argument("--once", action="store_true")
    ag.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.cmd == "launch":
        print(launch(a.spec))
    elif a.cmd == "status":
        print(status(a.all))
    elif a.cmd == "kill":
        print(kill(a.id))
    elif a.cmd == "check":
        print(check(a.branch))
    elif a.cmd == "metrics":
        print(metrics.text(a.id))
    elif a.cmd == "failures":
        print(failures.text(a.id, a.n))
    elif a.cmd == "compare":
        print(compare.text(a.a, a.b))
    elif a.cmd == "notebook":
        if a.nbcmd == "add":
            print(notebook.add(a.entry_file))
        elif a.nbcmd == "summary":
            print(notebook.show_summary())
        elif a.nbcmd == "show":
            print(notebook.show(a.last))
        else:
            print(notebook.set_summary(a.draft_file))
    elif a.cmd == "promote":
        print(promote(a.id, a.baseline))
    elif a.cmd == "report":
        from .report import write_report
        print(write_report(a.id))
    elif a.cmd == "pin":
        from .workspace import resolve
        commit = resolve(a.ref or config()["main_branch"])
        pin_reference(commit)
        print(f"reference pinned to {commit}")
    elif a.cmd == "daemon":
        from .daemon import run
        run(once=a.once)
    elif a.cmd == "agent":
        from .agent_loop import run as agent_run
        agent_run(once=a.once, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
