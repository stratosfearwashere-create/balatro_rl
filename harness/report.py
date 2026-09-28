"""Promotion reports for the human: the diff, the metrics and the paired comparison. Nothing merges."""
from __future__ import annotations

from .core import ROOT, load_registry, now, reference
from .workspace import git
from . import compare, failures, metrics


def write_report(exp_id: str) -> str:
    reg = load_registry()
    exp = reg["experiments"][exp_id]
    ref = reference()
    parent = exp.get("promoted_from")
    base = exp.get("baseline")
    parts = [f"# Promotion report: {exp_id} ({exp['spec'].get('name', '')})", "",
             f"Generated {now()}. Nothing has been merged; review and merge by hand if you agree.", "",
             f"**Hypothesis:** {exp['spec'].get('hypothesis', '')}", "",
             f"Branch `{exp['branch']}` at `{exp['commit']}`; full-tier experiment {exp_id}"
             + (f", promoted from cheap experiment {parent}" if parent else "") + ".", "",
             "## Metrics (full tier)", "```", metrics.text(exp_id), "```"]
    if base and base in reg["experiments"] and reg["experiments"][base]["status"] == "done":
        parts += ["", f"## Paired comparison with baseline {base}", "```", compare.text(exp_id, base), "```"]
    elif base:
        parts += ["", f"Baseline {base} has not finished yet; run `python -m harness compare {exp_id} {base}` later."]
    if parent:
        parts += ["", "## Cheap-tier result that earned the promotion", "```", metrics.text(parent), "```"]
    parts += ["", "## Failures", "```", failures.text(exp_id, 3), "```",
              "", f"## Diff against the reference commit {ref[:10]}", "```diff",
              git("diff", f"{ref}...{exp['commit']}", "--stat"), git("diff", f"{ref}...{exp['commit']}"), "```"]
    out = ROOT / "reports" / f"{exp_id}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return str(out)
