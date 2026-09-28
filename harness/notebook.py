"""The research journal: append-only entries, plus a short summary the agent keeps up to date."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from .core import ROOT, locked, now
from .metrics import summary
from .stats import fmt_ci

NB = Path(os.environ.get("HARNESS_NOTEBOOK", ROOT / "notebook"))
JOURNAL, SUMMARY, META = NB / "journal.md", NB / "summary.md", NB / ".meta.json"
REQUIRED = ("title", "hypothesis", "change", "experiments", "result", "conclusion", "outcome")
OUTCOMES = ("positive", "negative", "inconclusive")
SUMMARY_WORDS = 900
SUMMARY_EVERY = 5


def _meta() -> dict:
    return json.loads(META.read_text()) if META.exists() else {"entries": 0, "at_last_summary": 0}


def add(entry_file: str) -> str:
    e = yaml.safe_load(Path(entry_file).read_text())
    missing = [k for k in REQUIRED if not e.get(k)]
    if missing:
        return f"refused: missing fields {missing}"
    if e["outcome"] not in OUTCOMES:
        return f"refused: outcome must be one of {OUTCOMES}"
    exps = e["experiments"] if isinstance(e["experiments"], list) else [e["experiments"]]
    facts = []
    for x in exps:
        s = summary(str(x))
        facts.append(f"{x}: blinds {fmt_ci(*s['blinds'])}, wins {s['wins'][0]}/{s['wins'][1]}" if s
                     else f"{x}: no evaluation")
    with locked():
        NB.mkdir(exist_ok=True)
        m = _meta()
        m["entries"] += 1
        text = (f"\n## #{m['entries']} {now()} {e['title']} [{e['outcome']}]\n"
                f"- **Hypothesis:** {e['hypothesis']}\n- **Change:** {e['change']}\n"
                f"- **Experiments:** {', '.join(map(str, exps))}\n- **Result:** {e['result']}\n"
                f"- **Harness numbers:** {'; '.join(facts)}\n- **Conclusion:** {e['conclusion']}\n")
        with open(JOURNAL, "a", encoding="utf-8") as f:
            f.write(text)
        META.write_text(json.dumps(m))
    due = m["entries"] - m["at_last_summary"] >= SUMMARY_EVERY
    return f"added entry #{m['entries']}" + ("; the summary is due for a rewrite (notebook set-summary)" if due else "")


def show_summary() -> str:
    m = _meta()
    s = SUMMARY.read_text(encoding="utf-8") if SUMMARY.exists() else "(no summary yet)"
    since = m["entries"] - m["at_last_summary"]
    tail = ""
    if JOURNAL.exists() and since:
        heads = [l for l in JOURNAL.read_text(encoding="utf-8").splitlines() if l.startswith("## #")]
        tail = "\nentries since the summary:\n" + "\n".join(heads[-since:])
    due = " - rewrite due" if since >= SUMMARY_EVERY else ""
    return f"{s}\n[{m['entries']} journal entries, {since} since this summary{due}]{tail}"


def show(last: int = 3) -> str:
    if not JOURNAL.exists():
        return "(journal is empty)"
    parts = JOURNAL.read_text(encoding="utf-8").split("\n## #")[1:]
    return "\n".join("## #" + x for x in parts[-last:]) if parts else "(journal is empty)"


def set_summary(draft_file: str) -> str:
    text = Path(draft_file).read_text(encoding="utf-8").strip()
    words = len(text.split())
    if words > SUMMARY_WORDS:
        return f"refused: {words} words (limit {SUMMARY_WORDS})"
    with locked():
        NB.mkdir(exist_ok=True)
        if SUMMARY.exists():
            (NB / "summaries").mkdir(exist_ok=True)
            (NB / "summaries" / f"summary-{now().replace(':', '')}.md").write_text(SUMMARY.read_text(encoding="utf-8"),
                                                                              encoding="utf-8")
        SUMMARY.write_text(text + "\n", encoding="utf-8")
        m = _meta()
        m["at_last_summary"] = m["entries"]
        META.write_text(json.dumps(m))
    return f"summary replaced ({words} words)"
