"""Lessons from our games: the teacher (Claude) learns across games through its prompt.

  review       after a game, Claude reads a compact account of it (each decision with the money, the move, the outlook it was shown and its
               reasoning; the build at each ante; how it ended) and writes 1-5 general lessons, each with the decisions that show it
  consolidate  every few games, Claude merges the new lessons into the list (traces/lm/lessons.json): duplicates combined, each lesson with
               the games that support it; at most MAX_LESSONS
  use          the lessons supported by at least MIN_SUPPORT different games are written to balatro_rl/lm/lessons.md, which render.SYSTEM
               appends to the playbook ("LESSONS FROM OUR GAMES"); single-game lessons stay candidates

    python -m balatro_rl.lm.lessons review traces/lm/game1.jsonl            (review every finished game in a log, then consolidate)
    python -m balatro_rl.lm.lessons show
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
LESSONS_MD = os.path.join(HERE, "lessons.md")
DB = os.path.join(HERE, "..", "..", "traces", "lm", "lessons.json")
MIN_SUPPORT, MAX_LESSONS = 2, 15

REVIEW_SYSTEM = """You review a finished game of Balatro (Gold stake, Yellow Deck) played by an AI strategist that follows a playbook. Find
what lost the run (or what won it) and turn it into general lessons the strategist can apply in future runs.

A good lesson is a rule for a situation that recurs, specific enough to act on ("From ante 3, do not reroll or open packs when that takes
money below $25 unless the outlook says the next blind is below 70%"), not a description of this game. Point to the decisions that show
it (by their numbers). Prefer the mistakes that cost the most. If a playbook rule was broken, say which; if the playbook's advice itself
looks wrong in this situation, say that instead. Do not blame bad luck unless the outlook it was shown said the risk was small.

Answer in JSON: {"summary": "2-4 sentences: how the run went and what decided it",
 "lessons": [{"rule": "...", "evidence": "decision numbers and what happened", "cost": "what it cost the run"}]} with 1-5 lessons."""

REVIEW_SCHEMA = {"type": "object", "properties": {
    "summary": {"type": "string"},
    "lessons": {"type": "array", "items": {"type": "object", "properties": {"rule": {"type": "string"}, "evidence": {"type": "string"},
                                                                             "cost": {"type": "string"}},
                                           "required": ["rule", "evidence", "cost"], "additionalProperties": False}}},
    "required": ["summary", "lessons"], "additionalProperties": False}

MERGE_SYSTEM = f"""You keep the list of lessons an AI Balatro strategist (Gold stake, Yellow Deck) has learned from its games. You get the
current list (each lesson with the games that support it) and new lessons from newly reviewed games. Merge them:
- a new lesson that says the same as an existing one (even in other words) adds its game to that lesson's support; sharpen the wording
  if the new evidence makes the rule more precise;
- a new lesson that is really new is added with its game as its only support;
- two lessons that contradict each other: keep the one with more support and rewrite it to state when each applies, if the evidence
  shows both;
- keep at most {MAX_LESSONS} lessons: drop the ones with the least support first.
Answer in JSON: {{"lessons": [{{"rule": "...", "games": ["game ids"]}}]}}."""

MERGE_SCHEMA = {"type": "object", "properties": {"lessons": {"type": "array", "items": {"type": "object", "properties": {
    "rule": {"type": "string"}, "games": {"type": "array", "items": {"type": "string"}}}, "required": ["rule", "games"],
    "additionalProperties": False}}}, "required": ["lessons"], "additionalProperties": False}


def claude(system, user, schema, model="claude-opus-5-5", effort="high", timeout=1200):
    """one structured call through the claude command line on the signed-in account (a clean environment, as lm/play.py)"""
    import tempfile
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    f.write(system)
    f.close()
    cmd = ["claude", "-p", "--system-prompt-file", f.name, "--tools", "", "--safe-mode", "--strict-mcp-config", "--no-session-persistence",
           "--model", model, "--effort", effort, "--output-format", "json", "--json-schema", json.dumps(schema)]
    env = {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE") or k == "CLAUDECODE")}
    p = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env)
    r = json.loads(p.stdout)
    out = r.get("structured_output")
    if not isinstance(out, dict):
        out = json.loads(r.get("result") or "{}")
    return out


def _grab(pat, text, default=""):
    m = re.search(pat, text or "")
    return m.group(1) if m else default


def game_account(rows, game, result) -> str:
    """A compact account of one game from the play log: the decisions (numbered), the build at each ante, the result."""
    dec = [d for d in rows if d.get("kind") == "decision" and d.get("game") == game]
    L = [f"RESULT: {result['blinds']:.0f} blinds beaten, {'WON' if result.get('won') else 'lost'} (died in ante {result.get('ante')})."]
    last_ante = None
    for n, d in enumerate(dec, 1):
        u = d.get("user", "")
        if d["ante"] != last_ante and d["state"] != "ROUND":
            last_ante = d["ante"]
            L.append(f"\n--- ANTE {d['ante']} ---")
            L.append("Jokers: " + _grab(r"JOKERS \([^)]*\)[^:]*: ([^\n]*)", u)[:600])
            L.append(_grab(r"(HAND LEVELS[^\n]*)", u)[:300])
            L.append("Deck: " + _grab(r"DECK: ([^\n]*)", u)[:200])
        ans = d.get("answer") or {}
        safe = _grab(r"Safe \(90%\+\) through: ([^.]*)", u) or ("NOT SAFE for the next blind" if "Not safe even" in u else "")
        nxt = _grab(r"chance to clear the next blind[^:]*: (\d+%)", u)
        boss = _grab(r"Boss Blind too \([^)]*\): about (\d+%)", u)
        if d["state"] == "ROUND":
            est = _grab(r"playing on clears this blind about (\d+%)", u)
            st = json.dumps(ans.get("style")) if ans.get("style") else ""
            L.append(f"{n}. [round, blind {d['blind']}] ${d['money']} -> {d['move'][:110]} | clear if playing on {est} | style {st}"
                     f" | why: {ans.get('reasoning', '')[:220]}")
        else:
            L.append(f"{n}. [{d['state'].lower()}, blind {d['blind']}] ${d['money']} -> {d['move'][:120]} | outlook: next {nxt}, boss {boss},"
                     f" safe through {safe or '-'} | why: {ans.get('reasoning', '')[:260]}")
    if dec:
        L.append("\nLAST PLAN NOTE: " + str((dec[-1].get("answer") or {}).get("plan", ""))[:600])
    return "\n".join(L)


def load_db():
    if os.path.exists(DB):
        return json.load(open(DB, encoding="utf-8"))
    return {"lessons": [], "reviewed": [], "version": 0}


def save_db(db):
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    json.dump(db, open(DB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    keep = [l for l in db["lessons"] if len(set(l["games"])) >= MIN_SUPPORT]
    keep.sort(key=lambda l: -len(set(l["games"])))
    with open(LESSONS_MD, "w", encoding="utf-8") as f:
        f.write(f"LESSONS FROM OUR GAMES (version {db['version']}; each seen in at least {MIN_SUPPORT} games, most supported first)\n")
        for l in keep:
            f.write(f"- {l['rule']} [seen in {len(set(l['games']))} games]\n")


def review(rows, game, result, game_id):
    acc = game_account(rows, game, result)
    out = claude(REVIEW_SYSTEM, "PLAYBOOK THE STRATEGIST FOLLOWED (for reference):\n" + open(os.path.join(HERE, "playbook.md"), encoding="utf-8").read()[:6000]
                 + "\n\nTHE GAME (decisions numbered):\n" + acc, REVIEW_SCHEMA)
    return {"game": game_id, "summary": out.get("summary", ""), "lessons": out.get("lessons", [])}


def consolidate(db, new_reviews):
    cur = [{"rule": l["rule"], "games": sorted(set(l["games"]))} for l in db["lessons"]]
    new = [{"rule": l["rule"], "evidence": l.get("evidence", ""), "game": r["game"]} for r in new_reviews for l in r["lessons"]]
    if not new:
        return db
    out = claude(MERGE_SYSTEM, "CURRENT LESSONS:\n" + json.dumps(cur, ensure_ascii=False, indent=1) + "\n\nNEW LESSONS:\n"
                 + json.dumps(new, ensure_ascii=False, indent=1), MERGE_SCHEMA)
    db["lessons"] = [{"rule": l["rule"], "games": sorted(set(l["games"]))} for l in out.get("lessons", [])][:MAX_LESSONS]
    db["version"] += 1
    return db


def review_log(path, consolidate_now=True):
    """Reviews every finished game in a play log not reviewed yet; then merges the lessons. -> the new reviews"""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    db = load_db()
    done = set(db["reviewed"])
    news = []
    for r in rows:
        if r.get("kind") != "result":
            continue
        gid = f"{os.path.basename(path)}#{r['game']}#{r.get('seed')}"
        if gid in done:
            continue
        rv = review(rows, r["game"], r, gid)
        news.append(rv)
        db["reviewed"].append(gid)
        with open(os.path.join(os.path.dirname(DB), "reviews.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rv, ensure_ascii=False) + "\n")
    if news and consolidate_now:
        db = consolidate(db, news)
    save_db(db)
    return news


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review")
    r.add_argument("logs", nargs="+")
    sub.add_parser("show")
    a = p.parse_args()
    if a.cmd == "review":
        for path in a.logs:
            for rv in review_log(path):
                print(json.dumps(rv, ensure_ascii=False, indent=1))
    print(open(LESSONS_MD, encoding="utf-8").read() if os.path.exists(LESSONS_MD) else "(no lessons yet)")
    db = load_db()
    print(f"\n{len(db['lessons'])} lessons in the list (version {db['version']}), {len(db['reviewed'])} games reviewed; candidates:")
    for l in db["lessons"]:
        print(f"  [{len(set(l['games']))}] {l['rule']}")


if __name__ == "__main__":
    main()
