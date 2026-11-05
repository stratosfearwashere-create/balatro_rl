"""Text for the language-model player: the system prompt (role, rules, the playbook, the answer format) and each decision's message (the run's
state, a checkpoint block that applies the playbook's targets to this state, the engine's survival estimates, the legal moves numbered).

The same text is used for the teacher (Claude) and the student (a small open model), so the student is trained on exactly what it will see.
"""
from __future__ import annotations

import os
import re

import numpy as np

from .. import env as E
from ..az import guide as G
from ..llm_strategist import (BLIND, GOLD_BLINDS, HANDS, cons_key, describe_action, describe_state, interest, money_note, name_of, text_of)
_SURV = __import__("json").load(open(os.path.join(os.path.dirname(__file__), "survival_yellow.json")))


def survival(share):
    """P(clear the next blind) from the engine's share, calibrated for this setting (survival_yellow.json)."""
    return float(np.interp(share, _SURV["x"], _SURV["p"]))

PLAYBOOK = open(os.path.join(os.path.dirname(__file__), "playbook.md"), encoding="utf-8").read()
PLAYBOOK_ID = "playbook_v1_yellow"

SYSTEM = f"""You are the strategist in a game of Balatro on GOLD STAKE with the Yellow Deck. Your goal is to WIN THE RUN: beat the Boss Blind of
ante 8.

You make every decision outside a round: the shop (buy, sell, reroll, vouchers, packs, using consumables), picks from booster packs, and
selecting or skipping blinds. Inside a round, an honest round player plays the cards: it tries each sensible play and discard against many
possible draw orders (it never sees the real draws) and picks the move that clears the blind most often, then the one that leaves the most
money. Your job is the build: jokers, hand levels, deck, money.

You are also asked during a round (ROUND messages) at its start and whenever a consumable could be used. There you choose:
- whether to use a consumable now (a tarot on chosen cards, a planet, a spectral) or let the round player go on, and
- the STYLE of play for the rest of the round, which the round player follows as long as it does not lower the chance to clear:
  "grow_joker": the number of a joker whose counter should grow this round (Green Joker: play hands, no discards; Ride the Bus: no face
  card in scoring hands; Square: 4-card plays; Runner: straights; Spare Trousers: two pair; Obelisk: not the most played hand; ...), or 0;
  "hand": a poker hand to play when it keeps pace (to level up with planets, feed Supernova, Card Sharp, To Do List, Telescope), or "";
  "discard": a rank or suit to throw away when discarding (Hit the Road: "Jack"; Castle: its suit; Mail-In Rebate: its rank), or "".

Each message gives the state, the engine's estimates and the legal moves numbered. Outside a round it also gives CHECKPOINTS (the
playbook's targets applied to this state) and the plan note you wrote last time (your memory across decisions: keep it consistent unless
the shops give a clearly better direction).

Answer in JSON:
- "reasoning": 2-6 short sentences: what the build needs now (by the playbook's check: survival, multiplier, levels, money, slots, plan)
  and why this move is the best of the legal ones.
- "plan": your updated plan note (a few sentences: the build you are going for, what you are saving for, what to look for next). Outside
  rounds only.
- "move": the number of the move you choose.
- "targets": only when the move uses a tarot or spectral that acts on selected cards: the numbers of those cards (as listed in HAND or in
  the pack's cards); otherwise [].
- "style": only in ROUND messages: {{"grow_joker": number or 0, "hand": "hand name" or "", "discard": "rank or suit" or ""}}.

{PLAYBOOK}"""

SYSTEM_BASE = SYSTEM
LESSONS_MD = os.path.join(os.path.dirname(__file__), "lessons.md")


def lessons_text() -> str:
    """the lessons learned from our games (lm/lessons.py), if any"""
    if os.path.exists(LESSONS_MD):
        t = open(LESSONS_MD, encoding="utf-8").read().strip()
        if "\n- " in t:
            return t
    return ""


def lessons_version() -> str:
    m = re.search(r"version (\d+)", lessons_text())
    return m.group(1) if m else "0"


def system_prompt() -> str:
    """the system prompt with the current lessons appended"""
    t = lessons_text()
    if not t:
        return SYSTEM_BASE
    return (SYSTEM_BASE + "\n\n" + t + "\nThese lessons come from reviewing our own lost games; follow them unless the situation is "
            "clearly different.")


SYSTEM = system_prompt()

SCHEMA = {"type": "object", "properties": {"reasoning": {"type": "string"}, "plan": {"type": "string"}, "move": {"type": "integer"},
                                           "targets": {"type": "array", "items": {"type": "integer"}}},
          "required": ["reasoning", "plan", "move", "targets"], "additionalProperties": False}
ROUND_SCHEMA = {"type": "object", "properties": {
    "reasoning": {"type": "string"}, "move": {"type": "integer"}, "targets": {"type": "array", "items": {"type": "integer"}},
    "style": {"type": "object", "properties": {"grow_joker": {"type": "integer"}, "hand": {"type": "string"}, "discard": {"type": "string"}},
              "required": ["grow_joker", "hand", "discard"], "additionalProperties": False}},
    "required": ["reasoning", "move", "targets", "style"], "additionalProperties": False}

LEVEL_TARGET = {1: "1-2", 2: "2-3", 3: "3-5", 4: "3-5", 5: "6-9", 6: "6-9", 7: "about 12", 8: "about 12"}
BOSS_MULT = {"wall": 4.0, "violet_vessel": 6.0}


RETRIGGER = {"hanging_chad", "hack", "dusk", "sock_and_buskin", "selzer", "mime"}

_FILLED = os.path.join(os.path.dirname(__file__), "..", "data", "item_text_filled.json")
if os.path.exists(_FILLED):                                  # the descriptions with the game's numbers filled in (tools/build_item_text_filled.py)
    import json as _json
    from .. import llm_strategist as _LS
    _filled = _json.load(open(_FILLED, encoding="utf-8"))
    for _v in _filled.values():                                # a held joker's real counter is shown as 'current value' (llm_strategist.joker_str)
        _v["text"] = re.sub(r"\s*\(Currently [^)]*\)", "", _v["text"]).strip()
    _LS.TEXT.update(_filled)


def role(key: str) -> str:
    if key in RETRIGGER:
        return "retrigger"
    for name, st in (("xmult", G.XMULT), ("scaler", G.SCALERS), ("econ", G.ECON), ("mult", G.MULT), ("chips", G.CHIPS)):
        if key in st:
            return name
    return "other"


def boss_factor(d) -> float:
    """the coming blind's target / this ante's boss target"""
    bi = min(d["blind_idx"], 2)
    return {0: 1.0, 1: 1.5}.get(bi, 2.0) / BOSS_MULT.get(d["boss"], 2.0)


def checkpoints(d, fp=None) -> str:
    """The playbook's targets applied to this state. fp: the engine's fast_power row for this game ([share, max share, mean best, max best])."""
    a = int(d["ante"])
    js = d["jokers"]
    roles = [role(j["key"]) for j in js]
    rentals = sum(1 for j in js if j.get("rental"))
    L = ["CHECKPOINTS (the playbook's targets for this state):"]
    bank = 25 + 3 * rentals
    m = int(d["money"])
    if a >= 3:
        L.append(f"- Money ${m} vs the ${bank} bank target ({'met' if m >= bank else f'${bank - m} short'}; $25 + $3 per rental, from ante 3).")
    else:
        L.append(f"- Money ${m}: build toward $25 by ante 3 (interest now ${interest(m, d)} a round).")
    nx = roles.count("xmult")
    need = 2 if a >= 7 else 1 if a >= 5 else 0
    xs = ", ".join(name_of("j_" + j["key"]) for j, r in zip(js, roles) if r == "xmult") or "none"
    L.append(f"- xMult jokers: {nx} ({xs}); target {'at least ' + str(need) + ' now' if need else 'the first one by ante 4-5 (the top buy from ante 3)'}"
             + (" - BEHIND" if nx < need else "") + ".")
    played = d.get("hand_played") or [0] * 12
    lv = d["hand_levels"]
    main = int(np.argmax(played)) if max(played) > 0 else int(np.argmax(lv))
    L.append(f"- Main hand (most played): {HANDS[main]}, level {lv[main]}; planet-plan target by this ante: level {LEVEL_TARGET.get(a, 'about 12')}.")
    L.append(f"- Jokers by role: " + (", ".join(f"{name_of('j_' + j['key'])} = {r}" for j, r in zip(js, roles)) if js else "none")
             + f"; rentals held: {rentals}" + (" (playbook: none before ante 5)" if rentals and a < 5 else "") + ".")
    deck = d.get("full_deck") or []
    if deck:
        enh = sum(1 for c in deck if (c[2] if isinstance(c, (list, tuple)) else 0))
        L.append(f"- Deck: {len(deck)} cards, {enh} enhanced.")
    if fp is not None and a >= 7:
        best = float(fp[2])
        L.append(f"- Violet Vessel check (ante 8 boss, 1,200,000 in 4 hands = 300,000 per hand): best hand now about {best:,.0f} "
                 f"({best / 300000:.2f}x of what Vessel needs per hand).")
    return "\n".join(L)


def estimate(d, fp, sim=None) -> str:
    """fp: the engine's fast_power row (best hand); sim: (p_next, p_boss) from lm/rounds.ClearSim (the coming blinds played in simulated
    futures, boss effects included)."""
    if fp is None and sim is None:
        return ""
    bi = min(d["blind_idx"], 2)
    parts = []
    if fp is not None:
        parts.append(f"best hand now scores about {float(fp[2]):,.0f} chips")
    if sim is not None:
        p_next, p_boss = sim[0], sim[1]
        nb = BLIND[bi] + (f" ({name_of('bl_' + d['boss'])})" if bi == 2 else "")
        parts.append(f"chance to clear the next blind, the {nb}: {100 * p_next:.0f}%")
        if bi < 2:
            parts.append(f"to get through this ante's Boss Blind too ({name_of('bl_' + d['boss'])}): about {5 * round(20 * p_boss):.0f}% (+-10)")
    if sim is not None and len(sim) > 2 and sim[2]:
        b0 = int(d["furthest_blind"])
        steps, safe = [], None
        for k, p in enumerate(sim[2]):
            b = b0 + k
            an, bl = b // 3 + 1, b % 3
            tgt = GOLD_BLINDS[min(an, 8) - 1] * (1, 1.5, 2)[bl]
            nm = ("Boss " + (name_of("bl_" + d["boss"]) if an == d["ante"] else "(not known yet)")) if bl == 2 else ("Small", "Big")[bl]
            steps.append(f"ante {an} {nm} ({tgt:,.0f}): {5 * round(20 * p):.0f}%")
            if p >= 0.9:
                safe = f"ante {an} {nm}"
        parts.append("OUTLOOK - the chance to still be alive after each coming blind if nothing is bought: " + "; ".join(steps)
                     + (f". Safe (90%+) through: {safe}" if safe else ". Not safe even for the next blind"))
    return ("ESTIMATE (the coming blinds played out in 64 simulated futures with this build as it is - no more purchases - the boss's "
            "effect included, calibrated to the round player): " + "; ".join(parts) + ".")


def item_notes(d) -> str:
    """What the visible consumables, vouchers and tags do (jokers carry their text inline)."""
    keys = []
    for c in d.get("consumables", []):
        keys.append(cons_key(c))
    if d["state"] == "SHOP":
        for it in d["shop"]:
            k = it.get("key", "")
            if not k.startswith("j_") and it.get("kind") != "joker":
                keys.append(k)
        keys += [v["key"] for v in d.get("shop_vouchers", [])]
    if d["state"] == "BLIND_SELECT" and d["blind_idx"] < 2:
        keys.append("tag_" + d["tags_offered"][d["blind_idx"]])
    out, seen = [], set()
    for k in keys:
        t = text_of(k)
        if t and k not in seen and not re.match(r"c_(mercury|venus|earth|mars|jupiter|saturn|uranus|neptune|pluto|planet_x|ceres|eris)$", k):
            seen.add(k)
            out.append(f"- {name_of(k)}: {t}")
    return ("ITEMS (what they do):\n" + "\n".join(out)) if out else ""


def user_message(d, legal, plan, fp=None, notes=None, sim=None) -> str:
    notes = notes or {}
    parts = [describe_state(d, plan), checkpoints(d, fp)]
    if d["state"] == "PACK" and d.get("pack_hand"):
        parts.append("PACK HAND (numbered: the targets for a tarot you take from this pack):\n" + numbered_cards(d["pack_hand"]))
    est = estimate(d, fp, sim)
    if est:
        parts.append(est)
    it = item_notes(d)
    if it:
        parts.append(it)
    moves = "\n".join(f"{n}. {describe_action(int(a), d)}{money_note(int(a), d)}{notes.get(int(a), '')}" for n, a in enumerate(legal))
    parts.append("LEGAL MOVES:\n" + moves)
    return "\n\n".join(parts) + "\n\nAnswer with your reasoning, your updated plan note and the number of your move."


def response_text(out: dict) -> str:
    """The answer as the student writes it (the fine-tuning target): the same JSON the teacher returns."""
    import json
    return json.dumps({"reasoning": out.get("reasoning", ""), "plan": out.get("plan", ""), "move": int(out["move"])}, ensure_ascii=False)


def parse_answer(text: str, n_legal: int):
    """-> (index or None, plan or None, reasoning)"""
    import json
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None, None, ""
    try:
        out = json.loads(m.group(0))
        k = int(out["move"])
        return (k if 0 <= k < n_legal else None), str(out.get("plan", ""))[:1500], str(out.get("reasoning", ""))
    except Exception:
        return None, None, ""


SUIT_NAMES = ["Spades", "Hearts", "Clubs", "Diamonds"]
RANK_NAMES = {11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}


def numbered_cards(cards) -> str:
    from ..llm_strategist import card_str
    return "\n".join(f"  {k + 1}. {card_str(c)}" for k, c in enumerate(cards))


def pile_summary(cards) -> str:
    suits, ranks = [0, 0, 0, 0], {}
    for c in cards:
        if isinstance(c, (list, tuple)) and str(c[2]).upper() != "STONE":
            suits[int(c[1]) & 3] += 1
            ranks[int(c[0])] = ranks.get(int(c[0]), 0) + 1
    return (f"{len(cards)} cards; suits " + ", ".join(f"{SUIT_NAMES[k]} {suits[k]}" for k in range(4)) + "; ranks "
            + ", ".join(f"{RANK_NAMES.get(r, r)}:{n}" for r, n in sorted(ranks.items())))


def style_text(st) -> str:
    if not st:
        return "none (the round player only clears and keeps money)"
    parts = []
    if st.get("grow") is not None:
        parts.append(f"grow joker {st['grow'] + 1}")
    if st.get("hand") is not None:
        parts.append(f"play {HANDS[st['hand']]}")
    if st.get("discard"):
        kind, x = st["discard"]
        parts.append("discard " + (RANK_NAMES.get(x, str(x)) if kind == "rank" else SUIT_NAMES[x]))
    return ", ".join(parts) or "none"


def round_message(d, options, plan, style, est_continue=None, est_use=None) -> str:
    """A ROUND decision. options: [(action, text)] with option 0 = play on. est_*: the round player's estimated clear chance (0..1)."""
    from ..llm_strategist import joker_str
    bi = min(d["blind_idx"], 2)
    tgt, chips = float(d["target"]), float(d["chips"])
    L = [f"ROUND - ante {d['ante']}, {BLIND[bi]}" + (f" ({name_of('bl_' + d['boss'])}: {text_of('bl_' + d['boss'])})" if bi == 2 else "")
         + f". Target {tgt:,.0f}; scored {chips:,.0f}; {max(0.0, tgt - chips):,.0f} still needed. Hands left {d['hands_left']}, discards left "
         f"{d['discards_left']}. Money ${d['money']}."]
    L.append("HAND (numbered for targets):\n" + numbered_cards(d["hand"]))
    L.append("DRAW PILE: " + pile_summary(d.get("deck") or []))
    js = d["jokers"]
    L.append("JOKERS (numbered for grow_joker, left to right):\n" + ("\n".join(f"  {k + 1}. {joker_str(j)}" for k, j in enumerate(js)) if js else "  none"))
    cs = d["consumables"]
    L.append("CONSUMABLES: " + ("; ".join(f"{name_of(cons_key(c))} ({text_of(cons_key(c))})" for c in cs) if cs else "none"))
    lv, played = d["hand_levels"], d.get("hand_played") or [0] * 12
    L.append("HAND LEVELS (level, times played): " + (", ".join(f"{HANDS[h]} lvl {lv[h]} ({played[h]}x)" for h in range(12) if lv[h] > 1 or played[h] > 0) or "all level 1"))
    L.append("CURRENT STYLE: " + style_text(style))
    if est_continue is not None:
        e = f"ESTIMATE (the round player, over many possible draws): playing on clears this blind about {100 * est_continue:.0f}% of the time"
        if est_use:
            e += "; " + "; ".join(f"using {name} now (on the engine's own choice of target cards) about {100 * p:.0f}%" for name, p in est_use)
        L.append(e + ".")
    L.append("YOUR PLAN NOTE: " + (plan or "(none)"))
    L.append("LEGAL MOVES:\n" + "\n".join(f"{n}. {t}" for n, (_a, t) in enumerate(options)))
    return "\n\n".join(L) + ("\n\nAnswer with your reasoning, the move number, the target card numbers (for a tarot or spectral that needs "
                             "them, else []) and the style for the rest of the round.")


def parse_style(sty, d):
    """the answer's style object -> the round player's style dict"""
    out = {}
    if not isinstance(sty, dict):
        return out
    try:
        gj = int(sty.get("grow_joker") or 0)
        if 1 <= gj <= len(d["jokers"]):
            out["grow"] = gj - 1
    except Exception:
        pass
    h = str(sty.get("hand") or "").strip().lower()
    for k, name in enumerate(HANDS):
        if h and h == name.lower():
            out["hand"] = k
    x = str(sty.get("discard") or "").strip().lower()
    if x:
        xs = x[:-1] if x.endswith("s") and len(x) > 2 else x
        for k, nm in enumerate(SUIT_NAMES):
            if xs == nm.lower()[:-1]:
                out["discard"] = ("suit", k)
        short = {11: "j", 12: "q", 13: "k", 14: "a"}
        for r in range(2, 15):
            nm = RANK_NAMES.get(r, str(r)).lower()
            if xs == nm or x == nm or x == short.get(r, "#"):
                out["discard"] = ("rank", r)
    return out
