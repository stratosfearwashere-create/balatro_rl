"""Builds real_plays.json from the recorded real-game trajectories published in Attol8/balatro-ai.

Source: https://github.com/Attol8/balatro-ai, directory `evidence/` (the `trajectory.jsonl.gz` files of nine
games played on the real Balatro client through BalatroBot). That repository's NOTICE and LICENSE-DOCS place
`docs/`, `evidence/` and `benchmarks/results/` under CC BY 4.0 (its source code is AGPL-3.0; none of it is
used, copied or run here). This script only reads the recorded data: for every hand played it keeps what our
scorer needs of the state before the play, and the chips the real game then awarded.

    python tests/data/real_plays_build.py <path to their evidence directory>

Not a test; it is kept so the derived file can be rebuilt and audited.
"""
import glob
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from balatro_rl.sim.cards import Card, CHAR_RANKS, SUITS  # noqa: E402
from balatro_rl.sim.hands import HAND_NAMES, evaluate  # noqa: E402
from balatro_rl.sim.items import BOSS_BY_NAME, PLANETS  # noqa: E402

HN = {n: i for i, n in enumerate(HAND_NAMES)}
OUT = os.path.join(os.path.dirname(__file__), "real_plays.json")
# jokers whose running value the scorer needs; a play is left out when the recording doesn't show it
NEEDS_VALUE = {"green_joker", "ride_the_bus", "runner", "square", "trousers", "wee", "ice_cream", "obelisk",
               "loyalty_card", "lucky_cat", "vampire", "castle", "red_card", "flash", "glass", "campfire",
               "madness", "hit_the_road", "caino", "idol", "ancient"}


def card(c):
    """[rank, suit, enhancement, edition, seal, extra chips, debuffed], or None for a face-down card."""
    if not c or "rank" not in c:
        return None
    stone = c.get("enhancement") == "STONE" or c["rank"] not in CHAR_RANKS
    return [2 if stone else CHAR_RANKS[c["rank"]], 0 if stone else SUITS.index(c["suit"]),
            "STONE" if stone else (c.get("enhancement") or ""),
            (c.get("edition") or "").replace("HOLOGRAPHIC", "HOLO"), c.get("seal") or "",
            c.get("permanent_bonus") or 0, int(bool(c.get("debuffed")))]


def joker(j):
    """([key, edition, debuffed, sell value, state], tarots used or None, blinds skipped or None)."""
    key = j["key"][2:]
    r = j.get("runtime") or {}
    state = {}
    for f in ("current_mult", "current_chips", "current_x_mult"):
        if r.get(f) is not None:
            state["val"] = r[f]
    tarots = skipped = None
    if key == "fortune_teller":                    # +1 mult per Tarot used: the game shows the total
        tarots = r.get("current_mult")
        if tarots is None:
            m = re.search(r"Currently \+(\d+)", j.get("effect_text") or "")
            tarots = int(m.group(1)) if m else None
        if tarots is None:
            raise ValueError("fortune teller value unknown")
    if key == "throwback" and r.get("current_x_mult") is not None:
        skipped = round((r["current_x_mult"] - 1) / 0.25)
    if key in NEEDS_VALUE and "val" not in state:
        raise ValueError(f"{key} value unknown")
    row = [key, (j.get("edition") or "").replace("HOLOGRAPHIC", "HOLO"), int(bool(j.get("debuffed"))),
           j.get("sell_cost") or 0, state]
    return row, tarots, skipped


def main(root):
    plays, left_out = [], {}
    mouth = {}                                     # (run, round) -> the first hand type played under The Mouth
    for path in sorted(glob.glob(os.path.join(root, "*", "**", "trajectory.jsonl.gz"), recursive=True)):
        parts = path.replace("\\", "/").split("/")
        run, seg = parts[parts.index("evidence") + 1] if "evidence" in parts else parts[-4], parts[-2]
        for idx, line in enumerate(gzip.open(path, "rt", encoding="utf-8")):
            row = json.loads(line)
            if row.get("event") != "transition" or (row.get("action") or {}).get("type") != "play_cards":
                continue
            b, a = row["before"], row["after"]
            chips = a["round"]["chips"] - b["round"]["chips"]
            hand = [card(c) for c in b["hand"]]
            sel = sorted(row["action"]["cards"])
            cur = [x for x in b["blinds"] if x["status"] == "CURRENT"]
            boss = BOSS_BY_NAME.get(cur[0]["name"], "") if cur and cur[0]["kind"] == "BOSS" else ""
            disabled = int(bool(cur and cur[0]["disabled"]))
            rkey = (run, b["round_no"])
            one = [HN[s["name"]] for s in b["hand_stats"] if s["played_this_round"] > 0]
            if boss == "mouth" and rkey not in mouth and len(one) == 1:
                mouth[rkey] = one[0]               # the recording starts after the round's first hand
            if boss == "mouth" and rkey not in mouth and all(hand[i] is not None for i in sel):
                keys = {j.get("key", "")[2:] for j in b["jokers"] if not j.get("debuffed")}
                cs = [Card(hand[i][0], hand[i][1], enh=hand[i][2]) for i in sel]
                mouth[rkey] = evaluate(cs, "four_fingers" in keys, "shortcut" in keys, "smeared" in keys).hand
                first = True
            else:
                first = False
            why = None
            if a.get("round_no") != b.get("round_no") or chips < 0:
                why = "round changed"
            elif any(hand[i] is None for i in sel):
                why = "face-down card played"
            elif any("key" not in j for j in b["jokers"]):
                why = "face-down jokers"
            jokers, tarots, skipped = [], 0, 0
            if why is None:
                try:
                    for j in b["jokers"]:
                        jr, t, s = joker(j)
                        jokers.append(jr)
                        tarots = t if t is not None else tarots
                        skipped = s if s is not None else skipped
                except ValueError as e:
                    why = str(e)
            if why is not None:
                left_out[why] = left_out.get(why, 0) + 1
                continue
            levels, played, played_round = [1] * 12, [0] * 12, [0] * 12
            for s in b["hand_stats"]:
                h = HN[s["name"]]
                levels[h], played[h], played_round[h] = s["level"], s["played"], s["played_this_round"]
            fd = [e["card"].get("enhancement") for e in b["full_deck"] for _ in range(e["count"])]
            plays.append({
                "src": f"{run} {seg}:{idx}", "deck": b["deck"], "stake": b["stake"], "ante": b["ante"],
                "boss": boss, "boss_disabled": disabled,
                "mouth": -1 if (boss != "mouth" or first) else mouth.get(rkey, -1),
                "money": b["money"], "hands_left": b["round"]["hands_left"],
                "discards_left": b["round"]["discards_left"], "joker_slots": b["joker_limit"],
                "deck_len": b["draw_count"],
                "full": [len(fd), sum(e == "STEEL" for e in fd), sum(e == "STONE" for e in fd),
                         sum(bool(e) for e in fd)],
                "vouchers": sorted(v[2:] for v in b["used_vouchers"]),
                "planets": [c["key"][2:] for c in b["consumables"] if c["key"][2:] in PLANETS],
                "levels": levels, "played": played, "played_round": played_round,
                "tarots_used": tarots, "skipped": skipped, "jokers": jokers,
                "hand": hand, "play": sel, "chips": chips,
            })
    doc = {
        "attribution": "Derived from the real-game trajectories in Attol8/balatro-ai (https://github.com/Attol8/"
                       "balatro-ai, directory evidence/), copyright (C) 2026 Attol8, licensed CC BY 4.0 "
                       "(https://creativecommons.org/licenses/by/4.0/). Changes: reduced to the hands played, "
                       "re-encoded in this project's own format (see real_plays_build.py).",
        "left_out": left_out,
        "plays": plays,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(len(plays), "plays written;", "left out:", left_out)


if __name__ == "__main__":
    main(sys.argv[1])
