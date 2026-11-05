"""Builds balatro_rl/data/item_text.json: the real game's English name and description of every joker, planet, tarot, spectral card, voucher,
boss blind, tag, enhancement and edition, from the game's localization/en-us.lua (extracted from your copy of the game; read-only).
Formatting codes are removed and the numbers the game fills in (#1#, #2# ...) become '?'.

    python tools/build_item_text.py <path to en-us.lua>
"""
import json
import re
import sys

SECTIONS = ("Blind", "Joker", "Planet", "Spectral", "Tag", "Tarot", "Voucher", "Enhanced", "Edition")


def main():
    t = open(sys.argv[1], encoding="utf-8").read()
    out = {}
    for sec in SECTIONS:
        m = re.search("\n        " + sec + "=\\{\n(.*?)\n        \\},\n", t, re.S)
        if not m:
            print("missing section", sec)
            continue
        lines = m.group(1).split("\n")
        i = 0
        while i < len(lines):
            hm = re.match('^            ([a-z0-9_]+)=\\{$', lines[i])
            if not hm:
                i += 1
                continue
            key, name, body, j = hm.group(1), None, [], i + 1
            while j < len(lines) and not re.match("^            \\},?$", lines[j]):          # the entry's own closing brace
                nm = re.match('^                name="([^"]*)",', lines[j])
                if nm:
                    name = nm.group(1)
                if re.match("^                text=\\{", lines[j]):
                    k = j + 1
                    while k < len(lines) and not re.match("^                \\},?$", lines[k]):
                        body += re.findall('"([^"]*)"', lines[k])
                        k += 1
                    j = k
                j += 1
            i = j + 1
            if name is None:
                continue
            s = " ".join(body)
            s = re.sub("\\{[^}]*\\}", "", s)
            s = re.sub("#\\d+#", "?", s)
            s = re.sub("\\s+", " ", s).strip()
            out[key] = {"section": sec, "name": name, "text": s}
    counts = {s: sum(1 for v in out.values() if v["section"] == s) for s in SECTIONS}
    print(len(out), counts)
    json.dump(out, open("balatro_rl/data/item_text.json", "w", encoding="utf-8"), indent=0)


if __name__ == "__main__":
    main()
