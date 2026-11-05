"""Builds balatro_rl/data/item_text_filled.json: the same items, names and texts as item_text.json (see build_item_text.py), but with the
numbers the game fills in (#1#, #2# ...) replaced by the game's own BASE values, worked out the way the game does it:

  * localization/en-us.lua  -> description templates (formatting codes removed, as in build_item_text.py)
  * card.lua                -> which ability field fills #1#, #2# ... for each joker (Card:generate_UIBox_ability_table)
  * functions/common_events.lua -> the same for tarots, planets, spectrals, vouchers, enhancements, editions (generate_card_ui)
  * tag.lua                 -> the same for tags (Tag:get_uibox_table)
  * game.lua                -> each item's config (P_CENTERS, P_TAGS, P_BLINDS) and the base poker hand table (planet level-up amounts)

Values are what a freshly bought card shows: ability fields as Card:set_ability initialises them (so scaling jokers show their starting
counter), probabilities with G.GAME.probabilities.normal = 1. Values that depend on the run's state (deck contents, money, the rank/suit/
hand chosen each round ...) become a short description in brackets, e.g. "[current value]". Anything the script cannot evaluate stays '?'.

The game archive is only read (zipfile), never modified.

    python tools/build_item_text_filled.py [path to Balatro.zip]
"""
import json
import os
import re
import sys
import zipfile
from types import SimpleNamespace

ZIP_DEFAULT = r"C:\Users\strat\Desktop\Balatro.zip"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTIONS = ("Blind", "Joker", "Planet", "Spectral", "Tag", "Tarot", "Voucher", "Enhanced", "Edition")
UNKNOWN = "?"
CUR = "[current value]"

# Placeholders whose value depends on the run's state rather than on the card: (key, placeholder number) -> text.
STATE = {
    ("j_fortune_teller", 2): CUR, ("j_steel_joker", 2): CUR, ("j_stone", 2): CUR, ("j_blue_joker", 2): CUR,
    ("j_abstract", 2): CUR, ("j_bull", 2): CUR, ("j_bootstraps", 3): CUR, ("j_cloud_9", 2): CUR,
    ("j_erosion", 2): CUR, ("j_erosion", 3): "[your starting deck size]", ("j_satellite", 2): CUR,
    ("j_drivers_license", 2): "[current count]", ("j_swashbuckler", 1): CUR, ("j_throwback", 2): CUR,
    ("j_stencil", 1): CUR,
    ("j_todo_list", 2): "[a poker hand that changes every round]",
    ("j_mail", 2): "[a rank that changes every round]",
    ("j_idol", 2): "[a rank that changes every round]", ("j_idol", 3): "[a suit that changes every round]",
    ("j_ancient", 2): "[a suit that changes every round]",
    ("j_castle", 2): "[a suit that changes every round]",
    ("c_temperance", 2): CUR,
    ("tag_handy", 2): CUR, ("tag_garbage", 2): CUR, ("tag_skip", 2): CUR,
    ("tag_orbital", 1): "[a random poker hand]",
}
# Values the game computes in code at purchase time that the expression evaluator does not follow.
FIXED = {
    ("c_ectoplasm", 1): "1",                 # G.GAME.ecto_minus or 1 (grows by 1 each time Ectoplasm is used)
    ("e_negative_consumable", 1): "1",       # built in code with config = {extra = 1}
}


# ---------------------------------------------------------------- Lua literal tables
def _skip_ws(s, i):
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s.startswith("--", i):
            j = s.find("\n", i)
            i = len(s) if j < 0 else j + 1
        else:
            break
    return i


def _read_string(s, i):
    q, j, out = s[i], i + 1, []
    while s[j] != q:
        if s[j] == "\\":
            out.append(s[j + 1]); j += 2
        else:
            out.append(s[j]); j += 1
    return "".join(out), j + 1


def _read_raw(s, i):
    """Raw expression text up to the next top-level ',' or '}'."""
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c in "'\"":
            _, j = _read_string(s, j); continue
        if c in "({[":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif c == "," and depth == 0:
            break
        j += 1
    return s[i:j].strip(), j


def _value(s, i):
    i = _skip_ws(s, i)
    if s[i] == "{":
        return parse_table(s, i)
    if s[i] in "'\"":
        v, j = _read_string(s, i)
        k = _skip_ws(s, j)
        if k < len(s) and s[k] in ",}":
            return v, j
    raw, j = _read_raw(s, i)
    if raw == "true":
        return True, j
    if raw == "false":
        return False, j
    if raw == "nil":
        return None, j
    if re.fullmatch(r"[-+*/().\d\s]+", raw):
        return eval(raw, {"__builtins__": {}}), j        # numbers and constant arithmetic like 9.6/4
    return ("RAW", raw), j


def parse_table(s, i):
    """Parses the Lua table constructor starting at s[i] == '{'. Returns (dict or list, index after '}')."""
    assert s[i] == "{"
    i += 1
    d, arr = {}, []
    while True:
        i = _skip_ws(s, i)
        if s[i] == "}":
            return (d if d or not arr else arr), i + 1
        m = re.match(r"\[\s*(['\"])(.*?)\1\s*\]\s*=(?!=)", s[i:]) or re.match(r"([A-Za-z_]\w*)\s*=(?!=)", s[i:])
        if m:
            key = m.group(2) if m.lastindex == 2 else m.group(1)
            v, i = _value(s, i + m.end())
            d[key] = v
        else:
            v, i = _value(s, i)
            arr.append(v)
        i = _skip_ws(s, i)
        if s[i] in ",;":
            i += 1


def lua_tables(text, start_pat):
    """Entries `key = {...}` of the table that starts at the line matching start_pat (one level deep)."""
    m = re.search(start_pat, text)
    return parse_table(text, text.index("{", m.start()))[0]


# ---------------------------------------------------------------- localization
def parse_templates(t):
    """Like build_item_text.py, but keeps #n# placeholders."""
    out = {}
    for sec in SECTIONS:
        m = re.search("\n        " + sec + "=\\{\n(.*?)\n        \\},\n", t, re.S)
        lines = m.group(1).split("\n")
        i = 0
        while i < len(lines):
            hm = re.match('^            ([a-z0-9_]+)=\\{$', lines[i])
            if not hm:
                i += 1
                continue
            key, name, body, j = hm.group(1), None, [], i + 1
            while j < len(lines) and not re.match("^            \\},?$", lines[j]):
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
            out[key] = (sec, s)
    return out


def misc_table(t, name):
    m = re.search("\n        " + name + "=\\{", t)
    tab, _ = parse_table(t, m.end() - 1)
    return tab


# ---------------------------------------------------------------- loc_vars expressions in the Lua sources
def split_top(s):
    parts, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c in "'\"":
            _, j = _read_string(s, i)
            cur.append(s[i:j]); i = j; continue
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur).strip()); cur = []
        else:
            cur.append(c)
        i += 1
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def balanced(s, i):
    """s[i] == '{' -> index after the matching '}'."""
    depth = 0
    while True:
        c = s[i]
        if c in "'\"":
            _, i = _read_string(s, i); continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1


def loc_vars_by_name(block, name_expr):
    """Maps item name -> list of loc_vars argument expressions, for an if/elseif chain on `name_expr == '...'`."""
    out = {}
    for m in re.finditer(r"(?:\bif|\belseif)\b((?:(?!\bthen\b).)*?)\bthen\b", block, re.S):
        cond = m.group(1)
        names = re.findall(re.escape(name_expr) + r"\s*==\s*(?:'([^']*)'|\"([^\"]*)\")", cond)
        if not names:
            continue
        nxt = re.search(r"\belseif\b", block[m.end():])
        body = block[m.end(): m.end() + (nxt.start() if nxt else len(block))]
        lm = re.search(r"loc_vars\s*=\s*\{", body)
        args = None
        if lm:
            k = m.end() + lm.end() - 1
            inner = block[k + 1: balanced(block, k) - 1]
            args = [a for a in split_top(inner) if not re.match(r"^[A-Za-z_]\w*\s*=", a)]  # drop colours = {...}
        for a, b in names:
            out.setdefault(a or b, args)
    return out


def ns(v):
    if isinstance(v, dict):
        return SimpleNamespace(**{k: ns(x) for k, x in v.items()})
    return v


def fmt(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return ("%.14g" % v)
    return str(v)


class Evaluator:
    def __init__(self, loc, enh_names, tag_names):
        self.loc, self.enh, self.tag = loc, enh_names, tag_names

    def L(self, key, table):
        return self.loc[table][key]

    def NT(self, key, set_):
        return {"Enhanced": self.enh, "Tag": self.tag}[set_][key]

    def eval(self, expr, ability=None, config=None):
        """Evaluates one Lua loc_vars expression; returns None if it needs run state or cannot be followed."""
        e = expr
        e = e.replace("''..(G.GAME and G.GAME.probabilities.normal or 1)", "1")
        e = e.replace("G.GAME.probabilities.normal", "1")
        e = e.replace("(specific_vars and specific_vars.bonus_chips) or ", "")    # a fresh card has no extra bonus chips
        if "G." in e or "math." in e or "tally" in e or "localize{type = 'variable'" in e:
            return None
        e = re.sub(r"localize\{type\s*=\s*'name_text',\s*set\s*=\s*'(\w+)',\s*key\s*=\s*([^,}]+?)(?:,\s*nodes\s*=\s*\{\})?\}",
                   lambda m: "NT(%s, '%s')" % (m.group(2).strip(), m.group(1)), e)
        e = re.sub(r"localize\((.+?),\s*'(\w+)'\)", r"L(\1, '\2')", e)
        if "localize" in e or ".." in e:
            return None
        e = e.replace("self.ability.", "A.").replace("_c.config.", "C.").replace("self.config.", "C.")
        try:
            v = eval(e, {"__builtins__": {}}, {"A": ns(ability or {}), "C": ns(config or {}), "L": self.L, "NT": self.NT})
        except Exception:
            return None
        return None if v is None else fmt(v)


def joker_ability(c):
    """Card:set_ability for a freshly created joker/consumable."""
    cfg = c.get("config", {}) or {}
    a = {
        "name": c["name"], "mult": cfg.get("mult", 0), "h_mult": cfg.get("h_mult", 0), "h_x_mult": cfg.get("h_x_mult", 0),
        "h_dollars": cfg.get("h_dollars", 0), "p_dollars": cfg.get("p_dollars", 0), "t_mult": cfg.get("t_mult", 0),
        "t_chips": cfg.get("t_chips", 0), "x_mult": cfg.get("Xmult", 1), "h_size": cfg.get("h_size", 0),
        "d_size": cfg.get("d_size", 0), "extra": cfg.get("extra"), "type": cfg.get("type", ""),
    }
    if c["name"] == "Invisible Joker":
        a["invis_rounds"] = 0
    if c["name"] == "Caino":
        a["caino_xmult"] = 1
    if c["name"] == "Yorick":
        a["yorick_discards"] = a["extra"]["discards"]
    if c["name"] == "Loyalty Card":
        a["loyalty_remaining"] = a["extra"]["every"]
    return a


def section_block(text, start, end):
    i = text.index(start)
    return text[i: text.index(end, i)]


def main():
    zpath = sys.argv[1] if len(sys.argv) > 1 else ZIP_DEFAULT
    z = zipfile.ZipFile(zpath)

    def member(suffix):
        n = [x for x in z.namelist() if x.endswith(suffix)]
        return z.read(n[0]).decode("utf-8")

    en, game, card, events, tagsrc = (member("localization/en-us.lua"), member("/game.lua"), member("/card.lua"),
                                      member("functions/common_events.lua"), member("/tag.lua"))
    templates = parse_templates(en)
    loc = {k: misc_table(en, k) for k in ("poker_hands", "suits_plural", "suits_singular", "ranks", "dictionary", "v_dictionary")}

    centers = lua_tables(game, r"self\.P_CENTERS = \{")
    tags = lua_tables(game, r"self\.P_TAGS = \{")
    blinds = lua_tables(game, r"self\.P_BLINDS = \{")
    hm = re.search(r"\n\s*hands = \{\s*\n\s*\[\"Flush Five\"\]", game)       # base poker hand table (level-up amounts)
    hands, _ = parse_table(game, game.index("{", hm.start()))

    enh_names = {k: v_name(en, k) for k in centers if k.startswith("m_")}
    tag_names = {k: v_name(en, k) for k in tags}
    ev = Evaluator(loc, enh_names, tag_names)

    jok = loc_vars_by_name(section_block(card, "elseif self.ability.set == 'Joker' then", "local badges = {}"), "self.ability.name")
    voucher = loc_vars_by_name(section_block(events, "elseif _c.set == 'Voucher' then", "elseif _c.set == 'Edition' then"), "_c.name")
    spectral = loc_vars_by_name(section_block(events, "elseif _c.set == 'Spectral' then", "elseif _c.set == 'Planet' then"), "_c.name")
    tarot = loc_vars_by_name(section_block(events, "elseif _c.set == 'Tarot' then", "if main_end then"), "_c.name")
    enhanced = loc_vars_by_name(section_block(events, "elseif _c.set == 'Enhanced' then", "elseif _c.set == 'Booster' then"), "_c.effect")
    tagv = loc_vars_by_name(section_block(tagsrc, "function Tag:get_uibox_table", "tag_sprite.ability_UIBox_table"), "name_to_check")
    tagv["Speed Tag"] = tagv.get("Skip Tag")       # tag_skip is named 'Speed Tag' in en-us but 'Skip Tag' in code

    base = json.load(open(os.path.join(ROOT, "balatro_rl", "data", "item_text.json"), encoding="utf-8"))
    out, stats, unresolved = {}, {}, []
    for key, item in base.items():
        sec, tpl = templates[key]
        vals = {}                                   # placeholder number -> text
        c = centers.get(key) or tags.get(key) or blinds.get(key) or {}
        cname = c.get("name")
        args = None
        if sec == "Joker":
            args = jok.get(cname)
            ab = joker_ability(c)
            vals = {i + 1: ev.eval(a, ability=ab) for i, a in enumerate(args or [])}
            if cname == "Loyalty Card":             # 'loyalty_inactive' = "#1# remaining" with loyalty_remaining = every
                vals[3] = loc["v_dictionary"]["loyalty_inactive"].replace("#1#", fmt(ab["loyalty_remaining"]))
        elif sec in ("Tarot", "Spectral", "Voucher"):
            args = {"Tarot": tarot, "Spectral": spectral, "Voucher": voucher}[sec].get(cname)
            vals = {i + 1: ev.eval(a, config=c.get("config")) for i, a in enumerate(args or [])}
        elif sec == "Enhanced":
            args = enhanced.get(c.get("effect"))
            vals = {i + 1: ev.eval(a, config=c.get("config")) for i, a in enumerate(args or [])}
        elif sec == "Edition":
            cfg = c.get("config") if c else {"extra": 1}
            vals = {1: fmt(cfg["extra"])} if cfg and "extra" in cfg else {}
        elif sec == "Tag":
            args = tagv.get(cname)
            vals = {i + 1: ev.eval(a, config=c.get("config")) for i, a in enumerate(args or [])}
        elif sec == "Planet":
            h = hands[c["config"]["hand_type"]]
            vals = {1: "[current level]", 2: loc["poker_hands"][c["config"]["hand_type"]], 3: fmt(h["l_mult"]), 4: fmt(h["l_chips"])}
        elif sec == "Blind":
            bv = c.get("vars") or []
            for i, v in enumerate(bv):
                if isinstance(v, tuple) and v[0] == "RAW":
                    m = re.fullmatch(r"localize\('(\w+)'\)", v[1])
                    vals[i + 1] = loc["dictionary"].get(m.group(1)) if m else None
                else:
                    vals[i + 1] = fmt(v)
        for (k, n), v in list(STATE.items()) + list(FIXED.items()):
            if k == key:
                vals[n] = v

        def sub(m):
            n = int(m.group(1))
            v = vals.get(n)
            return UNKNOWN if v is None else v

        nums = [int(x) for x in re.findall(r"#(\d+)#", tpl)]
        filled = re.sub(r"#(\d+)#", sub, tpl)
        if key == "bl_wheel":                         # Blind:get_loc_debuff_text prefixes G.GAME.probabilities.normal
            filled = "1" + filled
        if key == "j_misprint":                       # text is drawn in code (main_start): '+' random in extra.min..extra.max ' Mult'
            filled = "+%s to +%s Mult (random each hand)" % (fmt(c["config"]["extra"]["min"]), fmt(c["config"]["extra"]["max"]))
        filled = re.sub(r"\s+", " ", filled).strip()
        missing = [n for n in nums if vals.get(n) is None]
        status = "none" if not nums else ("full" if not missing else ("partial" if len(set(missing)) < len(set(nums)) else "unfilled"))
        if missing:
            unresolved.append((key, sorted(set(missing)), args))
        st = stats.setdefault(sec, {"full": 0, "partial": 0, "unfilled": 0, "none": 0})
        st[status] += 1
        out[key] = {"section": item["section"], "name": item["name"], "text": filled}

    json.dump(out, open(os.path.join(ROOT, "balatro_rl", "data", "item_text_filled.json"), "w", encoding="utf-8"), indent=0)
    print(len(out), "items")
    for s, st in stats.items():
        print(" ", s, st)
    for u in unresolved:
        print("unresolved", u)


def v_name(en, key):
    m = re.search("\n            " + re.escape(key) + "=\\{\n                name=\"([^\"]*)\"", en)
    return m.group(1) if m else None


if __name__ == "__main__":
    main()
