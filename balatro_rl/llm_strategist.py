"""An LLM strategist for Gold stake: Claude makes every decision outside a round (shops, packs, blind select / skip, consumables used from
the shop); PPO v30's network plays the rounds (which cards to play and discard). Claude sees the run's state in plain words, the legal moves
numbered, the game's own description of every item (a cached reference in the system prompt), and the plan note it wrote last time - so it can
hold a build plan across shops - and answers with the number of its move and an updated plan note.

    python -m balatro_rl.llm_strategist --dry-run                       (print one decision's prompt; no API call)
    python -m balatro_rl.llm_strategist --games 5 --seed0 9000000       (needs ANTHROPIC_API_KEY or an `ant auth login` profile)

Every run is logged to --log (one json line per decision: state summary, legal moves, choice, plan note, tokens), and the results to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from . import env as E
from .engine_env import CppEnv, CppVecEnv, OBS_KEYS
from .planner import survival
from .model import load_model

TEXT = json.load(open(os.path.join(os.path.dirname(__file__), "data", "item_text.json"), encoding="utf-8"))
SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"]
RANKS = {11: "J", 12: "Q", 13: "K", 14: "A"}
HANDS = ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush",
         "Five of a Kind", "Flush House", "Flush Five"]
GOLD_BLINDS = [300, 1000, 3200, 9000, 25000, 60000, 110000, 200000]
BLIND = ["Small Blind", "Big Blind", "Boss Blind"]
MODEL = "claude-opus-5-5"


ALIAS = {"bl_violet_vessel": "bl_final_vessel", "bl_verdant_leaf": "bl_final_leaf", "bl_crimson_heart": "bl_final_heart",
         "bl_amber_acorn": "bl_final_acorn", "bl_cerulean_bell": "bl_final_bell", "tag_speed": "tag_skip"}   # engine name -> the game's key


def name_of(key: str) -> str:
    from .sim.items import PLANETS
    key = ALIAS.get(key, key)
    t = TEXT.get(key)
    nm = t["name"] if t else key
    if key.startswith("c_") and key[2:] in PLANETS:
        nm += f" (levels {HANDS[PLANETS[key[2:]]]})"
    return nm


def text_of(key: str) -> str:
    t = TEXT.get(ALIAS.get(key, key))
    return t["text"] if t else ""


def reference() -> str:
    """Every item's description, by kind (the cached part of the system prompt)."""
    parts = []
    for sec, title in (("Joker", "JOKERS"), ("Tarot", "TAROT CARDS"), ("Planet", "PLANET CARDS"), ("Spectral", "SPECTRAL CARDS"),
                       ("Voucher", "VOUCHERS"), ("Blind", "BOSS BLINDS"), ("Tag", "TAGS (taken by skipping a blind)"),
                       ("Enhanced", "CARD ENHANCEMENTS"), ("Edition", "EDITIONS")):
        rows = [f"- {v['name']}: {v['text']}" for k, v in TEXT.items() if v["section"] == sec]
        parts.append(title + "\n" + "\n".join(rows))
    return "\n\n".join(parts)


SYSTEM_RULES = f"""You are the strategist in a game of Balatro on GOLD STAKE with the Red Deck. Your goal is to WIN THE RUN: beat the Boss Blind of ante 8.

How the game is split: you make every decision outside a round - the shop (buy, sell, reroll, vouchers, packs, using consumables),
choosing cards from booster packs, and selecting or skipping blinds. A separate player plays the rounds (which cards to play and discard);
it plays them competently, so your job is the build: jokers, hand levels, deck, money.

Gold stake rules that matter: the Small Blind pays no money; 4 hands and 3 discards per round (Red Deck); jokers in shops and packs can
carry stickers - Eternal (cannot be sold), Perishable (debuffed after 5 rounds) and Rental (costs only $1 to buy but $3 at the end of every
round). Interest: $1 for every $5 you hold at the end of a round, at most $5 (more with the Seed Money / Money Tree vouchers). Unused hands
pay $1 each. Base chip targets per ante (Small Blind; Big Blind x1.5, Boss x2, some bosses more): {", ".join(f"ante {i + 1}: {v:,}" for i, v in enumerate(GOLD_BLINDS))}.
Joker order matters (left to right: put +Mult before xMult). Numbers marked '?' in the reference are the game's usual values.

STRATEGY - follow these phases:
1. SURVIVE (ante 1, early ante 2): get one or two cheap jokers that score right away (+Mult or +Chips) - just enough to clear the blinds.
   Do not spend beyond that on anything else.
2. ECONOMY (antes 2-4): build money up to $25 (the interest cap: $5 per round) and keep it there. Only spend the surplus above $25,
   unless the build would otherwise fail the next blind. Economy jokers and the Seed Money / Money Tree vouchers are good buys here.
   Do not buy booster packs or reroll with money that is below $25 - every $5 under the cap costs $1 every round from now on.
3. SCALING ENGINE (once the economy is running, usually from ante 3-4): use the surplus to find what wins Gold - xMult jokers and
   jokers that grow over the run (scalers), plus planets for one main hand. Reroll and buy packs with surplus money to look for them,
   and sell early survival jokers to make room for the engine.
Throughout: survival comes first - if the build cannot beat the next blind, spend what it takes to fix that.

Each time you are asked, you get the run's state, the legal moves (numbered) and the plan note you wrote last time. Reply with the number
of your move and an updated plan note (your strategy for this run in a few sentences: the build you are going for, what you are saving
for, what to look for in the next shops). Keep the plan consistent unless the shops give you a clearly better direction.

REFERENCE: what every item does (the game's own text)

{reference()}"""


def cons_key(c) -> str:
    """A consumable from the dump (a tuple (name, negative, ...), a 'c_...' string or a dict) -> its 'c_...' key."""
    if isinstance(c, dict):
        k = c.get("key", "")
    elif isinstance(c, (tuple, list)) and c and isinstance(c[0], str):
        k = c[0]
    else:
        k = str(c)
    return k if k.startswith("c_") else "c_" + k


def pack_item_str(x) -> str:
    """An item in an open pack: a joker (dict), a consumable ('c_...') or a playing card."""
    if isinstance(x, dict) and "key" in x and "rank" not in x:
        return joker_str(x)
    if isinstance(x, str):
        return name_of(cons_key(x)) + (f" ({text_of(cons_key(x))})" if text_of(cons_key(x)) and not cons_key(x)[2:] in __import__("balatro_rl.sim.items", fromlist=["PLANETS"]).PLANETS else "")
    return card_str(x)


def card_str(c) -> str:
    """A playing card from the dump: [rank, suit, enhancement, edition, seal, ...]."""
    if isinstance(c, dict):
        if "rank" in c:
            c = [c.get("rank"), c.get("suit"), c.get("enh") or c.get("enhancement") or "", c.get("edition") or "", c.get("seal") or ""]
        else:
            return c.get("key", str(c))
    c = list(c) + [""] * 5
    r, s, enh, ed, seal = c[0], c[1], c[2], c[3], c[4]
    if isinstance(s, str):
        suit = {"S": "Spades", "H": "Hearts", "C": "Clubs", "D": "Diamonds"}.get(s[:1].upper(), s)
    else:
        suit = SUITS[int(s)] if 0 <= int(s) < 4 else str(s)
    out = f"{RANKS.get(r, r)} of {suit}"
    extra = [x for x in (enh, ed, seal and f"{seal} seal") if x]
    return out + (f" ({', '.join(str(x).lower() for x in extra)})" if extra else "")


def joker_str(j: dict) -> str:
    key = "j_" + j["key"]
    s = name_of(key)
    tags = [t for t in (j.get("edition") or "").lower().split() if t]
    if j.get("eternal"):
        tags.append("eternal")
    if j.get("perishable") is not None:
        tags.append(f"perishable: {j['perishable']} rounds left")
    if j.get("rental"):
        tags.append("rental")
    if j.get("debuffed"):
        tags.append("DEBUFFED")
    st = j.get("state") or {}
    if "val" in st:
        tags.append(f"current value {st['val']}")
    return s + (f" [{'; '.join(tags)}]" if tags else "") + (f" ({text_of(key)})" if text_of(key) else "")


def item_str(it: dict) -> str:
    k = it.get("key", "")
    if it.get("kind") == "joker" and "joker" in it:
        return f"{joker_str(it['joker'])} - ${it['cost']}"
    nm = name_of(k)
    if k.startswith("p_"):
        nm = k[2:].replace("_", " ").title() + " Pack"
    return f"{nm} - ${it.get('cost', '?')}"


def interest_cap(d: dict) -> int:
    v = {str(x).replace("v_", "") for x in d.get("vouchers", [])}
    return 20 if "money_tree" in v else (10 if "seed_money" in v else 5)


def interest(money: int, d: dict) -> int:
    return max(0, min(money // 5, interest_cap(d)))


def action_cost(a: int, d: dict):
    """What the move costs (negative: what it pays), or None."""
    try:
        if E.A_BUY <= a < E.A_BUY_PACK:
            return d["shop"][a - E.A_BUY]["cost"]
        if E.A_BUY_PACK <= a < E.A_VOUCHER:
            return d["shop_packs"][a - E.A_BUY_PACK]["cost"]
        if E.A_VOUCHER <= a < E.A_REROLL:
            return d["shop_vouchers"][a - E.A_VOUCHER]["cost"]
        if a == E.A_REROLL:
            return 0 if d["free_rerolls"] else d["reroll_cost"]
        if E.A_SELL_J <= a < E.A_SELL_C:
            j = d["jokers"][a - E.A_SELL_J]
            return -max(1, j["cost"] // 2 + j.get("sell_bonus", 0))
        if a >= E.A_REPLACE:
            i, j = divmod(a - E.A_REPLACE, E.MAX_JOKERS)
            jj = d["jokers"][j]
            return d["shop"][i]["cost"] - max(1, jj["cost"] // 2 + jj.get("sell_bonus", 0))
    except Exception:
        return None
    return None


def money_note(a: int, d: dict) -> str:
    c = action_cost(a, d)
    if not c:
        return ""
    after = d["money"] - c
    return f"  [money ${d['money']} -> ${after}; interest ${interest(d['money'], d)} -> ${interest(after, d)} per round]"


def describe_action(a: int, d: dict) -> str:
    try:
        if a == E.A_SELECT:
            bi = d["blind_idx"]
            tgt = GOLD_BLINDS[min(d["ante"], 8) - 1] * (1 if bi == 0 else 1.5 if bi == 1 else 2)
            boss = f" - {name_of('bl_' + d['boss'])}: {text_of('bl_' + d['boss'])}" if bi == 2 else ""
            return f"Play the {BLIND[bi]} (about {int(tgt):,} chips){boss}"
        if a == E.A_SKIP:
            t = d["tags_offered"][d["blind_idx"]]
            return f"Skip the {BLIND[d['blind_idx']]} and take the {name_of('tag_' + t)} ({text_of('tag_' + t)})"
        if a == E.A_REROLL_BOSS:
            return f"Reroll the Boss Blind (now {name_of('bl_' + d['boss'])})"
        if E.A_BUY <= a < E.A_BUY_PACK:
            it = d["shop"][a - E.A_BUY]
            return f"Buy {item_str(it)}"
        if E.A_BUY_PACK <= a < E.A_VOUCHER:
            return f"Buy {item_str(d['shop_packs'][a - E.A_BUY_PACK])}"
        if E.A_VOUCHER <= a < E.A_REROLL:
            it = d["shop_vouchers"][a - E.A_VOUCHER]
            return f"Buy the voucher {name_of(it['key'])} - ${it['cost']} ({text_of(it['key'])})"
        if a == E.A_REROLL:
            return "Reroll the shop - " + ("free" if d["free_rerolls"] else f"${d['reroll_cost']}")
        if a == E.A_LEAVE:
            return "Leave the shop (go to the next blind)"
        if E.A_SELL_J <= a < E.A_SELL_C:
            j = d["jokers"][a - E.A_SELL_J]
            return f"Sell {joker_str(j)}"
        if E.A_SELL_C <= a < E.A_USE_C:
            return f"Sell {name_of(cons_key(d['consumables'][a - E.A_SELL_C]))}"
        if E.A_USE_C <= a < E.A_PICK:
            k = cons_key(d["consumables"][a - E.A_USE_C])
            return f"Use {name_of(k)}" + (f" ({text_of(k)})" if text_of(k) else "")
        if E.A_PICK <= a < E.A_PSKIP:
            x = d["pack_cards"][a - E.A_PICK]
            return f"Take {pack_item_str(x)} from the pack"
        if a == E.A_PSKIP:
            return "Skip the rest of the pack"
        if E.A_SWAP <= a < E.A_SWAP_END:
            i = a - E.A_SWAP
            return f"Swap jokers {i + 1} and {i + 2} ({joker_str(d['jokers'][i])} <-> {joker_str(d['jokers'][i + 1])})"
        if a >= E.A_REPLACE:
            i, j = divmod(a - E.A_REPLACE, E.MAX_JOKERS)
            return f"Sell {joker_str(d['jokers'][j])} and buy {item_str(d['shop'][i])} in its place"
    except Exception as ex:                                          # (a layout the describer does not know: say so rather than fail)
        return f"move {a} ({ex.__class__.__name__})"
    return f"move {a}"


def describe_state(d: dict, plan: str) -> str:
    L = []
    bi = min(d["blind_idx"], 2)
    L.append(f"ANTE {d['ante']} of 8 - next: {BLIND[bi]}. Money ${d['money']} (interest ${interest(d['money'], d)} per round now; "
             f"${interest_cap(d)} at ${5 * interest_cap(d)}+). Blinds beaten so far: {d['furthest_blind']}.")
    L.append(f"Phase: {d['state'].replace('_', ' ').lower()}.")
    L.append(f"Boss Blind of this ante: {name_of('bl_' + d['boss'])} - {text_of('bl_' + d['boss'])}")
    L.append(f"Hands {d['base_hands']}, discards {d['base_discards']} per round; hand size {d['hand_size']}.")
    js = d["jokers"]
    L.append(f"JOKERS ({len(js)}/{d['joker_slots']}, left to right): " + ("; ".join(joker_str(j) for j in js) if js else "none"))
    cs = d["consumables"]
    L.append(f"CONSUMABLES ({len(cs)}/{d['consumable_slots']}): " + (", ".join(name_of(cons_key(c)) for c in cs) if cs else "none"))
    if d["vouchers"]:
        L.append("VOUCHERS owned: " + ", ".join(name_of(v if str(v).startswith("v_") else "v_" + str(v)) for v in d["vouchers"]))
    lv = d["hand_levels"]
    played = d.get("hand_played") or [0] * 12
    L.append("HAND LEVELS (level, times played): " + ", ".join(f"{HANDS[h]} lvl {lv[h]} ({played[h]}x)" for h in range(12) if lv[h] > 1 or played[h] > 0)
             + ("" if any(lv[h] > 1 or played[h] > 0 for h in range(12)) else "all level 1"))
    deck = d.get("full_deck") or d.get("deck") or []
    if deck and isinstance(deck[0], (list, tuple)):
        suits = [0, 0, 0, 0]
        ranks = {}
        enh, eds, seals = {}, {}, {}
        for c in deck:
            suits[c[1]] += 1
            ranks[c[0]] = ranks.get(c[0], 0) + 1
            if c[2]:
                enh[c[2]] = enh.get(c[2], 0) + 1
            if c[3]:
                eds[c[3]] = eds.get(c[3], 0) + 1
            if c[4]:
                seals[c[4]] = seals.get(c[4], 0) + 1
        L.append(f"DECK: {len(deck)} cards; suits " + ", ".join(f"{SUITS[s]} {suits[s]}" for s in range(4)) + "; ranks "
                 + ", ".join(f"{RANKS.get(r, r)}:{n}" for r, n in sorted(ranks.items()))
                 + (f"; enhanced {enh}" if enh else "") + (f"; editions {eds}" if eds else "") + (f"; seals {seals}" if seals else ""))
    if d["state"] == "SHOP":
        items = [item_str(x) for x in d["shop"]] + [item_str(x) for x in d["shop_packs"]] + [f"voucher {name_of(v['key'])} ${v['cost']}" for v in d["shop_vouchers"]]
        L.append("SHOP: " + "; ".join(items) + f". Reroll: {'free' if d['free_rerolls'] else '$' + str(d['reroll_cost'])}.")
    if d["state"] == "PACK":
        L.append(f"OPEN PACK ({d['pack_kind']}), picks left {d['pack_picks']}: " + "; ".join(pack_item_str(x) for x in d["pack_cards"]))
        if d.get("pack_hand"):
            L.append("Cards drawn for the pack (targets for tarots): " + ", ".join(card_str(c) for c in d["pack_hand"]))
    if d["state"] == "BLIND_SELECT" and d["blind_idx"] < 2:
        t = d["tags_offered"][d["blind_idx"]]
        L.append(f"Skipping this blind would give: {name_of('tag_' + t)}")
    L.append("")
    L.append("YOUR PLAN NOTE FROM LAST TIME: " + (plan or "(none yet - this is the start of the run)"))
    return "\n".join(L)


class Twin:
    """The same game in the vector engine (exact mode: stepped with the same moves it stays identical), for the engine's power estimate:
    the build's best play over sampled hands, its share of the coming blind's target, and the chance real play clears that blind
    (planner.survival: measured on 10,027 Gold blinds). Also the change each build-changing move makes to that chance."""

    BUILD_MOVES = None

    def __init__(self, seed, stake):
        self.v = CppVecEnv(1, "RED", stake, seed0=seed, threads=1)
        self.v.set_exact(True)
        self.v.v.restart()
        self.v.set_auto_round(False, 1.5, 0)
        self.scr = CppVecEnv(48, "RED", stake, seed0=1, threads=1)
        self.scr.set_exact(True)
        self.scr.v.restart()
        self.scr.set_auto_round(False, 1.5, 0)
        self.ok = True
        if Twin.BUILD_MOVES is None:
            Twin.BUILD_MOVES = set(range(E.A_BUY, E.A_BUY_PACK)) | set(range(E.A_SELL_J, E.A_SELL_C)) | set(range(E.A_USE_C, E.A_PICK)) | \
                               set(range(E.A_PICK, E.A_PSKIP)) | set(range(E.A_REPLACE, E.N_ACTIONS))

    def step(self, a, mask_single):
        if not self.ok:
            return
        self.v.step(np.array([a], dtype=np.int64))

    def check(self, mask_single):
        if self.ok and not np.array_equal(np.asarray(self.v.obs["mask"])[0], np.asarray(mask_single)):
            self.ok = False                                         # (out of step: the estimates are dropped for the rest of this game)
        return self.ok

    @staticmethod
    def _boss_factor(d):
        bi = min(d["blind_idx"], 2)
        mult = {0: 1.0, 1: 1.5}.get(bi, 2.0)
        boss_mult = {"wall": 4.0, "violet_vessel": 6.0, "needle": 1.0}.get(d["boss"], 2.0)
        return mult / boss_mult

    def estimate(self, d, legal):
        """-> (state text, {move: note}) or ("", {})."""
        if not self.ok:
            return "", {}
        fp = self.v.fast_power([0], 1024)[0]
        share, best = float(fp[0]), float(fp[2])
        p_next = survival(share)
        p_boss = survival(share * self._boss_factor(d)) if min(d["blind_idx"], 2) < 2 else p_next
        txt = (f"ESTIMATE (engine, 1024 sampled hands): best hand scores about {best:,.0f} chips; with {d['base_hands']} hands that is "
               f"{share:.2f}x the next blind's target. Estimated chance to clear the next blind: {100 * p_next:.0f}%"
               + (f"; the Boss Blind of this ante: {100 * p_boss:.0f}%" if min(d["blind_idx"], 2) < 2 else "")
               + " (calibrated on 10,000 real Gold blinds played by the round player).")
        notes = {}
        moves = [int(a) for a in legal if int(a) in Twin.BUILD_MOVES][:self.scr.n]
        if moves:
            self.scr.copy_many(self.v, [0] * len(moves), list(range(len(moves))))
            act = np.full(self.scr.n, -1, dtype=np.int64)
            for k, a in enumerate(moves):
                act[k] = a
            self.scr.step(act)
            g = np.asarray(self.scr.obs["g"])
            ok = [k for k in range(len(moves)) if not np.asarray(self.scr.obs["mask"])[k, :E.A_SELECT].any()]
            if ok:
                fp2 = self.scr.fast_power(ok, 1024)
                for k, row in zip(ok, fp2):
                    p2 = survival(float(row[0]))
                    if abs(p2 - p_next) >= 0.05:
                        notes[moves[k]] = f"  [clear chance next blind {100 * p_next:.0f}% -> {100 * p2:.0f}%]"
        return txt, notes


SCHEMA = {"type": "object", "properties": {"plan": {"type": "string"}, "move": {"type": "integer"}},
          "required": ["plan", "move"], "additionalProperties": False}


class Strategist:
    def __init__(self, a):
        self.a = a
        self.round_net = load_model(a.round_net, "cuda")
        self.lock = threading.Lock()
        self.client = None
        self.calls = 0
        self.tokens = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
        self.log = open(a.log, "a", encoding="utf-8") if a.log else None

    def net_move(self, o):
        with self.lock, torch.no_grad():
            lg, _ = self.round_net({k: torch.from_numpy(np.asarray(o[k])[None]).cuda() for k in OBS_KEYS})
        lg = lg[0].float().cpu().numpy()
        return int(np.where(o["mask"], lg, -np.inf).argmax())

    def ask(self, d, legal, plan, seed, twin=None):
        est, notes = twin.estimate(d, legal) if twin is not None else ("", {})
        moves = "\n".join(f"{n}. {describe_action(int(a), d)}{money_note(int(a), d)}{notes.get(int(a), '')}" for n, a in enumerate(legal))
        user = describe_state(d, plan) + ("\n" + est if est else "") + "\n\nLEGAL MOVES:\n" + moves + "\n\nAnswer with the move number and your updated plan note."
        if self.a.dry_run:
            if d["state"] != "SHOP" or d["ante"] < 2:                # (show a shop decision of ante 2: the networks play until then)
                return None, plan, "dry run"
            print("=" * 100 + "\nSYSTEM PROMPT (cached; first 1500 of", len(SYSTEM_RULES), "characters):\n" + SYSTEM_RULES[:1500] + "\n...\n" + "=" * 100)
            print("USER MESSAGE:\n" + user)
            raise SystemExit
        with self.lock:
            if self.calls >= self.a.max_calls:
                raise RuntimeError("the --max-calls budget is used up")
            self.calls += 1
        if self.a.via == "cli":
            return self.ask_cli(user, legal, plan)
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic()
        r = self.client.beta.messages.create(
            model=self.a.model, max_tokens=8000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": self.a.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            system=[{"type": "text", "text": SYSTEM_RULES, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            messages=[{"role": "user", "content": user}])
        u = r.usage
        with self.lock:
            self.tokens["input"] += u.input_tokens or 0
            self.tokens["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            self.tokens["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            self.tokens["output"] += u.output_tokens or 0
        if r.stop_reason == "refusal":
            return None, plan, "refused"
        text = next((b.text for b in r.content if b.type == "text"), "")
        try:
            out = json.loads(text)
            k = int(out["move"])
            return (k if 0 <= k < len(legal) else None), str(out.get("plan", plan))[:1500], None
        except Exception as ex:
            return None, plan, f"bad answer: {ex}"

    def ask_cli(self, user, legal, plan):
        """The same question through the Claude Code command line (`claude -p`), which runs on the signed-in Claude account (a subscription plan)
        instead of an API key: our system prompt replaces Claude Code's, no tools, no settings or MCP servers, a structured answer."""
        import subprocess
        if not getattr(self, "sys_file", None):
            import tempfile
            f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
            f.write(SYSTEM_RULES)
            f.close()
            self.sys_file = f.name
        cmd = ["claude", "-p", user, "--system-prompt-file", self.sys_file, "--tools", "", "--safe-mode", "--strict-mcp-config",
               "--no-session-persistence", "--model", self.a.model, "--effort", self.a.effort, "--output-format", "json", "--json-schema", json.dumps(SCHEMA)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=600)
            r = json.loads(p.stdout)
        except Exception as ex:
            return None, plan, f"cli failed: {ex.__class__.__name__}"
        u = r.get("usage") or {}
        with self.lock:
            self.tokens["input"] += u.get("input_tokens", 0)
            self.tokens["cache_read"] += u.get("cache_read_input_tokens", 0)
            self.tokens["cache_write"] += u.get("cache_creation_input_tokens", 0)
            self.tokens["output"] += u.get("output_tokens", 0)
            self.cost_equiv = getattr(self, "cost_equiv", 0.0) + float(r.get("total_cost_usd") or 0)
        out = r.get("structured_output")
        if not isinstance(out, dict):
            try:
                out = json.loads(r.get("result") or "")
            except Exception:
                return None, plan, "bad answer"
        try:
            k = int(out["move"])
            return (k if 0 <= k < len(legal) else None), str(out.get("plan", plan))[:1500], None
        except Exception as ex:
            return None, plan, f"bad answer: {ex}"

    def play(self, seed):
        try:
            return self._play(seed)
        except Exception as ex:                                     # one game's failure ends only that game
            import traceback
            return {"seed": seed, "error": f"{ex.__class__.__name__}: {ex}", "trace": traceback.format_exc()[-600:], "blinds": -1, "won": False}

    def _play(self, seed):
        e = CppEnv("RED", self.a.stake)
        e.e.exact = True
        o = e.reset(seed)
        twin = Twin(seed, self.a.stake) if self.a.estimate else None
        plan = ""
        decisions = 0
        for _ in range(3000):
            d = e.game.dump()
            mask = np.asarray(o["mask"])
            in_round = mask[:E.A_SELECT].any()
            legal = np.flatnonzero(mask[E.A_SELECT:]) + E.A_SELECT
            if in_round or len(legal) <= 1:
                a = self.net_move(o) if in_round else int(legal[0])
            else:
                if twin is not None:
                    twin.check(mask)
                k, plan, err = self.ask(d, legal, plan, seed, twin)
                decisions += 1
                a = int(legal[k]) if k is not None else self.net_move(o)
                if self.log:
                    with self.lock:
                        self.log.write(json.dumps({"seed": seed, "ante": d["ante"], "blind": d["blind_idx"], "state": d["state"], "money": d["money"],
                                                   "move": describe_action(a, d), "error": err, "plan": plan}) + "\n")
                        self.log.flush()
            o, r, done, info = e.step(a)
            if twin is not None:
                twin.step(a, None)
            if done:
                dd = e.game.dump()
                return {"seed": seed, "blinds": dd["furthest_blind"], "won": dd["state"] == "WON", "decisions": decisions,
                        "twin_in_step": (twin.ok if twin is not None else None),
                        "jokers": [j["key"] for j in dd["jokers"]], "plan": plan}
        return {"seed": seed, "blinds": e.game.dump()["furthest_blind"], "won": False, "decisions": decisions, "plan": plan, "cut": True}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--seed0", type=int, default=9_000_000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--parallel", type=int, default=5, help="games played at once (each waits on its own API calls)")
    p.add_argument("--max-calls", type=int, default=400, help="a hard cap on API calls for the whole run (a spending guard)")
    p.add_argument("--round-net", default="checkpoints/ppo_gold_v30.pt")
    p.add_argument("--log", default="traces/llm_strategist.jsonl")
    p.add_argument("--estimate", type=int, default=1, help="1: show the engine's survival estimate for the state and for each build-changing move")
    p.add_argument("--via", default="api", choices=["api", "cli"], help="api: the Anthropic API (needs an API key); cli: the Claude Code command line on your signed-in Claude plan")
    p.add_argument("--dry-run", action="store_true", help="print the prompt of the first decision and stop (no API call)")
    a = p.parse_args()
    if a.log:
        os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    s = Strategist(a)
    if a.dry_run:
        s.play(a.seed0)
        return
    t = time.time()
    with ThreadPoolExecutor(a.parallel) as ex:
        res = list(ex.map(s.play, [a.seed0 + i for i in range(a.games)]))
    for r in res:
        print(json.dumps(r), flush=True)
    res_ok = [r for r in res if "error" not in r]
    bl = np.array([r["blinds"] for r in res_ok]) if res_ok else np.zeros(1)
    print(json.dumps({"games": len(res), "blinds": round(float(bl.mean()), 2), "wins": int(sum(r["won"] for r in res_ok)), "failed": len(res) - len(res_ok), "calls": s.calls,
                      "tokens": s.tokens, "cost_equivalent_usd": round(getattr(s, "cost_equiv", 0.0), 2), "minutes": round((time.time() - t) / 60, 1), "model": a.model, "effort": a.effort}), flush=True)


if __name__ == "__main__":
    main()
