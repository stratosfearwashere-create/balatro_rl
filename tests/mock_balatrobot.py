"""A fake BalatroBot server backed by the simulator, speaking the same JSON-RPC API.
Used to test the bridge end-to-end without the real game. The hand is exposed in
reverse order to make sure the bridge's index mapping is exercised.

    python tests/mock_balatrobot.py --port 12346
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from balatro_rl.sim.cards import SUITS, RANK_CHARS  # noqa: E402
from balatro_rl.sim.game import Game, Consumable  # noqa: E402
from balatro_rl.sim.hands import HAND_NAMES, hand_base  # noqa: E402
from balatro_rl.sim.items import ALL_BOSSES  # noqa: E402
from balatro_rl.sim.jokers import Joker  # noqa: E402


def card_json(c, buy=0):
    return {"key": f"{SUITS[c.suit]}_{RANK_CHARS[c.rank]}", "set": "ENHANCED" if c.enh else "DEFAULT",
            "value": {"suit": SUITS[c.suit], "rank": RANK_CHARS[c.rank],
                      "effect": f"+{c.extra_chips} extra chips" if c.extra_chips else ""},
            "modifier": {"seal": c.seal or None, "edition": c.edition or None, "enhancement": c.enh or None},
            "state": {"debuff": c.debuffed, "hidden": c.hidden}, "cost": {"sell": 1, "buy": buy}}


def joker_json(j: Joker, buy=0):
    val = j.state.get("val")
    eff = f"Currently +{val} Mult" if val is not None else ""
    suits = ["Spades", "Hearts", "Clubs", "Diamonds"]
    ranks = {11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}
    rank = ranks.get(j.state.get("rank"), str(j.state.get("rank")))
    if j.key == "ancient":
        eff = f"Each played card with {suits[j.state.get('suit', 0)]} suit gives X1.5 Mult when scored"
    elif j.key == "idol":
        eff = f"Each played {rank} of {suits[j.state.get('suit', 0)]} gives X2 Mult when scored"
    elif j.key == "mail":
        eff = f"Earn $5 for each discarded {rank}, rank changes every round"
    elif j.key == "todo_list":
        eff = f"Earn $4 if poker hand is a {HAND_NAMES[j.state.get('hand', 1)]}, poker hand changes at end of round"
    return {"key": f"j_{j.key}", "set": "JOKER", "value": {"effect": eff},
            "modifier": {"edition": j.edition or None, "eternal": j.eternal, "perishable": j.perishable,
                         "rental": j.rental},
            "state": {"debuff": j.debuffed, "hidden": j.hidden}, "cost": {"sell": j.sell_value(), "buy": buy}}


def cons_json(c: Consumable, buy=0):
    return {"key": c.key, "set": c.kind.upper(), "value": {"effect": ""},
            "modifier": {"edition": "NEGATIVE" if c.negative else None},
            "state": {}, "cost": {"sell": c.sell_value(), "buy": buy}}


class Mock:
    def __init__(self):
        self.g = None
        self.round_eval = False
        self.lock = threading.Lock()
        self.calls = 0

    def pile(self):
        # like the real game: between rounds every card is back in the deck
        return self.g.deck if self.g.state == "SELECTING_HAND" else self.g.full_deck

    def api_hand(self):
        return list(reversed(self.g.hand if self.g.state != "PACK" else self.g.pack_hand))

    def sim_pos(self, i):
        n = len(self.g.hand if self.g.state != "PACK" else self.g.pack_hand)
        if not 0 <= i < n:
            raise ValueError("bad card index")
        return n - 1 - i

    def state(self):
        g = self.g
        if g is None:
            return {"state": "MENU"}
        st = {"BLIND_SELECT": "BLIND_SELECT", "SELECTING_HAND": "SELECTING_HAND", "SHOP": "SHOP",
              "PACK": "SMODS_BOOSTER_OPENED", "GAME_OVER": "GAME_OVER", "WON": "GAME_OVER"}[g.state]
        if self.round_eval:
            st = "ROUND_EVAL"
        blinds = {}
        for i, t in enumerate(["SMALL", "BIG", "BOSS"]):
            status = "UPCOMING"
            if i < g.blind_idx:
                status = "DEFEATED"
            elif i == g.blind_idx:
                status = "CURRENT" if g.state == "SELECTING_HAND" else "SELECT"
            b = {"type": t, "status": status, "score": g.blind_target(i),
                 "name": ALL_BOSSES[g.boss][0] if i == 2 else ("Small Blind" if i == 0 else "Big Blind")}
            if i < 2:
                b["tag_name"] = g.tags_offered[i].replace("_", " ").title() + " Tag"
            blinds[t.lower()] = b
        hands = {}
        for i, n in enumerate(HAND_NAMES):
            c, m = hand_base(i, g.hand_levels[i])
            hands[n] = {"level": g.hand_levels[i], "chips": c, "mult": m, "played": g.hand_played[i],
                        "played_this_round": g.hand_played_round[i]}
        shop = []
        for it in g.shop:
            if it.kind == "joker":
                shop.append(joker_json(it.joker, it.cost))
            elif it.kind == "card":
                shop.append(card_json(it.card, it.cost))
            else:
                shop.append(cons_json(Consumable(it.kind, it.key[2:]), it.cost))
        pack = []
        for x in g.pack_cards:
            if isinstance(x, Joker):
                pack.append(joker_json(x))
            elif isinstance(x, Consumable):
                pack.append(cons_json(x))
            else:
                pack.append(card_json(x))
        return {
            "state": st, "round_num": g.blinds_beaten + 1, "ante_num": g.ante, "money": g.money,
            "deck": g.deck_type, "stake": "GOLD", "won": g.state == "WON",
            "used_vouchers": {f"v_{v}": "" for v in g.vouchers}, "hands": hands,
            "round": {"hands_left": g.hands_left, "discards_left": g.discards_left,
                      "discards_used": g.discards_used_round, "reroll_cost": g.reroll_cost, "chips": g.chips},
            "blinds": blinds,
            "jokers": {"count": len(g.jokers), "limit": g.joker_slots, "cards": [joker_json(j) for j in g.jokers]},
            "consumables": {"count": len(g.consumables), "limit": g.consumable_slots,
                            "cards": [cons_json(c) for c in g.consumables]},
            "cards": {"count": len(self.pile()), "cards": [card_json(c) for c in self.pile()]},
            "hand": {"count": len(g.hand), "limit": g.effective_hand_size(), "cards": [card_json(c) for c in self.api_hand()]},
            "shop": {"cards": shop},
            "vouchers": {"cards": [{"key": g.shop_voucher.key, "set": "VOUCHER",
                                    "cost": {"buy": g.shop_voucher.cost}}] if g.shop_voucher else []},
            "packs": {"cards": [{"key": it.key + "_1", "set": "BOOSTER", "cost": {"buy": it.cost}}
                                for it in g.shop_packs]},
            "pack": {"cards": pack, "highlighted_limit": g.pack_picks},
        }

    def handle(self, method, p):
        self.calls += 1
        g = self.g
        p = p or {}
        if method == "health":
            return {"status": "ok"}
        if method == "gamestate":
            return self.state()
        if method == "menu":
            self.g = None
            return self.state()
        if method == "start":
            seed = zlib.crc32(str(p.get("seed", "0")).encode())
            self.g = Game(seed=seed, deck_type=p["deck"], stake=p["stake"])
            self.round_eval = False
            return self.state()
        if method == "cash_out":
            assert self.round_eval
            self.round_eval = False
            return self.state()
        if method == "select":
            g.select_blind()
        elif method == "skip":
            g.skip_blind()
        elif method in ("play", "discard"):
            pos = sorted(self.sim_pos(i) for i in p["cards"])
            if g.boss_active() == "cerulean_bell" and g.forced_pos() not in pos:
                raise ValueError("forced card")
            before = g.blinds_beaten
            (g.play if method == "play" else g.discard)(pos)
            if g.blinds_beaten > before and g.state == "SHOP":
                self.round_eval = True
        elif method == "buy":
            if "card" in p:
                if not g.can_buy(g.shop[p["card"]]):
                    raise ValueError("cannot afford")
                g.buy_card(p["card"])
            elif "pack" in p:
                g.buy_pack(p["pack"])
            else:
                g.buy_voucher()
        elif method == "sell":
            if "joker" in p:
                g.sell_joker(p["joker"])
            else:
                g.sell_consumable(p["consumable"])
        elif method == "reroll":
            g.reroll()
        elif method == "rearrange":
            perm = p["jokers"]
            assert sorted(perm) == list(range(len(g.jokers)))
            g.jokers = [g.jokers[i] for i in perm]
        elif method == "next_round":
            g.leave_shop()
        elif method == "use":
            for i in p.get("cards", []):
                self.sim_pos(i)
            g.use_consumable(p["consumable"])
        elif method == "pack":
            if p.get("skip"):
                g.pack_skip()
            else:
                for i in p.get("targets", []):
                    self.sim_pos(i)
                if not g.pack_pick_ok(p["card"]):
                    raise ValueError("cannot pick")
                g.pack_pick(p["card"])
        else:
            raise ValueError(f"unknown method {method}")
        return self.state()


def serve(port=12346):
    mock = Mock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with mock.lock:
                try:
                    res = {"jsonrpc": "2.0", "id": body.get("id"),
                           "result": mock.handle(body["method"], body.get("params"))}
                except Exception as e:  # noqa: BLE001
                    res = {"jsonrpc": "2.0", "id": body.get("id"),
                           "error": {"code": -32003, "message": repr(e), "data": {"name": "NOT_ALLOWED"}}}
            out = json.dumps(res).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = HTTPServer(("127.0.0.1", port), H)
    return srv, mock


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=12346)
    srv, _ = serve(ap.parse_args().port)
    srv.serve_forever()
