"""Play the real Balatro with a trained model, through the BalatroBot mod.

Setup (once): install BalatroBot (https://github.com/coder/balatrobot) so the game
serves its JSON-RPC API on http://127.0.0.1:12346, then:

    python -m balatro_rl.bridge --model checkpoints/ppo_best.pt --deck RED --stake GOLD --runs 5

The real game state is converted into the simulator's `Game` object, encoded with the
same features used in training, and the chosen action is sent back as an API call.
Things the simulator doesn't model (unknown jokers, hand sizes above 8, ...) are mapped
to the closest supported representation.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
import urllib.request

import numpy as np

from .env import (encode, subset_of, Counters, A_PLAY, A_DISC, A_SELECT, A_SKIP, A_REROLL_BOSS, A_BUY,
                  A_BUY_PACK, A_VOUCHER, A_REROLL, A_LEAVE, A_SELL_J, A_SELL_C, A_USE_C, A_PICK, A_PSKIP,
                  A_SWAP, N_ACTIONS, describe_action)
from .heuristic import HeuristicPolicy
from .sim.cards import Card, CHAR_RANKS, SUITS, sort_hand
from .sim.game import Game, ShopItem, Consumable, MAX_HAND, MAX_SHOP
from .sim.hands import HAND_NAMES
from .sim.items import PLANETS, TAROTS, SPECTRALS, BOSS_BY_NAME, TAGS, PACKS
from .sim.jokers import Joker, JOKERS


class RPCError(Exception):
    pass


class Client:
    def __init__(self, url="http://127.0.0.1:12346", timeout=30):
        self.url = url
        self.timeout = timeout
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        body = {"jsonrpc": "2.0", "method": method, "id": self._id}
        if params:
            body["params"] = params
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read())
        if "error" in resp:
            raise RPCError(f"{method} {params}: {resp['error']}")
        return resp["result"]


# ------------------------------------------------------------------ state conversion
def _cards(area):
    if not area:
        return []
    return area.get("cards", []) if isinstance(area, dict) else list(area)


def _mod(c, k):
    return (c.get("modifier") or {}).get(k)


_EXTRA = re.compile(r"\+(\d+) (?:extra|bonus) chips", re.I)


def parse_card(c) -> Card:
    val = c.get("value") or {}
    key = c.get("key", "")
    suit = val.get("suit") or (key.split("_")[0] if "_" in key else "S")
    rank = val.get("rank") or (key.split("_")[1] if "_" in key else "A")
    card = Card(CHAR_RANKS.get(str(rank), 14), SUITS.index(suit) if suit in SUITS else 0)
    card.enh = (_mod(c, "enhancement") or "").upper()
    card.edition = (_mod(c, "edition") or "").upper()
    card.seal = (_mod(c, "seal") or "").upper()
    card.debuffed = bool((c.get("state") or {}).get("debuff"))
    card.hidden = bool((c.get("state") or {}).get("hidden"))
    m = _EXTRA.search(val.get("effect") or "")
    if m:
        card.extra_chips = int(m.group(1))
    if card.enh not in ("", "BONUS", "MULT", "WILD", "GLASS", "STEEL", "STONE", "GOLD", "LUCKY"):
        card.enh = ""
    if card.edition == "HOLOGRAPHIC":
        card.edition = "HOLO"
    return card


_SUIT_WORDS = {"spades": 0, "hearts": 1, "clubs": 2, "diamonds": 3}
_SUIT_RE = re.compile(r"\b(spades|hearts|clubs|diamonds)\b", re.I)
_RANK_WORDS = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "jack": 11,
               "queen": 12, "king": 13, "ace": 14}
_RANK_RE = re.compile(r"(?:discarded|played)\s+(10|[2-9]|jack|queen|king|ace)s?\b", re.I)
_NUM = re.compile(r"[Cc]urrently\s*[:]?\s*([+X]?)\$?(-?\d+(?:\.\d+)?)")


def parse_joker(c) -> Joker:
    key = c.get("key", "j_unknown")[2:]
    j = Joker(key if key in JOKERS else "unknown")
    j.edition = (_mod(c, "edition") or "").upper().replace("HOLOGRAPHIC", "HOLO")
    j.eternal = bool(_mod(c, "eternal"))
    per = _mod(c, "perishable")
    j.perishable = int(per) if per not in (None, False) else None
    j.rental = bool(_mod(c, "rental"))
    j.debuffed = bool((c.get("state") or {}).get("debuff"))
    j.hidden = bool((c.get("state") or {}).get("hidden"))
    sell = (c.get("cost") or {}).get("sell", 1)
    j.base_cost = int(sell) * 2
    eff = (c.get("value") or {}).get("effect", "") or ""
    m = _NUM.search(eff)
    if m:
        v = float(m.group(2))
        j.state["val"] = v
    # jokers whose target changes every round: read it from the description text
    if j.key in ("ancient", "idol", "castle"):
        sm = _SUIT_RE.search(eff)
        if sm:
            j.state["suit"] = _SUIT_WORDS[sm.group(1).lower()]
    if j.key in ("idol", "mail"):
        rm = _RANK_RE.search(eff)
        if rm:
            j.state["rank"] = _RANK_WORDS[rm.group(1).lower()]
    if j.key == "todo_list":
        for i in sorted(range(len(HAND_NAMES)), key=lambda i: -len(HAND_NAMES[i])):
            if HAND_NAMES[i].lower() in eff.lower():
                j.state["hand"] = i
                break
    return j


def parse_consumable(c) -> Consumable:
    name = c.get("key", "c_unknown")[2:]
    kind = "planet" if name in PLANETS else ("tarot" if name in TAROTS else "spectral")
    return Consumable(kind, name, negative=(_mod(c, "edition") or "").upper() == "NEGATIVE")


def _tag_key(name: str) -> str:
    k = (name or "").lower().replace(" tag", "").replace("-", "_").replace(" ", "_")
    k = {"d6": "d_six", "top_up": "top_up"}.get(k, k)
    return k if k in TAGS else "economy"


def game_from_state(gs: dict, deck="RED", stake="GOLD", memory: dict | None = None):
    """Build a simulator Game from BalatroBot's gamestate. Returns (game, hand_index_map).

    `memory` (a dict you keep between calls) remembers the full deck seen outside of rounds,
    because during a round the API only lists the cards still in the draw pile."""
    g = Game(seed=0, deck_type=gs.get("deck", deck) or deck, stake=gs.get("stake", stake) or stake)
    st = gs.get("state", "")
    g.state = {"BLIND_SELECT": "BLIND_SELECT", "SELECTING_HAND": "SELECTING_HAND", "SHOP": "SHOP",
               "SMODS_BOOSTER_OPENED": "PACK"}.get(st, st)
    g.ante = int(gs.get("ante_num", 1))
    g.money = int(gs.get("money", 0))
    rnd = gs.get("round") or {}
    g.hands_left = int(rnd.get("hands_left", 0))
    g.discards_left = int(rnd.get("discards_left", 0))
    g.discards_used_round = int(rnd.get("discards_used", 0))
    g.chips = float(rnd.get("chips", 0))
    g.reroll_cost = int(rnd.get("reroll_cost", 5))
    g.free_rerolls = 0

    # blinds
    blinds = gs.get("blinds") or {}
    blist = list(blinds.values()) if isinstance(blinds, dict) else list(blinds)
    order = {"SMALL": 0, "BIG": 1, "BOSS": 2}
    blist = sorted([b for b in blist if isinstance(b, dict)], key=lambda b: order.get(b.get("type"), 3))
    g.tags_offered = ["economy", "economy"]
    for b in blist:
        t = order.get(b.get("type"), 0)
        if t == 2:
            g.boss = BOSS_BY_NAME.get(b.get("name", ""), "")
        elif t < 2 and b.get("tag_name"):
            g.tags_offered[t] = _tag_key(b["tag_name"])
        if b.get("status") in ("CURRENT", "SELECT"):
            g.blind_idx = t
            g.target = int(b.get("score", 0))

    # hands
    hands = gs.get("hands") or {}
    for name, h in hands.items():
        if name in HAND_NAMES:
            i = HAND_NAMES.index(name)
            g.hand_levels[i] = int(h.get("level", 1))
            g.hand_played[i] = int(h.get("played", 0))
            g.hand_played_round[i] = int(h.get("played_this_round", 0))
    # The Eye / The Mouth remember hand types played this round
    g.round_hand_types = {i for i in range(len(HAND_NAMES)) if g.hand_played_round[i] > 0}
    if g.round_hand_types:
        g.mouth_hand = min(g.round_hand_types)

    # cards
    real_hand = [parse_card(c) for c in _cards(gs.get("hand"))]
    for i, c in enumerate(real_hand):
        c._real = i
    sim_hand = sort_hand(real_hand)[:MAX_HAND]      # real hands above 16 cards are cut (very rare)
    deck_cards = [parse_card(c) for c in _cards(gs.get("cards"))]
    if g.state == "PACK":
        g.pack_hand = sim_hand
        g.hand = []
    else:
        g.hand = sim_hand
    g.deck = deck_cards
    g.full_deck = real_hand + deck_cards
    if memory is not None:
        if g.state != "SELECTING_HAND":
            memory["full_deck"] = list(g.full_deck)
        elif len(memory.get("full_deck", [])) >= len(g.full_deck):
            # played/discarded cards this round aren't listed: use the deck seen before the round
            g.full_deck = memory["full_deck"]
    hand_map = [c._real for c in sim_hand]
    if isinstance(gs.get("hand"), dict) and gs["hand"].get("limit"):
        g.hand_size_override = min(MAX_HAND, int(gs["hand"]["limit"]))

    # jokers / consumables / vouchers
    g.jokers = [parse_joker(c) for c in _cards(gs.get("jokers"))]
    if isinstance(gs.get("jokers"), dict) and gs["jokers"].get("limit"):
        g.joker_slots = int(gs["jokers"]["limit"])
    g.consumables = [parse_consumable(c) for c in _cards(gs.get("consumables"))]
    if isinstance(gs.get("consumables"), dict) and gs["consumables"].get("limit"):
        g.consumable_slots = int(gs["consumables"]["limit"])
    uv = gs.get("used_vouchers") or {}
    g.vouchers = {k[2:] if k.startswith("v_") else k for k in (uv.keys() if isinstance(uv, dict) else uv)}

    # shop
    g.shop = []
    for c in _cards(gs.get("shop"))[:MAX_SHOP]:
        key = c.get("key", "")
        cost = int((c.get("cost") or {}).get("buy", 0))
        s = c.get("set", "")
        if s == "JOKER" or key.startswith("j_"):
            j = parse_joker(c)
            j.base_cost = cost
            g.shop.append(ShopItem("joker", f"j_{j.key}", cost, joker=j))
        elif s in ("TAROT", "PLANET", "SPECTRAL"):
            g.shop.append(ShopItem(s.lower(), key, cost))
        else:
            g.shop.append(ShopItem("card", "<playing_card>", cost, card=parse_card(c)))
    g.shop_packs = []
    for c in _cards(gs.get("packs"))[:2]:
        key = re.sub(r"_\d+$", "", c.get("key", "p_arcana_normal"))
        parts = key.split("_")
        kind, size = (parts[1], parts[2]) if len(parts) >= 3 else ("arcana", "normal")
        if (kind, size) not in PACKS:
            kind, size = "arcana", "normal"
        g.shop_packs.append(ShopItem("pack", f"p_{kind}_{size}", int((c.get("cost") or {}).get("buy", 4)),
                                     pack=(kind, size)))
    vs = _cards(gs.get("vouchers"))
    g.shop_voucher = None
    if vs:
        g.shop_voucher = ShopItem("voucher", vs[0].get("key", "v_unknown"), int((vs[0].get("cost") or {}).get("buy", 10)))

    # opened pack
    g.pack_cards = []
    if g.state == "PACK":
        kinds = set()
        for c in _cards(gs.get("pack")):
            s = c.get("set", "")
            if s == "JOKER":
                g.pack_cards.append(parse_joker(c)); kinds.add("buffoon")
            elif s in ("TAROT", "PLANET", "SPECTRAL"):
                g.pack_cards.append(parse_consumable(c))
                kinds.add({"TAROT": "arcana", "PLANET": "celestial", "SPECTRAL": "spectral"}[s])
            else:
                g.pack_cards.append(parse_card(c)); kinds.add("standard")
        g.pack_kind = kinds.pop() if len(kinds) == 1 else "arcana"
        g.pack_picks = int((gs.get("pack") or {}).get("highlighted_limit", 1) or 1) if isinstance(gs.get("pack"), dict) else 1
    return g, hand_map


# ------------------------------------------------------------------ action -> API call
def choose_targets(g: Game, a: int, cnt: Counters, policy) -> list[int] | None:
    """For using or picking a consumable that needs target cards: the cards the policy targets, as
    positions in the hand (or pack hand). Found by stepping a copy of the game into its targeting step."""
    g2 = copy.deepcopy(g)
    if A_USE_C <= a < A_PICK:
        g2.use_consumable(a - A_USE_C, choose_targets=True)
    elif A_PICK <= a < A_PSKIP:
        g2.pack_pick(a - A_PICK, choose_targets=True)
    if g2.targeting is None:
        return None
    obs2 = encode(g2, cnt)
    scores = np.where(obs2["mask"], policy(g2, obs2), -np.inf)
    return subset_of(obs2, int(np.argmax(scores)))


def to_rpc(g: Game, a: int, hand_map: list[int], obs: dict, targets: list[int] | None = None):
    """The BalatroBot call for action a. `targets` are the chosen target cards (positions) when the
    action uses or picks a consumable that needs them; without them the fixed rule picks the cards."""
    if a < A_SELECT:
        play = a < A_DISC
        sub = subset_of(obs, a)
        return ("play" if play else "discard"), {"cards": sorted(hand_map[p] for p in sub)}
    if A_SWAP <= a < N_ACTIONS:
        i = a - A_SWAP
        perm = list(range(len(g.jokers)))
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        return "rearrange", {"jokers": perm}
    if a == A_SELECT:
        return "select", None
    if a == A_SKIP:
        return "skip", None
    if A_BUY <= a < A_BUY_PACK:
        return "buy", {"card": a - A_BUY}
    if A_BUY_PACK <= a < A_VOUCHER:
        return "buy", {"pack": a - A_BUY_PACK}
    if a == A_VOUCHER:
        return "buy", {"voucher": 0}
    if a == A_REROLL:
        return "reroll", None
    if a == A_LEAVE:
        return "next_round", None
    if A_SELL_J <= a < A_SELL_C:
        return "sell", {"joker": a - A_SELL_J}
    if A_SELL_C <= a < A_USE_C:
        return "sell", {"consumable": a - A_SELL_C}
    if A_USE_C <= a < A_PICK:
        c = g.consumables[a - A_USE_C]
        params = {"consumable": a - A_USE_C}
        if targets is None and (TAROTS.get(c.name) or SPECTRALS.get(c.name)):
            targets = [g.hand.index(t) for t in g.auto_targets(c.name, g.hand)]
        if targets:
            params["cards"] = [hand_map[i] for i in targets]
        return "use", params
    if A_PICK <= a < A_PSKIP:
        i = a - A_PICK
        params = {"card": i}
        x = g.pack_cards[i]
        if isinstance(x, Consumable) and (TAROTS.get(x.name) or SPECTRALS.get(x.name)):
            if targets is None:
                targets = [g.pack_hand.index(t) for t in g.auto_targets(x.name, g.pack_hand)]
            if targets:
                params["targets"] = [hand_map[i] for i in targets]
        return "pack", params
    return "pack", {"skip": True}


# ------------------------------------------------------------------ main loop
def play(args):
    cli = Client(f"http://{args.host}:{args.port}")
    cli.call("health")
    if args.model == "heuristic":
        h = HeuristicPolicy(lookahead=True)
        policy = lambda g, obs: _one_hot(h.act(g, obs), len(obs["mask"]))
    else:
        import torch
        from .model import load_model, batch_obs
        model = load_model(args.model)

        def policy(g, obs):
            with torch.no_grad():
                logits, _ = model(batch_obs([obs]))
            return logits[0].numpy()

    results = []
    for run in range(args.runs):
        gs = cli.call("gamestate")
        if gs.get("state") != "MENU":
            gs = cli.call("menu")
        params = {"deck": args.deck, "stake": args.stake}
        if args.seed:
            params["seed"] = f"{args.seed}{run}" if args.runs > 1 else args.seed
        gs = cli.call("start", params)
        cnt = Counters()
        memory = {}
        while True:
            st = gs.get("state")
            if st == "GAME_OVER" or gs.get("won"):
                break
            if st == "ROUND_EVAL":
                gs = cli.call("cash_out"); cnt = Counters()
                continue
            if st not in ("BLIND_SELECT", "SELECTING_HAND", "SHOP", "SMODS_BOOSTER_OPENED"):
                time.sleep(0.2)
                gs = cli.call("gamestate")
                continue
            g, hmap = game_from_state(gs, args.deck, args.stake, memory)
            obs = encode(g, cnt)
            obs["mask"][A_REROLL_BOSS] = False      # BalatroBot has no Director's Cut call
            scores = np.where(obs["mask"], policy(g, obs), -np.inf)
            sent = False
            for a in np.argsort(-scores)[:6]:
                if not np.isfinite(scores[a]):
                    break
                targets = choose_targets(g, int(a), cnt, policy) if A_USE_C <= a < A_PSKIP else None
                method, params = to_rpc(g, int(a), hmap, obs, targets)
                if args.verbose:
                    print(f"ante {g.ante} {g.state:15s} ${g.money:<4} {describe_action(g, int(a), obs)}")
                try:
                    gs = cli.call(method, params)
                    sent = True
                    if a == A_REROLL:
                        cnt.rerolls += 1
                    elif A_SWAP <= a < N_ACTIONS:
                        cnt.swaps += 1
                    else:
                        if a == A_LEAVE:
                            cnt.rerolls = 0
                        if a in (A_LEAVE, A_SELECT, A_SKIP) or A_PLAY <= a < A_SELECT:
                            cnt.swaps = 0
                    break
                except RPCError as e:
                    print("  rejected:", e)
            if not sent:     # nothing the model wanted was accepted: fall back to a safe action
                fallback = {"SHOP": ("next_round", None), "SMODS_BOOSTER_OPENED": ("pack", {"skip": True}),
                            "BLIND_SELECT": ("select", None),
                            "SELECTING_HAND": ("play", {"cards": [0]})}[st]
                gs = cli.call(*fallback)
            time.sleep(args.delay)
        results.append({"run": run, "won": bool(gs.get("won")), "ante": gs.get("ante_num")})
        print(results[-1])
    print(json.dumps(results))


def _one_hot(a, n):
    v = np.full(n, -1e9)
    v[a] = 0.0
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="checkpoint path, or 'heuristic'")
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=12346)
    p.add_argument("--delay", type=float, default=0.0, help="seconds between actions (to watch it play)")
    p.add_argument("--verbose", action="store_true")
    play(p.parse_args())


if __name__ == "__main__":
    main()
