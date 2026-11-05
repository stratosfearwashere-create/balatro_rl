"""Balatro's shop generation, exactly: the same named random streams and pool orders as the game
(common_events.lua get_current_pool / create_card / poll_edition / get_pack / get_next_voucher_key,
UI_definitions.lua create_card_for_shop). The pools are the game's own (pools.json, dumped by running game.lua's prototype
tables); the random numbers come from seedrng.Streams. Checked against the game's LuaJIT running its real functions
(tests/test_shopgen.py).

State this needs from the game (and the game keeps up to date): `ante`, `used` (keys of cards that exist: owned jokers,
cards in the shop ...; the game marks every created card), `used_vouchers`, `pool_flags`, `banned`, `hands_played`
(hand type -> times played, for the planets of the secret hands), `showman`, `shop_vouchers`.
"""
import json
import os

from .seedrng import Streams

_P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pools.json")))
POOLS = {k: v for k, v in _P.items()}
UNAVAILABLE = None


class ShopGen:
    def __init__(self, seed: str, stake: int = 8, unlock_all: bool = True):
        """stake: the game's own level (1 white ... 8 gold)."""
        self.streams = Streams(seed)
        self.stake = stake
        self.unlock_all = unlock_all
        self.ante = 1
        self.used: set[str] = set()
        self.used_vouchers: set[str] = set()
        self.shop_vouchers: list[str] = []
        self.pool_flags: set[str] = set()
        self.banned: set[str] = set()
        self.hands_played: dict[str, int] = {}
        self.deck_enh: set[str] = set()                     # the enhancements ('m_gold' ...) some playing card has
        self.showman = False
        self.rates = {"Joker": 20, "Tarot": 4, "Planet": 4, "Base": 0, "Spectral": 0}
        self.edition_rate = 1
        self.first_shop_buffoon = False
        self.boss_counts = {b["key"]: 0 for b in POOLS["Boss"]}
        self.win_ante = 8

    # ---------------------------------------------------------------- pools
    def current_pool(self, typ, rarity=None, legendary=False, append=None):
        """-> (pool: list of keys with None for unavailable, pool key)"""
        s = self.streams
        if typ == "Joker":
            r = rarity if rarity is not None else s.pseudorandom("rarity%d%s" % (self.ante, append or ""))
            r = 4 if legendary else (3 if r > 0.95 else 2 if r > 0.7 else 1)
            start, key = POOLS["rarity%d" % r], "Joker%d%s" % (r, "" if legendary else (append or ""))
        else:
            start, key = POOLS[typ], typ + (append or "")
        pool, size = [], 0
        for v in start:
            add = False
            if typ == "Enhanced":
                add = True
            elif typ == "Tag":
                add = (v["min_ante"] or 0) <= self.ante
            elif not (v["key"] in self.used and not self.showman) and (self.unlock_all or v["unlocked"] is not False or v["rarity"] == 4):
                if v["set"] == "Voucher":
                    if v["key"] not in self.used_vouchers:
                        inc = all(r in self.used_vouchers for r in (v["requires"] or []))
                        if v["key"] in self.shop_vouchers:
                            inc = False
                        add = inc
                elif v["set"] == "Planet":
                    add = (not v["softlock"]) or self.hands_played.get(v["hand_type"], 0) > 0
                elif v["enhancement_gate"]:
                    add = v["enhancement_gate"] in self.deck_enh
                else:
                    add = True
                if v["key"] in ("c_black_hole", "c_soul"):
                    add = False
            if v.get("no_pool_flag") and v["no_pool_flag"] in self.pool_flags:
                add = False
            if v.get("yes_pool_flag") and v["yes_pool_flag"] not in self.pool_flags:
                add = False
            if add and v["key"] not in self.banned:
                pool.append(v["key"])
                size += 1
            else:
                pool.append(UNAVAILABLE)
        if size == 0:
            pool = [{"Tarot": "c_strength", "Tarot_Planet": "c_strength", "Planet": "c_pluto", "Spectral": "c_incantation",
                     "Voucher": "v_blank"}.get(typ, "j_joker")]
        return pool, key + ("" if legendary else str(self.ante))

    def _element(self, pool, key):
        """pseudorandom_element over a list: math.random(#pool) after seeding from the key's stream."""
        return pool[int(self.streams.pseudorandom(key, 1, len(pool))) - 1]

    def pick(self, typ, rarity=None, legendary=False, append=None):
        pool, key = self.current_pool(typ, rarity, legendary, append)
        center = self._element(pool, key)
        it = 1
        while center is UNAVAILABLE:
            it += 1
            center = self._element(pool, "%s_resample%d" % (key, it))
        return center

    # ---------------------------------------------------------------- cards
    def poll_edition(self, key, mod=1, no_neg=False, guaranteed=False):
        p = self.streams.pseudorandom(key)
        if guaranteed:
            if p > 1 - 0.003 * 25 and not no_neg: return "negative"
            if p > 1 - 0.006 * 25: return "polychrome"
            if p > 1 - 0.02 * 25: return "holo"
            if p > 1 - 0.04 * 25: return "foil"
            return None
        if p > 1 - 0.003 * mod and not no_neg: return "negative"
        if p > 1 - 0.006 * self.edition_rate * mod: return "polychrome"
        if p > 1 - 0.02 * self.edition_rate * mod: return "holo"
        if p > 1 - 0.04 * self.edition_rate * mod: return "foil"
        return None

    def create_card(self, typ, area="shop", legendary=False, rarity=None, forced_key=None, key_append=None, soulable=False):
        """Returns {'key', 'edition', 'eternal', 'perishable', 'rental'} (flags as requested; the card may refuse them)."""
        s = self.streams
        if not forced_key and soulable and "c_soul" not in self.banned:
            if typ in ("Tarot", "Spectral", "Tarot_Planet") and not ("c_soul" in self.used and not self.showman):
                if s.pseudorandom("soul_%s%d" % (typ, self.ante)) > 0.997:
                    forced_key = "c_soul"
            if typ in ("Planet", "Spectral") and not ("c_black_hole" in self.used and not self.showman):
                if s.pseudorandom("soul_%s%d" % (typ, self.ante)) > 0.997:
                    forced_key = "c_black_hole"
        if typ == "Base":
            raise NotImplementedError("playing cards in the shop")
        key = forced_key if forced_key and forced_key not in self.banned else self.pick(typ, rarity, legendary, key_append)
        card = {"key": key, "edition": None, "eternal": False, "perishable": False, "rental": False}
        if key.startswith("j_") or typ == "Joker":
            if area in ("shop", "pack"):
                poll = s.pseudorandom(("packetper" if area == "pack" else "etperpoll") + str(self.ante))
                if self.stake >= 4 and poll > 0.7:
                    card["eternal"] = True
                elif self.stake >= 7 and 0.4 < poll <= 0.7:
                    card["perishable"] = True
                if self.stake >= 8 and s.pseudorandom(("packssjr" if area == "pack" else "ssjr") + str(self.ante)) > 0.7:
                    card["rental"] = True
            card["edition"] = self.poll_edition("edi%s%d" % (key_append or "", self.ante))
        self.used.add(key)
        return card

    def shop_card(self):
        """create_card_for_shop(G.shop_jokers) without tags."""
        s = self.streams
        total = sum(self.rates.values())
        polled = s.pseudorandom("cdt%d" % self.ante) * total
        check = 0
        for typ in ("Joker", "Tarot", "Planet", "Base", "Spectral"):
            v = self.rates[typ]
            if polled > check and polled <= check + v:
                if typ == "Base":
                    return {"type": "Base", "key": None}        # (playing cards in the shop: not on the real streams yet)
                card = self.create_card(typ, "shop", key_append="sho")
                card["type"] = typ
                return card
            check += v
        raise RuntimeError("no slot type")

    # ---------------------------------------------------------------- packs, vouchers
    def pack(self, key="shop_pack", typ=None):
        """get_pack: the first shop of a run always opens with a Buffoon pack (the game's unseeded math.random(1, 2))."""
        if not self.first_shop_buffoon and "p_buffoon_normal_1" not in self.banned:
            self.first_shop_buffoon = True
            return "p_buffoon_normal_%d" % self.streams.rng.randint(1, 2)
        boosters = POOLS["Booster"]
        cume = sum((v["weight"] or 1) for v in boosters if (not typ or typ == v["kind"]) and v["key"] not in self.banned)
        poll = self.streams.pseudorandom((key or "pack_generic") + str(self.ante)) * cume
        it = 0
        for v in boosters:
            if v["key"] in self.banned:
                continue
            w = v["weight"] or 1
            if not typ or typ == v["kind"]:
                it += w
            if it >= poll and it - w <= poll:
                return v["key"]
        return None

    def next_voucher(self, from_tag=False):
        pool, key = self.current_pool("Voucher")
        if from_tag:
            key = "Voucher_fromtag"
        center = self._element(pool, key)
        it = 1
        while center is UNAVAILABLE:
            it += 1
            center = self._element(pool, "%s_resample%d" % (key, it))
        return center

    def next_tag(self):
        """get_next_tag_key(): the key of a tag ('tag_economy' ...)."""
        pool, key = self.current_pool("Tag")
        tag = self._element(pool, key)
        it = 1
        while tag is UNAVAILABLE:
            it += 1
            tag = self._element(pool, "%s_resample%d" % (key, it))
        return tag

    def new_boss(self):
        """get_new_boss(): the key of the boss for self.ante ('bl_hook' ...). Only bosses used least often are eligible; the
        pick is over the eligible keys in alphabetical order."""
        a = max(1, self.ante)
        elig = []
        for b in POOLS["Boss"]:
            if not b["showdown"] and b["min"] <= a and (a % self.win_ante != 0 or self.ante < 2):
                elig.append(b["key"])
            elif b["showdown"] and self.ante % self.win_ante == 0 and self.ante >= 2:
                elig.append(b["key"])
        elig = [k for k in elig if k not in self.banned]
        low = min(self.boss_counts[k] for k in elig)
        elig = sorted(k for k in elig if self.boss_counts[k] <= low)
        boss = elig[int(self.streams.pseudorandom("boss", 1, len(elig))) - 1]
        self.boss_counts[boss] += 1
        return boss
