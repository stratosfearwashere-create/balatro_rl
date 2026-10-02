"""One test per rule the simulator was corrected on, against the game's own code: scoring order, hand
detection, boss blinds, the economy, the shop, consumables and jokers."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from balatro_rl.az.actions import enumerate_candidates, target_sets  # noqa: E402
from balatro_rl.az.world import World  # noqa: E402
from balatro_rl.bridge import game_from_state, parse_joker  # noqa: E402
from balatro_rl.env import Counters, legal_mask, A_VOUCHER  # noqa: E402
from balatro_rl.sim.cards import Card  # noqa: E402
from balatro_rl.sim.game import Game, Consumable, ShopItem  # noqa: E402
from balatro_rl.sim.hands import evaluate, HC, PAIR, TWO_PAIR, FLUSH, QUADS, FIVE_KIND, FULL_HOUSE, FLUSH_FIVE  # noqa: E402
from balatro_rl.sim.jokers import JOKERS, Joker, EDITION_COST, NO_PERISHABLE  # noqa: E402
from balatro_rl.sim.scoring import Plan, score_hand  # noqa: E402
from mock_balatrobot import Mock, joker_json  # noqa: E402

S, H, C, D = 0, 1, 2, 3
PAIR_AA = 10 + 22            # pair of aces: base 10 chips + 11 + 11
FAR = 10 ** 9                # a target no hand reaches, so a round goes on


def game(*keys, stake="WHITE", seed=1, deck="RED"):
    g = Game(seed=seed, deck_type=deck, stake=stake)
    for k in keys:
        g.add_joker(Joker(k, base_cost=JOKERS[k].cost))
    return g


def boss_round(g, boss, target=FAR):
    g.blind_idx, g.boss = 2, boss
    g.select_blind()
    if target:
        g.target = target
    return g


def aces():
    return [Card(14, S), Card(14, H)]


def sc(g, played, held=()):
    return score_hand(g, list(played), list(held))[0]


def commit(g, played, held=()):
    return score_hand(g, list(played), list(held), rng=g.rng, commit=True)[0]


def fix_rng(g, u):
    """Every chance roll of the game comes out as u (choices and shuffles stay random)."""
    g.rng.random = lambda: u


# ------------------------------------------------------------------ scoring
def test_card_effects_come_before_the_per_card_jokers():
    g = game("scholar")
    # chips 5 + 11, Glass x2, then Scholar's +20 chips and +4 mult
    assert sc(g, [Card(14, S, enh="GLASS")]) == (5 + 11 + 20) * (1 * 2 + 4) == 216
    # the card's Polychrome also comes before Scholar
    assert sc(g, [Card(14, S, edition="POLYCHROME")]) == (5 + 11 + 20) * (1 * 1.5 + 4)
    # a retrigger repeats the whole sequence
    assert sc(g, [Card(14, S, enh="GLASS", seal="RED")]) == (5 + 2 * 31) * ((1 * 2 + 4) * 2 + 4)


def test_observatory_comes_after_the_jokers():
    g = game("joker")
    g.vouchers.add("observatory")
    g.consumables = [Consumable("planet", "mercury")]
    assert sc(g, aces()) == PAIR_AA * (2 + 4) * 1.5


def test_baseball_card_multiplies_each_uncommon_joker_in_turn():
    # Hack is Uncommon: x1.5 where Hack sits, so the +4 of a Joker after it is not multiplied
    assert sc(game("hack", "joker", "baseball"), aces()) == PAIR_AA * (2 * 1.5 + 4)
    assert sc(game("joker", "hack", "baseball"), aces()) == PAIR_AA * (2 + 4) * 1.5
    # once per Baseball Card, and once more for a Blueprint copying one
    assert sc(game("hack", "baseball", "baseball"), aces()) == PAIR_AA * 2 * 2.25
    assert sc(game("hack", "blueprint", "baseball"), aces()) == PAIR_AA * 2 * 2.25
    # nothing for jokers of other rarities (Baseball Card itself is Rare)
    assert sc(game("joker", "baseball"), aces()) == PAIR_AA * (2 + 4)


def test_raised_fist_is_a_held_card_effect():
    g = game("raised_fist")
    # the lowest held card gives 2x its chips as mult; its Red seal repeats that
    assert sc(g, aces(), [Card(13, S), Card(5, H), Card(2, D, seal="RED")]) == PAIR_AA * (2 + 4 + 4)
    # ties go to the rightmost card
    assert sc(g, aces(), [Card(3, S, seal="RED"), Card(3, H)]) == PAIR_AA * (2 + 6)
    assert sc(g, aces(), [Card(3, S), Card(3, H, seal="RED")]) == PAIR_AA * (2 + 12)
    assert sc(g, aces(), [Card(14, C)]) == PAIR_AA * (2 + 22)
    low = Card(2, D)
    low.debuffed = True
    assert sc(g, aces(), [Card(9, S), low]) == PAIR_AA * 2          # a debuffed lowest card gives nothing
    assert sc(game("raised_fist", "mime"), aces(), [Card(2, D)]) == PAIR_AA * (2 + 4 + 4)


def test_mime_and_red_seals_repeat_held_cards_that_did_something():
    g = game("baron", "mime")
    assert sc(g, aces(), [Card(13, S), Card(7, D, seal="RED")]) == PAIR_AA * 2 * 1.5 ** 2
    assert sc(g, aces(), [Card(13, S, enh="STEEL", seal="RED")]) == PAIR_AA * 2 * 1.5 ** 6
    assert sc(g, aces(), [Card(7, D, seal="RED"), Card(8, D)]) == PAIR_AA * 2


# ------------------------------------------------------------------ hand detection
def test_four_and_five_of_a_kind_do_not_contain_two_pair():
    quads = [Card(9, S), Card(9, H), Card(9, C), Card(9, D), Card(4, S)]
    five = [Card(9, S), Card(9, H), Card(9, C), Card(9, D), Card(9, S)]
    full = [Card(3, S), Card(3, H), Card(3, C), Card(9, D), Card(9, S)]
    assert evaluate(quads).hand == QUADS and TWO_PAIR not in evaluate(quads).contains
    assert evaluate(five).hand == FIVE_KIND and TWO_PAIR not in evaluate(five).contains
    assert evaluate(full).hand == FULL_HOUSE and TWO_PAIR in evaluate(full).contains
    g = game("mad")                                                 # +10 mult if the hand contains Two Pair
    assert sc(g, quads) == (60 + 36) * 7
    assert sc(g, full) == (40 + 27) * (4 + 10)
    assert g.predict_many([(0, 1, 2, 3, 4)], Plan(g), quads)[0][0] == (60 + 36) * 7   # compiled scorer too


# ------------------------------------------------------------------ playing a hand
def test_a_blocked_hand_scores_nothing_and_triggers_nothing():
    g = boss_round(game("green_joker", "ice_cream", "dna"), "psychic")
    card = g.hand[0]
    card.enh, card.seal = "GLASS", "GOLD"
    money, deck, hands = g.money, len(g.full_deck), g.hands_left
    ch = enumerate_candidates(World(g), random.Random(0))
    for c in ch.all_plays:                                           # the agent's own analysis agrees
        green = [d for d in c.jdiff if d[0] == 0]
        if len(c.action.cards) < 5:
            assert c.score == 0 and not green and not any(k == "money" for k, _ in c.effects)
        else:
            assert green
    g.play([0])                                                      # 1 card: The Psychic forbids it
    assert g.jokers[0].state["val"] == 0                             # Green Joker unchanged
    assert g.chips == 0 and g.money == money
    assert len(g.full_deck) == deck and card in g.full_deck          # no DNA copy, no Glass break
    assert g.hands_left == hands - 1 and g.hand_played[HC] == 1      # but it was a hand played
    assert g.jokers[1].state["val"] == 95                            # after-hand effects still run


def _eight(*first):
    rest = [Card(5, D), Card(4, C), Card(3, D), Card(2, C), Card(7, S), Card(9, H)]
    return list(first) + rest[:8 - len(first)]


def test_the_eye_and_the_mouth_count_blocked_hands():
    g = boss_round(game(), "eye")
    g.hand = _eight(Card(14, S), Card(14, H))
    g.play([0, 1])
    first = g.chips
    assert first == PAIR_AA * 2
    g.hand = _eight(Card(13, S), Card(13, H))
    g.play([0, 1])                                                   # a second Pair: blocked
    assert g.chips == first and g.hand_played_round[PAIR] == 2

    g = boss_round(game(), "mouth")
    g.hand = _eight(Card(14, S), Card(14, H))
    g.play([0, 1])
    g.hand = _eight(Card(13, S), Card(12, H))
    g.play([0])                                                      # High Card: blocked, but counted
    assert g.chips == first and g.mouth_hand == PAIR and g.hand_played_round[HC] == 1
    g.hand = _eight(Card(13, S), Card(13, H))
    g.play([0, 1])
    assert g.chips == first + (10 + 20) * 2


def test_the_hook_discards_before_scoring():
    g = boss_round(game("green_joker"), "hook")
    g.jokers[0].state["val"] = 5
    discards = g.discards_left
    g.hand = aces() + [Card(13, S, enh="STEEL"), Card(13, H, enh="STEEL", seal="PURPLE")]
    g.play([0, 1])
    # both Steel kings were discarded first: no x1.5; Green Joker lost 1 for the discard, gained 1 for the hand
    assert g.chips == PAIR_AA * (2 + 5)
    assert g.jokers[0].state["val"] == 5
    assert [c.kind for c in g.consumables] == ["tarot"]             # the Purple seal paid out
    assert g.discards_left == discards                               # it isn't one of your discards
    assert len(g.hand) == g.effective_hand_size()                    # drawn back up once

    g = boss_round(game("burnt"), "hook")                            # Burnt Joker ignores The Hook's discard
    g.hand = _eight(Card(14, S), Card(14, H))
    g.play([0, 1])
    assert g.hand_levels == [1] * len(g.hand_levels)


def test_predictions_average_over_the_hook_discards():
    g = boss_round(game("baron"), "hook", target=90)
    g.hand = aces() + [Card(13, S), Card(13, H), Card(5, D)]
    # held K K 5, two of them discarded first: 5 left (64), or a King left twice (64 x 1.5)
    assert g.predict([0, 1])[0] == (64 + 96 + 96) // 3
    assert g.predict_many([(0, 1)], Plan(g))[0][0] == (64 + 96 + 96) // 3        # the compiled scorer too
    g.chips = 10                                                     # 80 needed: 85 expected, 64 at worst
    ch = enumerate_candidates(World(g), random.Random(0))
    pair = next(c for c in ch.all_plays if sorted(c.action.cards) == [0, 1])
    assert pair.score == 85 and pair.clears and pair.score_min == 64 and not pair.certain
    g.chips = 30                                                     # 60 needed: certain
    ch = enumerate_candidates(World(g), random.Random(0))
    assert next(c for c in ch.all_plays if sorted(c.action.cards) == [0, 1]).certain
    # without a held-card effect the discard changes nothing
    g = boss_round(game(), "hook")
    g.hand = aces() + [Card(13, S), Card(13, H), Card(5, D)]
    assert g.predict([0, 1])[0] == 64


def test_the_arm_lowers_the_level_before_anything_else():
    g = boss_round(game(), "arm")
    g.hand_levels[PAIR] = 3
    g.hand = _eight(Card(14, S), Card(14, H))
    level2 = (10 + 15 + 22) * 3
    assert g.predict([0, 1])[0] == level2
    g.play([0, 1])
    assert g.chips == level2 and g.hand_levels[PAIR] == 2
    g.hand = _eight(Card(14, S), Card(13, H))
    g.play([0])                                                      # level 1 stays level 1
    assert g.hand_levels[HC] == 1

    g = boss_round(game("space"), "arm")                             # lowered first, then Space Joker's level
    g.hand_levels[PAIR] = 2
    g.hand = _eight(Card(14, S), Card(14, H))
    fix_rng(g, 0.0)
    g.play([0, 1])
    assert g.chips == level2 and g.hand_levels[PAIR] == 2


def test_the_ox_hand_is_fixed_when_the_round_starts():
    assert boss_round(game(), "ox").ox_hand == FLUSH_FIVE            # nothing played: the highest hand
    g = game()
    g.hand_played[HC] = g.hand_played[PAIR] = 3                     # a tie goes to the higher hand
    boss_round(g, "ox")
    assert g.ox_hand == PAIR
    g.money = 10
    for _ in range(2):                                               # High Card overtakes Pair: no change
        g.hand = _eight(Card(14, S), Card(13, H))
        g.play([0])
        assert g.money == 10 and g.ox_hand == PAIR
    g.hand = _eight(Card(14, S), Card(14, H))
    g.play([0, 1])
    assert g.money == 0


def test_matador_triggers():
    g = boss_round(game("matador"), "arm")
    money = g.money
    g.hand = _eight(Card(14, S), Card(13, H))
    g.play([0])                                                      # level 1: The Arm does nothing
    assert g.money == money
    g.hand_levels[HC] = 2
    g.hand = _eight(Card(14, S), Card(13, H))
    g.play([0])
    assert g.money == money + 8

    g = boss_round(game("matador"), "club")
    money = g.money
    kicker = Card(13, C)
    kicker.debuffed = True
    g.hand = _eight(Card(14, S), Card(14, H), kicker)
    g.play([0, 1, 2])                                                # the debuffed card doesn't score
    assert g.money == money
    ace = Card(14, C)
    ace.debuffed = True
    g.hand = _eight(ace, Card(14, H))
    g.play([0, 1])
    assert g.money == money + 8


# ------------------------------------------------------------------ boss rules
def test_suit_bosses_debuff_wild_cards_and_follow_smeared():
    g = game()
    wild = next(c for c in g.full_deck if c.suit == H)
    wild.enh = "WILD"
    boss_round(g, "club")
    assert wild.debuffed
    assert all(c.debuffed == (c.suit == C) for c in g.full_deck if c is not wild)
    g = boss_round(game("smeared"), "club")
    assert all(c.debuffed == (c.suit in (S, C)) for c in g.full_deck)


def test_selling_luchador_undoes_the_boss():
    g = boss_round(game("luchador"), "needle")
    assert g.hands_left == 1
    g.sell_joker(0)
    assert g.hands_left == g.round_hands() > 1

    g = boss_round(game("luchador"), "water")
    assert g.discards_left == 0
    g.sell_joker(0)
    assert g.discards_left == g.round_discards() > 0

    for boss, times in (("wall", 4), ("violet_vessel", 6)):
        g = boss_round(game("luchador"), boss, target=None)
        base = g.blind_target(0)
        assert g.target == base * times
        g.sell_joker(0)
        assert g.target == base * 2

    g = boss_round(game("luchador"), "manacle")
    assert len(g.hand) == 7
    g.sell_joker(0)
    assert len(g.hand) == g.effective_hand_size() == 8

    g = boss_round(game("luchador"), "house")
    assert all(c.hidden for c in g.hand)
    g.sell_joker(0)
    assert not any(c.hidden for c in g.hand)

    g = boss_round(game("luchador"), "cerulean_bell")
    assert g.forced_pos() >= 0
    g.sell_joker(0)
    assert g.forced_pos() == -1 and g.forced_uid == -1


def test_chicot_disables_the_boss_after_it_is_set_up():
    for boss in ("wall", "violet_vessel"):
        g = boss_round(game("chicot"), boss, target=None)
        assert g.target == g.blind_target(0) * 2 and g.boss_active() == ""
    g = boss_round(game("chicot"), "needle")
    assert g.hands_left == g.round_hands()
    g = boss_round(game("chicot"), "water")
    assert g.discards_left == g.round_discards()
    g = boss_round(game("chicot"), "manacle")
    assert len(g.hand) == 8
    g = boss_round(game("chicot"), "house")
    assert not any(c.hidden for c in g.hand)
    g = boss_round(game("chicot"), "club")
    assert not any(c.debuffed for c in g.full_deck)


# ------------------------------------------------------------------ economy
def _payout(g, money):
    """Win the current (small, White Stake: $3) blind holding `money`, with no hands left; the gain."""
    g.select_blind()
    g.money, g.hands_left, g.hand = money, 0, []
    g.win_round()
    return g.money - money


def test_to_the_moon_adds_interest_inside_the_cap():
    assert _payout(game(), 30) == 3 + 5
    assert _payout(game("to_the_moon"), 30) == 3 + 2 * 5
    assert _payout(game("to_the_moon"), 10) == 3 + 2 * 2
    g = game("to_the_moon")
    g.vouchers.add("seed_money")
    assert _payout(g, 100) == 3 + 2 * 10
    g = game("to_the_moon", deck="GREEN")                            # the Green Deck earns no interest
    g.select_blind()
    g.money, g.hands_left, g.discards_left, g.hand = 30, 0, 0, []
    g.win_round()
    assert g.money == 30 + 3


def test_sell_value_follows_the_price_paid():
    g = game()
    g.state, g.money = "SHOP", 20
    g.vouchers.add("liquidation")
    g.shop = [ShopItem("joker", "j_duo", g.price(8), joker=Joker("duo", base_cost=8))]
    assert g.shop[0].cost == 4
    g.buy_card(0)
    assert g.money == 16 and g.jokers[0].sell_value() == 2           # not 8 // 2
    assert Joker("duo", base_cost=8).sell_value() == 4
    assert Joker("duo", base_cost=8, cost=1).sell_value() == 1       # a Rental joker costs $1
    g = Game(seed=3, stake="GOLD")
    rental = [j for j in (g.random_joker() for _ in range(300)) if j.rental]
    assert rental and all(j.cost == 1 and j.sell_value() == 1 for j in rental)
    g.vouchers.add("clearance_sale")
    j = next(j for j in (g.random_joker() for _ in range(300)) if not j.rental)
    assert j.cost == g.price(j.base_cost) < j.base_cost
    # the bridge takes the game's own sell value
    owned = Joker("duo", base_cost=8, cost=4, sell_bonus=3)
    assert parse_joker(joker_json(owned)).sell_value() == owned.sell_value() == 5


def test_mr_bones_pays_no_blind_reward():
    g = game("mr_bones")
    g.select_blind()
    g.hands_left, g.money = 1, 0
    g.chips = int(g.target * 0.3)
    g.hand = [Card(2, S), Card(3, H), Card(9, C)]
    g.play([0])
    assert g.state == "SHOP" and not g.jokers and g.money == 0


def test_end_of_round_gold_cards_and_blue_seals_are_retriggered():
    def end(*keys):
        g = game(*keys)
        g.select_blind()
        g.money, g.hands_left, g.last_hand = 0, 0, PAIR
        g.hand = [Card(5, S, enh="GOLD", seal="RED"), Card(6, S, seal="BLUE")]
        g.win_round()
        return g
    g = end()
    assert g.money == 3 + 3 * 2 and [c.name for c in g.consumables] == ["mercury"]
    g = end("mime")
    assert g.money == 3 + 3 * 3 and [c.name for c in g.consumables] == ["mercury", "mercury"]
    g = end("blueprint", "mime")
    assert g.money == 3 + 3 * 4 and len(g.consumables) == 2          # no room for a third planet


# ------------------------------------------------------------------ shop, packs and tags
def _share(g, kind, n=20000):
    return sum(g.shop_card().kind == kind for _ in range(n)) / n


def test_merchant_and_tycoon_weights():
    g = game()
    g.vouchers.add("tarot_merchant")
    assert abs(_share(g, "tarot") - 9.6 / 33.6) < 0.015             # 28.6% (it was 25%)
    g.vouchers.add("tarot_tycoon")
    assert abs(_share(g, "tarot") - 32 / 56) < 0.02                 # 57.1% (it was 40%)
    g = game()
    g.vouchers.update({"planet_merchant", "planet_tycoon"})
    assert abs(_share(g, "planet") - 32 / 56) < 0.02


def test_uncommon_and_rare_tags_take_a_shop_slot():
    g = game()
    g.pending_tags = ["uncommon"]
    g.open_shop()
    assert len(g.shop) == g.shop_slots() == 2 and not g.pending_tags
    it = g.shop[0]
    assert it.kind == "joker" and it.cost == 0 and it.joker.d.rarity == 2
    g.buy_card(0)
    assert g.jokers[0].sell_value() == 1                             # it was free

    g = game()
    g.pending_tags = ["rare", "rare"]
    g.open_shop()
    assert [it.cost for it in g.shop] == [0, 0] and g.shop[0].key != g.shop[1].key

    g = game()                                                       # every Rare owned: the tag makes nothing
    g.joker_slots = 99
    for k, d in JOKERS.items():
        if d.rarity == 3:
            g.add_joker(Joker(k, base_cost=d.cost))
    g.pending_tags = ["rare"]
    g.open_shop()
    assert len(g.shop) == 2 and all(it.cost > 0 for it in g.shop) and not g.pending_tags


def test_edition_tags_wait_for_a_joker():
    g = game()
    g.money = 100
    g.pending_tags = ["negative"]
    g.shop_card = lambda exclude=(): ShopItem("tarot", "c_fool", 3)
    g.open_shop()
    assert g.pending_tags == ["negative"]                            # no joker in the shop: it waits
    g.leave_shop()
    g.open_shop()
    assert g.pending_tags == ["negative"]
    holo = Joker("jolly", base_cost=6, edition="HOLO")
    made = iter([ShopItem("joker", "j_jolly", 6, joker=holo),
                 ShopItem("joker", "j_joker", 2, joker=Joker("joker", base_cost=2))])
    g.shop_card = lambda exclude=(): next(made)
    g.reroll()
    assert not g.pending_tags and holo.edition == "HOLO"             # a joker that has an edition is skipped
    it = g.shop[1]
    assert it.joker.edition == "NEGATIVE" and it.cost == 0


def test_voucher_tag_adds_a_second_voucher():
    g = game()
    g.money = 100
    g.pending_tags = ["voucher"]
    g.open_shop()
    first, second = g.shop_vouchers
    assert first is g.ante_voucher and first.key != second.key
    m = legal_mask(g, Counters())
    assert m[A_VOUCHER] and m[A_VOUCHER + 1]
    g.buy_voucher(1)                                                 # the tag's voucher
    assert second.key[2:] in g.vouchers and g.shop_vouchers == [first]
    g.leave_shop()
    g.open_shop()
    assert g.shop_vouchers == [first]                                # the ante's voucher stays until bought
    g.buy_voucher(0)
    g.leave_shop()
    g.open_shop()
    assert g.shop_vouchers == [] and not legal_mask(g, Counters())[A_VOUCHER]


def test_illusion_playing_cards():
    g = game()
    g.vouchers.add("magic_trick")
    plain = [g.shop_playing_card() for _ in range(200)]
    assert all(c.enh == "" and c.edition == "" and c.seal == "" for c in plain)
    g.vouchers.add("illusion")
    cards = [g.shop_playing_card() for _ in range(6000)]
    assert all(c.seal == "" and c.edition != "NEGATIVE" for c in cards)
    assert abs(sum(c.enh != "" for c in cards) / 6000 - 0.4) < 0.03
    eds = [c.edition for c in cards if c.edition]
    assert abs(len(eds) / 6000 - 0.2) < 0.025
    assert abs(eds.count("FOIL") / len(eds) - 0.5) < 0.06 and abs(eds.count("POLYCHROME") / len(eds) - 0.15) < 0.05
    for _ in range(3000):                                            # $1 plus the edition, nothing for the rest
        it = g.shop_card()
        if it.kind == "card":
            assert it.cost == g.price(1 + EDITION_COST[it.card.edition])


def test_standard_packs_use_the_edition_rate():
    def share(g, n=8000):
        return sum(g.random_playing_card(enhanced=True).edition != "" for _ in range(n)) / n
    g = game()
    assert abs(share(g) - 0.08) < 0.015                              # 2 x 4%
    g.vouchers.update({"hone", "glow_up"})
    assert abs(share(g) - 0.32) < 0.025                              # Glow Up: 4 times as often
    assert all(g.random_playing_card(enhanced=True).edition != "NEGATIVE" for _ in range(2000))


def test_no_duplicates_in_a_shop_or_pack():
    for seed in range(20):
        g = game("joker", "duo", seed=seed)
        g.vouchers.update({"overstock_norm", "overstock_plus", "tarot_merchant", "planet_merchant"})
        g.money = 10 ** 6
        owned = {f"j_{j.key}" for j in g.jokers}
        g.open_shop()
        for _ in range(8):
            keys = [it.key for it in g.shop if it.kind != "card"]
            assert len(g.shop) == 4 and len(set(keys)) == len(keys) and not owned & set(keys)
            g.reroll()
        g.open_pack("buffoon", "mega")
        keys = [j.key for j in g.pack_cards]
        assert len(set(keys)) == len(keys) == 4 and not {"joker", "duo"} & set(keys)


def test_the_soul_and_black_hole_appear_once_per_run():
    def names(g, kind):
        g.open_pack(kind, "normal")
        got = [c.name for c in g.pack_cards]
        g.close_pack()
        return got
    g = game()
    fix_rng(g, 0.0)                                                  # every 0.3% roll succeeds
    assert names(g, "arcana").count("soul") == 1
    assert "soul" not in names(g, "arcana") and "soul" not in names(g, "spectral")
    assert names(g, "celestial").count("black_hole") == 1
    assert "black_hole" not in names(g, "celestial")
    g = game("ring_master")                                          # Showman lifts the limit
    fix_rng(g, 0.0)
    assert names(g, "arcana") == ["soul"] * 3 and names(g, "arcana") == ["soul"] * 3


def test_a_queued_double_tag_copies_the_next_tag():
    g = game()
    g.money = 10
    g.pending_tags = ["double", "juggle"]                            # another tag queued behind the Double
    g.add_tag("economy")
    assert g.money == 40 and g.pending_tags == ["juggle"]

    g = game()                                                       # it waits through shops for that tag
    g.add_tag("double")
    g.open_shop()
    g.leave_shop()
    assert g.pending_tags == ["double"]
    g.money = 10
    g.add_tag("economy")
    assert g.money == 40 and not g.pending_tags

    g = game(deck="ANAGLYPH")                                        # the Anaglyph Deck's tag after a boss
    boss_round(g, "club", target=1)
    g.play([0])
    assert g.state == "SHOP" and g.pending_tags == ["double"]

    g = game()                                                       # a third free joker waits for a slot
    g.pending_tags = ["uncommon", "uncommon", "uncommon"]
    g.open_shop()
    assert [it.cost for it in g.shop] == [0, 0] and g.pending_tags == ["uncommon"]


# ------------------------------------------------------------------ consumables
def test_wheel_of_fortune_uses_the_game_odds():
    for keys, hit in ((("joker",), False), (("joker", "oops"), True)):   # Oops! All 6s: 1 in 4 -> 1 in 2
        g = game(*keys)
        fix_rng(g, 0.4)
        g.consumables = [Consumable("tarot", "wheel_of_fortune")]
        g.use_consumable(0)
        assert any(j.edition for j in g.jokers) == hit


def test_consumable_use_conditions():
    g = game("joker")
    g.jokers[0].edition = "FOIL"
    two = [Card(5, S), Card(6, S)]
    needs_plain = [Consumable("spectral", "ectoplasm"), Consumable("spectral", "hex"),
                   Consumable("tarot", "wheel_of_fortune")]
    assert not any(g.consumable_usable(c, two) for c in needs_plain)     # no joker without an edition
    g.add_joker(Joker("duo", base_cost=8))
    assert all(g.consumable_usable(c, two) for c in needs_plain)

    for name in ("familiar", "grim", "incantation", "immolate", "sigil", "ouija"):
        c = Consumable("spectral", name)
        assert not g.consumable_usable(c, two[:1]) and g.consumable_usable(c, two)

    aura = Consumable("spectral", "aura")
    foil, plain = Card(5, S, edition="FOIL"), Card(6, S)
    assert not g.consumable_usable(aura, [foil]) and g.consumable_usable(aura, [foil, plain])
    assert target_sets(g, aura, [foil, plain]) == [(1,)]
    assert g.auto_targets("aura", [foil, plain]) == [plain]

    g.consumables = [Consumable("tarot", "emperor"), Consumable("tarot", "fool")]     # slots full
    g.last_consumable = Consumable("tarot", "empress")
    for name in ("emperor", "high_priestess", "fool"):
        c = Consumable("tarot", name)
        assert g.consumable_usable(c, [])                            # from its slot: it frees that slot
        assert not g.consumable_usable(c, [], from_slot=False)       # from a pack: no room
    g.consumables.pop()
    assert g.consumable_usable(Consumable("tarot", "emperor"), [], from_slot=False)


def test_wraith_sets_money_to_zero():
    for money in (-10, 25):
        g = game("credit_card")
        g.money = money
        g.apply_consumable(Consumable("spectral", "wraith"), [])
        assert g.money == 0 and g.jokers[-1].d.rarity == 3


# ------------------------------------------------------------------ jokers
def test_loyalty_card_fires_every_sixth_hand():
    g = game("loyalty_card")
    scores = [commit(g, aces()) for _ in range(18)]
    assert [i + 1 for i, s in enumerate(scores) if s == PAIR_AA * 2 * 4] == [6, 12, 18]
    assert all(s in (PAIR_AA * 2, PAIR_AA * 2 * 4) for s in scores)


def test_blueprint_and_brainstorm_copy_every_effect():
    g = boss_round(game("blueprint", "dna"), "club")
    deck = len(g.full_deck)
    g.play([0])
    assert len(g.full_deck) == deck + 2                              # effects before scoring (DNA)

    g = game("dna", "joker", "brainstorm")
    g.select_blind()
    g.target = FAR
    deck = len(g.full_deck)
    g.play([0])
    assert len(g.full_deck) == deck + 2

    g = game("blueprint", "faceless")
    g.select_blind()
    g.hand = _eight(Card(13, S), Card(13, H), Card(12, D))
    money = g.money
    g.discard([0, 1, 2])
    assert g.money == money + 10                                     # discard effects

    g = game("blueprint", "burnt")
    g.select_blind()
    g.hand = _eight(Card(13, S), Card(13, H))
    g.discard([0, 1])
    assert g.hand_levels[PAIR] == 3

    g = game("blueprint", "burglar")
    g.select_blind()
    assert g.hands_left == g.round_hands() + 6 and g.discards_left == 0   # blind-select effects

    g = game("blueprint", "certificate")
    deck = len(g.full_deck)
    g.select_blind()
    assert len(g.full_deck) == deck + 2

    g = game("blueprint", "hallucination")
    fix_rng(g, 0.0)
    g.open_pack("standard", "normal")
    assert [c.kind for c in g.consumables] == ["tarot", "tarot"]     # pack opening

    assert Plan(game("blueprint", "mime")).mime == 2                 # one more held-card retrigger

    g = game("blueprint", "green_joker")                             # a scaling joker doesn't grow twice
    g.select_blind()
    g.target = FAR
    g.play([0])
    assert g.jokers[1].state["val"] == 1


def test_sticker_restrictions():
    assert len(NO_PERISHABLE) == 18 and not any(JOKERS[k].perish_ok for k in NO_PERISHABLE)
    assert not JOKERS["luchador"].eternal_ok
    g = Game(seed=5, stake="GOLD")
    seen = set()
    for _ in range(6000):
        j = g.random_joker()
        seen.add(j.key)
        assert j.perishable is None or j.d.perish_ok
        assert not j.eternal or j.d.eternal_ok
    assert "luchador" in seen and set(NO_PERISHABLE) <= seen


def test_obelisk_does_not_grow_on_the_first_hand():
    g = game("obelisk")
    commit(g, aces())                                                # first hand of the run
    assert g.jokers[0].state["val"] == 1.0
    g.hand_played[PAIR] = 1
    commit(g, [Card(14, S)])                                         # a hand played less often: +0.2
    assert g.jokers[0].state["val"] == 1.2
    g.hand_played[HC] = 1
    commit(g, [Card(14, S)])                                         # as often as the most played: reset
    assert g.jokers[0].state["val"] == 1.0


def _straight(*suits_enh):
    return [Card(5 + i, s, enh=e) for i, (s, e) in enumerate(suits_enh)]


def test_flower_pot_fills_suits_in_the_game_order():
    base = 30 + 5 + 6 + 7 + 8 + 9
    g = game("flower_pot")
    assert sc(g, _straight((H, ""), (D, ""), (S, ""), (C, ""), (C, ""))) == base * 4 * 3
    assert sc(g, _straight((H, ""), (D, ""), (S, ""), (S, ""), (S, ""))) == base * 4
    assert sc(g, _straight((H, ""), (D, ""), (S, ""), (S, "WILD"), (S, ""))) == base * 4 * 3
    g = game("flower_pot", "smeared")                                # two red and two black cards are enough
    assert sc(g, _straight((H, ""), (H, ""), (S, ""), (S, ""), (H, ""))) == base * 4 * 3
    assert sc(g, _straight((H, ""), (S, ""), (S, ""), (S, ""), (C, ""))) == base * 4


# ------------------------------------------------------------------ copies, DNA, debuffed cards
def both(g, played, held=()):
    """Score of a play by the Python scorer, checked against the compiled one."""
    played, held = list(played), list(held)
    s = sc(g, played, held)
    assert g.predict_many([tuple(range(len(played)))], Plan(g), played + held)[0][0] == s
    return s


def deb(c):
    c.debuffed = True
    return c


def test_copies_of_photograph_double_every_trigger():
    # one King, High Card: 5 + 10 chips; Photograph x2 and Brainstorm's copy of it x2
    assert both(game("photograph", "brainstorm"), [Card(13, S)]) == (5 + 10) * 2 * 2 == 60
    assert both(game("blueprint", "photograph"), [Card(13, S)]) == 60
    # Blueprint copies Hanging Chad (2 + 2 retriggers: 5 triggers), Brainstorm copies Photograph:
    # chips 5 + 5 x 10, and x2 x2 on each of the 5 triggers
    g = game("photograph", "blueprint", "hanging_chad", "brainstorm")
    assert both(g, [Card(13, S)]) == (5 + 5 * 10) * 4 ** 5 == 56320
    assert both(g, [Card(13, S, enh="GLASS")]) == (5 + 5 * 10) * 8 ** 5 == 1802240
    g.jokers[0].debuffed = True                                      # no Photograph: its copy goes too
    assert both(g, [Card(13, S, enh="GLASS")]) == (5 + 5 * 10) * 2 ** 5 == 1760


def test_copier_chains_resolve_to_the_final_joker():
    king = [Card(13, S)]                                             # 15 chips, mult 1
    # Brainstorm copies Joker; Blueprint copies Brainstorm, so Joker too: +4 three times
    assert both(game("joker", "blueprint", "brainstorm"), king) == 15 * (1 + 4 + 4 + 4) == 195
    # Blueprint copies Joker; Brainstorm copies Blueprint (the leftmost), so Joker too
    assert both(game("blueprint", "joker", "brainstorm"), king) == 195
    # Blueprint -> Brainstorm -> the leftmost joker, that same Blueprint: a cycle copies nothing
    assert both(game("blueprint", "brainstorm", "joker"), king) == 15 * (1 + 4) == 75
    assert both(game("blueprint", "brainstorm"), king) == 15
    g = game("joker", "blueprint", "brainstorm")
    g.jokers[2].debuffed = True                                      # a debuffed link breaks the chain
    assert both(g, king) == 15 * (1 + 4) == 75
    g = game("mime", "egg", "blueprint", "brainstorm")               # passive effects follow the chain too
    assert Plan(g).mime == 3 and g.count_with_copies("mime") == 3
    # the joker Crimson Heart disabled gives nothing to copy
    g = boss_round(game("joker", "brainstorm"), "crimson_heart")
    g.crimson_disabled = g.jokers[0].uid
    assert both(g, king) == 15
    g.crimson_disabled = g.jokers[1].uid
    assert both(g, king) == 15 * (1 + 4)


def test_a_copy_never_grows_the_copied_joker():
    # Wee Joker gains +8 chips once for the scored 2; it and Blueprint's copy then each give those 8 chips
    g = game("blueprint", "wee")
    assert both(g, [Card(2, S)]) == 5 + 2 + 8 + 8 == 23
    assert commit(g, [Card(2, S)]) == 23 and g.jokers[1].state["val"] == 8
    # Lucky Cat: one Lucky hit (the +20 mult; the $20 roll misses at 0.1) is +0.25 once, and the copy to its
    # left already uses the grown value: 15 chips x (1 + 20) x 1.25 x 1.25 = 492.19
    g = game("blueprint", "lucky_cat")
    fix_rng(g, 0.1)
    assert commit(g, [Card(13, S, enh="LUCKY")]) == 492
    assert g.jokers[1].state["val"] == 1.25


def test_dna_copy_is_held_while_its_hand_scores():
    g = game("dna", "hologram")
    g.select_blind()
    g.target = FAR
    g.hand = _eight(Card(13, H, enh="STEEL"))
    deck = len(g.full_deck)
    # first hand, one card: the Steel King's copy is held (x1.5) and Hologram has grown to x1.25
    assert g.predict([0])[0] == g.predict_many([(0,)], Plan(g))[0][0] == 28      # 15 x 1.5 x 1.25 = 28.1
    cand = next(c for c in enumerate_candidates(World(g), random.Random(0)).all_plays if c.action.cards == (0,))
    assert cand.score == 28 and (1, "val", 1.0, 1.25) in cand.jdiff and ("dna", 1) in cand.effects
    g.play([0])
    assert g.chips == 28 and len(g.full_deck) == deck + 1
    assert g.jokers[1].state["val"] == 1.25                          # grown once, not again when it lands
    assert sum(1 for c in g.hand if c.enh == "STEEL" and c.rank == 13) == 1 and len(g.hand) == 8
    g.play([len(g.hand) - 1])                                        # not the first hand any more: no copy
    assert len(g.full_deck) == deck + 1 and g.jokers[1].state["val"] == 1.25


def test_debuffed_cards_are_not_faces_and_have_no_suit_for_jokers():
    # Ride the Bus: a debuffed King is not a face card, so it grows (+1); the card itself gives no chips
    g = game("ride_the_bus")
    assert both(g, [deb(Card(13, S))]) == 5 * (1 + 1) == 10
    assert commit(g, [Card(13, S)]) == 15 * 1 and g.jokers[0].state["val"] == 0     # a live face resets it
    # Photograph: the first face card is the first one that is not debuffed
    assert both(game("photograph"), [deb(Card(13, S)), Card(13, H)]) == (10 + 10) * 2 * 2 == 80
    # the boss's own check still sees them (The Plant debuffs every face card)
    g = boss_round(game(), "plant")
    faces = [c for c in g.full_deck if c.rank in (11, 12, 13)]
    assert len(faces) == 12 and all(c.debuffed and not c.is_face() and c.is_face(from_boss=True) for c in faces)
    # Seeing Double: a debuffed card has no suit
    g = game("seeing_double")
    assert both(g, [Card(14, C), Card(14, H)]) == PAIR_AA * 2 * 2
    assert both(g, [Card(14, C), deb(Card(14, H))]) == (10 + 11) * 2
    assert both(g, [Card(14, H, enh="WILD")]) == 16                  # a lone Wild card is only a Club
    assert both(g, [Card(14, H, enh="WILD"), Card(14, S, enh="WILD")]) == PAIR_AA * 2 * 2
    assert both(game("seeing_double", "smeared"), [Card(14, C)]) == 16 * 2      # a Club is also a Spade
    # a debuffed Wild card is not wild in a flush or for Blackboard, but a debuffed card keeps its suit there
    hearts = [Card(2, H), Card(5, H), Card(8, H), Card(11, H)]
    assert evaluate(hearts + [Card(13, S, enh="WILD")]).hand == FLUSH
    assert evaluate(hearts + [deb(Card(13, S, enh="WILD"))]).hand == HC
    assert evaluate(hearts + [deb(Card(13, H))]).hand == FLUSH
    g = game("blackboard")
    assert both(g, aces(), [deb(Card(5, S))]) == PAIR_AA * 2 * 3
    assert both(g, aces(), [Card(5, H, enh="WILD")]) == PAIR_AA * 2 * 3
    assert both(g, aces(), [deb(Card(5, H, enh="WILD"))]) == PAIR_AA * 2
    # Flower Pot ignores debuffs: the two debuffed Clubs (no chips) still complete the four suits
    cards = _straight((H, ""), (D, ""), (S, ""), (C, ""), (C, ""))
    deb(cards[3]), deb(cards[4])
    assert both(game("flower_pot"), cards) == (30 + 5 + 6 + 7) * 4 * 3


def test_four_fingers_flush_is_the_first_suit_that_has_four():
    # four Wild cards make the Spade flush (suits are tried Spades, Hearts, Clubs, Diamonds), so the
    # fifth card, a Heart, is not part of it and doesn't score
    cards = [Card(2, D, enh="WILD"), Card(5, C, enh="WILD"), Card(8, H, enh="WILD"), Card(11, H, enh="WILD"),
             Card(13, H)]
    res = evaluate(cards, four_fingers=True)
    assert res.hand == FLUSH and res.scoring == [0, 1, 2, 3]
    assert both(game("four_fingers"), cards) == (35 + 2 + 5 + 8 + 10) * 4 == 240


def test_hiker_chips_count_from_the_next_trigger():
    # Red seal Ace: 5 + 11, Hiker +5, then the retrigger scores 11 + 5; the same in a prediction
    g = game("hiker")
    ace = Card(14, S, seal="RED")
    assert both(g, [ace]) == 5 + 11 + (11 + 5) == 32 and ace.extra_chips == 0
    assert commit(g, [ace]) == 32 and ace.extra_chips == 10
    assert both(game("blueprint", "hiker"), [Card(14, S, seal="RED")]) == 5 + 11 + (11 + 10) == 37


def test_vampire_turns_a_stone_card_back_into_its_rank():
    assert both(game(), [Card(9, S, enh="STONE")]) == 5 + 50
    # stripped before it scores: 9 chips instead of 50, and Vampire is at x1.1: 14 x 1.1 = 15.4
    assert both(game("vampire"), [Card(9, S, enh="STONE")]) == 15


def test_plasma_deck_floors_the_balanced_value():
    # 16 chips and mult 1: both become floor(17 / 2) = 8
    assert both(game(deck="PLASMA"), [Card(14, S)]) == 8 * 8


# ------------------------------------------------------------------ bridge
def test_bridge_remembers_the_first_hand_and_the_ox_hand():
    mock = Mock()
    mock.handle("start", {"deck": "RED", "stake": "WHITE", "seed": "1"})
    g = mock.g
    boss_round(g, "mouth")
    memory = {}
    assert game_from_state(mock.state(), memory=memory)[0].mouth_hand == -1
    g.hand = _eight(Card(14, S), Card(14, H))
    g.play([0, 1])
    assert game_from_state(mock.state(), memory=memory)[0].mouth_hand == PAIR
    g.hand = _eight(Card(13, S), Card(12, H))
    g.play([0])                                                      # blocked, but counted as played
    g2 = game_from_state(mock.state(), memory=memory)[0]
    assert g2.mouth_hand == g.mouth_hand == PAIR
    assert g2.violates_boss([Card(13, S)]) and not g2.violates_boss(aces())

    mock.handle("start", {"deck": "RED", "stake": "WHITE", "seed": "2"})
    g = mock.g
    g.hand_played[PAIR] = g.hand_played[HC] = 2
    boss_round(g, "ox")
    for _ in range(2):
        g.hand = _eight(Card(13, S), Card(12, H))
        g.play([0])
    assert game_from_state(mock.state())[0].ox_hand == g.ox_hand == PAIR
