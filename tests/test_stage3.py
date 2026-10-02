"""Stage 3: the shop module (graded prior, rule switches, one-step V, paired rollouts on close calls)."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from balatro_rl.az import compare
from balatro_rl.az.actions import enumerate_candidates
from balatro_rl.az.agent import Agent, AgentConfig, override_summary, random_outcome
from balatro_rl.az.features import F_CAND, N_PRICE, encode_cands
from balatro_rl.az.net import AZNet, collate
from balatro_rl.az.shop import ShopConfig, ShopPricer, _eye_chance, rule_move
from balatro_rl.az.train import new_net, play_game, train_on
from balatro_rl.az.world import Action, World
from balatro_rl.rewards.config import PotentialConfig, RewardConfig, StrengthConfig
from balatro_rl.sim.game import Consumable, Game, ShopItem
from balatro_rl.sim.hands import N_HANDS
from balatro_rl.sim.jokers import Joker

RULES = ("rule_skip", "rule_arcana", "rule_hold", "rule_scaling", "rule_pace", "rule_copier", "rule_boss")
BOUNDED = PotentialConfig(value_bound="floor_sigmoid")


def joker_item(key: str, cost: int, edition: str = "") -> ShopItem:
    g = Game(seed=1, stake="WHITE")
    j = Joker(key, base_cost=cost, cost=cost, edition=edition)
    if j.d.init:
        j.d.init(g, j)
    return ShopItem("joker", f"j_{key}", cost, joker=j)


def shop(seed=9, money=12, jokers=(), items=None, packs=None, ante=1) -> Game:
    """A White Stake game in its first shop, with the given jokers held and shop contents."""
    g = Game(seed=seed, stake="WHITE")
    g.select_blind()
    g.chips = g.target
    g.win_round()
    g.ante = ante
    g.money = money
    for k in jokers:
        g.add_joker(Joker(k, base_cost=4))
    if items is not None:
        g.shop = list(items)
    if packs is not None:
        g.shop_packs = list(packs)
    return g


def twin(g: Game, seed: int) -> Game:
    """The same game as far as a player can tell: another draw order and another future."""
    g2 = copy.deepcopy(g)
    random.Random(seed).shuffle(g2.deck)
    g2.rng = random.Random(seed + 1)
    return g2


def priced(g: Game, cfg: ShopConfig | None = None, seed: int = 1, pricer: ShopPricer | None = None):
    w = World(g)
    ch = enumerate_candidates(w, random.Random(0))
    pr = pricer or ShopPricer(cfg)
    parts = pr.prices(w, ch, random.Random(seed))
    return {str(c.action): (float(p[0]), float(p[1])) for c, p in zip(ch.cands, parts)}, pr, ch


def agent(cfg: AgentConfig, seed: int = 11, pot: PotentialConfig = BOUNDED) -> Agent:
    torch.manual_seed(0)
    return Agent(new_net(pot, cfg.price_feature).eval(), cfg, seed=seed, potential=pot)


# ------------------------------------------------------------------ defaults
def test_defaults_are_unchanged():
    cfg = AgentConfig()
    assert (cfg.shop_prior, cfg.shop_eval, cfg.close_calls, cfg.price_feature) == ("rule", "search", False, False)
    assert not any(getattr(cfg.shop, r) for r in RULES)
    assert F_CAND == 1 + N_HANDS + 7 + 3 + 4 + 12 + 3 + 2
    torch.manual_seed(0)
    plain = AZNet()
    torch.manual_seed(0)
    with_price = AZNet(price_feats=True)
    assert plain.c_price is None and "price_feats" not in plain.config
    a, b = plain.state_dict(), with_price.state_dict()
    assert set(b) - set(a) == {"c_price.weight", "c_price.bias"}
    assert all(torch.equal(a[k], b[k]) for k in a)                  # every other weight is what it was
    g = shop()
    w = World(g)
    ch = enumerate_candidates(w, random.Random(0))
    enc = encode_cands(w, ch, np.zeros(len(ch.cands)))
    assert set(enc) == {"c_kind", "c_f", "c_ref", "c_jd", "c_prior"}
    # the default agent never builds a price
    ag = Agent(plain.eval(), AgentConfig(search=False), seed=3)
    d = ag.decide(w)
    assert not ag.pricer.stats and "c_price" not in d.enc[1]
    assert AgentConfig(shop={"rule_skip": True}).shop.rule_skip          # grid files pass a dict
    with pytest.raises(ValueError):
        AgentConfig(shop_prior="nope")


# ------------------------------------------------------------------ prices
def test_price_is_strength_change_minus_money_times_m():
    items = [joker_item("cavendish", 4), joker_item("credit_card", 1)]
    g = shop(money=12, items=items)
    p, pr, ch = priced(g)
    assert p["leave"] == (0.0, 0.0)
    strong, weak = p["buy 0"], p["buy 1"]
    assert strong[0] > 0.05 > weak[0]                      # X3 mult raises the build; Credit Card barely
    assert strong[1] == pytest.approx(-4 * pr.m(g, 4))     # money part = -cost x m(ante, money)
    assert sum(strong) > 0
    # the prior is the prices scaled to logits, for every option
    w = World(g)
    logits, feats = pr.prior(w, ch, random.Random(1))
    price = np.array([sum(p[str(c.action)]) for c in ch.cands])
    assert logits.max() == 0 and np.allclose(logits, np.clip((price - price.max()) * 100.0, -30, 0))
    assert feats.shape == (len(ch.cands), N_PRICE) and np.all(feats[:, 3] == 1)
    assert str(ch.cands[int(np.argmax(logits))].action) == "buy 0"


def test_m_is_high_under_an_interest_threshold_and_falls_with_ante():
    pr = ShopPricer()
    g = shop(money=25)
    at_cap = pr.m(g, 4)                      # $25 -> $21: an interest step is lost
    g.money = 29
    above = pr.m(g, 4)                       # $29 -> $25: none is
    assert at_cap > above > 0
    g.money = 24
    by_ante = []
    for a in range(1, 9):
        g.ante = a
        by_ante.append(pr.m(g, 4))
    assert all(x > y for x, y in zip(by_ante, by_ante[1:6])) and by_ante[7] < 0.1 * by_ante[0]


def test_full_slots_price_the_best_sell_then_buy_pair():
    g = shop(money=10, jokers=["credit_card", "joker", "jolly", "sly", "banner"], items=[joker_item("cavendish", 4)])
    assert len(g.jokers) == g.joker_slots
    p, pr, ch = priced(g)
    assert "buy 0" not in p                                # no free slot
    assert sum(p["sell_joker 0"]) > 0
    g.shop = []                                            # nothing to buy: selling Credit Card gains nothing
    alone, _, _ = priced(g)
    assert sum(alone["sell_joker 0"]) < sum(p["sell_joker 0"])
    # after the sale the purchase is the best option
    g.shop = [joker_item("cavendish", 4)]
    g.sell_joker(0)
    after, _, _ = priced(g)
    assert max(after, key=lambda k: sum(after[k])) == "buy 0"


def test_planets_vouchers_and_packs_get_prices():
    g = shop(money=20, jokers=["jolly"], items=[ShopItem("planet", "c_mercury", 3), ShopItem("planet", "c_eris", 3)],
             packs=[ShopItem("pack", "p_buffoon_normal", 4, pack=("buffoon", "normal")),
                    ShopItem("pack", "p_celestial_normal", 4, pack=("celestial", "normal"))])
    g.shop_voucher = ShopItem("voucher", "v_grabber", 10)
    p, pr, _ = priced(g)
    assert p["buy 0"][0] > p["buy 1"][0]                    # Pair is played; Flush Five never is
    assert p["voucher 0"][0] > 0                            # one more hand per round shows in the clear chances
    assert p["buy_pack 0"][0] > 0 and p["buy_pack 0"][1] < 0
    # a plain playing card is not worth a pick: it must beat card_margin
    std = shop(money=10, jokers=["jolly"])
    std.open_pack("standard", "normal")
    for c in std.pack_cards:
        c.enh = c.edition = c.seal = ""
    picks, _, _ = priced(std)
    assert max(picks, key=lambda k: sum(picks[k])) == "pack_skip"
    n = pr.stats["scorings"]
    again, _, _ = priced(g, pricer=pr)
    assert again == p and pr.stats["scorings"] == n         # same build: everything cached, same prices
    # a held planet: using it beats holding it
    g.consumables = [Consumable("planet", "mercury")]
    held, _, _ = priced(g)
    assert sum(held["use 0"]) > 0


def test_reroll_uses_the_running_average_of_fresh_shops():
    cfg = ShopConfig()
    pr = ShopPricer(cfg)
    g = shop(money=30, items=[joker_item("cavendish", 4)])
    ctx = pr.context(g)
    assert pr.reroll_value(ctx) == cfg.reroll_prior
    p, _, _ = priced(g, pricer=pr)
    best = sum(p["buy 0"])
    overall = (cfg.reroll_prior * 2 + best) / 3             # every shop of the game, with the prior's weight
    assert pr.reroll_value(pr.context(g)) == pytest.approx((overall * 2 + best) / 3)     # this ante's on top
    assert p["reroll"][0] == pytest.approx(pr.reroll_value(pr.context(g)) - cfg.act_margin)
    assert p["reroll"][1] == pytest.approx(-g.reroll_cost * pr.m(g, g.reroll_cost))
    priced(g, pricer=pr)
    assert pr.fresh[g.ante][1] == 1                         # the same shop is not counted twice
    pr.reset()
    assert pr.reroll_value(pr.context(g)) == cfg.reroll_prior
    # the agent starts every game with an empty average
    ag = agent(AgentConfig(search=False, shop_prior="graded"))
    ag.decide(World(g, steps=5))
    assert ag.pricer.fresh
    ag.decide(World(Game(seed=3, stake="WHITE")))
    assert not ag.pricer.fresh


# ------------------------------------------------------------------ rule switches
def test_rule_skip_prices_tags():
    g = Game(seed=4, stake="WHITE")
    g.money = 30
    g.tags_offered = ["economy", "economy"]
    off, _, _ = priced(g)
    on, _, _ = priced(g, ShopConfig(rule_skip=True))
    assert sum(off["skip"]) <= -1.0 and off["select"] == (0.0, 0.0)
    assert sum(on["skip"]) > 0                              # +$30 beats a $3 blind and its shop
    g.money = 0
    poor, _, _ = priced(g, ShopConfig(rule_skip=True))
    assert sum(poor["skip"]) < 0                            # doubling nothing does not
    g.tags_offered = ["buffoon", "economy"]                 # a tag that opens a pack is priced by its contents
    pack, pr, _ = priced(g, ShopConfig(rule_skip=True))
    assert pack["skip"][0] > poor["skip"][0]


def test_rule_arcana_prices_packs_and_shop_tarots():
    g = shop(money=20, jokers=["jolly"], items=[ShopItem("tarot", "c_empress", 3), ShopItem("tarot", "c_hermit", 3)],
             packs=[ShopItem("pack", "p_arcana_normal", 4, pack=("arcana", "normal")),
                    ShopItem("pack", "p_spectral_normal", 4, pack=("spectral", "normal"))])
    off, pr0, _ = priced(g)
    on, pr1, _ = priced(g, ShopConfig(rule_arcana=True))
    for k in ("buy 0", "buy 1", "buy_pack 0", "buy_pack 1"):
        assert sum(off[k]) < 0                              # never bought, as today
    assert on["buy_pack 0"][0] > off["buy_pack 0"][0] and on["buy 1"][0] > 0 and on["buy 0"][0] >= off["buy 0"][0]
    assert pr1.stats["scorings"] > pr0.stats["scorings"]


def test_rule_hold_waits_for_hermit():
    g = shop(money=4)
    g.consumables = [Consumable("tarot", "hermit")]
    off, _, _ = priced(g)
    on, _, _ = priced(g, ShopConfig(rule_hold=True))
    assert sum(off["use 0"]) > 0 > sum(on["use 0"])         # $4 now, or up to $20 later
    assert sum(on["sell_cons 0"]) < 0                       # and it is not sold for $1 either
    g.money = 22
    rich, _, _ = priced(g, ShopConfig(rule_hold=True))
    assert sum(rich["use 0"]) > 0


def test_rule_scaling_moves_scaling_jokers_forward():
    g = shop(money=10, items=[joker_item("green_joker", 4), joker_item("ice_cream", 4)])
    off, _, _ = priced(g)
    on, _, _ = priced(g, ShopConfig(rule_scaling=True))
    assert on["buy 0"][0] > off["buy 0"][0]                 # Green Joker at +0 now, more a few rounds on
    assert on["buy 1"][0] < off["buy 1"][0]                 # Ice Cream melts


def test_rule_pace_spends_when_behind():
    g = shop(money=25, items=[joker_item("joker", 4)], ante=3)       # no jokers at ante 3: far behind
    off, pr0, _ = priced(g)
    on, pr1, _ = priced(g, ShopConfig(rule_pace=True))
    assert pr1.last["ctx"].behind and not pr0.last["ctx"].behind and pr1.last["ctx"].p_boss < 0.6
    assert 0 > on["buy 0"][1] > off["buy 0"][1]             # the same $4 weighs less, interest step included
    assert on["reroll"][0] > off["reroll"][0]
    ahead = shop(money=25, jokers=["cavendish", "joker", "jolly"], items=[joker_item("joker", 4)])
    _, pr2, _ = priced(ahead, ShopConfig(rule_pace=True))
    assert not pr2.last["ctx"].behind


def test_rule_copier_places_blueprint_and_prices_moves():
    g = shop(money=20, jokers=["cavendish", "credit_card"], items=[joker_item("blueprint", 10)])
    off, _, _ = priced(g)
    on, _, _ = priced(g, ShopConfig(rule_copier=True))
    assert on["buy 0"][0] > off["buy 0"][0] + 0.01          # at the right end it copies nothing
    g.add_joker(Joker("blueprint", base_cost=10))           # cavendish, credit card, blueprint
    moves, _, ch = priced(g, ShopConfig(rule_copier=True))
    best = max((k for k in moves if k.startswith("move_joker")), key=lambda k: moves[k][0])
    assert moves[best][0] > 0
    w = World(copy.deepcopy(g))
    w.step(next(c.action for c in ch.cands if str(c.action) == best))
    i = [j.key for j in w.g.jokers].index("blueprint")
    assert w.g.jokers[i + 1].key == "cavendish"
    # without the rule: the rule-based player's order (chips, mult, xmult), never next to Blueprint
    g2 = shop(jokers=["cavendish", "joker"])
    assert rule_move(g2) == 0
    plain, _, _ = priced(g2)
    assert plain["move_joker 0 -> 1"][0] > 0 > plain["move_joker 1 -> 0"][0]


def test_rule_boss_models_the_visible_boss():
    g = shop(money=10, jokers=["joker"], ante=2)
    chances = {}
    for boss in ("hook", "needle", "water", "mouth", "eye", "club"):
        g.boss = boss
        pr = ShopPricer(ShopConfig(rule_boss=True))
        chances[boss] = pr.context(g).p_boss
        if boss == "hook":
            assert ShopPricer().context(g).p_boss == chances["hook"]        # no model: nothing changes
    plain = ShopPricer()
    g.boss = "water"
    assert chances["water"] < plain.context(g).p_boss       # no discards
    g.boss = "eye"
    assert chances["eye"] <= plain.context(g).p_boss
    g.boss = "mouth"
    assert chances["mouth"] <= plain.context(g).p_boss
    g.boss = "club"
    assert chances["club"] <= plain.context(g).p_boss
    # The Eye: a hand type counts once per round
    sc, ty = np.array([100.0, 100.0, 100.0, 10.0]), np.array([1, 1, 1, 0])
    scfg = StrengthConfig(discards="none")
    assert _eye_chance(sc, ty, 2, 0, 150, scfg, 1) == 0.0   # at most 100 + 10
    assert _eye_chance(sc, ty, 2, 0, 100, scfg, 1) > 0.5


# ------------------------------------------------------------------ no hidden-information leakage
ALL_RULES = {r: True for r in RULES}


def _states():
    s = shop(seed=9, money=14)
    full = shop(seed=10, money=30, jokers=["credit_card", "joker", "jolly", "sly", "banner"])
    full.consumables = [Consumable("tarot", "hermit"), Consumable("planet", "mercury")]
    pack = shop(seed=11, money=10, jokers=["jolly"])
    pack.open_pack("arcana", "normal")
    std = shop(seed=12, money=10, jokers=["jolly"])
    std.open_pack("standard", "normal")
    blind = shop(seed=13, money=22, jokers=["jolly"])
    blind.leave_shop()
    return [s, full, pack, std, blind]


def test_prices_do_not_depend_on_hidden_information():
    for g in _states():
        base, _, _ = priced(g, ShopConfig(**ALL_RULES), seed=5)
        for k in range(2):
            other, _, _ = priced(twin(g, 100 + k), ShopConfig(**ALL_RULES), seed=5)
            assert other == base                              # exactly: same samples, same copies


@pytest.mark.parametrize("mode", ["prior", "onestep", "close"])
def test_shop_decisions_do_not_depend_on_hidden_information(mode):
    cfg = dict(shop_prior="graded", shop=ShopConfig(**ALL_RULES), lam=0.5, solver_samples=4,
               solver_samples_inner=2, autoplay=False)
    if mode == "prior":
        cfg.update(search=False)
    else:
        cfg.update(shop_eval="onestep", close_calls=mode == "close", close_margin=1e9, close_rollouts=1,
                   close_horizon="blind", close_max_steps=12)
    states = _states()
    for g in ([states[0], states[4]] if mode == "close" else states[:3] + states[4:]):
        d1 = agent(AgentConfig(**cfg)).decide(World(g), explore=True)
        for k in range(2):
            d2 = agent(AgentConfig(**cfg)).decide(World(twin(g, 200 + k)), explore=True)
            assert d1.action == d2.action and np.array_equal(d1.policy, d2.policy)
            assert np.array_equal(d1.enc[1]["c_prior"], d2.enc[1]["c_prior"])


def test_pricing_leaves_the_game_untouched():
    g = _states()[1]
    before = copy.deepcopy(g)
    state = g.rng.getstate()
    priced(g, ShopConfig(**ALL_RULES))
    assert g.rng.getstate() == state and g.money == before.money
    assert [j.key for j in g.jokers] == [j.key for j in before.jokers]
    assert [c.uid for c in g.full_deck] == [c.uid for c in before.full_deck]


# ------------------------------------------------------------------ one-step V, close calls
def test_one_step_evaluation_records_a_search_policy():
    g = shop(money=14)
    ag = agent(AgentConfig(shop_prior="graded", shop_eval="onestep", lam=0.5))
    w = World(g)
    d = ag.decide(w)
    assert d.searched and d.reason == "onestep" and d.sims >= len(d.choice.cands)
    assert len(d.policy) == len(d.choice.cands) and abs(d.policy.sum() - 1) < 1e-9
    assert d.choice.cands[d.index].action == d.action
    assert ag.stats["budget_onestep"] == 1 and ag.stats["ovr_shop_n"] == 1 and ag.stats["ovr_shop_searched"] == 1
    assert override_summary(ag.stats)["shop"]["searched%"] == 100.0
    # the score is logits + weight x V of the state each option leads to
    root = ag.evaluate(w, True, random.Random(0), d.choice)
    i = next(k for k, c in enumerate(d.choice.cands) if c.kind == "leave")
    w2 = w.copy()
    w2.step(Action("leave"))
    v_leave = ag.values([w2])[0]
    j = next(k for k, c in enumerate(d.choice.cands) if c.kind == "sell_joker") if g.jokers else None
    lp = np.log(d.policy)
    assert 0.0 <= v_leave <= 1.0
    if j is not None:
        w3 = w.copy()
        w3.step(d.choice.cands[j].action)
        want = (root.logits[j] + 300.0 * ag.values([w3])[0]) - (root.logits[i] + 300.0 * v_leave)
        assert lp[j] - lp[i] == pytest.approx(want, abs=1e-6)
    # in a round nothing changes: the Gumbel search with its budgets
    g2 = Game(seed=5, stake="WHITE")
    g2.select_blind()
    d2 = ag.decide(World(g2))
    assert d2.reason not in ("onestep", "close", "prior")
    assert random_outcome(g, Action("reroll")) and not random_outcome(g, Action("leave"))
    assert ag.values([World(g)])[0] == pytest.approx(root.value, abs=1e-6)       # V without candidates


def test_search_off_means_priors_alone():
    ag = agent(AgentConfig(search=False, shop_prior="graded", shop_eval="onestep", close_calls=True))
    d = ag.decide(World(shop(money=14)))
    assert not d.searched and d.reason == "no search" and d.sims == 0


def test_close_calls_compare_the_top_two_on_common_futures(monkeypatch):
    g = shop(money=14)
    ag = agent(AgentConfig(shop_prior="graded", shop_eval="onestep", close_calls=True, close_margin=1e9,
                           close_rollouts=3, close_horizon="blind", close_max_steps=30, lam=0.5))
    seen = []
    real = Agent._rollout

    def spy(self, w, rng):
        seen.append((rng.getstate(), w.g.rng.getstate(), w.g.money))
        return real(self, w, rng)
    monkeypatch.setattr(Agent, "_rollout", spy)
    d = ag.decide(World(g))
    assert d.reason == "close" and d.searched and abs(d.policy.sum() - 1) < 1e-9
    assert ag.stats["close_calls"] == 1 and ag.stats["budget_close"] == 1 and len(seen) == 6
    for a, b in zip(seen[0::2], seen[1::2]):                # each future is shared by the two options
        assert a[0] == b[0]
    assert len({s[0] for s in seen}) == 3
    assert d.sims > len(d.choice.cands)                     # the rollouts' decisions are counted
    # a margin of 0 never asks for rollouts on a clear decision
    ag2 = agent(AgentConfig(shop_prior="graded", shop_eval="onestep", close_calls=True, close_margin=-1.0, lam=0.5))
    seen.clear()
    assert ag2.decide(World(g)).reason == "onestep" and not seen
    # outcome and horizon options
    ag3 = agent(AgentConfig(shop_prior="graded", shop_eval="prior", close_calls=True, close_margin=1e9,
                            close_rollouts=1, close_horizon="boss", close_outcome="blinds", close_max_steps=40))
    w = World(copy.deepcopy(g))
    w.step(Action("leave"))
    w2 = w.determinize(random.Random(1))
    w2.step(Action("select"))
    out, steps = real(ag3, w2, random.Random(2))
    assert 0 < steps <= 40 and out == w2.g.furthest_blind / 24.0


def test_rollouts_do_not_touch_the_reroll_average():
    ag = agent(AgentConfig(shop_prior="graded", shop_eval="prior", close_calls=True, close_margin=1e9,
                           close_rollouts=1, close_horizon="blind", close_max_steps=40, lam=0.5))
    g = shop(money=14)
    ag.decide(World(g, steps=3))
    assert sum(n for _, n in ag.pricer.fresh.values()) == 1          # only the shop the agent is really in


# ------------------------------------------------------------------ price inputs, training, tools
def test_price_inputs_reach_the_network_and_train(monkeypatch):
    monkeypatch.setattr("balatro_rl.az.world.MAX_STEPS", 40)        # a short game is enough
    cfg = AgentConfig(shop_prior="graded", shop_eval="onestep", price_feature=True, lam=0.5, budget_round=0,
                      budget_boss=0, budget_spectral=0)
    ag = agent(cfg)
    assert ag.price_feats and ag.net.config["price_feats"]
    rows, info = play_game(ag, 3, explore=True, record=True)
    built = [r for r in rows if not r["in_round"]]
    assert built and all(r["enc"][1]["c_price"].shape == (len(r["pi"]), N_PRICE) for r in rows)
    assert any(r["searched"] for r in built) and any(float(np.abs(r["enc"][1]["c_price"]).sum()) > 0 for r in built)
    assert all(float(np.abs(r["enc"][1]["c_price"]).sum()) == 0 for r in rows if r["in_round"])
    s, c, m = collate([r["enc"] for r in rows[:6]])
    assert c["c_price"].shape[:2] == m.shape
    rcfg = RewardConfig.from_dict({"potential": {"value_bound": "floor_sigmoid"}})
    before = ag.net.c_price.weight.detach().clone()
    logs, _ = train_on(ag.net, rows, 3, 16, 1e-3, "cpu", rcfg, rcfg.schedule(0))
    assert logs["policy"] > 0 and not torch.equal(before, ag.net.c_price.weight)
    # a network without the inputs plays the same game with the same priors
    plain = agent(AgentConfig(**{**cfg.__dict__, "price_feature": False}))
    rows2, info2 = play_game(plain, 3, explore=True, record=True)
    assert info2["blinds"] == info["blinds"] and len(rows2) == len(rows) and "c_price" not in rows2[0]["enc"][1]


def test_graded_game_and_override_statistics(monkeypatch):
    monkeypatch.setattr("balatro_rl.az.world.MAX_STEPS", 120)
    ag = agent(AgentConfig(search=False, shop_prior="graded", shop=ShopConfig(**ALL_RULES)))
    _, info = play_game(ag, 5, explore=False, record=False)
    s = override_summary(ag.stats)
    assert info["blinds"] >= 0 and s["shop"]["n"] > 0 and s["all"]["final%"] == 0     # priors alone: no override
    assert ag.pricer.stats["scorings"] > 0


def test_compare_runs_a_stage3_grid(tmp_path):
    grid = {"graded": {"cfg": {"shop_prior": "graded", "shop": {"rule_pace": True}, "shop_eval": "onestep",
                               "budget_round": 0, "budget_boss": 0},
                       "rewards": {"potential": {"value_bound": "floor_sigmoid"}}}}
    compare.run_grid(grid, str(tmp_path), games=1, workers=1, seed0=20_000)
    res = json.load(open(tmp_path / "graded.json"))
    assert res["per_game"][0][0] == 20_000 and res["agent_stats"]["budget_onestep"] > 0
    assert res["override"]["shop"]["searched%"] == 100.0
    rows = compare.table(str(tmp_path), None)
    assert rows[0]["cpu_sec/game"] > 0
