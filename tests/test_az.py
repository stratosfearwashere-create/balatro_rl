"""Tests for the unified agent (balatro_rl/az)."""
import copy
import random

import numpy as np
import pytest
import torch

from balatro_rl.sim.cards import Card
from balatro_rl.sim.game import Game, Consumable
from balatro_rl.sim.jokers import Joker
from balatro_rl.sim.scoring import Plan
from balatro_rl.az.world import World, Action, apply, determinize, infoset_key
from balatro_rl.az.actions import Config, enumerate_candidates, play_subsets, target_sets
from balatro_rl.az.solver import RoundSolver
from balatro_rl.az.agent import Agent, AgentConfig, autoplay_move, solver_prior
from balatro_rl.az.features import encode_state, encode_cands
from balatro_rl.az.net import AZNet, N_HEADS, collate

torch.set_num_threads(1)
ACCUMULATING = ["green_joker", "ride_the_bus", "ice_cream", "runner", "square", "wee", "lucky_cat", "hit_the_road",
                "castle", "yorick", "obelisk", "spare_trousers", "ramen", "selzer", "flash", "hiker", "vampire"]


def in_round(seed=0, jokers=(), stake="GOLD") -> Game:
    g = Game(seed=seed, stake=stake)
    for k in jokers:
        g.add_joker(Joker(k, base_cost=4))
    g.select_blind()
    return g


def mid_round(seed=0) -> Game:
    """A round after one discard, so part of the deck has been drawn."""
    g = in_round(seed)
    g.discard([0, 1])
    return g


# ------------------------------------------------------------------ A: enumeration and scoring
def test_every_play_is_enumerated_and_analyzed():
    g = in_round(3)
    ch = enumerate_candidates(World(g), random.Random(0))
    assert ch.complete
    assert len(ch.all_plays) == len(play_subsets(g)) == 218
    fast = dict(zip(play_subsets(g), g.predict_many(play_subsets(g), Plan(g))))
    for c in ch.all_plays:                        # the python analysis agrees with the compiled scorer
        key = tuple(sorted(c.action.cards))
        if c.action.cards == key:
            assert abs(c.score - fast[key][0]) < 1e-6


def test_identical_and_face_down_cards_together():
    g = in_round(1)
    g.hand[3].rank, g.hand[3].suit = g.hand[4].rank, g.hand[4].suit      # two identical cards: merged
    g.hand[0].hidden = True                                              # a face-down card: never merged
    ch = enumerate_candidates(World(g), random.Random(0))
    assert len(ch.all_plays) == len(play_subsets(g)) < 218
    assert target_sets(g, Consumable("tarot", "empress"), g.hand)


def test_card_order_is_optimised_when_it_matters():
    g = in_round(3)
    g.hand[0].enh, g.hand[1].enh = "GLASS", "MULT"
    g.hand[1].rank = g.hand[0].rank
    plan = Plan(g)
    glass_first = g.predict([0, 1], plan)[0]
    mult_first = g.predict([1, 0], plan)[0]
    assert mult_first > glass_first
    ch = enumerate_candidates(World(g), random.Random(0))
    pair = [c for c in ch.all_plays if sorted(c.action.cards) == [0, 1]]
    assert len(pair) == 1 and pair[0].action.cards == (1, 0) and pair[0].score == mult_first


def test_candidates_apply_cleanly():
    rng = random.Random(1)
    for seed in range(4):
        g = in_round(seed, ["green_joker", "castle"])
        g.consumables = [Consumable("tarot", "death"), Consumable("tarot", "strength"), Consumable("planet", "mars")]
        ch = enumerate_candidates(World(g), rng)
        kinds = {c.kind for c in ch.cands}
        assert {"play", "discard", "use", "sell_joker", "sell_cons", "move_joker"} <= kinds
        for c in ch.cands:
            w = World(determinize(g, rng))
            w.step(c.action)                       # raises if the action were illegal


def test_every_target_set_is_enumerated():
    g = in_round(5)
    death = Consumable("tarot", "death")
    empress = Consumable("tarot", "empress")
    n = len(g.hand)
    assert len(target_sets(g, death, g.hand)) <= n * (n - 1) // 2      # exactly 2 cards, identical ones merged
    subs = target_sets(g, empress, g.hand)
    assert all(1 <= len(s) <= 2 for s in subs) and len(subs) == n + n * (n - 1) // 2
    assert set(subs[0]) == {g.hand.index(c) for c in g.auto_targets("empress", g.hand)}   # the rule's choice first


def test_joker_state_deltas():
    g = in_round(2, ["green_joker", "ride_the_bus"])
    g.jokers[0].state["val"] = 3
    g.jokers[1].state["val"] = 4
    ch = enumerate_candidates(World(g), random.Random(0))
    face = [i for i, c in enumerate(g.hand) if c.is_face()]
    for c in ch.all_plays:
        d = {(s, k): (a, b) for s, k, a, b in c.jdiff}
        assert d[(0, "val")] == (3, 4)                                      # Green Joker +1 per hand
        has_face = any(g.hand[i].is_face() for i in c.action.cards)       # (every played card scores here or not)
        if not has_face:
            assert d[(1, "val")] == (4, 5)
    for c in ch.all_discards:
        assert dict(((s, k), (a, b)) for s, k, a, b in c.jdiff)[(0, "val")] == (3, 2)   # -1 per discard
    if face:
        resets = [c for c in ch.all_plays if (1, "val", 4, 0) in c.jdiff]
        assert resets, "a play scoring a face card resets Ride the Bus"


# ------------------------------------------------------------------ B: solver
def test_solver_clearing_and_hopeless():
    g = in_round(4)
    ch = enumerate_candidates(World(g), random.Random(0))
    g.chips = g.target - 1                          # any play clears
    ch = enumerate_candidates(World(g), random.Random(0))
    RoundSolver(samples=6).solve(g, ch, random.Random(0))
    assert all(c.p_clear == 1.0 for c in ch.cands if c.kind == "play")
    g2 = in_round(4)
    g2.target = 10 ** 9                             # nothing can clear
    ch2 = enumerate_candidates(World(g2), random.Random(0))
    RoundSolver(samples=6).solve(g2, ch2, random.Random(0))
    assert all(c.p_clear == 0.0 for c in ch2.cands if c.kind in ("play", "discard"))


def test_solver_is_deterministic_for_a_seed():
    g = mid_round(6)
    a = enumerate_candidates(World(g), random.Random(0))
    b = enumerate_candidates(World(g), random.Random(0))
    RoundSolver(samples=6).solve(g, a, random.Random(9))
    RoundSolver(samples=6).solve(g, b, random.Random(9))
    assert [c.p_clear for c in a.cands] == [c.p_clear for c in b.cands]


# ------------------------------------------------------------------ C: network
def test_untrained_network_plays_like_its_prior():
    torch.manual_seed(0)
    net = AZNet().eval()
    g = mid_round(7)
    w = World(g)
    rng = random.Random(0)
    ch = enumerate_candidates(w, rng)
    RoundSolver(samples=6).solve(g, ch, rng)
    prior = solver_prior(ch, 0.05)
    with torch.no_grad():
        s, c, m = collate([(encode_state(w), encode_cands(w, ch, prior))])
        logits, adj, heads = net(s, c, m)
    assert torch.all(adj == 0)
    assert np.allclose(logits[0].numpy(), prior, atol=1e-5)
    agent = Agent(net, AgentConfig(search=False, autoplay=False), seed=0)
    d = agent.decide(w)                             # without search it takes the prior's (the solver's) move
    assert d.action == d.choice.cands[int(np.argmax(solver_prior(d.choice, agent.cfg.tau)))].action


def test_network_batches_mixed_phases():
    net = AZNet().eval()
    samples = []
    rng = random.Random(0)
    for g in (in_round(1), Game(seed=2)):
        w = World(g)
        ch = enumerate_candidates(w, rng)
        samples.append((encode_state(w), encode_cands(w, ch, np.zeros(len(ch.cands)))))
    s, c, m = collate(samples)
    logits, adj, heads = net(s, c, m)
    assert logits.shape[0] == 2 and heads.shape == (2, N_HEADS)
    assert torch.isfinite(logits[m]).all()


# ------------------------------------------------------------------ D + no hidden-information leakage
def _twin(g: Game, seed: int) -> Game:
    """The same game as far as a player can tell: another draw order and another future (random generator)."""
    g2 = copy.deepcopy(g)
    r = random.Random(seed)
    r.shuffle(g2.deck)
    g2.rng = random.Random(seed + 1)
    return g2


def _decide(g: Game, search: bool, explore: bool):
    torch.manual_seed(0)
    net = AZNet().eval()
    cfg = AgentConfig(search=search, budget_round=4, budget_boss=4, budget_shop=4, budget_spectral=4,
                      solver_samples=4, solver_samples_inner=2, autoplay=False)
    d = Agent(net, cfg, seed=11).decide(World(g), explore=explore)
    return d.action, d.policy


@pytest.mark.parametrize("search", [False, True])
def test_decisions_do_not_depend_on_the_true_future(search):
    g = mid_round(8)
    g.hand[2].hidden = True                          # and a face-down card
    shop = Game(seed=9)
    shop.select_blind()
    shop.chips = shop.target
    shop.win_round()                                 # in the shop: its future rerolls / packs are hidden
    shop.money = 12
    for base in (g, shop):
        a1, p1 = _decide(base, search, explore=True)
        for k in range(2):
            twin = _twin(base, 100 + k)
            if base is g:
                assert [c.uid for c in twin.deck] != [c.uid for c in g.deck]
            a2, p2 = _decide(twin, search, explore=True)
            assert a1 == a2
            assert np.array_equal(p1, p2)


def test_determinize_hides_the_draw_order():
    g = mid_round(10)
    twin = _twin(g, 5)
    x, y = determinize(g, random.Random(3)), determinize(twin, random.Random(3))
    assert [c.uid for c in x.deck] == [c.uid for c in y.deck]
    assert infoset_key(World(g)) == infoset_key(World(twin))


def test_search_returns_a_legal_action_and_a_policy():
    g = mid_round(12)
    a, pi = _decide(g, True, explore=True)
    assert abs(pi.sum() - 1) < 1e-6
    World(copy.deepcopy(g)).step(a)


# ------------------------------------------------------------------ E: packs as mini-rounds
def test_arcana_pack_picks_with_targets():
    g = Game(seed=13)
    g.open_pack("arcana", "normal", return_to="BLIND_SELECT")
    g.pack_cards[0] = Consumable("tarot", "empress")
    ch = enumerate_candidates(World(g), random.Random(0), Config(max_targets=1000))
    picks = [c for c in ch.cands if c.kind == "pick" and c.action.idx == 0]
    n = len(g.pack_hand)
    assert len(picks) == len(target_sets(g, g.pack_cards[0], g.pack_hand)) and len(picks) > n
    assert all(c.best_after >= 0 and c.best_before >= 0 for c in picks)
    assert any(c.best_after > c.best_before for c in picks)      # Mult cards help the best hand
    w = World(copy.deepcopy(g))
    w.step(picks[0].action)
    assert sum(c.enh == "MULT" for c in w.g.full_deck) == len(picks[0].action.cards)


# ------------------------------------------------------------------ F: the auto-play guard
def _affected(g: Game, best, alternatives) -> bool:
    """Ground truth from the simulator: does any alternative play / discard change a joker's runtime
    state differently from `best`? (The blind is made unclearable in the copies so the round doesn't end.)"""
    def after(a: Action):
        x = copy.deepcopy(g)
        x.target = 10 ** 12
        x.rng = random.Random(0)
        apply(x, a)
        return [dict(j.state) for j in x.jokers], len(x.jokers)
    ref = after(best.action)
    return any(after(c.action) != ref for c in alternatives)


def test_autoplay_fires_in_the_plain_case():
    g = in_round(14, ["joker"])
    g.chips = g.target - 1
    ch = enumerate_candidates(World(g), random.Random(0))
    assert autoplay_move(ch) is not None


def test_autoplay_never_fires_when_an_accumulating_joker_is_affected():
    rng = random.Random(0)
    fired = affected_states = 0
    for seed in range(40):
        keys = rng.sample(ACCUMULATING, rng.randint(1, 3))
        g = in_round(seed, keys)
        for j in g.jokers:
            if "val" in j.state and isinstance(j.state["val"], (int, float)):
                j.state["val"] += rng.randint(0, 3)
        if rng.random() < 0.3:
            g.discards_left = 0
        g.chips = g.target - rng.choice([1, 50, 200])
        ch = enumerate_candidates(World(g), random.Random(seed))
        best = max([c for c in ch.all_plays if c.certain] or [None], key=lambda c: c.score if c else 0)
        if best is None:
            continue
        aff = _affected(g, best, ch.all_plays + ch.all_discards)
        affected_states += aff
        got = autoplay_move(ch)
        fired += got is not None
        if aff:
            assert got is None, f"auto-play fired with {keys} affected"
    assert affected_states > 10


def test_autoplay_blocked_by_green_joker_and_consumables():
    g = in_round(15, ["green_joker"])
    g.chips = g.target - 1
    assert autoplay_move(enumerate_candidates(World(g), random.Random(0))) is None   # discarding costs it mult
    g2 = in_round(15)
    g2.chips = g2.target - 1
    g2.consumables = [Consumable("planet", "mars")]
    assert autoplay_move(enumerate_candidates(World(g2), random.Random(0))) is None


def test_agent_logs_autoplay():
    g = in_round(14, ["joker"])
    g.chips = g.target - 1
    agent = Agent(AZNet().eval(), AgentConfig(search=False), seed=0)
    d = agent.decide(World(g))
    assert d.reason == "auto-play" and agent.stats["autoplay"] == 1 and agent.autoplay_rate() == 1.0


# ------------------------------------------------------------------ fast path for side-effect-free plays
def test_side_effect_free_fast_path_matches_full_analysis():
    """For states using only listed kinds, the compiled fast path equals analyze_play on every play."""
    from balatro_rl.az import actions as A
    from balatro_rl.sim import fastscore
    if not fastscore.ENABLED:
        pytest.skip("compiled scorer not built")
    rng = random.Random(3)
    safe = sorted(A.PLAY_SAFE_JOKERS | A.CONDITIONAL_JOKERS)
    bosses = sorted(A.PLAY_SAFE_BOSSES - {""})
    enh, seals, eds = (sorted(A.PLAY_SAFE_ENHANCEMENTS - {"HIDDEN"}), sorted(A.PLAY_SAFE_SEALS),
                       sorted(A.PLAY_SAFE_EDITIONS))
    covered = set()
    n_fast = n_full = 0
    for t in range(220):
        keys = [safe[(t * 3 + k) % len(safe)] for k in range(3)] + rng.sample(safe, 2)
        if t % 4 == 0:                                   # conditional jokers, also copied by Blueprint / Brainstorm
            keys = [rng.choice(sorted(A.CONDITIONAL_JOKERS)), rng.choice(["blueprint", "brainstorm", "joker"])] + keys[:3]
        g = in_round(700 + t, keys)
        covered |= set(keys)
        for j in g.jokers:
            if isinstance(j.state.get("val"), (int, float)):
                j.state["val"] += rng.randint(0, 5)
        if rng.random() < 0.5:
            g.boss, g.blind_idx, g.state = rng.choice(bosses), 2, "BLIND_SELECT"
            g.select_blind()
        for c in g.hand:
            c.enh, c.seal, c.edition = rng.choice(enh), rng.choice(seals), rng.choice(eds)
        if rng.random() < 0.2 and g.hand:
            g.hand[0].hidden = True
        plan = Plan(g)
        view = g.hand_view()
        filt = A.fast_play_filter(plan, view)
        assert filt is not None
        need = max(g.target - g.chips, 1) if rng.random() < 0.5 else 1
        slot_of = {j.uid: i for i, j in enumerate(g.jokers)}
        hs = A._HandScorer(g, plan, view)
        subs = A.play_subsets(g)
        ext = hs.fs.score_ext(subs)
        for s, e in zip(subs, ext):
            full = A.analyze_play(g, plan, view, s, need, slot_of, False)
            if filt(e[0], e[1], need):
                fast_ = A.light_play(s, e, need)
                assert vars(fast_) == vars(full), (keys, g.boss, s)
                n_fast += 1
            else:
                n_full += 1
    assert covered == set(safe)
    assert n_fast > 1000 and n_full > 50                 # both branches of the conditions were exercised


def test_unlisted_kinds_take_the_full_path():
    from balatro_rl.az import actions as A
    g = in_round(1, ["joker"])
    assert A.plays_side_effect_free(Plan(g), g.hand_view())
    for mutate in (lambda g: g.add_joker(Joker("green_joker", base_cost=4)),
                   lambda g: setattr(g.hand[0], "enh", "GLASS"),
                   lambda g: setattr(g.hand[0], "seal", "GOLD"),
                   lambda g: g.add_joker(Joker("some_future_joker", base_cost=4)),
                   lambda g: g.add_joker(Joker("todo_list", base_cost=4)),
                   lambda g: g.add_joker(Joker("misprint", base_cost=4))):
        h = copy.deepcopy(g)
        mutate(h)
        assert not A.plays_side_effect_free(Plan(h), h.hand_view())
