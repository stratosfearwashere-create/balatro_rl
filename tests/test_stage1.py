"""Search-versus-prior controls and their diagnostics: configurable search strength, common random numbers
at the root, the override statistics, the bounded value and the comparison tool."""
import json
import random
import subprocess
import sys

import numpy as np
import torch

from balatro_rl.az import compare
from balatro_rl.az.agent import Agent, AgentConfig, OVERRIDE_PHASES, override_summary, phase_of
from balatro_rl.az.net import AZNet
from balatro_rl.az.search import GumbelSearch, Node
from balatro_rl.az.train import new_net, play_game
from balatro_rl.az.world import World
from balatro_rl.rewards.config import PotentialConfig, RewardConfig
from balatro_rl.sim.game import Game

torch.set_num_threads(1)
SMALL = dict(budget_round=4, budget_boss=4, budget_shop=4, budget_spectral=4)


def in_round(seed=0) -> World:
    g = Game(seed=seed, stake="WHITE")
    g.select_blind()
    return World(g)


def test_search_strength_comes_from_the_agent_config():
    node = Node(None, logits=np.zeros(3))
    node.n[:] = [2, 1, 0]
    cq = np.array([1.0, 0.5, 0.0])
    weak, strong = GumbelSearch(None), GumbelSearch(None, c_scale=1.0)
    assert np.allclose(weak.sigma(node, cq), 5.2 * cq) and np.allclose(strong.sigma(node, cq), 52 * cq)
    seen = {}
    real = GumbelSearch.__init__

    def spy(self, *a, **kw):
        seen.update(kw)
        real(self, *a, **kw)
    GumbelSearch.__init__ = spy
    try:
        agent = Agent(AZNet().eval(), AgentConfig(c_scale=0.7, c_visit=20.0, crn=True, autoplay=False, **SMALL), seed=0)
        agent.decide(in_round(1))
    finally:
        GumbelSearch.__init__ = real
    assert seen["c_scale"] == 0.7 and seen["c_visit"] == 20.0 and seen["crn"] is True
    # a separate strength outside rounds (the interior nodes of a search use its root's strength)
    GumbelSearch.__init__ = spy
    try:
        agent = Agent(AZNet().eval(), AgentConfig(c_scale=0.7, c_scale_build=2.0, autoplay=False, **SMALL), seed=0)
        agent.decide(in_round(1))
        assert seen["c_scale"] == 0.7
        agent.decide(World(Game(seed=1, stake="WHITE")))            # blind select
        assert seen["c_scale"] == 2.0
    finally:
        GumbelSearch.__init__ = real


def _sweep_decks(crn: bool):
    """The draw order each root candidate was simulated on, visit by visit."""
    agent = Agent(AZNet().eval(), AgentConfig(crn=crn, autoplay=False, **SMALL), seed=0)
    w = in_round(2)
    rng = random.Random(5)
    root = agent.evaluate(w, True, rng)
    decks = []
    real = World.determinize

    def spy(self, r):
        out = real(self, r)
        if self is root.world:
            decks.append(tuple(c.uid for c in out.g.deck))
        return out
    World.determinize = spy
    try:
        search = GumbelSearch(agent._expand_inner(rng), m_root=4, crn=crn)
        search.run(root, 8, rng, explore=False)
    finally:
        World.determinize = real
    return decks


def test_crn_gives_every_candidate_of_a_sweep_the_same_future():
    decks = _sweep_decks(True)
    assert len(decks) >= 6 and len(set(decks[:4])) == 1            # first sweep: 4 candidates, one draw order
    assert len(set(decks)) > 1                                      # later sweeps draw new futures
    assert len(set(_sweep_decks(False)[:4])) == 4                   # without it: a new future per visit


def test_override_statistics():
    agent = Agent(AZNet().eval(), AgentConfig(search=False), seed=0)
    _, info = play_game(agent, 3, explore=False, record=False)
    s = override_summary(agent.stats)
    assert set(s) <= set(OVERRIDE_PHASES) | {"all"} and {"round", "shop", "blind", "all"} <= set(s)
    assert s["all"]["n"] == sum(v["n"] for k, v in s.items() if k != "all")
    assert s["all"]["n"] == agent.stats["decisions"] - agent.stats["autoplay"]
    # an untrained network without search is its prior: nothing is overridden, nothing searched
    assert all(v["final%"] == v["net%"] == v["search%"] == v["searched%"] == 0 for v in s.values())
    # a search that outweighs the prior overrides it somewhere (with the bounded value: the unbounded one
    # starts below 0 nearly everywhere, which the search's [0, 1] scale clips to 0 for every candidate)
    torch.manual_seed(0)
    pot = PotentialConfig(value_bound="floor_sigmoid")
    agent = Agent(new_net(pot).eval(), AgentConfig(c_scale=20.0, lam=0.5, **SMALL), seed=0, potential=pot)
    w = in_round(4)
    for _ in range(12):
        d = agent.decide(w)
        w.step(d.action)
        if w.done:
            break
    s = override_summary(agent.stats)
    assert s["all"]["searched%"] > 0 and s["all"]["final%"] == s["all"]["search%"] > 0 and s["all"]["net%"] == 0
    g = Game(seed=1, stake="WHITE")
    assert phase_of(g) == "blind"
    g.blind_idx = 2
    g.select_blind()
    assert phase_of(g) == "boss"


def test_bounded_value_stays_between_a_loss_now_and_one():
    torch.manual_seed(0)
    net = new_net(PotentialConfig(value_bound="floor_sigmoid", value_init_scale=1.0, value_init_bias=-1.9))
    assert net.config["value_bound"] == "floor_sigmoid"
    out = torch.randn(64, 12) * 5
    phi = torch.rand(64) * 1.7 - 0.7
    prog = torch.rand(64)
    v = net.value(out, phi, prog, 0.5)
    assert bool(((v >= 0.5 * prog) & (v <= 1.0)).all())
    assert float(net.heads.weight[0].abs().sum()) == 0 and float(net.heads.bias[0]) == 0   # R starts at 0
    zero = torch.zeros(64, 12)                                       # R = 0: the value starts from Phi
    want = 0.5 * prog + (1 - 0.5 * prog) * torch.sigmoid(1.0 * phi - 1.9)
    assert torch.allclose(net.value(zero, phi, prog, 0.5), want)
    assert torch.allclose(net.value(zero, phi, prog, 0.0), torch.sigmoid(phi - 1.9))   # lam 0: plain P(win)
    plain = AZNet()                                                  # the default is unchanged: V = Phi + R
    assert plain.config["value_bound"] == "none" and torch.allclose(plain.value(out, phi, prog, 0.5), out[:, 0] + phi)
    # the agent's value of a live state is never below the value of losing right now
    agent = Agent(net.eval(), AgentConfig(search=False, lam=0.5), seed=0,
                  potential=PotentialConfig(value_bound="floor_sigmoid"))
    w = in_round(3)
    w.g.furthest_blind = 6
    node = agent.evaluate(w, True, random.Random(0))
    assert agent.terminal(w, lost=True).value == 0.5 * 6 / 24 <= node.value <= 1.0


def test_training_with_the_bounded_value_logs_the_new_diagnostics(tmp_path):
    cfgfile = tmp_path / "rewards.yaml"
    cfgfile.write_text("rewards:\n  potential:\n    value_bound: floor_sigmoid\n  solver_kl:\n    kappa: 0.2\n")
    assert RewardConfig.load(str(cfgfile)).potential.value_bound == "floor_sigmoid"
    out, data = tmp_path / "az.pt", tmp_path / "data"
    cmd = [sys.executable, "-m", "balatro_rl.az.train", "run", "--iters", "1", "--games", "2", "--workers", "1",
           "--warmup", "0", "--steps", "2", "--batch", "8", "--out", str(out), "--data", str(data),
           "--reward-config", str(cfgfile), "--eval-every", "1", "--eval-games", "1",
           "--cfg", json.dumps({**SMALL, "tau": 0.2, "c_scale": 0.5, "heur_bonus": 1.0, "crn": True})]
    subprocess.run(cmd, check=True, capture_output=True)
    ck = torch.load(out, weights_only=False)
    assert ck["config"]["value_bound"] == "floor_sigmoid" and ck["extra"]["rewards"]["solver_kl"]["kappa"] == 0.2
    row = json.loads(open(tmp_path / "az_log.jsonl").readline())
    assert row["loss_policy_kl"] >= -1e-6 and row["loss_policy"] >= row["loss_policy_kl"]
    assert row["loss_value_mse"] >= 0 and row["kappa"] > 0.19
    assert row["lam"] == 0.5 and row["lambda_gate"]["open"] is False        # the gate holds lambda
    for o in (row["override"], row["eval"]["override"]):
        assert o["all"]["n"] > 0 and 0 <= o["all"]["final%"] <= 100
    assert len(row["eval"]["per_game"]) == 1 and row["eval"]["cpu_sec/game"] > 0


def test_compare_table_and_decisions_changed(tmp_path, capsys):
    def result(blinds):
        return {"per_game": [[10000 + i, b, int(b == 24), 3] for i, b in enumerate(blinds)], "workers": 2,
                "cpu_sec/game": 4.0, "override": {"all": {"n": 10, "final%": 12.5, "searched%": 50.0},
                                                  "shop": {"n": 4, "final%": 25.0, "searched%": 100.0}}}
    for name, blinds in (("base", [6, 9, 12, 3]), ("v", [9, 9, 15, 24])):
        with open(tmp_path / f"{name}.json", "w") as f:
            json.dump(result(blinds), f)
    with open(tmp_path / "v.changed.json", "w") as f:
        json.dump({"changed%": 7.0}, f)
    rows = {r["variant"]: r for r in compare.table(str(tmp_path), "base", detail=True)}
    assert "shop: 25.0% of 4" in capsys.readouterr().out
    assert rows["base"]["blinds"] == 7.5 and rows["base"]["games/hour"] == 1800 and "diff" not in rows["base"]
    v = rows["v"]
    assert v["wins"] == 1 and v["changed%"] == 7.0
    d = np.array([3, 0, 3, 21.0])
    assert v["diff"] == d.mean() and abs(v["diff_se"] - d.std(ddof=1) / 2) < 1e-12
    # the same variant asked again in the same positions never differs; a different prior does
    same = {"cfg": {}, "search": False}
    n, diff, n_net, diff_net = compare._changed_worker((same, same, [3]))
    assert n > 50 and diff == 0 and 0 < n_net <= n
    other = {"cfg": {"heur_bonus": -3.0}, "search": False}
    assert compare._changed_worker((same, other, [3]))[1] > 0


def test_in_round_search_only_on_hard_decisions():
    from balatro_rl.az.train import play_game as play
    # default: unchanged, nothing is "easy"
    agent = Agent(AZNet().eval(), AgentConfig(**SMALL), seed=0)
    play(agent, 5, explore=False, record=False)
    assert agent.stats["budget_easy"] == 0 and agent.stats["budget_round"] > 0
    # hard only: some in-round decisions are taken without search, bosses are still searched
    cfg = AgentConfig(round_search="hard", **SMALL)
    agent = Agent(AZNet().eval(), cfg, seed=0)
    play(agent, 5, explore=False, record=False)
    st = agent.stats
    assert st["budget_easy"] > 0 and st["budget_boss"] > 0
    agent.cfg.hard_margin = 1.0
    # the pieces of the definition
    w = in_round(6)
    node = agent.evaluate(w, True, random.Random(0))
    node.logits = np.full(len(node.logits), -9.0)
    node.logits[0], node.logits[1] = 0.0, -0.5
    assert agent.hard(node)                                          # top two within the margin
    node.logits[1] = -5.0
    for c in node.choice.cands:
        c.jdiff, c.effects = (), ()
    node.choice.cands = [c for c in node.choice.cands if c.kind != "use"]
    node.logits = node.logits[:len(node.choice.cands)]
    assert not agent.hard(node) and agent.budget(node)[0] == 0      # ("clear-cut" or "easy")
    node.logits[2] = -2.0                                            # a close option that treats a joker differently
    node.choice.cands[2].jdiff = ((0, "val", 1, 2),)
    assert agent.hard(node)
    node.choice.cands[2].jdiff = ()
    w.g.blind_idx = 2                                                # a boss blind is always hard
    assert agent.hard(node)
    agent.cfg.hard_boss = False
    assert not agent.hard(node)
