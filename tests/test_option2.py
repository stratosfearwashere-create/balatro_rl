"""Option 2: calculated priors everywhere, a learned adjustment, trained by policy-gradient (PPO) steps
without search. The growth-aware prior for rounds, the advantages, and a small end-to-end run."""
import json
import random
import subprocess
import sys

import numpy as np
import torch

from balatro_rl.az.actions import enumerate_candidates
from balatro_rl.az.agent import Agent, AgentConfig, solver_prior
from balatro_rl.az.net import AZNet
from balatro_rl.az.train import ppo_prepare
from balatro_rl.az.world import World
from balatro_rl.rewards.config import RewardConfig
from balatro_rl.sim.game import Game
from balatro_rl.sim.jokers import Joker

torch.set_num_threads(1)


def in_round(seed, *jokers) -> World:
    g = Game(seed=seed, stake="WHITE")
    for k in jokers:
        g.add_joker(Joker(k, base_cost=4))
    g.select_blind()
    return World(g)


def _near(agent, w):
    rng = random.Random(0)
    choice = enumerate_candidates(w, rng, agent.cfg.root_cfg)
    agent.solver.solve(w.g, choice, rng, samples=6, depth=1)
    prior = solver_prior(choice, agent.cfg.tau)
    return choice, prior, agent.growth_bonus(w, choice, prior)


def test_growth_bonus_prefers_what_grows_the_build():
    agent = Agent(AZNet().eval(), AgentConfig(search=False, round_growth=True), seed=0)
    w = in_round(3, "green_joker")
    w.g.jokers[0].state["val"] = 6
    w.g.target = 10                                   # a blind any hand clears: every option is "near"
    choice, prior, bonus = _near(agent, w)
    plays = [i for i, c in enumerate(choice.cands) if c.kind == "play" and prior[i] >= -1.0]
    discards = [i for i, c in enumerate(choice.cands) if c.kind == "discard" and prior[i] >= -1.0]
    assert plays and discards
    assert max(bonus[i] for i in plays) == 0.0                       # a hand grows Green Joker: the best
    assert max(bonus[i] for i in discards) < 0.0                     # a discard costs it mult
    assert bonus.min() >= -agent.cfg.growth_cap and bonus.max() <= 0.0
    assert all(bonus[i] == 0.0 for i, c in enumerate(choice.cands) if prior[i] < -1.0)   # outside the margin
    # without a joker that reacts, plays and discards change nothing lasting: no bonus anywhere
    w = in_round(3, "joker")
    w.g.target = 10
    assert not _near(agent, w)[2].any()
    # the flag is off by default and then evaluate() never calls it
    plain = Agent(AZNet().eval(), AgentConfig(search=False), seed=0)
    plain.growth_bonus = None
    plain.evaluate(in_round(3, "green_joker"), True, random.Random(0))


def test_growth_bonus_reads_nothing_hidden():
    """Two games that differ only in the draw order and the game's random generator get the same bonus."""
    agent = Agent(AZNet().eval(), AgentConfig(search=False, round_growth=True), seed=0)
    out = []
    for shuffle in (1, 2):
        w = in_round(5, "green_joker", "ride_the_bus")
        w.g.target = 10
        random.Random(shuffle).shuffle(w.g.deck)
        w.g.rng = random.Random(shuffle)
        choice, prior, bonus = _near(agent, w)
        out.append((prior.tolist(), bonus.tolist()))
    assert out[0] == out[1]


def test_ppo_advantages():
    pi = np.array([0.25, 0.75], np.float32)
    rows = [{"game": 7, "step": s, "value": v, "win": 0.0, "progress": 0.5, "pi": pi, "a": a}
            for s, v, a in ((2, 0.4, 1), (0, 0.2, 0), (1, 0.3, 1))]            # out of order on purpose
    rows.append({"game": 8, "step": 0, "value": 0.1, "win": 1.0, "progress": 1.0, "pi": pi, "a": 0})
    info = ppo_prepare(rows, lam=0.5, gae=0.5)
    by = {(r["game"], r["step"]): r for r in rows}
    z7 = 0.5 * 0.5                                                             # (1 - lam) * win + lam * progress
    d2, d1, d0 = z7 - 0.4, 0.4 - 0.3, 0.3 - 0.2
    assert np.isclose(by[(7, 2)]["adv_raw"], d2)
    assert np.isclose(by[(7, 1)]["adv_raw"], d1 + 0.5 * d2)
    assert np.isclose(by[(7, 0)]["adv_raw"], d0 + 0.5 * (d1 + 0.5 * d2))
    assert np.isclose(by[(8, 0)]["adv_raw"], 1.0 - 0.1)                        # a won game: z = 1
    assert np.isclose(by[(7, 0)]["logp"], np.log(0.25)) and np.isclose(by[(7, 1)]["logp"], np.log(0.75))
    adv = np.array([r["adv"] for r in rows])
    assert abs(adv.mean()) < 1e-6 and abs(adv.std() - 1.0) < 1e-3 and info["games"] == 2


def test_ppo_training_run_without_search(tmp_path):
    cfgfile = tmp_path / "rewards.yaml"
    cfgfile.write_text("rewards:\n  potential:\n    value_bound: floor_sigmoid\n"
                       "  solver_kl:\n    kappa: 0.2\n    all_phases: true\n")
    assert RewardConfig.load(str(cfgfile)).solver_kl.all_phases is True
    out, data = tmp_path / "az.pt", tmp_path / "data"
    cfg = {"shop_prior": "graded", "round_growth": True, "shop": {"rule_arcana": True}}
    cmd = [sys.executable, "-m", "balatro_rl.az.train", "run", "--algo", "ppo", "--iters", "1", "--games", "2",
           "--workers", "1", "--batch", "16", "--ppo-epochs", "1", "--out", str(out), "--data", str(data),
           "--reward-config", str(cfgfile), "--eval-every", "1", "--eval-games", "1", "--cfg", json.dumps(cfg)]
    subprocess.run(cmd, check=True, capture_output=True)
    row = json.loads(open(tmp_path / "az_log.jsonl").readline())
    assert row["algo"] == "ppo" and row["search"] is False and row["sims/decision"] == 0
    for k in ("loss_ppo_policy", "loss_entropy", "loss_approx_kl", "loss_clip_frac", "loss_value", "loss_solver_kl"):
        assert k in row and np.isfinite(row[k])
    assert row["loss_entropy"] > 0 and "loss_policy" not in row
    assert row["override"]["all"]["searched%"] == 0 and row["eval"]["override"]["all"]["searched%"] == 0
    ck = torch.load(out, weights_only=False)
    assert ck["config"]["value_bound"] == "floor_sigmoid"
