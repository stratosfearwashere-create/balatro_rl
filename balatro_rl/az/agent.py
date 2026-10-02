"""The agent: one decision process for the whole run.

decide(world):
  1. enumerate candidates with exact scores and effects (A); in a round, the solver adds P(clear) and
     expected chips (B)
  2. auto-play shortcut (F), only when every guard holds (see autoplay_move)
  3. priors: in a round, the solver's utility; elsewhere, the rule-based player's choice
  4. network: logits = prior + learned adjustment, value and auxiliary heads (C)
  5. search with an adaptive budget (D); consumable uses, holds, sells and joker moves are ordinary
     candidates everywhere (E)

The solver is a prior, never the final decider: its utility only enters as the prior logit, the network
adds its own adjustment on top, and the search compares actions by the network's value of what follows
(V = Phi + R, trained towards z; rewards/), not by the solver's P(clear). So playing instead of discarding with
Green Joker, resetting Ride the Bus, spending Mystic Summit's discards or a tarot a build needs later can
all be overruled once the value network has learned what they cost.

Override statistics (agent.stats, summarised by override_summary): per phase (round, boss, shop, pack,
blind), how many decisions the network saw, how many were searched, and how often
  - the final choice is not one of the prior's top choices            ("final")
  - ... and the prior had it at least 1 logit below its top choice    ("strong": a flat prior makes
    "final" large by itself, since near-ties are overridden by anything)
  - the network alone (prior + adjustment) would not pick one of them  ("net")
  - the search changed the network's own choice                        ("search")
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch

from ..rewards.config import PotentialConfig
from ..rewards.potential import Potential
from ..rewards.targets import z_target
from .actions import Choice, Config, enumerate_candidates
from .features import encode_cands, encode_state
from .net import AZNet, collate
from .search import GumbelSearch, Node, _softmax
from .solver import RoundSolver
from .world import Action, World


@dataclass
class AgentConfig:
    lam: float = 0.0                # value of a finished game: z = (1 - lam) * win + lam * blinds / 24
                                    # (set to the reward schedule's lambda at the network's training step)
    tau: float = 0.05               # solver prior: one logit per 0.05 of P(clear)
    heur_bonus: float = 3.0         # prior logit for the rule-based player's choice outside rounds
    budget_round: int = 8           # simulations per decision
    budget_boss: int = 16
    budget_shop: int = 24
    budget_spectral: int = 32
    budget_clear: int = 0           # in a round, when the policy is already this sure (clear_cut)
    clear_cut: float = 0.9
    m_root: int = 8
    c_visit: float = 50.0           # search vs prior: sigma = (c_visit + max N) * c_scale * Q
    c_scale: float = 0.1
    c_scale_build: float | None = None   # c_scale outside rounds (shop, packs, blind select); None: c_scale
    crn: bool = False               # root candidates share each sweep's sampled future (search.py)
    value_range: tuple | None = (0.0, 1.0)   # fixed scale for Q in the search (None: per-tree min-max)
    solver_samples: int = 12        # root
    solver_samples_inner: int = 4   # inside the search
    boss_depth: int = 2             # solver depth on boss blinds (root)
    autoplay: bool = True
    search: bool = True
    root_cfg: Config = field(default_factory=lambda: Config(max_analyzed=700))
    inner_cfg: Config = field(default_factory=lambda: Config(max_plays=24, max_discards=12, max_analyzed=160,
                                                             max_targets=8, use_samples=1))


@dataclass
class Decision:
    action: Action
    index: int                        # into choice.cands (-1 for the auto-play shortcut)
    choice: Choice
    policy: np.ndarray                # improved policy over choice.cands (training target)
    searched: bool                    # policy came from a search (else it is just the network's)
    enc: tuple | None = None          # (state, candidates) as fed to the network
    value: float = 0.0
    heads: dict | None = None
    sims: int = 0
    reason: str = ""


# ------------------------------------------------------------------ priors
def solver_prior(choice: Choice, tau: float) -> np.ndarray:
    evald = [c.p_clear for c in choice.cands if c.kind in ("play", "discard") and c.p_clear >= 0]
    best_p = max(evald, default=0.0)
    u = []
    for c in choice.cands:
        k = c.kind
        if k in ("play", "discard") and c.p_clear >= 0:
            v = c.p_clear + 0.02 * min(c.e_chips, 3.0) + (0.01 if k == "play" and c.clears else 0.0)
        elif k == "play":
            v = best_p - 0.3 - 0.2 * (1 - min(1.0, c.score / max(choice.need, 1)))
        elif k == "discard":
            v = best_p - 0.3
        elif k == "use":
            v = (c.p_clear if c.p_clear >= 0 else best_p) - 0.03
        else:                                               # sell / move: the solver has no opinion
            v = None
        u.append(v)
    known = [x for x in u if x is not None]
    floor = (min(known) if known else 0.0) - 0.5
    u = np.array([floor if x is None else x for x in u], dtype=float)
    return np.clip((u - u.max()) / tau, -30.0, 0.0)


def heuristic_action(w: World) -> Action | None:
    """The rule-based player's move outside rounds, as an Action (None if it can't be expressed)."""
    from .. import env as E
    from ..heuristic import HeuristicPolicy
    cnt = E.Counters()
    cnt.rerolls, cnt.swaps = w.rerolls, w.moves
    obs = E.encode(w.g, cnt)
    h = HeuristicPolicy()
    h.shop_rerolls = min(w.rerolls, 3)
    a = h.act(w.g, obs)
    simple = {E.A_SELECT: "select", E.A_SKIP: "skip", E.A_REROLL_BOSS: "reroll_boss",
              E.A_REROLL: "reroll", E.A_LEAVE: "leave", E.A_PSKIP: "pack_skip"}
    if a in simple:
        return Action(simple[a])
    for lo, hi, kind in ((E.A_BUY, E.A_BUY_PACK, "buy"), (E.A_BUY_PACK, E.A_VOUCHER, "buy_pack"),
                         (E.A_VOUCHER, E.A_REROLL, "voucher"),
                         (E.A_SELL_J, E.A_SELL_C, "sell_joker"), (E.A_SELL_C, E.A_USE_C, "sell_cons"),
                         (E.A_USE_C, E.A_PICK, "use"), (E.A_PICK, E.A_PSKIP, "pick")):
        if lo <= a < hi:
            return Action(kind, a - lo)
    if E.A_SWAP <= a < E.N_ACTIONS:
        return Action("move_joker", a - E.A_SWAP, to=a - E.A_SWAP + 1)
    return None


def rule_prior(w: World, choice: Choice, bonus: float) -> np.ndarray:
    pri = np.zeros(len(choice.cands))
    try:
        h = heuristic_action(w)
    except Exception:
        h = None
    if h is None:
        return pri
    for i, c in enumerate(choice.cands):
        a = c.action
        if a.kind == h.kind and a.idx == h.idx and (h.to < 0 or a.to == h.to):
            pri[i] = bonus                    # uses / picks: the first matching target set is the rule's own
            break
    return pri


# ------------------------------------------------------------------ F: auto-play shortcut
def autoplay_move(choice: Choice):
    """The solver's move, when taking it without network or search can't cost anything beyond this round.
    All of these must hold:
      - some play clears the blind with certainty (its score with every chance effect failing is enough);
      - every legal play and discard was examined (choice.complete), and every one of them changes the
        jokers' runtime state exactly as that play does (so no alternative this round treats Green Joker,
        Ride the Bus, Ice Cream, Castle ... differently);
      - no consumable can be used (a use changes the deck, levels, money or jokers);
      - no play or discard has lasting side effects different from that play's: money, deck changes, hand
        levels, created consumables, and the Gold cards / Blue seals it leaves held when the round ends.
    Returns the candidate to play, or None."""
    if choice.phase != "SELECTING_HAND" or not choice.complete:
        return None
    certain = [c for c in choice.all_plays if c.certain]
    if not certain:
        return None
    best = max(certain, key=lambda c: (c.score, -len(c.action.cards)))
    if any(c.kind == "use" for c in choice.cands):
        return None
    for c in choice.all_plays + choice.all_discards:
        if c.jdiff != best.jdiff or c.effects != best.effects:
            return None
    return best


# ------------------------------------------------------------------ the agent
class Agent:
    def __init__(self, net: AZNet | None = None, cfg: AgentConfig | None = None, seed: int = 0, device: str = "cpu",
                 potential: PotentialConfig | None = None):
        self.net = net if net is not None else AZNet().eval()
        self.cfg = cfg or AgentConfig()
        self.device = device
        self.rng = random.Random(seed)
        self.solver = RoundSolver()
        self.stats = Counter()
        self.potential = Potential(potential)

    def reseed(self, seed: int):
        self.rng = random.Random(seed)

    # network evaluation of one world
    def evaluate(self, w: World, root: bool, rng: random.Random, choice: Choice | None = None) -> Node:
        """Candidates, solver, priors and network outputs for `w` (not modified)."""
        if w.done:
            return self.terminal(w)
        cfg = self.cfg
        if choice is None:
            choice = enumerate_candidates(w, rng, cfg.root_cfg if root else cfg.inner_cfg)
        if not choice.cands:                    # nothing legal (an empty hand): the run is lost
            return self.terminal(w, lost=True)
        if choice.phase == "SELECTING_HAND":
            depth = cfg.boss_depth if (root and w.g.blind_idx == 2) else 1
            samples = cfg.solver_samples if root else cfg.solver_samples_inner
            if depth > 1:
                samples = max(4, samples // 2)
            self.solver.solve(w.g, choice, rng, samples=samples, depth=depth)
            prior = solver_prior(choice, cfg.tau)
        else:
            prior = rule_prior(w, choice, cfg.heur_bonus)
        state = encode_state(w)
        state["phi"] = np.float32(self.potential(w.g))
        state["prog"] = np.float32(self.potential.progress(w.g))
        enc = (state, encode_cands(w, choice, prior))
        with torch.no_grad():
            s, c, m = collate([enc], self.device)
            logits, _, out = self.net(s, c, m)
        heads = self.net.split_heads(out[0], s["phi"][0], s["prog"][0], cfg.lam)
        heads = {k: float(v) for k, v in heads.items() if v.dim() == 0}
        heads["phi"] = float(state["phi"])
        return Node(w, choice, enc, logits[0].float().cpu().numpy().astype(float), heads["value"], heads=heads)

    def terminal(self, w: World, lost: bool = False) -> Node:
        g = w.g
        v = z_target(float(g.state == "WON" and not lost), g.furthest_blind / 24.0, self.cfg.lam)
        return Node(w, value=v, terminal=True)

    def budget(self, node: Node) -> tuple[int, str]:
        cfg = self.cfg
        ch, g = node.choice, node.world.g
        if not cfg.search or len(ch.cands) <= 1:
            return 0, "single" if len(ch.cands) <= 1 else "no search"
        spectral = any(c.kind in ("use", "pick") and _is_spectral(g, c.action) for c in ch.cands)
        if spectral:
            return cfg.budget_spectral, "spectral"
        if ch.phase == "SELECTING_HAND":
            if _softmax(node.logits).max() >= cfg.clear_cut:
                return cfg.budget_clear, "clear-cut"
            if g.blind_idx == 2:
                return cfg.budget_boss, "boss"
            return cfg.budget_round, "round"
        return cfg.budget_shop, "build"

    def decide(self, w: World, explore: bool = False) -> Decision:
        """Choose an action for `w` (not modified). explore: Gumbel noise at the root (self-play)."""
        rng = self.rng
        self.stats["decisions"] += 1
        choice = enumerate_candidates(w, rng, self.cfg.root_cfg)
        if not choice.cands:
            return Decision(None, -1, choice, np.zeros(0), False, reason="no legal action")
        if w.g.state == "SELECTING_HAND" and self.cfg.autoplay:
            if any(c.certain for c in choice.all_plays):
                self.stats["autoplay_eligible"] += 1
            best = autoplay_move(choice)
            if best is not None:
                self.stats["autoplay"] += 1
                return Decision(best.action, -1, choice, np.zeros(0), False, reason="auto-play")
        root = self.evaluate(w, True, rng, choice)         # simulations start from resamplings of w
        budget, reason = self.budget(root)
        cfg = self.cfg
        build = w.g.state != "SELECTING_HAND" and cfg.c_scale_build is not None
        search = GumbelSearch(self._expand_inner(rng), c_visit=cfg.c_visit,
                              c_scale=cfg.c_scale_build if build else cfg.c_scale, m_root=cfg.m_root,
                              value_range=cfg.value_range, crn=cfg.crn)
        idx, pi = search.run(root, budget, rng, explore=explore)
        self.stats[f"budget_{reason}"] += 1
        self.stats["sims"] += getattr(search, "used", 0)
        self._count_override(w, root, idx, budget > 0)
        return Decision(root.choice.cands[idx].action, idx, root.choice, pi, budget > 0, enc=root.enc,
                        value=root.value, heads=root.heads, sims=getattr(search, "used", 0), reason=reason)

    def _count_override(self, w: World, root: Node, idx: int, searched: bool):
        prior = root.enc[1]["c_prior"]
        top = float(prior.max()) - 1e-6
        net_idx = int(np.argmax(root.logits))
        ph = phase_of(w.g)
        st = self.stats
        st[f"ovr_{ph}_n"] += 1
        st[f"ovr_{ph}_searched"] += int(searched)
        st[f"ovr_{ph}_final"] += int(prior[idx] < top)
        st[f"ovr_{ph}_strong"] += int(prior[idx] < top - 1.0)
        st[f"ovr_{ph}_net"] += int(prior[net_idx] < top)
        st[f"ovr_{ph}_search"] += int(idx != net_idx)

    def _expand_inner(self, rng):
        return lambda w, root: self.evaluate(w, root, rng)

    def autoplay_rate(self) -> float:
        return self.stats["autoplay"] / max(1, self.stats["decisions"])


OVERRIDE_PHASES = ("round", "boss", "shop", "pack", "blind")


def phase_of(g) -> str:
    if g.state == "SELECTING_HAND":
        return "boss" if g.blind_idx == 2 else "round"
    return {"SHOP": "shop", "PACK": "pack"}.get(g.state, "blind")


def override_summary(stats) -> dict:
    """Per phase and overall ("all"): decisions the network saw (n), % of them searched, and the override
    rates in % of n (see the module doc). Greedy play gives the meaningful numbers: with exploration the
    final choice also differs because of the Gumbel noise."""
    out = {}
    tot = Counter()
    for ph in OVERRIDE_PHASES + ("all",):
        if ph != "all":
            c = {k: stats.get(f"ovr_{ph}_{k}", 0) for k in ("n", "searched", "final", "strong", "net", "search")}
            tot.update(c)
        else:
            c = tot
        n = c["n"]
        if n:
            out[ph] = {"n": int(n), **{f"{k}%": round(100.0 * c[k] / n, 2)
                                       for k in ("searched", "final", "strong", "net", "search")}}
    return out


def _is_spectral(g, a: Action) -> bool:
    if a.kind == "use" and 0 <= a.idx < len(g.consumables):
        return g.consumables[a.idx].kind == "spectral"
    if a.kind == "pick" and g.state == "PACK" and 0 <= a.idx < len(g.pack_cards):
        return getattr(g.pack_cards[a.idx], "kind", "") == "spectral"
    return False
