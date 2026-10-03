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

import math
import random
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch

from ..rewards.config import PotentialConfig
from ..rewards.potential import Potential
from ..rewards.targets import z_target
from ..sim.game import MAX_JOKERS
from .actions import Choice, Config, enumerate_candidates
from .features import F_CAND, N_PRICE, N_REF, encode_cands, encode_state
from .net import AZNet, collate
from .search import GumbelSearch, Node, _softmax
from .shop import RANDOM_TAGS, RANDOM_USES, ShopConfig, ShopPricer
from .solver import RoundSolver
from .world import Action, World


@dataclass
class AgentConfig:
    """Stage 3 options (all defaults leave the agent as it was):
      shop_prior     "rule": the rule-based player's choice gets +heur_bonus (today). "graded": every option
                     outside a round gets a price (az/shop.py) and the prior is the prices scaled to logits.
      shop           the constants and rule switches of the graded prior (shop.ShopConfig; a dict is accepted)
      price_feature  an untrained network is built with the price inputs (features.N_PRICE per candidate); a
                     loaded network decides for itself. Changes the network, so never on by default.
      shop_eval      how decisions outside a round are improved when `search` is on:
                       "search"   the Gumbel search (today)
                       "onestep"  each option is applied to a redrawn copy and the network's V is read there
                                  (onestep_samples copies for options with a random outcome: packs, rerolls,
                                  blind select, random consumables); score = logits + onestep_weight x V; no
                                  search into later shops. Only the onestep_top options by logit are evaluated.
                       "prior"    nothing: the network's logits (close calls can still apply)
      close_calls    with shop_eval "onestep" or "prior": when the two best options are within close_margin
                     logits, each is played out close_rollouts times by the agent's own no-search policy on
                     the same sampled futures (common random numbers from the agent's generator) to the end
                     of the next boss blind ("boss") or of the next blind ("blind"), at most close_max_steps
                     decisions; outcome = V at the end ("value") or blinds cleared / 24 ("blinds"); the pair's
                     scores move by close_weight x (outcome - the pair's mean outcome).
    The improved choice is recorded like a search's: Decision.policy = softmax(scores), searched=True."""
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
    solver_samples_inner: int = 2   # inside the search (measured: SPEED_NOTES.md, stage 3)
    boss_depth: int = 2             # solver depth on boss blinds (root)
    autoplay: bool = True
    search: bool = True
    root_cfg: Config = field(default_factory=lambda: Config(max_analyzed=700))
    inner_cfg: Config = field(default_factory=lambda: Config(max_plays=24, max_discards=12, max_analyzed=160,
                                                             max_targets=8, use_samples=1))
    round_growth: bool = False      # in rounds, options the solver rates within growth_margin logits of its
    growth_margin: float = 1.0      #   best get growth_scale x (change in build strength they cause: joker
    growth_scale: float = 100.0     #   state, money), capped at +-growth_cap logits (see growth_bonus)
    growth_cap: float = 2.0
    shop_prior: str = "rule"
    shop: ShopConfig = field(default_factory=ShopConfig)
    price_feature: bool = False
    shop_eval: str = "search"
    onestep_weight: float = 300.0
    onestep_samples: int = 2
    onestep_top: int = 24
    close_calls: bool = False
    close_margin: float = 1.0
    close_rollouts: int = 4
    close_horizon: str = "boss"
    close_outcome: str = "value"
    close_weight: float = 300.0
    close_max_steps: int = 150
    value_model: str = ""           # build-value checkpoint (value/model.py): replaces the network's V in
    value_mode: str = "win"         #   one-step evaluation and close-call rollouts; win | antes | mix

    def __post_init__(self):
        if isinstance(self.shop, dict):
            self.shop = ShopConfig(**self.shop)
        if self.shop_prior not in ("rule", "graded"):
            raise ValueError(f"unknown shop_prior {self.shop_prior!r}")
        if self.shop_eval not in ("search", "onestep", "prior"):
            raise ValueError(f"unknown shop_eval {self.shop_eval!r}")
        if self.close_horizon not in ("boss", "blind") or self.close_outcome not in ("value", "blinds"):
            raise ValueError("close_horizon is 'boss' or 'blind'; close_outcome is 'value' or 'blinds'")
        if self.value_mode not in ("win", "antes", "mix"):
            raise ValueError(f"unknown value_mode {self.value_mode!r}")


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
        self.pricer = ShopPricer(self.cfg.shop, self.potential)
        self.price_feats = getattr(self.net, "c_price", None) is not None
        self._last_steps = -1
        self.build_value = None
        if self.cfg.value_model:
            from ..value.model import BuildValue
            self.build_value = BuildValue(self.cfg.value_model, self.potential, device, self.cfg.value_mode)

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
        price = None
        if choice.phase == "SELECTING_HAND":
            depth = cfg.boss_depth if (root and w.g.blind_idx == 2) else 1
            samples = cfg.solver_samples if root else cfg.solver_samples_inner
            if depth > 1:
                samples = max(4, samples // 2)
            self.solver.solve(w.g, choice, rng, samples=samples, depth=depth)
            prior = solver_prior(choice, cfg.tau)
            if cfg.round_growth:
                prior = prior + self.growth_bonus(w, choice, prior)
        elif cfg.shop_prior == "graded":
            prior, price = self.pricer.prior(w, choice, rng, root)
        else:
            prior = rule_prior(w, choice, cfg.heur_bonus)
        state = encode_state(w)
        state["phi"] = np.float32(self.potential(w.g))
        state["prog"] = np.float32(self.potential.progress(w.g))
        enc = (state, encode_cands(w, choice, prior, price, self.price_feats))
        with torch.no_grad():
            s, c, m = collate([enc], self.device)
            logits, _, out = self.net(s, c, m)
        heads = self.net.split_heads(out[0], s["phi"][0], s["prog"][0], cfg.lam)
        heads = {k: float(v) for k, v in heads.items() if v.dim() == 0}
        heads["phi"] = float(state["phi"])
        return Node(w, choice, enc, logits[0].float().cpu().numpy().astype(float), heads["value"], heads=heads)

    def growth_bonus(self, w: World, choice: Choice, prior: np.ndarray) -> np.ndarray:
        """The solver only asks "does this clear the blind". Where several plays or discards are about
        equally good by that measure (within growth_margin logits of its best), this adds what each does to
        the build beyond the round: the change in build strength (shop.ShopPricer: the same measure that
        prices shop options) from the joker state it changes -- Green Joker, Ride the Bus, Runner, Square
        Joker, Wee Joker, Castle, Hit the Road, Ramen, Ice Cream ... -- and the money it brings. So with a
        blind that is safe either way, the hand that grows the build is preferred, and a discard that costs
        Green Joker its mult has to be worth it. Options outside the margin keep the solver's logit."""
        cfg = self.cfg
        bonus = np.zeros(len(choice.cands))
        near = [i for i, c in enumerate(choice.cands)
                if c.kind in ("play", "discard") and c.analyzed and prior[i] >= -cfg.growth_margin]
        if len({(choice.cands[i].kind == "play", choice.cands[i].jdiff, choice.cands[i].effects) for i in near}) < 2:
            return bonus                                    # nothing to choose between
        g = w.g
        ctx = self.pricer.context(g)
        cache = {}
        for i in near:
            c = choice.cands[i]
            jd = c.jdiff + (choice.after_jdiff if c.kind == "play" else ())
            money = sum(v for k, v in c.effects if k == "money")
            key = (jd, round(money, 3))
            d = cache.get(key)
            if d is None:
                g2 = g.clone()
                gone = []
                for slot, k, _, new in jd:
                    if not 0 <= slot < len(g2.jokers):
                        continue
                    if k == "_removed":
                        gone.append(g2.jokers[slot])
                    else:
                        g2.jokers[slot].state[k] = new
                for j in gone:
                    g2.destroy_joker(j)
                g2.money = g.money + int(round(money))
                a, b = self.pricer._delta(ctx, g2)
                d = cache[key] = a + b
            bonus[i] = d
        if near:
            bonus[near] -= max(bonus[i] for i in near)      # relative to the best: the top logit stays 0
            self.stats["growth_decisions"] += 1
            before = max(near, key=lambda i: prior[i])
            after = max(near, key=lambda i: prior[i] + np.clip(bonus[i] * cfg.growth_scale, -cfg.growth_cap, 0.0))
            self.stats["growth_changed"] += int(after != before)
        return np.clip(bonus * cfg.growth_scale, -cfg.growth_cap, 0.0)

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
        if w.steps == 0 or w.steps < self._last_steps:      # a new game: the pricer forgets the shops it saw
            self.pricer.reset()
        self._last_steps = w.steps
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
        cfg = self.cfg
        if (cfg.search and cfg.shop_eval != "search" and w.g.state != "SELECTING_HAND"
                and len(root.choice.cands) > 1):
            return self._decide_build(w, root, rng, explore)
        budget, reason = self.budget(root)
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
        st[f"ovr_{ph}_strong"] += int(prior[idx] < top - 0.99)
        st[f"ovr_{ph}_net"] += int(prior[net_idx] < top)
        st[f"ovr_{ph}_search"] += int(idx != net_idx)

    def _expand_inner(self, rng):
        return lambda w, root: self.evaluate(w, root, rng)

    # ------------------------------------------------------------------ Stage 3: one-step V and close calls
    def values(self, worlds: list) -> list[float]:
        """The network's V of each world (the game's value for finished ones), in batches; no candidates
        are enumerated (V only reads the state)."""
        out = [0.0] * len(worlds)
        live = []
        for i, w in enumerate(worlds):
            if w.done:
                out[i] = self.terminal(w).value
            else:
                live.append(i)
        if self.build_value is not None:            # the learned build value (value/model.py)
            for i, v in zip(live, self.build_value.value([worlds[i] for i in live])):
                out[i] = float(v)
            return out
        dummy = {"c_kind": np.zeros(1, np.int64), "c_f": np.zeros((1, F_CAND), np.float32),
                 "c_ref": np.full((1, N_REF), -1, np.int64), "c_jd": np.zeros((1, MAX_JOKERS), np.float32),
                 "c_prior": np.zeros(1, np.float32)}
        if self.price_feats:
            dummy["c_price"] = np.zeros((1, N_PRICE), np.float32)
        for lo in range(0, len(live), 64):
            part = live[lo:lo + 64]
            encs = []
            for i in part:
                state = encode_state(worlds[i])
                state["phi"] = np.float32(self.potential(worlds[i].g))
                state["prog"] = np.float32(self.potential.progress(worlds[i].g))
                encs.append((state, dummy))
            with torch.no_grad():
                s, c, m = collate(encs, self.device)
                _, _, o = self.net(s, c, m)
                v = self.net.value(o, s["phi"], s["prog"], self.cfg.lam).float().cpu().numpy()
            for i, x in zip(part, v):
                out[i] = float(x)
        return out

    def _decide_build(self, w: World, root: Node, rng: random.Random, explore: bool) -> Decision:
        """A decision outside a round by one-step evaluation and / or paired rollouts (AgentConfig doc)."""
        cfg = self.cfg
        cands = root.choice.cands
        k = len(cands)
        score = np.array(root.logits, dtype=float)
        sims, reason = 0, "prior"
        if cfg.shop_eval == "onestep":
            worlds, owner = [], []
            for i in np.argsort(-score, kind="stable")[:cfg.onestep_top]:
                n = cfg.onestep_samples if random_outcome(w.g, cands[i].action) else 1
                for _ in range(max(1, n)):
                    w2 = w.determinize(rng)
                    try:
                        w2.step(cands[i].action)
                    except (AssertionError, IndexError):
                        continue
                    worlds.append(w2)
                    owner.append(int(i))
            vals = self.values(worlds)
            tot, cnt = np.zeros(k), np.zeros(k)
            for i, v in zip(owner, vals):
                tot[i] += v
                cnt[i] += 1
            seen = cnt > 0
            if seen.any():                              # options not evaluated count as the worst that was
                q = tot / np.maximum(cnt, 1)
                q = np.where(seen, q, q[seen].min())
                score = score + cfg.onestep_weight * q
            sims, reason = len(worlds), "onestep"
        if cfg.close_calls:
            top = [int(i) for i in np.argsort(-score, kind="stable")[:2]]
            self.stats["close_checked"] += 1
            if score[top[0]] - score[top[1]] <= cfg.close_margin:
                m, n = self._paired_rollouts(w, cands, top, rng)
                mean = 0.5 * (m[0] + m[1])
                for j, i in enumerate(top):
                    score[i] += cfg.close_weight * (m[j] - mean)
                sims += n
                reason = "close"
                self.stats["close_calls"] += 1
                self.stats["close_flipped"] += int(score[top[1]] > score[top[0]])
        g = np.array([-math.log(-math.log(max(rng.random(), 1e-12))) for _ in range(k)]) if explore else np.zeros(k)
        idx = int(np.argmax(g + score))
        self.stats[f"budget_{reason}"] += 1
        self.stats["sims"] += sims
        self._count_override(w, root, idx, sims > 0)
        return Decision(cands[idx].action, idx, root.choice, _softmax(score), sims > 0, enc=root.enc,
                        value=root.value, heads=root.heads, sims=sims, reason=reason)

    def _paired_rollouts(self, w: World, cands: list, pair: list, rng: random.Random) -> tuple[list, int]:
        """Mean outcome of each of the two options over close_rollouts futures; both options get the same
        redrawn copy and the same random generator for each future. Also returns the decisions played."""
        cfg = self.cfg
        tot = [0.0, 0.0]
        steps = 0
        n = max(1, cfg.close_rollouts)
        for _ in range(n):
            seed = rng.getrandbits(63)
            for j, i in enumerate(pair):
                r = random.Random(seed)
                w2 = w.determinize(r)
                try:
                    w2.step(cands[i].action)
                except (AssertionError, IndexError):
                    continue
                v, used = self._rollout(w2, r)
                tot[j] += v
                steps += used
        return [t / n for t in tot], steps

    def _rollout(self, w: World, rng: random.Random) -> tuple[float, int]:
        """Play `w` (a redrawn copy) on with the no-search policy to the horizon; (outcome, decisions)."""
        cfg = self.cfg
        steps = 0
        while not w.done and steps < cfg.close_max_steps:
            g = w.g
            beaten, boss = g.blinds_beaten, g.blind_idx == 2
            a = self.policy_action(w, rng)
            if a is None:
                g.state = "GAME_OVER"
                break
            w.step(a)
            steps += 1
            if g.blinds_beaten > beaten and (boss or cfg.close_horizon == "blind"):
                break
        if cfg.close_outcome == "blinds":
            return w.g.furthest_blind / 24.0, steps
        return self.values([w])[0], steps

    def policy_action(self, w: World, rng: random.Random) -> Action | None:
        """The no-search policy's move: the network's best logit over the (inner) candidates."""
        choice = enumerate_candidates(w, rng, self.cfg.inner_cfg)
        if not choice.cands:
            return None
        if len(choice.cands) == 1:
            return choice.cands[0].action
        node = self.evaluate(w, False, rng, choice)
        if node.terminal:
            return None
        return choice.cands[int(np.argmax(node.logits))].action

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


def random_outcome(g, a: Action) -> bool:
    """Whether carrying out `a` outside a round reveals something random (one-step evaluation averages
    a few copies of these)."""
    k = a.kind
    if k in ("buy_pack", "reroll", "select", "reroll_boss"):
        return True
    if k == "skip":
        return g.blind_idx < len(g.tags_offered) and g.tags_offered[g.blind_idx] in RANDOM_TAGS
    if k == "use" and 0 <= a.idx < len(g.consumables):
        return g.consumables[a.idx].name in RANDOM_USES
    if k == "pick" and g.state == "PACK" and 0 <= a.idx < len(g.pack_cards):
        return getattr(g.pack_cards[a.idx], "name", "") in RANDOM_USES
    if k == "voucher" and 0 <= a.idx < len(g.shop_vouchers):
        return g.shop_vouchers[a.idx].key[2:] in ("overstock_norm", "overstock_plus")
    return False


def _is_spectral(g, a: Action) -> bool:
    if a.kind == "use" and 0 <= a.idx < len(g.consumables):
        return g.consumables[a.idx].kind == "spectral"
    if a.kind == "pick" and g.state == "PACK" and 0 <= a.idx < len(g.pack_cards):
        return getattr(g.pack_cards[a.idx], "kind", "") == "spectral"
    return False
