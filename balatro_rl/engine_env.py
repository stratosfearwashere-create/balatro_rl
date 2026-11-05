"""Environments on the C++ engine (sim/_engine): the same observations, actions and rewards as env.BalatroEnv
(tests/test_engine.py compares them value for value), many times faster.

    env = CppEnv("RED", "WHITE"); obs = env.reset(seed); obs, r, done, info = env.step(a)     # like BalatroEnv
    venv = CppVecEnv(256, "RED", "WHITE", seed0=7, threads=8)                                  # like VecEnv
    obs = venv.obs            # dict of [n, ...] arrays: live views, overwritten by the next step
    rew, done, infos = venv.step(actions)

Not supported (use the Python environment): shape_phi (the build potential is Python code), the strategic
environment and a restricted joker pool. Build with `python setup_cython.py build`.
"""
from __future__ import annotations

import numpy as np

try:
    from .sim import _engine
    AVAILABLE = True
except ImportError:                                    # not built
    _engine = None
    AVAILABLE = False

OBS_KEYS = ("g", "cards", "jid", "jf", "cid", "aid", "af", "mask", "subs", "aid2", "priv")
F_PRIV = 2000         # the privileged observation (Env::encode_priv): zeros unless set_priv(True); [0,200) next shop, boss, deck top; [200,2000) the future list of the next 8 antes


def _check(reward_kw: dict):
    if not AVAILABLE:
        raise RuntimeError("the C++ engine is not built: run `python setup_cython.py build`")
    kw = dict(reward_kw or {})
    if kw.pop("shape_phi", 0.0):
        raise ValueError("shape_phi needs the Python environment (the build potential is Python code)")
    if kw.pop("joker_pool", None):
        raise ValueError("joker_pool needs the Python environment")
    kw.pop("reward_config", None)
    return {"ante_weight": kw.pop("ante_weight", 1.0), "win_bonus": kw.pop("win_bonus", 10.0),
            "win_ante": kw.pop("win_ante", 8), "shape_chips": kw.pop("shape_chips", 0.0),
            "shape_gamma": kw.pop("shape_gamma", 0.995)}


class CppEnv:
    """One environment (BalatroEnv's interface). `game` is the C++ game (dump(), state, money ...)."""

    def __init__(self, deck: str = "RED", stake: str = "GOLD", verdant_bonus: float = 0.0, **reward_kw):
        self.e = _engine.Env(deck, stake, **_check(reward_kw))
        self.e.verdant_bonus = float(verdant_bonus)
        self.obs = None

    @property
    def game(self):
        return self.e.game

    def reset(self, seed: int = 0) -> dict:
        self.obs = self.e.reset(int(seed))
        return self.obs

    def step(self, a: int):
        self.obs, r, done, info = self.e.step(int(a))
        return self.obs, r, done, info


class CppVecEnv:
    """n environments stepped together on `threads` threads (0: one per core), auto-reset like vec_env.VecEnv:
    environment i plays seeds seed0 + i + n * episode. `obs` holds live views of the engine's buffers: copy
    (or move to the GPU) what must outlive the next step."""

    def __init__(self, n: int, deck: str = "RED", stake: str = "GOLD", seed0: int = 0, threads: int = 0,
                 reward_kw: dict | None = None, start_xmult: float = 0.0, start_any: float = 0.0,
                 verdant_bonus: float = 0.0, stakes: list | None = None):
        # start_xmult: share of games that start holding one random multiplier joker (a training aid: the agent
        # meets scaling builds it would rarely assemble by exploring; evaluate without it)
        kw = _check(reward_kw or {})
        self.n = n
        self.v = _engine.VecEnv(n, deck, stake, int(seed0), int(threads), kw["ante_weight"], kw["win_bonus"],
                                kw["win_ante"], kw["shape_chips"], kw["shape_gamma"], float(start_xmult),
                                float(start_any), float(verdant_bonus), list(stakes or []))
        self.stakes = list(stakes) if stakes else [stake] * n          # the stake each environment plays
        self.seed0 = int(seed0)
        self.obs = self.v.obs()

    def set_start(self, xmult: float, any_: float, money_p: float = 0.0, money: int = 0, money2_p: float = 0.0,
                  money2: int = 0, grow: float = 0.0):
        """Head-start shares for the games that begin from now on."""
        if money_p > 0 or money2_p > 0 or grow > 0:       # (an engine built before the options takes two arguments)
            self.v.set_start(float(xmult), float(any_), float(money_p), int(money), float(money2_p), int(money2),
                             float(grow))
        else:
            self.v.set_start(float(xmult), float(any_))

    def set_ban(self, jokers: list, p: float):
        """In share p of the games that begin from now on, these jokers (engine ids: positions in sim.jokers.JOKERS)
        are out of the pool."""
        self.v.set_ban([int(j) for j in jokers], float(p))

    def set_level_cap(self, p: float, n: int = 2, ante: int = 4):
        """In share p of the games that begin from now on, until `ante` at most `n` planet cards may be bought, picked or used."""
        self.v.set_level_cap(float(p), int(n), int(ante))

    def set_xscale(self, share: float, val: float):
        """In share `share` of the games that begin from now on one scaling multiplier joker (Constellation, Hologram,
        Vampire, Lucky Cat, Glass) is held at the start with its counter already at `val`."""
        self.v.set_xscale(float(share), float(val))

    def set_stakes(self, stakes: list):
        """The stake of every environment, from its next game on."""
        self.v.set_stakes([str(x) for x in stakes])
        self.stakes = [str(x) for x in stakes]

    def set_school(self, p: float, students: list):
        """In share p of the games that begin from now on one of these growing jokers (engine joker ids) is given as an
        eternal, with a kit and a shelf item in every shop (the joker school)."""
        self.v.set_school(float(p), [int(x) for x in students])

    def school_student(self):
        """[n] the student joker id of the game each environment is playing, -1 for none."""
        return np.asarray(self.v.school_student(), dtype=np.int64)

    def school_stats(self):
        """[J, 2] finished school games since the last call, by student joker id: [games, sum of log final counter]."""
        return np.array(self.v.school_stats(), copy=True)

    def set_death_penalty(self, pen: float, cap: float = 100.0):
        """A game that ends in a loss costs pen * min(money, cap) / cap."""
        self.v.set_death_penalty(float(pen), float(cap))

    def copy_from(self, other: "CppVecEnv", i: int, j: int):
        """Environment j of this vector := a copy of the game environment i of `other` is playing (its observation made)."""
        self.v.copy_env_from(other.v, int(i), int(j))

    def expand_level(self, idx, dest, ends_per=8, sw=32, sl=7, node_cap=1000, money_w=0.01, pot_w=0.0, scale_w=0.0, target=(), target_w=0.0):
        """One level of the game search for each environment in idx (at a blind select): -> [(win, win_path, [(path, score, sig, slot)])];
        the end states (at the next blind select) are written to environments of `dest` (slot numbers). See ShopSearch in env.hpp."""
        return self.v.expand_level([int(x) for x in idx], dest.v, int(ends_per), int(sw), int(sl), int(node_cap), float(money_w), float(pot_w), float(scale_w), [int(x) for x in target], float(target_w))

    def copy_many(self, other: "CppVecEnv", src: list, dst: list):
        """Environment dst[k] of this vector := a copy of environment src[k] of `other`, in parallel."""
        self.v.copy_many(other.v, [int(x) for x in src], [int(x) for x in dst])

    def solve_rounds(self, idx: list, max_plans: int = 3, node_cap: int = 400):
        """For each environment in idx that is at the first decision of a blind: up to max_plans action lists that clear it
        (cards to play and discard), best first by the money they leave. The exact solver: the deck order and the rolls are known."""
        return self.v.solve_rounds([int(x) for x in idx], int(max_plans), int(node_cap))

    def set_priv(self, on: bool = True):
        """Fill the privileged observation "priv" (exact-seed mode): the next shop, the next ante's boss and the cards on top of the deck.
        For a critic that sees what the policy cannot (train.py --priv); the policy never reads it."""
        self.v.set_priv(bool(on))

    def set_exact(self, on: bool = True):
        """From the next game on, every game uses the real game's named random streams (the seed string made from the seed)."""
        self.v.set_exact(bool(on))

    def set_marginal(self, level: int = 1):
        """The marginal-value inputs (marginal.py) in every observation from now on (outside rounds): 1 the marginal
        values, 2 also the growth potential of growing jokers; 0 off."""
        self.v.set_marginal(int(level))

    def set_weight_shape(self, ante_weight: float, power: float = 1.0):
        """The blind rewards: 1 at ante 1 to ante_weight at ante 8 along (ante / 7) ** power, summing to 24."""
        self.v.set_weight_shape(float(ante_weight), float(power))

    def set_reward(self, blind_scale: float = 1.0, ante8_bonus: float = 0.0):
        """The reward for each blind beaten times blind_scale, and a bonus for reaching ante 8."""
        self.v.set_reward(float(blind_scale), float(ante8_bonus))

    def set_tilt(self, p: float, levels: int = 2):
        """In share p of the games that begin from now on, one random hand type starts `levels` levels up."""
        self.v.set_tilt(float(p), int(levels))

    def restricted(self):
        """[n] bool: the game each environment is playing has the ban."""
        return self.v.restricted()

    def step(self, actions):
        """-> (rewards [n] float32, dones [n] bool, infos). infos: ante / won / blinds arrays describing the
        step just taken (for a finished environment: the outcome of the episode that ended)."""
        rew, done, ante, won, blinds = self.v.step(np.ascontiguousarray(actions, dtype=np.int64))      # (an action < 0: leave it)
        return rew, done, {"ante": ante, "won": won, "blinds": blinds}

    def set_start_ante(self, ante: int):
        """Every game from its next one begins in the shop of this ante, the blinds before it counted as beaten (a late-game sandbox; with set_start for
        the money and set_level_cap for a limit on planet use)."""
        self.v.set_start_ante(int(ante))

    def set_scenario(self, p: float, w_one: float = 1.0, w_few: float = 0.0, w_full: float = 0.0, min_ante: int = 1, max_ante: int = 7,
                     reroll_cap: int = 3, couple: float = 0.5):
        """From their next game, a share p of the games are scenarios (engine Env::apply_scenario): a sampled build (2-5 jokers with counters, a themed
        deck, hand levels, money) at a sampled ante, of kind 0 (one blind), 1 (3-6 blinds with shops, rerolls capped) or 2 (the rest of the run), in the
        proportions w_one : w_few : w_full. A cut-off scenario is a truncation. The other games are played from the start as ever."""
        self.v.set_scenario(float(p), float(w_one), float(w_few), float(w_full), int(min_ante), int(max_ante), int(reroll_cap), float(couple))

    def set_auto_target(self, on: bool = True):
        """A consumable that needs target cards (Hanged Man, Death, Strength, the suit changers, the enhancers, ...) is applied at once on the engine's
        automatic targets (Game::auto_targets: the cards least worth keeping are destroyed, the best plain cards enhanced, ...) instead of a targeting step:
        the agent only decides whether to use it, in a round or when picking it from a pack."""
        self.v.set_auto_target(bool(on))

    def set_caps(self, reroll_cap: int = 0, hand_level_cap: int = 0, flash_cap: int = 0):
        """Rerolls allowed per shop visit (0: the default 8), the highest level a planet / Black Hole takes a hand to (0: no limit), and the rerolls of a
        game that grow Flash Card (the first flash_cap; 0: all)."""
        self.v.set_caps(int(reroll_cap), int(hand_level_cap), int(flash_cap))

    def set_weight_geom(self, ratio: float):
        """The blind rewards rise by this factor every ante (1, r, r^2 ...), scaled to add up to 24."""
        self.v.set_weight_geom(float(ratio))

    def set_archive(self, p_start: float, p_store: float, lo_ante: int = 4, hi_ante: int = 7, cap: int = 64, min_fill: int = 8, winners: bool = False):
        """The state archive: the start of a blind of antes lo..hi is kept with chance p_store (cap per stake and blind, a reservoir); a share p_start of the
        games from now on start from one (scn_kinds() reads 3 for them). A blind with fewer than min_fill snapshots is not used."""
        self.v.set_archive(float(p_start), float(p_store), int(lo_ante), int(hi_ante), int(cap), int(min_fill), bool(winners))

    def share_archive(self, other: "CppVecEnv"):
        """This vector environment's games keep to and start from `other`'s archive (a second one that plays only to fill it)."""
        self.v.share_archive(other.v)

    def archive_outcomes(self, reset: bool = True):
        """-> [2, 24]: the archive games that ended and that were won, by the blind they started from (since the last reset)."""
        return np.asarray(self.v.archive_outcomes(bool(reset)), dtype=np.int64).reshape(2, 24)

    def archive_starts(self, reset: bool = True):
        """-> the games started from each of the 24 blinds since the last call (read with reset=True)."""
        return np.asarray(self.v.archive_starts(bool(reset)), dtype=np.int64)

    def archive_counts(self, stake: int = -1):
        """-> the snapshots held for each of the 24 blinds (a stake index, or -1 for all)."""
        return np.asarray(self.v.archive_counts(int(stake)), dtype=np.int64)

    def set_immortal(self, on: bool = True, pay: bool = False):
        """From the next game: a blind that is not beaten does not end the run (nothing is paid, the next blind and its shop follow); only failing the
        ante-8 boss does. Rewards are the blinds actually beaten and the win; the 'blinds' of step() is the number beaten."""
        self.v.set_immortal(bool(on), bool(pay))

    def set_blind_mult(self, mult):
        """Immortal mode: the reward of blind b is multiplied by mult[b] (24 values; 1 = as is). The trainer's --immortal-fade moves them with the clear rates."""
        self.v.set_blind_mult([float(x) for x in mult])

    def set_autoplay(self, on: bool = True, nodes: int = 400, force: bool = False):
        """At a blind select the action A_LEAVE becomes legal and means: select the blind and let the exact round solver play it (the plan that clears it with the most money
        left); the ordinary select plays by hand. The solver sees the draw order and the rolls: a training aid, not the real game."""
        self.v.set_autoplay(bool(on), int(nodes), bool(force))           # force: the ordinary select too hands the round to the solver

    def last_bits(self):
        """-> for each environment, the bit mask (blind b = bit b) of the blinds beaten in the last game that ended there (read right after a step with done)."""
        return np.asarray(self.v.last_bits(), dtype=np.uint32)

    def set_power(self, lo: float = 1.0, hi: float = 1.0):
        """Scenarios: each build's power is log-uniform in [lo, hi]: it multiplies the hand level-ups and the rounds the counters have grown, and above 1.5 the build is
        made mostly of scoring jokers with more editions (builds near the pass / fail line of the late antes)."""
        self.v.set_power(float(lo), float(hi))

    def set_build_seeds(self, seeds, force_blind: int = -1, level_conc: float = 0.0):
        """Scenarios (set_scenario): env i draws its build from seeds[i] (0: from the game's own seed), so the same build meets different draws; force_blind
        0 / 1 / 2 makes a one-blind scenario face the small blind / big blind / boss; level_conc: share of builds whose hand levels pile up on one or two types."""
        self.v.set_build_seeds([int(x) for x in seeds], int(force_blind), float(level_conc))

    def set_state(self, ids, ante: int = 0, blind: int = 0, counters=None):
        """Hypothetical states of the games `ids` at a blind select: set the ante and blind (ante 0: leave them) and each joker slot's counter (counters [len(ids), 8], NaN: leave it);
        the observation, with the exact expected-score columns, is remade. For asking the judge about a build as it would be at a later ante."""
        ids = [int(i) for i in ids]
        c = np.full((len(ids), 8), np.nan) if counters is None else np.asarray(counters, dtype=np.float64).reshape(len(ids), 8)
        self.v.set_state(ids, int(ante), int(blind), c.reshape(-1).tolist())

    def set_marginal_blind_only(self, on: bool = True):
        """With set_marginal: the marginal-value columns only in blind-select observations (where the judge reads them), not at every shop step."""
        self.v.set_marginal_blind_only(bool(on))

    def reseed_rng(self, idx, salts):
        """(Not exact mode) new draws for the games `idx`: each game's own generator restarts from its salt; the state of the game stays as it is."""
        self.v.reseed_rng([int(i) for i in idx], [int(x) for x in salts])

    def set_build(self, ids, ante: int, blind: int, keys, counters=None, steel_cards: int = 0, levels=None, money: int = -1, editions=None, suit: int = -1, suit_n: int = 0):
        """A given build in the games `ids` at a blind select: joker engine ids `keys` with counters (NaN: its own), the first steel_cards deck cards Steel, hand levels, money."""
        c = [float("nan")] * len(keys) if counters is None else [float(x) for x in counters]
        self.v.set_build([int(i) for i in ids], int(ante), int(blind), [int(k) for k in keys], c, int(steel_cards), [int(x) for x in (levels or [])], int(money),
                         [int(e) for e in (editions or [])], int(suit), int(suit_n))

    def solve_best(self, idx, node_cap: int = 1000):
        """For each game at the start of a blind: [best share of the target any explored line reaches (>= 1: cleared), 1 if a clearing line was found, nodes used]."""
        return np.asarray(self.v.solve_best([int(i) for i in idx], int(node_cap)), dtype=np.float64)

    def set_round_resources(self, idx, hands: int, discards: int):
        """The blind each game in idx is at start with `hands` hands and `discards` discards left (small cases for checking the round solver against brute force)."""
        self.v.set_round_resources([int(i) for i in idx], int(hands), int(discards))

    def set_solver(self, n_plays: int = 4, n_discards: int = 3, first_plays: int = 0, first_discards: int = 0):
        """The exact round solver (solve_rounds, expand_level, autoplay) tries this many plays (best by predicted score) and this many discards at each step; at the first step of a round
        first_plays / first_discards (when larger)."""
        self.v.set_solver(int(n_plays), int(n_discards), int(first_plays), int(first_discards))

    def set_level_scripted(self, mode: int = -1):
        """The beam search's levels (expand_level) play each blind with the scripted player of this mode (0 score most, 1 scaler-friendly, 2 margin-blended) instead of the exact round
        solver: a shop-only search. In a returned path the blind is the single entry -3 (select with the scripted player). -1: the exact solver."""
        self.v.set_level_scripted(int(mode))

    def set_search_prune(self, bits: int = 3):
        """The beam's shop search (expand_level): bit 1 = a swap or a sell must raise the ranking score, else that line is pruned; bit 2 = of two states with the
        same future (state_key), only the one with more money is kept. A global switch (every environment, every search). 0: off."""
        self.v.set_search_prune(int(bits))

    def set_solver_hand_bound(self, on: bool = True):
        """The exact round solver: discards tried before plays, and once a clearing line is found a branch with no more hands left than it is not searched
        (it cannot end richer: each hand left pays $1). A global switch."""
        self.v.set_solver_hand_bound(bool(on))

    def set_solver_extra(self, n: int = 50):
        """The exact round solver (every search that uses it): stop n nodes after the first line that clears the blind is found, instead of searching on to the
        node cap for richer lines. -1: off (the default). A global switch."""
        self.v.set_solver_extra(int(n))

    def set_pot_mode(self, mode: int = 1, rounds: int = 6):
        """The growth potential (marginal 2, the beam's scaling-aware score): 0 = the old table (marginal.POTENTIAL, 3 rounds of optimistic rates);
        1 = the growth model's measured per-round rates over `rounds` rounds, with Madness, Ceremonial Dagger and Canio added. A global switch."""
        self.v.set_pot_mode(int(mode), int(rounds))

    def set_order_mode(self, mode: int = 1):
        """The jokers' order (a global switch): 0 = swaps when a joker acts on a neighbour (Blueprint, Brainstorm, Ceremonial Dagger), else +mult before xMult;
        1 = swaps only with Blueprint, Brainstorm or Ceremonial Dagger, every other build kept in chips / neutral, +mult, xMult order."""
        self.v.set_order_mode(int(mode))

    def load_nnue(self, path: str):
        """Loads an evaluation network exported by tools/nnue_export.py into the engine (one global network), with balatro_rl/nnue.py's layout."""
        from .nnue import engine_layout
        cols, off, consts = engine_layout()
        self.v.load_nnue(str(path), cols, off, consts)

    def set_round_eval(self, w: float = 10.0):
        """The round solver scores a clearing line by its money + w x the blinds the loaded network expects from the state it leaves (0: money only)."""
        self.v.set_round_eval(float(w))

    def nnue_eval(self, idx, as_next_select: bool = False):
        """The engine's network on environments idx as they stand (expected blinds)."""
        return np.asarray(self.v.nnue_eval([int(i) for i in idx], bool(as_next_select)), dtype=np.float64)

    def set_search_pack_macro(self, on: bool = True):
        """The beam's shop search opens a bought pack at once: each pick (and the skip; a mega pack's two picks) is a child of the buy, so a pack
        is judged by what it gives. Use with set_auto_target(True) (a tarot's targets chosen at once). A global switch."""
        self.v.set_search_pack_macro(bool(on))

    def set_search_reroll_cap(self, n: int = 3):
        """The beam's shop search tries at most n rerolls per shop visit (0: no cap beyond the game's). A global switch."""
        self.v.set_search_reroll_cap(int(n))

    def set_solver_speed(self, lite: bool = True, cache: bool = True, dead_f: float = 0.0):
        """The exact round solver's speed-ups (global switches, all off by default): lite = its nodes make only the candidate plays after a step, not the
        whole observation (the same search, cheaper nodes); cache = the beam's levels (expand_level) share solved rounds between beam states with the same
        round_key, each cached line replayed on the real state first; dead_f > 0 = a branch is dropped when the chips still needed exceed hands left x the
        best play now x dead_f."""
        self.v.set_solver_speed(bool(lite), bool(cache), float(dead_f))

    def solver_cache_stats(self, reset: bool = True):
        """-> (hits, misses, fallbacks) of the round cache since the last reset (a fallback: a cached line did not clear the new state)."""
        return tuple(self.v.solver_cache_stats(bool(reset)))

    def cell_info(self, idx):
        """Per game in idx the integers a cell key is made from (see VecEnv::cell_info in env.hpp)."""
        return self.v.cell_info([int(i) for i in idx])

    def hand_levels(self, idx):
        """The hand levels of each game in idx: [n][N_HANDS]."""
        return self.v.hand_levels([int(i) for i in idx])

    def state_keys(self, idx):
        """-> (keys [n] uint64, money [n]): for each game in idx (an exact-seed game at a blind select) a hash of everything that decides its future except the money, and the money. The same key and
        money: the same state. The same key and more money: dominates. The RNG stream positions are part of the key."""
        r = self.v.state_keys([int(i) for i in idx])
        return np.array([a for a, _ in r], dtype=np.uint64), np.array([b for _, b in r], dtype=np.int64)

    def set_round_feed(self, w: float = 0.0):
        """The exact round solver (solve_rounds, expand_level) prefers, among lines that clear a blind, the one with more money plus w dollars for every natural round of
        growth it gives the growing jokers the build holds (their counters' growth divided by their per-round rate); it also tries 8 plays at each step instead of 4. 0: money only."""
        self.v.set_round_feed(float(w))

    def set_modes(self, modes):
        """Each environment's scripted-player mode (set_auto_round) for its next round."""
        self.v.set_modes([int(x) for x in modes])

    def set_auto_round(self, on: bool = True, pace: float = 1.5, mode: int = 0):
        """From the next step, the select action plays the whole round with the scripted player (it sees only its hand: play the best predicted play if it clears the
        blind / is the last hand / is on pace (best x hands left >= pace x needed), else discard the cards outside it). The manager then only ever decides outside rounds."""
        self.v.set_auto_round(bool(on), float(pace), int(mode))

    def margin_report(self):
        """-> [n, 2]: the safety margin (P10 of the redrawn round scores / what the blind still needs) and the investment weight lambda at the first decision of each env's last
        scripted round (mode 2 only; 0 otherwise)."""
        return np.asarray(self.v.margin_report(), dtype=np.float64).reshape(self.n, 2)

    def set_margin_params(self, m_full: float = 1.6, samples: int = 24, force_lambda: float = -1.0):
        """Mode 2: lambda reaches 1 when the margin reaches m_full (+0.05 per ante above 1, +0.2 on a boss); samples = redrawn hands per round; force_lambda >= 0 fixes lambda."""
        self.v.set_margin_params(float(m_full), int(samples), float(force_lambda))

    def round_report(self):
        """-> the last scripted round of each environment: dict of arrays: cleared, hands, discards, money, ratio (best share of the target reached), c0 / c1 [n, 8]
        (each joker slot's counter before / after the round)."""
        a = np.asarray(self.v.round_report(), dtype=np.float64).reshape(self.n, 21)
        return {"cleared": a[:, 0] > 0.5, "hands": a[:, 1], "discards": a[:, 2], "money": a[:, 3], "ratio": a[:, 4], "c0": a[:, 5:13], "c1": a[:, 13:21]}

    def set_verdant_relative(self, on: bool = True):
        """The Verdant Leaf bonus times the weight of the blind being faced (train.py --shape-relative)."""
        self.v.set_verdant_relative(bool(on))

    def set_scaled(self, start_n: int = 0, shop: bool = False, rounds: int = 30):
        """The late-game sandbox's pre-scaled jokers: up to start_n growing jokers held from the start with counters grown `rounds` rounds (no ceiling), and
        (shop) every growing joker made for a shop or a pack arriving with its counter grown the same."""
        self.v.set_scaled(int(start_n), bool(shop), int(rounds))

    def play_scores(self):
        """-> the chips the play of the step just taken scored in each environment (0: the step was no play); a finished game's last step keeps its value."""
        return np.asarray(self.v.play_scores(), dtype=np.float32)

    def scn_kinds(self):
        """-> the kind of the game each environment is playing now: -1 a real game, 0/1/2 a scenario (read it BEFORE a step to label the game that ends in it)."""
        return np.asarray(self.v.scn_kinds(), dtype=np.int64)

    def set_shop_only(self, on: bool = True, nodes: int = 1000, partial_w: float = 0.5):
        """The shop-only game: every blind before the final one is won without play (paid as won with one hand left; jokers that grow by playing advance one
        round of growth); the final boss is played by the exact round solver (sees the draws), and no clearing line loses the run."""
        self.v.set_shop_only(bool(on), int(nodes), float(partial_w))

    def fast_power(self, idx, samples: int = 8):
        """Games at the start of a blind: a build's power without searching the round. -> [n, 4]: hands left x mean best play / target, hands left x max best play / target,
        mean best play, max best play (over `samples` hands redrawn from the hand + draw pile; every candidate play scored by the exact scorer)."""
        return np.asarray(self.v.fast_power([int(i) for i in idx], int(samples)), dtype=np.float64)

    def projected_power(self, idx, samples: int = 4):
        """Games outside a round: the build as it will be at the ante-8 boss (scaling counters moved by their natural rates over the rounds left), its fast_power share
        of that boss's target. >= 1: the build would clear it."""
        return np.asarray(self.v.projected_power([int(i) for i in idx], int(samples)), dtype=np.float64)

    def final_boss_clear(self, idx, samples: int = 2, node_cap: int = 200):
        """Games outside a round: the ante-8 oracle. The build with its scaling counters projected to the ante-8 boss, the round played by the exact round
        solver on `samples` redrawn decks (four final bosses in turn) -> [n, 2]: the share of samples that clear it, the mean min(1, best share of the target)."""
        return np.asarray(self.v.final_boss_clear([int(i) for i in idx], int(samples), int(node_cap)), dtype=np.float64).reshape(len(idx), 2)

    def set_fixed_seed(self, seed: int = -1):
        """seed >= 0: every game from the next one on (call v.restart() to apply now) is this seed; -1: back to seed0 + i + n x episode."""
        self.v.set_fixed_seed(int(seed))

    def set_rules(self, no_stickers: bool = False, small_blind_pay: bool = False):
        """From each game's next start: no eternal / perishable / rental jokers (no_stickers); the small blind pays its $3 at every stake (small_blind_pay). Call v.restart() to apply now."""
        self.v.set_rules(bool(no_stickers), bool(small_blind_pay))

    def set_menu(self, n: int = 0):
        """The shop shows n jokers (n > 2, up to MAX_SHOP) instead of its usual two cards, and only jokers: a curriculum with far more to choose from. 0: the usual shop. Applies from the next
        shop of every running game."""
        self.v.set_menu(int(n))

    def set_part_credit(self, on: bool = True):
        """False: a lost blind earns nothing for the chips scored in it. The blind rewards are then exactly the potential W(furthest blind beaten) (plus the win bonus at a win): the
        return of a game is a function of where it ended, whatever route led there."""
        self.v.set_part_credit(bool(on))

    def set_log_partial(self, on: bool = True):
        """The part credit for a lost blind on a log scale of the chips reached (a dense reward for scoring more, whatever the route)."""
        self.v.set_log_partial(bool(on))

    def set_sandbox(self, level_cap_mode: int = 0, pack_restock: int = 0, bias=(1, 1, 1, 1, 1)):
        """The late-game sandbox's rules: level_cap_mode 1 lets planets be bought and used without limit but raises no hand level past --level-cap-n level-ups
        (Constellation still grows); pack_restock: a pack bought is replaced by a new one this many times per shop visit; bias: weights of the pack kinds
        (arcana, celestial, standard, buffoon, spectral) are multiplied by these."""
        self.v.set_sandbox(int(level_cap_mode), int(pack_restock), [float(x) for x in bias])

    def set_sparse(self, from_blind: int):
        """The reward's sparse stream: the win bonus, the ante-8 bonus and the rewards of the blinds from this index on (3 a ante; 24: none)."""
        self.v.set_sparse(int(from_blind))

    def step_split(self, actions):
        """Like step, with the reward split in two: -> (rewards, dones, infos, sparse part of the rewards, truncated). `truncated` marks an
        episode the step limit ended (a loss or a win does not): its last state still has a value to bootstrap from."""
        rew, done, ante, won, blinds, sparse, trunc = self.v.step_split(np.ascontiguousarray(actions, dtype=np.int64))
        return rew, done, {"ante": ante, "won": won, "blinds": blinds}, sparse, trunc

    def close(self):
        self.v = None
        self.obs = None
