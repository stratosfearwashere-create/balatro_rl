"""Checks of the strength module (rewards/strength.py) on real games: calibration and cost.

    python -m balatro_rl.rewards.strength_check gen --games 120 --workers 2 --out strength_check.pkl
    python -m balatro_rl.rewards.strength_check report --data strength_check.pkl
    python -m balatro_rl.rewards.strength_check cost

gen: the untrained agent without search plays greedy White Stake games (seeds from 10000). Logged
  - at every decision the network sees: the old Phi and the K sampled best-hand scores (so any strength
    setting can be evaluated afterwards on the same states), and what happened later in that game;
  - at the start of every blind: the same scores, the blind's target, whether the blind was then cleared,
    and the round solver's playout policy (az/solver.py: exact scores, discards, a real shuffled deck) on
    fresh rounds of the same build against that target, this ante's boss target and the next ante's.
report: predicted clear chance against (a) the solver and (b) the real outcome, for several discard
  corrections; then Phi against the game's outcome with the old and the new head term.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time

import numpy as np

from .config import PotentialConfig, StrengthConfig
from .diagnostics import calibration
from .potential import Potential, _card_sig, _seed, fresh_hand_size, fresh_round, scoring_key
from .strength import LAST_ANTE, Strength, boss_targets, clear_chances, current_ante, strength_score, survives_through
from .targets import blind_index

OUTCOMES = ("win", "final_blinds", "blinds_after", "boss_cleared")
BUCKETS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001))


def solver_fresh_round(g, target: float, n: int = 100, seed: int = 0) -> float:
    """Share of `n` fresh rounds of this build (no boss effect) that the round solver's playout policy
    clears against `target`: exact scores, discards and a real deck that runs down."""
    from ..az.solver import RoundSolver, _Round
    from ..sim.cards import sort_hand
    probe = fresh_round(g)
    probe.target = target
    pool = sorted(probe.full_deck, key=lambda c: (_card_sig(c), c.uid))
    probe.hand, probe.deck = [], pool
    s = RoundSolver()
    pool = list(s._setup(probe).deck)                 # the solver's own copies, in the same canonical order
    k = min(fresh_hand_size(g), len(pool))
    rng = random.Random(seed)
    won = 0
    for _ in range(n):
        order = list(pool)
        rng.shuffle(order)
        r = _Round(sort_hand(order[len(order) - k:]), order[:len(order) - k], probe.hands_left,
                   probe.discards_left, 0, frozenset(), -1)
        s._playout(r, rng)
        won += r.chips >= target
    return won / n


# ------------------------------------------------------------------ generation
def _gen_worker(args):
    seeds, deck, stake, n_solver = args
    import torch
    torch.set_num_threads(1)
    from ..az.agent import Agent, AgentConfig
    from ..az.net import AZNet
    from ..az.world import World
    from ..sim.game import Game
    from .growth import snapshot
    torch.manual_seed(0)
    agent = Agent(AZNet().eval(), AgentConfig(search=False), seed=seeds[0])
    pot = agent.potential
    k32, k128 = Strength(StrengthConfig(samples=32)), Strength(StrengthConfig(samples=128))
    memo = {}

    def build(g):
        sk = scoring_key(g)
        got = memo.get(sk)
        if got is None:
            got = memo[sk] = {"scores": k32.best_scores(g, sk), "seed": _seed(("bootstrap", sk)),
                              "hands": g.round_hands(), "discards": g.round_discards()}
        return sk, got

    rows, starts, infos, growth = [], [], [], []
    for seed in seeds:
        w = World(Game(seed=seed, deck_type=deck, stake=stake))
        agent.reseed(seed * 7919 + 17)
        mine, my_starts, open_start, n_round = [], [], None, 0
        while not w.done:
            g = w.g
            d = agent.decide(w, explore=False)
            if d.action is None:
                g.state = "GAME_OVER"
                break
            if open_start is not None and "solver_real" not in open_start:
                ps = [c.p_clear for c in d.choice.cands if c.kind in ("play", "discard") and c.p_clear >= 0]
                open_start["solver_real"] = 1.0 if d.reason == "auto-play" else (max(ps) if ps else math.nan)
            if d.enc is not None:
                comp = pot.components(g)
                _, b = build(g)
                mine.append({"game": seed, "step": w.steps, "ante": current_ante(g), "blind": blind_index(g),
                             "furthest": g.furthest_blind, "in_round": g.state == "SELECTING_HAND",
                             "phi_old": comp["phi"], "headroom": comp["headroom"], "prog": comp["prog"],
                             "targets": boss_targets(g), **b})
            prev = g.state
            w.step(d.action)
            g = w.g
            if prev != "SELECTING_HAND" and g.state == "SELECTING_HAND":
                sk, b = build(g)
                tg = boss_targets(g)
                tests = {"blind": g.target, "boss": tg[0], "next": tg[1] if len(tg) > 1 else tg[0]}
                open_start = {"game": seed, "ante": current_ante(g), "idx": min(g.blind_idx, 2),
                              "boss": g.boss if g.blind_idx == 2 else "", "blind_no": blind_index(g),
                              "targets": tests, "scores128": k128.best_scores(g, sk),
                              "solver": {k: solver_fresh_round(g, t, n_solver, seed=_seed(("solver", sk)) % 2 ** 31)
                                         for k, t in tests.items()}, **b}
                my_starts.append(open_start)
                growth.append({"game": seed, "round": n_round, "jokers": snapshot(g)})
                n_round += 1
            if prev == "SELECTING_HAND" and g.state != "SELECTING_HAND" and open_start is not None:
                open_start["cleared"] = float(g.state in ("SHOP", "WON", "PACK", "BLIND_SELECT"))
                open_start["ratio"] = g.chips / g.target if g.target > 0 else 0.0
                open_start = None
        g = w.g
        for r in mine:
            r["win"] = float(g.state == "WON")
            r["blinds_after"] = float(g.furthest_blind - r["furthest"])
            r["final_blinds"] = float(g.furthest_blind)
            r["boss_cleared"] = float(g.furthest_blind >= 3 * r["ante"])
        for s in my_starts:
            s.setdefault("cleared", 0.0)
            s.setdefault("solver_real", math.nan)
        rows += mine
        starts += my_starts
        infos.append({"seed": seed, "won": g.state == "WON", "blinds": g.furthest_blind, "ante": g.ante})
    return rows, starts, infos, growth


def gen(games: int, workers: int, out: str, seed0: int = 10_000, deck: str = "RED", stake: str = "WHITE",
        n_solver: int = 100) -> dict:
    import multiprocessing as mp
    seeds = list(range(seed0, seed0 + games))
    chunks = [c for c in (seeds[i::workers] for i in range(workers)) if c]
    t = time.time()
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        parts = pool.map(_gen_worker, [(c, deck, stake, n_solver) for c in chunks])
    data = {"rows": sum((p[0] for p in parts), []), "starts": sum((p[1] for p in parts), []),
            "infos": sum((p[2] for p in parts), []), "growth": sum((p[3] for p in parts), []),
            "minutes": (time.time() - t) / 60}
    with open(out, "wb") as f:
        pickle.dump(data, f)
    return data


# ------------------------------------------------------------------ report
def _pred(entry: dict, target: float, cfg: StrengthConfig, key: str = "scores") -> float:
    return clear_chances(entry[key], entry["hands"], entry["discards"], [target], cfg, entry["seed"])[0]


def bucket_table(pred, *others) -> list[dict]:
    """Rows of (predicted bucket, n, mean predicted, mean of each other column)."""
    pred = np.asarray(pred, float)
    out = []
    for lo, hi in BUCKETS:
        m = (pred >= lo) & (pred < hi)
        if m.any():
            row = {"bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": int(m.sum()), "pred": round(float(pred[m].mean()), 3)}
            for name, col in others:
                col = np.asarray(col, float)
                ok = m & ~np.isnan(col)
                row[name] = round(float(col[ok].mean()), 3) if ok.any() else None
            out.append(row)
    return out


def clear_calibration(starts: list, cfg: StrengthConfig, key: str = "scores") -> dict:
    """Predicted clear chance at blind starts against the solver (fresh round, same target; three targets
    per state) and against whether the blind was really cleared (the blind's own target)."""
    p_all, s_all = [], []
    for name in ("blind", "boss", "next"):
        p_all += [_pred(s, s["targets"][name], cfg, key) for s in starts]
        s_all += [s["solver"][name] for s in starts]
    p_own = np.array([_pred(s, s["targets"]["blind"], cfg, key) for s in starts])
    cleared = np.array([s["cleared"] for s in starts])
    boss = np.array([s["idx"] == 2 for s in starts], bool)
    real = np.array([s["solver_real"] for s in starts], float)
    ok = ~np.isnan(real)
    p_all, s_all = np.array(p_all), np.array(s_all)

    def mean(x):
        return round(float(np.mean(x)), 4) if len(x) else None
    return {"vs_solver": bucket_table(p_all, ("solver", s_all)),
            "solver_mae": mean(np.abs(p_all - s_all)), "solver_bias": mean(p_all - s_all),
            "real_mae": mean(np.abs(p_own[ok] - real[ok])), "real_bias": mean(p_own[ok] - real[ok]),
            "actual_bias": mean(p_own - cleared),
            "vs_actual": bucket_table(p_own, ("cleared", cleared), ("solver", [s["solver"]["blind"] for s in starts]),
                                      ("solver_real", real)),
            "vs_actual_boss": bucket_table(p_own[boss], ("cleared", cleared[boss])),
            "vs_actual_nonboss": bucket_table(p_own[~boss], ("cleared", cleared[~boss])),
            "brier_actual": mean((p_own - cleared) ** 2),
            "brier_actual_boss": mean((p_own[boss] - cleared[boss]) ** 2)}


def fit(starts: list, scales=(0.8, 1.0, 1.25, 1.5, 2.0), weights=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)) -> list:
    """Every correction of a grid against the three references. "brier_heldout": the Brier score against the
    real outcome on the odd-seeded games (the grid is ranked on the even-seeded ones in `report`)."""
    even = [s for s in starts if s["game"] % 2 == 0]
    odd = [s for s in starts if s["game"] % 2 == 1]
    out = []
    for w in weights:
        for sc in scales:
            cfg = StrengthConfig(discards="none" if w == 0 else "extra_draws", discard_weight=w, score_scale=sc)
            c, a, b = clear_calibration(starts, cfg), clear_calibration(even, cfg), clear_calibration(odd, cfg)
            out.append({"discard_weight": w, "score_scale": sc, "solver_mae": c["solver_mae"],
                        "solver_bias": c["solver_bias"], "real_mae": c["real_mae"], "real_bias": c["real_bias"],
                        "brier_actual": c["brier_actual"], "actual_bias": c["actual_bias"],
                        "brier_fit_half": a["brier_actual"], "brier_heldout": b["brier_actual"]})
    return out


def _ranks(x) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    x = np.asarray(x, float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2.0)[inv]


def spearman(a, b) -> float:
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def within_ante(rows: list, cfg: StrengthConfig, pcfg: PotentialConfig | None = None) -> list[dict]:
    """Rank correlation, over the decisions of one ante at a time, of each quantity with what followed
    (blinds cleared after the state; this ante's boss cleared). Within an ante the progress so far is almost
    constant, so this asks the question without the "late states have little game left" effect."""
    pcfg = pcfg or PotentialConfig()
    out = []
    for ante in range(1, LAST_ANTE + 1):
        rs = [r for r in rows if r["ante"] == ante]
        if len(rs) < 200:
            continue
        new = [new_phi(r, cfg, pcfg) for r in rs]
        vals = {"phi_old": [r["phi_old"] for r in rs], "phi_new": [x[0] for x in new],
                "headroom_term": [r["headroom"] for r in rs], "strength_term": [x[1] for x in new]}
        row = {"ante": ante, "n": len(rs), "boss_cleared": round(float(np.mean([r["boss_cleared"] for r in rs])), 3)}
        for k, v in vals.items():
            row[f"{k}~blinds"] = round(spearman(v, [r["blinds_after"] for r in rs]), 3)
            row[f"{k}~boss"] = round(spearman(v, [r["boss_cleared"] for r in rs]), 3)
        out.append(row)
    return out


def new_phi(row: dict, cfg: StrengthConfig, pcfg: PotentialConfig) -> tuple[float, float, int]:
    """(Phi with the strength head term, the strength score, survives-through-ante) of a logged decision."""
    ch = clear_chances(row["scores"], row["hands"], row["discards"], row["targets"], cfg, row["seed"])
    s = strength_score(row["ante"], ch, cfg.score)
    return pcfg.w_head * s + pcfg.w_prog * row["prog"], s, survives_through(row["ante"], ch)


def phi_calibration(rows: list, cfg: StrengthConfig, pcfg: PotentialConfig | None = None, buckets: int = 8) -> dict:
    """diagnostics.calibration of the old Phi, the new Phi and their head terms alone against the win, the
    blinds cleared after the state, and whether the state's ante's boss was cleared."""
    pcfg = pcfg or PotentialConfig()
    for r in rows:
        r.setdefault("final_blinds", r["furthest"] + r["blinds_after"])
    games = [r["game"] for r in rows]
    new = [new_phi(r, cfg, pcfg) for r in rows]
    values = {"phi_old": [r["phi_old"] for r in rows], "phi_new": [x[0] for x in new],
              "headroom_term": [r["headroom"] for r in rows], "strength_term": [x[1] for x in new],
              "survives_minus_ante": [x[2] - r["ante"] for x, r in zip(new, rows)]}
    out = {}
    for name, v in values.items():
        out[name] = {y: calibration(v, [r[y] for r in rows], games, buckets=buckets)
                     for y in OUTCOMES}
        out[name]["range"] = [round(float(min(v)), 4), round(float(max(v)), 4)]
    return out


def _fmt_buckets(rows: list, cols) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(r.get(c)) for c in cols) + " |\n" for r in rows)


def _fmt_calibration(cal: dict, y: str) -> str:
    names = list(cal)
    n = max(len(cal[k][y]["buckets"]) for k in names)
    lines = ["| bucket | " + " | ".join(f"{k}: value / {y}" for k in names) + " |", "|---|" + "---|" * len(names)]
    for b in range(n):
        cells = []
        for k in names:
            bs = cal[k][y]["buckets"]
            cells.append(f"{bs[b]['value']:.3f} / {bs[b]['win_rate']:.3f} (n={bs[b]['n']})" if b < len(bs) else "")
        lines.append(f"| {b + 1} | " + " | ".join(cells) + " |")
    lines.append("| status | " + " | ".join(f"{cal[k][y]['status']} (rho {cal[k][y].get('spearman')})" for k in names) + " |")
    return "\n".join(lines) + "\n"


def report(data: dict, cfg: StrengthConfig | None = None) -> str:
    cfg = cfg or StrengthConfig()
    rows, starts, infos = data["rows"], data["starts"], data["infos"]
    out = [f"{len(infos)} games, {sum(i['won'] for i in infos)} won, mean blinds "
           f"{np.mean([i['blinds'] for i in infos]):.2f}; {len(rows)} decisions, {len(starts)} blind starts "
           f"({sum(s['idx'] == 2 for s in starts)} bosses)\n"]
    variants = {"no correction (independent hands, no discards)": StrengthConfig(discards="none"),
                "one extra draw per discard": StrengthConfig(discards="extra_draws", discard_weight=1.0)}
    if (cfg.discards, cfg.discard_weight, cfg.score_scale) != ("extra_draws", 1.0, 1.0):
        variants["configured"] = cfg
    for name, c in variants.items():
        cal = clear_calibration(starts, c)
        out.append(f"### Clear chance, {name}: discards={c.discards}, weight={c.discard_weight}, scale={c.score_scale}\n")
        out.append(f"(a) vs the solver's playout policy on fresh rounds (3 targets per blind start): MAE "
                   f"{cal['solver_mae']}, bias {cal['solver_bias']}\n\n"
                   + _fmt_buckets(cal["vs_solver"], ["bucket", "n", "pred", "solver"]))
        out.append(f"\n(b) vs the blind really cleared (its own target): Brier {cal['brier_actual']}, bias "
                   f"{cal['actual_bias']}; vs the solver at the real first hand: MAE {cal['real_mae']}, bias "
                   f"{cal['real_bias']}\n\n"
                   + _fmt_buckets(cal["vs_actual"], ["bucket", "n", "pred", "cleared", "solver", "solver_real"]))
        out.append("\nboss blinds only (boss effects are not modelled): Brier "
                   f"{cal['brier_actual_boss']}\n\n" + _fmt_buckets(cal["vs_actual_boss"], ["bucket", "n", "pred", "cleared"]))
        out.append("\nsmall and big blinds only\n\n" + _fmt_buckets(cal["vs_actual_nonboss"], ["bucket", "n", "pred", "cleared"]))
    k128 = clear_calibration(starts, cfg, "scores128")
    out.append(f"\nconfigured, K=128 hands instead of 32: solver MAE {k128['solver_mae']}, bias {k128['solver_bias']}, "
               f"Brier {k128['brier_actual']}, bias vs actual {k128['actual_bias']}\n")
    cl = np.array([s["cleared"] for s in starts])
    for name, key in (("playout policy on fresh rounds", None), ("estimate at the real first hand", "solver_real")):
        sv = np.array([s["solver"]["blind"] if key is None else s[key] for s in starts], float)
        ok = ~np.isnan(sv)
        out.append(f"\n### The solver's own {name} vs the blind really cleared (Brier "
                   f"{float(((sv[ok] - cl[ok]) ** 2).mean()):.4f})\n\n"
                   + _fmt_buckets(bucket_table(sv[ok], ("cleared", cl[ok])), ["bucket", "n", "pred", "cleared"]))
    grid = fit(starts)
    cols = ["discard_weight", "score_scale", "solver_mae", "solver_bias", "real_mae", "real_bias", "brier_actual",
            "actual_bias", "brier_fit_half", "brier_heldout"]
    out.append("\n### Grid of corrections, best 8 by MAE against the solver's fresh-round playouts\n\n"
               + _fmt_buckets(sorted(grid, key=lambda r: r["solver_mae"])[:8], cols))
    out.append("\n### Grid of corrections, best 8 by Brier against the real outcome on the even-seeded games "
               "(brier_heldout: the odd-seeded games)\n\n"
               + _fmt_buckets(sorted(grid, key=lambda r: (r["brier_fit_half"] is None, r["brier_fit_half"]))[:8], cols))
    for kind in ("horizon", "expected_antes", "expected_total", "current"):
        c = StrengthConfig(**{**cfg.__dict__, "score": kind})
        cal = phi_calibration(rows, c)
        out.append(f"\n### Phi calibration, strength score = {kind}\n\nranges: "
                   + ", ".join(f"{k} {cal[k]['range']}" for k in cal) + "\n")
        for y in OUTCOMES[1:]:
            out.append(f"\nagainst {y} (bucket mean value / mean outcome):\n\n" + _fmt_calibration(cal, y))
        wa = within_ante(rows, c)
        out.append("\nwithin one ante at a time (rank correlation with blinds cleared afterwards / this ante's boss "
                   "cleared):\n\n" + _fmt_buckets(wa, list(wa[0]) if wa else ["ante"]))
    return "\n".join(out)


# ------------------------------------------------------------------ cost
def cost(games: int = 6, seed0: int = 10_000) -> dict:
    """Milliseconds per uncached call and per decision (with the cache) of the headroom and the strength,
    on the decisions of a few games of the untrained agent without search."""
    import torch
    torch.set_num_threads(1)
    from ..az.agent import Agent, AgentConfig
    from ..az.net import AZNet
    from ..az.world import World
    from ..sim.game import Game
    from .growth import project
    torch.manual_seed(0)
    agent = Agent(AZNet().eval(), AgentConfig(search=False), seed=0)
    states = []
    for seed in range(seed0, seed0 + games):
        w = World(Game(seed=seed, stake="WHITE"))
        agent.reseed(seed * 7919 + 17)
        while not w.done:
            d = agent.decide(w, explore=False)
            if d.action is None:
                break
            if d.enc is not None:
                states.append(w.g.clone())
            w.step(d.action)
    res = {"decisions": len(states)}
    table = {j.key: (1.0, 0.0) for g in states for j in g.jokers}        # every joker "grows": the worst case

    class Growing(Strength):
        def table(self):
            return table
    for name, make in (("headroom", lambda: Potential(PotentialConfig()).headroom),
                       ("strength", lambda: Strength(StrengthConfig())),
                       ("strength_growth", lambda: Growing(StrengthConfig(growth=True)))):
        calc = make()
        call = calc.raw if name == "headroom" else calc.report
        t = time.perf_counter()
        for g in states:
            call(g)
        wall = time.perf_counter() - t
        st = calc.stats
        res[name] = {"ms/decision_cached": round(1e3 * wall / len(states), 4),
                     "ms/uncached_call": round(1e3 * st["seconds"] / max(1, st["calls"] - st["hits"]), 4),
                     "cache_hit%": round(100 * st["hits"] / st["calls"], 1)}
        t = time.perf_counter()
        for g in states:
            call(g)
        res[name]["ms/cache_hit"] = round(1e3 * (time.perf_counter() - t) / len(states), 4)
    return res


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--games", type=int, default=120)
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--seed0", type=int, default=10_000)
    g.add_argument("--solver-rounds", type=int, default=100)
    g.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--data", required=True)
    r.add_argument("--cfg", default="", help='StrengthConfig overrides as JSON, e.g. \'{"score_scale": 1.5}\'')
    c = sub.add_parser("cost")
    c.add_argument("--games", type=int, default=6)
    a = p.parse_args()
    if a.cmd == "gen":
        d = gen(a.games, a.workers, a.out, a.seed0, n_solver=a.solver_rounds)
        print(f"{len(d['infos'])} games, {len(d['rows'])} decisions, {len(d['starts'])} blind starts in "
              f"{d['minutes']:.1f} min -> {a.out}")
    elif a.cmd == "report":
        with open(a.data, "rb") as f:
            data = pickle.load(f)
        print(report(data, StrengthConfig(**json.loads(a.cfg)) if a.cfg else None))
    else:
        print(json.dumps(cost(a.games), indent=1))


if __name__ == "__main__":
    main()
