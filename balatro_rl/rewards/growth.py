"""Projected growth of scaling jokers: the table and the script that builds it from logged games.

A scaling joker keeps a number in state["val"] (Green Joker's mult, Runner's chips, Constellation's
Xmult ...). The table gives, per joker key, the average change of that number per round (blind) held:

    {"green_joker": {"gain": 2.1, "n": 340, "lo": 0.0}, ...}          (a bare number is read as the gain)

strength.py (config `growth: true`) moves each such joker forward by gain * (1 + d + ... + d^(r-1)) for the
r rounds until the boss in question (d = growth_discount) before scoring; a falling value (Ice Cream,
Popcorn, Ramen) is not taken below `lo`, the lowest value seen in the logs.

The built-in table (DEFAULT_TABLE) is EMPTY: with it, `growth: true` changes nothing. Build one from logs:

    python -m balatro_rl.rewards.growth log   --games 200 --workers 2 --out growth_log.jsonl
    python -m balatro_rl.rewards.growth build --logs growth_log.jsonl --out growth_table.json

Data needed (one JSON line per blind start, written by `log` or by anything else that plays games):
    {"game": id, "round": n, "jokers": [[uid, key, val], ...]}
with `round` counting the blinds started in that game (skipped blinds are not rounds) and every joker held
that has a numeric state["val"]. A joker's gain over a round is its val at the next line of the same game
minus its val now, for jokers (by uid) present in both. The gains are those of whoever played the logged
games: log the agent the table is for (a weak player grows its jokers less, and loses before late antes).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

DEFAULT_TABLE: dict = {}
MIN_COUNT = 20                      # a joker needs this many logged rounds to get an entry
NOT_SCALING = {"loyalty_card"}      # their "val" is a counter that cycles, not a value that grows


def load_table(path: str = "") -> dict:
    """{key: (gain, lo)} from a JSON file, or the built-in table when `path` is empty."""
    raw = DEFAULT_TABLE
    if path:
        with open(path) as f:
            raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[k] = (float(v["gain"]), float(v.get("lo", 0.0)))
        else:
            out[k] = (float(v), 0.0)
    return out


def discounted_rounds(rounds: int, discount: float) -> float:
    return float(sum(discount ** i for i in range(max(0, int(rounds)))))


def project(g, table: dict, rounds: int, discount: float):
    """A copy of g whose scaling jokers have moved `rounds` rounds forward (g is not modified)."""
    eff = discounted_rounds(rounds, discount)
    p = g.clone()
    for j in p.jokers:
        v = j.state.get("val")
        if j.key in table and isinstance(v, (int, float)) and not isinstance(v, bool):
            gain, lo = table[j.key]
            new = v + gain * eff
            if gain < 0:
                new = max(new, min(v, lo))
            j.state["val"] = new
    return p


def snapshot(g) -> list:
    """[uid, key, val] of every joker with a numeric val: what one log line holds."""
    return [[j.uid, j.key, float(j.state["val"])] for j in g.jokers
            if isinstance(j.state.get("val"), (int, float)) and not isinstance(j.state.get("val"), bool)]


def build_table(lines, min_count: int = MIN_COUNT) -> dict:
    """The table from log lines (dicts as described in the module doc)."""
    games = defaultdict(list)
    for ln in lines:
        games[ln["game"]].append(ln)
    gains, lows = defaultdict(list), {}
    for rounds in games.values():
        rounds.sort(key=lambda r: r["round"])
        for a, b in zip(rounds, rounds[1:]):
            nxt = {uid: val for uid, _, val in b["jokers"]}
            for uid, key, val in a["jokers"]:
                lows[key] = min(lows.get(key, val), val)
                if uid in nxt:
                    gains[key].append(nxt[uid] - val)
                    lows[key] = min(lows[key], nxt[uid])
    return {k: {"gain": round(sum(v) / len(v), 4), "n": len(v), "lo": lows[k]}
            for k, v in sorted(gains.items()) if len(v) >= min_count and k not in NOT_SCALING}


def _log_worker(args):
    seeds, deck, stake = args
    import torch
    torch.set_num_threads(1)
    from ..az.agent import Agent, AgentConfig
    from ..az.net import AZNet
    from ..az.world import World
    from ..sim.game import Game
    torch.manual_seed(0)
    agent = Agent(AZNet().eval(), AgentConfig(search=False), seed=seeds[0])
    out = []
    for s in seeds:
        w = World(Game(seed=s, deck_type=deck, stake=stake))
        agent.reseed(s * 7919 + 17)
        n = 0
        while not w.done:
            prev = w.g.state
            d = agent.decide(w, explore=False)
            if d.action is None:
                break
            w.step(d.action)
            if prev != "SELECTING_HAND" and w.g.state == "SELECTING_HAND":
                out.append({"game": s, "round": n, "jokers": snapshot(w.g)})
                n += 1
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log", help="play games with the untrained agent (no search) and log joker values")
    lg.add_argument("--games", type=int, default=200)
    lg.add_argument("--workers", type=int, default=2)
    lg.add_argument("--seed0", type=int, default=20_000)
    lg.add_argument("--deck", default="RED")
    lg.add_argument("--stake", default="WHITE")
    lg.add_argument("--out", required=True)
    b = sub.add_parser("build", help="the growth table from logged games")
    b.add_argument("--logs", nargs="+", required=True)
    b.add_argument("--min-count", type=int, default=MIN_COUNT)
    b.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "log":
        import multiprocessing as mp
        seeds = list(range(a.seed0, a.seed0 + a.games))
        chunks = [c for c in (seeds[i::a.workers] for i in range(a.workers)) if c]
        with mp.get_context("spawn").Pool(len(chunks)) as pool:
            parts = pool.map(_log_worker, [(c, a.deck, a.stake) for c in chunks])
        with open(a.out, "w") as f:
            for part in parts:
                for ln in part:
                    f.write(json.dumps(ln) + "\n")
        print(f"{sum(len(x) for x in parts)} rounds of {a.games} games -> {a.out}")
    else:
        lines = []
        for path in a.logs:
            with open(path) as f:
                lines += [json.loads(x) for x in f if x.strip()]
        table = build_table(lines, a.min_count)
        with open(a.out, "w") as f:
            json.dump(table, f, indent=1)
        print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
