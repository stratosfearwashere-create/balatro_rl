"""Usage statistics from the generator's event log (value/gen.py writes chunk*.events.jsonl.gz and games.jsonl).

    python -m balatro_rl.value.stats --data checkpoints/value_data                 # every stake
    python -m balatro_rl.value.stats --data checkpoints/value_data --stake WHITE --top 25

Tables: jokers (bought / picked, sold, held at an ante start, win rate of runs that held them), vouchers,
tags skipped for, consumables (bought, used in a round / in the shop), packs, rerolls per shop by ante, and
where runs died. Win rates are per run (a joker held in several antes counts once). Everything is read
from the log: no games are played.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import Counter, defaultdict


def load_events(data_dir: str):
    games = {}
    with open(os.path.join(data_dir, "games.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            games[r["seed"]] = r
    events = []
    for path in sorted(glob.glob(os.path.join(data_dir, "chunk*.events.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            events += [json.loads(l) for l in f]
    return games, events


def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def report(games: dict, events: list, stake: str | None = None, top: int = 20, min_runs: int = 20) -> str:
    if stake:
        games = {s: g for s, g in games.items() if g["stake"] == stake}
        events = [e for e in events if e["game"] in games]
    n_games = len(games)
    wins = sum(g["won"] for g in games.values())
    out = [f"{n_games} games{' at ' + stake if stake else ''}, {wins} won ({100 * wins / max(1, n_games):.1f}%), "
           f"{len(events)} events"]
    if not n_games:
        return out[0]
    won = {s for s, g in games.items() if g["won"]}

    # ---- jokers: held at an ante start (per run), bought / picked, sold
    held_runs = defaultdict(set)
    first_ante = defaultdict(list)
    for e in events:
        if e["kind"] == "ante":
            for j in e["jokers"]:
                key = j.split(":")[0]
                if e["game"] not in held_runs[key]:
                    first_ante[key].append(e["ante"])
                held_runs[key].add(e["game"])
    bought = Counter(e["key"] for e in events if e["kind"] in ("buy", "pick") and e.get("key", "").startswith("j_"))
    sold = Counter(e["key"] for e in events if e["kind"] == "sell_joker")
    rows = []
    for key, runs in held_runs.items():
        n = len(runs)
        if n < min_runs:
            continue
        w = len(runs & won)
        lo, hi = wilson(w, n)
        rows.append((w / n, key, n, w, lo, hi, bought[key], sold[key], sum(first_ante[key]) / len(first_ante[key])))
    rows.sort(reverse=True)
    out.append(f"\nJokers by win rate of the runs that held them at an ante start (>= {min_runs} runs); base rate "
               f"{100 * wins / n_games:.1f}%")
    out.append(f"{'joker':22s} {'runs':>5s} {'win%':>6s} {'95% CI':>13s} {'bought':>6s} {'sold':>5s} {'1st ante':>8s}")
    for wr, key, n, w, lo, hi, b, s, fa in rows[:top]:
        out.append(f"{key[2:]:22s} {n:5d} {100 * wr:6.1f} {100 * lo:6.1f}-{100 * hi:5.1f} {b:6d} {s:5d} {fa:8.1f}")
    if len(rows) > top:
        out.append("  ...")
        for wr, key, n, w, lo, hi, b, s, fa in rows[-min(top // 2, len(rows) - top):]:
            out.append(f"{key[2:]:22s} {n:5d} {100 * wr:6.1f} {100 * lo:6.1f}-{100 * hi:5.1f} {b:6d} {s:5d} {fa:8.1f}")
    most_bought = bought.most_common(top)
    out.append("\nMost bought / picked jokers: " + ", ".join(f"{k[2:]} {v}" for k, v in most_bought))

    # ---- vouchers and tags
    def item_table(title, kind, prefix=""):
        runs = defaultdict(set)
        for e in events:
            if e["kind"] == kind and e.get("key", "").startswith(prefix):
                runs[e["key"]].add(e["game"])
        rows = sorted(((len(r), k, len(r & won)) for k, r in runs.items()), reverse=True)
        out.append(f"\n{title} (runs that took it, win rate):")
        for n, k, w in rows[:top]:
            lo, hi = wilson(w, n)
            out.append(f"  {k:24s} {n:5d} runs  {100 * w / n:5.1f}%  ({100 * lo:.0f}-{100 * hi:.0f})")
    item_table("Vouchers", "voucher", "v_")
    item_table("Tags skipped for", "skip", "tag_")

    # ---- consumables
    cons_buy = Counter(e["key"] for e in events if e["kind"] in ("buy", "pick") and e.get("key", "").startswith("c_"))
    use_round = Counter(e["key"] for e in events if e["kind"] == "use" and e["in_round"])
    use_shop = Counter(e["key"] for e in events if e["kind"] == "use" and not e["in_round"])
    cons_sold = Counter(e["key"] for e in events if e["kind"] == "sell_cons")
    keys = sorted(set(cons_buy) | set(use_round) | set(use_shop), key=lambda k: -(use_round[k] + use_shop[k]))
    out.append(f"\nConsumables: {'key':20s} {'got':>6s} {'used/round':>10s} {'used/shop':>9s} {'sold':>5s}")
    for k in keys[:top]:
        out.append(f"             {k[2:]:20s} {cons_buy[k]:6d} {use_round[k]:10d} {use_shop[k]:9d} {cons_sold[k]:5d}")
    packs = Counter(e["key"] for e in events if e["kind"] == "buy_pack")
    skips = Counter(e.get("pack") for e in events if e["kind"] == "pack_skip")
    out.append("\nPacks bought: " + ", ".join(f"{k[2:]} {v}" for k, v in packs.most_common()))
    out.append("Packs skipped without a pick: " + ", ".join(f"{k} {v}" for k, v in skips.most_common()))

    # ---- rerolls and money by ante
    rer = defaultdict(int)
    shops = defaultdict(set)
    money_at = defaultdict(list)
    for e in events:
        if e["kind"] == "reroll":
            rer[e["ante"]] += 1
        if e["kind"] in ("buy", "buy_pack", "voucher", "reroll", "leave", "sell_joker"):
            shops[e["ante"]].add((e["game"], e["ante"], e["blind"]))
        if e["kind"] == "ante":
            money_at[e["ante"]].append(e["money"])
    out.append("\nAnte:   runs reaching   money at start (mean)   rerolls per shop")
    for a in range(1, 9):
        n = len(money_at[a])
        if n:
            out.append(f"  {a}     {n:6d}          {sum(money_at[a]) / n:8.1f}             {rer[a] / max(1, len(shops[a])):.2f}")

    # ---- deaths
    died = Counter((g["ante"], g.get("died_blind", -1)) for g in games.values() if not g["won"])
    boss_deaths = Counter(g.get("boss", "") for g in games.values() if not g["won"] and g.get("died_blind") == 2)
    out.append("\nDeaths by ante / blind (small, big, boss): " + ", ".join(
        f"ante {a} {'sbB'[b] if 0 <= b < 3 else '?'}: {v}" for (a, b), v in sorted(died.items())))
    out.append("Boss deaths: " + ", ".join(f"{k} {v}" for k, v in boss_deaths.most_common(12)))
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="checkpoints/value_data")
    p.add_argument("--stake", default=None)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--min-runs", type=int, default=20)
    a = p.parse_args()
    games, events = load_events(a.data)
    print(report(games, events, a.stake.upper() if a.stake else None, a.top, a.min_runs))


if __name__ == "__main__":
    main()
