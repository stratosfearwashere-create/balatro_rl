"""Where and how a policy checkpoint loses: greedy games on fixed seeds on the C++ engine, with the state
at the end of every game.

    python tools/loss_analysis.py checkpoints/ppo_cpp_white3.pt --games 300 --seed0 10000 --stake WHITE

Reports: where games end (ante, blind, boss), how close the lost blind was, and what the build looked like at
the end, for wins against losses: jokers, xMult, hand levels, money, unused consumables, deck changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from balatro_rl.engine_env import CppEnv, OBS_KEYS          # noqa: E402
from balatro_rl.heuristic import CHIPS, XMULT                # noqa: E402
from balatro_rl.model import load_model                      # noqa: E402
from balatro_rl.sim.hands import HAND_NAMES                  # noqa: E402

ECON = {"golden", "rocket", "business", "mail", "reserved_parking", "trading", "faceless", "delayed_grat", "to_the_moon",
        "egg", "cloud_9", "hallucination", "cartomancer", "8_ball", "certificate", "dna", "midas_mask", "gift",
        "satellite", "vagabond", "riff_raff", "todo_list", "rough_gem", "matador", "ticket", "chaos", "credit_card"}
BLIND = ("small", "big", "boss")


def action_kind(ai: int) -> str:
    """The kind of an action index, from the environment's own layout (it moves when a slot count changes)."""
    from balatro_rl import env as E
    if ai < E.A_SELECT:
        return "play" if ai < E.A_DISC else "discard"
    for lo, hi, name in ((E.A_SELECT, E.A_SELECT + 1, "select"), (E.A_SKIP, E.A_SKIP + 1, "skip"), (E.A_REROLL_BOSS, E.A_REROLL_BOSS + 1, "reroll_boss"), (E.A_BUY, E.A_BUY_PACK, "buy"),
                         (E.A_BUY_PACK, E.A_VOUCHER, "buy_pack"), (E.A_VOUCHER, E.A_REROLL, "voucher"), (E.A_REROLL, E.A_REROLL + 1, "reroll"), (E.A_LEAVE, E.A_LEAVE + 1, "leave"),
                         (E.A_SELL_J, E.A_SELL_C, "sell_joker"), (E.A_SELL_C, E.A_USE_C, "sell_cons"), (E.A_USE_C, E.A_PICK, "use"), (E.A_PICK, E.A_PSKIP, "pick"), (E.A_PSKIP, E.A_PSKIP + 1, "pack_skip"),
                         (E.A_SWAP, E.A_REPLACE, "swap")):
        if lo <= ai < hi:
            return name
    return "replace"


def play_exact(path: str, games: int, seed0: int, deck: str, stake: str, device: str) -> list[dict]:
    """Exact-seed games with the privileged observation on (a network that reads the future list): the games are played together in a
    vector environment, greedily, and each one's actions are then replayed in a single environment for its end state."""
    from balatro_rl.engine_env import CppVecEnv
    model = load_model(path, device)
    v = CppVecEnv(games, deck, stake, seed0=seed0, threads=0)
    if getattr(model, "marg", 0):
        v.set_marginal(int(model.marg))
    v.set_exact(True)
    v.v.restart()
    v.set_priv(True)
    lines = [[] for _ in range(games)]
    live = np.ones(games, dtype=bool)
    while live.any():
        idx = np.nonzero(live)[0]
        batch = {k: torch.from_numpy(np.array(v.obs[k])[idx]).to(device) for k in OBS_KEYS}
        with torch.no_grad():
            logits, _ = model(batch)
        act = np.full(games, -1, dtype=np.int64)               # (a finished game has been reset: leave it alone)
        act[idx] = logits.argmax(1).cpu().numpy()
        for i in idx:
            lines[i].append(int(act[i]))
        _, done, _ = v.step(act)
        live &= ~np.asarray(done, dtype=bool)
    out = []
    for i in range(games):
        e = CppEnv(deck, stake)
        e.e.exact = True
        e.reset(seed0 + i)
        acts, mx = Counter(), 0
        for ai in lines[i]:
            acts[action_kind(ai)] += 1
            _, _, done, _ = e.step(ai)
            mx = max(mx, int(e.game.money))
            if done:
                break
        d = e.game.dump()
        d["won"] = d["state"] == "WON"
        d["acts"] = dict(acts)
        d["steps"] = len(lines[i])
        d["max_money"] = mx
        out.append(d)
    return out


def play(path: str, games: int, seed0: int, deck: str, stake: str, device: str, money=None) -> list[dict]:
    model = load_model(path, device)
    if getattr(model, "priv", 0):
        del model
        return play_exact(path, games, seed0, deck, stake, device)
    envs = [CppEnv(deck, stake) for _ in range(games)]
    if getattr(model, "marg", 0):                              # a network made with the marginal-value inputs
        for e in envs:
            e.e.marginal = int(model.marg)
    obs = [e.reset(seed0 + i) for i, e in enumerate(envs)]
    if money is not None:                          # every game starts with this much (a rich start)
        for i, e in enumerate(envs):
            e.game.money = money
            obs[i] = e.e.observe()
    live = list(range(games))
    out = [None] * games
    acts = [Counter() for _ in range(games)]
    steps = [0] * games
    max_money = [0] * games
    while live:
        batch = {k: torch.from_numpy(np.stack([obs[i][k] for i in live])).to(device) for k in OBS_KEYS}
        with torch.no_grad():
            logits, _ = model(batch)
        a = logits.argmax(1).cpu().numpy()
        nxt = []
        for i, ai in zip(live, a):
            ai = int(ai)
            kind = action_kind(ai)
            acts[i][kind] += 1
            steps[i] += 1
            o, r, done, info = envs[i].step(ai)
            max_money[i] = max(max_money[i], int(envs[i].game.money))
            if done:
                d = envs[i].game.dump()
                d["won"] = d["state"] == "WON"
                d["acts"] = dict(acts[i])
                d["steps"] = steps[i]
                d["max_money"] = max_money[i]
                out[i] = d
            else:
                obs[i] = o
                nxt.append(i)
        live = nxt
    return out


def order_stats(gs: list[dict]) -> dict:
    """Joker order: additive mult should sit left of multipliers (jokers score left to right)."""
    from balatro_rl.az.guide import MULT
    both = pairs = good = perfect = 0
    for g in gs:
        keys = [j["key"] for j in g["jokers"]]
        add = [i for i, k in enumerate(keys) if k in MULT and k not in XMULT]
        mul = [i for i, k in enumerate(keys) if k in XMULT]
        if not add or not mul:
            continue
        both += 1
        ok = sum(1 for a_ in add for m_ in mul if a_ < m_)
        pairs += len(add) * len(mul)
        good += ok
        perfect += ok == len(add) * len(mul)
    return {"games": both, "pairs": pairs, "pairs in the right order %": 100 * good / max(pairs, 1),
            "builds fully in order %": 100 * perfect / max(both, 1)}


def pct(n, d):
    return f"{100 * n / max(d, 1):5.1f}%"


def build_stats(gs: list[dict]) -> dict:
    if not gs:
        return {}
    n = len(gs)
    jk = [[j["key"] for j in g["jokers"]] for g in gs]
    top = [max(range(12), key=lambda h: (g["hand_played"][h], h)) for g in gs]
    return {
        "games": n,
        "jokers held": np.mean([len(j) for j in jk]),
        "joker slots free": np.mean([g["joker_slots"] - len(g["jokers"]) for g in gs]),
        "xMult jokers": np.mean([sum(k in XMULT for k in j) for j in jk]),
        "games with no xMult joker %": 100 * np.mean([not any(k in XMULT for k in j) for j in jk]),
        "chips jokers": np.mean([sum(k in CHIPS for k in j) for j in jk]),
        "economy jokers": np.mean([sum(k in ECON for k in j) for j in jk]),
        "level of the most played hand": np.mean([g["hand_levels"][t] for g, t in zip(gs, top)]),
        "share of hands that are the most played type %": 100 * np.mean(
            [g["hand_played"][t] / max(sum(g["hand_played"]), 1) for g, t in zip(gs, top)]),
        "money at the end": np.mean([g["money"] for g in gs]),
        "most money held": np.mean([g["max_money"] for g in gs]),
        "consumables left unused": np.mean([len(g["consumables"]) for g in gs]),
        "vouchers": np.mean([len(g["vouchers"]) for g in gs]),
        "deck size": np.mean([len(g["full_deck"]) for g in gs]),
        "enhanced / sealed / edition cards": np.mean(
            [sum(1 for c in g["full_deck"] if c[2] or c[3] or c[4]) for g in gs]),
        "blinds skipped": np.mean([g["blinds_skipped"] for g in gs]),
        "rerolls per game": np.mean([g["acts"].get("reroll", 0) for g in gs]),
        "packs bought per game": np.mean([g["acts"].get("buy_pack", 0) for g in gs]),
        "cards / jokers bought per game": np.mean([g["acts"].get("buy", 0) for g in gs]),
        "jokers sold per game": np.mean([g["acts"].get("sell_joker", 0) for g in gs]),
        "jokers replaced per game (sell + buy in one)": np.mean([g["acts"].get("replace", 0) for g in gs]),
        "joker position swaps per game": np.mean([g["acts"].get("swap", 0) for g in gs]),
        "consumables used per game": np.mean([g["acts"].get("use", 0) for g in gs]),
        "discards per hand played": np.mean([g["acts"].get("discard", 0) / max(g["acts"].get("play", 0), 1) for g in gs]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--deck", default="RED")
    p.add_argument("--stake", default="WHITE")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", default="", help="write the end states to this JSON file")
    p.add_argument("--money", type=int, default=None, help="every game starts with this much money")
    a = p.parse_args()
    gs = play(a.checkpoint, a.games, a.seed0, a.deck, a.stake, a.device, a.money)
    if a.save:
        json.dump(gs, open(a.save, "w"))
    wins = [g for g in gs if g["won"]]
    losses = [g for g in gs if not g["won"]]
    n = len(gs)
    print(f"{n} greedy games, seeds {a.seed0}.., {a.deck} deck, {a.stake} stake: {len(wins)} wins ({pct(len(wins), n).strip()}), "
          f"{np.mean([g['furthest_blind'] for g in gs]):.2f} blinds beaten on average\n")

    print("WHERE THE LOSSES HAPPEN")
    print(f"  {'ante':>4} {'lost':>5} {'share':>7}   small   big  boss   mean score / target at the loss")
    for ante in range(1, 9):
        at = [g for g in losses if g["ante"] == ante]
        if not at:
            continue
        by = Counter(min(g["blind_idx"], 2) for g in at)
        ratio = np.mean([g["chips"] / max(g["target"], 1) for g in at])
        print(f"  {ante:>4} {len(at):>5} {pct(len(at), len(losses)):>7}   {by[0]:5d} {by[1]:5d} {by[2]:5d}   {ratio:.2f}")
    kind = Counter(min(g["blind_idx"], 2) for g in losses)
    print(f"  by blind: " + ", ".join(f"{BLIND[k]} {kind[k]} ({pct(kind[k], len(losses)).strip()})" for k in range(3)))
    ratios = np.array([g["chips"] / max(g["target"], 1) for g in losses])
    print(f"  how close: {pct((ratios >= 0.75).sum(), len(losses)).strip()} of losses scored at least 75% of the target, "
          f"{pct((ratios < 0.5).sum(), len(losses)).strip()} under half, {pct((ratios < 0.25).sum(), len(losses)).strip()} under a quarter")

    print("\nBOSSES THAT END RUNS (losses on the boss blind)")
    faced = Counter()
    for g in gs:
        for b in g["used_bosses"]:
            faced[b] += 1
    killed = Counter(g["boss"] for g in losses if min(g["blind_idx"], 2) == 2)
    rows = sorted(killed.items(), key=lambda kv: -kv[1])[:12]
    for b, k in rows:
        print(f"  {b:16s} ended {k:3d} runs")

    print("\nTHE BUILD AT THE END: wins against losses")
    late = [g for g in losses if g["ante"] >= 6]
    early = [g for g in losses if g["ante"] <= 3]
    cols = [("wins", build_stats(wins)), ("lost ante 6-8", build_stats(late)), ("lost ante 4-5", build_stats([g for g in losses if 4 <= g["ante"] <= 5])),
            ("lost ante 1-3", build_stats(early))]
    keys = next((list(st) for _, st in cols if st), [])               # the row names: from whichever column has games
    print(f"  {'':46s}" + "".join(f"{name:>15s}" for name, _ in cols))
    for k in keys:
        print(f"  {k:46s}" + "".join(f"{(st.get(k, float('nan'))):15.2f}" for _, st in cols))

    o = order_stats(gs)
    print(f"\nJOKER ORDER (+mult left of xMult; chance is 50%): {o['games']} end states hold both; "
          f"{o['pairs in the right order %']:.1f}% of +mult / xMult pairs in the right order, "
          f"{o['builds fully in order %']:.1f}% of those builds fully in order")
    print("\nMOST PLAYED HAND (share of games where it is the main hand)")
    for name, part in (("wins", wins), ("losses", losses)):
        c = Counter(max(range(12), key=lambda h: (g["hand_played"][h], h)) for g in part)
        print(f"  {name:7s} " + ", ".join(f"{HAND_NAMES[h]} {pct(k, len(part)).strip()}" for h, k in c.most_common(5)))

    print("\nJOKERS: held at the end in wins against losses (at least 8 games)")
    cw, cl = Counter(), Counter()
    for g in wins:
        cw.update({j["key"] for j in g["jokers"]})
    for g in losses:
        cl.update({j["key"] for j in g["jokers"]})
    rows = []
    for k in set(cw) | set(cl):
        tot = cw[k] + cl[k]
        if tot >= 8:
            rows.append((cw[k] / tot, k, cw[k], cl[k]))
    rows.sort(reverse=True)
    base = len(wins) / n
    print(f"  (overall win rate {100 * base:.1f}%)")
    print("  best:  " + ", ".join(f"{k} {w}/{w + l}" for _, k, w, l in rows[:10]))
    print("  worst: " + ", ".join(f"{k} {w}/{w + l}" for _, k, w, l in rows[-10:]))
    held = Counter()
    for g in gs:
        held.update({j["key"] for j in g["jokers"]})
    print("  most held: " + ", ".join(f"{k} {v}" for k, v in held.most_common(12)))


if __name__ == "__main__":
    main()
