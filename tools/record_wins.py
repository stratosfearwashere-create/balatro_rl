"""Records what a late-game sandbox network wins with (run beside the trainer; python tools/record_wins.py).

Every --every-iters updates of the run it copies the checkpoint, samples --games games with the sandbox's rules (start ante, money, planet cap, banned jokers) on the CPU
and appends to --out (JSON lines):
  {"type": "win", ...}      one per won game: the whole end state (jokers with editions / stickers / counters, consumables, vouchers, hand levels and plays, planets used,
                            money, deck make-up, boss, blinds skipped) and the game's seed and ACTIONS, so the win can be replayed exactly (tools/record_wins.py --replay FILE LINE);
  {"type": "summary", ...}  one per snapshot: games, win rate, blinds beaten, the jokers most held in wins against losses, the money left in wins (a warning for money routes).
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppEnv, OBS_KEYS      # noqa: E402
from balatro_rl.model import load_model                  # noqa: E402
from balatro_rl.sim.hands import HAND_NAMES              # noqa: E402
from balatro_rl.sim.jokers import JOKERS                 # noqa: E402

DROP = {"deck", "discard_pile", "hand", "pack_cards", "shop", "shop_packs", "full_deck", "ante_played", "flags"}


def make_env(a, model, seed):
    e = CppEnv(a.deck, a.stake)
    e.e.start_ante, e.e.start_money_p, e.e.start_money = a.start_ante, 1.0, a.money
    e.e.level_cap_p, e.e.level_cap_n, e.e.level_cap_ante = 1.0, a.cap_n, 99
    e.e.level_cap_mode, e.e.pack_restock = a.level_cap_mode, a.pack_restock
    e.e.scaled_start, e.e.scaled_shop, e.e.scaled_rounds = getattr(a, "scaled_start", 0), getattr(a, "scaled_shop", False), getattr(a, "scaled_rounds", 30)
    if a.pack_bias:
        names = ("arcana", "celestial", "standard", "buffoon", "spectral")
        w = {k: 1.0 for k in names}
        for part in a.pack_bias.split(","):
            k, v = part.split(":")
            w[k.strip()] = float(v)
        e.e.pack_bias = [w[k] for k in names]
    if a.ban and a.ban.lower() != "none":
        keys = list(JOKERS)
        e.e.banned = [keys.index(k) for k in a.ban.split(",")]
        e.e.ban_p = 1.0
    if getattr(model, "marg", 0):
        e.e.marginal = int(model.marg)
    return e


def play(a, model, seed0, games, greedy=False):
    envs = [make_env(a, model, seed0 + i) for i in range(games)]
    obs = [e.reset(seed0 + i) for i, e in enumerate(envs)]
    live = list(range(games))
    lines = [[] for _ in range(games)]
    out = [None] * games
    gen = torch.Generator().manual_seed(seed0)
    while live:
        batch = {k: torch.from_numpy(np.stack([obs[i][k] for i in live])).to(a.device) for k in OBS_KEYS}
        with torch.no_grad():
            logits, _ = model(batch)
        act = (logits.argmax(1) if greedy else torch.multinomial(torch.softmax(logits.float(), 1), 1, generator=gen).squeeze(1)).cpu().numpy()
        nxt = []
        for i, ai in zip(live, act):
            lines[i].append(int(ai))
            o, r, done, info = envs[i].step(int(ai))
            if done:
                out[i] = envs[i].game.dump()
            else:
                obs[i] = o
                nxt.append(i)
        live = nxt
    return out, lines


def deck_makeup(d):
    cards = d.get("full_deck", [])
    return {"size": len(cards)}


def record(d, line, seed, it, a):
    rec = {k: v for k, v in d.items() if k not in DROP}
    kinds = Counter()
    for x in line:
        kinds["play" if x < 218 else "discard" if x < 436 else "select" if x == 436 else "skip" if x == 437 else "other"] += 1
    rec.update({"type": "win" if d["state"] == "WON" else "loss", "iter": it, "seed": seed, "stake": a.stake, "n_actions": len(line), "action_kinds": dict(kinds),
                "actions": line if d["state"] == "WON" else None, "deck_size": len(d.get("full_deck", [])),
                "hand_levels_named": {n: l for n, l in zip(HAND_NAMES, d["hand_levels"]) if l > 1},
                "most_played": Counter({n: p for n, p in zip(HAND_NAMES, d["hand_played"]) if p}).most_common(3)})
    return rec


def snapshot(a, it):
    tmp = a.ckpt.replace(".pt", "_record_snap.pt")
    for _ in range(5):
        try:
            shutil.copy2(a.ckpt, tmp)
            return load_model(tmp, a.device)
        except Exception:
            time.sleep(3)
    raise RuntimeError("could not read the checkpoint")


def summary(a, it, ds, n):
    wins = [d for d in ds if d["state"] == "WON"]
    jw = Counter(j["key"] for d in wins for j in d["jokers"])
    jl = Counter(j["key"] for d in ds if d["state"] != "WON" for j in d["jokers"])
    return {"type": "summary", "iter": it, "games": n, "wins": len(wins), "win_rate": round(len(wins) / n, 4),
            "blinds_beyond_start": round(float(np.mean([d["furthest_blind"] - 3 * (a.start_ante - 1) for d in ds])), 3),
            "jokers_in_wins": jw.most_common(15), "jokers_in_losses": jl.most_common(10),
            "money_left_in_wins": round(float(np.mean([d["money"] for d in wins])), 1) if wins else None,
            "planet_uses_in_wins": round(float(np.mean([d.get("planet_uses", 0) for d in wins])), 2) if wins else None,
            "hand_levels_in_wins": {n_: round(float(l), 1) for n_, l in zip(HAND_NAMES, np.mean([d["hand_levels"] for d in wins], axis=0)) if l > 1.05} if wins else {}}


def replay(a):
    path, idx = a.replay
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    wins = [r for r in rows if r.get("type") == "win"]
    r = wins[int(idx)]
    model = None
    e = make_env(a, model, r["seed"])
    e.reset(r["seed"])
    for x in r["actions"]:
        _, _, done, _ = e.step(x)
    d = e.game.dump()
    print("replayed win", idx, "of", path, "->", d["state"], "jokers", [j["key"] for j in d["jokers"]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ppo_ante8_v1.pt")
    p.add_argument("--log", default="checkpoints/ppo_ante8_v1_log.jsonl")
    p.add_argument("--out", default="traces/ante8_builds.jsonl")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--every-iters", type=int, default=50)
    p.add_argument("--stake", default="WHITE")
    p.add_argument("--deck", default="RED")
    p.add_argument("--start-ante", type=int, default=8)
    p.add_argument("--money", type=int, default=1000)
    p.add_argument("--cap-n", type=int, default=4)
    p.add_argument("--ban", default="bootstraps,bull")
    p.add_argument("--level-cap-mode", type=int, default=0)
    p.add_argument("--pack-restock", type=int, default=0)
    p.add_argument("--pack-bias", default="")
    p.add_argument("--scaled-start", type=int, default=0)
    p.add_argument("--scaled-shop", action="store_true")
    p.add_argument("--scaled-rounds", type=int, default=30)
    p.add_argument("--seed0", type=int, default=555000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--once", action="store_true")
    p.add_argument("--replay", nargs=2, metavar=("FILE", "WIN_INDEX"))
    a = p.parse_args()
    if a.replay:
        replay(a)
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    last = -10 ** 9
    while True:
        rows = [json.loads(l) for l in open(a.log, encoding="utf-8") if l.strip().startswith("{")] if os.path.exists(a.log) else []
        it = rows[-1]["iter"] if rows else 0
        if it - last >= a.every_iters and os.path.exists(a.ckpt):
            ck_iter = torch.load(a.ckpt, map_location="cpu", weights_only=False)["extra"].get("iter", it)
            model = snapshot(a, ck_iter)
            t = time.time()
            ds, lines = play(a, model, a.seed0 + ck_iter * 1000, a.games)
            with open(a.out, "a", encoding="utf-8") as f:
                for i, (d, ln) in enumerate(zip(ds, lines)):
                    if d["state"] == "WON":
                        f.write(json.dumps(record(d, ln, a.seed0 + ck_iter * 1000 + i, ck_iter, a), default=str) + "\n")
                f.write(json.dumps(summary(a, ck_iter, ds, a.games), default=str) + "\n")
            s = summary(a, ck_iter, ds, a.games)
            print(f"update {ck_iter}: {s['wins']}/{a.games} wins ({100 * s['win_rate']:.1f}%), {time.time() - t:.0f}s; jokers in wins {s['jokers_in_wins'][:6]}", flush=True)
            last = it
        if a.once:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
