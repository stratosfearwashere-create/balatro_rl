"""What does a late-game sandbox network win with? Samples games from a checkpoint with the sandbox's start (--start-ante, money, planet cap) and prints the end states of the wins
against the losses: jokers, hand levels, planets used, consumables, vouchers, money left, how the blinds were got through (skips).

    python tools/sandbox_wins.py checkpoints/ppo_ante8_v1.pt --games 400 --stake WHITE
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppEnv, OBS_KEYS      # noqa: E402
from balatro_rl.model import load_model                  # noqa: E402
from balatro_rl.sim.hands import HAND_NAMES              # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--games", type=int, default=300)
p.add_argument("--stake", default="WHITE")
p.add_argument("--deck", default="RED")
p.add_argument("--start-ante", type=int, default=8)
p.add_argument("--money", type=int, default=1000)
p.add_argument("--cap-n", type=int, default=4)
p.add_argument("--seed0", type=int, default=777000)
p.add_argument("--level-cap-mode", type=int, default=0)
p.add_argument("--pack-restock", type=int, default=0)
p.add_argument("--pack-bias", default="")
p.add_argument("--greedy", action="store_true")
p.add_argument("--device", default="cpu")
a = p.parse_args()
model = load_model(a.checkpoint, a.device)
envs, obs = [], []
for i in range(a.games):
    e = CppEnv(a.deck, a.stake)
    e.e.start_ante, e.e.start_money_p, e.e.start_money = a.start_ante, 1.0, a.money
    e.e.level_cap_p, e.e.level_cap_n, e.e.level_cap_ante = 1.0, a.cap_n, 99
    e.e.level_cap_mode, e.e.pack_restock = a.level_cap_mode, a.pack_restock
    if a.pack_bias:
        names = ("arcana", "celestial", "standard", "buffoon", "spectral")
        w = {k: 1.0 for k in names}
        for part in a.pack_bias.split(","):
            k, v = part.split(":")
            w[k.strip()] = float(v)
        e.e.pack_bias = [w[k] for k in names]
    if getattr(model, "marg", 0):
        e.e.marginal = int(model.marg)
    envs.append(e)
    obs.append(e.reset(a.seed0 + i))
live = list(range(a.games))
acts = [Counter() for _ in range(a.games)]
out = [None] * a.games
gen = torch.Generator().manual_seed(0)
KIND = lambda x: ("play" if x < 218 else "discard" if x < 436 else {436: "select", 437: "skip", 438: "reroll_boss"}.get(x) or
                  ("buy" if x < 443 else "buy_pack" if x < 445 else "voucher" if x < 447 else "reroll" if x == 447 else "leave" if x == 448 else
                   "sell_joker" if x < 457 else "sell_cons" if x < 460 else "use" if x < 463 else "pick" if x < 468 else "pack_skip" if x == 468 else "swap" if x < 476 else "replace"))
while live:
    batch = {k: torch.from_numpy(np.stack([obs[i][k] for i in live])).to(a.device) for k in OBS_KEYS}
    with torch.no_grad():
        logits, _ = model(batch)
    if a.greedy:
        act = logits.argmax(1).cpu().numpy()
    else:
        act = torch.multinomial(torch.softmax(logits.float(), 1), 1, generator=gen).squeeze(1).cpu().numpy()
    nxt = []
    for i, ai in zip(live, act):
        acts[i][KIND(int(ai))] += 1
        o, r, done, info = envs[i].step(int(ai))
        if done:
            d = envs[i].game.dump()
            d["acts"] = dict(acts[i])
            out[i] = d
        else:
            obs[i] = o
            nxt.append(i)
    live = nxt
wins = [d for d in out if d["state"] == "WON"]
print(f"{a.games} games ({'greedy' if a.greedy else 'sampled'}), stake {a.stake}, start ante {a.start_ante}: {len(wins)} wins ({100 * len(wins) / a.games:.1f}%), "
      f"mean blinds beaten beyond the start {np.mean([d['furthest_blind'] - 3 * (a.start_ante - 1) for d in out]):.2f}")


def summarize(ds, name):
    if not ds:
        print(f"\n{name}: none")
        return
    jc = Counter(j["key"] for d in ds for j in d["jokers"])
    print(f"\n{name} ({len(ds)} games)")
    print("  jokers held at the end:", jc.most_common(14))
    print("  jokers per game %.2f; money left %.0f; skips %.2f; rerolls %.1f; packs bought %.1f; vouchers %.1f; consumables used %.1f" % (
        np.mean([len(d["jokers"]) for d in ds]), np.mean([d["money"] for d in ds]), np.mean([d["acts"].get("skip", 0) for d in ds]),
        np.mean([d["acts"].get("reroll", 0) for d in ds]), np.mean([d["acts"].get("buy_pack", 0) for d in ds]),
        np.mean([bin(d.get("vouchers", 0)).count("1") if isinstance(d.get("vouchers"), int) else len(d.get("vouchers", [])) for d in ds]),
        np.mean([d["acts"].get("use", 0) for d in ds])))
    lv = np.mean([d["hand_levels"] for d in ds], axis=0)
    print("  mean hand levels:", {n: round(float(l), 1) for n, l in zip(HAND_NAMES, lv) if l > 1.05})
    print("  planet uses:", np.mean([d.get("planet_uses", 0) for d in ds]).round(2), "| deck size", np.mean([len(d["full_deck"]) for d in ds]).round(1),
          "| most played:", Counter({n: sum(d["hand_played"][i] for d in ds) for i, n in enumerate(HAND_NAMES)}).most_common(3))
    print("  boss:", Counter(d["boss"] for d in ds).most_common(3))


summarize(wins, "WINS")
summarize([d for d in out if d["state"] != "WON"], "LOSSES")
