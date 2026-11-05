"""Build archetypes, from the jokers a game holds, and a check that they separate real builds.

    scale   at least one scaling joker (its counter grows), a copy joker (Blueprint / Brainstorm) or a multiplier joker (xMult): the build gets stronger faster than the blinds' targets, or at least
            multiplies
    econ    no scale, but at least one economy joker, or $25+ banked (interest) at the check
    flat    the rest: chips / plus-mult jokers, a few conditional ones

classify() is the label the strategy-conditioned policy is trained on; checked(): the labels are read at the first blind select of a given ante (default 4) from the games a policy plays, and each
archetype's outcome (blinds beaten, ante reached, win rate) is reported. If the archetypes do not differ in outcome the labels carry no information.

    python tools/archetypes.py checkpoints/ppo_gold_v30.pt --games 256 --ante 4
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("BALATRO_DEV_ENGINE", "")
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS      # noqa: E402
from balatro_rl.env import item_id                           # noqa: E402
from balatro_rl.model import load_model                      # noqa: E402
from balatro_rl.sim.jokers import JOKERS                     # noqa: E402
from seed_report import role_of                              # noqa: E402

ITEM2KEY = {item_id("j_" + k): k for k in JOKERS}
ARCHETYPES = ("flat", "econ", "scale")


def classify(keys, money=0):
    roles = [role_of(k) for k in keys]
    if any(r in ("scaler", "xmult", "copy") for r in roles):
        return 2
    if any(r == "econ" for r in roles) or money >= 25:
        return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--ante", type=int, default=4, help="the label is read at the first blind select of this ante")
    p.add_argument("--seed0", type=int, default=920000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    model = load_model(a.checkpoint, a.device)
    N = a.games
    venv = CppVecEnv(N, "RED", a.stake, seed0=a.seed0, threads=0)
    if getattr(model, "marg", 0):
        venv.set_marginal(int(model.marg))
    label = np.full(N, -1)
    snap = [None] * N
    blinds = np.full(N, -1)
    won = np.zeros(N, dtype=bool)
    gen = torch.Generator(device=a.device).manual_seed(0)
    for _ in range(a.max_steps):
        if (blinds >= 0).all():
            break
        ob = {k: torch.from_numpy(np.asarray(venv.obs[k])).to(a.device) for k in OBS_KEYS}
        g, mask, jid = np.asarray(venv.obs["g"]), np.asarray(venv.obs["mask"]), np.asarray(venv.obs["jid"])
        for e in range(N):
            if blinds[e] < 0 and label[e] < 0 and mask[e, 436] and int(round(g[e, 0] * 8)) == a.ante:
                keys = [ITEM2KEY[int(x)] for x in jid[e] if int(x) in ITEM2KEY]
                money = int(round(g[e, 17] * 50))
                label[e] = classify(keys, money)
                snap[e] = (keys, money)
        with torch.no_grad():
            logits, _ = model(ob)
        act = torch.multinomial(torch.softmax(logits.float(), 1), 1, generator=gen).squeeze(1).cpu().numpy()
        _, done, info = venv.step(act)
        for e in np.flatnonzero(done):
            if blinds[e] < 0:
                blinds[e] = info["blinds"][e]
                won[e] = info["won"][e]
    reached = label >= 0
    print(f"{a.checkpoint}: {N} games; {int(reached.sum())} reached ante {a.ante}'s first blind select ({100 * reached.mean():.0f}%)")
    print(f"{'archetype':10s} {'games':>6s} {'share':>6s} {'blinds':>7s} {'sd':>5s} {'>=ante 6':>9s} {'wins':>5s}")
    for k, name in enumerate(ARCHETYPES):
        m = label == k
        if m.sum() == 0:
            print(f"{name:10s} {0:6d}")
            continue
        b = blinds[m]
        print(f"{name:10s} {int(m.sum()):6d} {100 * m.sum() / max(reached.sum(), 1):5.0f}% {b.mean():7.2f} {b.std():5.2f} {100 * (b >= 15).mean():8.0f}% {int(won[m].sum()):5d}")
    print("common jokers by archetype:")
    for k, name in enumerate(ARCHETYPES):
        cnt = {}
        for e in np.flatnonzero(label == k):
            for j in snap[e][0]:
                cnt[j] = cnt.get(j, 0) + 1
        print(f"  {name}: " + ", ".join(f"{j} {c}" for j, c in sorted(cnt.items(), key=lambda t: -t[1])[:8]))


if __name__ == "__main__":
    main()
