"""Do the multiplier jokers the policy holds actually fire? At the blind selects of greedy games (solver rounds), for each held xMult joker
(joker_desc 'xmult'): the build's fast power (redrawn hands, best single play by the exact scorer) with the joker vs on a copy where it was sold.
ratio = power with / without: ~1.0 means the joker adds nothing to the best plays this deck can make (its condition is not met).

    python tools/xmult_activation.py --ckpt checkpoints/ppo_gold_v48.pt --games 200 --catalogue 0.5
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl import env as E                                  # noqa: E402
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS            # noqa: E402
from balatro_rl.joker_desc import FEATURES, item_descriptors     # noqa: E402
from balatro_rl.model import load_model                          # noqa: E402
from balatro_rl.runeval import Cols                              # noqa: E402
from balatro_rl.sim.items import VOCAB                           # noqa: E402

D = item_descriptors()
XM = {i for i in range(len(VOCAB)) if VOCAB[i].startswith("j_") and D[i, FEATURES.index("xmult")] > 0.5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v48.pt")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--catalogue", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    net = load_model(a.ckpt, "cpu").eval()

    def mk(n, seed):
        v = CppVecEnv(n, "RED", "GOLD", seed0=seed, threads=a.threads)
        v.set_marginal(int(getattr(net, "marg", 0)))
        v.set_autoplay(True, 400, True)
        v.v.set_solver_tarots(True, 3)
        if a.catalogue:
            v.v.set_catalogue(a.catalogue)
        v.v.restart()
        return v
    v = mk(a.games, 9_300_000)
    S = mk(64, 77)
    fin = np.zeros(a.games, bool)
    ratios = defaultdict(list)
    held = defaultdict(int)
    for _ in range(4000):
        o = {k: np.asarray(v.obs[k]) for k in OBS_KEYS}
        st = np.argmax(o["g"][:, Cols.STATE:Cols.STATE + 4], 1)
        pairs = []                                              # (game, slot, joker id) in a shop, where the joker can be sold
        for i in np.flatnonzero((st == 2) & ~fin & o["mask"][:, E.A_LEAVE]):
            for s, j in enumerate(o["jid"][i]):
                if int(j) in XM and o["mask"][i, E.A_SELL_J + s]:
                    pairs.append((i, s, int(j)))
        for c0 in range(0, len(pairs), S.n // 2):
            ch = pairs[c0:c0 + S.n // 2]
            m = len(ch)
            S.copy_many(v, [i for i, _, _ in ch] * 2, list(range(2 * m)))
            S.set_autoplay(False, 400, False)                    # (enter the round without playing it: fast_power needs the hand dealt)
            act = np.full(S.n, -1, dtype=np.int64)
            act[m:2 * m] = [E.A_SELL_J + s for _, s, _ in ch]      # the second copy sells the joker
            S.step(act)
            for a_ in (E.A_LEAVE, E.A_SELECT):
                act = np.full(S.n, -1, dtype=np.int64)
                act[:2 * m] = a_
                S.step(act)
            S.set_autoplay(True, 400, True)
            ok = np.argmax(np.asarray(S.obs["g"])[:2 * m, Cols.STATE:Cols.STATE + 4], 1) == 1
            f = S.fast_power(list(range(2 * m)), 8)
            for k, (_, _, j) in enumerate(ch):
                if not (ok[k] and ok[m + k]):
                    continue
                w, wo = f[k, 2], f[m + k, 2]
                if wo > 0:
                    ratios[VOCAB[j][2:]].append(w / wo)
                held[VOCAB[j][2:]] += 1
        with torch.no_grad():
            lg, _ = net({k: torch.from_numpy(x) for k, x in o.items()})
        _, d, _ = v.step(np.where(fin, -1, lg.argmax(1).numpy()))
        fin |= d
        if fin.all():
            break
    allr = np.concatenate([np.array(x) for x in ratios.values()]) if ratios else np.zeros(1)
    print(f"{a.games} games: {sum(held.values())} shop visits holding a multiplier joker; median power with / without it {np.median(allr):.2f}; "
          f"it adds under 10% in {100 * np.mean(allr < 1.1):.0f}% of them")
    for k, x in sorted(ratios.items(), key=lambda t: -len(t[1])):
        x = np.array(x)
        print(f"  {k:20s} held in {held[k]:4d} shop visits: median x{np.median(x):.2f}, adds <10% in {100 * np.mean(x < 1.1):3.0f}%")


if __name__ == "__main__":
    main()
