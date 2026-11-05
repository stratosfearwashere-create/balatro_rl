"""Where does a policy's build power come from, and what would raise it most? Counterfactuals at the ante-4 boss.

The policy plays greedy Gold games; at its first blind select of ante 4 each game's build is taken (jokers, hand levels, the deck as it is) and rebuilt in the engine facing the ante-4 boss
(VecEnv.set_build: scaling jokers at the counter they would have by then, the same for every variant, so only the change differs). The exact solver gives the best share of the target for:

    base          the build as it is
    levels=1      every hand level reset to 1                       -> what the levels contribute
    +3 levels     three more levels on the build's highest-level hand -> what a few planets would add
    -joker        each joker removed in turn (the mean and the largest drop) -> what the jokers contribute
    +xmult        the least useful joker replaced by Hologram (x-mult scaler) or Cavendish (x3), the better of the two
    +scaler       the least useful joker replaced by Green Joker / Ride the Bus, the better
    +slot         a sixth joker: the better of the strong ones above

Reported as the median change in log10 share against base, and how often the variant crosses the target when base does not.

    python tools/power_sources.py checkpoints/ppo_gold_v30_best.pt --games 300
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("BALATRO_DEV_ENGINE", "")
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS       # noqa: E402
from balatro_rl.env import A_SELECT                           # noqa: E402
from balatro_rl.model import load_model                      # noqa: E402
from balatro_rl.sim.jokers import JOKERS                     # noqa: E402
from joker_strength import counter_at                        # noqa: E402

KEYS = list(JOKERS)
XM = ["hologram", "cavendish"]
SC = ["green_joker", "ride_the_bus"]


def collect(model, n, seed0, stake, ante, device):
    """Plays n greedy games; returns a VecEnv whose env i holds game i at its first blind select of `ante` (or None if it died first)."""
    v = CppVecEnv(n, "RED", stake, seed0=seed0, threads=0)
    if getattr(model, "marg", 0):
        v.set_marginal(int(model.marg))
    keep = CppVecEnv(n, "RED", stake, seed0=1, threads=0)
    got = np.zeros(n, dtype=bool)
    dead = np.zeros(n, dtype=bool)
    for _ in range(8000):
        g, mask = np.asarray(v.obs["g"]), np.asarray(v.obs["mask"])
        at = [e for e in range(n) if not got[e] and not dead[e] and mask[e, A_SELECT] and int(round(g[e, 0] * 8)) == ante]
        if at:
            keep.copy_many(v, at, at)
            got[at] = True
        if (got | dead).all():
            break
        ob = {k: torch.from_numpy(np.asarray(v.obs[k])).to(device) for k in OBS_KEYS}
        with torch.no_grad():
            logits, _ = model(ob)
        act = logits.argmax(1).cpu().numpy()
        act[got | dead] = -1
        _, done, _ = v.step(act)
        dead |= done & ~got
    return keep, [e for e in range(n) if got[e]]


def shares(src, ids, builds, ante, node_cap):
    """builds: per id, a list of (keys, levels) variants -> [len(ids)][variants] best share at the ante's boss."""
    flat = [(i, kv) for i, vs in zip(ids, builds) for kv in vs]
    out = []
    w = CppVecEnv(len(flat), "RED", "GOLD", seed0=1, threads=0)
    w.copy_many(src, [i for i, _ in flat], list(range(len(flat))))
    w.set_auto_round(False, 1.5, 0)
    for k, (_, (keys, levels)) in enumerate(flat):
        w.set_build([k], ante, 2, [KEYS.index(x) for x in keys], [counter_at(x, ante) for x in keys], 0, levels, -1)
    act = np.full(w.n, A_SELECT, dtype=np.int64)
    w.step(act)
    r = w.solve_best(list(range(len(flat))), node_cap)[:, 0]
    pos = 0
    for vs in builds:
        out.append(list(r[pos:pos + len(vs)]))
        pos += len(vs)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed0", type=int, default=10000)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--ante", type=int, default=4)
    p.add_argument("--node-cap", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    model = load_model(a.checkpoint, a.device)
    keep, ids = collect(model, a.games, a.seed0, a.stake, a.ante, a.device)
    info = keep.cell_info(ids)
    lv = keep.hand_levels(ids)
    builds, meta = [], []
    for v, levels in zip(info, lv):
        nj = v[7]
        keys = [KEYS[v[8 + 3 * i]] for i in range(nj)]
        levels = list(levels)
        top = int(np.argmax(levels))
        up = list(levels)
        up[top] += 3
        variants = [(keys, levels), (keys, [1] * len(levels)), (keys, up)]
        for j in range(len(keys)):                                       # each joker removed
            variants.append((keys[:j] + keys[j + 1:], levels))
        meta.append((len(keys),))
        builds.append(variants)
    r1 = shares(keep, ids, builds, a.ante, a.node_cap)
    # the least useful joker: the one whose removal costs least
    builds2 = []
    for (keys, levels), res in zip([b[0] for b in builds], r1):
        nj = len(keys)
        drops = [np.log10(max(res[0], 1e-3)) - np.log10(max(res[3 + j], 1e-3)) for j in range(nj)]
        weak = int(np.argmin(drops)) if nj else -1
        vs = []
        for cand in XM + SC:
            ks = list(keys)
            if nj >= 5:
                ks[weak] = cand
            else:
                ks = ks + [cand]
            vs.append((ks, levels))
        for cand in XM + SC:
            vs.append((list(keys) + [cand], levels))                     # a sixth (or extra) slot
        builds2.append(vs)
    r2 = shares(keep, ids, builds2, a.ante, a.node_cap)
    L = lambda x: np.log10(max(x, 1e-3))
    rows = {"levels=1": [], "+3 levels": [], "-joker (mean)": [], "-joker (largest)": [], "+xmult (swap)": [], "+scaler (swap)": [], "+strong joker in an extra slot": []}
    cross = {k: [] for k in rows}
    for res, res2, (nj,) in zip(r1, r2, meta):
        b = L(res[0])
        def add(name, s):
            rows[name].append(L(s) - b)
            cross[name].append(float(s >= 1 and res[0] < 1))
        add("levels=1", res[1])
        add("+3 levels", res[2])
        if nj:
            d = [L(res[3 + j]) - b for j in range(nj)]
            rows["-joker (mean)"].append(float(np.mean(d)))
            rows["-joker (largest)"].append(float(np.min(d)))
            cross["-joker (mean)"].append(0.0)
            cross["-joker (largest)"].append(0.0)
        add("+xmult (swap)", max(res2[0], res2[1]))
        add("+scaler (swap)", max(res2[2], res2[3]))
        add("+strong joker in an extra slot", max(res2[4:8]))
    base = np.array([L(r[0]) for r in r1])
    print(f"{os.path.basename(a.checkpoint)}: {len(ids)} of {a.games} games reached ante {a.ante}; base power at the ante-{a.ante} boss: median {np.median(base):+.2f} (log10), "
          f"{100 * (base >= 0).mean():.0f}% clear; jokers held {np.mean([m[0] for m in meta]):.1f}, top hand level {np.mean([max(l) for l in lv]):.1f}")
    print(f"{'variant':34s} {'median change':>14s} {'p25':>7s} {'p75':>7s} {'newly clears':>13s}")
    for k, v in rows.items():
        v = np.array(v)
        print(f"{k:34s} {np.median(v):+14.2f} {np.percentile(v, 25):+7.2f} {np.percentile(v, 75):+7.2f} {100 * np.mean(cross[k]):12.0f}%")


if __name__ == "__main__":
    main()
