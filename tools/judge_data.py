"""Training data for the build judge (network 1) and the scaling model (network 2).

Builds come from the engine's scenario generator (jokers with counters as if owned for many rounds, a deck with a theme and random edits, hand levels piled on one or two hand
types, money), at every ante, with a build power that grows with the ante so that builds sit around the pass / fail line of that ante's blinds; they are screened by the
engine's exact expected best-play score against the blind's target. Each accepted build is labelled on each of the three blinds of its ante: the scripted round player (it
sees only its hand: Env::scripted_action) plays the blind R times against different draws.

    python tools/judge_data.py --out-dir data/judge1 --builds 400000 --reps 8

One .npz shard per batch: the observation at the blind select (global vector with the engine's marginal-value columns, jokers with their counters, consumables), and per sample
  pass      share of the R rounds cleared                      ratio   mean of the best share of the target reached (log10)
  hands     hands played / discards used / money gained (means)   dc    mean change of each joker slot's counter over the round
  ante, blind, build_seed, n (R)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/judge1")
    p.add_argument("--builds", type=int, default=400_000, help="accepted builds (each is labelled on 3 blinds)")
    p.add_argument("--reps", type=int, default=8, help="draws per (build, blind)")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--pace", type=float, default=1.5)
    p.add_argument("--modes", default="0,1", help="the scripted player's modes to label each build with (0: score most; 1: scaler-friendly)")
    p.add_argument("--antes", default="1,2,3,4,5,6,7,8")
    p.add_argument("--ante-weights", default="1,1,1,1,1,1,1.5,2.5")
    p.add_argument("--band", type=float, default=1.0, help="builds whose log10(expected best play / target) is within +-band are all kept, the others with --keep-out")
    p.add_argument("--keep-out", type=float, default=0.15)
    p.add_argument("--level-conc", type=float, default=0.7)
    p.add_argument("--seed0", type=int, default=100_000_000)
    p.add_argument("--dev-engine", default="", help="a directory with _engine_dev (a development build of the engine)")
    a = p.parse_args()
    if a.dev_engine:
        sys.path.insert(0, a.dev_engine)
        import _engine_dev
        import balatro_rl.sim as S
        sys.modules["balatro_rl.sim._engine"] = _engine_dev
        S._engine = _engine_dev
    from balatro_rl.engine_env import CppVecEnv
    from balatro_rl.env import global_size, A_SELECT, F_JOKER_W
    from balatro_rl.marginal import MARG_G, POT_G

    os.makedirs(a.out_dir, exist_ok=True)
    antes = [int(x) for x in a.antes.split(",")]
    w = np.array([float(x) for x in a.ante_weights.split(",")], dtype=np.float64)
    w = w / w.sum()
    rng = np.random.default_rng(a.seed0)
    N = a.batch
    base_col = global_size() - MARG_G - POT_G
    v = CppVecEnv(N, "RED", "GOLD", seed0=a.seed0, threads=0)
    v.set_marginal(1)
    accepted, shard, t0 = 0, 0, time.time()
    stats = {x: [0, 0] for x in antes}                           # candidates, accepted
    meta = {"reps": a.reps, "pace": a.pace, "band": a.band, "level_conc": a.level_conc, "g_dim": global_size(), "jf_dim": F_JOKER_W}
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"))
    next_seed = a.seed0 + 1
    while accepted < a.builds:
        ante = int(rng.choice(antes, p=w))
        hi = min(24.0, 3.0 + 2.6 * ante)
        v.set_scenario(1.0, 1.0, 0.0, 0.0, ante, ante, 3, 0.5)
        v.set_power(1.0, hi)
        keep_seeds = []
        while len(keep_seeds) < N:                                # screen candidates until a batch is accepted
            seeds = np.arange(next_seed, next_seed + N, dtype=np.int64)
            next_seed += N
            v.set_build_seeds(seeds, -1, a.level_conc)
            v.v.restart()
            g = np.asarray(v.obs["g"])
            ratio = 7.0 * g[:, base_col] - 6.0 * g[:, 8]
            ok = (np.abs(ratio) <= a.band) | (rng.random(N) < a.keep_out)
            stats[ante][0] += N
            stats[ante][1] += int(ok.sum())
            keep_seeds += seeds[ok].tolist()
        seeds = np.array(keep_seeds[:N], dtype=np.int64)
        out = {k: [] for k in ("g", "jid", "jf", "cid", "pass", "ratio", "hands", "discards", "money", "dc", "ante", "blind", "seed", "mode")}
        modes = [int(x) for x in a.modes.split(",")]
        for blind in (0, 1, 2):
            v.set_build_seeds(seeds, blind, a.level_conc)
            v.v.restart()
            g0 = np.array(v.obs["g"], dtype=np.float32); jid0 = np.array(v.obs["jid"], dtype=np.int16)
            jf0 = np.array(v.obs["jf"])[:, :, :F_JOKER_W].astype(np.float16); cid0 = np.array(v.obs["cid"], dtype=np.int16)
            for mode in modes:
                v.set_build_seeds(seeds, blind, a.level_conc)
                v.v.restart()                                          # (the same build and the same draws for each mode)
                v.set_auto_round(True, a.pace, mode)
                cl = np.zeros(N); rt = np.zeros(N); hd = np.zeros(N); dd = np.zeros(N); mn = np.zeros(N); dc = np.zeros((N, 8))
                for _ in range(a.reps):
                    v.step_split(np.full(N, A_SELECT))
                    r = v.round_report()
                    cl += r["cleared"]; rt += np.log10(np.maximum(r["ratio"], 1e-3)); hd += r["hands"]; dd += r["discards"]; mn += r["money"]; dc += r["c1"] - r["c0"]
                v.set_auto_round(False, a.pace)
                k = float(a.reps)
                for key, val in (("g", g0), ("jid", jid0), ("jf", jf0), ("cid", cid0)):
                    out[key].append(val)
                out["pass"].append((cl / k).astype(np.float32)); out["ratio"].append((rt / k).astype(np.float32)); out["hands"].append((hd / k).astype(np.float32))
                out["discards"].append((dd / k).astype(np.float32)); out["money"].append((mn / k).astype(np.float32)); out["dc"].append((dc / k).astype(np.float32))
                out["ante"].append(np.full(N, ante, dtype=np.int8)); out["blind"].append(np.full(N, blind, dtype=np.int8)); out["seed"].append(seeds)
                out["mode"].append(np.full(N, mode, dtype=np.int8))
        np.savez_compressed(os.path.join(a.out_dir, f"shard{shard:05d}.npz"), **{k: np.concatenate(x) for k, x in out.items()})
        shard += 1
        accepted += N
        el = time.time() - t0
        print(json.dumps({"builds": accepted, "samples": 3 * len(modes) * accepted, "min": round(el / 60, 1), "builds_per_s": round(accepted / el, 1), "ante": ante,
                          "pass_mean_by_blind": [round(float(x.mean()), 3) for x in out["pass"]], "accept": {str(k): round(s[1] / max(1, s[0]), 2) for k, s in stats.items() if s[0]}}), flush=True)


if __name__ == "__main__":
    main()
