"""Speed of the C++ engine against the Python environment, on the same policy.

    python tools/bench_engine.py                 # everything
    python tools/bench_engine.py --seconds 5 --stake WHITE

Policy (works from observations alone, so both engines can run it): in a round the highest-scoring play, or a
discard when no play clears and discards are left; outside rounds a random legal action. It reaches the
middle antes, so the mix of round and shop decisions is realistic.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from balatro_rl import env as E                                  # noqa: E402
from balatro_rl.sim import _engine                               # noqa: E402

RATIO = E.RATIO_COL
IN_ROUND = 5                    # obs["g"] column: state == SELECTING_HAND


def policy(obs: dict, rng) -> np.ndarray:
    """Batched: obs arrays have a leading [n] axis."""
    m = obs["mask"]
    plays = m[:, :E.N_SUB]
    ratio = np.where(plays, obs["af"][:, :E.N_SUB, RATIO], -1.0)
    best = ratio.argmax(1)
    clears = ratio.max(1) >= 1.0
    discs = m[:, E.A_DISC:E.A_DISC + E.N_SUB]
    can_disc = discs.any(1)
    disc = E.A_DISC + np.argmax(rng.random(discs.shape) * discs, 1)
    in_round = plays.any(1) & (obs["g"][:, IN_ROUND] > 0)
    rand = np.argmax(rng.random(m.shape) * m, 1)
    a = np.where(in_round, np.where(clears | ~can_disc, best, disc), rand)
    return a


def one(obs: dict) -> dict:
    return {k: np.asarray(v)[None] for k, v in obs.items()}


def bench_python(stake: str, seconds: float):
    env = E.BalatroEnv("RED", stake)
    rng = np.random.default_rng(0)
    obs = env.reset(1)
    n = games = 0
    seed = 1
    t_step = 0.0
    t0 = time.perf_counter()
    rounds = 0
    while time.perf_counter() - t0 < seconds:
        a = int(policy(one(obs), rng)[0])
        rounds += obs["g"][IN_ROUND] > 0
        t = time.perf_counter()
        obs, r, done, _ = env.step(a)
        t_step += time.perf_counter() - t
        n += 1
        if done:
            games += 1
            seed += 1
            obs = env.reset(seed)
    return n / t_step, n / max(games, 1), rounds / n


def bench_single(stake: str, seconds: float):
    env = _engine.Env("RED", stake)
    rng = np.random.default_rng(0)
    obs = env.reset(1)
    n = games = 0
    seed = 1
    t_step = 0.0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = int(policy(one(obs), rng)[0])
        t = time.perf_counter()
        obs, r, done, _ = env.step(a)
        t_step += time.perf_counter() - t
        n += 1
        if done:
            games += 1
            seed += 1
            obs = env.reset(seed)
    return n / t_step, n / max(games, 1)


def bench_vec(stake: str, n_envs: int, threads: int, seconds: float):
    v = _engine.VecEnv(n_envs, "RED", stake, 0, threads)
    obs = v.obs()
    rng = np.random.default_rng(0)
    steps = eps = 0
    t_step = 0.0
    rounds = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = policy(obs, rng)
        rounds += int((obs["g"][:, IN_ROUND] > 0).sum())
        t = time.perf_counter()
        rew, done, ante, won, blinds = v.step(a)
        t_step += time.perf_counter() - t
        steps += n_envs
        eps += int(done.sum())
    return steps / t_step, steps / max(eps, 1), rounds / steps, eps / t_step


def components(stake: str):
    """Microseconds per call of the parts of one step, on a state in a round and on a shop state."""
    env = _engine.Env("RED", stake)
    rng = np.random.default_rng(0)
    obs = env.reset(3)
    out = {}
    seen = set()
    for _ in range(400):
        st = env.game.state
        if st in ("SELECTING_HAND", "SHOP") and st not in seen:
            seen.add(st)
            n = 2000
            row = {"encode": 1e6 * env.bench_encode(n) / n, "features": 1e6 * env.bench_features(n) / n}
            if st == "SELECTING_HAND":
                row["candidates (218 predictions)"] = 1e6 * env.bench_candidates(n) / n
            out[st] = row
        a = int(policy(one(obs), rng)[0])
        obs, r, done, _ = env.step(a)
        if done:
            obs = env.reset(4)
    scratch = _engine.Env("RED", stake)
    out["reset (new game + first observation)"] = {"": 1e6 * scratch.bench_reset(2000) / 2000}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stake", default="WHITE")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--no-python", action="store_true")
    a = p.parse_args()
    cores = os.cpu_count() or 1
    print(f"stake {a.stake}, {cores} logical cores")
    if not a.no_python:
        sps, spg, share = bench_python(a.stake, a.seconds)
        print(f"Python env, 1 core:   {sps:9,.0f} steps/s   ({1e6 / sps:6.1f} us/step, {spg:.0f} steps/game, "
              f"{100 * share:.0f}% of steps in a round)")
        py = sps
    else:
        py = None
    sps, spg = bench_single(a.stake, a.seconds)
    print(f"C++ Env, 1 core:      {sps:9,.0f} steps/s   ({1e6 / sps:6.1f} us/step, incl. building the Python obs dict)"
          + (f"   x{sps / py:.1f}" if py else ""))
    for n_envs, threads in ((256, 1), (256, 2), (512, 4), (1024, 8), (1024, cores)):
        if threads > cores:
            continue
        sps, spg, share, gps = bench_vec(a.stake, n_envs, threads, a.seconds)
        print(f"C++ VecEnv n={n_envs:4d} threads={threads:2d}: {sps:9,.0f} steps/s   ({spg:.0f} steps/game, "
              f"{gps:,.0f} games/s, {100 * share:.0f}% in a round)" + (f"   x{sps / py:.1f}" if py else ""))
    print("components (microseconds per call, one core):")
    for st, row in components(a.stake).items():
        print(f"  {st}: " + ", ".join(f"{k} {v:.1f}".strip() for k, v in row.items()))


if __name__ == "__main__":
    main()
