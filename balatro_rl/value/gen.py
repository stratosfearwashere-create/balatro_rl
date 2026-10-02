"""Play games with a fixed agent and write build records with outcome labels (records.py).

    python -m balatro_rl.value.gen --games 50000 --workers 7 --out checkpoints/value_data
    python -m balatro_rl.value.gen --games 200 --workers 2 --out checkpoints/value_smoke   # a check

Stakes are drawn per game from --stakes (name:weight,...). The agent is the no-search graded prior
(the strongest fixed policy); --explore-frac games add Gumbel noise at the root for coverage. Output: one
.npz per worker chunk (records + labels) and a games.jsonl with one line per game (seed, stake, outcome).
Resumable: chunks already written are skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

from ..rewards.config import RewardConfig
from ..rewards.targets import blind_index
from .records import GameRecorder, pack

DEFAULT_STAKES = "WHITE:4,RED:1,GREEN:1,BLACK:1,BLUE:1,PURPLE:1,ORANGE:1,GOLD:2"
DEFAULT_CFG = {"shop_prior": "graded", "shop": {"rule_arcana": True}}
SEED0 = 1_000_000


def parse_stakes(s: str) -> tuple[list[str], list[float]]:
    names, weights = [], []
    for part in s.split(","):
        n, _, w = part.partition(":")
        names.append(n.strip().upper())
        weights.append(float(w) if w else 1.0)
    return names, weights


def play_recorded(agent, seed: int, deck: str, stake: str, explore: bool, game_id: int):
    """One game; returns (labelled rows, info)."""
    from ..az.world import World
    from ..sim.game import Game
    w = World(Game(seed=seed, deck_type=deck, stake=stake))
    agent.reseed(seed * 7919 + 17)
    rec = GameRecorder(agent.potential, game_id)
    while not w.done:
        g = w.g
        if g.state == "BLIND_SELECT" or g.state in ("SHOP", "PACK"):
            rec.maybe(w)
        d = agent.decide(w, explore=explore)
        if d.action is None:
            g.state = "GAME_OVER"
            break
        prev_state, prev_ante, prev_blind = g.state, g.ante, blind_index(g)
        w.step(d.action)
        if prev_state != "SELECTING_HAND" and g.state == "SELECTING_HAND":
            rec.maybe(w, blind_start=True)
        if prev_state == "SELECTING_HAND" and g.state != "SELECTING_HAND" and prev_blind % 3 == 2:
            rec.boss(prev_ante, g.state in ("SHOP", "WON", "PACK", "BLIND_SELECT"))
    g = w.g
    rows = rec.finish(g)
    info = {"seed": seed, "stake": stake, "deck": deck, "won": g.state == "WON", "blinds": g.furthest_blind,
            "ante": g.ante, "steps": w.steps, "explore": explore, "records": len(rows)}
    return rows, info


def _worker(args):
    import torch
    torch.set_num_threads(1)
    from ..az.train import _make_agent
    model, cfg, pot, seeds, stakes, weights, explore_frac, deck, out_path = args
    if os.path.exists(out_path):
        return []
    agent = _make_agent(model, False, seeds[0], cfg, pot)
    rows, infos = [], []
    for s in seeds:
        rng = random.Random(s)
        stake = rng.choices(stakes, weights=weights)[0]
        explore = rng.random() < explore_frac
        r, info = play_recorded(agent, s, deck, stake, explore, s)
        rows += r
        infos.append(info)
    if rows:
        np.savez_compressed(out_path + ".tmp.npz", **pack(rows))
        os.replace(out_path + ".tmp.npz", out_path)
    return infos


def generate(a):
    os.makedirs(a.out, exist_ok=True)
    if a.log:                                   # a detached run writes its own log (no inherited pipes)
        import sys
        sys.stdout = sys.stderr = open(a.log, "a", buffering=1)
    stakes, weights = parse_stakes(a.stakes)
    rcfg = RewardConfig.load(a.reward_config)
    pot = dict(rcfg.potential.__dict__)
    cfg = json.loads(a.cfg) if a.cfg else DEFAULT_CFG
    seeds = list(range(a.seed0, a.seed0 + a.games))
    chunks = [seeds[i:i + a.chunk] for i in range(0, len(seeds), a.chunk)]
    jobs = [(a.model, cfg, pot, c, stakes, weights, a.explore_frac, a.deck,
             os.path.join(a.out, f"chunk{k:05d}.npz")) for k, c in enumerate(chunks)]
    todo = [j for j in jobs if not os.path.exists(j[-1])]
    print(f"{len(seeds)} games in {len(chunks)} chunks ({len(todo)} to do), {a.workers} workers, stakes {dict(zip(stakes, weights))}",
          flush=True)
    t = time.time()
    done = 0
    log = open(os.path.join(a.out, "games.jsonl"), "a")
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for infos in pool.imap_unordered(_worker, todo):
            for i in infos:
                log.write(json.dumps(i) + "\n")
            log.flush()
            done += len(infos)
            if infos:
                rate = done / (time.time() - t)
                left = (len(todo) * a.chunk - done) / max(rate, 1e-9) / 60
                wins = sum(i["won"] for i in infos)
                print(f"  {done} games, {rate * 3600:.0f}/h, ~{left:.0f} min left; last chunk: {wins}/{len(infos)} won, "
                      f"{np.mean([i['blinds'] for i in infos]):.1f} blinds, {np.mean([i['records'] for i in infos]):.0f} records/game",
                      flush=True)
    log.close()
    print(f"done in {(time.time() - t) / 60:.1f} min -> {a.out}")


def load(out_dir: str) -> dict:
    """All chunks of a data directory as one dict of arrays."""
    from .records import concat
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(out_dir, "chunk*.npz")))]
    if not parts:
        raise SystemExit(f"no chunks in {out_dir}")
    return concat(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--chunk", type=int, default=50, help="games per output file")
    p.add_argument("--seed0", type=int, default=SEED0)
    p.add_argument("--stakes", default=DEFAULT_STAKES)
    p.add_argument("--deck", default="RED")
    p.add_argument("--explore-frac", type=float, default=0.2)
    p.add_argument("--model", default="none", help="network checkpoint, or 'none' (priors only)")
    p.add_argument("--cfg", default="", help="AgentConfig overrides as JSON (default: the graded prior + arcana)")
    p.add_argument("--reward-config", default=None)
    p.add_argument("--out", default="checkpoints/value_data")
    p.add_argument("--log", default="", help="append progress to this file instead of stdout")
    generate(p.parse_args())


if __name__ == "__main__":
    main()
