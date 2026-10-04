"""Play games with a fixed agent and write build records with outcome labels (records.py).

    python -m balatro_rl.value.gen --games 50000 --workers 7 --out checkpoints/value_data
    python -m balatro_rl.value.gen --games 200 --workers 2 --out checkpoints/value_smoke   # a check

Stakes are drawn per game from --stakes (name:weight,...). The agent is the no-search graded prior
(the strongest fixed policy); --explore-frac games add Gumbel noise at the root for coverage. Output: one
.npz per worker chunk (records + labels) and a games.jsonl with one line per game (seed, stake, outcome).
Resumable: chunks already written are skipped.

Branch rollouts (--branch-k K --branch-prob p): in a share p of the games, at one decision outside a round
(the t-th, t uniform in 1..--branch-span), the K best options by the agent's policy are each carried out on a
redrawn copy and played to the end by the same agent. Their records are ordinary labelled records (more
data), and the first record of each branch carries the decision's group id (records.py `group`): the
trainer ranks a group's records against each other, a direct signal about one decision that whole-game
outcomes give only weakly. Costs about K/2 extra games per branched game.
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


def event(w, a) -> dict | None:
    """What a decision outside a round (or a consumable use) did, as plain facts for stats.py; None for plays
    and discards. Read before the step: the indices refer to the state the agent saw."""
    g = w.g
    k = a.kind
    e = {"ante": g.ante, "blind": min(g.blind_idx, 2), "step": w.steps, "money": g.money, "kind": k,
         "in_round": g.state == "SELECTING_HAND"}
    if k in ("play", "discard"):
        return None
    if k == "buy" and 0 <= a.idx < len(g.shop):
        it = g.shop[a.idx]
        e["key"], e["cost"] = it.key, it.cost
        if it.joker is not None:
            e["edition"] = it.joker.edition
    elif k == "buy_pack" and 0 <= a.idx < len(g.shop_packs):
        e["key"], e["cost"] = g.shop_packs[a.idx].key, g.shop_packs[a.idx].cost
    elif k == "voucher" and 0 <= a.idx < len(g.shop_vouchers):
        e["key"], e["cost"] = g.shop_vouchers[a.idx].key, g.shop_vouchers[a.idx].cost
    elif k == "sell_joker" and 0 <= a.idx < len(g.jokers):
        j = g.jokers[a.idx]
        e["key"], e["edition"], e["value"] = f"j_{j.key}", j.edition, j.sell_value()
    elif k == "sell_cons" and 0 <= a.idx < len(g.consumables):
        e["key"] = g.consumables[a.idx].key
    elif k == "use" and 0 <= a.idx < len(g.consumables):
        e["key"], e["targets"] = g.consumables[a.idx].key, len(a.cards)
    elif k == "pick" and g.state == "PACK" and 0 <= a.idx < len(g.pack_cards):
        x = g.pack_cards[a.idx]
        key = getattr(x, "key", None)
        e["key"] = (f"j_{key}" if hasattr(x, "state") else key) if isinstance(key, str) else "<playing_card>"
        e["pack"] = g.pack_kind
    elif k == "skip" and g.blind_idx < len(g.tags_offered):
        e["key"] = f"tag_{g.tags_offered[g.blind_idx]}"
    elif k == "reroll":
        e["cost"] = 0 if g.free_rerolls > 0 else g.reroll_cost
    elif k == "select":
        e["key"] = g.boss if g.blind_idx == 2 else ""
    elif k == "pack_skip":
        e["pack"] = g.pack_kind
    return e


def lineup(g) -> dict:
    return {"kind": "ante", "ante": g.ante, "money": g.money, "step": 0, "blind": 0, "in_round": False,
            "jokers": [f"j_{j.key}" + (f":{j.edition}" if j.edition else "") for j in g.jokers],
            "vouchers": sorted(g.vouchers), "levels": list(g.hand_levels), "deck": len(g.full_deck)}


def play_out(agent, w, rec: GameRecorder, explore: bool = False, events: list | None = None,
             branch_at: int = -1, branch_k: int = 0, branch_rng=None):
    """Play `w` to the end with `agent`, recording into `rec`; returns the branch worlds and their first
    actions (when the branch_at-th decision outside a round was reached), else []."""
    g = w.g
    seen_ante = set()
    n_build = 0
    branches = []
    while not w.done:
        g = w.g
        if events is not None and g.state == "BLIND_SELECT" and g.blind_idx == 0 and g.ante not in seen_ante:
            seen_ante.add(g.ante)
            events.append(lineup(g))
        if g.state == "BLIND_SELECT" or g.state in ("SHOP", "PACK"):
            rec.maybe(w)
        d = agent.decide(w, explore=explore)
        if d.action is None:
            g.state = "GAME_OVER"
            break
        if g.state != "SELECTING_HAND" and len(d.choice.cands) >= 2:
            n_build += 1
            if n_build == branch_at and branch_k > 0:
                order = [int(i) for i in np.argsort(-d.policy, kind="stable")] if len(d.policy) == len(d.choice.cands) else [d.index]
                for i in order[:branch_k]:
                    branches.append((w.determinize(branch_rng), d.choice.cands[i].action))
        if events is not None:
            ev = event(w, d.action)
            if ev is not None:
                events.append(ev)
        prev_state, prev_ante, prev_blind = g.state, g.ante, blind_index(g)
        w.step(d.action)
        if prev_state != "SELECTING_HAND" and g.state == "SELECTING_HAND":
            rec.maybe(w, blind_start=True)
        if prev_state == "SELECTING_HAND" and g.state != "SELECTING_HAND" and prev_blind % 3 == 2:
            rec.boss(prev_ante, g.state in ("SHOP", "WON", "PACK", "BLIND_SELECT"))
    return branches


def play_recorded(agent, seed: int, deck: str, stake: str, explore: bool, game_id: int, branch_k: int = 0,
                  branch_prob: float = 0.0, branch_span: int = 40):
    """One game (plus its branch rollouts, module doc); returns (labelled rows, info, events)."""
    from ..az.world import World
    from ..sim.game import Game
    w = World(Game(seed=seed, deck_type=deck, stake=stake))
    agent.reseed(seed * 7919 + 17)
    rec = GameRecorder(agent.potential, game_id)
    events = []
    brng = random.Random(seed * 31 + 7)
    branch_at = brng.randint(1, branch_span) if (branch_k > 0 and brng.random() < branch_prob) else -1
    branches = play_out(agent, w, rec, explore, events, branch_at, branch_k, brng)
    g = w.g
    rows = rec.finish(g)
    n_branch = 0
    for b, (wb, action) in enumerate(branches):
        try:
            wb.step(action)
        except (AssertionError, IndexError):
            continue
        agent.reseed(seed * 7919 + 101 + b)
        rb = GameRecorder(agent.potential, -(8 * seed + 1 + b), group=seed)
        if wb.g.state == "SELECTING_HAND":
            rb.maybe(wb, blind_start=True)
        play_out(agent, wb, rb)
        rows += rb.finish(wb.g)
        n_branch += 1
    info = {"seed": seed, "stake": stake, "deck": deck, "won": g.state == "WON", "blinds": g.furthest_blind,
            "ante": g.ante, "steps": w.steps, "explore": explore, "records": len(rows),
            "boss": g.boss if g.state != "WON" else "", "died_blind": min(g.blind_idx, 2), "money_end": g.money,
            "branches": n_branch}
    return rows, info, events


def _worker(args):
    import torch
    torch.set_num_threads(1)
    from ..az.train import _make_agent
    model, cfg, pot, seeds, stakes, weights, explore_frac, deck, out_path, branch = args
    if os.path.exists(out_path):
        return []
    # one-step / prior configs need search=True to take their shop path (agent.decide); without it every
    # "V agent" generation so far was the plain graded prior
    search = cfg.get("shop_eval") in ("onestep", "prior")
    agent = _make_agent(model, search, seeds[0], cfg, pot)
    rows, infos, events = [], [], []
    for s in seeds:
        rng = random.Random(s)
        stake = rng.choices(stakes, weights=weights)[0]
        explore = rng.random() < explore_frac
        r, info, ev = play_recorded(agent, s, deck, stake, explore, s, *branch)
        rows += r
        infos.append(info)
        for e in ev:
            e["game"] = s
        events += ev
    if rows:
        import gzip
        with gzip.open(out_path[:-4] + ".events.jsonl.gz", "wt", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
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
    branch = (a.branch_k, a.branch_prob, a.branch_span)
    jobs = [(a.model, cfg, pot, c, stakes, weights, a.explore_frac, a.deck,
             os.path.join(a.out, f"chunk{k:05d}.npz"), branch) for k, c in enumerate(chunks)]
    todo = [j for j in jobs if not os.path.exists(j[-2])]
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
    p.add_argument("--branch-k", type=int, default=0, help="branch rollouts: options per branched decision")
    p.add_argument("--branch-prob", type=float, default=0.5, help="share of games with a branched decision")
    p.add_argument("--branch-span", type=int, default=40, help="the branched decision is the t-th outside a round, t <= this")
    p.add_argument("--out", default="checkpoints/value_data")
    p.add_argument("--log", default="", help="append progress to this file instead of stdout")
    generate(p.parse_args())


if __name__ == "__main__":
    main()
