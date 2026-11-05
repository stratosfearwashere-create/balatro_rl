"""Evaluation of a network on UNMODIFIED games (ante 1, no head start, no scenario, no sandbox), and a sanity check of its critic.

    python tools/eval_report.py --ckpt checkpoints/x.pt --stake GOLD --games 200
    python tools/eval_report.py --ckpt checkpoints/x.pt --critic-check 1500            # also: does V(s) rank builds the way the engine's score does?

Part 1, greedy games on seeds seed0+i: the win rate, the distribution of the blinds reached, and per ante (at the moment a blind is selected, and per blind
cleared): money, interest, how many jokers and of what kinds (+mult, xMult, chips, scaling, economy), and the ratio of the score at the clearing play to the
requirement (the margin the build clears blinds with).
Part 2 (--critic-check N): N build states sampled by the scenario generator outside rounds; the value V(s) against the engine's own expected score of the build
(marginal.py, the global column the network is not shown unless it was made with --potential), within each ante: Spearman rank correlation, and the share of
pairs of builds of the same ante whose order V gets right.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppEnv, CppVecEnv, OBS_KEYS      # noqa: E402
from balatro_rl.model import load_model                            # noqa: E402
from balatro_rl.marginal import MARG_G, POT_G                      # noqa: E402
from balatro_rl.env import global_size                             # noqa: E402
from balatro_rl.joker_desc import FEATURES, item_descriptors       # noqa: E402
from balatro_rl.sim.jokers import JOKERS                           # noqa: E402

KINDS = ("mult", "xmult", "chips", "grows", "economy")


def joker_kinds():
    d = item_descriptors()
    col = {n: FEATURES.index(n) for n in KINDS}
    from balatro_rl.sim.items import VOCAB
    out = {}
    for i, k in enumerate(VOCAB):
        if k.startswith("j_") and k[2:] in JOKERS:
            out[k[2:]] = {n: bool(d[i, col[n]]) for n in KINDS}
    return out


def play_games(a, model):
    kinds = joker_kinds()
    per_ante = defaultdict(lambda: defaultdict(list))           # ante -> metric -> values
    ratios = defaultdict(list)                                  # ante -> score / requirement of the clearing play
    results = []
    for i in range(a.games):
        e = CppEnv(a.deck, a.stake)
        if getattr(model, "marg", 0):
            e.e.marginal = int(model.marg)
        obs = e.reset(a.seed0 + i)
        done = False
        seen_ante = set()
        while not done:
            d = e.game.dump()
            if d["state"] == "BLIND_SELECT" and d["blind_idx"] == 0 and d["ante"] not in seen_ante:
                seen_ante.add(d["ante"])
                js = [j["key"] for j in d["jokers"]]
                m = per_ante[d["ante"]]
                m["money"].append(d["money"])
                m["interest"].append(min(5, max(0, d["money"]) // 5))
                m["jokers"].append(len(js))
                for n in KINDS:
                    m[n].append(sum(kinds[k][n] for k in js))
            batch = {k: torch.from_numpy(np.asarray(obs[k]))[None] for k in OBS_KEYS}
            with torch.no_grad():
                logits, _ = model(batch)
            act = int(logits.argmax(1))
            before_chips, target, ante, was_round = d["chips"], d["target"], d["ante"], d["state"] == "SELECTING_HAND"
            fb = d["furthest_blind"]
            obs, _, done, info = e.step(act)
            if was_round and act < 218:
                d2 = e.game.dump()
                if d2["furthest_blind"] > fb:                          # this play cleared the blind
                    ratios[ante].append((before_chips + e.e.last_score) / max(1.0, target))
        d = e.game.dump()
        results.append((d["state"] == "WON", d["furthest_blind"]))
    return results, per_ante, ratios


def critic_check(a, model):
    v = CppVecEnv(a.critic_check, "RED", a.stake, seed0=a.seed0 + 777_000, threads=0)
    v.set_scenario(1.0, 0.0, 1.0, 0.0, 1, 7, 3, 0.5)
    v.set_marginal(1)
    v.v.restart()
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):                                              # states outside rounds only; random play moves on to new ones
        o = v.obs
        g = np.asarray(o["g"])
        ante = np.rint(g[:, 0] * 8).astype(int)
        outside = g[:, 5] < 0.5
        base_col = global_size() - MARG_G - POT_G
        score = g[:, base_col]
        if outside.any():
            batch = {k: torch.from_numpy(np.asarray(o[k]))[outside] for k in OBS_KEYS}
            with torch.no_grad():
                _, val = model(batch)
            val = val.float()
            val = val.sum(-1) if val.dim() > 1 else val
            for an, sc, vv in zip(ante[outside], score[outside], val.numpy()):
                rows.append((int(an), float(sc), float(vv)))
        mask = np.asarray(o["mask"])
        v.step(np.argmax(rng.random(mask.shape) * mask, axis=1))
    out = {}
    pair_ok = pair_n = 0
    cors = []
    by = defaultdict(list)
    for an, sc, vv in rows:
        by[an].append((sc, vv))
    for an, xs in sorted(by.items()):
        if len(xs) < 30:
            continue
        sc, vv = np.array(xs).T
        rs = np.argsort(np.argsort(sc)); rv = np.argsort(np.argsort(vv))
        c = float(np.corrcoef(rs, rv)[0, 1])
        cors.append(c)
        idx = rng.integers(0, len(xs), size=(2000, 2))
        diff = sc[idx[:, 0]] - sc[idx[:, 1]]
        keep = np.abs(diff) > 0.05                                    # builds whose expected scores differ by more than ~12% (log10/7 units)
        ok = (np.sign(diff[keep]) == np.sign(vv[idx[:, 0]][keep] - vv[idx[:, 1]][keep])).sum()
        pair_ok += int(ok); pair_n += int(keep.sum())
        out[an] = {"n": len(xs), "spearman": round(c, 3)}
    return {"states": len(rows), "per_ante": out, "mean_spearman": round(float(np.mean(cors)), 3) if cors else None,
            "pairs_ordered_right": round(pair_ok / max(1, pair_n), 3), "pairs": pair_n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed0", type=int, default=9_000_000)
    p.add_argument("--critic-check", type=int, default=0)
    a = p.parse_args()
    model = load_model(a.ckpt, "cpu")
    rep = {}
    if a.games > 0:
        results, per_ante, ratios = play_games(a, model)
        n = len(results)
        rep["games"] = n
        rep["win_rate"] = round(sum(w for w, _ in results) / n, 4)
        hist = Counter(min(b, 24) for _, b in results)
        rep["blinds_reached"] = {str(k): hist[k] for k in sorted(hist)}
        rep["blinds_mean"] = round(float(np.mean([b for _, b in results])), 3)
        rep["per_ante"] = {str(an): {"games": len(m["money"]), **{k: round(float(np.mean(v)), 2) for k, v in m.items()},
                                     **({"score_over_requirement": round(float(np.median(ratios[an])), 2), "clears": len(ratios[an])} if ratios[an] else {})}
                           for an, m in sorted(per_ante.items())}
    if a.critic_check > 0:
        rep["critic"] = critic_check(a, model)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
