"""Audit of the build judge on the builds a real player reaches (not the generator's).

A network plays full Gold games. At every blind select from ante `--min-ante` on, the state is copied `--copies` times, each copy gets new draws (VecEnv.reseed_rng) and the blind is
played by the scripted player; the share cleared is that state's true pass rate (for that player and mode). The judge's prediction for the same state is compared with it.

    python tools/audit_judge.py --policy checkpoints/ppo_gold_v7_best_greedy.pt --judge checkpoints/judge2.pt --games 256
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--judge", required=True)
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--copies", type=int, default=16)
    p.add_argument("--min-ante", type=int, default=3)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--max-states", type=int, default=6000)
    p.add_argument("--seed0", type=int, default=41_000_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dev-engine", default="")
    p.add_argument("--out", default="")
    p.add_argument("--save", default="", help="a directory: every audited state is written there as labelled training data (the relabel loop: the builds this player really reaches)")
    p.add_argument("--seed-tag", type=int, default=0)
    a = p.parse_args()
    if a.dev_engine:
        sys.path.insert(0, a.dev_engine)
        import _engine_dev
        import balatro_rl.sim as S
        sys.modules["balatro_rl.sim._engine"] = _engine_dev
        S._engine = _engine_dev
    from balatro_rl.engine_env import CppVecEnv, OBS_KEYS
    from balatro_rl.env import A_SELECT, F_JOKER_W
    from balatro_rl.model import load_model
    from balatro_rl import judge as J

    dev = a.device
    policy = load_model(a.policy, "cpu")
    judge = J.load(a.judge, dev)
    G, K, CH = a.games, a.copies, a.chunk
    venv = CppVecEnv(G, "RED", "GOLD", seed0=a.seed0, threads=0)
    venv.set_marginal(1)
    la = CppVecEnv(CH * K, "RED", "GOLD", seed0=1, threads=0)
    recs = {m: {"p": [], "y": [], "ante": [], "blind": []} for m in (0, 1)}
    save = {k: [] for k in ("g", "jid", "jf", "cid", "pass", "ratio", "hands", "discards", "money", "dc", "ante", "blind", "seed", "mode")} if a.save else None
    if a.save:
        os.makedirs(a.save, exist_ok=True)
    shard = 0
    t0, n_states = time.time(), 0
    for step in range(3000):
        o = {k: torch.from_numpy(np.array(venv.obs[k])) for k in OBS_KEYS if k in venv.obs}
        g = np.asarray(venv.obs["g"])
        ante = np.rint(g[:, 0] * 8).astype(int)
        ids = np.flatnonzero((g[:, 4] > 0.5) & (ante >= a.min_ante))
        for c0 in range(0, len(ids), CH):
            chunk = ids[c0:c0 + CH]
            gg = torch.from_numpy(g[chunk]).to(dev)
            jid = torch.from_numpy(np.asarray(venv.obs["jid"])[chunk]).to(dev)
            jf = torch.from_numpy(np.asarray(venv.obs["jf"])[chunk][:, :, :F_JOKER_W]).to(dev)
            cid = torch.from_numpy(np.asarray(venv.obs["cid"])[chunk]).to(dev)
            for mode in (0, 1):
                pj = judge.predict(gg, jid, jf, cid, torch.full((len(chunk),), mode, device=dev, dtype=torch.long))["p_pass"].cpu().numpy()
                src = [int(e) for e in chunk for _ in range(K)]
                dst = list(range(len(src)))
                la.copy_many(venv, src, dst)
                la.reseed_rng(dst, [a.seed0 + 1000003 * n_states + 7919 * i + 13 * mode for i in range(len(src))])
                la.set_auto_round(True, 1.5, mode)
                la.step_split(np.concatenate([np.full(len(src), A_SELECT), np.full(la.n - len(src), -1)]))
                rep_ = la.round_report()
                y = rep_["cleared"][:len(src)].reshape(len(chunk), K).mean(1)
                if save is not None:
                    n_ = len(src)
                    rs = lambda x: x[:n_].reshape(len(chunk), K, *x.shape[1:]).mean(1)
                    save["g"].append(g[chunk].astype(np.float32)); save["jid"].append(np.asarray(venv.obs["jid"])[chunk].astype(np.int16))
                    save["jf"].append(np.asarray(venv.obs["jf"])[chunk][:, :, :F_JOKER_W].astype(np.float16)); save["cid"].append(np.asarray(venv.obs["cid"])[chunk].astype(np.int16))
                    save["pass"].append(y.astype(np.float32)); save["ratio"].append(np.log10(np.maximum(rep_["ratio"][:n_], 1e-3)).reshape(len(chunk), K).mean(1).astype(np.float32))
                    save["hands"].append(rs(rep_["hands"]).astype(np.float32)); save["discards"].append(rs(rep_["discards"]).astype(np.float32)); save["money"].append(rs(rep_["money"]).astype(np.float32))
                    save["dc"].append(rs(rep_["c1"] - rep_["c0"]).astype(np.float32)); save["ante"].append(ante[chunk].astype(np.int8))
                    save["blind"].append(np.argmax(g[chunk][:, 1:4], 1).astype(np.int8)); save["mode"].append(np.full(len(chunk), mode, dtype=np.int8))
                    save["seed"].append((a.seed_tag * 1_000_003 + n_states * 8 + chunk.astype(np.int64) % 8 + 17).astype(np.int64))
                recs[mode]["p"].append(pj); recs[mode]["y"].append(y)
                recs[mode]["ante"].append(ante[chunk]); recs[mode]["blind"].append(np.argmax(g[chunk][:, 1:4], 1))
            n_states += len(chunk)
        if save is not None and sum(len(x) for x in save["pass"]) >= 60000:
            np.savez_compressed(os.path.join(a.save, f"shard{shard:05d}.npz"), **{k: np.concatenate(v) for k, v in save.items()})
            shard += 1
            save = {k: [] for k in save}
        if n_states >= a.max_states:
            break
        with torch.no_grad():
            lg, _ = policy({k: v for k, v in o.items()})
        act = lg.float().argmax(1).numpy()
        venv.step(act)
    if save is not None and save["pass"]:
        np.savez_compressed(os.path.join(a.save, f"shard{shard:05d}.npz"), **{k: np.concatenate(v) for k, v in save.items()})
        json.dump({"reps": K, "pace": 1.5, "source": a.policy, "g_dim": int(g.shape[1]), "jf_dim": F_JOKER_W}, open(os.path.join(a.save, "meta.json"), "w"))
    out = {"states": n_states, "min": round((time.time() - t0) / 60, 1)}
    for mode in (0, 1):
        p_ = np.concatenate(recs[mode]["p"]); y_ = np.concatenate(recs[mode]["y"]); an = np.concatenate(recs[mode]["ante"])
        ece, brier, table = J.calibration(p_, y_)
        res = {"n": len(p_), "mean_pred": round(float(p_.mean()), 3), "mean_true": round(float(y_.mean()), 3), "brier": round(brier, 4), "ece": round(ece, 4),
               "auc": round(J.auc(p_, y_), 4) if 0 < (y_ > 0.5).sum() < len(y_) else None, "base_brier": round(float(np.var(y_)), 4),
               "mae": round(float(np.abs(p_ - y_).mean()), 4), "by_ante": {}}
        for x in range(a.min_ante, 9):
            m = an == x
            if m.sum() >= 30:
                res["by_ante"][x] = {"n": int(m.sum()), "pred": round(float(p_[m].mean()), 3), "true": round(float(y_[m].mean()), 3), "brier": round(float(np.mean((p_[m] - y_[m]) ** 2)), 4)}
        out[f"mode{mode}"] = res
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
