"""Best-of-N demonstrations from the policy's own play (expert iteration on the tail of the outcome distribution): every seed is played N times by the
current network (actions sampled, the same start for all N copies: exact mode), the group's best run (won, else the most blinds; kept only if it won or beat
the group's median by --margin blinds) is kept, and its decisions outside rounds are written as demonstrations in tools/make_demos.py's format, plus the
oracle inputs ob["ex"] the network reads (train.py --oracle-inputs), for train.py --demo. With --loop the file is rebuilt every --every minutes from the
newest checkpoint (the trainer reloads it when it changes); the last --keep batches are kept.

    python tools/bestofn_demos.py --ckpt checkpoints/ppo_gold_big2.pt --seeds 64 --n 8 --money 25 --out checkpoints/demos_bestofn.npz --loop --every 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from balatro_rl import azc, env as E                       # noqa: E402
from balatro_rl.engine_env import CppVecEnv                # noqa: E402
from balatro_rl.model import load_model                    # noqa: E402
from balatro_rl.heur import G_LEVELS as GL, G_FULL as GF, G_ENH as GE    # noqa: E402

DENSE = ("g", "cards", "jid", "jf", "cid", "subs")


class Oracle:
    """train.py's attach_ex: [ante-8 clear share, scaled log best share, P(find a strong multiplier with this money)] for the games outside a round, the
    solver part cached per build."""

    def __init__(self, samples=1, nodes=150):
        mt = json.load(open(os.path.join(ROOT, "balatro_rl", "data", "money_option.json")))
        self.money, self.p = np.array(mt["money"], dtype=np.float64), np.array(mt["p"], dtype=np.float64)
        self.cache, self.samples, self.nodes = {}, samples, nodes

    def attach(self, v, ex):
        mask = np.asarray(v.obs["mask"])
        outr = np.flatnonzero(~mask[:, :E.A_SELECT].any(1))
        if not len(outr):
            return
        g = np.asarray(v.obs["g"])[outr]
        jid = np.asarray(v.obs["jid"])[outr]
        jf = np.asarray(v.obs["jf"])[outr]
        parts = np.concatenate([jid, np.rint(jf[:, :, 9]), np.rint(g[:, GL:GL + 12] * 10), np.rint(g[:, GF:GF + 1] * 52), np.rint(g[:, GE:GE + 8] * 10)], 1).astype(np.int32)
        sigs = [row.tobytes() for row in parts]
        if len(self.cache) > 400_000:
            self.cache.clear()
        need = [k for k, sg in enumerate(sigs) if sg not in self.cache]
        if need:
            fb = v.final_boss_clear([int(outr[k]) for k in need], self.samples, self.nodes)
            for k, row in zip(need, fb):
                self.cache[sigs[k]] = (float(row[0]), float(np.clip((np.log10(max(row[1], 1e-4)) + 4.0) / 4.0, 0.0, 1.0)))
        vals = np.array([self.cache[sg] for sg in sigs], dtype=np.float64)
        ex[outr, 0], ex[outr, 1] = vals[:, 0], vals[:, 1]
        money = 10.0 ** (3.0 * g[:, 18]) - 1.0
        ante = np.clip(np.rint(g[:, 0] * 8.0), 1, 8).astype(int)
        pf = np.zeros(len(outr))
        for an in range(1, 9):
            sel = ante == an
            if sel.any():
                pf[sel] = np.interp(money[sel], self.money, self.p[an - 1])
        ex[outr, 2] = pf


def play_batch(model, oracle, seed0, groups, n, a):
    """groups seeds (seed0..), n copies each -> (records per kept game: list of dicts, outcomes [groups, n] blinds, won)."""
    N = groups * n
    roots = CppVecEnv(groups, "RED", a.stake, seed0=seed0, threads=0)
    roots.set_exact(True)
    if a.money > 0:
        roots.set_start(0.0, 0.0, 1.0, a.money)
    roots.v.restart()
    v = CppVecEnv(N, "RED", a.stake, seed0=1, threads=0)
    v.set_exact(True)
    v.v.restart()
    v.set_auto_round(False, 1.5, 0)
    v.copy_many(roots, [i // n for i in range(N)], list(range(N)))
    ex = np.zeros((N, 3), dtype=np.float32)
    recs = [[] for _ in range(N)]
    done = np.zeros(N, dtype=bool)
    bl = np.zeros(N)
    won = np.zeros(N, dtype=bool)
    for _ in range(9000):
        mask = np.asarray(v.obs["mask"])
        live = np.flatnonzero(~done)
        oracle.attach(v, ex)
        ob = azc.obs_batch(v, list(live), "cpu")
        ob["ex"] = torch.from_numpy(ex[live])
        with torch.no_grad():
            lg, _ = model(ob)
        lg = lg.float().masked_fill(~torch.from_numpy(mask[live]), -1e9)
        pick = torch.multinomial(torch.softmax(lg / a.temp, 1), 1).squeeze(1).numpy()
        act = np.full(N, -1, dtype=np.int64)
        act[live] = pick
        outr = ~mask[:, :E.A_SELECT].any(1)
        for i in live:
            if outr[i] and mask[i].sum() > 1:
                idx = np.flatnonzero(mask[i])
                recs[i].append({**{k: np.asarray(v.obs[k])[i].copy() for k in DENSE}, "ex": ex[i].copy(), "idx": idx.astype(np.int16),
                                "af": np.asarray(v.obs["af"])[i][idx].astype(np.float16), "aid": np.asarray(v.obs["aid"])[i][idx].astype(np.int32),
                                "aid2": np.asarray(v.obs["aid2"])[i][idx].astype(np.int32), "act": int(act[i])})
        _, d, info = v.step(act)
        f = d & ~done
        bl[f], won[f] = info["blinds"][f], info["won"][f]
        done |= d
        if done.all():
            break
    kept = []
    for g_ in range(groups):
        ids = np.arange(g_ * n, (g_ + 1) * n)
        score = won[ids] * 100.0 + bl[ids]
        best = ids[int(np.argmax(score))]
        if won[best] or bl[best] >= np.median(bl[ids]) + a.margin:
            kept.append((seed0 + g_, int(bl[best]), bool(won[best]), recs[best]))
    return kept, bl.reshape(groups, n), won.reshape(groups, n)


def write(batches, out):
    rec = {k: [] for k in DENSE + ("ex",)}
    legal_idx, legal_af, legal_aid, legal_aid2, offs = [], [], [], [], [0]
    acts, seeds, levels, won = [], [], [], []
    for kept in batches:
        for seed, level, w, rs in kept:
            for r in rs:
                for k in rec:
                    rec[k].append(r[k])
                legal_idx.append(r["idx"]); legal_af.append(r["af"]); legal_aid.append(r["aid"]); legal_aid2.append(r["aid2"])
                offs.append(offs[-1] + len(r["idx"]))
                acts.append(r["act"]); seeds.append(seed); levels.append(level); won.append(w)
    if not acts:
        return 0
    o = {k: np.stack(rec[k]) for k in rec}
    o["cards"] = o["cards"].astype(np.float16)
    o["subs"] = o["subs"].astype(np.int8)
    o.update(legal_idx=np.concatenate(legal_idx), legal_af=np.concatenate(legal_af), legal_aid=np.concatenate(legal_aid), legal_aid2=np.concatenate(legal_aid2),
             offs=np.array(offs, dtype=np.int64), act=np.array(acts, dtype=np.int64), seed=np.array(seeds), level=np.array(levels), won=np.array(won))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **o)
    os.replace(tmp, out)
    return len(acts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_big2.pt")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--money", type=int, default=25, help="extra starting money, as in the run")
    p.add_argument("--seeds", type=int, default=64, help="seed groups per batch")
    p.add_argument("--n", type=int, default=8, help="copies per seed")
    p.add_argument("--margin", type=float, default=2.0, help="a group's best is kept if it won or beat the group's median by this many blinds")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--batches", type=int, default=2, help="batches per refresh")
    p.add_argument("--keep", type=int, default=6, help="batches kept in the file (the newest)")
    p.add_argument("--seed0", type=int, default=50_000_000)
    p.add_argument("--out", default="checkpoints/demos_bestofn.npz")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--every", type=float, default=10.0, help="minutes between refreshes")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (the network runs on the CPU beside the trainer)")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    oracle = Oracle()
    history = []
    seed = a.seed0
    mtime = None
    model = None
    while True:
        t0 = time.time()
        m = os.path.getmtime(a.ckpt)
        if m != mtime:
            try:
                model = load_model(a.ckpt, "cpu").eval()
                mtime = m
            except Exception as ex:                                  # (the trainer may be writing it)
                print(time.strftime("%H:%M"), "checkpoint not readable yet:", ex.__class__.__name__, flush=True)
                if model is None:
                    time.sleep(30)
                    continue
        stats = []
        for _ in range(a.batches):
            kept, bl, won = play_batch(model, oracle, seed, a.seeds, a.n, a)
            seed += a.seeds
            history.append(kept)
            stats.append((len(kept), float(bl.mean()), float(bl.max(1).mean()), int(won.sum()), int(won.any(1).sum())))
        history = history[-a.keep:]
        nd = write(history, a.out)
        s = np.array(stats)
        print(time.strftime("%H:%M"), json.dumps({"ckpt_time": time.strftime("%H:%M", time.localtime(mtime)), "groups_kept": int(s[:, 0].sum()), "of": a.seeds * a.batches,
                                                   "mean_blinds": round(float(s[:, 1].mean()), 2), "best_of_n_blinds": round(float(s[:, 2].mean()), 2),
                                                   "wins": int(s[:, 3].sum()), "groups_with_a_win": int(s[:, 4].sum()), "decisions_in_file": nd,
                                                   "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
        if not a.loop:
            break
        time.sleep(max(0.0, a.every * 60 - (time.time() - t0)))


if __name__ == "__main__":
    main()
