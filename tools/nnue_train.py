"""Trains the evaluation network (balatro_rl/nnue.py) on tools/nnue_data.py shards: target = blinds reached / 24, mean squared error; seeds split
into train / held-out. Reports the held-out error, Spearman overall and within one beam level of one seed (the comparisons the search makes:
sibling states), and how well the top-ranked state of each level agrees with the best label.

    python tools/nnue_train.py data/nnue --out checkpoints/nnue1.pt
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.nnue import NNUE, N_FEATURES                     # noqa: E402
from runeval_fit import spearman                                 # noqa: E402


def load_shards(d):
    idx, offs, y, seed, level, ante = [], [0], [], [], [], []
    for f in sorted(glob.glob(os.path.join(d, "nnue_*.npz"))):
        z = np.load(f)
        o = z["offs"]
        idx.append(z["idx"])
        offs += list(o[1:] + offs[-1])
        y.append(z["y"])
        seed.append(z["seed"])
        level.append(z["level"])
        ante.append(z["ante"])
    return (np.concatenate(idx), np.array(offs, dtype=np.int64), np.concatenate(y), np.concatenate(seed), np.concatenate(level), np.concatenate(ante))


def batch(idx, offs, rows):
    parts = [idx[offs[r]:offs[r + 1]] for r in rows]
    o = np.cumsum([0] + [len(p) for p in parts])
    return np.concatenate(parts), o


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="?", default="data/nnue")
    p.add_argument("--out", default="checkpoints/nnue1.pt")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.0, help="weight decay (AdamW)")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    idx, offs, y, seed, level, ante = load_shards(a.data)
    n = len(y)
    useeds = np.unique(seed)
    test_seeds = set(useeds[::5].tolist())
    test = np.array([s_ in test_seeds for s_ in seed])
    tr = np.flatnonzero(~test)
    te = np.flatnonzero(test)
    print(f"{n} states from {len(useeds)} seeds; train {len(tr)}, held-out {len(te)} ({len(test_seeds)} seeds); mean label {y.mean():.2f} blinds")
    net = NNUE(N_FEATURES, a.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    yt = y / 24.0

    def predict(rows):
        out = []
        net.eval()
        with torch.no_grad():
            for c0 in range(0, len(rows), 8192):
                bi, bo = batch(idx, offs, rows[c0:c0 + 8192])
                out.append(net(torch.as_tensor(bi, device=dev), torch.as_tensor(bo, device=dev)).cpu().numpy())
        net.train()
        return np.concatenate(out) * 24.0

    rng = np.random.default_rng(0)
    t0 = time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        tot = 0.0
        for c0 in range(0, len(perm), a.bs):
            rows = perm[c0:c0 + a.bs]
            bi, bo = batch(idx, offs, rows)
            pred = net(torch.as_tensor(bi, device=dev), torch.as_tensor(bo, device=dev))
            loss = torch.mean((pred - torch.as_tensor(yt[rows], device=dev)) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(rows)
        sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            ph = predict(te)
            print(f"epoch {ep + 1}: train rmse {np.sqrt(tot / len(tr)) * 24:.2f} blinds, held-out rmse {np.sqrt(np.mean((ph - y[te]) ** 2)):.2f}, "
                  f"Spearman {spearman(ph, y[te]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    ph = predict(te)
    # sibling comparisons: within one seed and level (what the beam ranks)
    sp, top_hit, regret = [], [], []
    keys = {}
    for j, r in enumerate(te):
        keys.setdefault((seed[r], level[r]), []).append(j)
    for js in keys.values():
        if len(js) < 8:
            continue
        yy, pp = y[te[js]], ph[js]
        if np.std(yy) > 0:
            sp.append(spearman(pp, yy))
        top_hit.append(float(yy[np.argmax(pp)] >= np.max(yy) - 0.5))
        regret.append(float(np.max(yy) - yy[np.argmax(pp)]))
    print(f"held-out sibling groups ({len(regret)}): mean Spearman within a group {np.nanmean(sp):.3f}; the network's top pick is (near) the best in "
          f"{100 * np.mean(top_hit):.0f}% of groups; mean regret {np.mean(regret):.2f} blinds (a random pick: "
          f"{np.mean([np.max(y[te[js]]) - np.mean(y[te[js]]) for js in keys.values() if len(js) >= 8]):.2f})")
    for an in range(1, 9):
        m = ante[te] == an
        if m.sum() > 100:
            print(f"  ante {an}: n {m.sum()}, Spearman {spearman(ph[m], y[te][m]):.3f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"config": net.config, "state": net.state_dict()}, a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
