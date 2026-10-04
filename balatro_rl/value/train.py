"""Train the build-value network on recorded games, and judge it against Phi.

    python -m balatro_rl.value.train --data checkpoints/value_data --out checkpoints/value.pt --epochs 12
    python -m balatro_rl.value.train --data checkpoints/value_data --out checkpoints/value.pt --report-only

Games are split by game id (5% held out). The report gives, on the held-out games and per ante:
  - log-loss of P(win) and of P(reach the next ante) for the model, for a logistic fit of Phi alone (fitted
    on the training games, one per ante), and for the base rate;
  - AUC of P(reach next ante);
  - calibration buckets of P(win);
  - decision accuracy: on held-out branch groups (gen.py --branch-k), the share of pairs of options whose
    outcomes differ that the model's E[antes] orders correctly, against Phi's ordering.
The model only earns its place in the agent if it beats the Phi fit on held-out log-loss.
--rank-weight > 0 adds a pairwise ranking loss between a group's records (the better outcome first).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .gen import load
from .model import BuildValueNet, ordinal_targets, save_model, load_model, to_tensors
from .records import N_ORDINAL, root_game


def split(data: dict, val_frac: float = 0.05, seed: int = 0):
    games = root_game(data["game"])                   # a game and its branch rollouts stay on one side
    uniq = np.unique(games)
    rng = np.random.default_rng(seed)
    val_games = set(rng.choice(uniq, size=max(1, int(len(uniq) * val_frac)), replace=False).tolist())
    is_val = np.array([g in val_games for g in games])
    return {k: v[~is_val] for k, v in data.items()}, {k: v[is_val] for k, v in data.items()}


def per_game_sample(data: dict, k: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of at most k records per game (a random k): the ~70 records of one game share its outcome,
    so training on all of them every epoch makes the network recognise games instead of builds."""
    games = data["game"]
    order = rng.permutation(len(games))
    seen = {}
    keep = []
    for i in order:
        g = games[i]
        c = seen.get(g, 0)
        if c < k:
            seen[g] = c + 1
            keep.append(i)
    return rng.permutation(np.asarray(keep))


def batches(data: dict, idx: np.ndarray, batch: int):
    for lo in range(0, len(idx), batch):
        sel = idx[lo:lo + batch]
        yield {k: v[sel] for k, v in data.items()}


def groups_of(data: dict) -> list[np.ndarray]:
    """Record indices of each decision group with at least two different outcomes."""
    if "group" not in data:
        return []
    grp = data["group"]
    order = np.argsort(grp, kind="stable")
    order = order[grp[order] >= 0]
    out = []
    if len(order) == 0:
        return out
    g_sorted = grp[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        if len(idx) >= 2 and len(np.unique(data["ante_reached"][idx])) >= 2:
            out.append(idx)
    return out


def rank_batches(data: dict, groups: list[np.ndarray], per_batch: int, rng: np.random.Generator):
    """Batches of whole groups: (batch dict, group id per record)."""
    order = rng.permutation(len(groups))
    for lo in range(0, len(order), per_batch):
        chosen = [groups[i] for i in order[lo:lo + per_batch]]
        sel = np.concatenate(chosen)
        gid = np.concatenate([np.full(len(c), k) for k, c in enumerate(chosen)])
        yield {k: v[sel] for k, v in data.items()}, gid


def rank_loss(scores: torch.Tensor, labels: torch.Tensor, gid: torch.Tensor, scale: float) -> tuple[torch.Tensor, int]:
    """Pairwise logistic loss over pairs within a group whose labels differ: the better outcome should score
    higher. (mean loss, pairs)."""
    same = gid.unsqueeze(0) == gid.unsqueeze(1)
    better = labels.unsqueeze(1) > labels.unsqueeze(0)              # [i, j]: i's outcome beats j's
    m = same & better
    if not m.any():
        return scores.sum() * 0.0, 0
    diff = (scores.unsqueeze(1) - scores.unsqueeze(0))[m]
    return F.softplus(-scale * diff).mean(), int(m.sum().item())


def losses(net, b: dict, device) -> tuple[torch.Tensor, dict]:
    x = to_tensors(b, device)
    out = net(x)
    ante = torch.as_tensor(b["ante"], device=device)
    reached = torch.as_tensor(b["ante_reached"], device=device)
    tgt, mask = ordinal_targets(ante, reached)
    if "td_target" in b:                                   # TD(lambda) targets replace the Monte-Carlo ones
        tgt = torch.as_tensor(b["td_target"], device=device, dtype=torch.float32)
    ol = (F.binary_cross_entropy_with_logits(out["ord"], tgt, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
    bn = F.binary_cross_entropy_with_logits(out["boss_now"], torch.as_tensor(b["boss_now"], device=device))
    bx = F.binary_cross_entropy_with_logits(out["boss_next"], torch.as_tensor(b["boss_next"], device=device))
    bl = F.mse_loss(out["blinds"], torch.as_tensor(b["blinds_after"], device=device))
    loss = ol + 0.25 * bn + 0.25 * bx + 0.5 * bl
    return loss, {"ord": ol.item(), "boss_now": bn.item(), "boss_next": bx.item(), "blinds": bl.item()}


def rank_step(net, b: dict, gid: np.ndarray, device, scale: float) -> tuple[torch.Tensor, int]:
    out = net(to_tensors(b, device))
    e = BuildValueNet.expected_antes(out["ord"])
    labels = torch.as_tensor(b["ante_reached"], device=device)
    return rank_loss(e, labels, torch.as_tensor(gid, device=device), scale)


def decision_accuracy(reach: np.ndarray, data: dict, groups: list[np.ndarray]) -> dict:
    """Share of within-group pairs with different outcomes ordered correctly by the model's E[antes] and by
    Phi (ties count half)."""
    e = 1.0 + reach.sum(-1)
    n = ok_m = ok_p = 0.0
    for idx in groups:
        y, em, ph = data["ante_reached"][idx], e[idx], data["phi"][idx]
        for a in range(len(idx)):
            for b in range(len(idx)):
                if y[a] > y[b]:
                    n += 1
                    ok_m += 1.0 if em[a] > em[b] else (0.5 if em[a] == em[b] else 0.0)
                    ok_p += 1.0 if ph[a] > ph[b] else (0.5 if ph[a] == ph[b] else 0.0)
    return {"groups": len(groups), "pairs": int(n), "model": float(ok_m / n) if n else math.nan,
            "phi": float(ok_p / n) if n else math.nan}


def td_targets(data: dict, reach: np.ndarray, lam: float) -> np.ndarray:
    """TD(lambda) targets for the ordinal heads. Within a game, records in step order: the target of record t
    for head k is (1 - lam) * V_k(s_{t+1}) + lam * target_{t+1}, ending in the game's label; a record's own
    antes (k <= ante) keep the Monte-Carlo label (they are masked out of the loss anyway)."""
    n = len(data["game"])
    k = np.arange(2, 2 + N_ORDINAL)
    mc = (data["ante_reached"][:, None] >= k[None, :]).astype(np.float32)
    out = mc.copy()
    order = np.lexsort((data["step"], data["game"]))
    games = data["game"][order]
    starts = np.flatnonzero(np.r_[True, games[1:] != games[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        nxt = mc[idx[-1]]
        for i in idx[::-1]:
            out[i] = nxt
            nxt = (1.0 - lam) * reach[i] + lam * out[i]
    return out


def predict(net, data: dict, device, batch: int = 2048) -> dict:
    net.eval()
    ords, bn, bx = [], [], []
    with torch.no_grad():
        for b in batches(data, np.arange(len(data["game"])), batch):
            out = net(to_tensors(b, device))
            ords.append(torch.sigmoid(out["ord"]).float().cpu().numpy())
            bn.append(torch.sigmoid(out["boss_now"]).float().cpu().numpy())
            bx.append(torch.sigmoid(out["boss_next"]).float().cpu().numpy())
    return {"reach": np.concatenate(ords), "boss_now": np.concatenate(bn), "boss_next": np.concatenate(bx)}


# ------------------------------------------------------------------ the Phi yardstick
def _logistic_fit(x: np.ndarray, y: np.ndarray, iters: int = 300) -> tuple[float, float]:
    """1-D logistic regression y ~ sigmoid(a x + b) by Newton's method (with a little ridge)."""
    a = b = 0.0
    for _ in range(iters):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1 - p) + 1e-6
        g_a, g_b = ((p - y) * x).sum() + 1e-3 * a, (p - y).sum()
        h_aa, h_ab, h_bb = (w * x * x).sum() + 1e-3, (w * x).sum(), w.sum()
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 0:
            break
        da, db = (h_bb * g_a - h_ab * g_b) / det, (h_aa * g_b - h_ab * g_a) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-9:
            break
    return a, b


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def report(net, train: dict, val: dict, device) -> dict:
    pred = predict(net, val, device)
    out = {"n_val": int(len(val["game"])), "val_games": int(len(np.unique(val["game"]))),
           "win_rate_val": float(val["win"].mean()), "by_ante": {}}
    for a in range(1, 9):
        tr, va = train["ante"] == a, val["ante"] == a
        if va.sum() < 50 or tr.sum() < 200:
            continue
        row = {"n": int(va.sum())}
        # P(win): model vs Phi-logistic vs base rate
        y = val["win"][va]
        p_model = pred["reach"][va, -1]
        ca, cb = _logistic_fit(train["phi"][tr], train["win"][tr])
        p_phi = 1.0 / (1.0 + np.exp(-(ca * val["phi"][va] + cb)))
        base = train["win"][tr].mean()
        row["win"] = {"model": _logloss(p_model, y), "phi_fit": _logloss(p_phi, y),
                      "base": _logloss(np.full(len(y), base), y), "rate": float(y.mean())}
        # P(reach the next ante)
        if a < 8:
            k = a + 1
            y2 = (val["ante_reached"][va] >= k).astype(float)
            p2 = pred["reach"][va, k - 2]
            ca2, cb2 = _logistic_fit(train["phi"][tr], (train["ante_reached"][tr] >= k).astype(float))
            p2_phi = 1.0 / (1.0 + np.exp(-(ca2 * val["phi"][va] + cb2)))
            row["next_ante"] = {"model": _logloss(p2, y2), "phi_fit": _logloss(p2_phi, y2),
                                "auc_model": _auc(p2, y2), "auc_phi": _auc(val["phi"][va], y2), "rate": float(y2.mean())}
        out["by_ante"][a] = row
    # calibration of P(win) on all held-out states
    p = pred["reach"][:, -1]
    y = val["win"]
    edges = np.quantile(p, np.linspace(0, 1, 9))
    buckets = []
    for i in range(8):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == 7 else p < edges[i + 1])
        if m.any():
            buckets.append({"p": float(p[m].mean()), "win_rate": float(y[m].mean()), "n": int(m.sum())})
    out["calibration_win"] = buckets
    by_stake = {}
    for s in np.unique(val["stake"]):
        m = val["stake"] == s
        by_stake[int(s)] = {"n": int(m.sum()), "win_rate": float(y[m].mean()), "model_logloss": _logloss(p[m], y[m])}
    out["by_stake"] = by_stake
    out["decision"] = decision_accuracy(pred["reach"], val, groups_of(val))
    tot_model = np.mean([r["win"]["model"] for r in out["by_ante"].values()]) if out["by_ante"] else math.nan
    tot_phi = np.mean([r["win"]["phi_fit"] for r in out["by_ante"].values()]) if out["by_ante"] else math.nan
    out["verdict"] = {"mean_logloss_win_model": float(tot_model), "mean_logloss_win_phi_fit": float(tot_phi),
                      "beats_phi": bool(tot_model < tot_phi)}
    return out


def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(a.data)
    n = len(data["game"])
    print(f"{n} records from {len(np.unique(data['game']))} games; win rate {data['win'].mean():.3f}; "
          f"antes reached mean {data['ante_reached'].mean():.2f}", flush=True)
    tr, va = split(data, a.val_frac)
    if a.report_only:
        net = load_model(a.out, device)
        rep = report(net, tr, va, device)
        print(json.dumps(rep, indent=1, default=float))
        return
    torch.manual_seed(0)
    net = BuildValueNet(d=a.d, layers=a.layers, dropout=a.dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    per_epoch = len(tr["game"]) if a.per_game <= 0 else min(len(tr["game"]), a.per_game * len(np.unique(tr["game"])))
    steps_per_epoch = math.ceil(per_epoch / a.batch)
    total = max(10, a.epochs * steps_per_epoch)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(            # linear warm-up, then cosine to 2% of the peak
        opt, lambda s: (s + 1) / warm if s < warm else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    rng = np.random.default_rng(0)
    best = math.inf
    log_path = a.out.replace(".pt", "_log.jsonl")
    groups = groups_of(tr) if a.rank_weight > 0 else []
    print(f"{len(groups)} decision groups for the ranking loss (rank_weight {a.rank_weight})", flush=True)
    for ep in range(1, a.epochs + 1):
        if a.td_lambda > 0 and ep > 1:                  # epoch 1 trains on the Monte-Carlo labels
            tr["td_target"] = td_targets(tr, predict(net, tr, device)["reach"], a.td_lambda)
        net.train()
        t = time.time()
        idx = rng.permutation(len(tr["game"])) if a.per_game <= 0 else per_game_sample(tr, a.per_game, rng)
        agg, k = {}, 0
        rb = rank_batches(tr, groups, a.rank_groups, rng) if groups else None
        for b in batches(tr, idx, a.batch):
            loss, parts = losses(net, b, device)
            if rb is not None:                          # one batch of decision groups per ordinary batch
                nxt = next(rb, None)
                if nxt is None:
                    rb = rank_batches(tr, groups, a.rank_groups, rng)
                    nxt = next(rb, None)
                if nxt is not None:
                    rl, pairs = rank_step(net, nxt[0], nxt[1], device, a.rank_scale)
                    loss = loss + a.rank_weight * rl
                    parts["rank"] = rl.item()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            for key, v in parts.items():
                agg[key] = agg.get(key, 0.0) + v
            k += 1
        net.eval()
        vl, vk = {}, 0
        with torch.no_grad():
            for b in batches(va, np.arange(len(va["game"])), 2048):
                _, parts = losses(net, b, device)
                for key, v in parts.items():
                    vl[key] = vl.get(key, 0.0) + v
                vk += 1
        row = {"epoch": ep, "train": {key: v / k for key, v in agg.items()}, "val": {key: v / vk for key, v in vl.items()},
               "min": round((time.time() - t) / 60, 2)}
        if row["val"]["ord"] < best:
            best = row["val"]["ord"]
            save_model(net, a.out, {"epoch": ep, "val": row["val"], "data": a.data, "records": n})
            row["saved"] = True
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    net = load_model(a.out, device)
    rep = report(net, tr, va, device)
    with open(a.out.replace(".pt", "_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(json.dumps(rep["verdict"]), flush=True)
    print("  decision accuracy (held-out branch groups): " + json.dumps(rep["decision"]), flush=True)
    for ante, r in rep["by_ante"].items():
        nxt = r.get("next_ante", {})
        print(f"  ante {ante}: n={r['n']:6d} win logloss model {r['win']['model']:.4f} phi {r['win']['phi_fit']:.4f} "
              f"base {r['win']['base']:.4f} | next-ante AUC model {nxt.get('auc_model', float('nan')):.3f} "
              f"phi {nxt.get('auc_phi', float('nan')):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="checkpoints/value_data")
    p.add_argument("--out", default="checkpoints/value.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--per-game", type=int, default=8, help="records per game per epoch (0: all)")
    p.add_argument("--td-lambda", type=float, default=0.0,
                   help="> 0: TD(lambda) targets (the model's own estimate of the next build, blended with the outcome)")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--rank-weight", type=float, default=0.0, help="> 0: pairwise ranking loss on decision groups")
    p.add_argument("--rank-scale", type=float, default=4.0, help="logistic scale on E[antes] differences")
    p.add_argument("--rank-groups", type=int, default=128, help="decision groups per ranking batch")
    p.add_argument("--report-only", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
