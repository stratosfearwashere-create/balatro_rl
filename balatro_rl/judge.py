"""The build judge (network 1) and the scaling model (network 2): supervised networks on tools/judge_data.py's shards.

Both read a build at a blind select: the global vector (ante, blind, money, the deck's make-up, hand levels, vouchers, ... and the engine's exact expected-score columns), the
jokers with their counters and the joker descriptors (what each does and, with spec, what grows it and by how much), the consumables. The encoder is the attention encoder of
model.ActorCritic over those tokens; there are no cards in hand at a blind select.

  kind "judge"    heads: pass probability of the blind (a logit), log10 of the best share of the target reached, hands played, discards used, money gained
  kind "scaling"  heads: the change of each joker slot's counter over the round (signed log scale), hands played, discards used, money gained

    python -m balatro_rl.judge --data data/judge1 --kind judge --out checkpoints/judge1.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import F_CARD
from .model import ActorCritic, mlp

MAX_HAND_SLOTS = 16
N_SLOTS = 8


def slog(x):
    """signed log: counters change by 0.1 (a multiplier) or by 200 (chips): one scale for both."""
    return np.sign(x) * np.log1p(np.abs(x))


class JudgeNet(nn.Module):
    def __init__(self, kind: str = "judge", attn: int = 2, tok: int = 128, spec: bool = True):
        super().__init__()
        self.kind = kind
        self.enc = ActorCritic(attn=attn, tok=tok, desc=1, spec=int(spec), marg=1)
        d = self.enc.config["d"]
        self.n_out = 5 if kind == "judge" else N_SLOTS + 3
        self.mode_emb = nn.Embedding(2, d)                    # the scripted player's mode (0: score most, 1: scaler-friendly) is told to the net
        nn.init.zeros_(self.mode_emb.weight)
        self.head = mlp(d, 256, self.n_out)
        self.config = {"kind": kind, "attn": attn, "tok": tok, "spec": spec}

    def forward(self, g, jid, jf, cid, mode=None):
        B = g.shape[0]
        ob = {"g": g.float(), "cards": torch.zeros((B, MAX_HAND_SLOTS, F_CARD), device=g.device), "jid": jid.long(), "jf": jf.float(), "cid": cid.long()}
        _, s = self.enc._encode_attn(ob)
        if mode is not None:
            s = s + self.mode_emb(mode.long()).to(s.dtype)
        return self.head(s)

    @torch.no_grad()
    def predict(self, g, jid, jf, cid, mode=None):
        out = self(g, jid, jf, cid, mode)
        if self.kind == "judge":
            return {"p_pass": torch.sigmoid(out[:, 0]), "log_ratio": out[:, 1], "hands": out[:, 2], "discards": out[:, 3], "money": out[:, 4]}
        return {"dc": torch.sign(out[:, :N_SLOTS]) * torch.expm1(out[:, :N_SLOTS].abs()), "hands": out[:, N_SLOTS], "discards": out[:, N_SLOTS + 1], "money": out[:, N_SLOTS + 2]}


def save(path, net, extra=None):
    torch.save({"state_dict": net.state_dict(), "config": net.config, **(extra or {})}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = JudgeNet(**ck["config"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


def load_shards(path, max_shards=0, repeat=None):
    """path: a directory of shards, or several separated by commas (a generator's data and the builds a real player reaches); repeat: how many times each directory's samples count
    (the real builds are fewer and matter more)."""
    dirs = path.split(",")
    reps = [int(x) for x in repeat.split(",")] if repeat else [1] * len(dirs)
    parts = []
    for d, r in zip(dirs, reps):
        files = sorted(glob.glob(os.path.join(d, "shard*.npz")))
        if max_shards:
            files = files[:max_shards]
        for f in files:
            z = np.load(f)
            parts += [{k: z[k] for k in z.files}] * r
    keys = set.intersection(*[set(p) for p in parts])
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def split(data, held_out=0.05):
    """held-out BUILDS (the same build on its three blinds goes the same way), by the build seed."""
    h = (data["seed"].astype(np.int64) * 2654435761 % 1000003) / 1000003.0
    ho = h < held_out
    return ~ho, ho


def calibration(p, y, bins=10):
    """reliability of predicted pass probabilities against the observed pass rates: expected calibration error, Brier score, and the table."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append((round(float(p[m].mean()), 3), round(float(y[m].mean()), 3), int(m.sum())))
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece), float(np.mean((p - y) ** 2)), rows


def auc(p, y_hard):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y_hard > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def train(a):
    dev = a.device
    t0 = time.time()
    data = load_shards(a.data, a.max_shards, a.repeat)
    n = len(data["pass"])
    tr, ho = split(data, a.held_out)
    meta = json.load(open(os.path.join(a.data.split(",")[0], "meta.json")))
    print(json.dumps({"samples": n, "train": int(tr.sum()), "held_out": int(ho.sum()), "load_s": round(time.time() - t0, 1), "reps": meta["reps"]}), flush=True)

    def T(x, dt=None):
        t = torch.from_numpy(np.ascontiguousarray(x))
        return (t.to(dt) if dt else t).to(dev)
    G, JID, JF, CID = T(data["g"]), T(data["jid"], torch.int16), T(data["jf"]), T(data["cid"], torch.int16)
    MODE = T(data["mode"] if "mode" in data else np.zeros(n, dtype=np.int8), torch.int8)
    if a.kind == "judge":
        Y = T(np.stack([data["pass"], data["ratio"], data["hands"], data["discards"], data["money"] / 5.0], 1).astype(np.float32))
    else:
        Y = T(np.concatenate([slog(data["dc"]), data["hands"][:, None], data["discards"][:, None], (data["money"] / 5.0)[:, None]], 1).astype(np.float32))
    ANTE, BLIND = data["ante"], data["blind"]
    tr_idx = torch.from_numpy(np.flatnonzero(tr)).to(dev)
    ho_idx = np.flatnonzero(ho)
    net = JudgeNet(a.kind, a.attn, a.tok, spec=not a.no_spec).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr_idx) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    amp = str(dev).startswith("cuda")
    log_path = a.out.replace(".pt", "_log.jsonl")

    def loss_fn(out, y):
        if a.kind == "judge":
            l_pass = F.binary_cross_entropy_with_logits(out[:, 0], y[:, 0])
            l_aux = F.smooth_l1_loss(out[:, 1:], y[:, 1:])
            return l_pass + 0.3 * l_aux, l_pass
        l = F.smooth_l1_loss(out, y)
        return l, l

    def evaluate():
        net.eval()
        outs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            for s0 in range(0, len(ho_idx), 8192):
                ii = torch.from_numpy(ho_idx[s0:s0 + 8192]).to(dev)
                outs.append(net(G[ii], JID[ii], JF[ii], CID[ii], MODE[ii]).float())
        out = torch.cat(outs).cpu().numpy()
        net.train()
        res = {}
        if a.kind == "judge":
            p = 1 / (1 + np.exp(-out[:, 0]))
            y = data["pass"][ho_idx]
            ece, brier, table = calibration(p, y)
            res = {"ece": round(ece, 4), "brier": round(brier, 4), "auc": round(auc(p, y), 4), "base_brier": round(float(np.var(y)), 4),
                   "mae_ratio": round(float(np.abs(out[:, 1] - data["ratio"][ho_idx]).mean()), 4), "table": table}
            by = {}
            for an in range(1, 9):
                m = ANTE[ho_idx] == an
                if m.sum() > 200:
                    e2, b2, _ = calibration(p[m], y[m])
                    by[an] = {"n": int(m.sum()), "ece": round(e2, 4), "brier": round(b2, 4), "auc": round(auc(p[m], y[m]), 3), "mean_pass": round(float(y[m].mean()), 3), "mean_pred": round(float(p[m].mean()), 3)}
            res["by_ante"] = by
        else:
            y = Y[torch.from_numpy(ho_idx).to(dev)].cpu().numpy()
            res = {"mae_dc_slog": round(float(np.abs(out[:, :N_SLOTS] - y[:, :N_SLOTS]).mean()), 4), "base_mae_dc": round(float(np.abs(y[:, :N_SLOTS] - np.median(y[:, :N_SLOTS], 0)).mean()), 4),
                   "mae_hands": round(float(np.abs(out[:, N_SLOTS] - y[:, N_SLOTS]).mean()), 4), "mae_discards": round(float(np.abs(out[:, N_SLOTS + 1] - y[:, N_SLOTS + 1]).mean()), 4),
                   "mae_money": round(float(np.abs(out[:, N_SLOTS + 2] - y[:, N_SLOTS + 2]).mean() * 5.0), 4)}
        return res

    net.train()
    step = 0
    for ep in range(a.epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        run = []
        for s0 in range(0, len(perm) - a.batch + 1, a.batch):
            ii = perm[s0:s0 + a.batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = net(G[ii], JID[ii], JF[ii], CID[ii], MODE[ii])
            loss, main = loss_fn(out.float(), Y[ii])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                run.append(float(main))
        res = evaluate()
        row = {"epoch": ep + 1, "step": step, "train_loss": round(float(np.mean(run)), 4) if run else None, "min": round((time.time() - t0) / 60, 1), **res}
        print(json.dumps({k: v for k, v in row.items() if k != "table"}), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        save(a.out, net, {"epoch": ep + 1, "eval": {k: v for k, v in res.items() if k != "table"}})
    return net


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--kind", choices=["judge", "scaling"], default="judge")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--attn", type=int, default=2)
    p.add_argument("--tok", type=int, default=128)
    p.add_argument("--no-spec", action="store_true")
    p.add_argument("--held-out", type=float, default=0.05)
    p.add_argument("--repeat", default="", help="comma list: how many times each --data directory counts")
    p.add_argument("--max-shards", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse())
