"""Fits the run eval's weights (balatro_rl/runeval.py) on labelled states (tools/runeval_data.py) and measures it, as the spec asks:

  1. the nine terms are computed once per state;
  2. a fixed held-out set is split off by game (states of one game stay together; --holdout share);
  3. each phase potential is a logistic regression of its label on the standardised terms, ridge-regularised toward the hand-set priors:
       early  P(reach ante 4 with $10+) on states at antes 1-3 (--early-label reach4: without the money condition)
       mid    P(reach ante 7) on states at antes 1-6
       late   P(win) on every state
  4. the ante blend (centres, width, the conversion of the early and mid potentials to a win chance) is fitted last against the win label;
  5. metrics on the held-out states: Spearman rank correlation of the score with the simulated win rate (and with the mean ante reached),
     calibration error (10 equal-count bins), per-phase Spearman against each phase's own label; the power term against the engine's
     fast_power probe;
  6. ablations: each term removed in turn and the phases refitted.
The weights go to balatro_rl/data/runeval_weights.json (RunEval loads them), the report to traces/runeval_report.md.

    python tools/runeval_fit.py data/runeval/states_v1.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from balatro_rl.runeval import RunEval, TERMS, PHASES, PRIOR_W, PRIOR_B, PRIOR_BLEND, default_weights, blend_weights, sigmoid, WEIGHTS_PATH   # noqa: E402


def rankdata(x):
    """Average ranks (ties share their mean rank)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def calibration_error(p, y, w=None, bins=10):
    """Expected calibration error over equal-count bins: sum of |mean p - mean y| weighted by the bin's share."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = np.ones(len(p)) if w is None else np.asarray(w, dtype=np.float64)
    order = np.argsort(p)
    edges = np.linspace(0, len(p), bins + 1).astype(int)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = order[lo:hi]
        if len(idx) == 0:
            continue
        ww = w[idx]
        err += ww.sum() / w.sum() * abs(np.average(p[idx], weights=ww) - np.average(y[idx], weights=ww))
    return float(err)


def logloss(p, y, w=None):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(ll, weights=w))


def fit_logistic(Z, y, w, w0, b0, lam=1.0, iters=50):
    """Ridge-regularised logistic regression with soft labels y in [0, 1] and sample weights w, by Newton's method.
    Z standardised features [n, k]; the penalty is lam x |beta - w0|^2 (toward the prior, in standardised units); the intercept starts at b0."""
    n, k = Z.shape
    X = np.concatenate([Z, np.ones((n, 1))], 1)
    beta = np.concatenate([w0, [b0]])
    prior = np.concatenate([w0, [b0]])
    pen = np.full(k + 1, lam)
    pen[-1] = 1e-3 * lam                                                   # the intercept is nearly free
    W = w / w.mean()

    def objective(b):
        q = np.clip(sigmoid(X @ b), 1e-9, 1 - 1e-9)
        return float(-(W * (y * np.log(q) + (1 - y) * np.log(1 - q))).sum() + (pen * (b - prior) ** 2).sum())

    obj = objective(beta)
    for _ in range(iters):
        p = sigmoid(X @ beta)
        g = X.T @ (W * (p - y)) + 2 * pen * (beta - prior)
        s = W * p * (1 - p)
        H = (X * s[:, None]).T @ X + 2 * np.diag(pen) + 1e-6 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        t = 1.0                                                            # damped Newton: halve the step until the objective falls
        while t > 1e-4:
            cand = beta - t * step
            o = objective(cand)
            if np.isfinite(o) and o <= obj:
                break
            t *= 0.5
        if t <= 1e-4:
            break
        beta, obj = cand, o
        if np.abs(t * step).max() < 1e-7:
            break
    return beta[:-1], beta[-1]


def phase_sets(d, early_label, late_label="final_boss"):
    """The states each phase is fitted on, with their labels. late_label "win": the share of rollouts that win; "final_boss": the engine's probe, the
    build with its counters projected to the ante-8 boss and played by the exact round solver (min(1, share of the target), states at a blind select)."""
    ante = d["ante"].astype(int)
    econ = d["reach4"] * d["econ4"] if early_label == "reach4_econ" else d["reach4"]
    fb = d["final_boss"][:, 1]
    if late_label == "final_boss":
        late = (np.isfinite(fb), np.where(np.isfinite(fb), fb, 0.0))
    else:
        late = (np.ones(len(ante), dtype=bool), d["win"])
    return {"early": (ante <= 3, econ), "mid": (ante <= 6, d["reach7"]), "late": late}


def fit_phases(T, d, train, early_label, lam, drop=None, late_label="final_boss"):
    """-> weights dict (per phase: w, b, mean, std) fitted on the training states; `drop`: a term index set to zero (ablations)."""
    out = {}
    for ph in PHASES:
        sel, y = phase_sets(d, early_label, late_label)[ph]
        m = sel & train
        Tm = T[m]
        mu, sd = Tm.mean(0), Tm.std(0) + 1e-6
        Z = (Tm - mu) / sd
        w0 = np.array([PRIOR_W[ph][t] for t in TERMS]) * sd                 # the prior, in standardised units
        if drop is not None:
            Z = Z.copy()
            Z[:, drop] = 0.0
            w0 = w0.copy()
            w0[drop] = 0.0
        w, b = fit_logistic(Z, y[m], d["n_roll"][m], w0, PRIOR_B[ph], lam)
        if drop is not None:
            w[drop] = 0.0
        out[ph] = {"w": w.tolist(), "b": float(b), "mean": mu.tolist(), "std": sd.tolist(), "n": int(m.sum())}
    return out


def fit_blend(ev, T, d, train):
    """The ante blend: centres, width and the early / mid conversions, by a coarse search on the win label's log loss."""
    pot = ev.potentials(T)
    ante = d["ante"].astype(float)
    y = d["win"]
    best = None
    for c_e in (2.5, 3.0, 3.5, 4.0, 4.5):
        for c_l in (5.5, 6.0, 6.5, 7.0, 7.5):
            for tau in (0.4, 0.7, 1.0, 1.5):
                bw = blend_weights(ante[train], dict(c_early=c_e, c_late=c_l, tau=tau))
                # the conversions: least-squares on the win label given the blend (both in [0, 1])
                A = np.stack([bw[:, 0] * pot[train, 0], bw[:, 1] * pot[train, 1]], 1)
                rhs = y[train] - bw[:, 2] * pot[train, 2]
                conv, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                conv = np.clip(conv, 0.0, 1.0)
                p = A @ conv + bw[:, 2] * pot[train, 2]
                ll = logloss(np.clip(p, 0, 1), y[train], d["n_roll"][train])
                if best is None or ll < best[0]:
                    best = (ll, dict(c_early=c_e, c_late=c_l, tau=tau, conv_early=float(conv[0]), conv_mid=float(conv[1])))
    return best[1], best[0]


def evaluate(ev, T, d, test, early_label, e_deal=None, late_label="final_boss"):
    """Held-out metrics."""
    out = {}
    r = score_all(ev, T, d)
    ante = d["ante"].astype(int)
    e_deal = np.ones(len(ante)) if e_deal is None else e_deal
    out["spearman_win"] = spearman(r["win"][test], d["win"][test])
    out["spearman_blend_win"] = spearman(r["score"][test], d["win"][test])
    out["spearman_ante_mean"] = spearman(r["score"][test], d["ante_mean"][test])
    out["spearman_blinds_mean"] = spearman(r["score"][test], d["blinds_mean"][test])
    out["calibration_win"] = calibration_error(r["win"][test], d["win"][test], d["n_roll"][test])
    out["logloss_win"] = logloss(r["win"][test], d["win"][test], d["n_roll"][test])
    out["per_ante"] = {}
    fb = d["final_boss"][:, 1]
    for a in range(1, 9):
        m = test & (ante == a)
        if m.sum() >= 30:
            mf = m & np.isfinite(fb)
            out["per_ante"][a] = dict(n=int(m.sum()), spearman_win=spearman(r["win"][m], d["win"][m]), spearman_ante=spearman(r["score"][m], d["ante_mean"][m]),
                                      spearman_blinds=spearman(r["score"][m], d["blinds_mean"][m]), spearman_late_fb=spearman(r["late"][mf], fb[mf]),
                                      spearman_mid_ante=spearman(r["mid"][m], d["ante_mean"][m]), win_rate=float(d["win"][m].mean()), pred=float(r["win"][m].mean()),
                                      ante_mean=float(d["ante_mean"][m].mean()))
    out["phases"] = {}
    for k, ph in enumerate(PHASES):
        sel, y = phase_sets(d, early_label, late_label)[ph]
        m = sel & test
        out["phases"][ph] = dict(n=int(m.sum()), spearman=spearman(r[ph][m], y[m]), calibration=calibration_error(r[ph][m], y[m], d["n_roll"][m]),
                                 logloss=logloss(r[ph][m], y[m], d["n_roll"][m]), label_mean=float(y[m].mean()), pred_mean=float(r[ph][m].mean()))
    # the power term against the engine's probe (states at a blind select have one): overall and within each ante
    fp = d["fast_power"]
    ok = test & np.isfinite(fp[:, 0])
    k_power = TERMS.index("power")
    out["power_vs_fast_power"] = spearman(T[ok, k_power], np.log10(np.maximum(fp[ok, 0], 1e-3)))
    out["deal_vs_engine_best_play"] = spearman(np.log10(np.maximum(e_deal[ok], 1.0)), np.log10(np.maximum(fp[ok, 2], 1.0)))
    out["deal_vs_engine_by_ante"] = {int(a): spearman(np.log10(np.maximum(e_deal[ok & (ante == a)], 1.0)), np.log10(np.maximum(fp[ok & (ante == a), 2], 1.0)))
                                     for a in range(1, 9) if (ok & (ante == a)).sum() >= 30}
    out["deal_median_ratio"] = float(np.median(e_deal[ok] / np.maximum(fp[ok, 2], 1.0)))
    out["power_vs_final_boss"] = spearman(T[ok, k_power] + T[ok, TERMS.index("scaling")], d["final_boss"][ok, 1])
    return out


def score_all(ev, T, d):
    pot = ev.potentials(T)
    bw = blend_weights(d["ante"].astype(float), ev.w["blend"])
    conv = np.array([ev.w["blend"]["conv_early"], ev.w["blend"]["conv_mid"], 1.0])
    return dict(score=(bw * pot).sum(1), win=(bw * pot * conv[None]).sum(1), early=pot[:, 0], mid=pot[:, 1], late=pot[:, 2])


def compute_terms(d, chunk=4096):
    ev = RunEval(default_weights())
    n = len(d["ante"])
    T = np.zeros((n, len(TERMS)))
    main = np.zeros(n, dtype=np.int64)
    e_deal = np.zeros(n)
    for c0 in range(0, n, chunk):
        sl = slice(c0, min(n, c0 + chunk))
        Tc, s, dg = ev.terms(d["g"][sl], d["jid"][sl], d["jf"][sl].astype(np.float32), d["cid"][sl], want_deal=True)
        T[sl] = Tc
        main[sl] = dg["main"]
        e_deal[sl] = dg["e_deal"]
    return T, main, e_deal


def load(paths):
    """One or more state files (tools/runeval_data.py), concatenated; n_roll per state from each file's meta."""
    import ast as _ast
    parts, metas = [], []
    for path in paths:
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files if k != "meta"}
        meta = str(z["meta"]) if "meta" in z.files else ""
        m = _ast.literal_eval(meta) if meta.startswith("{") else {}
        d["n_roll"] = np.full(len(d["ante"]), float(m.get("rollouts", 16)))
        parts.append(d)
        metas.append(f"{os.path.basename(path)}: {len(d['ante'])} states, {m}")
    d = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    return d, "; ".join(metas)


def report_md(path, args, d, meta, weights, metrics, ablations, main_hist, t_terms, n):
    from balatro_rl.sim.hands import HAND_NAMES
    L = []
    L.append("# Run eval: fitted weights and metrics\n")
    L.append(f"Data: {meta} ({n} states). Held-out share {args.holdout}; ridge {args.lam} toward the priors; early label `{args.early_label}`, late label `{args.late_label}`"
             + (" (the rollouts win too rarely under the simulator's policy to fit the late potential on them: see the per-phase table)" if args.late_label == "final_boss" else "") + ".\n")
    L.append(f"Term computation: {1e6 * t_terms / n:.0f} us per state (batches of 4096, numpy, one core).\n")
    L.append("\n## Weights per phase (logistic regression on standardised terms; `raw` = per unit of the term)\n")
    L.append("| term | " + " | ".join(f"{ph} std / raw / prior" for ph in PHASES) + " |")
    L.append("|---|" + "---|" * len(PHASES))
    for k, t in enumerate(TERMS):
        cells = []
        for ph in PHASES:
            w = weights["phases"][ph]
            cells.append(f"{w['w'][k]:+.2f} / {w['w'][k] / w['std'][k]:+.2f} / {PRIOR_W[ph][t]:.2f}")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("| intercept | " + " | ".join(f"{weights['phases'][ph]['b']:+.2f} (n={weights['phases'][ph]['n']})" for ph in PHASES) + " |")
    b = weights["blend"]
    L.append(f"\nBlend: early centre {b['c_early']}, late centre {b['c_late']}, width {b['tau']}; conversions early {b['conv_early']:.3f}, mid {b['conv_mid']:.3f}.\n")
    L.append("\n## Held-out metrics\n")
    L.append("| metric | value | spec target |\n|---|---|---|")
    L.append(f"| Spearman, win estimate vs simulated win rate | {metrics['spearman_win']:.3f} | >= 0.80 |")
    L.append(f"| Spearman, blended score vs simulated win rate | {metrics['spearman_blend_win']:.3f} | |")
    L.append(f"| Spearman, blended score vs mean ante reached | {metrics['spearman_ante_mean']:.3f} | |")
    L.append(f"| Spearman, blended score vs mean blinds beaten | {metrics['spearman_blinds_mean']:.3f} | |")
    L.append(f"| calibration error, win estimate | {metrics['calibration_win']:.3f} | < 0.05 |")
    L.append(f"| log loss, win estimate | {metrics['logloss_win']:.3f} | |")
    L.append(f"| Spearman, power term vs engine fast_power | {metrics['power_vs_fast_power']:.3f} | |")
    L.append(f"| Spearman, expected best play of the dealt hand vs the engine's measured best play | {metrics['deal_vs_engine_best_play']:.3f} (by ante: "
             + ", ".join(f"{a}: {v:.2f}" for a, v in metrics["deal_vs_engine_by_ante"].items()) + f"); median ratio eval / engine {metrics['deal_median_ratio']:.2f} | |")
    L.append(f"| Spearman, power + scaling vs engine final-boss share | {metrics['power_vs_final_boss']:.3f} | |")
    L.append("\n### Per phase (each potential against its own label)\n")
    L.append("| phase | n | Spearman | calibration | log loss | label mean | predicted mean |\n|---|---|---|---|---|---|---|")
    for ph in PHASES:
        m = metrics["phases"][ph]
        L.append(f"| {ph} | {m['n']} | {m['spearman']:.3f} | {m['calibration']:.3f} | {m['logloss']:.3f} | {m['label_mean']:.3f} | {m['pred_mean']:.3f} |")
    L.append("\n### Per ante (held-out): does the eval order the states of one ante correctly?\n")
    L.append("| ante | n | blended score vs mean ante reached | vs mean blinds beaten | mid potential vs mean ante reached | late potential vs final-boss probe | win estimate vs win rate | win rate | mean ante reached |\n|---|---|---|---|---|---|---|---|---|")
    for a, m in metrics["per_ante"].items():
        L.append(f"| {a} | {m['n']} | {m['spearman_ante']:.3f} | {m['spearman_blinds']:.3f} | {m['spearman_mid_ante']:.3f} | {m['spearman_late_fb']:.3f} | {m['spearman_win']:.3f} | {m['win_rate']:.4f} | {m['ante_mean']:.2f} |")
    L.append("\n## Ablations (held-out, each term removed and the phases refitted)\n")
    L.append("| removed | Spearman win | calibration | log loss | early Spearman | mid Spearman | late Spearman |\n|---|---|---|---|---|---|---|")
    L.append(f"| (none) | {metrics['spearman_win']:.3f} | {metrics['calibration_win']:.3f} | {metrics['logloss_win']:.3f} | "
             + " | ".join(f"{metrics['phases'][ph]['spearman']:.3f}" for ph in PHASES) + " |")
    for t, m in ablations.items():
        L.append(f"| {t} | {m['spearman_win']:.3f} | {m['calibration_win']:.3f} | {m['logloss_win']:.3f} | " + " | ".join(f"{m['phases'][ph]['spearman']:.3f}" for ph in PHASES) + " |")
    L.append("\n## Main hand the eval picks (all states)\n")
    L.append(", ".join(f"{HAND_NAMES[h]} {c}" for h, c in enumerate(main_hist) if c))
    L.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="*", default=["data/runeval/states_v1.npz"])
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--lam", type=float, default=1.0, help="ridge weight toward the priors (per standardised unit)")
    p.add_argument("--early-label", default="reach4_econ", choices=["reach4_econ", "reach4"])
    p.add_argument("--late-label", default="final_boss", choices=["final_boss", "win"],
                   help="final_boss: the engine's ante-8 probe (the simulator's exact solver on the projected build); win: the rollouts")
    p.add_argument("--win-conv", action="store_true", help="also convert the early and mid potentials to win chances in the blend (needs wins in the data)")
    p.add_argument("--out", default=WEIGHTS_PATH)
    p.add_argument("--report", default="traces/runeval_report.md")
    p.add_argument("--no-ablations", action="store_true")
    args = p.parse_args()
    d, meta = load(args.data)
    n = len(d["ante"])
    t0 = time.time()
    T, main, e_deal = compute_terms(d)
    t_terms = time.time() - t0
    print(f"terms for {n} states in {t_terms:.1f} s ({1e6 * t_terms / n:.0f} us/state)")
    rng = np.random.default_rng(0)
    games = d["game"]
    ug = np.unique(games)
    hold_games = set(rng.choice(ug, size=int(len(ug) * args.holdout), replace=False).tolist())
    test = np.array([g in hold_games for g in games])
    train = ~test
    print(f"train {train.sum()} / held-out {test.sum()} states ({len(hold_games)} of {len(ug)} games)")
    weights = default_weights()
    weights["phases"] = fit_phases(T, d, train, args.early_label, args.lam, late_label=args.late_label)
    ev = RunEval(weights)
    if args.win_conv:
        weights["blend"], ll = fit_blend(ev, T, d, train)
    else:                                                                             # the spec's blend as given: the potentials mixed by ante, no win conversion
        weights["blend"] = dict(PRIOR_BLEND, conv_early=1.0, conv_mid=1.0)
    ev = RunEval(weights)
    metrics = evaluate(ev, T, d, test, args.early_label, e_deal, args.late_label)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_ante", "phases")}, indent=1))
    for ph in PHASES:
        print(ph, metrics["phases"][ph])
    ablations = {}
    if not args.no_ablations:
        for k, t in enumerate(TERMS):
            w2 = default_weights()
            w2["phases"] = fit_phases(T, d, train, args.early_label, args.lam, drop=k, late_label=args.late_label)
            w2["blend"] = weights["blend"]
            ablations[t] = evaluate(RunEval(w2), T, d, test, args.early_label, e_deal, args.late_label)
            print(f"  without {t}: Spearman win {ablations[t]['spearman_win']:.3f} (full {metrics['spearman_win']:.3f})")
    weights["fitted"] = True
    weights["metrics"] = {k: v for k, v in metrics.items() if k not in ("per_ante",)}
    weights["source"] = dict(data=args.data, meta=meta, n=int(n), holdout=args.holdout, lam=args.lam, early_label=args.early_label, late_label=args.late_label)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=1)
    report_md(args.report, args, d, meta, weights, metrics, ablations, np.bincount(main, minlength=12), t_terms, n)
    print(f"-> {args.out}, {args.report}")


if __name__ == "__main__":
    main()
