"""Readable summary of a training log (the *_log.jsonl that az.train run writes).

    python -m balatro_rl.az.report                       # checkpoints/az_log.jsonl
    python -m balatro_rl.az.report --log other_log.jsonl --last 10
    python -m balatro_rl.az.report --detail              # also losses and shaping per iteration
"""
from __future__ import annotations

import argparse
import json


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def report(rows: list[dict], last: int = 0, detail: bool = False):
    if not rows:
        print("no iterations logged yet")
        return
    show = rows[-last:] if last else rows
    print(f"{'it':>3} {'step':>7} {'srch':>4} {'min':>5} | {'self-play':^20} | {'held-out eval':^26} | notes")
    print(f"{'':>3} {'':>7} {'':>4} {'':>5} | {'win%':>6} {'blinds':>6} {'ante':>5} | {'win%':>6} {'blinds':>7} {'ante':>5} {'calib':>6} |")
    for r in show:
        ev = r.get("eval")
        mins = r.get("gen_min", 0) + r.get("train_min", 0)
        evs = (f"{ev['win%']:6.1f} {ev['blinds']:7.2f} {ev['ante']:5.2f} {ev['calibration_v']['status'][:6]:>6}"
               if ev else f"{'':>6} {'':>7} {'':>5} {'':>6}")
        notes = []
        if "ALARM" in r:
            notes.append("ALARM")
        if r.get("flags"):
            notes.append(f"{len(r['flags'])} flag(s)")
        if "WARNING" in r:
            notes.append("WARNING")
        print(f"{r['iter']:>3} {r['step']:>7} {'yes' if r['search'] else 'no':>4} {mins:5.1f} | "
              f"{r['win%']:6.1f} {r['blinds']:6.2f} {r['ante']:5.2f} | {evs} | {' '.join(notes)}")
        if detail:
            losses = {k[5:]: v for k, v in r.items() if k.startswith("loss_")}
            print("      losses " + "  ".join(f"{k} {v:.4f}" for k, v in losses.items()))
            print("      shaping " + "  ".join(f"{k} {v:+.4f}" for k, v in r["shaping"].items()))
            g = r.get("lambda_gate")
            gate = (f"  (lambda clock {r.get('lambda_clock', 0):,}, gate {'open' if g['open'] else 'closed'}: "
                    f"win rate {g['win_rate']} vs {g['threshold']})") if g else ""
            print(f"      lam {r['lam']:.3f}{gate}  beta {r['beta']:.4f}  kappa {r['kappa']:.3f}  sims/decision "
                  f"{r['sims/decision']}  autoplay {r['autoplay%']}%  headroom {r['headroom_ms/decision']} ms/decision")
    last_r = rows[-1]
    evals = [r for r in rows if "eval" in r]
    print()
    total_min = sum(r.get("gen_min", 0) + r.get("train_min", 0) for r in rows)
    total_min += sum(0 for _ in evals)
    print(f"{len(rows)} iterations, {last_r['step']:,} decisions; schedule now lam {last_r['lam']:.3f}, "
          f"beta {last_r['beta']:.4f}, kappa {last_r['kappa']:.3f}")
    g = last_r.get("lambda_gate")
    if g:
        print(f"lambda clock {last_r.get('lambda_clock', 0):,} decisions; gate {'OPEN' if g['open'] else 'closed'} "
              f"(self-play win rate {g['win_rate']} vs threshold {g['threshold']}): lambda only fades while open")
    searched = [r for r in rows if r["search"]]
    if searched:
        m = sum(r["gen_min"] + r["train_min"] for r in searched[-5:]) / len(searched[-5:])
        print(f"recent iteration time {m:.1f} min (+ evaluations every few iterations)")
    if evals:
        best = max(evals, key=lambda r: (r["eval"]["win%"], r["eval"]["blinds"]))
        e = evals[-1]["eval"]
        print(f"latest held-out: {e['win%']:.1f}% wins, {e['blinds']:.2f} blinds (iteration {evals[-1]['iter']}); "
              f"best: {best['eval']['win%']:.1f}% wins, {best['eval']['blinds']:.2f} blinds (iteration {best['iter']})")
        print(f"  where latest runs ended: {e['breakdown']['by_ante']}")
        bosses = e["breakdown"]["by_boss"]
        hard = sorted(bosses.items(), key=lambda kv: kv[1]["cleared%"])[:5]
        print("  hardest bosses (cleared %): " + ", ".join(f"{k} {v['cleared%']:.0f}% of {v['met']}" for k, v in hard))
        for name in ("calibration_phi", "calibration_v"):
            c = e[name]
            print(f"  {name}: {c['status']} ({c.get('won_games', 0)} won games)")
    else:
        print("no held-out evaluation yet (every --eval-every iterations)")
    alarms = [r for r in rows if "ALARM" in r]
    for r in alarms:
        print(f"ALARM at iteration {r['iter']}: {r['ALARM']}")
    for r in rows:
        for f in r.get("flags", []):
            print(f"flag at iteration {r['iter']}: {f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="checkpoints/az_log.jsonl")
    p.add_argument("--last", type=int, default=0, help="only the last N iterations")
    p.add_argument("--detail", action="store_true", help="losses, shaping and schedule per iteration")
    a = p.parse_args()
    report(load(a.log), a.last, a.detail)


if __name__ == "__main__":
    main()
