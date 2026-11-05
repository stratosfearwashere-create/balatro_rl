"""A permanent diagnostic loop over a running PPO trainer (CPU only: the trainer owns the GPU).

Every --every minutes: copy the run's newest checkpoint (a stable snapshot), then on CPU
  1. greedy Gold games on two fixed seed blocks (10000+ and 10300+, --games each): blinds, wins, ante reached;
  2. tools/loss_analysis.py on the first block: where it loses, how close, the build at the end (xMult jokers, hand level, rerolls, packs, consumables);
  3. tools/offer_take.py: when a strong joker is legal to take, what the policy does.
The compact result is appended to traces/agent_journal.md (and traces/diag_loop.jsonl), with the change against the previous round and against the reference checkpoint.

    python tools/diag_loop.py --run checkpoints/ppo_curr_v1.pt --ref checkpoints/ppo_gold_v30_best.pt --every 30
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PY = sys.executable


def run(args, timeout=3600):
    r = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return r.stdout + r.stderr


def eval_block(ck, seed0, games):
    out = run(["tools/loss_analysis.py", ck, "--games", str(games), "--seed0", str(seed0), "--stake", "GOLD", "--device", "cpu"])
    m = re.search(r"(\d+) wins \(([\d.]+)%\), ([\d.]+) blinds beaten on average", out)
    res = {"wins": int(m.group(1)), "blinds": float(m.group(3))} if m else {"error": out[-400:]}
    keep = {}
    for key, pat in (("lost_ante_1_3", r"lost ante 1-3"),):
        pass
    for line in out.splitlines():
        s = line.strip()
        for name in ("xMult jokers", "games with no xMult joker %", "level of the most played hand", "rerolls per game", "packs bought per game", "consumables used per game",
                     "money at the end", "most money held", "vouchers", "blinds skipped"):
            if s.startswith(name + " ") or s == name:
                nums = re.findall(r"-?[\d.]+|nan", s[len(name):])
                keep[name] = nums[-3:]                    # lost ante 6-8, 4-5, 1-3 (wins column is first when there are wins)
        if s.startswith("how close:"):
            keep["how close"] = s[len("how close: "):]
        if s.startswith("by blind:"):
            keep["by blind"] = s[len("by blind: "):]
    m2 = re.search(r"WHERE THE LOSSES HAPPEN(.*?)by blind", out, re.S)
    if m2:
        rows = re.findall(r"^\s+(\d)\s+(\d+)\s+([\d.]+)%", m2.group(1), re.M)
        res["loss_share_by_ante"] = {int(a): float(s) for a, _, s in rows}
    res["build"] = keep
    return res


def offer(ck, games):
    out = run(["tools/offer_take.py", ck, "--games", str(games)])
    m = re.search(r"^all\s+(\d+)\s+(.*)$", out, re.M)
    if not m:
        return {"error": out[-300:]}
    vals = [float(x) for x in re.findall(r"([\d.]+)%", m.group(2))]
    names = ["bought", "swapped_in", "bought_other", "swapped_other", "pack_voucher", "rerolled", "left", "other"]
    return {"steps": int(m.group(1)), **dict(zip(names, vals))}


def journal(lines):
    with open(os.path.join(ROOT, "traces", "agent_journal.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="checkpoints/ppo_curr_v1.pt")
    p.add_argument("--ref", default="checkpoints/ppo_gold_v30_best.pt")
    p.add_argument("--every", type=float, default=30.0, help="minutes between rounds")
    p.add_argument("--games", type=int, default=150)
    p.add_argument("--rounds", type=int, default=0, help="0: forever")
    a = p.parse_args()
    snap_dir = os.path.join(ROOT, "checkpoints", "diag")
    os.makedirs(snap_dir, exist_ok=True)
    log_path = os.path.join(ROOT, "traces", "diag_loop.jsonl")
    prev = None
    ref = None
    k = 0
    journal([f"", f"## Diagnostic loop (tools/diag_loop.py), run {a.run}, reference {a.ref}, {a.games} greedy Gold games per block, started {time.strftime('%Y-%m-%d %H:%M')}"])
    while a.rounds == 0 or k < a.rounds:
        k += 1
        t0 = time.time()
        if ref is None:
            ref = {"A": eval_block(a.ref, 10000, a.games), "B": eval_block(a.ref, 10300, a.games), "offer": offer(a.ref, a.games)}
            journal([f"- reference {os.path.basename(a.ref)}: block A {ref['A'].get('blinds')} blinds / {ref['A'].get('wins')} wins, block B {ref['B'].get('blinds')} / {ref['B'].get('wins')}; "
                     f"offer: {ref['offer']}"])
        src = os.path.join(ROOT, a.run)
        it = None
        logp = src.replace(".pt", "_log.jsonl")
        try:
            last = open(logp).read().strip().splitlines()[-1]
            it = json.loads(last)["iter"]
        except Exception:
            pass
        snap = os.path.join(snap_dir, f"{os.path.basename(src)[:-3]}_u{it}.pt")
        shutil.copyfile(src, snap)
        res = {"time": time.strftime("%Y-%m-%d %H:%M"), "iter": it, "A": eval_block(snap, 10000, a.games), "B": eval_block(snap, 10300, a.games), "offer": offer(snap, a.games)}
        with open(log_path, "a") as f:
            f.write(json.dumps(res) + "\n")

        def d(x, y):
            try:
                return f"{x - y:+.2f}"
            except Exception:
                return "?"
        bA, bB = res["A"].get("blinds"), res["B"].get("blinds")
        mean_run = (bA + bB) / 2 if bA is not None and bB is not None else None
        mean_ref = (ref["A"].get("blinds", 0) + ref["B"].get("blinds", 0)) / 2
        lines = [f"- {res['time']} update {it}: blinds A {bA} B {bB} (mean {mean_run if mean_run is None else round(mean_run, 2)}; vs reference {d(mean_run, mean_ref)}"
                 + (f", vs last round {d(mean_run, prev)}" if prev is not None else "") + f"), wins {res['A'].get('wins')}+{res['B'].get('wins')}",
                 f"  loss share by ante (A): {res['A'].get('loss_share_by_ante')}; {res['A']['build'].get('by blind', '')}; close: {res['A']['build'].get('how close', '')}",
                 f"  build (lost ante 6-8 / 4-5 / 1-3): " + "; ".join(f"{n} {v}" for n, v in res["A"]["build"].items() if n not in ("by blind", "how close")),
                 f"  strong joker legal to take: {res['offer']}  ({time.time() - t0:.0f}s)"]
        journal(lines)
        prev = mean_run
        wait = a.every * 60 - (time.time() - t0)
        if wait > 0 and (a.rounds == 0 or k < a.rounds):
            time.sleep(wait)


if __name__ == "__main__":
    main()
