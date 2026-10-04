"""Overnight, unattended and restartable (every step skips when its output exists). Log: overnight_chain.log,
summary: overnight_summary.txt (both in checkpoints/eval).

  1. round-2 A/B: value_r2.pt in the shop (weights 200/300, margins 1/2), seeds 10000-10299 and 10300-10599
  2. hand-built fixes from the usage stats, each on both blocks: rule_skip, Hieroglyph/Petroglyph devalued,
     interest_rounds 4, dollar 0.012; a combination of those that gain > 1 paired SE on both blocks
  3. 20k more games with the best hand-built config (seeds 1200000+), merged -> 50k
  4. train value_r3.pt on everything
  5. A/B value_r3.pt against the best hand-built config on both blocks
"""
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = r"C:\balatro_rl"
EV = os.path.join(ROOT, "checkpoints", "eval")
LOG = os.path.join(EV, "overnight_chain.log")
SUMMARY = os.path.join(EV, "overnight_summary.txt")
PY = sys.executable
if ROOT not in sys.path:                       # the chain runs from checkpoints/eval: make the package importable
    sys.path.insert(0, ROOT)
BLOCKS = (("a", 10000, "value_ab"), ("b", 10300, "value_repl"))


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def note(text):
    with open(SUMMARY, "a") as f:
        f.write(text.rstrip() + "\n")


def run(args, out):
    with open(out, "a") as f:
        return subprocess.run([PY, "-u"] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def grid_run(name, grid, base_name="graded"):
    """Run a grid on both blocks (skipping finished variants); return {block: {variant: result dict}}."""
    gp = os.path.join(ROOT, "eval_grids", f"{name}.json")
    json.dump(grid, open(gp, "w"), indent=1)
    res = {}
    for blk, seed0, prev in BLOCKS:
        out = os.path.join(EV, f"{name}_{blk}")
        os.makedirs(out, exist_ok=True)
        src = os.path.join(EV, prev, "graded.json")
        if base_name == "graded" and os.path.exists(src) and not os.path.exists(os.path.join(out, "graded.json")):
            shutil.copy(src, os.path.join(out, "graded.json"))
        rc = run(["-m", "balatro_rl.az.compare", "grid", "--grid", gp, "--out", out, "--games", "300",
                  "--workers", "7", "--seed0", str(seed0)], os.path.join(EV, f"{name}_{blk}.log"))
        run(["-m", "balatro_rl.az.compare", "table", "--grid", gp, "--out", out, "--base", base_name],
            os.path.join(EV, f"{name}_{blk}_table.txt"))
        log(f"{name} block {blk} exit {rc}")
        res[blk] = {}
        for v in grid:
            p = os.path.join(out, v + ".json")
            if os.path.exists(p):
                res[blk][v] = json.load(open(p))
    return res


def paired(res_v, res_b):
    """(mean blinds difference, SE, wins v, wins base) over the seeds both have."""
    bv = {g[0]: g[1] for g in res_v["per_game"]}
    bb = {g[0]: g[1] for g in res_b["per_game"]}
    d = [bv[s] - bb[s] for s in bv if s in bb]
    if len(d) < 2:
        return 0.0, 99.0, 0, 0
    m = sum(d) / len(d)
    se = statistics.pstdev(d) / (len(d) ** 0.5)
    wv = sum(g[2] for g in res_v["per_game"])
    wb = sum(g[2] for g in res_b["per_game"])
    return m, se, wv, wb


def table(title, res, base="graded"):
    lines = [f"\n## {title}"]
    for blk in res:
        if base not in res[blk]:
            continue
        lines.append(f"block {blk}: {base} {res[blk][base]['blinds']:.2f} blinds, {res[blk][base]['win%']:.1f}% wins")
        for v, r in res[blk].items():
            if v == base:
                continue
            m, se, wv, wb = paired(r, res[blk][base])
            lines.append(f"  {v:28s} {r['blinds']:6.2f} blinds  {m:+.2f} +- {se:.2f}   wins {wv} vs {wb}")
    text = "\n".join(lines)
    note(text)
    return text


def gains_on_both(res, v, base):
    for blk in res:
        if v not in res[blk] or base not in res[blk]:
            return False
        m, se, _, _ = paired(res[blk][v], res[blk][base])
        if not m > se:
            return False
    return True


def main():
    os.chdir(ROOT)
    log("overnight chain started")
    base = {"shop_prior": "graded", "shop": {"rule_arcana": True}}
    kept_fixes = {}
    try:
        # ---- 1. round-2 A/B
        one = {**base, "shop_eval": "onestep", "value_model": "checkpoints/value_r2.pt", "value_mode": "antes",
               "budget_round": 0, "budget_boss": 0}
        g1 = {"graded": {"cfg": base, "search": False},
              "r2_w200": {"cfg": {**one, "onestep_weight": 200.0}},
              "r2_w300": {"cfg": {**one, "onestep_weight": 300.0}},
              "r2_m1": {"cfg": {**one, "onestep_margin": 1.0}},
              "r2_m2": {"cfg": {**one, "onestep_margin": 2.0}}}
        r1 = grid_run("round2", g1)
        table("1. value_r2.pt in the shop (30k-game model)", r1)
    except Exception:
        log("step 1 failed: " + traceback.format_exc())
    try:
        # ---- 2. hand-built fixes
        from balatro_rl.az.shop import VOUCHER_DOLLARS
        vd = dict(VOUCHER_DOLLARS)
        vd.update({"hieroglyph": -8.0, "petroglyph": -8.0})
        fixes = {"skip": {"shop": {"rule_arcana": True, "rule_skip": True}},
                 "hiero": {"shop": {"rule_arcana": True, "voucher_dollars": vd}},
                 "interest4": {"shop": {"rule_arcana": True, "interest_rounds": 4.0}},
                 "dollar12": {"shop": {"rule_arcana": True, "dollar": 0.012}}}
        g2 = {"graded": {"cfg": base, "search": False}}
        for k, over in fixes.items():
            g2[k] = {"cfg": {**base, **over}, "search": False}
        r2 = grid_run("fixes", g2)
        table("2. hand-built fixes from the usage stats", r2)
        good = [k for k in fixes if gains_on_both(r2, k, "graded")]
        note(f"fixes that gain > 1 SE on both blocks: {good}")
        if len(good) >= 2:
            shop = {"rule_arcana": True}
            for k in good:
                shop.update(fixes[k]["shop"])
            g3 = {"graded": {"cfg": base, "search": False}, "combo": {"cfg": {**base, "shop": shop}, "search": False}}
            r3 = grid_run("fixes_combo", g3)
            table("2b. combination of the kept fixes", r3)
            if gains_on_both(r3, "combo", "graded"):
                kept_fixes = {"shop": shop}
            else:
                kept_fixes = fixes[good[0]]
        elif len(good) == 1:
            kept_fixes = fixes[good[0]]
        note(f"best hand-built config: {json.dumps({**base, **kept_fixes})}")
    except Exception:
        log("step 2 failed: " + traceback.format_exc())
    best = {**base, **kept_fixes}
    try:
        # ---- 3. more games with the best hand-built config
        games_file = "checkpoints/value_data3/games.jsonl"
        have = sum(1 for _ in open(games_file)) if os.path.exists(games_file) else 0
        if have < 20000:
            rc = run(["-m", "balatro_rl.value.gen", "--games", "20000", "--seed0", "1200000", "--workers", "8",
                      "--chunk", "50", "--out", "checkpoints/value_data3", "--cfg", json.dumps(best)],
                     "checkpoints/value_gen3.log")
            log(f"generation 3 exit {rc}")
        n = 0
        for src in sorted(glob.glob("checkpoints/value_data3/chunk*.npz")):
            dst = os.path.join("checkpoints/value_data", "chunk2" + os.path.basename(src)[5:])
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                n += 1
        seen = {json.loads(l)["seed"] for l in open("checkpoints/value_data/games.jsonl")}
        with open("checkpoints/value_data/games.jsonl", "a") as f_out:
            for l in open(games_file):
                if json.loads(l)["seed"] not in seen:
                    f_out.write(l)
        log(f"merged {n} chunks of data3")
        run(["-m", "balatro_rl.value.stats", "--data", "checkpoints/value_data3", "--stake", "WHITE", "--top", "30"],
            os.path.join(EV, "usage_round3.txt"))
    except Exception:
        log("step 3 failed: " + traceback.format_exc())
    try:
        # ---- 4. train on everything
        if not os.path.exists("checkpoints/value_r3.pt"):
            rc = run(["-m", "balatro_rl.value.train", "--data", "checkpoints/value_data", "--out",
                      "checkpoints/value_r3.pt", "--epochs", "30"], "checkpoints/value_train_r3.log")
            log(f"train r3 exit {rc}")
        note("\n## 4. value_r3.pt training report")
        for line in open("checkpoints/value_train_r3.log"):
            if "beats_phi" in line or line.startswith("  ante"):
                note(line.rstrip())
    except Exception:
        log("step 4 failed: " + traceback.format_exc())
    try:
        # ---- 5. A/B value_r3 against the best hand-built config
        one = {**best, "shop_eval": "onestep", "value_model": "checkpoints/value_r3.pt", "value_mode": "antes",
               "budget_round": 0, "budget_boss": 0}
        g5 = {"best": {"cfg": best, "search": False},
              "r3_w200": {"cfg": {**one, "onestep_weight": 200.0}},
              "r3_w300": {"cfg": {**one, "onestep_weight": 300.0}},
              "r3_m1": {"cfg": {**one, "onestep_margin": 1.0}},
              "r3_m2": {"cfg": {**one, "onestep_margin": 2.0}}}
        r5 = grid_run("round3", g5, base_name="best")
        table("5. value_r3.pt (50k-game model) vs the best hand-built config", r5, base="best")
    except Exception:
        log("step 5 failed: " + traceback.format_exc())
    log("chain=done")
    note("\n(chain finished)")


if __name__ == "__main__":
    main()
