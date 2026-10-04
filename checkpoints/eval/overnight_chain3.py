"""Keeps the loop running after the fixes grid: adopts the kept hand-built fixes, then rounds of
play (25k games with the best agent) -> train -> A/B until DEADLINE. Summary appended to overnight_summary.txt."""
import glob
import json
import os
import re
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight_chain as C  # noqa: E402
import overnight_chain2 as C2  # noqa: E402

DEADLINE = time.mktime(time.strptime(time.strftime("%Y-%m-%d") + " 23:30", "%Y-%m-%d %H:%M"))
EV = C.EV


def wait_for(marker):
    while True:
        try:
            if marker in open(C.LOG).read():
                return
        except OSError:
            pass
        time.sleep(60)


def kept_fixes_from_summary():
    kept = []
    for line in open(C.SUMMARY):
        if line.startswith("fixes that gain > 1 SE on both blocks: "):
            kept = json.loads(line.split(": ", 1)[1].replace("'", '"'))
    return kept


def last_round():
    ks = [int(m.group(1)) for d in glob.glob(os.path.join(EV, "round*_a")) for m in [re.search(r"round(\d+)_a$", d)] if m]
    return max(ks) if ks else 3


def main():
    os.chdir(C.ROOT)
    wait_for("fixes=done")
    C.log("chain3 started")
    base = {"shop_prior": "graded", "shop": {"rule_arcana": True}}
    from balatro_rl.az.shop import VOUCHER_DOLLARS
    vd = dict(VOUCHER_DOLLARS)
    vd.update({"hieroglyph": -8.0, "petroglyph": -8.0})
    fixes = {"skip": {"rule_skip": True}, "hiero": {"voucher_dollars": vd}, "interest4": {"interest_rounds": 4.0},
             "dollar12": {"dollar": 0.012}, "boss": {"rule_boss": True}}
    kept = [k for k in kept_fixes_from_summary() if k in fixes]
    best = base
    if kept:
        shop = {"rule_arcana": True}
        for k in kept:
            shop.update(fixes[k])
        cand = {**base, "shop": shop}
        if len(kept) >= 2:
            res = C.grid_run("fixes_combo", {"graded": {"cfg": base, "search": False},
                                             "combo": {"cfg": cand, "search": False}})
            C.table("combination of the kept fixes", res)
            if C.gains_on_both(res, "combo", "graded"):
                best = cand
            else:                                   # the single best fix instead
                best = {**base, "shop": {"rule_arcana": True, **fixes[kept[0]]}}
        else:
            best = cand
    C.note(f"\nbest hand-built config (chain3): {json.dumps(best)}")
    changed = best != base
    k = last_round()
    seed0 = 1_300_000 + 100_000 * (k - 3)
    prev = k
    while time.time() < DEADLINE:
        k += 1
        try:
            variants_prev = [f"r{prev}_w200", f"r{prev}_w300", f"r{prev}_m1", f"r{prev}_m2"]
            won = C2.winner(f"round{prev}", "best", ["best"] + variants_prev)
            model_prev = f"checkpoints/value_r{prev}.pt"
            if won and os.path.exists(model_prev):
                key = won.split("_", 1)[1]
                gen_cfg = {**best, "shop_eval": "onestep", "value_model": model_prev, "value_mode": "antes",
                           "budget_round": 0, "budget_boss": 0, "onestep_weight": 200.0 if key == "w200" else 300.0}
                if key.startswith("m"):
                    gen_cfg["onestep_margin"] = float(key[1:])
                C.note(f"\n## round {k}: generating with the V agent ({won})")
            else:
                gen_cfg = best
                C.note(f"\n## round {k}: generating with the best hand-built config")
            data = f"checkpoints/value_data{k}"
            games_file = os.path.join(data, "games.jsonl")
            have = sum(1 for _ in open(games_file)) if os.path.exists(games_file) else 0
            if have < 25000:
                rc = C.run(["-m", "balatro_rl.value.gen", "--games", "25000", "--seed0", str(seed0), "--workers", "8",
                            "--chunk", "50", "--out", data, "--cfg", json.dumps(gen_cfg)], f"checkpoints/value_gen{k}.log")
                C.log(f"generation {k} exit {rc}")
            seed0 += 100_000
            n = 0
            for src in sorted(glob.glob(os.path.join(data, "chunk*.npz"))):
                dst = os.path.join("checkpoints/value_data", f"chunk{k}" + os.path.basename(src)[5:])
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
                    n += 1
            seen = {json.loads(l)["seed"] for l in open("checkpoints/value_data/games.jsonl")}
            with open("checkpoints/value_data/games.jsonl", "a") as f_out:
                for l in open(games_file):
                    if json.loads(l)["seed"] not in seen:
                        f_out.write(l)
            C.log(f"merged {n} chunks of data{k}")
            model = f"checkpoints/value_r{k}.pt"
            if not os.path.exists(model):
                rc = C.run(["-m", "balatro_rl.value.train", "--data", "checkpoints/value_data", "--out", model,
                            "--epochs", "30"], f"checkpoints/value_train_r{k}.log")
                C.log(f"train r{k} exit {rc}")
            C.note(f"## round {k}: value_r{k}.pt training report")
            for line in open(f"checkpoints/value_train_r{k}.log"):
                if "beats_phi" in line or line.startswith("  ante"):
                    C.note(line.rstrip())
            one = {**best, "shop_eval": "onestep", "value_model": model, "value_mode": "antes",
                   "budget_round": 0, "budget_boss": 0}
            variants = [f"r{k}_w200", f"r{k}_w300", f"r{k}_m1", f"r{k}_m2"]
            grid = {"best": {"cfg": best, "search": False},
                    variants[0]: {"cfg": {**one, "onestep_weight": 200.0}},
                    variants[1]: {"cfg": {**one, "onestep_weight": 300.0}},
                    variants[2]: {"cfg": {**one, "onestep_margin": 1.0}},
                    variants[3]: {"cfg": {**one, "onestep_margin": 2.0}}}
            if not changed:                         # the baseline's results carry over from round 3
                for blk, _, _ in C.BLOCKS:
                    src = os.path.join(EV, f"round3_{blk}", "best.json")
                    dst_dir = os.path.join(EV, f"round{k}_{blk}")
                    os.makedirs(dst_dir, exist_ok=True)
                    if os.path.exists(src) and not os.path.exists(os.path.join(dst_dir, "best.json")):
                        shutil.copy(src, os.path.join(dst_dir, "best.json"))
            res = C.grid_run(f"round{k}", grid, base_name="best")
            C.table(f"round {k}: value_r{k}.pt vs the best hand-built config", res, base="best")
            if changed:                             # later rounds reuse this round's baseline
                changed = False
                for blk, _, _ in C.BLOCKS:
                    src = os.path.join(EV, f"round{k}_{blk}", "best.json")
                    dst = os.path.join(EV, "round3_%s" % blk, "best.json")
                    if os.path.exists(src):
                        shutil.copy(src, dst)
            prev = k
        except Exception:
            C.log(f"chain3 round {k} failed: " + traceback.format_exc())
            break
    C.log("chain3=done")
    C.note("\n(chain3 finished)")


if __name__ == "__main__":
    main()
