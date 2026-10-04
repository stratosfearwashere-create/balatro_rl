"""Keeps the loop going after overnight_chain.py until a deadline: more rounds of play -> train -> A/B.

Each round: 25k games with the current best agent (the V agent if it beat the hand-built config by more than
one paired SE on both seed blocks, else the best hand-built config), merged into checkpoints/value_data;
value_r{k}.pt trained on everything; A/B (weights 200 / 300, margins 1 / 2) against the best hand-built
config on both blocks. No new round starts after DEADLINE. Summary: overnight_summary.txt (appended).
"""
import glob
import json
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight_chain as C  # noqa: E402

DEADLINE = time.mktime(time.strptime(time.strftime("%Y-%m-%d") + " 13:30", "%Y-%m-%d %H:%M"))
ROOT = C.ROOT
EV = C.EV


def best_config() -> dict:
    cfg = {"shop_prior": "graded", "shop": {"rule_arcana": True}}
    try:
        for line in open(C.SUMMARY):
            if line.startswith("best hand-built config: "):
                cfg = json.loads(line.split(": ", 1)[1])
    except OSError:
        pass
    return cfg


def winner(name: str, base: str, grid_names) -> str | None:
    """The variant of a finished two-block grid that beats `base` by > 1 SE on both blocks (best mean), or None."""
    res = {}
    for blk, _, _ in C.BLOCKS:
        out = os.path.join(EV, f"{name}_{blk}")
        res[blk] = {}
        for v in grid_names:
            p = os.path.join(out, v + ".json")
            if os.path.exists(p):
                res[blk][v] = json.load(open(p))
    cands = []
    for v in grid_names:
        if v == base or not C.gains_on_both(res, v, base):
            continue
        m = sum(C.paired(res[b][v], res[b][base])[0] for b in res) / len(res)
        cands.append((m, v))
    return max(cands)[1] if cands else None


def main():
    os.chdir(ROOT)
    # wait for the first chain
    while True:
        try:
            if "chain=done" in open(C.LOG).read():
                break
        except OSError:
            pass
        time.sleep(60)
    C.log("chain2 started")
    best = best_config()
    k = 3
    seed0 = 1_300_000
    variants = ["r3_w200", "r3_w300", "r3_m1", "r3_m2"]
    prev_grid, prev_base = "round3", "best"
    while time.time() < DEADLINE:
        k += 1
        try:
            won = winner(prev_grid, prev_base, [prev_base] + variants)
            if won:
                w = {"r{}_w200": 200.0, "r{}_w300": 300.0, "r{}_m1": 300.0, "r{}_m2": 300.0}
                key = won.split("_", 1)[1]
                gen_cfg = {**best, "shop_eval": "onestep", "value_model": f"checkpoints/value_r{k - 1}.pt",
                           "value_mode": "antes", "budget_round": 0, "budget_boss": 0,
                           "onestep_weight": 200.0 if key == "w200" else 300.0}
                if key.startswith("m"):
                    gen_cfg["onestep_margin"] = float(key[1:])
                C.note(f"\n## round {k}: generating with the V agent ({won})")
            else:
                gen_cfg = best
                C.note(f"\n## round {k}: generating with the best hand-built config (no V variant won both blocks)")
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
            if time.time() > DEADLINE + 3600:
                break
            one = {**best, "shop_eval": "onestep", "value_model": model, "value_mode": "antes",
                   "budget_round": 0, "budget_boss": 0}
            variants = [f"r{k}_w200", f"r{k}_w300", f"r{k}_m1", f"r{k}_m2"]
            grid = {"best": {"cfg": best, "search": False},
                    variants[0]: {"cfg": {**one, "onestep_weight": 200.0}},
                    variants[1]: {"cfg": {**one, "onestep_weight": 300.0}},
                    variants[2]: {"cfg": {**one, "onestep_margin": 1.0}},
                    variants[3]: {"cfg": {**one, "onestep_margin": 2.0}}}
            # reuse the best hand-built config's results from round 3
            for blk, _, _ in C.BLOCKS:
                src = os.path.join(EV, f"round3_{blk}", "best.json")
                dst_dir = os.path.join(EV, f"round{k}_{blk}")
                os.makedirs(dst_dir, exist_ok=True)
                if os.path.exists(src) and not os.path.exists(os.path.join(dst_dir, "best.json")):
                    shutil.copy(src, os.path.join(dst_dir, "best.json"))
            res = C.grid_run(f"round{k}", grid, base_name="best")
            C.table(f"round {k}: value_r{k}.pt vs the best hand-built config", res, base="best")
            prev_grid, prev_base = f"round{k}", "best"
        except Exception:
            C.log(f"round {k} failed: " + traceback.format_exc())
            break
    C.log("chain2=done")
    C.note("\n(chain2 finished)")


if __name__ == "__main__":
    main()
