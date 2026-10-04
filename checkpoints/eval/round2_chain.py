"""Round 2, unattended (each step skips if its output exists, so this can be relaunched):
  1. wait for the 20k-game generation (checkpoints/value_gen2.log says "done in")
  2. merge value_data2 into value_data (chunks renamed 1xxxx) and run the usage report on the new games
  3. train value_r2.pt on all 30k games
  4. one-step A/B with value_r2.pt: weights 200 / 300 and the margin guard (1 and 2 logits), two 300-seed blocks
Progress: checkpoints/eval/round2_chain.log
"""
import glob, json, os, shutil, subprocess, sys, time

ROOT = r"C:\balatro_rl"
EV = os.path.join(ROOT, "checkpoints", "eval")
LOG = os.path.join(EV, "round2_chain.log")
PY = sys.executable


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def run(args, out):
    with open(out, "a") as f:
        return subprocess.run([PY, "-u"] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    os.chdir(ROOT)
    log("chain started")
    while True:
        try:
            if "done in" in open("checkpoints/value_gen2.log").read():
                break
        except OSError:
            pass
        time.sleep(60)
    log("generation finished")
    # 2. merge + stats
    n = 0
    for src in sorted(glob.glob("checkpoints/value_data2/chunk*.npz")):
        dst = os.path.join("checkpoints/value_data", "chunk1" + os.path.basename(src)[5:])
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            n += 1
    with open("checkpoints/value_data/games.jsonl", "a") as f_out, open("checkpoints/value_data2/games.jsonl") as f_in:
        seen = set()
        try:
            seen = {json.loads(l)["seed"] for l in open("checkpoints/value_data/games.jsonl")}
        except OSError:
            pass
        for l in f_in:
            if json.loads(l)["seed"] not in seen:
                f_out.write(l)
    log(f"merged {n} chunks")
    if not os.path.exists(os.path.join(EV, "usage_round2.txt")):
        rc = run(["-m", "balatro_rl.value.stats", "--data", "checkpoints/value_data2", "--stake", "WHITE", "--top", "30"],
                 os.path.join(EV, "usage_round2.txt"))
        run(["-m", "balatro_rl.value.stats", "--data", "checkpoints/value_data2", "--top", "30"],
            os.path.join(EV, "usage_round2.txt"))
        log(f"usage report exit {rc}")
    # 3. train
    if not os.path.exists("checkpoints/value_r2.pt"):
        rc = run(["-m", "balatro_rl.value.train", "--data", "checkpoints/value_data", "--out", "checkpoints/value_r2.pt",
                  "--epochs", "30"], "checkpoints/value_train_r2.log")
        log(f"train exit {rc}")
    # 4. A/B
    base = {"shop_prior": "graded", "shop": {"rule_arcana": True}}
    one = {**base, "shop_eval": "onestep", "value_model": "checkpoints/value_r2.pt", "value_mode": "antes",
           "budget_round": 0, "budget_boss": 0}
    grid = {"graded": {"cfg": base, "search": False},
            "r2_w200": {"cfg": {**one, "onestep_weight": 200.0}},
            "r2_w300": {"cfg": {**one, "onestep_weight": 300.0}},
            "r2_m1": {"cfg": {**one, "onestep_margin": 1.0}},
            "r2_m2": {"cfg": {**one, "onestep_margin": 2.0}}}
    gp = os.path.join(ROOT, "eval_grids", "round2_grid.json")
    json.dump(grid, open(gp, "w"), indent=1)
    for name, seed0, prev in (("round2_a", 10000, "value_ab"), ("round2_b", 10300, "value_repl")):
        out = os.path.join(EV, name)
        os.makedirs(out, exist_ok=True)
        src = os.path.join(EV, prev, "graded.json")
        if os.path.exists(src) and not os.path.exists(os.path.join(out, "graded.json")):
            shutil.copy(src, os.path.join(out, "graded.json"))
        rc = run(["-m", "balatro_rl.az.compare", "grid", "--grid", gp, "--out", out, "--games", "300", "--workers", "7",
                  "--seed0", str(seed0)], os.path.join(EV, name + ".log"))
        run(["-m", "balatro_rl.az.compare", "table", "--grid", gp, "--out", out, "--base", "graded"],
            os.path.join(EV, name + "_table.txt"))
        log(f"{name} exit {rc}")
    log("chain=done")


if __name__ == "__main__":
    main()
