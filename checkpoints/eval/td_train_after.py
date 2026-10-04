"""Train the TD(lambda) value model once chain2's round-7 training has released the RAM."""
import os, subprocess, sys, time
ROOT = r"C:\balatro_rl"
LOG = os.path.join(ROOT, "checkpoints", "eval", "overnight_chain.log")
os.chdir(ROOT)
while True:
    try:
        if "train r7 exit" in open(LOG).read() or "chain2=done" in open(LOG).read():
            break
    except OSError:
        pass
    time.sleep(60)
with open(LOG, "a") as f:
    f.write(time.strftime("%H:%M:%S") + " td training started\n")
with open("checkpoints/value_train_td.log", "a") as out:
    rc = subprocess.run([sys.executable, "-u", "-m", "balatro_rl.value.train", "--data", "checkpoints/value_data",
                         "--out", "checkpoints/value_td.pt", "--epochs", "24", "--td-lambda", "0.8"],
                        cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
with open(LOG, "a") as f:
    f.write(time.strftime("%H:%M:%S") + f" td training exit {rc}\n")
