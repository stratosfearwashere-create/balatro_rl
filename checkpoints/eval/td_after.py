"""After chain3: A/B the TD(lambda) model (value_td.pt) against the best hand-built config on both blocks."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight_chain as C  # noqa: E402

os.chdir(C.ROOT)
while True:
    try:
        if "chain3=done" in open(C.LOG).read():
            break
    except OSError:
        pass
    time.sleep(60)
C.log("td A/B started")
best = {"shop_prior": "graded", "shop": {"rule_arcana": True}}
for line in open(C.SUMMARY):
    if line.startswith("best hand-built config (chain3): "):
        best = json.loads(line.split(": ", 1)[1])
one = {**best, "shop_eval": "onestep", "value_model": "checkpoints/value_td.pt", "value_mode": "antes",
       "budget_round": 0, "budget_boss": 0}
grid = {"best": {"cfg": best, "search": False},
        "td_w200": {"cfg": {**one, "onestep_weight": 200.0}},
        "td_w300": {"cfg": {**one, "onestep_weight": 300.0}}}
res = C.grid_run("td", grid, base_name="best")
C.table("TD(lambda=0.8) model vs the best hand-built config", res, base="best")
C.log("td=done")
