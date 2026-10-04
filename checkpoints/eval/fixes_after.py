"""After chain2 finishes: the hand-built fixes grid that step 2 of overnight_chain.py skipped (import bug), on
both seed blocks, against the graded baseline; appended to overnight_summary.txt."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight_chain as C  # noqa: E402

os.chdir(C.ROOT)
while True:
    try:
        if "chain2=done" in open(C.LOG).read():
            break
    except OSError:
        pass
    time.sleep(60)
C.log("fixes grid started")
from balatro_rl.az.shop import VOUCHER_DOLLARS  # noqa: E402

base = {"shop_prior": "graded", "shop": {"rule_arcana": True}}
vd = dict(VOUCHER_DOLLARS)
vd.update({"hieroglyph": -8.0, "petroglyph": -8.0})
fixes = {"skip": {"shop": {"rule_arcana": True, "rule_skip": True}},
         "hiero": {"shop": {"rule_arcana": True, "voucher_dollars": vd}},
         "interest4": {"shop": {"rule_arcana": True, "interest_rounds": 4.0}},
         "dollar12": {"shop": {"rule_arcana": True, "dollar": 0.012}},
         "boss": {"shop": {"rule_arcana": True, "rule_boss": True}}}
grid = {"graded": {"cfg": base, "search": False}}
for k, over in fixes.items():
    grid[k] = {"cfg": {**base, **over}, "search": False}
res = C.grid_run("fixes", grid)
C.table("2 (rerun). hand-built fixes from the usage stats", res)
good = [k for k in fixes if C.gains_on_both(res, k, "graded")]
C.note(f"fixes that gain > 1 SE on both blocks: {good}")
C.log("fixes=done")
