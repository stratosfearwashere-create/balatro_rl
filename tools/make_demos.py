"""Demonstrations of strong builds for imitation: the non-round decisions of the beam search's lines.

The lines come from tools/solve_novel.py (exact-seed beam, Gold; traces/demo_lines/*.json: full action sequences, rounds included). Each line with a win, or deep
enough (--min-level blinds), is replayed in the exact engine with its seed, and at every decision OUTSIDE a round (blind select / skip, shop, packs: action index >=
A_SELECT) the observation the policy would see and the line's action are recorded. Round play is left out: the beam plays rounds with the clairvoyant solver, which no
policy can copy; the build is what is wanted.

Storage is compact: the per-action arrays (aid, aid2, af, mask) are kept only for the legal actions; tools/demo_data.py rebuilds the dense observation for a batch.

    python tools/make_demos.py --lines traces/demo_lines --min-level 15 --out checkpoints/demos_gold.npz
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solve_fast import make                                                      # noqa: E402
from balatro_rl.env import A_SELECT                                              # noqa: E402

DENSE = ("g", "cards", "jid", "jf", "cid", "subs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lines", default="traces/demo_lines")
    p.add_argument("--min-level", type=int, default=15, help="keep lines that won or reached at least this many blinds")
    p.add_argument("--out", default="checkpoints/demos_gold.npz")
    p.add_argument("--priv", action="store_true", help="also record the privileged observation (the future list: next shops, bosses ...) for a network that reads it (--priv-actor)")
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.lines, "*.json")))
    dense = DENSE + (("priv",) if a.priv else ())
    rec = {k: [] for k in dense}
    legal_idx, legal_af, legal_aid, legal_aid2, offs = [], [], [], [], [0]
    acts, seeds, levels, won = [], [], [], []
    used = 0
    mismatch = 0
    for f in files:
        d = json.load(open(f))
        path = d.get("actions") or []
        level = 24 if d.get("solved") else int(d.get("info", {}).get("level", 0))
        if not path or (not d.get("solved") and level < a.min_level):
            continue
        v = make(1, d.get("deck", "RED"), d.get("stake", "GOLD"), int(d["seed"]), 0)
        if a.priv:                                       # the future list is filled from the next encode on: restart the (same) game with it
            v.set_priv(True)
            v.v.restart()
        used += 1
        for ai in path:
            ai = int(ai)
            mask = np.asarray(v.obs["mask"])[0]
            if ai < 0 or ai >= len(mask) or not mask[ai]:
                mismatch += 1
                break
            if ai >= A_SELECT:
                for k in dense:
                    rec[k].append(np.asarray(v.obs[k])[0].copy())
                idx = np.flatnonzero(mask)
                legal_idx.append(idx.astype(np.int16))
                legal_af.append(np.asarray(v.obs["af"])[0][idx].astype(np.float16))
                legal_aid.append(np.asarray(v.obs["aid"])[0][idx].astype(np.int32))
                legal_aid2.append(np.asarray(v.obs["aid2"])[0][idx].astype(np.int32))
                offs.append(offs[-1] + len(idx))
                acts.append(ai)
                seeds.append(int(d["seed"]))
                levels.append(level)
                won.append(bool(d.get("solved")))
            act = np.array([ai], dtype=np.int64)
            _, done, _ = v.step(act)
            if done[0]:
                break
    out = {k: np.stack(rec[k]) for k in dense}
    if a.priv:
        out["priv"] = out["priv"].astype(np.float16)
    out["cards"] = out["cards"].astype(np.float16)
    out["subs"] = out["subs"].astype(np.int8)
    out.update(legal_idx=np.concatenate(legal_idx), legal_af=np.concatenate(legal_af), legal_aid=np.concatenate(legal_aid),
               legal_aid2=np.concatenate(legal_aid2), offs=np.array(offs, dtype=np.int64), act=np.array(acts, dtype=np.int64),
               seed=np.array(seeds), level=np.array(levels), won=np.array(won))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, a.out)                                           # (atomic: a trainer reloads the file while this runs)
    print(f"{used} lines ({int(sum(1 for f in files if json.load(open(f)).get('solved')))} won among all files), {len(acts)} decisions, {mismatch} lines cut by an illegal action; "
          f"-> {a.out} ({os.path.getsize(a.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
