"""Does the beam's win come from the build or from knowing the draws? Take a beam line (found with the real seed's future), keep only its decisions
outside rounds (blind select, skips, shop, packs: "the build"), and let an ordinary network play the rounds without any foresight.

    python tools/build_transfer.py --lines "traces/novel128_scale/GOLD_*.json" --model checkpoints/ppo_gold_v22.pt --samples 4

For each line: the blinds the beam's own line beat (24 for a win), then the blinds beaten when the network plays the rounds (greedy, and sampled
several times), with the line's non-round actions as a script (an action that is illegal at that moment is dropped).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppEnv, OBS_KEYS      # noqa: E402
from balatro_rl.model import load_model                   # noqa: E402

A_SELECT, A_LEAVE, A_PSKIP = 436, 448, 468


def script_of(d):
    """The line's actions taken outside rounds, in order; and the line's own result."""
    e = CppEnv(d["deck"], d["stake"])
    e.e.exact = True
    e.reset(int(d["seed"]))
    script, done = [], False
    for a in d["actions"]:
        if e.game.state != "SELECTING_HAND":
            script.append(int(a))
        _, _, done, _ = e.step(int(a))
        if done:
            break
    return script, e.game.dump()


def blinds_of(dump):
    return int(dump.get("blinds_beaten", dump.get("furthest_blind", 0)))


def play(d, script, model, greedy, gen, max_steps=4000):
    e = CppEnv(d["deck"], d["stake"])
    e.e.exact = True
    if getattr(model, "marg", 0):
        e.e.marginal = int(model.marg)
    obs = e.reset(int(d["seed"]))
    ptr, done = 0, False
    for _ in range(max_steps):
        st = e.game.state
        mask = obs["mask"]
        if st == "SELECTING_HAND":
            batch = {k: torch.from_numpy(np.asarray(obs[k])[None]) for k in OBS_KEYS}
            with torch.no_grad():
                logits, _ = model(batch)
            logits = logits[0]
            if greedy:
                a = int(logits.argmax())
            else:
                a = int(torch.multinomial(torch.softmax(logits.float(), 0), 1, generator=gen))
        else:
            a = None
            while ptr < len(script):
                c = script[ptr]
                ptr += 1
                if mask[c]:
                    a = c
                    break
            if a is None:
                a = A_SELECT if st == "BLIND_SELECT" else A_LEAVE if st == "SHOP" else A_PSKIP
        obs, _, done, _ = e.step(a)
        if done:
            break
    dump = e.game.dump()
    return dump["state"] == "WON", blinds_of(dump), dump["ante"], ptr, len(script)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lines", required=True)
    p.add_argument("--model", default="checkpoints/ppo_gold_v22.pt")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args()
    model = load_model(a.model, "cpu")
    gen = torch.Generator().manual_seed(0)
    files = sorted(glob.glob(a.lines))
    if a.limit:
        files = files[:a.limit]
    rows = []
    for f in files:
        d = json.load(open(f))
        if not d.get("actions"):
            continue
        script, dump = script_of(d)
        own_won = dump["state"] == "WON"
        own = 24 if own_won else blinds_of(dump)
        gw, gb, ga, used, n = play(d, script, model, True, gen)
        sw, sb = [], []
        for _ in range(a.samples):
            w, b, _, _, _ = play(d, script, model, False, gen)
            sw.append(w)
            sb.append(b)
        rows.append(dict(file=os.path.basename(f), seed=d["seed"], own_win=own_won, own_blinds=own, greedy_win=gw, greedy_blinds=gb,
                         sampled_wins=int(sum(sw)), sampled_n=a.samples, sampled_blinds=float(np.mean(sb)) if sb else 0.0, used=used, script=n))
        r = rows[-1]
        print(f"{r['file']:22s} beam {'WIN' if own_won else 'lost'} ({own:2d} blinds) | with {os.path.basename(a.model)} in rounds: greedy "
              f"{'WIN' if gw else 'lost'} ({gb:2d}), sampled wins {r['sampled_wins']}/{a.samples} avg {r['sampled_blinds']:.1f} blinds"
              f"  [script {used}/{n}]", flush=True)
    if rows:
        wins = [r for r in rows if r["own_win"]]
        print(f"\n{len(rows)} lines ({len(wins)} beam wins).")
        if wins:
            print(f"  beam-win lines: greedy wins {sum(r['greedy_win'] for r in wins)}/{len(wins)}, sampled win rate "
                  f"{sum(r['sampled_wins'] for r in wins) / (len(wins) * a.samples):.2f}, mean blinds "
                  f"{np.mean([r['greedy_blinds'] for r in wins]):.1f} (greedy) vs 24 for the beam")
        lost = [r for r in rows if not r["own_win"]]
        if lost:
            print(f"  beam-loss lines: mean blinds {np.mean([r['greedy_blinds'] for r in lost]):.1f} (greedy) vs the beam's {np.mean([r['own_blinds'] for r in lost]):.1f}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
