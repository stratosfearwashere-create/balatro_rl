"""A game played one decision at a time, kept in traces/session_<seed>.json (the actions so far; exact mode replays them to the same state).

    python tools/play_session.py 43003 show            # the state, the shop, the legal actions with their numbers
    python tools/play_session.py 43003 do 112 114      # take these actions
    python tools/play_session.py 43003 round           # select the blind and play it with the exact round solver (the most money left)
    python tools/play_session.py 43003 undo 2          # take back the last 2 actions (a played round counts as its plays)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from balatro_rl import env as E                                          # noqa: E402
from balatro_rl.engine_env import CppVecEnv                               # noqa: E402
from balatro_rl.heur import C_COST, describe                              # noqa: E402
from balatro_rl.runeval import Cols, State                                # noqa: E402
from balatro_rl.sim.hands import HAND_NAMES                               # noqa: E402
from balatro_rl.sim.items import VOCAB, BOSS_KEYS, ALL_BOSSES, TAGS        # noqa: E402
from balatro_rl.sim.cards import ENHANCEMENTS, EDITIONS                   # noqa: E402


def path(seed):
    return os.path.join(ROOT, "traces", f"session_{seed}.json")


def load(seed):
    acts = json.load(open(path(seed)))["actions"] if os.path.exists(path(seed)) else []
    v = CppVecEnv(1, "RED", "GOLD", seed0=seed, threads=1)
    v.set_exact(True)
    v.set_auto_target(True)
    v.v.restart()
    done = False
    for a in acts:
        _, d, info = v.step(np.array([a]))
        done = bool(d[0])
        if done:
            break
    return v, acts, done


def save(seed, acts):
    json.dump({"seed": seed, "actions": acts}, open(path(seed), "w"))


def name(i):
    n = VOCAB[int(i)] if 0 <= int(i) < len(VOCAB) else "?"
    return n.split("_", 1)[1] if "_" in n else n


def show(v):
    g = np.asarray(v.obs["g"])[0]
    s = State(np.asarray(v.obs["g"]), np.asarray(v.obs["jid"]), np.asarray(v.obs["jf"]), np.asarray(v.obs["cid"]))
    st = ["blind select", "in a round", "shop", "pack"][int(s.state[0])]
    blind = ["small", "big", "boss"][int(s.blind[0])]
    boss = BOSS_KEYS[int(s.boss_idx[0])] if s.boss_known[0] else "?"
    print(f"ANTE {int(s.ante[0])}  {blind} blind  ({st})   money ${s.money[0]:.0f}   hands {int(s.hands[0])}  discards {int(s.discards[0])}  hand size {int(s.hand_size[0])}")
    print(f"  targets: small {s.targets[0, 0]:,.0f}  big {s.targets[0, 1]:,.0f}  boss {s.targets[0, 2]:,.0f}  (boss: {boss})")
    tag = int(np.argmax(s.tag[0])) if s.tag[0].max() > 0.5 else -1
    if tag >= 0:
        print(f"  skipping this blind gives the tag: {TAGS[tag]}")
    jid, jf = np.asarray(v.obs["jid"])[0], np.asarray(v.obs["jf"])[0]
    js = []
    for k in range(len(jid)):
        if jid[k] <= 0:
            continue
        ed = [EDITIONS[e] for e in range(1, 5) if jf[k, 1 + e] > 0.5]        # (jf 1..5: the edition one-hot, "" first)
        stick = [n for n, c in (("eternal", 6), ("perishable", 7), ("rental", 8), ("debuffed", 9)) if jf[k, c] > 0.05]
        cnt = s.counter[0, k]
        js.append(f"{name(jid[k])}" + (f"[{cnt:g}]" if abs(cnt) > 1e-6 else "") + (f" {'/'.join(ed).lower()}" if ed else "") + (f" ({','.join(stick)})" if stick else "")
                  + f" sell ${jf[k, 10] * 10:.0f}")
    print(f"  jokers ({int(s.njokers[0])}/{int(s.jslots[0])}): " + (", ".join(js) if js else "none"))
    cid = np.asarray(v.obs["cid"])[0]
    cs = [name(c) for c in cid if c > 0]
    print(f"  consumables: {', '.join(cs) if cs else 'none'}")
    lv = [f"{HAND_NAMES[h]} {int(s.levels[0, h])}" for h in range(12) if s.levels[0, h] > 1]
    print(f"  hand levels: {', '.join(lv) if lv else 'all 1'}")
    enh = [f"{ENHANCEMENTS[e + 1].lower()} {int(c)}" for e, c in enumerate(s.enh[0]) if c > 0]
    print(f"  deck {int(s.full[0])} cards, suits S{int(s.suits[0, 0])} H{int(s.suits[0, 1])} C{int(s.suits[0, 2])} D{int(s.suits[0, 3])}" + (f", {', '.join(enh)}" if enh else ""))
    mask = np.asarray(v.obs["mask"])[0]
    aid = np.asarray(v.obs["aid"])[0]
    af = np.asarray(v.obs["af"])[0]
    legal = np.flatnonzero(mask)
    print("  actions:")
    for a in legal:
        if a < E.A_SELECT:
            continue
        cost = af[a, C_COST] * 10 if a < E.A_SELL_J or a >= E.A_REPLACE else 0
        extra = f"  ${cost:.0f}" if cost > 0 else ""
        if a >= E.A_REPLACE:
            extra = f"  (gives up {name(np.asarray(v.obs['aid2'])[0][a])})" + extra
        if (E.A_BUY <= a < E.A_BUY_PACK) or (E.A_PICK <= a < E.A_PSKIP) or a >= E.A_REPLACE:      # the item's edition and stickers
            from balatro_rl.heur import C_ED, C_ETERNAL, C_PERISH, C_RENTAL
            ed = EDITIONS[int(np.argmax(af[a, C_ED:C_ED + 5]))]
            tags = ([ed.lower()] if ed else []) + [t for t, c in (("ETERNAL", C_ETERNAL), ("PERISHABLE", C_PERISH), ("RENTAL $3/round", C_RENTAL)) if af[a, c] > 0.05]
            if tags:
                extra += "  [" + ", ".join(tags) + "]"
        print(f"    {a:4d}  {describe(int(a), aid)}{extra}")
    if s.state[0] == 1:
        print("    (in a round: use `round` to have the solver finish it)")


def main():
    seed = int(sys.argv[1])
    cmd = sys.argv[2] if len(sys.argv) > 2 else "show"
    v, acts, done = load(seed)
    if cmd == "do":
        for x in sys.argv[3:]:
            a = int(x)
            if not np.asarray(v.obs["mask"])[0][a]:
                print(f"action {a} is not legal here; stopped")
                break
            _, d, info = v.step(np.array([a]))
            acts.append(a)
            if d[0]:
                print("GAME OVER" if not info["won"][0] else "WON")
                done = True
                break
    elif cmd == "round":
        g = np.asarray(v.obs["g"])[0]
        if g[Cols.STATE] > 0.5:
            v.step(np.array([E.A_SELECT]))
            acts.append(E.A_SELECT)
        v.set_solver_hand_bound(True)                      # (the richest clearing line: fewest hands, discards used freely)
        plans = []
        for cap in (2000, 50000, 300000):                  # a bigger search when a smaller one finds no clearing line
            plans = v.solve_rounds([0], 1, cap)[0]
            if plans:
                break
        if not plans:
            print("the solver finds no line that clears this blind: the run would end here")
            save(seed, acts)
            return
        for a in plans[0]:
            _, d, info = v.step(np.array([a]))
            acts.append(int(a))
            if d[0]:
                print("WON" if info["won"][0] else "GAME OVER")
                done = True
                break
        plays = sum(1 for a in plans[0] if a < E.A_DISC)
        print(f"round played: {plays} hands, {len(plans[0]) - plays} discards")
    elif cmd == "undo":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        acts = acts[:-n]
        save(seed, acts)
        v, acts, done = load(seed)
    save(seed, acts)
    if not done:
        show(v)


if __name__ == "__main__":
    main()
