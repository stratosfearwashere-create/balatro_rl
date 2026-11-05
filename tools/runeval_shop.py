"""Shop decision agreement (the spec's fourth gate): does the eval pick the option the simulator prefers?

Shops are taken from games the labelling policy plays (scripted rounds). At each shop with at least two options, every option (each buy, the
voucher, a pack, a reroll when affordable, leave) is applied on a copy; the state it leads to is scored by the eval, and played to the end
--rollouts times by the same policy (new draws each time). The simulator's verdict on an option is the mean blinds beaten (and the win rate);
agreement is the share of shops where the eval's best option is the simulator's best, or within --tol blinds of it. Also reported: the regret
(blinds the eval's pick loses against the best), the same for a greedy money-keeping baseline (always leave) and for the policy's own choice.

    python tools/runeval_shop.py --shops 256 --rollouts 32
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from balatro_rl.engine_env import CppVecEnv                                        # noqa: E402
from balatro_rl.env import A_BUY, A_BUY_PACK, A_VOUCHER, A_REROLL, A_LEAVE, A_SELL_J, A_SELL_C   # noqa: E402
from balatro_rl.runeval import RunEval, Cols, I_SHOP, from_vecenv                   # noqa: E402
from runeval_data import Policy, play_out                                           # noqa: E402


def options_of(mask_row, max_opts):
    """The decisive shop actions: buys, vouchers, packs, reroll, leave (sells left out: they are a different kind of choice)."""
    rows = [a for a in range(A_BUY, A_REROLL) if mask_row[a]]
    if mask_row[A_REROLL]:
        rows.append(A_REROLL)
    if mask_row[A_LEAVE]:
        rows.append(A_LEAVE)
    return rows[:max_opts]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v30_best.pt")
    p.add_argument("--shops", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--max-opts", type=int, default=6)
    p.add_argument("--batch", type=int, default=32, help="shops per batch (batch x options x rollouts environments)")
    p.add_argument("--tol", type=float, default=0.5, help="blinds within which an option counts as agreeing with the best")
    p.add_argument("--seed0", type=int, default=300_000)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--snap-p", type=float, default=0.25)
    p.add_argument("--per-ante", type=int, default=0, help="at most this many shops from any one ante (0: shops / 5), so the late antes are represented")
    p.add_argument("--dev", default="cuda")
    p.add_argument("--out", default="traces/runeval_shop.json")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed0)
    ev = RunEval()
    policy = Policy(a.ckpt, a.dev)
    roots = CppVecEnv(a.games, "RED", "GOLD", seed0=a.seed0, threads=0)
    roots.set_auto_round(True, 1.5, 0)
    roots.v.restart()
    bank = CppVecEnv(a.batch, "RED", "GOLD", seed0=1, threads=0)
    bank.set_auto_round(True, 1.5, 0)
    opts = CppVecEnv(a.batch * a.max_opts, "RED", "GOLD", seed0=2, threads=0)
    opts.set_auto_round(True, 1.5, 0)
    work = CppVecEnv(a.batch * a.max_opts * a.rollouts, "RED", "GOLD", seed0=3, threads=0)
    work.set_auto_round(True, 1.5, 0)
    done = np.zeros(a.games, dtype=bool)
    seen = set()
    results = []
    fill, pend = 0, []
    t0 = time.time()
    cap = a.per_ante if a.per_ante > 0 else max(1, a.shops // 5)
    taken = {}

    def flush():
        nonlocal fill
        if fill == 0:
            return
        m = fill
        mask = np.asarray(bank.obs["mask"])
        # apply every option on a copy
        slots, meta = [], []
        for i in range(m):
            for k, act in enumerate(pend[i]["opts"]):
                slots.append((i, k, act))
        opts.copy_many(bank, [i for i, _, _ in slots], list(range(len(slots))))
        acts = np.full(opts.n, -1, dtype=np.int64)
        for j, (_, _, act) in enumerate(slots):
            acts[j] = act
        opts.step(acts)
        inp = from_vecenv(opts, list(range(len(slots))))
        sc = ev(**inp)
        # rollouts from each option's state
        K = a.rollouts
        N = len(slots) * K
        work.copy_many(opts, [j // K for j in range(N)], list(range(N)))
        work.reseed_rng(list(range(N)), [int(a.seed0 * 7 + j * 7919 + len(results) * 13) for j in range(N)])
        out = play_out(work, policy, np.arange(N), watch_ante4=False)
        bl = out["blinds"].reshape(len(slots), K).astype(np.float64)
        wn = out["won"].reshape(len(slots), K).astype(np.float64)
        for i in range(m):
            js = [j for j, (ii, _, _) in enumerate(slots) if ii == i]
            if len(js) < 2:
                continue
            blm = bl[js].mean(1)
            se = bl[js].std(1).mean() / np.sqrt(K)
            wm = wn[js].mean(1)
            e_score = sc["win"][js]
            e_blend = sc["score"][js]
            pick = int(np.argmax(e_score))
            pick_b = int(np.argmax(e_blend))
            best = int(np.argmax(blm))
            acts = pend[i]["opts"]
            leave = acts.index(A_LEAVE) if A_LEAVE in acts else None
            pol = pend[i]["policy_choice"]
            pol_k = acts.index(pol) if pol in acts else None
            results.append(dict(ante=pend[i]["ante"], money=pend[i]["money"], n_opts=len(acts), blinds=blm.tolist(), wins=wm.tolist(), se=float(se),
                                eval_pick=pick, eval_pick_blend=pick_b, sim_best=best, gap=float(blm[best] - np.sort(blm)[-2]) if len(js) > 1 else 0.0,
                                regret_eval=float(blm[best] - blm[pick]), regret_eval_blend=float(blm[best] - blm[pick_b]),
                                regret_leave=float(blm[best] - blm[leave]) if leave is not None else None,
                                regret_policy=float(blm[best] - blm[pol_k]) if pol_k is not None else None,
                                win_regret_eval=float(wm.max() - wm[pick]), win_regret_leave=float(wm.max() - wm[leave]) if leave is not None else None,
                                win_regret_policy=float(wm.max() - wm[pol_k]) if pol_k is not None else None))
        print(f"  {time.strftime('%H:%M')} shops judged {len(results)} ({time.time() - t0:.0f}s)", flush=True)
        fill = 0
        pend.clear()

    for _ in range(6000):
        live = np.flatnonzero(~done)
        if not len(live) or len(results) + fill >= a.shops:
            break
        g = np.asarray(roots.obs["g"])
        mask = np.asarray(roots.obs["mask"])
        act = policy.act(roots, live)
        for i in live:
            if g[i, Cols.STATE + I_SHOP] <= 0.5:
                continue
            key = (int(i), int(round(g[i, Cols.ANTE] * 8)), int(np.argmax(g[i, Cols.BLIND:Cols.BLIND + 3])))
            if key in seen or rng.random() >= a.snap_p or taken.get(key[1], 0) >= cap:
                continue
            rows = options_of(mask[i], a.max_opts)
            if len(rows) < 2:
                continue
            seen.add(key)
            taken[key[1]] = taken.get(key[1], 0) + 1
            bank.copy_from(roots, int(i), fill)
            money = g[i, Cols.MONEY] * 50.0 if g[i, Cols.MONEY] < 1.99 else 10.0 ** (3.0 * g[i, Cols.LOG_MONEY]) - 1.0
            pend.append(dict(opts=rows, ante=key[1], money=float(money), policy_choice=int(act[i])))
            fill += 1
            if fill == a.batch:
                flush()
                break
        _, d, _ = roots.step(act)
        done |= d
    flush()
    results = results[:a.shops]
    R = results
    agree = np.mean([r["regret_eval"] <= a.tol for r in R])
    agree_b = np.mean([r["regret_eval_blend"] <= a.tol for r in R])
    clear = [r for r in R if r["gap"] > 2 * r["se"]]
    agree_clear = np.mean([r["regret_eval"] <= a.tol for r in clear]) if clear else float("nan")
    summary = dict(shops=len(R), rollouts=a.rollouts, tol=a.tol, agreement=float(agree), agreement_blend=float(agree_b), clear_cut_shops=len(clear),
                   agreement_clear_cut=float(agree_clear), mean_regret_eval=float(np.mean([r["regret_eval"] for r in R])),
                   mean_regret_leave=float(np.mean([r["regret_leave"] for r in R if r["regret_leave"] is not None])),
                   mean_regret_policy=float(np.mean([r["regret_policy"] for r in R if r["regret_policy"] is not None])),
                   agreement_leave=float(np.mean([r["regret_leave"] <= a.tol for r in R if r["regret_leave"] is not None])),
                   agreement_policy=float(np.mean([r["regret_policy"] <= a.tol for r in R if r["regret_policy"] is not None])),
                   win_regret_eval=float(np.mean([r["win_regret_eval"] for r in R])),
                   win_regret_leave=float(np.mean([r["win_regret_leave"] for r in R if r["win_regret_leave"] is not None])),
                   win_regret_policy=float(np.mean([r["win_regret_policy"] for r in R if r["win_regret_policy"] is not None])),
                   mean_se=float(np.mean([r["se"] for r in R])), mean_gap=float(np.mean([r["gap"] for r in R])))
    print(summary)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    import json
    json.dump(dict(summary=summary, shops=R), open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
