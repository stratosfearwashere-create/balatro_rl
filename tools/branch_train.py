"""Policy improvement by branching rollouts (expert iteration on the build decisions).

    python tools/branch_train.py --init checkpoints/ppo_gold_imm2.pt --out checkpoints/branch1.pt

The game is Gold on the real game's named random streams (exact mode), immortal with pay-on-fail (every game runs to the ante-8 boss; the reward is the blinds beaten and the win).
Each iteration:
  1. ROOTS. The current policy plays `--games` games; at a fraction of its decisions outside rounds (shop, packs, blind select) the game is copied into a root slot.
  2. BRANCHES. At a root the alternatives are the policy's top few legal actions, leaving the shop, and a few random legal ones. Each alternative is played out to the end of the
     game on a copy of the root, with the policy (sampled, with a small share of random shop actions so that it does not only repeat its habits), under `--futures` different
     futures (the named streams re-seeded; the same futures for every alternative, so the alternatives are compared under the same luck).
  3. LABELS. Q(alternative) = mean return over the futures; the target distribution over the alternatives is softmax((Q - mean Q) / tau).
  4. TRAINING. A KL-regularised improvement, as in advantage-weighted regression: the target over the alternatives is softmax(kappa * logit_old + (Q - mean Q) / tau) (the policy's own
     preference, tempered, times the advantage, so an alternative the policy thought absurd is not lifted to the top by one lucky rollout); the loss is cross entropy among the
     alternatives, plus KL(old || new) over all legal actions at the roots (a trust region), plus a KL anchor to the starting policy on states the policy itself met (rounds
     included). Its evaluation is greedy AND sampled, and a drop in either puts the starting weights back.
It logs `headroom`: how much better the best alternative is than the policy's own greedy choice (0: the policy is already the best of what was tried), and how often they differ.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from balatro_rl.engine_env import CppVecEnv, OBS_KEYS     # noqa: E402
from balatro_rl.model import load_model                    # noqa: E402
from balatro_rl import env as E                            # noqa: E402

rng_anchor = np.random.default_rng(1)
ROUND_COL = 5                                              # obs "g" column 5: in a round (selecting hand)


def make_vec(n, a, seed0):
    v = CppVecEnv(n, a.deck, a.stake, seed0=seed0, threads=0, reward_kw={"win_bonus": a.win_bonus})
    v.set_weight_geom(a.weight_geom)
    v.set_exact(True)
    v.set_immortal(True, True)
    v.v.restart()
    return v


def obs_at(v, idx):
    return {k: np.asarray(v.obs[k])[idx] for k in OBS_KEYS if k in v.obs and k != "priv"}


def to_dev(ob, dev):
    return {k: torch.as_tensor(x).to(dev) for k, x in ob.items()}


def policy_logits(model, ob, dev):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
        lg, _ = model(to_dev(ob, dev))
    return lg.float()


def sample_actions(lg, mask, gen, eps, nonround):
    """Sampled actions from the logits; at decisions outside rounds a share eps is a uniformly random legal action."""
    a = torch.multinomial(torch.softmax(lg, 1), 1, generator=gen).squeeze(1)
    if eps > 0 and nonround.any():
        rnd = torch.multinomial(mask.float() + 1e-9, 1, generator=gen).squeeze(1)
        use = nonround & (torch.rand(lg.shape[0], generator=gen, device=lg.device) < eps)
        a = torch.where(use, rnd, a)
    return a


def collect_roots(model, main, a, dev, gen, n_roots, anchors):
    """Play the main environments with the policy until n_roots decisions outside rounds are chosen. -> list of dicts (obs row, logits row), and the slots they were copied to."""
    roots, slots = [], []
    snap = a._snap
    per = max(1, n_roots // 8)                                   # roots per ante: every ante is sampled equally (the games run on to ante 8, so each has its turn)
    quota = np.zeros(9, dtype=int)
    steps = 0
    while len(roots) < n_roots and steps < 4000:
        steps += 1
        ob = obs_at(main, np.arange(main.n))
        lg = policy_logits(model, ob, dev)
        mask = torch.as_tensor(ob["mask"]).to(dev)
        nonround = torch.as_tensor(ob["g"][:, ROUND_COL] < 0.5).to(dev)
        pick = (nonround & (torch.rand(main.n, generator=gen, device=dev) < a.p_root)).cpu().numpy()
        for i in np.flatnonzero(pick):
            if len(roots) >= n_roots:
                break
            ante = int(round(float(ob["g"][i, 0]) * 8))
            if quota[min(max(ante, 1), 8)] >= per:
                continue
            quota[min(max(ante, 1), 8)] += 1
            s = len(roots)
            snap.copy_many(main, [int(i)], [s])
            roots.append({"obs": {k: x[i].copy() for k, x in ob.items()}, "logits": lg[i].cpu().numpy(), "ante": ante})
        for i in rng_anchor.choice(main.n, size=min(8, main.n), replace=False):                 # states the policy meets, rounds included
            anchors.append({k: (x[i].astype(np.float16) if x.dtype == np.float32 else x[i].copy()) for k, x in ob.items()})
        del anchors[:-a.anchor_buffer]
        act = sample_actions(lg, mask, gen, 0.0, nonround).cpu().numpy()
        main.step(act)
    return roots


def pick_alternatives(logits, mask, a, rng):
    legal = np.flatnonzero(mask)
    lg = np.where(mask, logits, -1e9)
    top = [int(x) for x in np.argsort(-lg)[:a.top]]
    alts = [x for x in top if mask[x]]
    if mask[E.A_LEAVE] and E.A_LEAVE not in alts:
        alts.append(E.A_LEAVE)
    rest = [int(x) for x in legal if int(x) not in alts]
    rng.shuffle(rest)
    alts += rest[:max(0, a.alts - len(alts))]
    return alts[:a.alts]


def rollouts(model, roll, snap, roots, alts_list, a, dev, gen, base_salt):
    """Play every (root, alternative, future) to the end. -> returns[root][alt] = list over futures of the undiscounted return."""
    jobs = [(r, k, j) for r, alts in enumerate(alts_list) for k in range(len(alts)) for j in range(a.futures)]
    ret = {(r, k): [0.0] * a.futures for r, alts in enumerate(alts_list) for k in range(len(alts))}
    for c0 in range(0, len(jobs), roll.n):
        chunk = jobs[c0:c0 + roll.n]
        m = len(chunk)
        roll.copy_many(snap, [r for r, _, _ in chunk], list(range(m)))
        roll.v.reseed(list(range(m)), [int(base_salt + 7919 * r + 104729 * j) for r, _, j in chunk])
        act = np.full(roll.n, -1, dtype=np.int64)
        for s, (r, k, j) in enumerate(chunk):
            act[s] = alts_list[r][k]
        G = np.zeros(roll.n)
        live = np.zeros(roll.n, dtype=bool)
        live[:m] = True
        t = 0
        while live.any() and t < 3000:
            rew, done, info = roll.step(act)
            G += np.where(live, rew, 0.0)
            live &= ~np.asarray(done)
            if not live.any():
                break
            idx = np.flatnonzero(live)
            ob = obs_at(roll, idx)
            lg = policy_logits(model, ob, dev)
            mask = torch.as_tensor(ob["mask"]).to(dev)
            nonround = torch.as_tensor(ob["g"][:, ROUND_COL] < 0.5).to(dev)
            nxt = sample_actions(lg, mask, gen, a.eps, nonround).cpu().numpy()
            act = np.full(roll.n, -1, dtype=np.int64)
            act[idx] = nxt
            t += 1
        for s, (r, k, j) in enumerate(chunk):
            ret[(r, k)][j] = float(G[s])
    return ret


def evaluate(model, a, dev, n=100, sampled=False):
    ev = CppVecEnv(n, a.deck, a.stake, seed0=9_000_000, threads=0)
    gen_e = torch.Generator(device=dev).manual_seed(11)
    done = np.zeros(n, dtype=bool)
    blinds = np.zeros(n)
    won = np.zeros(n, dtype=bool)
    for _ in range(3000):
        ob = obs_at(ev, np.arange(n))
        lg = policy_logits(model, ob, dev)
        act = (torch.multinomial(torch.softmax(lg, 1), 1, generator=gen_e).squeeze(1) if sampled else lg.argmax(1)).cpu().numpy()
        act = np.where(done, -1, act)
        _, d, info = ev.step(act)
        fin = np.asarray(d) & ~done
        blinds[fin] = np.asarray(info["blinds"])[fin]
        won[fin] = np.asarray(info["won"])[fin]
        done |= fin
        if done.all():
            break
    ev.close()
    return {"blinds": round(float(blinds.mean()), 3), "win_rate": round(float(won.mean()), 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True)
    p.add_argument("--out", default="checkpoints/branch1.pt")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--deck", default="RED")
    p.add_argument("--games", type=int, default=64, help="main environments the policy plays to find roots")
    p.add_argument("--roots", type=int, default=48, help="roots per iteration")
    p.add_argument("--p-root", type=float, default=0.05, help="chance that a decision outside a round becomes a root (until its ante's share is full)")
    p.add_argument("--alts", type=int, default=6, help="alternatives per root")
    p.add_argument("--top", type=int, default=3, help="of them, the policy's own top choices")
    p.add_argument("--futures", type=int, default=4, help="futures each alternative is played under")
    p.add_argument("--capacity", type=int, default=1152, help="rollout environments at once")
    p.add_argument("--eps", type=float, default=0.05, help="share of random actions outside rounds in the rollouts")
    p.add_argument("--tau-scale", type=float, default=0.5)
    p.add_argument("--weight-geom", type=float, default=1.6)
    p.add_argument("--win-bonus", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--train-steps", type=int, default=16)
    p.add_argument("--kappa", type=float, default=0.5, help="how much of the policy's own logit goes into the target (0: only the advantage)")
    p.add_argument("--root-beta", type=float, default=1.0, help="weight of KL(old || new) at the roots")
    p.add_argument("--anchor-beta", type=float, default=5.0, help="weight of the KL to the starting policy on states the policy met")
    p.add_argument("--anchor-batch", type=int, default=128)
    p.add_argument("--anchor-buffer", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=1500)
    p.add_argument("--iters", type=int, default=100000)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev = torch.device(a.device)
    model = load_model(a.init, dev)
    ref = load_model(a.init, dev)                         # the starting policy, frozen
    for q in ref.parameters():
        q.requires_grad_(False)
    ref.eval()
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    anchors = []
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    main_env = make_vec(a.games, a, 5_000_000 + a.seed * 100_000)
    a._snap = CppVecEnv(a.roots, a.deck, a.stake, seed0=1, threads=0)
    roll = CppVecEnv(a.capacity, a.deck, a.stake, seed0=2, threads=0, reward_kw={"win_bonus": a.win_bonus})
    roll.set_weight_geom(a.weight_geom)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    buf = []
    log = a.out.replace(".pt", "_log.jsonl")
    t_start = time.time()
    print(json.dumps({"start": vars(a) | {"_snap": None}}, default=str), flush=True)
    base = evaluate(model.eval(), a, dev)
    base_s = evaluate(model.eval(), a, dev, sampled=True)
    print(json.dumps({"initial_eval": base, "initial_eval_sampled": base_s}), flush=True)
    model.train()
    for it in range(1, a.iters + 1):
        t0 = time.time()
        model.eval()
        roots = collect_roots(model, main_env, a, dev, gen, a.roots, anchors)
        alts_list = [pick_alternatives(r["logits"], r["obs"]["mask"], a, rng) for r in roots]
        t1 = time.time()
        ret = rollouts(model, roll, a._snap, roots, alts_list, a, dev, gen, base_salt=1_000_003 * it)
        t2 = time.time()
        # labels
        Qs, head, differ, own_q, qbest = [], [], 0, [], []
        for r, (root, alts) in enumerate(zip(roots, alts_list)):
            Q = np.array([np.mean(ret[(r, k)]) for k in range(len(alts))])
            lg = np.where(root["obs"]["mask"], root["logits"], -1e9)
            greedy = int(np.argmax(lg))
            kg = alts.index(greedy) if greedy in alts else 0
            head.append(Q.max() - Q[kg])
            differ += int(int(np.argmax(Q)) != kg)
            own_q.append(Q[kg]); qbest.append(Q.max())
            Qs.append(Q)
        antes = np.array([r["ante"] for r in roots])
        tau_by = {}
        for ante in np.unique(antes):                              # each ante's labels are scaled by that ante's own spread (a late root has little left to gain)
            sel = [q - q.mean() for q, an in zip(Qs, antes) if an == ante]
            tau_by[int(ante)] = max(0.01, a.tau_scale * (float(np.std(np.concatenate(sel))) + 1e-6))
        tau = float(np.mean(list(tau_by.values())))
        for root, alts, Q in zip(roots, alts_list, Qs):
            lo = np.array([root["logits"][al] for al in alts], dtype=np.float64)
            z = a.kappa * (lo - lo.max()) + (Q - Q.max()) / tau_by[int(root["ante"])]
            pi = np.exp(z - z.max())
            pi /= pi.sum()
            buf.append({"obs": {k: (x.astype(np.float16) if x.dtype == np.float32 else x) for k, x in root["obs"].items()}, "alts": np.array(alts), "pi": pi.astype(np.float32),
                        "old": np.maximum(root["logits"], -1e4).astype(np.float16)})
        buf = buf[-a.buffer:]
        # training
        model.train()
        losses = []
        for _ in range(a.train_steps):
            idx = rng.integers(0, len(buf), size=min(a.batch, len(buf)))
            ob = {k: torch.as_tensor(np.stack([buf[i]["obs"][k] for i in idx])).to(dev) for k in buf[0]["obs"]}
            ob = {k: (x.float() if x.dtype == torch.float16 else x) for k, x in ob.items()}
            lg, _ = model(ob)
            lg = lg.float()
            A = max(len(buf[i]["alts"]) for i in idx)
            alts = torch.full((len(idx), A), 0, dtype=torch.long, device=dev)
            pi = torch.zeros((len(idx), A), device=dev)
            valid = torch.zeros((len(idx), A), dtype=torch.bool, device=dev)
            for b, i in enumerate(idx):
                n = len(buf[i]["alts"])
                alts[b, :n] = torch.as_tensor(buf[i]["alts"], device=dev)
                pi[b, :n] = torch.as_tensor(buf[i]["pi"], device=dev)
                valid[b, :n] = True
            la = lg.gather(1, alts).masked_fill(~valid, -1e9)
            loss_alt = -(pi * F.log_softmax(la, 1)).sum(1).mean()                  # cross entropy among the alternatives only
            old = torch.as_tensor(np.stack([buf[i]["old"] for i in idx])).to(dev).float()
            lp_old = F.log_softmax(old, 1)
            lp_cur = F.log_softmax(lg, 1)
            kl_root = (lp_old.exp() * (lp_old - lp_cur)).masked_fill(~ob["mask"], 0.0).sum(1).mean()      # a trust region at the roots
            loss = loss_alt + a.root_beta * kl_root
            if anchors and a.anchor_beta > 0:                                      # the policy stays as it was where nothing was learned
                ai = rng.integers(0, len(anchors), size=min(a.anchor_batch, len(anchors)))
                aob = {k: torch.as_tensor(np.stack([anchors[i][k] for i in ai])).to(dev) for k in anchors[0]}
                aob = {k: (x.float() if x.dtype == torch.float16 else x) for k, x in aob.items()}
                lg2, _ = model(aob)
                with torch.no_grad():
                    lgr, _ = ref(aob)
                msk = aob["mask"]
                lp_new = F.log_softmax(lg2.float(), 1)
                lp_ref = F.log_softmax(lgr.float(), 1)
                kl = (lp_ref.exp() * (lp_ref - lp_new)).masked_fill(~msk, 0.0).sum(1).mean()
                loss = loss + a.anchor_beta * kl
                ent_now = float(-(lp_new.exp() * lp_new).masked_fill(~msk, 0.0).sum(1).mean())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            losses.append(float(loss))
        t3 = time.time()
        head_a = {int(an): round(float(np.mean([h for h, x in zip(head, antes) if x == an])), 4) for an in np.unique(antes)}
        q_a = {int(an): round(float(np.mean([q for q, x in zip(own_q, antes) if x == an])), 3) for an in np.unique(antes)}
        row = {"headroom_by_ante": head_a, "q_by_ante": q_a, "iter": it, "roots": len(roots), "headroom": round(float(np.mean(head)), 4), "differ": round(differ / len(roots), 3), "q_greedy": round(float(np.mean(own_q)), 3),
               "q_best": round(float(np.mean(qbest)), 3), "tau": round(tau, 4), "loss": round(float(np.mean(losses)), 4), "buffer": len(buf),
               "secs": {"roots": round(t1 - t0, 1), "rollouts": round(t2 - t1, 1), "train": round(t3 - t2, 1)}, "min": round((time.time() - t_start) / 60, 1)}
        row["alt_loss"] = round(float(loss_alt), 4)
        row["kl"] = round(float(kl), 5) if anchors and a.anchor_beta > 0 else None
        row["entropy"] = round(ent_now, 3) if anchors and a.anchor_beta > 0 else None
        row["kl_root"] = round(float(kl_root), 5)
        if it % a.eval_every == 0:
            model.eval()
            row["eval"] = evaluate(model, a, dev)
            row["eval_sampled"] = evaluate(model, a, dev, sampled=True)
            if row["eval"]["blinds"] < base["blinds"] - 1.0 or row["eval_sampled"]["blinds"] < base_s["blinds"] - 1.0:                          # the policy got worse: back to the start with half the step
                model.load_state_dict(init_state)
                for g_ in opt.param_groups:
                    g_["lr"] *= 0.5
                row["reset"] = True
            else:
                torch.save({"config": model.config, "state_dict": model.state_dict()}, a.out)
            model.train()
        print(json.dumps(row), flush=True)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
