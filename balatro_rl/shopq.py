"""The shop policy improved from rollouts (policy iteration on the shop):

    1. DATA     full games are played with the current policy (a little sampling noise for variety); at some shop states every candidate action (and "no forced action": what the policy
                would do) is valued by paired rollouts (shopvalue.PairedRollouts): the same draws for every candidate, the run played to its end with the policy + the mode-2 round player.
                Label = Gain(a) = V(a) - V(policy's own choice).
    2. FIT      a Q head on the shop network's (state, action) rows regresses the Gain of every candidate (weighted by the rollouts' standard errors); the policy head is untouched.
    3. IMPROVE  QPolicy: in a shop take the action with the highest predicted Gain when it is above a threshold, else do what the network does.
    4. CHECK    exact-seed games with the old and the new policy (same seeds), blinds beaten.
    Then the new policy is the rollout policy of the next round.

    python -m balatro_rl.shopq --base checkpoints/shopA_best.pt --out checkpoints/shopq1 --rounds 3 --states 3000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import shopvalue as SV
from .env import A_BUY, A_LEAVE, A_PICK, A_PSKIP, A_SELECT

SAVE_KEYS = ["g", "cards", "jid", "jf", "cid", "aid", "af", "mask", "subs", "aid2"]
CMAX = 32


def dev_engine(path):
    if path:
        import sys
        sys.path.insert(0, path)
        import _engine_dev
        import balatro_rl.sim as S
        sys.modules["balatro_rl.sim._engine"] = _engine_dev
        S._engine = _engine_dev


def load_q(path, base, dev):
    """The shop network with a Q head: a shopq checkpoint ({'config', 'state_dict'}) or a hier checkpoint (the manager gets a fresh zero Q head)."""
    from .model import ActorCritic
    ck = torch.load(path, map_location=dev, weights_only=False)
    if "state_dict" in ck:
        m = ActorCritic(**ck["config"]).to(dev)
        m.load_state_dict(ck["state_dict"], strict=False)
    else:
        from .hier import load_hier
        _, m = load_hier(path, dev)
        m.add_qhead()
    m.eval()
    return m


def save_q(m, path):
    torch.save({"config": m.config, "state_dict": m.state_dict()}, path)


def make_env(n, a_deck, a_stake, seed0, exact=False, threads=0):
    from .engine_env import CppVecEnv
    v = CppVecEnv(n, a_deck, a_stake, seed0=seed0, threads=threads)
    if exact:
        v.set_exact(True)
        v.v.restart()
    v.set_auto_round(True, 1.5, 2)
    v.set_modes([2] * n)
    v.set_marginal(0)
    return v


@torch.no_grad()
def play_games(policy, n, seed0, deck="RED", stake="GOLD", max_steps=900):
    """Exact-seed games (the same seeds for every policy): -> (blinds [n], won [n])."""
    v = make_env(n, deck, stake, seed0, exact=True)
    done = np.zeros(n, dtype=bool)
    blinds, won = np.zeros(n), np.zeros(n, dtype=bool)
    for _ in range(max_steps):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        a = np.full(n, -1, dtype=np.int64)
        a[live] = policy(v, live)
        _, d, info = v.step(a)
        first = d & ~done
        blinds[first], won[first] = info["blinds"][first], info["won"][first]
        done |= d
    v.close()
    return blinds, won


@torch.no_grad()
def gen_data(rollout_policy, data_policy, n_states, n_seeds, out_path, envs=96, p_label=0.08, per_call=12, seed=1, deck="RED", stake="GOLD", log=print, scratch=False):
    rng = np.random.default_rng(seed)
    v = make_env(envs, deck, stake, seed * 1_000_000 + 17)
    pr = SV.PairedRollouts(rollout_policy, cap=4096, deck=deck, stake=stake)
    store = {k: [] for k in SAVE_KEYS}
    cands_all, v_all, se_all = [], [], []
    t0, salt, steps = time.time(), 1_000_000 * seed, 0
    while len(cands_all) < n_states:
        mask = np.asarray(v.obs["mask"])
        shop = np.flatnonzero(mask[:, A_SELECT:].any(1)) if scratch else np.flatnonzero(mask[:, A_LEAVE])      # (scratch: every decision outside a round; else shop states)
        pick = shop[rng.random(len(shop)) < p_label][:per_call]
        if len(pick):
            states = [(int(e), SV.all_candidates(mask[e]) if scratch else [-1] + SV.shop_candidates(mask[e])) for e in pick]
            while sum(len(c) for _, c in states) * n_seeds > pr.cap:
                states.pop()
            res = pr.values_many(v, states, n_seeds, salt0=salt)
            salt += 1000
            for (e, cands), r in zip(states, res):
                for k in SAVE_KEYS:
                    x = np.array(v.obs[k][e])
                    store[k].append(x.astype(np.float16) if k == "af" else x.astype(np.int8) if k == "subs" else x)
                c = np.full(CMAX, -2, dtype=np.int16); c[:len(cands)] = cands
                vv = np.zeros(CMAX, dtype=np.float32); vv[:len(cands)] = r["v"]
                ss = np.zeros(CMAX, dtype=np.float32); ss[:len(cands)] = r["se"]
                cands_all.append(c); v_all.append(vv); se_all.append(ss)
            if len(cands_all) % 240 < len(pick):
                log(f"  {len(cands_all)}/{n_states} labelled states, {time.time() - t0:.0f}s")
        a = data_policy(v, np.arange(envs))
        v.step_split(a)
        steps += 1
    n = len(cands_all)
    np.savez_compressed(out_path, **{k: np.stack(x) for k, x in store.items()}, cands=np.stack(cands_all), v=np.stack(v_all), se=np.stack(se_all))
    v.close()
    return n


def load_shards(paths):
    out = None
    for p in paths:
        z = np.load(p)
        d = {k: z[k] for k in z.files}
        out = d if out is None else {k: np.concatenate([out[k], d[k]]) for k in out}
    return out


def to_batch(d, idx, dev):
    o = {k: torch.from_numpy(d[k][idx]).to(dev) for k in SAVE_KEYS if k in d}
    o["af"] = o["af"].float()
    o["subs"] = o["subs"].long()
    o["aid"] = o["aid"].long(); o["jid"] = o["jid"].long(); o["cid"] = o["cid"].long()
    if "aid2" in o:
        o["aid2"] = o["aid2"].long()
    return o


def train_q(model, d, dev, epochs=20, batch=256, lr=3e-4, holdout=0.1, all_params=False, log=print, seed=0, centered=False):
    """Regress Gain(a) = V(a) - V(policy) on the Q head. Returns the held-out report."""
    rng = np.random.default_rng(seed)
    n = len(d["cands"])
    perm = rng.permutation(n)
    nv = max(1, int(n * holdout))
    va, tr = perm[:nv], perm[nv:]
    for p in model.parameters():
        p.requires_grad_(all_params)
    for p in model.qhead.parameters():
        p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    def targets(idx):
        c = torch.from_numpy(d["cands"][idx].astype(np.int64)).to(dev)
        v = torch.from_numpy(d["v"][idx]).to(dev)
        se = torch.from_numpy(d["se"][idx]).to(dev)
        valid = c >= 0
        if centered:                                   # from scratch there is no policy to compare with: an action's value against the mean of the candidates of its state
            gain = (v - (v * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp(min=1)) * valid
            var = se ** 2 + 0.05
        else:
            gain = v - v[:, :1]
            var = se ** 2 + se[:, :1] ** 2 + 0.05
        return c, gain, 1.0 / var, valid

    def predict(idx):
        o = to_batch(d, idx, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
            model(o)
        q = model.q_values().float()
        c, gain, w, valid = targets(idx)
        qc = q.gather(1, c.clamp(min=0))
        return qc, gain, w, valid, c

    def report(idx):
        model.eval()
        with torch.no_grad():
            qc, gain, w, valid, c = predict(idx)
            m = valid
            corr = np.corrcoef(qc[m].cpu().numpy(), gain[m].cpu().numpy())[0, 1]
            # the action the Q head would take against the best candidate, in true gain
            qq = qc.masked_fill(~m, -1e9)
            best_q = qq.argmax(1)
            ch = gain.gather(1, best_q[:, None]).squeeze(1)
            top = gain.masked_fill(~m, -1e9).max(1).values
            sign = ((qc > 0) == (gain > 0))[m].float().mean().item()
            take = qq.max(1).values > 0
            chosen = torch.where(take, ch, torch.zeros_like(ch))
            return {"corr": round(float(corr), 3), "sign_acc": round(sign, 3), "gain_if_follow_q": round(float(chosen.mean()), 3), "gain_of_best_possible": round(float(top.clamp(min=0).mean()), 3),
                    "share_taking_q": round(float(take.float().mean()), 3)}
    log("  before fit: " + json.dumps(report(va)))
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        tot = 0.0
        for s0 in range(0, len(order), batch):
            idx = order[s0:s0 + batch]
            qc, gain, w, valid, c = predict(idx)
            loss = ((qc - gain) ** 2 * w * valid).sum() / (w * valid).sum().clamp(min=1e-6)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 4 or ep == epochs - 1:
            log(f"  epoch {ep + 1}: train loss {tot / len(order):.4f}; held out " + json.dumps(report(va)))
    model.eval()
    return report(va)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="checkpoints/shopA_best.pt", help="a hier checkpoint (its manager is the first policy) or a shopq checkpoint")
    p.add_argument("--out", default="checkpoints/shopq1")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--states", type=int, default=3000, help="labelled shop states per round")
    p.add_argument("--seeds", type=int, default=32, help="rollouts per candidate")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--eval-games", type=int, default=256)
    p.add_argument("--eval-seed0", type=int, default=9_000_000)
    p.add_argument("--scratch", action="store_true", help="learn from scratch: an untrained network whose policy is its own Q head (softmax(Q / tau)), every decision outside rounds labelled")
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--thr", type=float, default=0.0, help="the predicted Gain a Q action needs to override the network")
    p.add_argument("--all-params", action="store_true", help="fine-tune the whole network on the Q loss (default: only the Q head)")
    p.add_argument("--dev-engine", default="C:/Users/strat/AppData/Local/Temp/claude/C--balatro-rl/85bbfca6-42a2-4515-a6d7-78ba00e9b1d8/scratchpad/dev")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    dev_engine(a.dev_engine)
    dev = a.device
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.scratch:
        from .model import ActorCritic
        ck = torch.load(a.base, map_location="cpu", weights_only=False)
        cfg = dict(ck["config"]["manager"]) if "config" in ck and "manager" in ck["config"] else dict(ck["config"])
        cfg.pop("goal_out", None); cfg.pop("goal_in", None); cfg.pop("qhead", None)
        torch.manual_seed(a.seed)
        model = ActorCritic(**cfg).to(dev)                                    # the architecture of the base network, untrained weights
        model.add_qhead()
        model.eval()
        base_pol = SV.SoftQPolicy(model, dev, tau=0.0)
        label0 = "round 0 (the untrained starting policy)"
    else:
        model = load_q(a.base, a.base, dev)
        base_pol = SV.ManagerPolicy(model, dev)
        label0 = "round 0 (the base shop network)"
    b0, w0 = play_games(base_pol, a.eval_games, a.eval_seed0)
    print(f"{label0}: blinds {b0.mean():.3f}, wins {int(w0.sum())}/{a.eval_games}", flush=True)
    log = lambda s: print(s, flush=True)
    for rd in range(1, a.rounds + 1):
        t0 = time.time()
        if a.scratch:
            roll = SV.SoftQPolicy(model, dev, tau=a.tau, seed=rd)
            data_pol = SV.SoftQPolicy(model, dev, tau=a.tau, explore=0.05, seed=rd)
        else:
            roll = SV.QPolicy(model, dev, thr=a.thr) if rd > 1 else SV.ManagerPolicy(model, dev)
            data_pol = SV.QPolicy(model, dev, thr=a.thr, temperature=0.7, explore=0.04, seed=rd) if rd > 1 else SV.ManagerPolicy(model, dev, temperature=0.7, explore=0.04, seed=rd)
        path = f"{a.out}_data{rd}.npz"
        n = gen_data(roll, data_pol, a.states, a.seeds, path, seed=a.seed + rd, log=log, scratch=a.scratch, p_label=0.04 if a.scratch else 0.08)
        log(f"round {rd}: {n} states labelled in {time.time() - t0:.0f}s")
        d = load_shards([f"{a.out}_data{k}.npz" for k in range(1, rd + 1)])
        rep = train_q(model, d, dev, a.epochs, all_params=a.all_params or a.scratch, log=log, centered=a.scratch)
        save_q(model, f"{a.out}_r{rd}.pt")
        pol = SV.SoftQPolicy(model, dev, tau=0.0) if a.scratch else SV.QPolicy(model, dev, thr=a.thr)
        b, w = play_games(pol, a.eval_games, a.eval_seed0)
        print(f"round {rd}: held-out {json.dumps(rep)}\n         new policy: blinds {b.mean():.3f} (base {b0.mean():.3f}; paired difference {np.mean(b - b0):+.3f} +- {np.std(b - b0) / np.sqrt(len(b)):.3f}), wins {int(w.sum())}/{a.eval_games}, "
              + (f"Q overrode the network at {pol.used}/{pol.calls} shop decisions" if not a.scratch else "the Q head is the policy: greedy"), flush=True)


if __name__ == "__main__":
    main()
