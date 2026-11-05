"""States and simulator labels for the run eval (balatro_rl/runeval.py; tools/runeval_fit.py fits the weights on them).

States come from two sources: (a) blind-select states (and a share of shop states) along games the labelling policy plays itself, from ante 1;
(b) random builds (tools/gen_builds.make_build) loaded with VecEnv.set_build at antes 2-8, so the late antes are covered. Every state is copied
--rollouts times, each copy's generator reseeded (new draws, new shops, new bosses), and played to the end by the same policy: the scripted round
player for rounds (auto_round mode 0: the spec's greedy v1, it sees only its hand) and the network for the decisions outside rounds (--ckpt;
"none" = select every blind and leave every shop: the build as it stands). The labels of a state are the shares of its rollouts that reach ante 4,
reach ante 7 and win, with the mean blinds beaten, the mean ante reached and the share that reach ante 4 holding $10+. The engine's exact probes
(fast_power, projected_power, final_boss_clear) are recorded for checking the hand-written terms.

    python tools/runeval_data.py --games 1024 --builds 2048 --rollouts 16 --out data/runeval/states_v1.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from balatro_rl import azc                                                       # noqa: E402
from balatro_rl.engine_env import CppVecEnv                                      # noqa: E402
from balatro_rl.env import A_SELECT, A_LEAVE, A_PSKIP, A_SKIP                    # noqa: E402
from balatro_rl.model import load_model                                          # noqa: E402
from balatro_rl.runeval import Cols, I_BLIND_SELECT, I_SHOP, I_PACK, JF_COUNTER  # noqa: E402

STATE_KEYS = ("g", "jid", "jf", "cid")


class Policy:
    """The decisions outside rounds: a network's greedy choice, or the fixed 'none' rule (select, leave, skip packs)."""

    def __init__(self, ckpt, dev="cuda", chunk=4096):
        self.none = ckpt in (None, "", "none")
        self.dev = dev if torch.cuda.is_available() else "cpu"
        self.chunk = chunk
        self.model = None if self.none else load_model(ckpt, self.dev).eval()

    @torch.no_grad()
    def act(self, v, live):
        mask = np.asarray(v.obs["mask"])
        act = np.full(v.n, -1, dtype=np.int64)
        if not len(live):
            return act
        if self.none:
            g = np.asarray(v.obs["g"])[live]
            st = np.argmax(g[:, Cols.STATE:Cols.STATE + 4], axis=1)
            for k, i in enumerate(live):
                m = mask[i]
                if st[k] == I_BLIND_SELECT and m[A_SELECT]:
                    act[i] = A_SELECT
                elif st[k] == I_SHOP and m[A_LEAVE]:
                    act[i] = A_LEAVE
                elif st[k] == I_PACK and m[A_PSKIP]:
                    act[i] = A_PSKIP
                else:
                    act[i] = int(np.flatnonzero(m)[0])
            return act
        for c0 in range(0, len(live), self.chunk):
            idx = live[c0:c0 + self.chunk]
            ob = azc.obs_batch(v, list(idx), self.dev)
            lg, _ = self.model(ob)
            lg = lg.float().masked_fill(~torch.from_numpy(mask[idx]).to(self.dev), -1e9)
            act[idx] = lg.argmax(1).cpu().numpy()
        return act


def play_out(v, policy, idx=None, max_steps=6000, watch_ante4=True):
    """Plays the environments idx (all by default) to the end. -> dict: won, blinds (beaten in the game), ante (reached: blinds // 3 + 1, 9 = won),
    money4 (money at the first blind select of ante 4, NaN if never there)."""
    n = v.n
    idx = np.arange(n) if idx is None else np.asarray(idx)
    done = np.ones(n, dtype=bool)
    done[idx] = False
    won = np.zeros(n, dtype=bool)
    blinds = np.zeros(n, dtype=np.int64)
    money4 = np.full(n, np.nan)
    for _ in range(max_steps):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        if watch_ante4:
            g = np.asarray(v.obs["g"])
            at4 = live[(np.rint(g[live, Cols.ANTE] * 8) == 4) & (g[live, Cols.STATE + I_BLIND_SELECT] > 0.5) & np.isnan(money4[live])]
            if len(at4):
                money4[at4] = np.where(g[at4, Cols.MONEY] < 1.99, g[at4, Cols.MONEY] * 50.0, 10.0 ** (3.0 * g[at4, Cols.LOG_MONEY]) - 1.0)
        act = policy.act(v, live)
        _, d, info = v.step(act)
        f = d & ~done
        won[f] = np.asarray(info["won"])[f] > 0.5
        blinds[f] = np.asarray(info["blinds"])[f]
        done |= d
    ante = np.where(won, 9, np.minimum(blinds // 3 + 1, 8))
    return dict(won=won[idx], blinds=blinds[idx], ante=ante[idx], money4=money4[idx])


def label_states(bank, m, K, policy, work, seed, probes=True, probe_env=None):
    """The m states in bank[0:m], K rollouts each, in `work` (capacity >= m x K). -> dict of label arrays [m] and probes (fast_power needs a dealt
    hand: it is measured on a copy in probe_env, without the scripted round, after the blind is selected)."""
    N = m * K
    work.copy_many(bank, [i // K for i in range(N)], list(range(N)))
    work.reseed_rng(list(range(N)), [int(seed * 1_000_003 + i * 7919 + 1) for i in range(N)])
    out = play_out(work, policy, np.arange(N))
    won = out["won"].reshape(m, K)
    ante = out["ante"].reshape(m, K)
    blinds = out["blinds"].reshape(m, K)
    money4 = out["money4"].reshape(m, K)
    lab = dict(win=won.mean(1), reach4=(ante >= 4).mean(1), reach7=(ante >= 7).mean(1), blinds_mean=blinds.mean(1), ante_mean=ante.mean(1),
               econ4=np.where((ante >= 4).any(1), np.nansum(np.where(ante >= 4, (money4 >= 10).astype(np.float64), 0.0), axis=1) / np.maximum((ante >= 4).sum(1), 1), 0.0))
    if probes:
        g = np.asarray(bank.obs["g"])[:m]
        sel = np.flatnonzero(g[:, Cols.STATE + I_BLIND_SELECT] > 0.5)
        fp = np.full((m, 4), np.nan)
        pp = np.full(m, np.nan)
        fbc = np.full((m, 2), np.nan)
        if len(sel):
            if probe_env is not None:
                probe_env.copy_many(bank, [int(i) for i in sel], list(range(len(sel))))
                act = np.full(probe_env.n, -1, dtype=np.int64)
                act[:len(sel)] = A_SELECT
                probe_env.step(act)
                fp[sel] = probe_env.fast_power(list(range(len(sel))), 8)
            pp[sel] = bank.projected_power(sel, 4)
            fbc[sel] = bank.final_boss_clear(sel, 1, 150)
        lab.update(fast_power=fp, projected_power=pp, final_boss=fbc)
    return lab


class Store:
    def __init__(self):
        self.rec = {k: [] for k in STATE_KEYS + ("ante", "blind", "state", "source", "game", "win", "reach4", "reach7", "blinds_mean", "ante_mean", "econ4",
                                                 "fast_power", "projected_power", "final_boss")}

    def add(self, bank, m, lab, source, game_ids):
        o = bank.obs
        self.rec["g"].append(np.asarray(o["g"])[:m].astype(np.float32))
        self.rec["jid"].append(np.asarray(o["jid"])[:m].astype(np.int16))
        self.rec["jf"].append(np.asarray(o["jf"])[:m, :, :JF_COUNTER + 2].astype(np.float16))
        self.rec["cid"].append(np.asarray(o["cid"])[:m].astype(np.int16))
        g = np.asarray(o["g"])[:m]
        self.rec["ante"].append(np.rint(g[:, Cols.ANTE] * 8).astype(np.int8))
        self.rec["blind"].append(np.argmax(g[:, Cols.BLIND:Cols.BLIND + 3], 1).astype(np.int8))
        self.rec["state"].append(np.argmax(g[:, Cols.STATE:Cols.STATE + 4], 1).astype(np.int8))
        self.rec["source"].append(np.full(m, source, dtype=np.int8))
        self.rec["game"].append(np.asarray(game_ids, dtype=np.int64))
        for k in ("win", "reach4", "reach7", "blinds_mean", "ante_mean", "econ4", "fast_power", "projected_power", "final_boss"):
            self.rec[k].append(np.asarray(lab[k]))

    def save(self, path, meta):
        out = {k: np.concatenate(v) for k, v in self.rec.items() if v}
        out["meta"] = np.array(str(meta))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path)
        return len(out["ante"])


def play_states(a, policy, store, bank, work, rng, probe):
    """Source (a): snapshots along the policy's own games."""
    G = a.games
    roots = CppVecEnv(G, "RED", a.stake, seed0=a.seed0, threads=0)
    roots.set_auto_round(True, 1.5, 0)
    roots.v.restart()
    done = np.zeros(G, dtype=bool)
    seen = set()
    B = bank.n
    fill = 0
    pending = []
    t0 = time.time()
    labelled = 0

    def flush():
        nonlocal fill, labelled
        if fill == 0:
            return
        lab = label_states(bank, fill, a.rollouts, policy, work, a.seed0 + labelled, probe_env=probe)
        store.add(bank, fill, lab, 0, pending)
        labelled += fill
        print(f"  {time.strftime('%H:%M')} play states labelled {labelled} (win {lab['win'].mean():.3f}, reach4 {lab['reach4'].mean():.2f}) {time.time() - t0:.0f}s", flush=True)
        fill = 0
        pending.clear()

    for _ in range(a.max_steps):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        g = np.asarray(roots.obs["g"])
        st = np.argmax(g[:, Cols.STATE:Cols.STATE + 4], 1)
        ante = np.rint(g[:, Cols.ANTE] * 8).astype(int)
        blind = np.argmax(g[:, Cols.BLIND:Cols.BLIND + 3], 1)
        for i in live:
            key = (int(i), int(ante[i]), int(blind[i]), int(st[i]))
            if key in seen:
                continue
            p = a.snap_p if st[i] == I_BLIND_SELECT else (a.snap_p / 3 if st[i] == I_SHOP else 0.0)
            if ante[i] >= 5:
                p = min(1.0, p * 2)                        # the late antes are rare: keep them
            if rng.random() < p:
                seen.add(key)
                bank.copy_from(roots, int(i), fill)
                pending.append(int(a.seed0 + i))
                fill += 1
                if fill == B:
                    flush()
        act = policy.act(roots, live)
        _, d, _ = roots.step(act)
        done |= d
    flush()


def build_states(a, policy, store, bank, work, rng, probe):
    """Source (b): random builds at antes 2-8 (tools/gen_builds.make_build), facing a random blind of the ante."""
    from gen_builds import make_build, strength_weights, KEYS
    W = strength_weights()
    B = bank.n
    made = 0
    t0 = time.time()
    while made < a.builds:
        m = min(B, a.builds - made)
        bank.v.restart()
        lo, hi = [int(x) for x in a.build_antes.split("-")]
        antes = rng.integers(lo, hi + 1, size=m)
        blinds = rng.integers(0, 3, size=m)
        for i in range(m):
            ks, cs, es, lv, su, sn, st, mo = make_build(rng, int(antes[i]), W)
            bank.set_build([i], int(antes[i]), int(blinds[i]), [KEYS.index(k) for k in ks], cs, st, lv, mo, es, su, sn)
        lab = label_states(bank, m, a.rollouts, policy, work, a.seed0 + 777_000 + made, probe_env=probe)
        store.add(bank, m, lab, 1, np.arange(made, made + m) + 10_000_000)
        made += m
        print(f"  {time.strftime('%H:%M')} build states labelled {made} (win {lab['win'].mean():.3f}, reach7 {lab['reach7'].mean():.2f}) {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/ppo_gold_v30_best.pt", help="the shop / blind policy of the labelling simulator ('none': select, leave)")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--games", type=int, default=1024, help="games the policy plays for source (a)")
    p.add_argument("--snap-p", type=float, default=0.5, help="chance to keep a blind-select state (shop states a third of it)")
    p.add_argument("--builds", type=int, default=2048, help="random builds for source (b)")
    p.add_argument("--build-antes", default="2-8", help="the antes the random builds are placed at")
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--bank", type=int, default=512, help="states labelled per batch (bank x rollouts environments at once)")
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--seed0", type=int, default=100_000)
    p.add_argument("--dev", default="cuda")
    p.add_argument("--out", default="data/runeval/states_v1.npz")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed0)
    policy = Policy(a.ckpt, a.dev)
    bank = CppVecEnv(a.bank, "RED", a.stake, seed0=1, threads=0)
    bank.set_auto_round(True, 1.5, 0)
    work = CppVecEnv(a.bank * a.rollouts, "RED", a.stake, seed0=2, threads=0)
    work.set_auto_round(True, 1.5, 0)
    probe = CppVecEnv(a.bank, "RED", a.stake, seed0=3, threads=0)          # (no scripted round: a select deals the hand the probe measures)
    store = Store()
    t0 = time.time()
    if a.games > 0:
        play_states(a, policy, store, bank, work, rng, probe)
    if a.builds > 0:
        build_states(a, policy, store, bank, work, rng, probe)
    n = store.save(a.out, dict(ckpt=a.ckpt, stake=a.stake, games=a.games, builds=a.builds, build_antes=a.build_antes, rollouts=a.rollouts, seed0=a.seed0, snap_p=a.snap_p))
    print(f"-> {a.out}: {n} states in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
