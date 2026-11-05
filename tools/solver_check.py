"""Phase 3 exit check: does the round solver match brute force on small cases?

Blinds are started with few hands and discards left (so that trying every play and every discard is possible), and the solver at several widths is compared with the exhaustive one (every play and
discard at every step): the best share of the target any line reaches, and whether it finds a clearing line.

    python tools/solver_check.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from balatro_rl.engine_env import CppVecEnv             # noqa: E402
from balatro_rl.env import A_SELECT                     # noqa: E402

N = 512
v = CppVecEnv(N, "RED", "GOLD", seed0=31, threads=0)
v.set_exact(True)
v.v.restart()
rng = np.random.default_rng(0)
ids = set()
for step in range(900):
    m = np.asarray(v.obs["mask"])
    act = np.array([A_SELECT if m[e, A_SELECT] else rng.choice(np.flatnonzero(m[e])) for e in range(N)])
    inr = np.flatnonzero(m[:, :A_SELECT].any(1))
    if len(inr) and step % 5 == 0:
        g = np.asarray(v.obs["g"]); hl = np.rint(g[inr, 11] * 5).astype(int)
        ids |= set(inr[hl >= 4].tolist())
    v.step(act)
    if len(ids) > 200:
        break
ids = sorted(ids)[:150]
print(f"{len(ids)} blinds at their start")
WIDTHS = [("approximate (2 plays, 1 discard)", 2, 1, 0, 0, 60), ("default (4, 3)", 4, 3, 0, 0, 400), ("default + all at step 1", 4, 3, 218, 218, 30000), ("wide (8, 6)", 8, 6, 0, 0, 3000), ("wide + all at step 1", 8, 6, 218, 218, 60000), ("wider (16, 12)", 16, 12, 0, 0, 30000)]
for hands, discards in ((2, 0), (1, 1)):
    ref_cap = 4_000_000 if (hands, discards) != (2, 1) else 6_000_000
    # the same small cases for every width: the games are copies, then given the small resources
    t = time.time()
    v.set_round_resources(ids, hands, discards)
    v.set_solver(218, 218)
    ref = v.solve_best(ids, ref_cap)
    t_ref = time.time() - t
    truncated = int((ref[:, 2] >= ref_cap * 0.99).sum())
    print(f"\n{hands} hands, {discards} discards left: brute force {t_ref:.1f}s, {int(ref[:, 1].sum())}/{len(ids)} can be cleared (nodes cut off by the cap in {truncated} cases)")
    for name, npl, nd, f1, f2, cap in WIDTHS:
        v.set_round_resources(ids, hands, discards)
        v.set_solver(npl, nd, f1, f2)
        t = time.time()
        r = v.solve_best(ids, cap)
        dt = time.time() - t
        exact = np.isclose(r[:, 0], ref[:, 0], rtol=1e-6)
        clear_ok = (r[:, 1] >= ref[:, 1])
        missed = int(((ref[:, 1] > 0) & (r[:, 1] == 0)).sum())
        regret = np.maximum(ref[:, 0] - r[:, 0], 0)
        print(f"   {name:34s} {1000 * dt / len(ids):7.2f} ms/blind; same best score as brute force in {100 * exact.mean():5.1f}% of cases; clears missed {missed:3d}; mean shortfall of the best score {100 * regret.mean():.2f}% of the target")
    v.set_solver(4, 3)
