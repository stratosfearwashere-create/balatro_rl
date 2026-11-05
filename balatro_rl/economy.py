"""Future economy: how much money a run state will have over the coming rounds, worked out from the state alone (no search, no engine call).

The projection plays the next `rounds` blinds forward in expectation, in order, with the game's own payout rules (sim/game.py win_round):
    blind reward      small $0 at Red stake and up ($3 at White), big $4, boss $5
    unused hands      $1 each: the hands a round leaves = hands - the plays needed (the blind's target over the build's expected best play,
                      runeval's estimate; no growth assumed, so later rounds need more plays). A blind the build cannot clear ends the projection.
    interest          (1 + To the Moon) x $1 per $5 held at the end of the round, up to the cap (5, Seed Money 10, Money Tree 20)
    end of round      Golden Joker $4, Rocket its value (+$2 after a boss), Cloud 9 $1 per 9 in the deck, Satellite $1 per Planet type used,
                      Delayed Gratification $2 per discard if none were used, Gold cards held $3 each (Mime retriggers), rentals -$3 each
    in the round      Business Card, Reserved Parking, Golden Ticket, Rough Gem, Gold seals, Lucky cards, To Do List, Mail-In Rebate, Trading Card,
                      Faceless Joker, Matador: their expected payouts from the deck's make-up and the plays / discards the round uses
    sell value        Egg (+$3 a round on itself) and Gift Card (+$1 a round on every joker and consumable): not cash, reported apart
Perishable jokers stop paying when they expire; nothing is bought or sold and every blind is played (no skips) unless `spend` says how much
is spent in each shop.

    from balatro_rl.economy import future_economy
    out = future_economy(g, jid, jf, cid, rounds=9)          # out["money"][:, k] = money at the k-th coming blind select
    python -m balatro_rl.economy --check 256                 # against the engine playing the same rounds out
"""
from __future__ import annotations

import numpy as np

from .runeval import (RunEval, State, Cols, E, V, N_CARDS, FACE_MASK, JF_SELL, JF_RENTAL, JF_PERISH, JF_DEBUFF, I_BLIND_SELECT, default_weights)
from .sim.items import BLIND_BASE, BOSS_KEYS, ALL_BOSSES, item_id

JK = {k: item_id(f"j_{k}") for k in ("golden", "rocket", "cloud_9", "satellite", "delayed_grat", "egg", "gift", "to_the_moon", "business",
                                     "reserved_parking", "ticket", "rough_gem", "todo_list", "mail", "trading", "faceless", "matador", "mime",
                                     "credit_card", "oops")}
SEAL_GOLD = 2                                          # SEALS[1:] = RED, BLUE, GOLD, PURPLE
NINE = 7                                               # rank index of 9 (ranks 2 .. A)
DIAMONDS = 3                                           # suit order S H C D
_EV = None


def _ev():
    global _EV
    if _EV is None:
        _EV = RunEval(default_weights())
    return _EV


def future_economy(g, jid, jf, cid=None, rounds=9, spend=0.0, e_best=None, grow=0.0, pace=1.5):
    """Batch projection. spend: dollars spent in each shop (a number, or [n]); e_best: the build's expected best play per hand (default: runeval's);
    grow: the build's score grows by this factor per round (0: none); pace: plays needed = target / (expected best play / pace) (the player's
    plays fall short of the best one; 1.5 fits the engine's scripted player: python -m balatro_rl.economy). -> dict of arrays:
        money [n, rounds + 1]   money at the coming blind selects (column 0: now)
        income [n, rounds]      money gained in each round, by kind in parts[kind] [n, rounds]
        alive [n, rounds]       the build is projected to clear that blind
        sell_value [n]          sell value gained over the rounds (Egg, Gift Card)
        per_ante [n]            income over the first three rounds; sustainable [n]: a round's income with money held at the interest cap
    """
    s = State(g, jid, jf, cid)
    n = s.n
    if e_best is None:
        T, s2, dg = _ev().terms(g, jid, jf, cid)
        e_best = dg["e_best"]
        main = dg["main"]
    else:
        main = np.full(n, 1)
    e_best = np.asarray(e_best, dtype=np.float64)
    spend = np.broadcast_to(np.asarray(spend, dtype=np.float64), (n,))
    live = s.live
    has = lambda k: ((s.jid == JK[k]) & live).sum(1).astype(np.float64)
    perish = np.where(s.jf[:, :, JF_PERISH] > 0, np.rint(s.jf[:, :, JF_PERISH] * 5.0), 1e9)      # rounds left before a perishable joker is debuffed
    rental = ((s.jf[:, :, JF_RENTAL] > 0.5) & (s.jid > 0)).sum(1).astype(np.float64)
    cap = np.where(s.vouchers[:, V["money_tree"]], 20, np.where(s.vouchers[:, V["seed_money"]], 10, 5)).astype(np.float64)
    moon = has("to_the_moon")
    full = np.maximum(s.full, 1.0)
    face = (s.rank_share * FACE_MASK).sum(1)
    gold_c = s.enh[:, E["GOLD"]] / full
    lucky_c = s.enh[:, E["LUCKY"]] / full
    gold_seal = s.seals[:, SEAL_GOLD] / full
    diamonds = s.suit_share[:, DIAMONDS]
    nines = s.ranks[:, NINE]
    ncards = N_CARDS[np.asarray(main)]
    held = np.maximum(s.hand_size - ncards, 0.0)
    rocket_val = np.where((s.jid == JK["rocket"]) & live, np.maximum(s.counter, 1.0), 0.0).sum(1)
    odds = 1.0 + has("oops")
    scaling = np.where(s.stake >= 5, 3, np.where(s.stake >= 2, 2, 1))
    # the blind sequence from here: the next blind to be played and its ante
    nxt = np.where(s.state == I_BLIND_SELECT, s.blind, s.blind + 1)
    ante = s.ante.copy() + (nxt > 2)
    nxt = nxt % 3
    money = np.zeros((n, rounds + 1))
    money[:, 0] = s.money
    income = np.zeros((n, rounds))
    parts = {k: np.zeros((n, rounds)) for k in ("blind", "hands", "interest", "jokers", "cards", "rental")}
    alive = np.zeros((n, rounds), dtype=bool)
    plays_out = np.zeros((n, rounds))
    disc_out = np.zeros((n, rounds))
    ok = np.ones(n, dtype=bool)
    sell_value = np.zeros(n)
    m = s.money.astype(np.float64).copy()
    score = e_best.copy()
    for r in range(rounds):
        a = np.minimum(ante, 8)
        base = np.array([BLIND_BASE[int(sc)][int(x) - 1] for sc, x in zip(scaling, a)], dtype=np.float64)
        boss_mult = np.where((ante == s.ante) & s.boss_known, np.array([ALL_BOSSES[BOSS_KEYS[i]][2] for i in s.boss_idx], dtype=np.float64), 2.0)
        target = base * np.where(nxt == 0, 1.0, np.where(nxt == 1, 1.5, boss_mult)) * np.where(s.stake >= 8, 1.25, 1.0)
        plays = np.ceil(target * pace / np.maximum(score, 1.0))
        clears = ok & (plays <= s.hands) & (ante <= 8)
        alive[:, r] = clears
        plays = np.minimum(plays, s.hands)
        hands_left = np.maximum(s.hands - plays, 0.0)
        disc_used = np.where(hands_left >= 1, 0.0, s.discards)                         # a comfortable round keeps its discards
        plays_out[:, r] = plays
        disc_out[:, r] = disc_used
        active = lambda k: np.where((s.jid == JK[k]) & live & (perish > r), 1.0, 0.0).sum(1)
        blind = np.where(nxt == 0, np.where(s.stake >= 1, 0.0, 3.0), np.where(nxt == 1, 4.0, 5.0))
        interest = (1.0 + moon) * np.minimum(np.floor(np.maximum(m, 0.0) / 5.0), cap)
        jokers = (4.0 * active("golden") + np.where(active("rocket") > 0, rocket_val, 0.0) + nines * active("cloud_9") + s.planets_used * active("satellite")
                  + 2.0 * s.discards * (disc_used == 0) * active("delayed_grat") + 4.0 * active("todo_list") * plays / 8.0
                  + 5.0 * active("mail") * disc_used * 3.0 / 13.0 + 3.0 * active("trading") * (disc_used > 0) + 5.0 * active("faceless") * disc_used * 0.1
                  + 8.0 * active("matador") * (nxt == 2) * 0.7
                  + 2.0 * np.minimum(0.5 * odds, 1.0) * active("business") * plays * ncards * face
                  + 1.0 * np.minimum(0.5 * odds, 1.0) * active("reserved_parking") * plays * held * face)
        cards = (3.0 * (1.0 + active("mime")) * s.hand_size * gold_c + 4.0 * active("ticket") * plays * ncards * gold_c
                 + 1.0 * active("rough_gem") * plays * ncards * diamonds + 3.0 * plays * ncards * gold_seal
                 + 20.0 * np.minimum(odds / 15.0, 1.0) * plays * ncards * lucky_c)
        rent = 3.0 * rental
        earned = np.where(clears, blind + hands_left + interest + jokers + cards - rent, 0.0)
        for k, v in (("blind", blind), ("hands", hands_left), ("interest", interest), ("jokers", jokers), ("cards", cards), ("rental", -rent)):
            parts[k][:, r] = np.where(clears, v, 0.0)
        income[:, r] = earned
        sell_value += np.where(clears, 3.0 * active("egg") + active("gift") * (s.njokers + s.cused), 0.0)
        m = np.where(clears, m + earned - spend, m)
        m = np.maximum(m, -20.0 * (has("credit_card") > 0))
        money[:, r + 1] = np.where(clears, m, np.nan)
        rocket_val = rocket_val + 2.0 * (nxt == 2)
        ok = clears
        score = score * (1.0 + grow)
        nxt = nxt + 1
        ante = ante + (nxt > 2)
        nxt = nxt % 3
    per_ante = income[:, :3].sum(1)
    sust = parts["blind"][:, :3].mean(1) + parts["hands"][:, :3].mean(1) + parts["jokers"][:, :3].mean(1) + parts["cards"][:, :3].mean(1) \
        + parts["rental"][:, :3].mean(1) + (1.0 + moon) * cap
    return dict(money=money, income=income, parts=parts, alive=alive, sell_value=sell_value, per_ante=per_ante, sustainable=sust,
                plays=plays_out, discards=disc_out, main=np.asarray(main))


# ------------------------------------------------------------------ the check against the engine
def check(n_states=256, rounds=9, ckpt="checkpoints/ppo_gold_v30_best.pt", seed0=500_000, paces=(1.0,)):
    """States at blind selects along the network's games (scripted rounds), each played on by the engine for `rounds` blinds with every blind
    selected and every shop left (what the projection assumes); the money at each blind select against the projection."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    from runeval_data import Policy, play_out                                   # noqa: F401
    from .engine_env import CppVecEnv
    from .runeval import from_vecenv
    rng = np.random.default_rng(seed0)
    pol = Policy(ckpt, "cuda")
    none = Policy("none")
    G = 512
    roots = CppVecEnv(G, "RED", "GOLD", seed0=seed0, threads=0)
    roots.set_auto_round(True, 1.5, 0)
    roots.v.restart()
    bank = CppVecEnv(n_states, "RED", "GOLD", seed0=1, threads=0)
    bank.set_auto_round(True, 1.5, 0)
    fill, done = 0, np.zeros(G, dtype=bool)
    for _ in range(4000):
        live = np.flatnonzero(~done)
        if not len(live) or fill >= n_states:
            break
        g = np.asarray(roots.obs["g"])
        for i in live:
            if fill < n_states and g[i, Cols.STATE + I_BLIND_SELECT] > 0.5 and rng.random() < 0.08:
                bank.copy_from(roots, int(i), fill)
                fill += 1
        _, d, _ = roots.step(pol.act(roots, live))
        done |= d
    idx = list(range(fill))
    inputs = from_vecenv(bank, idx)
    projs = {pc: future_economy(**inputs, rounds=rounds, pace=pc) for pc in paces}
    proj = projs[paces[0]]
    actual = np.full((fill, rounds + 1), np.nan)
    gb = np.asarray(bank.obs["g"])[:fill]
    actual[:, 0] = State(gb, np.asarray(bank.obs["jid"])[:fill], np.asarray(bank.obs["jf"])[:fill]).money
    count = np.zeros(fill, dtype=int)
    alive = np.ones(fill, dtype=bool)
    for _ in range(3000):
        live = np.flatnonzero(alive)
        if not len(live):
            break
        g = np.asarray(bank.obs["g"])
        prev_sel = g[:, Cols.STATE + I_BLIND_SELECT] > 0.5
        _, d, _ = bank.step(none.act(bank, live))
        g = np.asarray(bank.obs["g"])
        st = State(g, np.asarray(bank.obs["jid"]), np.asarray(bank.obs["jf"]))
        for i in live:
            if d[i]:
                alive[i] = False
                continue
            if prev_sel[i] and g[i, Cols.STATE + I_BLIND_SELECT] <= 0.5:
                pass
            if g[i, Cols.STATE + I_BLIND_SELECT] > 0.5 and not prev_sel[i]:
                count[i] += 1
                if count[i] <= rounds:
                    actual[i, count[i]] = st.money[i]
                if count[i] >= rounds:
                    alive[i] = False
    am = actual
    for pc, pr in projs.items():
        pm = pr["money"]
        print(f"pace {pc}: {fill} states; money at the coming blind selects, projection vs the engine (both alive):")
        print(f"{'rounds ahead':>12} {'n':>5} {'mean actual':>12} {'mean proj':>10} {'mean |err|':>10} {'corr':>6}")
        for k in range(1, rounds + 1):
            okk = np.isfinite(pm[:, k]) & np.isfinite(am[:, k])
            if okk.sum() < 5:
                continue
            c = np.corrcoef(pm[okk, k], am[okk, k])[0, 1]
            print(f"{k:12d} {okk.sum():5d} {am[okk, k].mean():12.1f} {pm[okk, k].mean():10.1f} {np.abs(pm[okk, k] - am[okk, k]).mean():10.1f} {c:6.2f}")
    surv_p = proj["alive"].sum(1)
    surv_a = np.array([np.isfinite(am[i, 1:]).sum() for i in range(fill)])
    print(f"blinds survived of the next {rounds}: projected mean {surv_p.mean():.2f}, engine {surv_a.mean():.2f}, corr {np.corrcoef(surv_p, surv_a)[0, 1]:.2f}")
    return proj, actual


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", type=int, default=256)
    p.add_argument("--rounds", type=int, default=9)
    a = p.parse_args()
    check(a.check, a.rounds, paces=(1.0, 1.5, 2.0, 3.0))
