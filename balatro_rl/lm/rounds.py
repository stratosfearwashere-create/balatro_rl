"""An honest round player: each move is chosen by its average result over many possible draw orders, toward a style the strategist sets.

The old honest player (tools/gap_study.Honest) solves K copies of the round, each with one redrawn future that the solver then sees in full,
and lets them vote on the first move. Every copy "knows" its draws, so chasing a flush with both discards looks right in every copy, and the
real game loses the opening Small Blind 7.5% of the time with a plain deck (the solver that sees the real draws: 0%; this player: 1.5%).

Here a move is judged at its real odds:
  candidates   the best plays by exact score, the most promising discards (what the kept cards build toward: the engine's kept-card
               features), the moves the style asks for (the best play of the style's hand type, the best play with no face card when Ride
               the Bus is being grown, the discard of the style's cards), and each usable consumable unless the strategist decides those
  futures      --futures copies of the game with the unseen cards redrawn (Env::hide_future); the same redrawn orders for every candidate,
               so the comparison is paired
  continuation each copy finishes the round with a base policy that never peeks: play a hand that clears; otherwise, behind pace (best play
               x hands left < chips needed) with discards left, discard; otherwise play the best hand - leaning toward the style (its hand
               type when that keeps pace, face-free plays for Ride the Bus, the style's cards when discarding)
  value        cleared (1/0), then the style's points x STYLE_W (capped at STYLE_CAP), then money x MONEY_W (the money held after the
               round's pay: blind reward, unused hands, interest, Gold cards, jokers that pay). The weights keep the order: one more clear
               in 64 outweighs any style or money difference of a few points or dollars.

Styles (set per game by the strategist, cleared at the end of each round: styles[game] = dict):
  grow     a joker slot: points = the joker's counter growth over the round / its step (GROW_STEP); a reset (Ride the Bus on a face card,
           Green Joker on discards) counts as negative growth
  hand     a hand type (0-11): points = hands of that type played
  discard  ("rank", r) or ("suit", s): points = matching cards discarded (Hit the Road: Jacks; Castle: its suit; Mail-In Rebate: its rank)
  consumables  False: consumable uses are not candidates (the strategist decides them)
Same interface as tools/gap_study.Honest: moves(v, ids, mask, af).
"""
from __future__ import annotations

import numpy as np

from ..engine_env import CppVecEnv
from ..env import A_DISC, A_PLAY, A_USE_C, N_HANDS, N_SUB, RATIO_COL, SUBSETS, _OFF_PLAY

OFF_DISC = RATIO_COL + 5
OFF_HAND = _OFF_PLAY + 1                                       # a play row's hand-type one-hot
SUB_INDEX = {tuple(s): j for j, s in enumerate(SUBSETS)}
STYLE_W, STYLE_CAP, MONEY_W = 0.006, 0.05, 0.003
GROW_STEP = {"ride_the_bus": 1, "green_joker": 1, "runner": 15, "square": 4, "castle": 3, "trousers": 2, "wee": 8, "obelisk": 0.2,
             "yorick": 1, "lucky_cat": 0.25, "ice_cream": 5, "ramen": 0.01, "selzer": 1, "hiker": 5, "flash": 2, "red_card": 3,
             "constellation": 0.1, "hologram": 0.25, "madness": 0.5, "vampire": 0.1, "glass": 0.75, "campfire": 0.25, "swashbuckler": 1,
             "spare_trousers": 2}
FACE = (11, 12, 13)


def money_of(g):
    """money from the global features (env.encode_global: min(money, 100) / 50, then log10(money + 1) / 3)"""
    return np.where(g[:, 17] < 1.99, g[:, 17] * 50.0, 10.0 ** (3.0 * g[:, 18]) - 1.0)


def disc_potential(kf):
    """kf [..., 6]: the kept-card features of a discard (env.action_features) -> how promising it is (as nnue_ac.topk_mask)"""
    return np.maximum(np.maximum(kf[..., 0], kf[..., 1] * 1.25), kf[..., 2]) + 0.25 * kf[..., 3] + 0.05 * (1.0 - kf[..., 5])


def card_matches(c, crit) -> bool:
    if not crit or not isinstance(c, (list, tuple)) or str(c[2]).upper() == "STONE":
        return False
    kind, x = crit
    return (kind == "rank" and int(c[0]) == int(x)) or (kind == "suit" and int(c[1]) == int(x))


def style_discard_slot(hand, crit, legal_d):
    """the discard of the style's cards in this hand (at most 5), if legal -> slot or -1"""
    pos = tuple(k for k, c in enumerate(hand) if card_matches(c, crit))[:5]
    j = SUB_INDEX.get(pos, -1) if pos else -1
    return j if j >= 0 and legal_d[j] else -1


def face_free_plays(hand, legal_p):
    ok = np.zeros(N_SUB, bool)
    face = [isinstance(c, (list, tuple)) and int(c[0]) in FACE and str(c[2]).upper() != "STONE" for c in hand]
    for j in np.flatnonzero(legal_p):
        ok[j] = not any(face[k] for k in SUBSETS[j] if k < len(face))
    return ok


def base_actions(mask, af, hands=None, styles=None, slot_key=None):
    """The base policy for a batch of games in a round (mask [n, A], af [n, A, F]) -> actions [n]. hands: each game's hand (for the
    style's card checks); styles: each game's style dict or None; slot_key: each game's key of the grown joker (or None)."""
    n = mask.shape[0]
    mp = mask[:, A_PLAY:A_PLAY + N_SUB]
    md = mask[:, A_DISC:A_DISC + N_SUB]
    ratio = np.where(mp, af[:, A_PLAY:A_PLAY + N_SUB, RATIO_COL], -1.0)
    pace_p = np.where(mp, af[:, A_PLAY:A_PLAY + N_SUB, RATIO_COL + 4], -1.0)
    best_p = ratio.argmax(1)
    best_r = ratio.max(1)
    pace = pace_p.max(1)
    pot = np.where(md, disc_potential(af[:, A_DISC:A_DISC + N_SUB, OFF_DISC:OFF_DISC + 6]), -1e9)
    best_d = pot.argmax(1)
    can_d = md.any(1)
    act = A_PLAY + best_p
    chase = (best_r < 1.0) & (pace < 1.0) & can_d
    act = np.where(chase, A_DISC + best_d, act)
    if styles is not None:
        for k in range(n):
            st = styles[k]
            if not st:
                continue
            ok = None                                          # plays the style prefers
            if st.get("hand") is not None:
                ok = mp[k] & (af[k, A_PLAY:A_PLAY + N_SUB, OFF_HAND + int(st["hand"])] > 0.5)
            if slot_key is not None and slot_key[k] == "ride_the_bus" and hands is not None:
                ff = face_free_plays(hands[k], mp[k])
                ok = ff if ok is None else (ok & ff)
            if ok is not None and ok.any():
                good = ok & ((pace_p[k] >= 1.0) | (ratio[k] >= 1.0))
                if good.any() and (best_r[k] < 1.0 or ratio[k][good].max() >= 1.0):
                    act[k] = A_PLAY + int(np.argmax(np.where(good, ratio[k], -1.0)))
            crit = st.get("discard")
            if crit and hands is not None and can_d[k] and best_r[k] < 1.0:
                j = style_discard_slot(hands[k], crit, md[k])
                if j >= 0:
                    act[k] = A_DISC + j
    nothing = ~mp.any(1)                                       # (no play legal: a discard, else the first legal action)
    if nothing.any():
        for k in np.flatnonzero(nothing):
            act[k] = A_DISC + best_d[k] if can_d[k] else int(np.flatnonzero(mask[k])[0])
    return act.astype(np.int64)


class MCRounds:
    def __init__(self, v, deck, stake, size=4096, futures=64, plays=5, discards=6, threads=0, max_steps=40):
        self.H = CppVecEnv(size, deck, stake, seed0=2, threads=threads)    # (named H like Honest's scratch, so callers can configure it)
        self.H.v.restart()
        self.futures, self.plays, self.discards, self.max_steps = futures, plays, discards, max_steps
        self.salt = 0
        self.styles = {}                                       # game -> style dict (see the top of the file)
        self.last_values = {}                                  # game -> {action: mean value} of its last decision

    def candidates(self, mask_i, af_i, hand, st, grow_key):
        mp = mask_i[A_PLAY:A_PLAY + N_SUB]
        md = mask_i[A_DISC:A_DISC + N_SUB]
        out = []
        if mp.any():
            sc = np.where(mp, af_i[A_PLAY:A_PLAY + N_SUB, RATIO_COL + 1], -1e9)    # (log score: the exact order of the scores)
            out += [A_PLAY + int(j) for j in np.argsort(-sc)[:min(self.plays, int(mp.sum()))]]
            if st.get("hand") is not None:
                ok = mp & (af_i[A_PLAY:A_PLAY + N_SUB, OFF_HAND + int(st["hand"])] > 0.5)
                if ok.any():
                    out.append(A_PLAY + int(np.argmax(np.where(ok, sc, -1e9))))
            if grow_key == "ride_the_bus" and hand is not None:
                ff = face_free_plays(hand, mp)
                if ff.any():
                    out.append(A_PLAY + int(np.argmax(np.where(ff, sc, -1e9))))
        if md.any():
            pot = np.where(md, disc_potential(af_i[A_DISC:A_DISC + N_SUB, OFF_DISC:OFF_DISC + 6]), -1e9)
            out += [A_DISC + int(j) for j in np.argsort(-pot)[:min(self.discards, int(md.sum()))]]
            if st.get("discard") and hand is not None:
                j = style_discard_slot(hand, st["discard"], md)
                if j >= 0:
                    out.append(A_DISC + j)
        if st.get("consumables", True):
            out += [A_USE_C + k for k in range(3) if A_USE_C + k < len(mask_i) and mask_i[A_USE_C + k]]
        return list(dict.fromkeys(out))

    def moves(self, v, ids, mask, af, consider=None):
        """-> one action per game in ids. consider: an optional {game: [actions]} that replaces the candidates of those games."""
        ids = [int(i) for i in ids]
        out = np.zeros(len(ids), dtype=np.int64)
        jobs, cands, info_of = [], {}, {}
        for r, i in enumerate(ids):
            st = self.styles.get(i) or {}
            d = v.v.dump(i) if st else None
            hand = d["hand"] if d else None
            gslot = st.get("grow")
            gkey = d["jokers"][gslot]["key"] if (d and gslot is not None and 0 <= gslot < len(d["jokers"])) else None
            gval0 = float((d["jokers"][gslot].get("state") or {}).get("val", 0) or 0) if gkey else 0.0
            c = list(consider[i]) if consider and i in consider else self.candidates(mask[i], af[i], hand, st, gkey)
            if len(c) <= 1:
                out[r] = c[0] if c else int(np.flatnonzero(mask[i])[0])
                continue
            cands[r] = c
            info_of[r] = (st, gslot, gkey, gval0)
            jobs += [(r, a) for a in c]
        if not jobs:
            return out
        F = self.futures
        per = max(1, self.H.n // F)                            # candidates per batch of copies
        vals = np.zeros(len(jobs))
        self.salt += 1
        base_salt = {r: (self.salt * 1_000_003 + 7919 * r) for r in cands}
        for c0 in range(0, len(jobs), per):
            ch = jobs[c0:c0 + per]
            src = [ids[r] for r, _ in ch for _ in range(F)]
            m = len(src)
            dst = list(range(m))
            self.H.copy_many(v, src, dst)
            self.H.v.hide_future(dst, [base_salt[r] + 104729 * f for r, _ in ch for f in range(F)])   # the same F orders for every candidate
            styles = [info_of[r][0] for r, _ in ch for _ in range(F)] + [None] * (self.H.n - m)
            gslot = [info_of[r][1] for r, _ in ch for _ in range(F)]
            gkey = [info_of[r][2] for r, _ in ch for _ in range(F)] + [None] * (self.H.n - m)
            gval0 = np.array([info_of[r][3] for r, _ in ch for _ in range(F)])
            any_style = any(s for s in styles[:m])
            need_hand = any(s and (s.get("discard") or k == "ride_the_bus") for s, k in zip(styles[:m], gkey[:m]))
            act = np.full(self.H.n, -1, dtype=np.int64)
            act[:m] = [a for _, a in ch for _ in range(F)]
            live = np.zeros(self.H.n, bool)
            live[:m] = True
            val = np.zeros(self.H.n)
            pts = np.zeros(self.H.n)
            af_now = np.asarray(self.H.obs["af"], dtype=np.float32) if any_style else None
            for _ in range(self.max_steps):
                if any_style:                                  # style points of the moves about to be made
                    for k in np.flatnonzero(live & (act >= 0)):
                        st = styles[k]
                        if not st:
                            continue
                        a = int(act[k])
                        if st.get("hand") is not None and A_PLAY <= a < A_PLAY + N_SUB and af_now[k, a, OFF_HAND + int(st["hand"])] > 0.5:
                            pts[k] += 1
                        if st.get("discard") and A_DISC <= a < A_DISC + N_SUB:
                            hand = self.H.v.dump(int(k))["hand"]
                            pts[k] += sum(1 for p in SUBSETS[a - A_DISC] if p < len(hand) and card_matches(hand[p], st["discard"]))
                _, done, _info = self.H.step(act)
                g = np.asarray(self.H.obs["g"])
                inr = g[:, 5] > 0.5
                lost = live & done
                won = live & ~done & ~inr
                if won.any():                                  # cleared: in the shop with the round's pay already added
                    for k in np.flatnonzero(won):
                        if gkey[k] is not None:
                            js = self.H.v.dump(int(k))["jokers"]
                            sl = gslot[k]
                            if sl < len(js) and js[sl]["key"] == gkey[k]:
                                g1 = float((js[sl].get("state") or {}).get("val", 0) or 0)
                                pts[k] += (g1 - gval0[k]) / GROW_STEP.get(gkey[k], 1.0)
                    val[won] = 1.0 + np.clip(STYLE_W * pts[won], -STYLE_CAP, STYLE_CAP) + MONEY_W * money_of(g[won])
                live &= ~(lost | won)
                if not live.any():
                    break
                mk = np.asarray(self.H.obs["mask"])
                a_all = np.asarray(self.H.obs["af"], dtype=np.float32)
                af_now = a_all if any_style else None
                act = np.full(self.H.n, -1, dtype=np.int64)
                li = np.flatnonzero(live)
                hands = [self.H.v.dump(int(k))["hand"] for k in li] if need_hand else None
                act[li] = base_actions(mk[li], a_all[li], hands, [styles[k] for k in li] if any_style else None,
                                       [gkey[k] for k in li] if any_style else None)
            vals[c0:c0 + len(ch)] = val[:m].reshape(len(ch), F).mean(1)
        pos = 0
        for r, i in enumerate(ids):
            if r not in cands:
                continue
            c = cands[r]
            vv = vals[pos:pos + len(c)]
            pos += len(c)
            out[r] = c[int(np.argmax(vv))]
            self.last_values[i] = dict(zip(c, vv.tolist()))
        return out


class ClearSim:
    """The chance to clear the coming blinds, by playing them: copies of a game outside a round with the unseen future redrawn, every blind
    selected (no skips), shops left and packs skipped, rounds played by the honest base policy (base_actions; boss effects, the real targets
    and the deck included). Optionally a first move (a buy, a sale, a pack ...) is made on the copies before that.
    -> per job (p_next, p_boss): the chance to clear the next blind, and to get through this ante's Boss Blind, for the real round player:
    each blind's simulated clear rate (given the blinds before it) is mapped through sim_calib.json (the simple policy is weaker than
    MCRounds), and the boss chance is the product over the blinds to it."""

    def __init__(self, deck, stake, size=2048, threads=0, max_steps=400):
        self.E = CppVecEnv(size, deck, stake, seed0=5, threads=threads)
        self.E.set_auto_target(True)
        self.E.v.restart()
        self.max_steps = max_steps
        self.salt = 0

    def run(self, v, jobs, futures=32, through_boss=True, antes_ahead=0):
        """jobs: [(game in v, first action or None, blinds beaten now, ante now)]"""
        from ..env import A_LEAVE, A_PSKIP, A_SELECT
        E = self.E
        F = futures
        per = max(1, E.n // F)
        out = []
        for c0 in range(0, len(jobs), per):
            ch = jobs[c0:c0 + per]
            src = [g for g, _, _, _ in ch for _ in range(F)]
            m = len(src)
            dst = list(range(m))
            E.copy_many(v, src, dst)
            self.salt += 1
            E.v.hide_future(dst, [self.salt * 1_000_003 + 104729 * f + 7 for _ in ch for f in range(F)])   # the same futures for every job
            b0 = np.array([b for _, _, b, _ in ch for _ in range(F)] + [0] * (E.n - m))
            a0 = np.array([a for _, _, _, a in ch for _ in range(F)] + [0] * (E.n - m))
            stop_at = 3 * (a0 + (antes_ahead if through_boss else 0))  # play on through the boss of this ante (+ antes_ahead more)
            first = [a for _, a, _, _ in ch for _ in range(F)]
            live = np.zeros(E.n, bool)
            live[:m] = True
            best = b0.copy()                                   # the most blinds each copy has beaten
            boss = np.zeros(E.n, bool)
            act = np.full(E.n, -1, dtype=np.int64)
            for k in range(m):
                if first[k] is not None:
                    act[k] = first[k]
            if (act >= 0).any():
                E.step(act)
            for _ in range(self.max_steps):
                obs = E.obs
                g = np.asarray(obs["g"])
                mask = np.asarray(obs["mask"])
                act = np.full(E.n, -1, dtype=np.int64)
                li = np.flatnonzero(live)
                inr = g[li, 5] > 0.5
                ri = li[inr]
                if len(ri):
                    act[ri] = base_actions(mask[ri], np.asarray(obs["af"], dtype=np.float32)[ri])
                for k in li[~inr]:
                    mk = mask[k]
                    act[k] = A_SELECT if mk[A_SELECT] else A_LEAVE if mk[A_LEAVE] else A_PSKIP if mk[A_PSKIP] else int(np.flatnonzero(mk)[0])
                _, done, info = E.step(act)
                bl = np.asarray(info["blinds"])
                best = np.where(live, np.maximum(best, bl), best)
                boss |= live & (bl >= 3 * a0)
                live &= ~done
                live &= ~((best >= stop_at) if through_boss else (best > b0))
                if not live.any():
                    break
            for j, (_g, _a, bj, aj) in enumerate(ch):
                s = slice(j * F, (j + 1) * F)
                cond = []                                      # the clear rate of each blind up to this ante's boss, given the ones before it
                for k in range(max(1, 3 * (aj + (antes_ahead if through_boss else 0)) - bj)):
                    before = (best[s] >= bj + k).sum()
                    cond.append(float((best[s] > bj + k).sum() / before) if before else 0.0)
                    if not through_boss:
                        break
                cal = [calibrate(c) for c in cond]
                n_this = max(1, 3 * aj - bj)
                out.append((cal[0], float(np.prod(cal[:n_this])), np.cumprod(cal).tolist()))   # (next, this ante's boss, alive after each blind)
        return out


_CAL = None


def calibrate(p_sim):
    """the real round player's clear rate for a blind the simple base policy clears p_sim of the time (sim_calib.json)"""
    global _CAL
    if _CAL is None:
        import json
        import os
        _CAL = json.load(open(os.path.join(os.path.dirname(__file__), "sim_calib.json")))
    return float(np.interp(p_sim, _CAL["x"], _CAL["p"]))
