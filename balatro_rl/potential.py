"""The judge as a potential for the shop network: how well the build would do at ante 8, projected, from the networks of judge.py.

Phi(s), at a blind select, is the log of the chance of getting through the next blind and the ante-8 boss with the build as it is, in the scripted player's better mode, with
each joker's counter carried forward to the ante-8 boss at the rate the scaling model predicts for a round. It is clipped to [floor, 0]; a win is 0 and a death the floor.
The shop network is paid weight x (Phi after - Phi before) when a blind select is reached: a potential, so what it adds up to over a game is where the game ends (0 won, floor lost)
minus where it began, and passing a blind or improving the build raises it.

The projection asks the engine, not a guess, about the future state: the game is copied (VecEnv.copy_many), the copy is put at the ante-8 boss with the projected counters
(VecEnv.set_state) and its observation, with the exact expected-score columns, goes to the judge.
"""
from __future__ import annotations

import numpy as np
import torch

from .env import F_JOKER_W, item_id

# the engine's own growth table (game.hpp SCN_GROW): name -> (start, per round, floor, ceiling). 'Fully scaled' = this joker held to the ante-8 boss at this rate.
GROW = {"ride_the_bus": (0, 2.5, 0, 40), "green_joker": (0, 2, 0, 60), "runner": (0, 10, 0, 400), "square": (0, 8, 0, 300), "red_card": (0, 3, 0, 60), "trousers": (0, 2, 0, 60),
        "flash": (0, 6, 0, 120), "wee": (0, 8, 0, 300), "castle": (0, 5, 0, 200), "constellation": (1, .12, 1, 6), "obelisk": (1, .25, 1, 8), "vampire": (1, .15, 1, 6),
        "hologram": (1, .2, 1, 8), "lucky_cat": (1, .15, 1, 6), "campfire": (1, .3, 1, 5), "glass": (1, .2, 1, 6), "hit_the_road": (1, .3, 1, 5), "caino": (1, .3, 1, 8),
        "ice_cream": (100, -12, 5, 100), "popcorn": (20, -4, 4, 20), "ramen": (2.0, -.1, 1.05, 2.0), "selzer": (10, -2, 1, 10), "turtle_bean": (5, -1, 1, 5)}


class JudgePotential:
    # the ante-8 boss's pass probability as a smooth function of how close the build gets to its requirement (the judge's ratio output), fitted on the judge's data
    # (ante 8, boss, mode 1: pass = sigmoid(5.24 x (log10 best share of the target + 0.09)), Brier 0.017): the probability itself is ~0 for any build a player makes, which has no gradient
    A8, C8 = 5.24, -0.09

    def __init__(self, venv, la, judge, scaling, dev, floor=-20.0, money_w=0.0, full=False):
        self.venv, self.la, self.judge, self.scaling, self.dev, self.floor = venv, la, judge, scaling, dev, floor
        self.money_w = money_w                                  # nats for a full bank ($25, where interest stops): spending is a trade, not free
        self.calls = 0
        self.full = full                                        # project every scaler to its natural ante-8 value (the table above) and drop the next-blind term
        if full:
            n = 1 + max(item_id("j_" + k) for k in GROW)
            self.tab = torch.zeros(n, 5, device=dev)           # [is scaler, start, rate, lo, hi] by item id
            for k, (a, r, lo, hi) in GROW.items():
                self.tab[item_id("j_" + k)] = torch.tensor([1.0, a, r, lo, hi], device=dev)

    def _obs(self, v, ids):
        g = torch.from_numpy(np.asarray(v.obs["g"])[ids]).to(self.dev)
        jid = torch.from_numpy(np.asarray(v.obs["jid"])[ids]).to(self.dev)
        jf = torch.from_numpy(np.asarray(v.obs["jf"])[ids][:, :, :F_JOKER_W]).to(self.dev)
        cid = torch.from_numpy(np.asarray(v.obs["cid"])[ids]).to(self.dev)
        g, jf = torch.nan_to_num(g, nan=0.0, posinf=10.0, neginf=-10.0), torch.nan_to_num(jf, nan=0.0, posinf=10.0, neginf=-10.0)      # (a state whose features overflowed)
        return g, jid, jf, cid

    @torch.no_grad()
    def _p_pass(self, g, jid, jf, cid):
        p = [self.judge.predict(g, jid, jf, cid, torch.full((g.shape[0],), m, device=self.dev, dtype=torch.long))["p_pass"] for m in (0, 1)]
        return torch.maximum(p[0], p[1])                                      # the manager can choose the better mode

    @torch.no_grad()
    def _log_ratio(self, g, jid, jf, cid):
        r = [self.judge.predict(g, jid, jf, cid, torch.full((g.shape[0],), m, device=self.dev, dtype=torch.long))["log_ratio"] for m in (0, 1)]
        return torch.maximum(r[0], r[1])

    @torch.no_grad()
    def phi(self, ids, src=None, proj=None) -> np.ndarray:
        """Phi for the environments `ids` of `src` (default: the training venv; at any decision point) -> [len(ids)] in [floor, 0]. `proj`: the scratch VecEnv
        the projected copies are made in (default: the potential's own), indexed like `src`."""
        src = self.venv if src is None else src
        proj = self.la if proj is None else proj
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return np.zeros(0)
        self.calls += len(ids)
        g, jid, jf, cid = self._obs(src, ids)
        if not self.full:
            p_now = self._p_pass(g, jid, jf, cid)
            dc = self.scaling.predict(g, jid, jf, cid, torch.ones(len(ids), device=self.dev, dtype=torch.long))["dc"]       # per round, in the counters' own units
        ante = (g[:, 0] * 8.0).round().clamp(1, 8)
        blind = g[:, 1:4].argmax(1).float()
        rounds = (3.0 * (8.0 - ante) + (2.0 - blind)).clamp(min=0)                # rounds until the ante-8 boss
        cur = torch.expm1(4.0 * jf[:, :, 11].float())                             # the observation holds log1p(|counter|) / 4
        if self.full:
            t = self.tab[jid.clamp(0, self.tab.shape[0] - 1).long()]
            proj_t = torch.minimum(torch.maximum(cur + t[..., 2] * rounds[:, None], t[..., 3]), t[..., 4])
            live = (jid > 0) & (t[..., 0] > 0)
        else:
            proj_t = torch.minimum((cur + dc * rounds[:, None]).clamp(min=0.0), cur * 3.0 + 20.0)        # at most about triple the counter: a growth rate is not a promise
            live = (jid > 0) & (dc.abs() > 1e-3)
        proj_t = torch.where(live, proj_t, torch.full_like(proj_t, float("nan")))
        pr = proj
        pr.copy_many(src, ids.tolist(), ids.tolist())
        pr.set_state(ids.tolist(), 8, 2, proj_t.cpu().numpy())
        g2, jid2, jf2, cid2 = self._obs(pr, ids)
        lr8 = self._log_ratio(g2, jid2, jf2, cid2)
        phi = torch.nn.functional.logsigmoid(self.A8 * (lr8 - self.C8))
        if not self.full:
            phi = phi + torch.log(p_now.clamp(min=1e-3))
        phi = torch.nan_to_num(phi, nan=self.floor).clamp(self.floor, 0.0)
        if self.money_w > 0:
            phi = phi + self.money_w * (g[:, 17] * 50.0 / 25.0).clamp(0.0, 1.0)         # g column 17 is min(money, 100) / 50
        return phi.cpu().numpy().astype(np.float64)
