"""A Stockfish-style evaluation network (NNUE) for the known-seed game.

Input: a sparse set of active features (about 3,600 possible, 60-120 active), read off the engine's observation of a state outside a round:
    jokers held (which, edition, counter bucket, rental / perishable), hand levels and how often each hand was played, the deck's make-up
    (rank / suit / enhancement / seal counts, size), vouchers, money, ante x blind x phase, the boss, the consumables held, hands / discards /
    hand size / joker slots, and the seed's own upcoming shops (the jokers they offer this ante, the next two, and later: the engine's future
    list, VecEnv.set_priv).
Network: the active features' weight columns summed (the "accumulator", 256 wide) -> clipped ReLU -> 32 -> clipped ReLU -> 32 -> 1.
Output: the share of the 24 blinds the run is expected to reach (x 24 = blinds), trained on search-generated states labelled by playouts
(tools/nnue_data.py, tools/nnue_train.py).

    from balatro_rl.nnue import features, NNUE, load
    idx, offs = features(obs)           # obs: the observation arrays with "priv" (VecEnv.set_priv(True))
    net = load("checkpoints/nnue1.pt"); blinds = net.predict(idx, offs) * 24
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .runeval import Cols
from .sim.items import VOCAB, VOUCHERS, BOSS_KEYS
from .sim.jokers import JOKERS

NJ = len(JOKERS)
J0 = VOCAB.index("j_" + next(iter(JOKERS)))                  # VOCAB id of the first joker
C0 = VOCAB.index("j_unknown") + 1                             # VOCAB id of the first consumable
NC = 52
FUT0, FUT_W, FUT_K = 200, NJ + 32 + 15 + 28, 8


class _Layout:
    def __init__(self):
        self.off = {}
        n = 0
        for name, size in (("j_id", NJ), ("j_ed", NJ * 4), ("j_cnt", NJ * 8), ("j_rent", NJ), ("j_perish", NJ),
                           ("lvl", 12 * 10), ("played", 12 * 6), ("rank", 13 * 9), ("suit", 4 * 12), ("enh", 8 * 6), ("seal", 4 * 6), ("deck", 12),
                           ("vouch", 32), ("money", 16), ("phase", 8 * 3 * 4), ("boss", len(BOSS_KEYS)), ("cons", NC),
                           ("hands", 7), ("disc", 8), ("hsize", 10), ("jslots", 8), ("fut0", NJ), ("fut1", NJ), ("fut2", NJ), ("fut_later", NJ)):
            self.off[name] = n
            n += size
        self.n = n


L = _Layout()
N_FEATURES = L.n
MONEY_EDGES = np.array([0, 2, 4, 7, 11, 15, 19, 24, 29, 39, 49, 74, 99, 149, 249])


def _bucket(x, edges):
    return np.searchsorted(edges, x, side="left")


def features(o):
    """Observation arrays (g, jid, jf, cid, priv; batch first) -> (flat feature indices int64, offsets int64 [n + 1]).
    The states must be outside a round (blind select, shop or pack)."""
    g = np.asarray(o["g"], dtype=np.float32)
    jid = np.asarray(o["jid"]).astype(np.int64)
    jf = np.asarray(o["jf"], dtype=np.float32)
    cid = np.asarray(o["cid"]).astype(np.int64)
    priv = np.asarray(o["priv"], dtype=np.float32) if "priv" in o and o["priv"] is not None else None
    n = len(g)
    out, offs = [], [0]
    ante = np.clip(np.rint(g[:, Cols.ANTE] * 8), 1, 8).astype(int)
    blind = np.argmax(g[:, Cols.BLIND:Cols.BLIND + 3], 1)
    state = np.argmax(g[:, Cols.STATE:Cols.STATE + 4], 1)
    money = np.where(g[:, Cols.MONEY] < 1.99, g[:, Cols.MONEY] * 50.0, 10.0 ** (3.0 * g[:, Cols.LOG_MONEY]) - 1.0)
    money = np.where(g[:, Cols.NEG] > 0.5, -1.0, money)
    lv = np.clip(np.rint(g[:, Cols.LEVELS:Cols.LEVELS + 12] * 10), 1, 10).astype(int)
    played = g[:, Cols.PLAYED:Cols.PLAYED + 12] * 30.0
    ranks = np.clip(np.rint(g[:, Cols.RANKS:Cols.RANKS + 13] * 4), 0, 8).astype(int)
    suits = np.clip(np.rint(g[:, Cols.SUITS:Cols.SUITS + 4] * 13 / 3), 0, 11).astype(int)
    enh = np.rint(g[:, Cols.ENH:Cols.ENH + 8] * 10)
    seal = np.rint(g[:, Cols.SEAL:Cols.SEAL + 4] * 10)
    deck = np.clip(((np.rint(g[:, Cols.FULL] * 52) - 28) // 4), 0, 11).astype(int)
    vou = g[:, Cols.VOUCH:Cols.VOUCH + 32] > 0.5
    boss = g[:, Cols.BOSS:Cols.BOSS + len(BOSS_KEYS)]
    hands = np.clip(np.rint(g[:, Cols.ROUND_HANDS] * 5), 0, 6).astype(int)
    disc = np.clip(np.rint(g[:, Cols.ROUND_DISC] * 5), 0, 7).astype(int)
    hsize = np.clip(np.rint(g[:, Cols.HAND_SIZE] * 8) - 4, 0, 9).astype(int)
    jsl = np.clip(np.rint(g[:, Cols.JSLOTS] * 7), 0, 7).astype(int)
    eb = np.array([0, 1, 2, 4, 8, 1e9])

    def eb6(x):
        return np.minimum(np.searchsorted(eb, x, side="left"), 5)
    pb = np.array([0, 2, 5, 10, 20, 1e9])
    for i in range(n):
        f = []
        for k in range(jid.shape[1]):
            j = jid[i, k] - J0
            if not (0 <= j < NJ):
                continue
            f.append(L.off["j_id"] + j)
            for e in range(4):
                if jf[i, k, 2 + e] > 0.5:
                    f.append(L.off["j_ed"] + j * 4 + e)
            c = jf[i, k, 11]
            if c > 0:
                f.append(L.off["j_cnt"] + j * 8 + min(int(c * 8), 7))
            if jf[i, k, 8] > 0.5:
                f.append(L.off["j_rent"] + j)
            if jf[i, k, 7] > 0.05:
                f.append(L.off["j_perish"] + j)
        for h in range(12):
            if lv[i, h] > 1:
                f.append(L.off["lvl"] + h * 10 + lv[i, h] - 1)
            if played[i, h] > 0.5:
                f.append(L.off["played"] + h * 6 + int(np.searchsorted(pb, played[i, h], side="left")))
        for r in range(13):
            f.append(L.off["rank"] + r * 9 + ranks[i, r])
        for s_ in range(4):
            f.append(L.off["suit"] + s_ * 12 + suits[i, s_])
        for e in range(8):
            if enh[i, e] > 0:
                f.append(L.off["enh"] + e * 6 + int(eb6(enh[i, e])))
        for e in range(4):
            if seal[i, e] > 0:
                f.append(L.off["seal"] + e * 6 + int(eb6(seal[i, e])))
        f.append(L.off["deck"] + deck[i])
        f += [L.off["vouch"] + v for v in np.flatnonzero(vou[i])]
        f.append(L.off["money"] + int(min(_bucket(money[i], MONEY_EDGES), 15)))
        f.append(L.off["phase"] + ((ante[i] - 1) * 3 + blind[i]) * 4 + state[i])
        if boss[i].max() > 0.5:
            f.append(L.off["boss"] + int(np.argmax(boss[i])))
        for c in cid[i]:
            if 0 <= c - C0 < NC:
                f.append(L.off["cons"] + int(c - C0))
        f += [L.off["hands"] + hands[i], L.off["disc"] + disc[i], L.off["hsize"] + hsize[i], L.off["jslots"] + jsl[i]]
        if priv is not None:
            fut = priv[i, FUT0:FUT0 + FUT_K * FUT_W].reshape(FUT_K, FUT_W)[:, :NJ] > 0.5
            for r, name in ((0, "fut0"), (1, "fut1"), (2, "fut2")):
                f += [L.off[name] + j for j in np.flatnonzero(fut[r])]
            f += [L.off["fut_later"] + j for j in np.flatnonzero(fut[3:].any(0))]
        out.append(np.asarray(f, dtype=np.int64))
        offs.append(offs[-1] + len(f))
    return (np.concatenate(out) if out else np.zeros(0, dtype=np.int64)), np.asarray(offs, dtype=np.int64)


class NNUE(nn.Module):
    def __init__(self, n_features=N_FEATURES, width=256, h1=32, h2=32):
        super().__init__()
        self.acc = nn.EmbeddingBag(n_features, width, mode="sum")
        self.acc_bias = nn.Parameter(torch.zeros(width))
        self.l1 = nn.Linear(width, h1)
        self.l2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, 1)
        nn.init.normal_(self.acc.weight, std=0.02)
        self.config = dict(n_features=n_features, width=width, h1=h1, h2=h2)

    def forward(self, idx, offs):
        a = self.acc(idx, offs[:-1]) + self.acc_bias
        x = torch.clamp(a, 0.0, 1.0)                                  # clipped ReLU, as in Stockfish
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        return torch.sigmoid(self.out(x)).squeeze(-1)                 # the share of the 24 blinds reached

    @torch.no_grad()
    def predict(self, idx, offs, dev=None):
        dev = dev or next(self.parameters()).device
        return self(torch.as_tensor(idx, device=dev), torch.as_tensor(offs, device=dev)).cpu().numpy()


def load(path, dev="cpu"):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = NNUE(**ck["config"]).to(dev)
    net.load_state_dict(ck["state"])
    return net.eval()


def engine_layout():
    """The column indices, feature offsets and constants the engine's copy of features() uses (VecEnv.load_nnue)."""
    names = ("ANTE", "BLIND", "STATE", "MONEY", "LOG_MONEY", "NEG", "LEVELS", "PLAYED", "RANKS", "SUITS", "ENH", "SEAL", "FULL", "VOUCH", "BOSS",
             "ROUND_HANDS", "ROUND_DISC", "HAND_SIZE", "JSLOTS")
    cols = [int(getattr(Cols, k)) for k in names]
    order = ("j_id", "j_ed", "j_cnt", "j_rent", "j_perish", "lvl", "played", "rank", "suit", "enh", "seal", "deck", "vouch", "money", "phase", "boss",
             "cons", "hands", "disc", "hsize", "jslots", "fut0", "fut1", "fut2", "fut_later")
    off = [int(L.off[k]) for k in order]
    return cols, off, [int(J0), int(C0), int(NJ), int(NC), len(BOSS_KEYS)]


def export(net, path):
    """Writes the network for the engine (VecEnv.load_nnue): int32 header [features, width, h1, h2], then float32 accumulator weights
    [features x width], its bias, l1 [h1 x width], l1 bias, l2 [h2 x h1], l2 bias, out [h2], out bias."""
    import struct
    sd = {k: v.detach().cpu().float().numpy() for k, v in net.state_dict().items()}
    c = net.config
    with open(path, "wb") as f:
        f.write(struct.pack("<4i", c["n_features"], c["width"], c["h1"], c["h2"]))
        for k in ("acc.weight", "acc_bias", "l1.weight", "l1.bias", "l2.weight", "l2.bias", "out.weight", "out.bias"):
            f.write(np.ascontiguousarray(sd[k], dtype=np.float32).tobytes())
