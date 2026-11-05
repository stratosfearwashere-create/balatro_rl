"""Demonstrations for imitation (tools/make_demos.py): the observation at each non-round decision of a strong line and the action the line took.

The file keeps the per-action arrays only for the legal actions; sample() rebuilds dense observations, shaped and typed like the trainer's own (a template observation
dict gives the keys, shapes and dtypes), on the device.
"""
from __future__ import annotations

import numpy as np
import torch


class DemoSet:
    def __init__(self, path: str, template: dict, device, seed: int = 0, won_weight: float = 1.0):
        import os
        self.path, self.won_weight, self.calls = path, won_weight, 0
        self.mtime = os.path.getmtime(path)
        self.template = {k: (tuple(v.shape[1:]), v.dtype) for k, v in template.items()}
        self.rng = np.random.default_rng(seed)
        self.dev = device
        self._load()

    def maybe_reload(self):
        """The file is rebuilt as more lines are found: reload it when it changes (checked every 50 samples)."""
        import os
        self.calls += 1
        if self.calls % 50:
            return False
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            return False
        if m != self.mtime:
            try:
                self._load()
                self.mtime = m
                return True
            except Exception:                         # (a file being written: try again later)
                return False
        return False

    def _load(self):
        z = np.load(self.path)
        device = self.dev
        self.n = int(len(z["act"]))
        self.act = torch.from_numpy(z["act"]).to(device)
        self.dense = {k: torch.from_numpy(z[k].astype(np.float32) if z[k].dtype == np.float16 else z[k]).to(device) for k in ("g", "cards", "jid", "jf", "cid", "subs", "priv", "ex") if k in z.files}
        self.offs = z["offs"]
        self.l_idx = torch.from_numpy(z["legal_idx"].astype(np.int64)).to(device)
        self.l_af = torch.from_numpy(z["legal_af"].astype(np.float32)).to(device)
        self.l_aid = torch.from_numpy(z["legal_aid"].astype(np.int64)).to(device)
        self.l_aid2 = torch.from_numpy(z["legal_aid2"].astype(np.int64)).to(device)
        self.rows = torch.from_numpy(np.repeat(np.arange(self.n), np.diff(self.offs))).to(device)   # the sample each legal row belongs to
        w = np.where(z["won"], self.won_weight, 1.0)
        self.p = w / w.sum()
        self.won_share = float(z["won"].mean())

    def sample(self, b: int):
        """-> (observation dict for b demonstrations, their actions)."""
        self.maybe_reload()
        ids = np.sort(self.rng.choice(self.n, size=b, p=self.p))
        idt = torch.from_numpy(ids).to(self.dev)
        out = {}
        for k, (shape, dt) in self.template.items():
            if k in self.dense:
                out[k] = self.dense[k][idt].to(dt)
            else:
                out[k] = torch.zeros((b,) + shape, dtype=dt, device=self.dev)
        # the legal rows of the chosen samples, scattered into the dense per-action arrays
        lo, hi = self.offs[ids], self.offs[ids + 1]
        sel = torch.from_numpy(np.concatenate([np.arange(s, e) for s, e in zip(lo, hi)])).to(self.dev)
        pos = torch.from_numpy(np.repeat(np.arange(b), hi - lo)).to(self.dev)
        a_idx = self.l_idx[sel]
        out["mask"][pos, a_idx] = True
        af = self.l_af[sel]
        w = min(af.shape[1], out["af"].shape[2])
        out["af"][pos, a_idx, :w] = af[:, :w].to(out["af"].dtype)
        out["aid"][pos, a_idx] = self.l_aid[sel].to(out["aid"].dtype)
        out["aid2"][pos, a_idx] = self.l_aid2[sel].to(out["aid2"].dtype)
        return out, self.act[idt]
