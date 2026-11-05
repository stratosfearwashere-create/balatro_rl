"""Grow a network without changing what it computes: more self-attention layers over the jokers / hand / consumables, the new layers starting
as the identity (their two output projections are zero, and the layers are residual), so the grown network plays exactly as the old one and
training goes on from there with more capacity.

    python tools/grow_model.py checkpoints/azc_cloud.pt checkpoints/azc_big_start.pt --attn 4
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from balatro_rl.model import ActorCritic, save_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--attn", type=int, default=4, help="self-attention layers of the grown network")
    a = p.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    old = int(cfg.get("attn", 0))
    if old == 0 or a.attn <= old:
        raise SystemExit(f"the network has {old} attention layers: growing needs an attention network and more layers than it has")
    small = ActorCritic(**cfg)
    small.load_state_dict(ck["state_dict"], strict=False)
    cfg["attn"] = a.attn
    big = ActorCritic(**cfg)
    missing = big.load_state_dict(ck["state_dict"], strict=False)
    new = [f"mix.{i}." for i in range(old, a.attn)]
    bad = [k for k in missing.missing_keys if not any(k.startswith(n) for n in new) and not k.startswith("replace.")] + list(missing.unexpected_keys)
    if bad:
        raise SystemExit(f"unexpected weights: {bad}")
    with torch.no_grad():
        for i in range(old, a.attn):
            blk = big.mix[i]
            blk.out.weight.zero_(); blk.out.bias.zero_()
            blk.ff[2].weight.zero_(); blk.ff[2].bias.zero_()
    # check: the same outputs on a real batch
    from balatro_rl.engine_env import CppVecEnv
    from balatro_rl.azc import KEYS, obs_batch
    v = CppVecEnv(64, "RED", "GOLD", seed0=3, threads=0)
    small.eval(); big.eval()
    with torch.no_grad():
        o = obs_batch(v, list(range(64)), "cpu")
        (l1, v1), (l2, v2) = small(o), big(o)
    dl = (l1 - l2).abs().nan_to_num(0).max().item()
    dv = (v1 - v2).abs().max().item()
    print(f"grown {old} -> {a.attn} attention layers: {sum(t.numel() for t in small.parameters()):,} -> {sum(t.numel() for t in big.parameters()):,} weights; "
          f"largest output difference: logits {dl:.2e}, value {dv:.2e}")
    if dl > 1e-4 or dv > 1e-4:
        raise SystemExit("the grown network does not match")
    save_model(big, a.dst, dict(ck.get("extra", {}), grown_from=a.src))
    print("saved", a.dst)


if __name__ == "__main__":
    main()
