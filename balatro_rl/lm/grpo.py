"""GRPO for the student: group-relative policy optimisation with the game as the judge, one decision at a time.

Games are played as in lm/play.py (Yellow Gold, rounds by the honest money-maximising solver). At each decision outside a round the student
writes G answers to the same prompt (sampling, --temperature). Every distinct legal move among them is scored by rollouts: the game is copied
--futures times with the unknown future (draws, shops, packs, bosses) redrawn - the same redrawn futures for every candidate move - the move is
played, and each copy is finished by a fast continuation player (v51's network with the round solver) to the end of the run. A move's
reward = mean blinds beaten (+ --win-bonus for a win); an answer that does not parse or names no legal move gets the group's worst reward
minus 1. Advantage = (reward - group mean) / (group std + eps); groups where every answer chose the same move carry no signal and are dropped.

The update (every --batch-groups groups): loss = - mean over answers of advantage x the answer's mean token log-probability (one on-policy
step per batch: the GRPO objective with ratio 1), plus --kl x a KL penalty to the starting adapter estimated on the same tokens. Only the
LoRA weights train. The real game then plays the group's best move (its trajectory follows the improved policy).

Why rollouts and not whole games played by the student: a whole game is ~70 decisions, so a game-level score gives every decision the same
credit; the rollouts give each decision its own, and the shared futures remove most of the luck from the comparison.

    C:/venvs/balatro-lm/Scripts/python -m balatro_rl.lm.grpo --adapter checkpoints/lm/sft_v1 --out checkpoints/lm/grpo_v1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch

from .. import env as E
from ..engine_env import CppVecEnv, OBS_KEYS
from ..model import load_model
from . import render as R
from .play import BUILD_MOVES, default_move, make_rounds, solver_money_settings
from .sft import SHORT_SYSTEM


class Rollouts:
    """Finishes copies of a game with a fast player: v51's network outside rounds, the engine's round solver (foresight autoplay) inside."""

    def __init__(self, ckpt, deck, stake, size, threads, win_bonus, max_steps=6000, autoplay_nodes=400):
        self.net = load_model(ckpt, "cuda").eval()
        self.R = CppVecEnv(size, deck, stake, seed0=3, threads=threads)
        if getattr(self.net, "marg", 0):
            self.R.set_marginal(int(self.net.marg))
        self.R.set_autoplay(True, autoplay_nodes, True)
        self.R.v.restart()
        self.size, self.win_bonus, self.max_steps = size, win_bonus, max_steps
        self.games = 0

    @torch.no_grad()
    def values(self, V, jobs, salts):
        """jobs [(game in V, action)], salts [one per job] -> blinds beaten at the end (+ win bonus) per job."""
        out = np.zeros(len(jobs))
        R = self.R
        for c0 in range(0, len(jobs), self.size):
            ch = jobs[c0:c0 + self.size]
            m = len(ch)
            dst = list(range(m))
            R.copy_many(V, [g for g, _ in ch], dst)
            R.v.hide_future(dst, salts[c0:c0 + m])
            fin = np.ones(self.size, bool)
            fin[:m] = False
            act = np.full(self.size, -1, dtype=np.int64)
            act[:m] = [a for _, a in ch]
            val = np.zeros(self.size)
            _, d, info = R.step(act)
            for _ in range(self.max_steps):
                f = d & ~fin
                val[f] = info["blinds"][f] + self.win_bonus * info["won"][f]
                fin |= d
                if fin.all():
                    break
                live = np.flatnonzero(~fin)
                o = {k: torch.from_numpy(np.asarray(R.obs[k])[live]).cuda() for k in OBS_KEYS if k in R.obs}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = self.net(o)
                act = np.full(self.size, -1, dtype=np.int64)
                act[live] = lg.float().argmax(1).cpu().numpy()
                _, d, info = R.step(act)
            live = np.flatnonzero(~fin)                   # (cut at max_steps: what it reached)
            if len(live):
                val[live] = np.asarray(info["blinds"])[live]
            out[c0:c0 + m] = val[:m]
            self.games += m
        return out


def parse_move(text, n_legal):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        k = int(json.loads(m.group(0))["move"])
        return k if 0 <= k < n_legal else None
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="the starting LoRA adapter (lm/sft.py output)")
    p.add_argument("--out", default="checkpoints/lm/grpo_v1")
    p.add_argument("--games", type=int, default=8, help="games played side by side")
    p.add_argument("--seed0", type=int, default=7_300_000)
    p.add_argument("--deck", default="YELLOW")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--group", type=int, default=6, help="answers sampled per decision")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--futures", type=int, default=8, help="redrawn futures per candidate move")
    p.add_argument("--win-bonus", type=float, default=10.0)
    p.add_argument("--batch-groups", type=int, default=16, help="groups with signal per update")
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl", type=float, default=0.02)
    p.add_argument("--max-new", type=int, default=384)
    p.add_argument("--gen-chunk", type=int, default=2, help="prompts per generate call (x group answers)")
    p.add_argument("--continuation", default="checkpoints/ppo_gold_v51.pt")
    p.add_argument("--rollout-envs", type=int, default=1024)
    p.add_argument("--rounds", default="mc", choices=["mc", "vote"])
    p.add_argument("--round-futures", type=int, default=64, help="the round player's redrawn draw orders per candidate move")
    p.add_argument("--honest", type=int, default=5)
    p.add_argument("--node-cap", type=int, default=1000)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--save-every", type=int, default=10)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "grpo_log.jsonl"), "a", encoding="utf-8")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    meta = json.load(open(os.path.join(a.adapter, "lm_meta.json")))
    tok = AutoTokenizer.from_pretrained(a.adapter)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(meta["base"], torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, a.adapter, is_trainable=True)
    model.load_adapter(a.adapter, adapter_name="ref", is_trainable=False)        # the starting adapter, for the KL penalty
    model.set_adapter("default")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    system = SHORT_SYSTEM if meta.get("system", "short") == "short" else R.SYSTEM

    def prompt_text(user):
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True)

    @torch.no_grad()
    def sample(users):
        model.eval()
        texts = [prompt_text(u) for u in users]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=a.temperature, top_p=1.0,
                             num_return_sequences=a.group, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        txt = tok.batch_decode(gen, skip_special_tokens=True)
        return [txt[i * a.group:(i + 1) * a.group] for i in range(len(users))]

    def seq_logp(ptxt, ctxt, adapter):
        """mean token log-probability of the completion ctxt after the prompt ptxt, under the given adapter"""
        model.set_adapter(adapter)
        pi = tok(ptxt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ci = tok(ctxt + tok.eos_token, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.cat([pi, ci], 1).cuda()
        logits = model(input_ids=ids).logits[:, pi.shape[1] - 1:-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[:, pi.shape[1]:, None])[..., 0]
        return lp.mean(), lp

    def update(groups):
        model.train()
        opt.zero_grad(set_to_none=True)
        n = sum(len(g["answers"]) for g in groups)
        tot, kl_tot = 0.0, 0.0
        for g in groups:
            ptxt = prompt_text(g["user"])
            for ans, adv in zip(g["answers"], g["adv"]):
                with torch.no_grad():
                    _, lp_ref = seq_logp(ptxt, ans, "ref")
                mlp, lp = seq_logp(ptxt, ans, "default")
                kl = (torch.exp(lp_ref - lp) - (lp_ref - lp) - 1).mean()          # the k3 estimator of KL(policy || start)
                loss = (-adv * mlp + a.kl * kl) / n
                loss.backward()
                tot += float(loss) * n
                kl_tot += float(kl)
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
        opt.step()
        model.set_adapter("default")
        return tot / n, kl_tot / n

    # ---- the games
    V = CppVecEnv(a.games, a.deck, a.stake, seed0=a.seed0, threads=a.threads)
    solver_money_settings(V)
    honest = make_rounds(V, a)
    honest.H.set_auto_target(True)
    S = CppVecEnv(64, a.deck, a.stake, seed0=1, threads=a.threads)
    S.set_auto_target(True)
    S.v.restart()
    V.v.restart()
    ro = Rollouts(a.continuation, a.deck, a.stake, a.rollout_envs, a.threads, a.win_bonus)
    plan = [""] * a.games
    salt = 1
    buf, upd, t0 = [], 0, time.time()
    stats = {"groups": 0, "signal": 0, "bad": 0, "answers": 0, "games": 0, "blinds": []}

    def fp(env, ids):
        env.v.set_curve_crn(12345, False)
        f = env.fast_power(ids, 256)
        env.v.set_curve_crn(0, False)
        return f

    while upd < a.updates:
        obs = V.obs
        g = np.asarray(obs["g"])
        mask = np.asarray(obs["mask"])
        act = np.full(a.games, -1, dtype=np.int64)
        inr = g[:, 5] > 0.5
        ids = np.flatnonzero(inr)
        if len(ids):
            act[ids] = honest.moves(V, ids, mask, np.asarray(obs["af"], dtype=np.float32))
        ask = []
        for i in np.flatnonzero(~inr):
            legal = [int(x) for x in np.flatnonzero(mask[i, E.A_SELECT:]) + E.A_SELECT]
            if not legal:
                act[i] = int(np.flatnonzero(mask[i])[0])
            elif len(legal) == 1:
                act[i] = legal[0]
            else:
                ask.append((int(i), legal))
        if ask:
            fps = fp(V, [i for i, _ in ask])
            users = []
            for (i, legal), f in zip(ask, fps):
                d = V.v.dump(i)
                users.append((i, legal, d, R.user_message(d, legal, plan[i], f)))
            for c0 in range(0, len(users), a.gen_chunk):
                chunk = users[c0:c0 + a.gen_chunk]
                outs = sample([u for _, _, _, u in chunk])
                for (i, legal, d, user), answers in zip(chunk, outs):
                    moves = [parse_move(t, len(legal)) for t in answers]
                    cands = sorted({legal[k] for k in moves if k is not None})
                    stats["answers"] += len(answers)
                    stats["bad"] += sum(1 for k in moves if k is None)
                    stats["groups"] += 1
                    if not cands:
                        act[i] = default_move(legal)
                        continue
                    if len(cands) == 1 and all(k is not None for k in moves):
                        act[i] = cands[0]                             # (every answer chose the same move: no signal)
                        best_text = answers[0]
                    else:
                        jobs, salts = [], []
                        for c in cands:
                            for f_ in range(a.futures):
                                jobs.append((i, c))
                                salts.append(salt * 1_000_003 + 7919 * f_ + 13)
                        salt += 1
                        vals = ro.values(V, jobs, salts).reshape(len(cands), a.futures).mean(1)
                        val_of = dict(zip(cands, vals))
                        worst = float(min(vals)) - 1.0
                        rew = np.array([val_of[legal[k]] if k is not None else worst for k in moves])
                        if rew.std() > 1e-6:
                            adv = (rew - rew.mean()) / (rew.std() + 1e-4)
                            buf.append({"user": user, "answers": answers, "adv": adv.tolist()})
                            stats["signal"] += 1
                        top = int(np.argmax(rew))
                        act[i] = legal[moves[top]] if moves[top] is not None else default_move(legal)
                        best_text = answers[top]
                    try:
                        plan[i] = str(json.loads(re.search(r"\{.*\}", best_text, re.S).group(0)).get("plan", plan[i]))[:1500]
                    except Exception:
                        pass
        if (act < 0).all():
            continue
        _, done, info = V.step(act)
        for i in np.flatnonzero(done):                          # a finished game (the engine starts the next seed in its place)
            stats["games"] += 1
            stats["blinds"].append(float(info["blinds"][i]))
            plan[i] = ""
        if len(buf) >= a.batch_groups:
            loss, kl = update(buf[:a.batch_groups])
            buf = buf[a.batch_groups:]
            upd += 1
            row = {"update": upd, "loss": round(loss, 4), "kl": round(kl, 4), "groups": stats["groups"], "signal": stats["signal"],
                   "bad_rate": round(stats["bad"] / max(1, stats["answers"]), 3), "games": stats["games"],
                   "blinds_recent": round(float(np.mean(stats["blinds"][-20:])), 2) if stats["blinds"] else None,
                   "rollout_games": ro.games, "min": round((time.time() - t0) / 60, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if upd % a.save_every == 0:
                model.save_pretrained(a.out, selected_adapters=["default"])
                tok.save_pretrained(a.out)
                json.dump({**meta, "grpo_from": a.adapter, "updates": upd}, open(os.path.join(a.out, "lm_meta.json"), "w"))
    model.save_pretrained(a.out, selected_adapters=["default"])
    tok.save_pretrained(a.out)
    json.dump({**meta, "grpo_from": a.adapter, "updates": upd}, open(os.path.join(a.out, "lm_meta.json"), "w"))


if __name__ == "__main__":
    main()
