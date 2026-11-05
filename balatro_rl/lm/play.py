"""Plays Gold stake games where a language model makes every decision outside a round, and logs each decision as a training example.

Rounds: --rounds mc (default): lm/rounds.MCRounds, each round move chosen by its clear rate over --futures redrawn draw orders (honest:
it never sees the real draws); --rounds vote: the older tools/gap_study.Honest (K copies with the future hidden are solved, their first moves vote),
with the solver's money settings on (smart move order, the hands-left bound that keeps looking for the richest clearing line, the full
node budget, held tarots on their best targets). Nothing in a round sees the real draw order.

Decisions: every state outside a round with more than one legal move goes to the model (render.SYSTEM + render.user_message). Games run
side by side; a game waiting for its answer is simply not stepped (action -1), the others go on.

Backends:
  claude   the Claude Code command line (`claude -p`) on the signed-in Claude account; no API key is read here
  student  a local fine-tuned model (lm/student.py), batched

Log (--log, JSON lines): one row per decision {"kind": "decision", game, seed, ante, blind, state, money, user, answer, move_index, action,
error, ms}, one row per finished game {"kind": "result", game, seed, blinds, won, ante}. lm/sft.py turns these into training data.

    python -m balatro_rl.lm.play --dry-run                                    (print one ante-2 shop prompt, no model call)
    python -m balatro_rl.lm.play --games 2 --parallel 2 --log traces/lm/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import env as E
from ..engine_env import CppVecEnv
from ..llm_strategist import cons_key, describe_action, name_of, text_of
from .rounds import GROW_STEP, SUB_INDEX

# the jokers for which the style of a round matters (a counter that grows with play, cards to discard, a hand type to play); a round start
# with none of them and no usable consumable is not asked about
STYLE_JOKERS = set(GROW_STEP) | {"hit_the_road", "castle", "mail", "faceless", "burnt", "supernova", "card_sharp", "todo_list", "obelisk",
                                 "space", "seance", "superposition", "trading", "sixth_sense", "8_ball", "trousers", "dna", "vagabond"}
from . import render as R

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from gap_study import Honest                                              # noqa: E402

BUILD_MOVES = set(range(E.A_BUY, E.A_BUY_PACK)) | set(range(E.A_SELL_J, E.A_SELL_C)) | set(range(E.A_USE_C, E.A_PICK)) | \
              set(range(E.A_PICK, E.A_PSKIP)) | set(range(E.A_REPLACE, E.N_ACTIONS))


def solver_money_settings(v, targets=3):
    """The round solver's settings (engine globals) that find the richest clearing line."""
    v.v.set_solver_tarots(True, targets)
    v.v.set_solver_order(True)
    v.v.set_solver_speed(True, False, 0.0)
    v.v.set_solver_tt(True)
    v.v.set_solver_hand_bound(True)
    v.v.set_solver_extra(-1)
    v.set_auto_target(True)                           # a tarot that needs target cards is applied at once on the engine's targets (Game::auto_targets)


def make_rounds(v, a):
    """The honest round player (--rounds): MCRounds (each move by its clear rate over redrawn draw orders) or the older voting Honest."""
    if getattr(a, "rounds", "mc") == "mc":
        from .rounds import MCRounds
        return MCRounds(v, a.deck, a.stake, size=4096, futures=getattr(a, "round_futures", getattr(a, "futures", 64)), threads=a.threads)
    return Honest(v, a.honest, a.node_cap, getattr(a, "scratch", 1024), a.deck, a.stake, a.threads)


def default_move(legal):
    for a in (E.A_LEAVE, E.A_SELECT, E.A_PSKIP):
        if a in legal:
            return a
    return int(legal[0])


class ClaudeCLI:
    """One decision per `claude -p` call: our system prompt replaces Claude Code's, no tools, no settings or MCP servers, a JSON answer."""

    def __init__(self, model, effort, timeout=900):
        self.model, self.effort, self.timeout = model, effort, timeout
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        f.write(R.SYSTEM)
        f.close()
        self.sys_file = f.name
        self.lock = threading.Lock()
        self.usage = {"calls": 0, "input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "cost_equiv": 0.0}
        self.fatal = None

    def refresh_system(self, text):
        """a new system prompt (new lessons) for the calls from now on"""
        with open(self.sys_file, "w", encoding="utf-8") as f:
            f.write(text)

    def ask(self, user, schema=None):
        cmd = ["claude", "-p", "--system-prompt-file", self.sys_file, "--tools", "", "--safe-mode", "--strict-mcp-config",
               "--no-session-persistence", "--model", self.model, "--effort", self.effort, "--output-format", "json",
               "--json-schema", json.dumps(schema or R.SCHEMA)]
        delay = 30
        while True:                                   # a usage limit or an overloaded server: wait and ask again (never a default move)
            out, err, retry = self._ask_once(cmd, user)
            if not retry:
                return out, err
            with self.lock:
                self.waiting = f"{err} - retrying in {delay}s"
            time.sleep(delay)
            delay = min(delay * 2, 900)

    def _ask_once(self, cmd, user):
        """-> (answer, error, retry)"""
        try:
            # a clean environment: launched from inside a Claude Code session, the child would inherit that session's variables and try to
            # refresh its sign-in through the host app; without them it uses the saved login (~/.claude) like a terminal does
            env = {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE") or k == "CLAUDECODE")}
            p = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8", timeout=self.timeout, env=env)
            r = json.loads(p.stdout)
        except Exception as ex:
            return None, f"cli failed: {ex.__class__.__name__}: {str(ex)[:200]}", True
        u = r.get("usage") or {}
        with self.lock:
            self.usage["calls"] += 1
            self.usage["input"] += u.get("input_tokens", 0)
            self.usage["cache_read"] += u.get("cache_read_input_tokens", 0)
            self.usage["cache_write"] += u.get("cache_creation_input_tokens", 0)
            self.usage["output"] += u.get("output_tokens", 0)
            self.usage["cost_equiv"] += float(r.get("total_cost_usd") or 0)
        res = str(r.get("result") or "")
        if r.get("terminal_reason") == "api_error" and ("authenticate" in res or "OAuth" in res or "login" in res.lower()):
            self.fatal = "the claude command is not signed in: " + res[:200]
            return None, self.fatal, False
        if r.get("is_error") or r.get("terminal_reason") == "api_error":
            return None, "api error: " + res[:200], True
        out = r.get("structured_output")
        if isinstance(out, dict) and "move" in out:
            return out, None, False
        try:
            return json.loads(r.get("result") or ""), None, False
        except Exception:
            return None, f"bad answer: {str(r.get('result'))[:200]}", False


class Session:
    def __init__(self, a):
        self.a = a
        self.v = CppVecEnv(a.games, a.deck, a.stake, seed0=a.seed0, threads=a.threads)
        solver_money_settings(self.v)
        self.v.set_auto_target(False)                 # tarots act on the cards the model names (resolve_targets); copies keep auto targets
        self.honest = make_rounds(self.v, a)
        self.honest.H.set_auto_target(True)
        from .rounds import ClearSim
        self.cs = ClearSim(a.deck, a.stake, size=2048, threads=a.threads)       # the coming blinds played out, for the estimates
        self.S = CppVecEnv(64, a.deck, a.stake, seed0=1, threads=a.threads)     # scratch copies
        self.S.set_auto_target(True)
        self.S.v.restart()
        self.v.v.restart()
        self.log = open(a.log, "a", encoding="utf-8") if a.log else None
        self.lessons_v = R.lessons_version()
        self.next_review, self.review_job = getattr(a, "lessons_every", 0) or 0, None
        self.lock = threading.Lock()

    def fp(self, ids):
        if not len(ids):
            return np.zeros((0, 4))
        self.v.v.set_curve_crn(12345, False)
        out = self.v.fast_power(ids, self.a.fp_samples)
        self.v.v.set_curve_crn(0, False)
        return out

    def sims(self, items):
        """items: [(game, legal)] -> per game: (p_next, p_boss) as it stands, and {move: note} for the build-changing moves that change
        the next blind's chance by 5 points or the boss chance by 15 points or more (each move played on the copies first; the same simulated futures for every move)."""
        jobs, owners, base = [], [], []
        for i, legal in items:
            d = self.v.v.dump(i)
            b0, a0 = int(d["furthest_blind"]), int(d["ante"])
            moves = [int(x) for x in legal if int(x) in BUILD_MOVES] if self.a.move_notes else []
            owners.append((i, moves, len(jobs), len(base)))
            base.append((i, None, b0, a0))
            jobs += [(i, m, b0, a0) for m in moves]
        res0 = self.cs.run(self.v, base, futures=self.a.sim_futures, antes_ahead=1)       # the state as it is: through the next ante
        res = self.cs.run(self.v, jobs, futures=self.a.sim_futures) if jobs else []      # each move: through this ante's boss
        out = {}
        for i, moves, j0, k0 in owners:
            p0 = res0[k0]
            notes = {}
            for k, m in enumerate(moves):
                p1 = res[j0 + k]
                if abs(p1[0] - p0[0]) >= 0.05 or abs(p1[1] - p0[1]) >= 0.15:          # (the boss chance is noisier: about +-8 points)
                    notes[m] = f"  [clear next blind {100 * p0[0]:.0f}% -> {100 * p1[0]:.0f}%; through this ante's boss {100 * p0[1]:.0f}% -> {100 * p1[1]:.0f}%]"
            out[i] = (p0, notes)
        return out

    def write(self, row):
        if self.log:
            with self.lock:
                self.log.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.log.flush()

    def resolve_targets(self, i, mask_i, want):
        """A targeting step: the play slot whose cards are the ones the model named (0-based positions), else the largest legal slot
        inside them, else the first legal slot. -> (action, exact)"""
        legal = [j for j in np.flatnonzero(mask_i[E.A_PLAY:E.A_PLAY + E.N_SUB])]
        if not legal:
            return int(np.flatnonzero(mask_i)[0]), False
        want = tuple(sorted(set(int(x) for x in (want or []))))
        j = SUB_INDEX.get(want, -1)
        if j >= 0 and j in set(legal):
            return E.A_PLAY + j, True
        inside = [j for j in legal if set(E.SUBSETS[j]) <= set(want)] if want else []
        if inside:
            return E.A_PLAY + max(inside, key=lambda j: len(E.SUBSETS[j])), False
        return E.A_PLAY + int(legal[0]), False

    def round_options(self, i, d, mask_i, af_i):
        """-> options [(action or None, text)], estimate of playing on, [(name, estimate)] of each consumable use"""
        uses = [E.A_USE_C + k for k in range(len(d["consumables"])) if E.A_USE_C + k < len(mask_i) and mask_i[E.A_USE_C + k]]
        opts = [(None, "Play on: the round player makes the next play or discard (following your style)")]
        names = []
        for a_ in uses:
            k = cons_key(d["consumables"][a_ - E.A_USE_C])
            opts.append((a_, f"Use {name_of(k)} now ({text_of(k)})"))
            names.append(name_of(k))
        est_c, est_u = None, []
        if hasattr(self.honest, "last_values"):
            st = self.honest.styles.get(i) or {}
            base = self.honest.candidates(mask_i, af_i, d["hand"], {**st, "consumables": False},
                                          d["jokers"][st["grow"]]["key"] if st.get("grow") is not None and st["grow"] < len(d["jokers"]) else None)
            self.honest.moves(self.v, [i], np.asarray(self.v.obs["mask"]), np.asarray(self.v.obs["af"], dtype=np.float32),
                              consider={i: base + uses})
            lv = self.honest.last_values.get(i, {})
            vals = [lv[c] for c in base if c in lv]
            est_c = min(1.0, max(vals)) if vals else None
            est_u = [(nm, min(1.0, lv.get(a_, 0.0))) for nm, a_ in zip(names, uses)]
        return opts, est_c, est_u

    def run(self, backend):
        a, v = self.a, self.v
        n = a.games
        fin = np.zeros(n, bool)
        plan = [""] * n
        pending = {}                                  # game -> dict(fut, kind, legal / options, dump, user, t0)
        res = [None] * n
        pool = ThreadPoolExecutor(max(1, a.parallel)) if backend is not None else None
        t_start = time.time()
        steps = 0
        mc = hasattr(self.honest, "styles")
        ask_rounds = backend is not None and mc and a.round_decisions
        prev_in = np.zeros(n, bool)
        round_asked, skip_ask, expect_target = set(), set(), {}
        while not fin.all():
            if getattr(backend, "fatal", None):
                raise SystemExit("stopped: " + backend.fatal)
            steps += 1
            obs = v.obs
            g = np.asarray(obs["g"])
            mask = np.asarray(obs["mask"])
            af = np.asarray(obs["af"], dtype=np.float32)
            act = np.full(n, -1, dtype=np.int64)
            inround = (g[:, 5] > 0.5) & ~fin
            for i in np.flatnonzero(prev_in & ~inround):            # a round ended: its style and questions with it
                if mc:
                    self.honest.styles.pop(int(i), None)
                round_asked.discard(int(i))
            prev_in = inround.copy()
            # targeting steps after a tarot use or a pack pick: the cards the model named
            for i in list(expect_target):
                if i in pending or fin[i]:
                    continue
                want = expect_target.pop(i)
                d = v.v.dump(i)
                if d["targeting"]:
                    act[i], exact = self.resolve_targets(i, mask[i], want)
                    if not exact:
                        self.write({"kind": "target_fallback", "game": int(i), "want": list(want or []), "action": int(act[i])})
            # finished answers
            for i in list(pending):
                pd = pending[i]
                if not pd["fut"].done():
                    continue
                del pending[i]
                out, err = pd["fut"].result()
                d = pd["dump"]
                k = None
                if out is not None:
                    try:
                        k = int(out["move"])
                        k = k if 0 <= k < len(pd["choices"]) else None
                    except Exception as ex:
                        err = f"bad answer: {ex}"
                tg = []
                if out is not None and isinstance(out.get("targets"), list):
                    tg = [int(x) - 1 for x in out["targets"] if isinstance(x, (int, float))]
                if pd["kind"] == "round":
                    st = R.parse_style((out or {}).get("style"), d)
                    self.honest.styles[i] = {**st, "consumables": False}
                    choice = pd["choices"][k][0] if k is not None else None
                    if choice is None:
                        skip_ask.add(i)                      # play on: the round player moves next
                    else:
                        act[i] = choice
                        expect_target[i] = tg
                    move_txt = pd["choices"][k][1] if k is not None else "play on (no valid answer)"
                    err = err if k is not None or err else "move out of range"
                else:
                    legal = pd["choices"]
                    if k is None:
                        act[i] = default_move(legal)
                        err = err or "move out of range"
                    else:
                        act[i] = int(legal[k])
                        plan[i] = str(out.get("plan", "") or "")[:1500] or plan[i]
                        if E.A_USE_C <= act[i] < E.A_PICK or E.A_PICK <= act[i] < E.A_PSKIP:
                            expect_target[i] = tg
                    move_txt = describe_action(int(act[i]), d)
                self.write({"kind": "decision", "game": int(i), "seed": a.seed0 + int(i), "ante": d["ante"], "blind": d["blind_idx"],
                            "state": "ROUND" if pd["kind"] == "round" else d["state"], "money": d["money"],
                            "system": f"{R.PLAYBOOK_ID}+lessons_v{self.lessons_v}",
                            "user": pd["user"], "answer": out, "move_index": k, "action": int(act[i]), "move": move_txt, "error": err,
                            "ms": int(1000 * (time.time() - pd["t0"]))})
            # rounds: ask the model at a round's start and whenever a consumable could be used; otherwise the round player moves
            rids = []
            for i in np.flatnonzero(inround):
                i = int(i)
                if i in pending or act[i] >= 0:
                    continue
                if ask_rounds:
                    self.honest.styles.setdefault(i, {"consumables": False})
                    if i not in skip_ask:
                        uses = any(mask[i, E.A_USE_C + k] for k in range(3) if E.A_USE_C + k < mask.shape[1])
                        if uses or i not in round_asked:
                            d = v.v.dump(i)
                            if d["targeting"]:
                                rids.append(i)
                                continue
                            if uses or any(j["key"] in STYLE_JOKERS for j in d["jokers"]):
                                round_asked.add(i)
                                opts, est_c, est_u = self.round_options(i, d, mask[i], af[i])
                                user = R.round_message(d, opts, plan[i], self.honest.styles.get(i), est_c, est_u)
                                pending[i] = {"fut": pool.submit(backend.ask, user, R.ROUND_SCHEMA), "kind": "round", "choices": opts,
                                              "dump": d, "user": user, "t0": time.time()}
                                continue
                            round_asked.add(i)
                rids.append(i)
            if rids:
                act[rids] = self.honest.moves(v, rids, mask, af)
            # new decisions outside rounds
            outside = [int(i) for i in np.flatnonzero(~inround & ~fin) if i not in pending and act[i] < 0]
            ask = []
            for i in outside:
                legal = [int(x) for x in np.flatnonzero(mask[i, E.A_SELECT:]) + E.A_SELECT]
                if not legal:                                                   # (no move outside a round: whatever the engine allows)
                    act[i] = int(np.flatnonzero(mask[i])[0])
                elif len(legal) == 1:
                    act[i] = legal[0]
                else:
                    ask.append((i, legal))
            if ask:
                fps = self.fp([i for i, _ in ask])
                sims = self.sims(ask)
                for (i, legal), f in zip(ask, fps):
                    d = v.v.dump(i)
                    sim, notes = sims[i]
                    user = R.user_message(d, legal, plan[i], f, notes, sim)
                    if backend is None:                                         # dry run: print an ante-2 shop prompt
                        if d["state"] == "SHOP" and d["ante"] >= a.dry_ante:
                            print("=" * 100 + f"\nSYSTEM PROMPT ({len(R.SYSTEM)} characters) - see balatro_rl/lm/render.py\n" + "=" * 100)
                            print(user)
                            return None
                        act[i] = default_move(legal) if E.A_LEAVE in legal or d["state"] != "SHOP" else int(legal[0])
                        if d["state"] == "SHOP" and E.A_LEAVE in legal:                 # (buy something cheap now and then, to have a build)
                            buys = [x for x in legal if E.A_BUY <= x < E.A_BUY_PACK]
                            act[i] = buys[0] if buys and d["money"] > 8 else E.A_LEAVE
                        continue
                    pending[i] = {"fut": pool.submit(backend.ask, user, R.SCHEMA), "kind": "outside", "choices": legal, "dump": d,
                                  "user": user, "t0": time.time()}
            if not (act >= 0).any():
                time.sleep(0.2)
                continue
            _, done, info = v.step(act)
            for i in np.flatnonzero(act >= 0):
                skip_ask.discard(int(i))
            for i in np.flatnonzero(done & ~fin):
                res[i] = {"blinds": float(info["blinds"][i]), "won": bool(info["won"][i]), "ante": int(info["ante"][i])}
                self.write({"kind": "result", "game": int(i), "seed": a.seed0 + int(i), "deck": a.deck, "stake": a.stake, **res[i]})
                print(json.dumps({"game": int(i), **res[i], "min": round((time.time() - t_start) / 60, 1)}), flush=True)
            fin |= done
            if a.lessons_every and backend is not None and hasattr(backend, "refresh_system"):
                n_done = int(fin.sum())
                if n_done >= self.next_review and (self.review_job is None or self.review_job.done()):
                    self.next_review = n_done + a.lessons_every
                    self.review_job = pool.submit(self.learn, backend)
        if self.review_job is not None:
            self.review_job.result()
        if pool:
            pool.shutdown()
        return res

    def learn(self, backend):
        """review the finished games, merge the lessons, give the new system prompt to the calls from now on"""
        from . import lessons as LS
        try:
            news = LS.review_log(self.a.log)
            backend.refresh_system(R.system_prompt())
            self.lessons_v = R.lessons_version()
            self.write({"kind": "lessons", "version": self.lessons_v, "reviewed": [r["game"] for r in news],
                        "summaries": [r["summary"] for r in news], "text": R.lessons_text()})
        except Exception as ex:
            self.write({"kind": "lessons_error", "error": f"{ex.__class__.__name__}: {str(ex)[:300]}"})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=2)
    p.add_argument("--seed0", type=int, default=7_000_000)
    p.add_argument("--deck", default="YELLOW")
    p.add_argument("--stake", default="GOLD")
    p.add_argument("--backend", default="claude", choices=["claude", "student"])
    p.add_argument("--adapter", default="", help="--backend student: the LoRA adapter folder (lm/sft.py output)")
    p.add_argument("--batch", type=int, default=8, help="--backend student: requests per generate call")
    p.add_argument("--temperature", type=float, default=0.0, help="--backend student: 0 = greedy")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--parallel", type=int, default=4, help="model calls at once")
    p.add_argument("--rounds", default="mc", choices=["mc", "vote"], help="the round player (see the top of this file)")
    p.add_argument("--futures", type=int, default=64, help="--rounds mc: redrawn draw orders per candidate move")
    p.add_argument("--honest", type=int, default=5, help="--rounds vote: hidden-future copies voting on each round move")
    p.add_argument("--node-cap", type=int, default=1000, help="round solver nodes per copy")
    p.add_argument("--scratch", type=int, default=1024)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--fp-samples", type=int, default=256, help="sampled hands for the survival estimates")
    p.add_argument("--round-decisions", type=int, default=1, help="1: the model decides consumable uses and the style of play in rounds")
    p.add_argument("--lessons-every", type=int, default=0, help="> 0: after every this many finished games, review them and update the lessons")
    p.add_argument("--sim-futures", type=int, default=64, help="simulated futures for the clear-chance estimates")
    p.add_argument("--move-notes", type=int, default=1, help="1: show each build-changing move's change in the clear chance")
    p.add_argument("--log", default="traces/lm/teacher.jsonl")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-ante", type=int, default=2)
    a = p.parse_args()
    if a.log and not a.dry_run:
        os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    else:
        a.log = ""
    s = Session(a)
    if a.dry_run:
        backend = None
    elif a.backend == "student":
        from .student import Student
        backend = Student(a.adapter or None, batch=a.batch, temperature=a.temperature)
        a.parallel = max(a.parallel, a.batch)
    else:
        backend = ClaudeCLI(a.model, a.effort)
    t = time.time()
    res = s.run(backend)
    if res is None:
        return
    bl = np.array([r["blinds"] for r in res])
    print(json.dumps({"games": len(res), "blinds": round(float(bl.mean()), 2), "wins": int(sum(r["won"] for r in res)),
                      "minutes": round((time.time() - t) / 60, 1), "model": a.model, "effort": a.effort, "usage": backend.usage}), flush=True)


if __name__ == "__main__":
    main()
