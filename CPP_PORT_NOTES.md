# C++ core: notes

The C++ core lives in `balatro_rl/sim/cpp/` and builds into `balatro_rl.sim._core` (pybind11) with
`python setup_cython.py build`, after the Cython extensions. `BALATRO_PURE=1` disables it like every other
compiled extension; the Python code stays the reference, and `tests/test_cpp_core.py` plus the differential
test (`tests/test_differential.py`) hold the two to the same results.

## What is ported

| piece | file | status |
|---|---|---|
| `random.Random` replica (MT19937 and CPython's `random`, `getrandbits`, `_randbelow`, `randrange`, `randint`, `choice`, `shuffle`, `sample`, `choices`, `uniform`; `getstate` / `setstate` round trip) | `cpp/pyrandom.hpp`, `_core.Random` | done, bit for bit (test_cpp_core) |
| playing card access (zero copy, through the Cython card's struct) | `cpp/card.hpp` | done |
| `hands.evaluate` (1-5 cards; the Python version keeps the general case) | `cpp/hands.hpp`, `_core.evaluate` | done, 15x faster than Python |
| round solver playouts (`az/solver.py` RoundSolver) | `cpp/solver.hpp`, `_core.SolverCore`, `az/solver.py` CoreRoundSolver | done, bit for bit on the regression states |
| scoring + jokers (`scoring.score_hand`) | - | not ported: `Game.predict` already uses the compiled scorer (`_fastscore.pyx`); the Python scorer only runs for real plays and `analyze_play` |
| Game (rounds, shop, packs, consumables, bosses, tags, vouchers) | - | not ported (see "Data model" below) |

## Design decisions

**The card stays the Cython card.** `sim/_cards.pyx` is declared a Cython `api` class, so the generated
`_cards.h` gives C++ the object layout (`struct CardObject`) and C++ code reads rank, suit, enhancement,
edition, seal, extra chips, debuff, hidden and uid straight from the object (`CardView`). Nothing is copied,
`_clone.pyx` and `_solver.pyx` keep working, and Python code sees the same objects.

**The compiled scorer is called through its C API.** `_fastscore.pyx` exports `fs_best_two`, `fs_score_one`
and `fs_subset` (`cdef api`), reachable from C++ through `_fastscore_api.h` with no Python objects in the
call. The solver core scores every hand of every playout this way.

**The solver core replicates the agent's generator.** `CoreRoundSolver.solve` loads the agent's
`random.Random` state into the C++ replica, runs the futures and playouts in C++ (every draw in the order of
the Python code: the futures' shuffles, The Hook's `sample`, the inner re-shuffles) and writes the state
back. Scores are doubles in C++ and Python ints in the Python solver: identical below 2^53 chips, which no
real game reaches (a blind at ante 8 is about 10^5 on White Stake). The hidden-card path scores with the
hand types / Mouth hand "as the last `_preds` left them in the game", as the Python solver does, so the two
agree even there. A simulated hand past 16 cards (The Serpent) falls back to the Cython solver with the
generator restored to its state before the solve.

**Random, CPython-exact.** `seed(int)` is `init_by_array` over the absolute value's 32-bit words;
`random()` is the 53-bit construction; `getrandbits(k)` fills 32-bit words little-endian; `_randbelow`
rejects by `bit_length`; `shuffle` runs `i` from `len - 1` down to 1; `sample` uses CPython's set / pool
switch (`setsize = 21 + 4 ** ceil(log(3k, 4))` for `k > 5`); `choices` without weights takes
`floor(random() * n)`, with weights `bisect` on the cumulative weights with `hi = n - 1`.

## Data model: why the game itself is not ported

Everything outside `sim/` (az/, rewards/, env.py, heuristic.py, the tests) reads and *writes* the game's
state as plain Python objects: `g.hand = [...]`, `g.jokers = rest[:pos] + [new] + rest[pos:]`,
`g.consumables.append(...)`, `g.vouchers.add(...)`, `j.state["val"] = ...`, `g.rng = random.Random(0)`,
`vars(g)` (the differential snapshot), identity-keyed dicts (`{id(c): i for c in pool}`) and Card equality
by all fields. A C++ game with a C++ data model (vectors of structs, a memcpy clone) would have to expose
every list, set and dict attribute through proxies that behave like the Python containers (slicing,
concatenation, in-place mutation, identity of elements, dataclass equality), keep Python's arbitrary
precision for chips (scores past 2^63 occur in the fuzz tests), keep int / float distinctions in joker
state, effects and money, and replicate every random draw of 1400 lines of game logic and 150 joker hooks.
That is the whole of the remaining work and it is the risky part: a C++ game that cannot run az/
unchanged is worth nothing. A C++ game whose attributes are the same Python containers (the "port the
methods, keep the data" variant) is possible with the card-struct access used here, but it leaves the cost
that dominates the glue today (cloning Python objects: `Game.clone`, `determinize`, `fresh_round`) where it
is.

The pieces that were ported are the ones where the data already is, or can cheaply be made, plain C data:
the solver (card indices into a pool, the compiled scorer's tables, the generator's state).

## Where the time goes now (no-search graded bench, this container)

See SPEED_NOTES.md for the numbers. After stages 2-4 the pricer's Python glue (clones, `apply`, the
pricing logic, numpy bootstraps), candidate enumeration (`az/actions.py`) and the network are what is
left; the compiled scorer and the solver core are a small share.

## Things noticed in the Python simulator (not changed)

- `az/_solver.pyx` / `solver.py`: with The Eye or The Mouth and face-down cards, the hidden-card score of a
  future reads the hand types the *last* `_preds` call left in the solver's game copy (whatever playout
  ran before), not the future's own. The core reproduces this.
- `_fastscore.Scorer.first_hand` (DNA) is fixed for the whole solve, so a DNA copy is predicted on every
  playout hand of a round whose first hand is still to come. Same in both solvers.
