"""Hyperparameters for tabular preflop 6-max CFR."""

import os

# ── Game constants (must match deep_cfr/config.py and engine/game.py) ─────────
N_PLAYERS     = 6
INITIAL_STACK = 10_000
SMALL_BLIND   = 50
BIG_BLIND     = 100

# ── Abstract action indices (same as deep_cfr/config.py) ──────────────────────
FOLD          = 0
CHECK_CALL    = 1
BET_0_27X_POT = 2
BET_THIRD_POT = 3
BET_HALF_POT  = 4
BET_FULL_POT  = 5
BET_1_72X_POT = 6
BET_2X_POT    = 7
ALL_IN        = 8
N_ACTIONS     = 9

# ── Preflop-active subset ──────────────────────────────────────────────────────
# Raise sizes that make sense preflop. Pot-relative formulas match _abstract_to_raw
# in bot.py so chip amounts are identical at solve-time and inference-time.
#
# Deliberately coarse: FOLD / CHECK_CALL / one pot-sized raise / ALL_IN.  A
# full-pot preflop raise lands at a natural ~3.5bb open (pot 150 + owed 100 →
# raise-by 250 → to 350), so two sizes (pot, jam) cover open/3-bet/jam lines well.
# The point is to keep the game tree small enough that every reachable info set
# gets enough visits to converge — the previous 6-action × 4-raise tree had
# ~4.9M info sets (≈0.08 visits/set after 400k traversals) and never converged.
PREFLOP_ACTIONS = [FOLD, CHECK_CALL, BET_FULL_POT, ALL_IN]

# Cap on non-jam raises in the preflop tree.  2 = open + 3-bet with a sized raise;
# 4-bets and beyond are still reachable via ALL_IN (always legal), so 4-bet-jam
# lines survive.  Dropping from 3→2 prunes the deepest, rarest re-raise subtrees,
# which (a) shrinks the reachable info-set count so the visit budget converges
# faster and (b) removes the most expensive multiway all-in equity leaves.  This
# is the one knob here that trades strategy resolution (non-jam 4-bet sizing) for
# convergence speed — raise back to 3 if the wall-clock budget allows.
MAX_RAISES_PREFLOP = 2

# ── Training ───────────────────────────────────────────────────────────────────
# With the coarse 4-action / 2-raise tree the reachable info-set count drops to
# ~1e5 (vs ~4.9M before), so a full run can actually push average visits/set into
# the thousands.  "Convergence" is judged by the diagnostics train.py prints each
# checkpoint — visits/set histogram + premium-hand strategy drift — NOT by raw
# iteration count.  Stop when TARGET_VISITS_PER_SET is broadly met and the
# premium-hand drift between checkpoints has flattened.
# Upper bound on traversals — a *safety ceiling*, not the real stop condition.
# The convergence gate below stops the run as soon as the premium-hand strategy
# has stopped drifting, which under CFR+ + the coarse tree happens well inside
# this ceiling (≈7M traversals to hit TARGET_VISITS_PER_SET; ~7h at ~300 it/s on
# 8 workers).  The ceiling just guarantees termination if the gate never trips.
ITERATIONS         = 20_000_000  # ES-MCCFR traversal ceiling
QUICK_ITERATIONS   = 5_000       # smoke-test run (--quick flag)
CHECKPOINT_EVERY   = 1_000_000
PRUNE_MIN_VISITS   = 10        # drop info sets visited < N times at export
# Convergence gate (diagnostic only): target average visits per kept info set.
# Once most info sets clear this and the premium-hand drift has flattened across
# consecutive checkpoints, the table is effectively converged.
TARGET_VISITS_PER_SET = 1_000

# ── Early-stop convergence gate ────────────────────────────────────────────────
# Stop training once the premium UTG-open strategy mix (AA/KK/.../AKo) changes by
# less than CONVERGENCE_DRIFT_EPS (max |Δ probability|) between consecutive
# checkpoints, for CONVERGENCE_PATIENCE checkpoints in a row.  This is what
# actually ends the run — so the budget is spent reaching convergence, not
# chasing rare info sets long after the strategy has settled.
CONVERGENCE_DRIFT_EPS = 0.01
CONVERGENCE_PATIENCE  = 3
# Parallel mode: iterations between cross-worker merges. Smaller = closer to
# true sequential CFR (less worker divergence) but more broadcast overhead.
SYNC_EVERY         = 100_000
# Parallel mode RAM budget (GB) for the live regret/strategy tables and their
# per-worker copies.  Each round the trainer caps the number of concurrent
# workers so that  workers × (~3 × table_size)  stays under this budget.  As the
# tables grow over a long run the effective worker count scales down, which is
# what prevents the OOM-kill on the full 500k run.  Override with the
# PREFLOP_MEM_BUDGET_GB env var or the --mem-budget-gb CLI flag.
MEM_BUDGET_GB      = float(os.environ.get("PREFLOP_MEM_BUDGET_GB", "6"))

# ── Equity tables ──────────────────────────────────────────────────────────────
HU_EQUITY_BOARDS    = 2_000    # MC boards for 169×169 HU table build
# MC boards per multiway (3+-player) leaf rollout.  Lowered 400→250: these leaves
# are the dominant cost, and the result is cached per suit-isomorphic matchup, so
# 250 boards (±~3% per-leaf stderr) is enough resolution for the 169-bucket
# abstraction.  Don't drop below ~200 — the frozen cache value bakes the MC noise
# into the equilibrium, so too few boards biases the solution.
MULTIWAY_MC_BOARDS  = 250      # MC boards per multiway leaf rollout

# ── Export ─────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_PATH     = os.path.join(_ROOT, "bots", "vlad", "data", "preflop_cfr",
                               "preflop_strategy.npz")
CHECKPOINT_PATH = os.path.join(_ROOT, "preflop_cfr", "checkpoint.npz")
# Cached 169×169 HU equity table — built once, reused across runs/workers.
HU_TABLE_PATH   = os.path.join(_ROOT, "preflop_cfr", "hu_equity_table.npz")
