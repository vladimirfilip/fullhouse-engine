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
# Two sized raises + jam.  The single-pot-raise tree ([F,C,POT,JAM]) converged
# but played jam-happy poker: with only one non-jam size the solver expresses all
# aggression as ALL_IN (open-shoves AKs, jams QQ for 100bb), which bleeds chips vs
# disciplined fields.  Adding a smaller raise gives a genuine non-jam open/3-bet
# ladder:  ⅓-pot ≈ a ~2.3bb open, full-pot scales up for 3-bets, ALL_IN for
# 4-bet/jam.  This grows the reachable tree to ~2-4M info sets (vs ~430k), so it
# needs the bigger visit budget below + the parallel throughput fixes in train.py.
# Do NOT expand to the full 6-raise set — that was the ~4.9M tree that never
# converged (≈0.08 visits/set after 400k traversals).
PREFLOP_ACTIONS = [FOLD, CHECK_CALL, BET_THIRD_POT, BET_FULL_POT, ALL_IN]

# Cap on non-jam raises in the preflop tree.  2 = open + 3-bet with sized raises;
# 4-bets and beyond are reachable via ALL_IN (always legal), so 4-bet-jam lines
# survive.  With two sized raises active this already fixes the jam-happy
# pathology (the solver can 3-bet to a real size instead of shoving) WITHOUT the
# tree blow-up of MAX=3: calibration showed [⅓,POT]+MAX=3 reaches ~5M info sets
# (won't converge in 16h on the current solver), so 2 is the tractable sweet spot.
MAX_RAISES_PREFLOP = 2

# Info-set key history truncation: keep only the last N abstract actions.
# Full histories explode to ~4.9M info sets (untrainable in 16h); truncating
# to 4 collapses many distinct long sequences that share the same recent
# betting context, targeting ~250k–500k info sets.  Must be mirrored in
# bot.py (_preflop_infoset_key) and abstraction.py.
HISTORY_TRUNCATION_LEN = 4

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
PRUNE_MIN_VISITS   = 40        # drop info sets visited < N times at export
# Convergence gate (diagnostic only): target average visits per kept info set.
# Once most info sets clear this and the premium-hand drift has flattened across
# consecutive checkpoints, the table is effectively converged.
TARGET_VISITS_PER_SET = 400

# ── Early-stop convergence gate ────────────────────────────────────────────────
# Stop training once the premium UTG-open strategy mix (AA/KK/.../AKo) changes by
# less than CONVERGENCE_DRIFT_EPS (max |Δ probability|) between consecutive
# checkpoints, for CONVERGENCE_PATIENCE checkpoints in a row.  This is what
# actually ends the run — so the budget is spent reaching convergence, not
# chasing rare info sets long after the strategy has settled.
CONVERGENCE_DRIFT_EPS = 0.01
CONVERGENCE_PATIENCE  = 3
# Guard against the failure that shipped the last table: the premium-hand drift
# gate tripped while >50% of the tree was still diffuse (premiums settle early).
# Don't allow early-stop until this fraction of kept info sets has also cleared
# TARGET_VISITS_PER_SET — i.e. the *whole* tree, not just 6 hands, is well-visited.
CONVERGENCE_MIN_MET_FRAC = 0.80
# Parallel mode: iterations between cross-worker merges. The parent merge is the
# scaling bottleneck (single-threaded delta-sum over the touched rows), so on a
# many-core box raise this to amortize the merge over more per-worker work: at 96
# workers, 1M iters/round ≈ 10k iters/worker before a merge. Smaller = closer to
# true sequential CFR (less worker divergence) but a merge-bound throughput floor.
SYNC_EVERY         = 1_000_000
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
