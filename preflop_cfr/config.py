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
# Three sized raises + jam.  History:
#   [F,C,POT,JAM]        converged but jam-happy (one non-jam size => all
#                        aggression collapses to ALL_IN: open-shoves AKs, jams QQ).
#   [F,C,⅓,POT,JAM]      added a small open size, but premiums STILL open-jammed
#                        24–34% (see field audit) because the jam, not the grid,
#                        was the cause: in a preflop-only model ALL_IN realises
#                        full equity while the play-line is discounted, so value
#                        hands shove to get stacks in.  Fixed structurally by the
#                        ALL_IN gate (game.legal_actions) + the hand-aware leaf
#                        (game._realization_weights), NOT by more sizes.
#   [F,C,⅓,POT,2POT,JAM] current: ⅓-pot ≈ ~2.3bb open, full-pot for 3-bets, 2×-pot
#                        gives a big NON-jam value/4-bet size so premiums express
#                        value with a sized raise instead of shoving.  The
#                        betting-context key is size-agnostic (it stores n_raises +
#                        a 4-level facing bucket, not which size), so this added
#                        size barely grows the info-set count (~180k) — the old
#                        "more sizes => untrainable" warning applied to the FULL-
#                        HISTORY key, which is no longer used.
# Do NOT expand to the full 6-raise set — combined with full-history keys that was
# the ~4.9M tree that never converged (≈0.08 visits/set after 400k traversals).
PREFLOP_ACTIONS = [FOLD, CHECK_CALL, BET_THIRD_POT, BET_FULL_POT, BET_2X_POT, ALL_IN]

# Cap on non-jam raises in the preflop tree.  3 = open + 3-bet + 4-bet with sized
# raises; the 5-bet (n_raises==3) and beyond collapse to ALL_IN via the gate in
# game.legal_actions.  This is the correct ladder at 100bb: a 5-bet IS a shove at
# this depth (~60bb+ after open≈2.5bb / 3-bet≈9bb / 4-bet≈24bb), so jamming it is
# GTO, not a leak — whereas a forced 4-bet-JAM (the old MAX=2 behaviour) was the
# last remnant of the open-shove pathology.  Do NOT raise to 4: sized 5-bet/6-bet
# nodes are shove-equivalent at 100bb (non-GTO) and only dilute the visit budget.
#
# Cost of 2->3: the betting-context key already caps n_raises at 3
# (abstraction.infoset_key), so this adds NO new key dimension — only the rare,
# mostly-heads-up 4-bet/5-bet subtree (~+10-25% info sets, ~+5-15% per-iter).
# Measured equilibrium reach: 4-bet nodes ≈7.4% of decisions (well-visited),
# 5-bet nodes ≈0.9% (thin but near-deterministic call/fold/jam → fine, and the
# rarest fall under PRUNE_MIN_VISITS).
MAX_RAISES_PREFLOP = 3

# NOTE: the info-set key is now the compact betting-context tuple in
# abstraction.py (hero_pos, n_raises, facing, n_live, hero_committed,
# last_aggr_rel, bucket) — an imperfect-recall abstraction, NOT last-N-actions
# history truncation.  That key collapses the reachable tree to ~180k info sets
# (≈1,065 distinct public betting contexts × 169 hand buckets), well under the
# naive 6·4·4·5·2·6·169 ≈ 973k ceiling.  The old HISTORY_TRUNCATION_LEN knob has
# been removed.  Any change to the key encoding must still be mirrored in
# bot.py (_preflop_infoset_key) and abstraction.py.

# ── Training ───────────────────────────────────────────────────────────────────
# "Convergence" is judged by the diagnostics train.py prints each checkpoint —
# visits/set histogram + premium-hand strategy drift — NOT by raw iteration
# count.  The ceiling below is a safety cap; the convergence gate stops the run
# as soon as the drift gate fires.
#
# Budget estimate for the ~180k betting-context tree:
#   ~400 visits/set × ~180k sets / ~3 opponent touches per iter ≈ 24M traversals
#   for average convergence; 80th-percentile needs ~2× that (~48M).  300M is a
#   deliberate over-cover so SB/BB/3-bet nodes (slower to settle than the UTG-open
#   node the drift gate watches) are well-visited, not just the premium hands.
#   At ~2500 it/s (96-worker VM): 300M iters ≈ 33h.
ITERATIONS         = 300_000_000  # ES-MCCFR traversal ceiling
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
# Early-stop master switch.  Disabled for the guaranteed fixed-iteration run:
# the premium-drift gate watches only the UTG-open node (perfect recall, settles
# first) so it fired prematurely while SB/BB/3-bet nodes were still diffuse —
# the main reason the shipped table looked unconverged.  With a fixed --iters
# budget sized to over-cover the tree (see ITERATIONS / plan §E), run the whole
# budget; the drift/visit prints below stay on as diagnostics only.
EARLY_STOP_ENABLED    = False
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

# ── Shared-memory parallel mode ────────────────────────────────────────────────
# Number of open-addressing slots for the shared hash table.  Must exceed the
# expected info-set count by at least 2× (load factor ≤ 0.5).  The 5-action
# 2-raise tree with the betting-context key reaches ~180k info sets, so 1M slots
# gives a comfortable ~0.18 load factor.  Each slot consumes 9×8 + 9×4 + 8 + 8 =
# 124 bytes → 1M slots ≈ 124 MB total.  Windows commits shared memory to the page
# file lazily (on access), so the physical-RAM footprint grows with the actual
# info-set count, not the capacity.  Bump via PREFLOP_SHARED_CAPACITY if a future
# key encoding grows the tree.
SHARED_TABLE_CAPACITY = int(os.environ.get("PREFLOP_SHARED_CAPACITY", "1000000"))

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
