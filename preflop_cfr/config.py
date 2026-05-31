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
PREFLOP_ACTIONS = [FOLD, CHECK_CALL, BET_HALF_POT, BET_FULL_POT, BET_2X_POT, ALL_IN]

MAX_RAISES_PREFLOP = 4   # mirrors engine raise cap

# ── Training ───────────────────────────────────────────────────────────────────
ITERATIONS         = 500_000   # ES-MCCFR game traversals (full run)
QUICK_ITERATIONS   = 5_000     # smoke-test run (--quick flag)
CHECKPOINT_EVERY   = 50_000
PRUNE_MIN_VISITS   = 10        # drop info sets visited < N times at export
# Parallel mode: iterations between cross-worker merges. Smaller = closer to
# true sequential CFR (less worker divergence) but more broadcast overhead.
SYNC_EVERY         = 20_000

# ── Equity tables ──────────────────────────────────────────────────────────────
HU_EQUITY_BOARDS    = 2_000    # MC boards for 169×169 HU table build
MULTIWAY_MC_BOARDS  = 400      # MC boards per multiway leaf rollout

# ── Export ─────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_PATH     = os.path.join(_ROOT, "bots", "vlad", "data", "preflop_cfr",
                               "preflop_strategy.npz")
CHECKPOINT_PATH = os.path.join(_ROOT, "preflop_cfr", "checkpoint.npz")
# Cached 169×169 HU equity table — built once, reused across runs/workers.
HU_TABLE_PATH   = os.path.join(_ROOT, "preflop_cfr", "hu_equity_table.npz")
