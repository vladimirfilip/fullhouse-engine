"""Hyperparameters and action-space constants for Deep CFR training."""

# ── Abstract action indices ────────────────────────────────────────────────
# Standard sizes: 27% is a nano-blocker/probe, 33%/50%/100%/200% are GTO staples.
# Weird sizes: 27% and 172% fall off the grids most GTO solvers are trained on,
# which exploits opponents whose defences are calibrated only to standard nodes.
FOLD           = 0
CHECK_CALL     = 1
BET_0_27X_POT  = 2   # 27 %  – weird nano-bet
BET_THIRD_POT  = 3   # 33 %  – standard small
BET_HALF_POT   = 4   # 50 %  – standard medium
BET_FULL_POT   = 5   # 100 % – pot-sized
BET_1_72X_POT  = 6   # 172 % – weird overbet
BET_2X_POT     = 7   # 200 % – standard overbet
ALL_IN         = 8
N_ACTIONS      = 9

ACTION_NAMES = [
    "fold", "check_call",
    "bet_0_27x", "bet_third", "bet_half", "bet_full",
    "bet_1_72x", "bet_2x", "all_in",
]

# ── Game constants ─────────────────────────────────────────────────────────
N_PLAYERS     = 6
INITIAL_STACK = 10_000
SMALL_BLIND   = 50
BIG_BLIND     = 100

# ── Feature vector ─────────────────────────────────────────────────────────
# Layout (must match build_feature_vector in deep_cfr_cpp/src/features.cpp AND
# _build_feature_vector in bots/the_house/bot.py byte-for-byte — any drift
# silently corrupts inference).
#   [0:52]    hole cards one-hot
#   [52:104]  board cards one-hot
#   [104:110] hero position rel. dealer (one-hot, 6)
#   [110]     pot / INITIAL_STACK
#   [111:117] per-seat stack / INITIAL_STACK (6)
#   [117:123] per-seat is_folded mask (6)
#   [123:129] per-seat is_all_in mask (6)
#   [129:135] per-seat bet_this_street / INITIAL_STACK (6)
#   [135:139] street one-hot (4)
#   [139]     pot odds
#   [140]     SPR log-scaled
#   [141]     amount owed / INITIAL_STACK
#   [142]     n_raises_this_street / MAX_RAISES_PER_STREET
#   [143]     hero bet_this_street / INITIAL_STACK
#   [144]     min effective stack / INITIAL_STACK
#   [145:151] last-aggressor seat one-hot (6)
#   [151]     last-aggressor amount / INITIAL_STACK
#   [152:155] board texture (flush-draw, paired, connected) (3)
#   [155]     n_active / N_PLAYERS
#   [156:252] action history 16 slots × 6 floats (seat, 4 action one-hot, amount/INITIAL_STACK)
# Tier-2b tightening (was 308): dropped last-aggressor rel-pos one-hot (-6),
# board monotone+two-paired bits (-2), action-history 24->16 slots (-48).
INPUT_DIM  = 252
# Training-tree raise cap AND the feature[142] normaliser. MUST match
# MAX_RAISES_PER_STREET in deep_cfr_cpp/src/config.hpp and
# _MAX_RAISES_PER_STREET in bots/the_house/bot.py (see config.hpp for the trade-off).
MAX_RAISES_PER_STREET = 4
# Tier-2b: 3×384. Calibration showed TRAINING (not CPU data-gen) is the
# bottleneck on the 96-core+5060Ti box, so gen has headroom for a bigger net that
# makes fewer value errors than the 3×256 it replaces. MUST match HIDDEN_DIM /
# N_LAYERS in deep_cfr_cpp/src/config.hpp (rebuild the extension after changing)
# — bot.py infers both from the .npz weight shapes, so it needs no edit.
HIDDEN_DIM = 384
N_LAYERS   = 3     # hidden layers

# ── Memory buffers ─────────────────────────────────────────────────────────
# 8M cap: at 25k games/iter the reservoir saturates around iter 320; past that
# it keeps the most recent, highest-weight (DCFR-discounted) samples.
REGRET_BUF_CAP   = 8_000_000
STRATEGY_BUF_CAP = 8_000_000

# ── Training loop ─────────────────────────────────────────────────────────
# Data-gen on CPU is the bottleneck (training runs on GPU), so the budget is
# rebalanced toward more iterations with fewer games each: CFR converges through
# the time-averaging across iterations, not raw samples per iteration. 600 × 25k
# = 15M traversals — same total work as the old 300 × 50k but 2× the net refits
# (averaging steps). Use --quick (5 iters × 200 games) to smoke-test the build.
K_ITERATIONS      = 600
GAMES_PER_ITER    = 25_000
# Calibration on the 96-core + RTX 5060 Ti box showed TRAINING (not CPU data-gen)
# is the bottleneck: ~40 ms/step for this tiny MLP is host/launch-bound, not
# compute-bound. A bigger batch amortises the per-step Python/transfer/sampling
# overhead — the net fits easily, so 65 536 (4× the old 16 384) cuts step count
# and H2D copies 4× for the same buffer coverage. LR kept at 2e-3 (conservative;
# raise toward 3-4e-3 only if the loss curve stays smooth) — watch calibration.
BATCH_SIZE        = 65_536
LEARNING_RATE     = 2e-3

# Regret net is retrained from scratch each iteration. CFR converges through the
# across-iteration time-average, not per-iter fit depth, so fewer steps/iter +
# more iterations is the right trade under a fixed wall-clock budget. ~6 passes
# over the 8M buffer at batch 65 536: REGRET_BUF_CAP / BATCH_SIZE * 6 ≈ 732.
REGRET_TRAIN_STEPS    = 750

# Strategy net: trained once at the end and ships in production. Needs more passes
# than the regret net but at batch 65 536 far fewer steps reach them: ~33 passes =
# STRATEGY_BUF_CAP / BATCH_SIZE * 33 ≈ 4 030. The final LR is cosine-decayed in
# train.py. Sized to fit the ~1h budget reserve for the final fit.
STRATEGY_TRAIN_STEPS  = 4_000

# ── Export ─────────────────────────────────────────────────────────────────
MODEL_FILENAME = "gto_strategy"

# ── Parallelism ────────────────────────────────────────────────────────────
import os
N_WORKERS = max(1, os.cpu_count() or 1)   # one worker per logical CPU
