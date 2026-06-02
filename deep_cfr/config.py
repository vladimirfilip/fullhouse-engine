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
# _build_feature_vector in bots/vlad/bot.py byte-for-byte — any drift silently
# corrupts inference).
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
#   [152:158] last-aggressor pos rel. hero one-hot (6)
#   [158:163] board texture (flush-draw, monotone, paired, two-paired, connected)
#   [163]     n_active / N_PLAYERS
#   [164:308] action history 24 slots × 6 floats (seat, 4 action one-hot, amount/INITIAL_STACK)
INPUT_DIM  = 308
# Training-tree raise cap AND the feature[142] normaliser. MUST match
# MAX_RAISES_PER_STREET in deep_cfr_cpp/src/config.hpp and
# _MAX_RAISES_PER_STREET in bots/vlad/bot.py (see config.hpp for the trade-off).
MAX_RAISES_PER_STREET = 4
# 3×256 (~180k params) instead of 4×512 (~1M): the smaller net cuts the
# per-node forward cost in the C++ data-gen path (the CPU bottleneck) ~3–4× and
# generalises better on the data a ~30 h run produces. MUST match HIDDEN_DIM /
# N_LAYERS in deep_cfr_cpp/src/config.hpp (rebuild the extension after changing)
# — bot.py infers both from the .npz weight shapes, so it needs no edit.
HIDDEN_DIM = 256
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
# Bigger batch better utilises the GPU on this tiny 4×512 MLP: at 4 096 the
# per-step Python/transfer overhead dominated and GPU occupancy was low. 16 384
# (4×) cuts step count for the same buffer coverage and raises occupancy. LR is
# scaled by sqrt(4)=2× (the conservative Adam rule) to keep the update size
# stable; the cosine scheduler in train.py decays from here.
BATCH_SIZE        = 16_384
LEARNING_RATE     = 2e-3

# Step counts hold the epoch count fixed under the 4× larger batch (else a bigger
# batch would silently quarter the passes over the buffer).
# Regret net: retrained from scratch each iteration; ~5 passes over the full
# buffer.  Formula: REGRET_BUF_CAP / BATCH_SIZE * 5 ≈ 2 441.
REGRET_TRAIN_STEPS    = 2_500

# Strategy net: trained once at the end and ships in production.  Needs more
# passes than the regret net.  Formula: STRATEGY_BUF_CAP / BATCH_SIZE * 25
# ≈ 12 207.  The final LR is decayed by the cosine scheduler in train.py.
STRATEGY_TRAIN_STEPS  = 12_500

# ── Export ─────────────────────────────────────────────────────────────────
MODEL_FILENAME = "gto_strategy"

# ── Parallelism ────────────────────────────────────────────────────────────
import os
N_WORKERS = max(1, os.cpu_count() or 1)   # one worker per logical CPU
