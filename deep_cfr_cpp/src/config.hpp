#pragma once

// Abstract action indices — mirrors config.py exactly.
// Sizes 0.27x and 1.72x are deliberately off-grid relative to standard GTO
// solver trees, which exploits opponents whose range defences are calibrated
// only to common nodes (25%, 33%, 50%, 100%, 150%, 200%).
constexpr int FOLD          = 0;
constexpr int CHECK_CALL    = 1;
constexpr int BET_0_27X_POT = 2;   // 27 %  – weird nano-bet
constexpr int BET_THIRD_POT = 3;   // 33 %  – standard small
constexpr int BET_HALF_POT  = 4;   // 50 %  – standard medium
constexpr int BET_FULL_POT  = 5;   // 100 % – pot-sized
constexpr int BET_1_72X_POT = 6;   // 172 % – weird overbet
constexpr int BET_2X_POT    = 7;   // 200 % – standard overbet
constexpr int ALL_IN        = 8;
constexpr int N_ACTIONS     = 9;

// Game constants
constexpr int N_PLAYERS    = 6;
constexpr int INITIAL_STACK = 10000;
constexpr int SMALL_BLIND  = 50;
constexpr int BIG_BLIND    = 100;

// Feature vector
// 164 fixed (cards 104 + hero pos 6 + pot/stacks 7 + status masks 18 + street 4 +
// scalars 6 + last-aggressor 13 + texture 5 + n_active 1) + 144 action history
// (16 slots × 6 floats). See features.cpp for the exact layout — must match
// _build_feature_vector in bots/the_house/bot.py byte-for-byte.
constexpr int INPUT_DIM    = 252;
// Tier-2b: 3×384 net. Calibration showed training (not CPU data-gen) is the
// bottleneck on the 96-core+5060Ti box, so gen has headroom for a bigger net
// that makes fewer value errors. HIDDEN_DIM/N_LAYERS are read by make_net() +
// forward_single, so changing them DOES require rebuilding the extension.
constexpr int HIDDEN_DIM   = 384;
constexpr int N_LAYERS     = 3;   // hidden layers

// Memory buffers
constexpr int REGRET_BUF_CAP   = 8'000'000;
constexpr int STRATEGY_BUF_CAP = 8'000'000;

// Training loop. NOTE: these are NOT read at runtime — train.py owns the loop
// and passes its own values from deep_cfr/config.py. They are kept in sync here
// only so the C++ header isn't a stale source of truth for anyone reading it.
constexpr int   K_ITERATIONS      = 600;
constexpr int   GAMES_PER_ITER    = 25'000;
constexpr int   BATCH_SIZE        = 16'384;
constexpr float LR                = 2e-3f;
constexpr int   REGRET_TRAIN_STEPS   = 2'500;   // ~5 passes over the 8M buffer
constexpr int   STRATEGY_TRAIN_STEPS = 12'500;  // runs once; is the exported model

// Network
constexpr float LEAKY_ALPHA = 0.01f;

// DCFR-style sample discounting.  Deep CFR weights each (state, target) sample by
// the iteration t at which it was produced; the weighted-mean loss then lets
// later, better-trained iterations dominate.  Raising the exponent from 1.0
// (plain linear CFR) to DCFR_ALPHA down-weights the early, random-net iterations
// more aggressively — the practical benefit DCFR has over linear CFR — by using
// weight = t^DCFR_ALPHA instead of t.  α=1.5 is the canonical DCFR value and
// keeps weights numerically tame (t≤300 → t^1.5 ≤ ~5 200, safe in float32).
constexpr float DCFR_ALPHA = 1.5f;

// ── Regret-based pruning (Tier-2b) ──────────────────────────────────────────
// At traverser nodes, skip recursion into actions that regret matching has
// already zeroed (strategy=0, i.e. non-positive regret) AND whose regret sits
// in the worst fraction of the node's regret range. They contribute 0 to the
// node EV (strategy=0, so the EV stays exact) and keep their carried-forward
// negative regret target, so they remain pruned. ~30-55% fewer traversed nodes
// once regrets stabilise.
//  - Gated off until the regret net is no longer random (iteration_t >= start),
//    because raw_regrets from a fresh net are meaningless.
//  - Scale-free margin (fraction of the node's max-min regret span), so no
//    chip-EV tuning is needed.
//  - Always keeps at least MIN_TRAVERSE_ACTIONS to bound variance.
constexpr int   PRUNE_START_ITER     = 30;
constexpr float PRUNE_MARGIN_FRAC    = 0.60f;
constexpr int   MIN_TRAVERSE_ACTIONS = 2;

// Adam defaults matching PyTorch
constexpr float ADAM_BETA1 = 0.9f;
constexpr float ADAM_BETA2 = 0.999f;
constexpr float ADAM_EPS   = 1e-8f;

// MCCFR
constexpr int MAX_DEPTH             = 200;
// Raise cap per street (training tree only — production engine has no cap).
// TRADE-OFF: a lower cap shrinks the traverser subtree (each node costs a full
// feature build + MLP forward), so data-gen throughput and convergence speed
// improve markedly — the dominant lever for a fixed wall-clock budget. The cost
// is a known boundary artifact: at exactly n_raises==cap the only aggressive
// action left is ALL_IN, so the net learns inflated jam frequency in deep
// re-raise pots. 4 still covers open/3-bet/4-bet/5-bet — 6+ bet wars are
// vanishingly rare — and bot.py's SPR-gated ALL_IN dampener compensates for the
// residual artifact. MUST equal _MAX_RAISES_PER_STREET in bots/vlad/bot.py and
// MAX_RAISES_PER_STREET in deep_cfr/config.py (feature[142] normaliser); a
// mismatch silently corrupts inference. Changing this requires a full retrain.
constexpr int MAX_RAISES_PER_STREET = 4;

// Export
constexpr const char* MODEL_FILENAME = "gto_strategy";
