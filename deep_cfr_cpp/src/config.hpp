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
// (24 slots × 6 floats). See features.cpp for the exact layout — must match
// _build_feature_vector in bots/vlad/bot.py byte-for-byte.
constexpr int INPUT_DIM    = 308;
constexpr int HIDDEN_DIM   = 512;
constexpr int N_LAYERS     = 4;   // hidden layers

// Memory buffers
constexpr int REGRET_BUF_CAP   = 4'000'000;
constexpr int STRATEGY_BUF_CAP = 4'000'000;

// Training loop
constexpr int   K_ITERATIONS      = 100;
constexpr int   GAMES_PER_ITER    = 10'000;
constexpr int   BATCH_SIZE        = 4'096;
constexpr float LR                = 1e-3f;
constexpr int   REGRET_TRAIN_STEPS   = 5'000;   // ~5 passes over 4M buffer
constexpr int   STRATEGY_TRAIN_STEPS = 15'000;  // ~15 passes; runs once, is the exported model

// Network
constexpr float LEAKY_ALPHA = 0.01f;

// Adam defaults matching PyTorch
constexpr float ADAM_BETA1 = 0.9f;
constexpr float ADAM_BETA2 = 0.999f;
constexpr float ADAM_EPS   = 1e-8f;

// MCCFR
constexpr int MAX_DEPTH             = 200;
constexpr int MAX_RAISES_PER_STREET = 4;   // cap to bound game tree; mirrors common cap rules

// Export
constexpr const char* MODEL_FILENAME = "gto_strategy";
