#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <array>
#include "config.hpp"
#include "features.hpp"

struct Layer {
    Eigen::MatrixXf W;  // [out_dim, in_dim]
    Eigen::VectorXf b;  // [out_dim]
    // Adam moment matrices — allocated by make_layer(), zero-sized in inference nets.
    Eigen::MatrixXf mW, vW;
    Eigen::VectorXf mb, vb;
};

struct MLP {
    std::vector<Layer> layers;  // hidden layers + output layer
    bool softmax_output;        // true for strategy net
    int  adam_t = 0;            // global step counter
};

// Initialise nets matching PyTorch nn.Linear defaults (Kaiming uniform, zero bias)
MLP make_regret_net();
MLP make_strategy_net();

// Lightweight inference-only net: weights loaded from caller, no Adam moment
// matrices allocated.  Use this in generate_and_add() to save ~7.6 MB per
// iteration and avoid the associated heap fragmentation.
MLP make_inference_net(bool softmax_output = false);

// Forward pass (batch=1) — used in MCCFR hot path; avoids Eigen overhead for tiny input
std::array<float, N_ACTIONS> forward_single(const MLP& net, const FeatureVec& x);

