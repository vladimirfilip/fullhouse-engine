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
    // Adam moments
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

// One Adam training step on a batch. Returns mean loss.
// regret net: MSE(pred, target)
// strategy net: iteration-weighted MSE
float train_step_regret(MLP& net,
                        const float* states,   // [batch, INPUT_DIM]
                        const float* targets,  // [batch, N_ACTIONS]
                        int batch);

float train_step_strategy(MLP& net,
                          const float* states,   // [batch, INPUT_DIM]
                          const float* targets,  // [batch, N_ACTIONS]
                          const float* weights,  // [batch] — iteration weights
                          int batch);

// Export all layer weights as a flat list of (W_ptr, rows, cols, b_ptr, size) tuples
// for the .npz writer — avoids copying
struct LayerWeights {
    const float* W;  // [out_dim * in_dim], row-major
    int out_dim, in_dim;
    const float* b;  // [out_dim]
};
std::vector<LayerWeights> get_layer_weights(const MLP& net);
