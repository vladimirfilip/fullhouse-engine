#include "network.hpp"
#include <cmath>
#include <cassert>

// ── Kaiming uniform initialisation ───────────────────────────────────────────
// PyTorch nn.Linear default: Uniform(-sqrt(1/fan_in), sqrt(1/fan_in))
static void kaiming_uniform(Eigen::MatrixXf& W, std::mt19937& rng) {
    float bound = std::sqrt(1.0f / W.cols());
    std::uniform_real_distribution<float> dist(-bound, bound);
    for (int i = 0; i < W.rows(); i++)
        for (int j = 0; j < W.cols(); j++)
            W(i, j) = dist(rng);
}

static Layer make_layer(int in_dim, int out_dim, std::mt19937& rng) {
    Layer l;
    l.W  = Eigen::MatrixXf(out_dim, in_dim);
    l.b  = Eigen::VectorXf::Zero(out_dim);
    l.mW = Eigen::MatrixXf::Zero(out_dim, in_dim);
    l.vW = Eigen::MatrixXf::Zero(out_dim, in_dim);
    l.mb = Eigen::VectorXf::Zero(out_dim);
    l.vb = Eigen::VectorXf::Zero(out_dim);
    kaiming_uniform(l.W, rng);
    return l;
}

// Inference-only layer: allocates W and b but leaves Adam moments as 0×0 matrices
// (no heap allocation).  Used by make_inference_net() to cut ~7.6 MB per call.
static Layer make_inference_layer(int in_dim, int out_dim) {
    Layer l;
    l.W = Eigen::MatrixXf(out_dim, in_dim);
    l.b = Eigen::VectorXf::Zero(out_dim);
    // mW, vW, mb, vb stay default-constructed (size 0) — forward_single never
    // touches them, and backprop_and_update is never called on this net.
    return l;
}

static MLP make_net(bool softmax_output) {
    std::mt19937 rng(std::random_device{}());
    MLP net;
    net.softmax_output = softmax_output;
    net.adam_t         = 0;
    int in_dim = INPUT_DIM;
    for (int i = 0; i < N_LAYERS; i++) {
        net.layers.push_back(make_layer(in_dim, HIDDEN_DIM, rng));
        in_dim = HIDDEN_DIM;
    }
    net.layers.push_back(make_layer(HIDDEN_DIM, N_ACTIONS, rng));
    return net;
}

MLP make_regret_net()   { return make_net(false); }
MLP make_strategy_net() { return make_net(true);  }

MLP make_inference_net(bool softmax_output) {
    MLP net;
    net.softmax_output = softmax_output;
    net.adam_t         = 0;
    int in_dim = INPUT_DIM;
    for (int i = 0; i < N_LAYERS; i++) {
        net.layers.push_back(make_inference_layer(in_dim, HIDDEN_DIM));
        in_dim = HIDDEN_DIM;
    }
    net.layers.push_back(make_inference_layer(HIDDEN_DIM, N_ACTIONS));
    return net;
}

// ── Forward pass (batch=1) ────────────────────────────────────────────────────
// MCCFR calls this millions of times per iteration. Three thread-local buffers,
// none ever resized: a/b ping-pong at HIDDEN_DIM for hidden layers, out_buf at
// N_ACTIONS for the output layer. Layer 0 uses Eigen::Map over the FeatureVec
// (zero-copy) so there is no element-by-element input copy either.
std::array<float, N_ACTIONS> forward_single(const MLP& net, const FeatureVec& x) {
    thread_local Eigen::VectorXf a(HIDDEN_DIM);
    thread_local Eigen::VectorXf b(HIDDEN_DIM);
    thread_local Eigen::VectorXf out_buf(N_ACTIONS);

    // Layer 0: INPUT_DIM → HIDDEN_DIM (Map avoids copying x into a temporary)
    {
        const Layer& l0 = net.layers[0];
        a.noalias() = l0.W * Eigen::Map<const Eigen::VectorXf>(x.data(), INPUT_DIM) + l0.b;
        a = (a.array() < 0.0f).select(LEAKY_ALPHA * a, a);
    }
    Eigen::VectorXf* cur = &a;
    Eigen::VectorXf* nxt = &b;

    // Hidden layers 1..N_LAYERS-1: HIDDEN_DIM → HIDDEN_DIM, no resize needed
    for (int li = 1; li < (int)net.layers.size() - 1; li++) {
        const Layer& l = net.layers[li];
        nxt->noalias() = l.W * (*cur) + l.b;
        *nxt = (nxt->array() < 0.0f).select(LEAKY_ALPHA * (*nxt), *nxt);
        std::swap(cur, nxt);
    }

    // Output layer: HIDDEN_DIM → N_ACTIONS into fixed-size out_buf
    {
        const Layer& lo = net.layers.back();
        out_buf.noalias() = lo.W * (*cur) + lo.b;
    }

    std::array<float, N_ACTIONS> out{};
    if (net.softmax_output) {
        float mx = out_buf.maxCoeff();
        float sum = 0.0f;
        for (int i = 0; i < N_ACTIONS; i++) {
            float e = std::exp(out_buf(i) - mx);
            out[i]  = e;
            sum    += e;
        }
        float inv = 1.0f / sum;
        for (int i = 0; i < N_ACTIONS; i++) out[i] *= inv;
    } else {
        for (int i = 0; i < N_ACTIONS; i++) out[i] = out_buf(i);
    }
    return out;
}

