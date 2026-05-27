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

// ── Forward pass (batch=1) ────────────────────────────────────────────────────
// MCCFR calls this millions of times per iteration. Each call previously
// heap-allocated N_LAYERS+1 Eigen::VectorXf temporaries; here we reuse a pair
// of thread-local scratch buffers (ping-pong A/B) sized to the widest layer
// (HIDDEN_DIM, which exceeds INPUT_DIM and N_ACTIONS).
std::array<float, N_ACTIONS> forward_single(const MLP& net, const FeatureVec& x) {
    thread_local Eigen::VectorXf a(HIDDEN_DIM);
    thread_local Eigen::VectorXf b(HIDDEN_DIM);

    int in_dim = INPUT_DIM;
    a.resize(in_dim);
    for (int i = 0; i < in_dim; i++) a(i) = x[i];
    Eigen::VectorXf* cur = &a;
    Eigen::VectorXf* nxt = &b;

    for (int li = 0; li < (int)net.layers.size(); li++) {
        const Layer& l   = net.layers[li];
        int out_dim      = (int)l.W.rows();
        nxt->resize(out_dim);
        nxt->noalias() = l.W * (*cur) + l.b;
        bool is_last = (li == (int)net.layers.size() - 1);
        if (!is_last) {
            *nxt = (nxt->array() < 0.0f).select(LEAKY_ALPHA * (*nxt), *nxt);
        }
        std::swap(cur, nxt);
    }

    std::array<float, N_ACTIONS> out{};
    if (net.softmax_output) {
        float mx = cur->maxCoeff();
        float sum = 0.0f;
        for (int i = 0; i < N_ACTIONS; i++) {
            float e = std::exp((*cur)(i) - mx);
            out[i]  = e;
            sum    += e;
        }
        float inv = 1.0f / sum;
        for (int i = 0; i < N_ACTIONS; i++) out[i] *= inv;
    } else {
        for (int i = 0; i < N_ACTIONS; i++) out[i] = (*cur)(i);
    }
    return out;
}

// ── Adam update helper ────────────────────────────────────────────────────────
static void adam_update(Eigen::MatrixXf& param, Eigen::MatrixXf& m, Eigen::MatrixXf& v,
                        const Eigen::MatrixXf& grad, int t) {
    float bc1 = 1.0f - std::pow(ADAM_BETA1, (float)t);
    float bc2 = 1.0f - std::pow(ADAM_BETA2, (float)t);
    m = ADAM_BETA1 * m + (1.0f - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1.0f - ADAM_BETA2) * grad.cwiseProduct(grad);
    Eigen::MatrixXf m_hat = m / bc1;
    Eigen::MatrixXf v_hat = v / bc2;
    param -= LR * m_hat.cwiseQuotient((v_hat.cwiseSqrt().array() + ADAM_EPS).matrix());
}
static void adam_update(Eigen::VectorXf& param, Eigen::VectorXf& m, Eigen::VectorXf& v,
                        const Eigen::VectorXf& grad, int t) {
    float bc1 = 1.0f - std::pow(ADAM_BETA1, (float)t);
    float bc2 = 1.0f - std::pow(ADAM_BETA2, (float)t);
    m = ADAM_BETA1 * m + (1.0f - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1.0f - ADAM_BETA2) * grad.cwiseProduct(grad);
    Eigen::VectorXf m_hat = m / bc1;
    Eigen::VectorXf v_hat = v / bc2;
    param -= LR * m_hat.cwiseQuotient((v_hat.cwiseSqrt().array() + ADAM_EPS).matrix());
}

// ── Batch forward pass (for training) ────────────────────────────────────────
// Returns activations at each layer (needed for backprop)
struct FwdCache {
    std::vector<Eigen::MatrixXf> acts; // acts[i] = input to layer i, [in_dim, batch]
    Eigen::MatrixXf output;            // [N_ACTIONS, batch]
};

static FwdCache forward_batch(const MLP& net, const Eigen::MatrixXf& X) {
    // X: [INPUT_DIM, batch]
    FwdCache cache;
    cache.acts.resize(net.layers.size());
    Eigen::MatrixXf cur = X;

    for (int li = 0; li < (int)net.layers.size(); li++) {
        cache.acts[li] = cur;
        const Layer& l = net.layers[li];
        // y = W * cur + b (broadcast bias)
        Eigen::MatrixXf y = (l.W * cur).colwise() + l.b;
        bool is_last = (li == (int)net.layers.size() - 1);
        if (!is_last) {
            // LeakyReLU
            y = (y.array() < 0.0f).select(LEAKY_ALPHA * y, y);
        }
        cur = y;
    }
    // Softmax or linear output
    if (net.softmax_output) {
        Eigen::MatrixXf shifted = cur.colwise() - cur.colwise().maxCoeff().transpose();
        Eigen::MatrixXf exp_    = shifted.array().exp();
        cur = exp_.array().rowwise() / exp_.colwise().sum().array();
    }
    cache.output = cur;
    return cache;
}

// ── Backprop + Adam ───────────────────────────────────────────────────────────
static float backprop_and_update(MLP& net, const FwdCache& cache,
                                  const Eigen::MatrixXf& targets,     // [N_ACTIONS, batch]
                                  const Eigen::RowVectorXf* weights) { // [batch] or nullptr
    int batch = (int)cache.output.cols();

    // MSE loss gradient at output: dL/dy = 2*(pred-target)/batch
    // With per-sample weighting: dL/dy_i = w_i * 2*(pred_i - target_i) / batch
    Eigen::MatrixXf delta = 2.0f * (cache.output - targets) / batch; // [N_ACTIONS, batch]
    if (weights) {
        // Normalise weights: w_norm = w / mean(w)
        float mean_w = weights->mean() + 1e-8f;
        Eigen::RowVectorXf w_norm = weights->array() / mean_w;
        delta = delta.array().rowwise() * w_norm.array();
    }

    // Compute mean loss for logging
    float loss = (cache.output - targets).array().square().mean();

    // Backprop through layers (last → first)
    Eigen::MatrixXf d = delta;
    net.adam_t++;
    for (int li = (int)net.layers.size() - 1; li >= 0; li--) {
        Layer& layer = net.layers[li];
        const Eigen::MatrixXf& a = cache.acts[li]; // [in_dim, batch]

        // dW = d * a^T / (we already divided by batch above)
        Eigen::MatrixXf dW = d * a.transpose();
        Eigen::VectorXf db = d.rowwise().sum();

        // Propagate through activation (LeakyReLU) before passing to previous layer
        Eigen::MatrixXf d_prev = layer.W.transpose() * d;
        if (li > 0) {
            // The input to layer li is the output of LeakyReLU of layer li-1
            // So we need: d_prev *= leaky_relu_grad(cache.acts[li])
            // cache.acts[li] = input to layer li (= output of leaky_relu of li-1)
            d_prev = d_prev.array() * (a.array() > 0.0f).cast<float>()
                   + d_prev.array() * (a.array() <= 0.0f).cast<float>() * LEAKY_ALPHA;
        }

        adam_update(layer.W, layer.mW, layer.vW, dW, net.adam_t);
        adam_update(layer.b, layer.mb, layer.vb, db, net.adam_t);

        d = d_prev;
    }
    return loss;
}

// ── Training steps ────────────────────────────────────────────────────────────

float train_step_regret(MLP& net,
                        const float* states_raw,
                        const float* targets_raw,
                        int batch) {
    // Map raw arrays to Eigen matrices [dim, batch]
    Eigen::Map<const Eigen::MatrixXf> X(states_raw,  INPUT_DIM, batch);
    Eigen::Map<const Eigen::MatrixXf> T(targets_raw, N_ACTIONS, batch);

    auto cache = forward_batch(net, X);
    return backprop_and_update(net, cache, T, nullptr);
}

float train_step_strategy(MLP& net,
                          const float* states_raw,
                          const float* targets_raw,
                          const float* weights_raw,
                          int batch) {
    Eigen::Map<const Eigen::MatrixXf> X(states_raw,  INPUT_DIM, batch);
    Eigen::Map<const Eigen::MatrixXf> T(targets_raw, N_ACTIONS, batch);
    Eigen::RowVectorXf W = Eigen::Map<const Eigen::RowVectorXf>(weights_raw, batch);

    auto cache = forward_batch(net, X);
    return backprop_and_update(net, cache, T, &W);
}

// ── Weight export ─────────────────────────────────────────────────────────────

std::vector<LayerWeights> get_layer_weights(const MLP& net) {
    std::vector<LayerWeights> result;
    for (const auto& l : net.layers) {
        LayerWeights lw;
        lw.W       = l.W.data();
        lw.out_dim = (int)l.W.rows();
        lw.in_dim  = (int)l.W.cols();
        lw.b       = l.b.data();
        result.push_back(lw);
    }
    return result;
}
