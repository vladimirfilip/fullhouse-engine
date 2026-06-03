#define _CRT_SECURE_NO_WARNINGS
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include "train.hpp"
#include "network.hpp"
#include "config.hpp"
#include "features.hpp"
#include "engine.hpp"
#include "card.hpp"

namespace py = pybind11;
using farray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// ── Feature-parity helper ───────────────────────────────────────────────────
// Build the INPUT_DIM feature vector from a Python game-state dict in the SAME
// format bots/the_house/bot.py consumes, so a test can assert the C++ producer
// and the Python production mirror agree byte-for-byte. dealer_seat and
// n_raises_this_street are passed explicitly (the bot derives them from the log
// via shared, unchanged helpers — this isolates the feature LAYOUT parity).
static ActionType _action_from_str(const std::string& a) {
    if (a == "fold")        return ActionType::FOLD;
    if (a == "check")       return ActionType::CHECK;
    if (a == "call")        return ActionType::CALL;
    if (a == "raise")       return ActionType::RAISE;
    if (a == "all_in")      return ActionType::ALL_IN;
    if (a == "small_blind") return ActionType::SMALL_BLIND;
    if (a == "big_blind")   return ActionType::BIG_BLIND;
    throw std::runtime_error("bad action: " + a);
}

static farray build_features_py(py::dict d) {
    StateDict s{};
    s.seat_to_act          = d["seat_to_act"].cast<int>();
    s.dealer_seat          = d["dealer_seat"].cast<int>();
    s.pot                  = d["pot"].cast<int>();
    s.current_bet          = d.contains("current_bet") ? d["current_bet"].cast<int>() : 0;
    s.min_raise_to         = d.contains("min_raise_to") ? d["min_raise_to"].cast<int>() : 0;
    s.amount_owed          = d["amount_owed"].cast<int>();
    s.your_stack           = d["your_stack"].cast<int>();
    s.your_bet_this_street = d["your_bet_this_street"].cast<int>();
    s.n_raises_this_street = d["n_raises_this_street"].cast<int>();
    std::string st = d["street"].cast<std::string>();
    s.street = (st == "preflop") ? 0 : (st == "flop") ? 1 : (st == "turn") ? 2 : 3;

    auto yc = d["your_cards"].cast<std::vector<std::string>>();
    s.your_cards[0] = card_from_str(yc[0]);
    s.your_cards[1] = card_from_str(yc[1]);
    auto cc = d["community_cards"].cast<std::vector<std::string>>();
    s.n_community = (int)cc.size();
    for (int i = 0; i < s.n_community; i++) s.community_cards[i] = card_from_str(cc[i]);

    auto players = d["players"].cast<std::vector<py::dict>>();
    s.n_players_seated = (int)players.size();
    for (int i = 0; i < s.n_players_seated; i++) {
        py::dict p = players[i];
        PlayerState ps{};
        ps.seat            = p["seat"].cast<int>();
        ps.stack           = p["stack"].cast<int>();
        ps.bet_this_street = p["bet_this_street"].cast<int>();
        ps.is_folded       = p["is_folded"].cast<bool>();
        ps.is_all_in       = p["is_all_in"].cast<bool>();
        s.players[i] = ps;
    }

    for (auto item : d["action_log"].cast<std::vector<py::dict>>()) {
        ActionEntry ae{};
        ae.seat   = item["seat"].cast<int>();
        py::object amt = item["amount"];
        ae.amount = amt.is_none() ? 0 : amt.cast<int>();
        ae.action = _action_from_str(item["action"].cast<std::string>());
        s.action_log.push_back(ae);
    }

    FeatureVec fv = build_feature_vector(s);
    farray out((py::ssize_t)INPUT_DIM);
    std::memcpy(out.mutable_data(), fv.data(), INPUT_DIM * sizeof(float));
    return out;
}

// Reconstruct MLP from flat list of numpy arrays [W0, b0, W1, b1, ...].
// W_i shape: [out_dim, in_dim] row-major (matches PyTorch param.numpy() layout).
static MLP mlp_from_weights(const std::vector<farray>& arrs) {
    // Use inference-only net (no Adam moment matrices) — saves ~7.6 MB per
    // generate_and_add() call and eliminates associated heap fragmentation.
    MLP net = make_inference_net(false);
    int n_layers = (int)net.layers.size();
    if ((int)arrs.size() < n_layers * 2)
        throw std::runtime_error("Not enough weight arrays for this network architecture");

    for (int li = 0; li < n_layers; li++) {
        auto W_info = arrs[li * 2].request();
        auto b_info = arrs[li * 2 + 1].request();

        int out_dim = (int)net.layers[li].W.rows();
        int in_dim  = (int)net.layers[li].W.cols();

        net.layers[li].W = Eigen::Map<const Eigen::Matrix<float,
            Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            static_cast<const float*>(W_info.ptr), out_dim, in_dim);

        net.layers[li].b = Eigen::Map<const Eigen::VectorXf>(
            static_cast<const float*>(b_info.ptr), out_dim);
    }
    return net;
}

// ── DeepCFRBuffers ────────────────────────────────────────────────────────────
// Owns the reservoir buffers for the full training run. Python creates one
// instance and calls generate_and_add each iteration, then sample_*() inside
// the training loop. No large intermediate arrays cross the Python/C++ boundary.

class DeepCFRBuffers {
    ReservoirBuffer<RegretSample>   regret_buf_;
    ReservoirBuffer<StrategySample> strategy_buf_;

public:
    DeepCFRBuffers(int regret_cap, int strategy_cap)
        : regret_buf_(regret_cap), strategy_buf_(strategy_cap) {}

    // Generate n_games MCCFR traversals and add samples directly to the
    // internal reservoir buffers. No large intermediate allocation.
    void generate_and_add(int n_games, int n_workers, int iteration_t,
                          const std::vector<farray>& weights) {
        MLP net = mlp_from_weights(weights);
        parallel_generate(n_games, net, regret_buf_, strategy_buf_,
                          iteration_t, n_workers);
    }

    // Returns (states [B, INPUT_DIM], targets [B, N_ACTIONS])
    py::tuple sample_regret(int batch_size) {
        auto batch = regret_buf_.sample(batch_size);
        int k = (int)batch.size();

        auto states  = py::array_t<float>({k, INPUT_DIM});
        auto targets = py::array_t<float>({k, N_ACTIONS});
        auto rs = states .mutable_unchecked<2>();
        auto rt = targets.mutable_unchecked<2>();

        for (int i = 0; i < k; i++) {
            for (int j = 0; j < INPUT_DIM; j++) rs(i, j) = batch[i].state[j];
            for (int j = 0; j < N_ACTIONS;  j++) rt(i, j) = batch[i].regrets[j];
        }
        return py::make_tuple(states, targets);
    }

    // Returns (states [B, INPUT_DIM], targets [B, N_ACTIONS], weights [B])
    py::tuple sample_strategy(int batch_size) {
        auto batch = strategy_buf_.sample(batch_size);
        int k = (int)batch.size();

        auto states  = py::array_t<float>({k, INPUT_DIM});
        auto targets = py::array_t<float>({k, N_ACTIONS});
        auto weights = py::array_t<float>({k});
        auto ss = states .mutable_unchecked<2>();
        auto st = targets.mutable_unchecked<2>();
        auto sw = weights.mutable_unchecked<1>();

        for (int i = 0; i < k; i++) {
            for (int j = 0; j < INPUT_DIM; j++) ss(i, j) = batch[i].state[j];
            for (int j = 0; j < N_ACTIONS;  j++) st(i, j) = batch[i].strategy[j];
            sw(i) = batch[i].weight;
        }
        return py::make_tuple(states, targets, weights);
    }

    // Write a sampled batch into pre-allocated caller-owned arrays.
    // Releases the GIL for the entire sampling + copy, enabling true background
    // prefetch from Python: submit this in a ThreadPoolExecutor while the main
    // thread runs PyTorch forward/backward on the previous batch.
    // Returns the actual number of rows written (≤ states_out.shape(0)).
    // Fill pre-allocated arrays in-place. Releases GIL during sampling.
    // Returns actual rows written (≤ states_out.shape[0]).
    // weights_out receives the per-sample iteration_t for Linear-CFR weighting.
    int sample_regret_into(farray states_out, farray targets_out, farray weights_out) {
        auto s_info = states_out.request();
        auto t_info = targets_out.request();
        auto w_info = weights_out.request();
        float* s = static_cast<float*>(s_info.ptr);
        float* t = static_cast<float*>(t_info.ptr);
        float* w = static_cast<float*>(w_info.ptr);
        int batch_size = (int)s_info.shape[0];
        {
            py::gil_scoped_release release;
            auto batch = regret_buf_.sample(batch_size);
            int k = (int)batch.size();
            for (int i = 0; i < k; i++) {
                std::memcpy(s + (size_t)i * INPUT_DIM,
                            batch[i].state.data(), INPUT_DIM * sizeof(float));
                std::memcpy(t + (size_t)i * N_ACTIONS,
                            batch[i].regrets,     N_ACTIONS * sizeof(float));
                w[i] = batch[i].weight;
            }
            return k;
        }
    }

    int sample_strategy_into(farray states_out, farray targets_out, farray weights_out) {
        auto s_info = states_out.request();
        auto t_info = targets_out.request();
        auto w_info = weights_out.request();
        float* s = static_cast<float*>(s_info.ptr);
        float* t = static_cast<float*>(t_info.ptr);
        float* w = static_cast<float*>(w_info.ptr);
        int batch_size = (int)s_info.shape[0];
        {
            py::gil_scoped_release release;
            auto batch = strategy_buf_.sample(batch_size);
            int k = (int)batch.size();
            for (int i = 0; i < k; i++) {
                std::memcpy(s + (size_t)i * INPUT_DIM,
                            batch[i].state.data(), INPUT_DIM * sizeof(float));
                std::memcpy(t + (size_t)i * N_ACTIONS,
                            batch[i].strategy,    N_ACTIONS * sizeof(float));
                w[i] = batch[i].weight;
            }
            return k;
        }
    }

    int  regret_size()            const { return regret_buf_.size(); }
    int  strategy_size()          const { return strategy_buf_.size(); }
    bool regret_ready(int min)    const { return regret_buf_.is_ready(min); }
    bool strategy_ready(int min)  const { return strategy_buf_.is_ready(min); }
    void clear() { regret_buf_.clear(); strategy_buf_.clear(); }
};

PYBIND11_MODULE(deep_cfr_gen, m) {
    m.doc() = "C++ MCCFR data generation for Deep CFR (pybind11 binding)";
    m.attr("INPUT_DIM")  = INPUT_DIM;
    m.attr("N_ACTIONS")  = N_ACTIONS;
    m.attr("HIDDEN_DIM") = HIDDEN_DIM;
    m.attr("N_LAYERS")   = N_LAYERS;

    m.def("build_features", &build_features_py, py::arg("state"),
          "Build the INPUT_DIM feature vector from a game-state dict (feature "
          "parity check vs bots/the_house/bot.py._build_feature_vector).");

    py::class_<DeepCFRBuffers>(m, "DeepCFRBuffers")
        .def(py::init<int, int>(),
             py::arg("regret_cap"), py::arg("strategy_cap"),
             "Create reservoir buffers. Call generate_and_add() each iteration, "
             "then sample_*() inside the training loop.")
        .def("generate_and_add", &DeepCFRBuffers::generate_and_add,
             py::arg("n_games"), py::arg("n_workers"), py::arg("iteration_t"),
             py::arg("weights"),
             "Run parallel MCCFR and add samples directly to internal reservoirs.")
        .def("sample_regret", &DeepCFRBuffers::sample_regret,
             py::arg("batch_size"),
             "Sample a training batch. Returns (states [B,308], targets [B,9]).")
        .def("sample_strategy", &DeepCFRBuffers::sample_strategy,
             py::arg("batch_size"),
             "Sample a training batch. Returns (states, targets [B,9], weights [B]).")
        .def("sample_regret_into", &DeepCFRBuffers::sample_regret_into,
             py::arg("states_out"), py::arg("targets_out"), py::arg("weights_out"),
             "Fill pre-allocated arrays in-place. Releases GIL during sampling. "
             "weights_out receives per-sample iteration_t for Linear-CFR weighting. "
             "Returns actual rows written (≤ states_out.shape[0]).")
        .def("sample_strategy_into", &DeepCFRBuffers::sample_strategy_into,
             py::arg("states_out"), py::arg("targets_out"), py::arg("weights_out"),
             "Fill pre-allocated arrays in-place. Releases GIL during sampling. "
             "Returns actual rows written (≤ states_out.shape[0]).")
        .def("regret_size",     &DeepCFRBuffers::regret_size)
        .def("strategy_size",   &DeepCFRBuffers::strategy_size)
        .def("regret_ready",    &DeepCFRBuffers::regret_ready,   py::arg("min_size"))
        .def("strategy_ready",  &DeepCFRBuffers::strategy_ready, py::arg("min_size"))
        .def("clear",           &DeepCFRBuffers::clear,
             "Reset both buffers (call before starting a fresh training run).");
}
