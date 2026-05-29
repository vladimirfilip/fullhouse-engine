#pragma once
#include <vector>
#include <random>
#include <array>
#include "config.hpp"
#include "engine.hpp"
#include "features.hpp"
#include "network.hpp"

// Samples collected during MCCFR traversal
struct RegretSample {
    FeatureVec state;
    float regrets[N_ACTIONS];
    float weight; // iteration_t — Linear-CFR weighting (matches strategy samples)
};
struct StrategySample {
    FeatureVec state;
    float strategy[N_ACTIONS];
    float weight; // iteration_t
};

// Fixed-size action set — avoids heap allocation on every get_legal_actions() call.
struct LegalActions {
    int  acts[N_ACTIONS];
    int  n = 0;
    int  size()        const { return n; }
    const int* begin() const { return acts; }
    const int* end()   const { return acts + n; }
};

// GameState wraps PokerEngine — one node in the MCCFR game tree.
// Copyable by value for fast branching (no heap allocation beyond action_log vector).
class GameState {
public:
    explicit GameState(int n_players = N_PLAYERS);

    // MCCFR interface
    bool is_terminal()   const;
    bool is_chance_node() const;
    LegalActions get_legal_actions() const;
    GameState sample_chance_event(std::mt19937& rng) const;
    GameState apply_action(int abstract_idx) const;
    int   current_player() const;
    float get_payoff(int player) const; // returns payoff for player index

    const StateDict& state_dict() const { return state_; }

private:
    PokerEngine engine_;
    StateDict   state_;
    bool        started_;
    bool        done_;
    float       payoffs_[N_PLAYERS]; // zero-padded for absent seats
    int         n_players_;

    RawAction abstract_to_raw(int action_idx) const;
};

// External Sampling MCCFR traversal.
// Collects samples into caller-supplied vectors (thread-local in workers).
float mccfr(const GameState& state,
            int traverser,
            const MLP& regret_net,
            std::vector<RegretSample>&   regret_buf,
            std::vector<StrategySample>& strategy_buf,
            int iteration_t,
            int depth,
            std::mt19937& rng);
