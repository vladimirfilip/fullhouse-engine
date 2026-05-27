#include "mccfr.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

// ── GameState ─────────────────────────────────────────────────────────────────

GameState::GameState(int n_players)
    : started_(false), done_(false), n_players_(n_players) {
    std::fill(payoffs_, payoffs_ + N_PLAYERS, 0.0f);
}

bool GameState::is_terminal()    const { return done_; }
bool GameState::is_chance_node() const { return !started_; }

int GameState::current_player() const {
    if (done_ || !started_) return -1;
    return state_.seat_to_act;
}

float GameState::get_payoff(int player) const {
    if (player < 0 || player >= N_PLAYERS) return 0.0f;
    return payoffs_[player];
}

std::vector<int> GameState::get_legal_actions() const {
    if (done_ || !started_) return {};
    std::vector<int> acts;
    acts.reserve(N_ACTIONS);
    // Insert in sorted order so callers get a sorted list without a sort pass.
    if (state_.amount_owed > 0)
        acts.push_back(FOLD);           // 0
    acts.push_back(CHECK_CALL);         // 1
    if (state_.your_stack > 0 && state_.n_raises_this_street < MAX_RAISES_PER_STREET) {
        acts.push_back(BET_0_27X_POT);  // 2
        acts.push_back(BET_THIRD_POT);  // 3
        acts.push_back(BET_HALF_POT);   // 4
        acts.push_back(BET_FULL_POT);   // 5
        acts.push_back(BET_1_72X_POT);  // 6
        acts.push_back(BET_2X_POT);     // 7
        acts.push_back(ALL_IN);         // 8
    }
    return acts;
}

GameState GameState::sample_chance_event(std::mt19937& rng) const {
    Card deck[52];
    build_shuffled_deck(deck, rng);

    int dealer_seat = std::uniform_int_distribution<int>(0, n_players_ - 1)(rng);

    GameState next(n_players_);
    next.engine_.init(n_players_, dealer_seat, deck);
    next.state_   = next.engine_.start_hand();
    next.started_ = true;
    return next;
}

GameState GameState::apply_action(int action_idx) const {
    GameState next = *this;  // copy by value
    RawAction raw  = next.abstract_to_raw(action_idx);
    HandResult result = next.engine_.apply_action(next.state_.seat_to_act, raw);

    if (result.is_complete) {
        next.done_    = true;
        next.state_   = {};
        // Payoffs: (final_stack - initial_stack). Absent seats → 0.
        for (int i = 0; i < N_PLAYERS; i++) {
            if (i < n_players_)
                next.payoffs_[i] = (float)(result.final_stacks[i] - INITIAL_STACK);
            else
                next.payoffs_[i] = 0.0f;
        }
    } else {
        next.state_ = result.state;
    }
    return next;
}

// ── Abstract action → raw action (mirrors env.py _abstract_to_raw) ────────────

RawAction GameState::abstract_to_raw(int action_idx) const {
    const StateDict& sd = state_;
    int pot      = sd.pot;
    int owed     = sd.amount_owed;
    int cur_bet  = sd.current_bet;
    int min_r    = sd.min_raise_to;   // already = current_bet + min_raise
    int stack    = sd.your_stack;
    int my_bet   = sd.your_bet_this_street;
    int all_in_tot = my_bet + stack;
    int eff_pot    = pot + owed;       // pot if we call

    auto bet_action = [&](int target) -> RawAction {
        if (target >= all_in_tot) return {ActionType::ALL_IN, 0};
        return {ActionType::RAISE, target};
    };

    if (action_idx == FOLD)      return {ActionType::FOLD, 0};
    if (action_idx == CHECK_CALL)
        return (owed == 0) ? RawAction{ActionType::CHECK, 0}
                           : RawAction{ActionType::CALL,  owed};

    if (action_idx == BET_0_27X_POT) {
        // lround (not truncation) — odd pot sizes were biased low.
        int raise_by = std::max((int)std::lround(eff_pot * 0.27f), min_r - cur_bet);
        return bet_action(cur_bet + raise_by);
    }
    if (action_idx == BET_THIRD_POT) {
        int raise_by = std::max(eff_pot / 3, min_r - cur_bet);
        return bet_action(cur_bet + raise_by);
    }
    if (action_idx == BET_HALF_POT) {
        int raise_by = std::max(eff_pot / 2, min_r - cur_bet);
        return bet_action(cur_bet + raise_by);
    }
    if (action_idx == BET_FULL_POT) {
        int raise_by = std::max(eff_pot, min_r - cur_bet);
        return bet_action(cur_bet + raise_by);
    }
    if (action_idx == BET_1_72X_POT) {
        int raise_by = std::max((int)std::lround(eff_pot * 1.72f), min_r - cur_bet);
        return bet_action(cur_bet + raise_by);
    }
    if (action_idx == BET_2X_POT) {
        int raise_by = std::max(eff_pot * 2, min_r - cur_bet);
        return bet_action(cur_bet + raise_by);
    }
    if (action_idx == ALL_IN)    return {ActionType::ALL_IN, 0};

    return {ActionType::FOLD, 0};  // fallback
}

// ── External Sampling MCCFR ────────────────────────────────────────────────────

float mccfr(const GameState& state,
            int traverser,
            const MLP& regret_net,
            std::vector<RegretSample>&   regret_buf,
            std::vector<StrategySample>& strategy_buf,
            int iteration_t,
            int depth,
            std::mt19937& rng) {

    // ── Terminal ──────────────────────────────────────────────────────────────
    if (state.is_terminal() || depth >= MAX_DEPTH)
        return state.get_payoff(traverser);

    // ── Chance node ───────────────────────────────────────────────────────────
    if (state.is_chance_node())
        return mccfr(state.sample_chance_event(rng), traverser,
                     regret_net, regret_buf, strategy_buf,
                     iteration_t, depth + 1, rng);

    auto legal  = state.get_legal_actions();
    int n_legal = (int)legal.size();

    // ── Feature vector and network forward pass ────────────────────────────────
    FeatureVec fvec = build_feature_vector(state.state_dict());
    auto raw_regrets = forward_single(regret_net, fvec);

    // ── Regret matching ───────────────────────────────────────────────────────
    float strategy[N_ACTIONS] = {};
    float pos_sum = 0.0f;
    for (int a : legal) {
        float v = std::max(raw_regrets[a], 0.0f);
        strategy[a] = v;
        pos_sum += v;
    }
    if (pos_sum > 1e-12f) {
        for (int a : legal) strategy[a] /= pos_sum;
    } else {
        float uniform = 1.0f / n_legal;
        for (int a : legal) strategy[a] = uniform;
    }

    int player = state.current_player();

    // ── Opponent node ──────────────────────────────────────────────────────────
    if (player != traverser) {
        // Record strategy sample
        StrategySample ss;
        ss.state  = fvec;
        std::copy(strategy, strategy + N_ACTIONS, ss.strategy);
        ss.weight = (float)iteration_t;
        strategy_buf.push_back(ss);

        // Sample one action proportional to strategy
        float prob[N_ACTIONS] = {};
        float psum = 0.0f;
        for (int a : legal) { prob[a] = strategy[a]; psum += prob[a]; }
        float r = std::uniform_real_distribution<float>(0.0f, psum)(rng);
        int chosen = legal.back();
        float cumul = 0.0f;
        for (int a : legal) {
            cumul += prob[a];
            if (r <= cumul) { chosen = a; break; }
        }
        return mccfr(state.apply_action(chosen), traverser,
                     regret_net, regret_buf, strategy_buf,
                     iteration_t, depth + 1, rng);
    }

    // ── Traverser node ────────────────────────────────────────────────────────
    float action_evs[N_ACTIONS] = {};
    for (int a : legal) {
        action_evs[a] = mccfr(state.apply_action(a), traverser,
                               regret_net, regret_buf, strategy_buf,
                               iteration_t, depth + 1, rng);
    }

    // Node EV = sum(strategy[a] * EV[a])
    float node_ev = 0.0f;
    for (int a : legal) node_ev += strategy[a] * action_evs[a];

    // Instantaneous regret
    RegretSample rs;
    rs.state = fvec;
    std::fill(rs.regrets, rs.regrets + N_ACTIONS, 0.0f);
    for (int a : legal) rs.regrets[a] = action_evs[a] - node_ev;
    regret_buf.push_back(rs);

    return node_ev;
}
