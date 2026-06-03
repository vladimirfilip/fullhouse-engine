#include "mccfr.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

// ── GameState ─────────────────────────────────────────────────────────────────

GameState::GameState(int n_players)
    : started_(false), done_(false), n_players_(n_players) {
    std::fill(payoffs_, payoffs_ + N_PLAYERS, 0.0f);
    // state_.action_log is NOT pre-reserved here: state_ is always replaced
    // immediately by a move-assign from start_hand() / apply_action(), so any
    // upfront reservation would be wasted (alloc + free per node expansion).
    // build_state() resizes to the exact action count, allocating once.
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

LegalActions GameState::get_legal_actions() const {
    LegalActions acts;
    if (done_ || !started_) return acts;

    if (state_.amount_owed > 0)
        acts.acts[acts.n++] = FOLD;
    acts.acts[acts.n++] = CHECK_CALL;

    if (state_.your_stack > 0) {
        int cur_bet      = state_.current_bet;
        int min_r        = state_.min_raise_to;
        int my_bet       = state_.your_bet_this_street;
        int all_in_tot   = my_bet + state_.your_stack;
        int eff_pot      = state_.pot + state_.amount_owed;
        int min_raise_by = min_r - cur_bet;

        // Fractional bet sizes only when raise cap not yet hit.
        if (state_.n_raises_this_street < MAX_RAISES_PER_STREET) {
            // Add a bet size only if (a) it doesn't collapse to ALL_IN and
            // (b) it isn't a duplicate of the previous size (two fractions can
            // both clamp to the same min-raise target at shallow effective stacks).
            int last_target = -1;
            auto add_bet = [&](int action_idx, int raise_by_raw) {
                int raise_by = std::max(raise_by_raw, min_raise_by);
                int target   = cur_bet + raise_by;
                if (target < all_in_tot && target != last_target) {
                    last_target = target;
                    acts.acts[acts.n++] = action_idx;
                }
            };

            add_bet(BET_0_27X_POT, (int)std::lround(eff_pot * 0.27f));
            add_bet(BET_THIRD_POT, eff_pot / 3);
            add_bet(BET_HALF_POT,  eff_pot / 2);
            add_bet(BET_FULL_POT,  eff_pot);
            add_bet(BET_1_72X_POT, (int)std::lround(eff_pot * 1.72f));
            add_bet(BET_2X_POT,    eff_pot * 2);
        }
        // ALL_IN is always available when the player has chips, even past the
        // raise cap — committing all chips is not a re-raise in the usual sense.
        acts.acts[acts.n++] = ALL_IN;
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
    // Compute raw action from *this before constructing next, so we don't
    // copy the state_.action_log vector only to immediately overwrite it.
    RawAction raw = abstract_to_raw(action_idx);
    GameState next(n_players_);
    next.engine_  = engine_;
    next.started_ = started_;
    std::copy(payoffs_, payoffs_ + N_PLAYERS, next.payoffs_);
    HandResult result = next.engine_.apply_action(state_.seat_to_act, raw);
    if (result.is_complete) {
        next.done_ = true;
        for (int i = 0; i < N_PLAYERS; i++)
            next.payoffs_[i] = (i < n_players_)
                ? (float)(result.final_stacks[i] - INITIAL_STACK) / INITIAL_STACK : 0.0f;
    } else {
        next.state_ = std::move(result.state);
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
    int n_legal = legal.n;

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
        float inv = 1.0f / pos_sum;
        for (int a : legal) strategy[a] *= inv;
    } else {
        float uniform = 1.0f / n_legal;
        for (int a : legal) strategy[a] = uniform;
    }

    int player = state.current_player();

    // ── Opponent node ──────────────────────────────────────────────────────────
    if (player != traverser) {
        StrategySample ss;
        ss.state  = fvec;
        std::copy(strategy, strategy + N_ACTIONS, ss.strategy);
        ss.weight = std::pow((float)iteration_t, DCFR_ALPHA);  // DCFR discount
        strategy_buf.push_back(ss);

        // strategy[] is already normalised over legal actions (sums to 1.0).
        float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
        int chosen = legal.acts[n_legal - 1];
        float cumul = 0.0f;
        for (int i = 0; i < n_legal; i++) {
            cumul += strategy[legal.acts[i]];
            if (r <= cumul) { chosen = legal.acts[i]; break; }
        }
        return mccfr(state.apply_action(chosen), traverser,
                     regret_net, regret_buf, strategy_buf,
                     iteration_t, depth + 1, rng);
    }

    // ── Traverser node ────────────────────────────────────────────────────────
    // Regret-based pruning (see config.hpp): once the regret net is trained,
    // skip recursion into regret-matching-zeroed actions deep in the regret
    // range. They contribute 0 to node EV (strategy=0, EV stays exact) and keep
    // their carried-forward negative regret target so they stay pruned.
    float max_reg = raw_regrets[legal.acts[0]];
    float min_reg = raw_regrets[legal.acts[0]];
    for (int a : legal) {
        max_reg = std::max(max_reg, raw_regrets[a]);
        min_reg = std::min(min_reg, raw_regrets[a]);
    }
    const bool  prune_on     = iteration_t >= PRUNE_START_ITER;
    const float prune_thresh = max_reg - PRUNE_MARGIN_FRAC * (max_reg - min_reg);

    float action_evs[N_ACTIONS] = {};
    bool  pruned[N_ACTIONS]     = {};
    int   n_kept = n_legal;
    for (int a : legal) {
        if (prune_on && strategy[a] == 0.0f && raw_regrets[a] < prune_thresh
                && n_kept > MIN_TRAVERSE_ACTIONS) {
            pruned[a] = true;
            n_kept--;
        }
    }
    for (int a : legal) {
        if (pruned[a]) continue;
        action_evs[a] = mccfr(state.apply_action(a), traverser,
                               regret_net, regret_buf, strategy_buf,
                               iteration_t, depth + 1, rng);
    }

    // Node EV = sum(strategy[a] * EV[a]) — pruned actions have strategy[a]=0.
    float node_ev = 0.0f;
    for (int a : legal) node_ev += strategy[a] * action_evs[a];

    // Instantaneous regret. For pruned actions we carry forward the net's current
    // (negative) regret estimate rather than a fabricated 0, so they stay pruned.
    RegretSample rs;
    rs.state  = fvec;
    rs.weight = std::pow((float)iteration_t, DCFR_ALPHA);  // DCFR discount
    std::fill(rs.regrets, rs.regrets + N_ACTIONS, 0.0f);
    for (int a : legal)
        rs.regrets[a] = pruned[a] ? raw_regrets[a] : (action_evs[a] - node_ev);
    regret_buf.push_back(rs);

    return node_ev;
}
