#include "engine.hpp"
#include "hand_eval.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>

// ── Init ─────────────────────────────────────────────────────────────────────

void PokerEngine::init(int n_players, int dealer_seat, const Card* shuffled_deck) {
    n_           = n_players;
    dealer_seat_ = dealer_seat % n_players;
    pot_         = 0;
    street_      = 0;
    current_bet_ = 0;
    min_raise_   = BIG_BLIND;
    last_aggression_size_ = BIG_BLIND;
    n_community_ = 0;
    deck_idx_    = 0;
    n_log_actions_ = 0;
    n_raises_this_street_ = 0;
    needs_clear_all();

    std::copy(shuffled_deck, shuffled_deck + 52, deck_);

    for (int i = 0; i < n_; i++) {
        players_[i].seat            = i;
        players_[i].stack           = INITIAL_STACK;
        players_[i].is_folded       = false;
        players_[i].is_all_in       = false;
        players_[i].bet_this_street = 0;
        players_[i].total_invested  = 0;
        starting_stacks_[i]         = INITIAL_STACK;
    }
}

// ── Deck helpers ──────────────────────────────────────────────────────────────

Card PokerEngine::deal_one() { return deck_[deck_idx_++]; }

void PokerEngine::deal_community(int n) {
    for (int i = 0; i < n; i++) community_[n_community_++] = deal_one();
}

void PokerEngine::deal_hole_cards() {
    for (int i = 0; i < n_; i++) {
        players_[i].hole_cards[0] = deal_one();
        players_[i].hole_cards[1] = deal_one();
    }
}

// ── Seat helpers ──────────────────────────────────────────────────────────────

int PokerEngine::sb_seat() const {
    return (n_ == 2) ? dealer_seat_ : (dealer_seat_ + 1) % n_;
}
int PokerEngine::bb_seat() const {
    return (n_ == 2) ? (dealer_seat_ + 1) % n_ : (dealer_seat_ + 2) % n_;
}
int PokerEngine::utg_seat() const {
    if (n_ == 2) return sb_seat();
    int bb = bb_seat();
    for (int offset = 1; offset <= n_; offset++) {
        int s = (bb + offset) % n_;
        if (!players_[s].is_folded && !players_[s].is_all_in && players_[s].stack > 0)
            return s;
    }
    return bb;
}
int PokerEngine::first_postflop_actor() const {
    if (n_ == 2) {
        int seats[2] = {bb_seat(), sb_seat()};
        for (int seat : seats)
            if (!players_[seat].is_folded && !players_[seat].is_all_in && players_[seat].stack > 0)
                return seat;
        return -1;
    }
    for (int offset = 1; offset <= n_; offset++) {
        int s = (dealer_seat_ + offset) % n_;
        if (!players_[s].is_folded && !players_[s].is_all_in && players_[s].stack > 0)
            return s;
    }
    return -1;
}
int PokerEngine::next_actor(int from_seat) const {
    if (!needs_any()) return -1;
    for (int offset = 1; offset <= n_; offset++) {
        int s = (from_seat + offset) % n_;
        if (needs(s) && !players_[s].is_folded && !players_[s].is_all_in && players_[s].stack > 0)
            return s;
    }
    return -1;
}

// ── Chips ─────────────────────────────────────────────────────────────────────

void PokerEngine::put_in(int seat, int amount) {
    amount = std::max(0, std::min(amount, players_[seat].stack));
    players_[seat].stack           -= amount;
    players_[seat].bet_this_street += amount;
    players_[seat].total_invested  += amount;
    pot_                           += amount;
    if (players_[seat].bet_this_street > current_bet_)
        current_bet_ = players_[seat].bet_this_street;
}

// ── Post blinds ───────────────────────────────────────────────────────────────

void PokerEngine::post_blinds() {
    int sb = sb_seat(), bb = bb_seat();
    int sb_amt = std::min(SMALL_BLIND, players_[sb].stack);
    int bb_amt = std::min(BIG_BLIND,   players_[bb].stack);
    put_in(sb, sb_amt);
    put_in(bb, bb_amt);
    current_bet_          = std::max(current_bet_, bb_amt);
    min_raise_            = BIG_BLIND;
    last_aggression_size_ = BIG_BLIND;
    if (players_[sb].stack == 0) players_[sb].is_all_in = true;
    if (players_[bb].stack == 0) players_[bb].is_all_in = true;
    action_log_[n_log_actions_++] = {sb, sb_amt, ActionType::SMALL_BLIND};
    action_log_[n_log_actions_++] = {bb, bb_amt, ActionType::BIG_BLIND};
}

// ── start_hand ────────────────────────────────────────────────────────────────

StateDict PokerEngine::start_hand() {
    post_blinds();
    deal_hole_cards();
    for (int i = 0; i < n_; i++)
        if (!players_[i].is_folded && !players_[i].is_all_in && players_[i].stack > 0)
            needs_set(i);
    return build_state(utg_seat());
}

// ── Validate action ───────────────────────────────────────────────────────────

RawAction PokerEngine::validate(int seat, RawAction raw) const {
    const PlayerState& p = players_[seat];
    int owed = current_bet_ - p.bet_this_street;

    if (raw.action == ActionType::CHECK || raw.action == ActionType::CALL) {
        if (owed == 0) return {ActionType::CHECK, 0};
        return {ActionType::CALL, owed};
    }
    if (raw.action == ActionType::RAISE) {
        int min_total    = current_bet_ + min_raise_;
        int amount       = std::max(raw.amount, min_total);
        int chips_needed = amount - p.bet_this_street;
        if (chips_needed >= p.stack)
            return {ActionType::ALL_IN, p.stack + p.bet_this_street};
        return {ActionType::RAISE, amount};
    }
    if (raw.action == ActionType::ALL_IN)
        return {ActionType::ALL_IN, p.stack + p.bet_this_street};
    return {ActionType::FOLD, 0};
}

// ── Aggression ────────────────────────────────────────────────────────────────

void PokerEngine::handle_aggression(int seat, int raise_size) {
    if (raise_size >= last_aggression_size_) {
        last_aggression_size_ = raise_size;
        min_raise_            = raise_size;
        n_raises_this_street_++;
        needs_clear_all();
        for (int i = 0; i < n_; i++)
            if (i != seat && !players_[i].is_folded && !players_[i].is_all_in && players_[i].stack > 0)
                needs_set(i);
    }
}

// ── apply_action ─────────────────────────────────────────────────────────────

HandResult PokerEngine::apply_action(int seat, RawAction raw) {
    RawAction action = validate(seat, raw);

    if (n_log_actions_ < MAX_HAND_ACTIONS)
        action_log_[n_log_actions_] = {seat, action.amount, action.action};
    n_log_actions_++;
    needs_clear(seat);

    PlayerState& p = players_[seat];

    if (action.action == ActionType::FOLD) {
        p.is_folded = true;

    } else if (action.action == ActionType::CHECK) {
        // no chips move

    } else if (action.action == ActionType::CALL) {
        int owed = current_bet_ - p.bet_this_street;
        int paid = std::min(owed, p.stack);
        put_in(seat, paid);
        if (p.stack == 0) p.is_all_in = true;

    } else if (action.action == ActionType::RAISE) {
        int prev_bet = current_bet_;
        int chips_in = action.amount - p.bet_this_street;
        put_in(seat, std::min(chips_in, p.stack));
        if (p.stack == 0) p.is_all_in = true;
        handle_aggression(seat, current_bet_ - prev_bet);

    } else if (action.action == ActionType::ALL_IN) {
        int prev_bet = current_bet_;
        put_in(seat, p.stack);
        p.is_all_in  = true;
        handle_aggression(seat, current_bet_ - prev_bet);
    }

    int surv = sole_survivor();
    if (surv >= 0) return award_uncontested(surv);
    return advance_if_street_over(seat);
}

// ── Flow ──────────────────────────────────────────────────────────────────────

HandResult PokerEngine::advance_if_street_over(int last_seat) {
    int nxt = next_actor(last_seat);
    if (nxt >= 0) {
        HandResult r; r.is_complete = false;
        r.state = build_state(nxt);
        return r;
    }
    return advance_street();
}

HandResult PokerEngine::advance_street() {
    for (int i = 0; i < n_; i++) players_[i].bet_this_street = 0;
    current_bet_          = 0;
    min_raise_            = BIG_BLIND;
    last_aggression_size_ = BIG_BLIND;
    n_raises_this_street_ = 0;

    if      (street_ == 0) { deal_community(3); street_ = 1; }
    else if (street_ == 1) { deal_community(1); street_ = 2; }
    else if (street_ == 2) { deal_community(1); street_ = 3; }
    else                    { return showdown(); }

    int first = first_postflop_actor();
    if (first < 0) return run_it_out();

    needs_clear_all();
    for (int i = 0; i < n_; i++)
        if (!players_[i].is_folded && !players_[i].is_all_in && players_[i].stack > 0)
            needs_set(i);

    HandResult r; r.is_complete = false;
    r.state = build_state(first);
    return r;
}

HandResult PokerEngine::run_it_out() {
    if (street_ == 0) { deal_community(3); street_ = 1; }
    if (street_ == 1) { deal_community(1); street_ = 2; }
    if (street_ == 2) { deal_community(1); street_ = 3; }
    return showdown();
}

HandResult PokerEngine::award_uncontested(int winner_seat) {
    players_[winner_seat].stack += pot_;
    HandResult r; r.is_complete = true;
    for (int i = 0; i < N_PLAYERS; i++)
        r.final_stacks[i] = (i < n_) ? players_[i].stack : 0;
    return r;
}

// ── Side pots ─────────────────────────────────────────────────────────────────

void PokerEngine::compute_side_pots(SidePot* out, int* n_out) const {
    // Collect unique total_invested levels
    int levels[N_PLAYERS];
    int n_levels = 0;
    for (int i = 0; i < n_; i++)
        if (players_[i].total_invested > 0)
            levels[n_levels++] = players_[i].total_invested;
    std::sort(levels, levels + n_levels);
    n_levels = (int)(std::unique(levels, levels + n_levels) - levels);

    *n_out = 0;
    int prev = 0;
    for (int li = 0; li < n_levels; li++) {
        int lvl = levels[li];
        int per_player = lvl - prev;
        int contributors = 0;
        for (int i = 0; i < n_; i++)
            if (players_[i].total_invested >= lvl) contributors++;
        int pot_amount = per_player * contributors;

        SidePot& sp = out[*n_out];
        sp.amount    = pot_amount;
        sp.n_eligible = 0;
        for (int i = 0; i < n_; i++)
            if (!players_[i].is_folded && players_[i].total_invested >= lvl)
                sp.eligible[sp.n_eligible++] = i;
        if (pot_amount > 0 && sp.n_eligible > 0)
            (*n_out)++;
        prev = lvl;
    }

    if (*n_out == 0) {
        SidePot& sp = out[0];
        sp.amount    = pot_;
        sp.n_eligible = 0;
        for (int i = 0; i < n_; i++)
            if (!players_[i].is_folded)
                sp.eligible[sp.n_eligible++] = i;
        *n_out = 1;
        return;
    }

    int total = 0;
    for (int i = 0; i < *n_out; i++) total += out[i].amount;
    if (total != pot_) out[*n_out - 1].amount += pot_ - total;
}

// ── Showdown ──────────────────────────────────────────────────────────────────

HandResult PokerEngine::showdown() {
    SidePot side_pots[N_PLAYERS];
    int n_pots = 0;
    compute_side_pots(side_pots, &n_pots);

    for (int pi = 0; pi < n_pots; pi++) {
        SidePot& sp = side_pots[pi];
        if (sp.n_eligible == 0) continue;

        // Evaluate each eligible hand once and cache the score.
        uint32_t scores[N_PLAYERS];
        uint32_t best_score = 0;
        for (int ei = 0; ei < sp.n_eligible; ei++) {
            int s = sp.eligible[ei];
            Card hand7[7];
            hand7[0] = players_[s].hole_cards[0];
            hand7[1] = players_[s].hole_cards[1];
            for (int c = 0; c < n_community_; c++) hand7[2 + c] = community_[c];
            scores[ei] = evaluate_7(hand7);
            if (scores[ei] > best_score) best_score = scores[ei];
        }

        int winners[N_PLAYERS];
        int n_winners = 0;
        for (int ei = 0; ei < sp.n_eligible; ei++) {
            if (scores[ei] == best_score) winners[n_winners++] = sp.eligible[ei];
        }

        int split     = sp.amount / n_winners;
        int remainder = sp.amount % n_winners;
        for (int wi = 0; wi < n_winners; wi++)
            players_[winners[wi]].stack += split + (wi == 0 ? remainder : 0);
    }

#ifndef NDEBUG
    int total_start = 0, total_now = 0;
    for (int i = 0; i < n_; i++) { total_start += starting_stacks_[i]; total_now += players_[i].stack; }
    assert(total_now == total_start);
#endif

    HandResult r; r.is_complete = true;
    for (int i = 0; i < N_PLAYERS; i++)
        r.final_stacks[i] = (i < n_) ? players_[i].stack : 0;
    return r;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

int PokerEngine::sole_survivor() const {
    int last = -1, cnt = 0;
    for (int i = 0; i < n_; i++) if (!players_[i].is_folded) { last = i; cnt++; }
    return (cnt == 1) ? last : -1;
}

StateDict PokerEngine::build_state(int seat) const {
    StateDict s;
    s.seat_to_act          = seat;
    s.dealer_seat          = dealer_seat_;
    s.pot                  = pot_;
    s.current_bet          = current_bet_;
    s.street               = street_;
    s.amount_owed          = std::max(0, current_bet_ - players_[seat].bet_this_street);
    s.min_raise_to         = current_bet_ + min_raise_;
    s.your_stack           = players_[seat].stack;
    s.your_bet_this_street = players_[seat].bet_this_street;
    s.your_cards[0]        = players_[seat].hole_cards[0];
    s.your_cards[1]        = players_[seat].hole_cards[1];
    s.n_community          = n_community_;
    for (int c = 0; c < n_community_; c++) s.community_cards[c] = community_[c];
    s.n_players_seated     = n_;
    s.n_raises_this_street = n_raises_this_street_;
    for (int i = 0; i < n_; i++) s.players[i] = players_[i];
    int log_entries = std::min(n_log_actions_, MAX_HAND_ACTIONS);
    s.action_log.resize(log_entries);  // never reallocates: GameState ctor pre-reserves
    for (int i = 0; i < log_entries; i++) s.action_log[i] = action_log_[i];
    return s;
}
