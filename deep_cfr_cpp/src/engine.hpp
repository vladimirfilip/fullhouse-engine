#pragma once
#include <string>
#include <vector>
#include <random>
#include <stdexcept>
#include "card.hpp"
#include "config.hpp"

enum class ActionType : uint8_t {
    FOLD, CHECK, CALL, RAISE, ALL_IN, SMALL_BLIND, BIG_BLIND
};

struct ActionEntry {
    int seat;
    int amount;
    ActionType action;
};
static_assert(sizeof(ActionEntry) <= 12, "ActionEntry size");

struct PlayerState {
    int  stack;
    int  bet_this_street;
    int  total_invested;
    Card hole_cards[2];
    int  seat;
    bool is_folded;
    bool is_all_in;
};

struct RawAction {
    ActionType action;
    int amount;
};

// Snapshot returned to the MCCFR algorithm.
// action_log here is a vector (built once per decision point, not on every copy).
struct StateDict {
    int  seat_to_act;
    int  dealer_seat;         // button seat — authoritative source for position calc
    int  pot;
    int  current_bet;
    int  min_raise_to;
    int  amount_owed;
    int  your_stack;
    int  your_bet_this_street;
    int  street;              // 0=preflop,1=flop,2=turn,3=river
    int  n_raises_this_street;
    Card your_cards[2];
    Card community_cards[5];
    int  n_community;
    int  n_players_seated;
    PlayerState players[N_PLAYERS];      // inline — no heap alloc, trivially copyable
    std::vector<ActionEntry> action_log; // pre-reserved; resize() never reallocates
};

struct HandResult {
    bool is_complete = false;
    StateDict state;             // valid when !is_complete
    int final_stacks[N_PLAYERS]; // valid when is_complete
};

// ── PokerEngine ───────────────────────────────────────────────────────────────
// Uses fixed-size arrays (no heap allocation) so copying is a flat memcpy.
// Max actions per hand: 2 blinds + 6 streets×4 betting rounds×6 players = ~74 actions.

static constexpr int MAX_HAND_ACTIONS = 300;

class PokerEngine {
public:
    PokerEngine() = default;
    PokerEngine(const PokerEngine&) = default;
    PokerEngine& operator=(const PokerEngine&) = default;

    void     init(int n_players, int dealer_seat, const Card* shuffled_deck);
    StateDict start_hand();
    HandResult apply_action(int seat, RawAction raw);

private:
    // Fixed layout — no pointers, pure value type, trivially copyable
    PlayerState players_[N_PLAYERS];
    ActionEntry action_log_[MAX_HAND_ACTIONS];
    Card        deck_[52];
    Card        community_[5];
    int         starting_stacks_[N_PLAYERS];
    int         n_;
    int         dealer_seat_;
    int         pot_;
    int         street_;
    int         current_bet_;
    int         min_raise_;
    int         last_aggression_size_;
    int         deck_idx_;
    int         n_community_;
    int         n_log_actions_;
    int         n_raises_this_street_;
    uint8_t     needs_mask_;  // bit i = player i needs to act (6 players max)

    // Needs-to-act helpers (replaces std::set<int>)
    bool needs(int s) const          { return (needs_mask_ >> s) & 1; }
    void needs_set(int s)            { needs_mask_ |= (uint8_t)(1 << s); }
    void needs_clear(int s)          { needs_mask_ &= (uint8_t)~(1 << s); }
    bool needs_any() const           { return needs_mask_ != 0; }
    void needs_clear_all()           { needs_mask_ = 0; }

    // Seat helpers
    int sb_seat() const;
    int bb_seat() const;
    int utg_seat() const;
    int first_postflop_actor() const;
    int next_actor(int from_seat) const;

    // Deck
    Card deal_one();
    void deal_community(int n);
    void deal_hole_cards();

    // Chips
    void put_in(int seat, int amount);
    void post_blinds();

    // Aggression
    void handle_aggression(int seat, int raise_size);

    // Flow
    HandResult advance_if_street_over(int last_seat);
    HandResult advance_street();
    HandResult run_it_out();
    HandResult award_uncontested(int winner_seat);
    HandResult showdown();

    // Serialise
    StateDict  build_state(int seat) const;
    RawAction  validate(int seat, RawAction raw) const;

    int  sole_survivor() const;

    struct SidePot { int amount; int eligible[N_PLAYERS]; int n_eligible; };
    void compute_side_pots(SidePot* out, int* n_out) const;
};
