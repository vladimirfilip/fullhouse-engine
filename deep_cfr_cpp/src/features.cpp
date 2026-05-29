#include "features.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

// One-hot index for action type: 0=fold, 1=check_or_call, 2=raise, 3=all_in
// Returns -1 for blind actions (caller skips them before this point)
static int action_onehot_idx(ActionType a) {
    switch (a) {
        case ActionType::FOLD:   return 0;
        case ActionType::CHECK:  return 1;
        case ActionType::CALL:   return 1;
        case ActionType::RAISE:  return 2;
        case ActionType::ALL_IN: return 3;
        default:                 return -1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature vector layout (INPUT_DIM = 308):
//   [0:52]    hole cards one-hot
//   [52:104]  board cards one-hot
//   [104:110] hero position rel. dealer (one-hot, 6)
//   [110]     pot / INITIAL_STACK
//   [111:117] per-seat stack / INITIAL_STACK (6)
//   [117:123] per-seat is_folded mask (6)
//   [123:129] per-seat is_all_in mask (6)
//   [129:135] per-seat bet_this_street / INITIAL_STACK (6)
//   [135:139] street one-hot (4)
//   [139]     pot odds = owed / (pot + owed)
//   [140]     SPR log-scaled: log10(spr+1) / log10(101), clamped to [0,1]
//   [141]     amount owed / INITIAL_STACK
//   [142]     n_raises_this_street / MAX_RAISES_PER_STREET
//   [143]     hero bet_this_street / INITIAL_STACK
//   [144]     min effective stack vs opponents / INITIAL_STACK
//   [145:151] last-aggressor seat one-hot (6)
//   [151]     last-aggressor amount / INITIAL_STACK
//   [152:158] last-aggressor position rel. hero one-hot (6)
//   [158:163] board texture: flush-draw, monotone, paired, two-paired, connected (5)
//   [163]     n_active (active+all_in) / N_PLAYERS
//   [164:308] action history 24 slots × 6 floats (seat, 4 action one-hot, amount/INITIAL_STACK)
// Mirror exactly in bots/vlad/bot.py — any drift silently corrupts inference.
// ─────────────────────────────────────────────────────────────────────────────

FeatureVec build_feature_vector(const StateDict& s) {
    FeatureVec vec{};  // zero-initialise

    // ── 1. Hole cards [0:52] ──────────────────────────────────────────────────
    vec[s.your_cards[0]] = 1.0f;
    vec[s.your_cards[1]] = 1.0f;

    // ── 2. Board cards [52:104] ───────────────────────────────────────────────
    for (int c = 0; c < s.n_community; c++)
        vec[52 + s.community_cards[c]] = 1.0f;

    // ── 3. Hero position relative to dealer [104:110] ────────────────────────
    int n_in_game = std::max(s.n_players_seated, 1);
    int hero_pos = ((s.seat_to_act - s.dealer_seat) % n_in_game + n_in_game) % n_in_game;
    vec[104 + hero_pos] = 1.0f;

    // ── 4+5. Pot, stacks, and per-seat status masks [110:135] ────────────────
    vec[110] = (float)s.pot / INITIAL_STACK;
    for (int i = 0; i < s.n_players_seated; i++) {
        const PlayerState& p = s.players[i];
        vec[111 + p.seat] = (float)p.stack / INITIAL_STACK;
        if (p.is_folded) vec[117 + p.seat] = 1.0f;
        if (p.is_all_in) vec[123 + p.seat] = 1.0f;
        vec[129 + p.seat] = (float)p.bet_this_street / INITIAL_STACK;
    }

    // ── 6. Street one-hot [135:139] ───────────────────────────────────────────
    if (s.street >= 0 && s.street < 4)
        vec[135 + s.street] = 1.0f;

    // ── 7. Pot odds, SPR, owed [139:142] ─────────────────────────────────────
    float owed = (float)s.amount_owed;
    float pot  = (float)s.pot;
    vec[139] = owed / std::max(pot + owed, 1.0f);
    // SPR log-scaled — uncapped, unlike the old min(spr,10)/10 which collapsed
    // all deep-stack situations to 1.0.
    float spr = (float)s.your_stack / std::max(pot, 1.0f);
    vec[140] = std::min(std::log10(spr + 1.0f) / std::log10(101.0f), 1.0f);
    vec[141] = owed / INITIAL_STACK;

    // ── 8. Raises this street, hero bet, effective stack, n_active [142:145,163]
    vec[142] = (float)std::min(s.n_raises_this_street, MAX_RAISES_PER_STREET) / std::max(MAX_RAISES_PER_STREET, 1);
    vec[143] = (float)s.your_bet_this_street / INITIAL_STACK;
    int max_opp_stack = 0, n_active = 0;
    for (int i = 0; i < s.n_players_seated; i++) {
        const PlayerState& p = s.players[i];
        if (!p.is_folded) n_active++;
        if (p.seat != s.seat_to_act && !p.is_folded)
            max_opp_stack = std::max(max_opp_stack, p.stack);
    }
    vec[144] = (float)std::min(s.your_stack, max_opp_stack) / INITIAL_STACK;
    vec[163] = (float)n_active / std::max(N_PLAYERS, 1);

    // ── 9. Last aggressor [145:158] ───────────────────────────────────────────
    // ── 10. Board texture [158:163] ───────────────────────────────────────────
    // ── 11. Action history [164:308] ──────────────────────────────────────────
    //
    // Single pass over action_log:
    //   • detects last RAISE / ALL_IN / BIG_BLIND for aggressor features
    //   • fills a 24-slot ring buffer of non-blind entries for history encoding
    //   Both previous loops are merged here, and the old MAX_HAND_ACTIONS-entry
    //   stack pointer array is replaced with a 24-slot ring (176 bytes vs 2400).
    if (s.n_community > 0) {
        int suit_counts[4]  = {};
        int rank_counts[13] = {};
        for (int c = 0; c < s.n_community; c++) {
            suit_counts[card_suit(s.community_cards[c])]++;
            rank_counts[card_rank(s.community_cards[c])]++;
        }
        int max_suit = *std::max_element(suit_counts, suit_counts + 4);
        vec[158] = (max_suit >= 2) ? 1.0f : 0.0f;
        vec[159] = (max_suit >= 3) ? 1.0f : 0.0f;
        int pairs = 0;
        for (int r = 0; r < 13; r++) if (rank_counts[r] >= 2) pairs++;
        vec[160] = (pairs >= 1) ? 1.0f : 0.0f;
        vec[161] = (pairs >= 2) ? 1.0f : 0.0f;
        bool connected = false;
        for (int c1 = 0; c1 < s.n_community && !connected; c1++)
            for (int c2 = c1 + 1; c2 < s.n_community && !connected; c2++)
                if (std::abs(card_rank(s.community_cards[c1]) -
                             card_rank(s.community_cards[c2])) == 1)
                    connected = true;
        vec[162] = connected ? 1.0f : 0.0f;
    }

    // Single pass: find last aggressor + fill ring buffer of last 24 non-blind
    const ActionEntry* ring[24];
    int ring_head = 0, n_seen = 0;
    int last_agg_seat = -1, last_agg_amount = 0;
    for (const auto& e : s.action_log) {
        if (e.action == ActionType::RAISE ||
            e.action == ActionType::ALL_IN ||
            e.action == ActionType::BIG_BLIND) {
            last_agg_seat   = e.seat;
            last_agg_amount = e.amount;
        }
        if (e.action != ActionType::SMALL_BLIND && e.action != ActionType::BIG_BLIND) {
            ring[ring_head] = &e;
            ring_head = (ring_head == 23) ? 0 : ring_head + 1;
            n_seen++;
        }
    }

    if (last_agg_seat >= 0 && last_agg_seat < N_PLAYERS) {
        vec[145 + last_agg_seat] = 1.0f;
        // Normalise by INITIAL_STACK (stable scale) rather than current pot,
        // which already includes this bet and would understate its size.
        vec[151] = (float)last_agg_amount / INITIAL_STACK;
        int rel_pos = ((last_agg_seat - s.seat_to_act) % n_in_game + n_in_game) % n_in_game;
        vec[152 + rel_pos] = 1.0f;
    }

    int n_slots   = n_seen <= 24 ? n_seen : 24;
    int start_pos = n_seen <= 24 ? 0 : ring_head;
    for (int slot = 0; slot < n_slots; slot++) {
        const ActionEntry& e = *ring[(start_pos + slot) % 24];
        int base  = 164 + slot * 6;
        vec[base] = (float)e.seat / std::max(N_PLAYERS - 1, 1);
        int atype = action_onehot_idx(e.action);
        if (atype >= 0) vec[base + 1 + atype] = 1.0f;
        // Normalise by INITIAL_STACK so early small bets are not compressed
        // by the large current pot (which would happen if we divided by pot_now).
        vec[base + 5] = (float)e.amount / INITIAL_STACK;
    }

    return vec;
}
