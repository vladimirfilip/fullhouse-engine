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
//   [151]     last-aggressor amount / pot
//   [152:158] last-aggressor position rel. hero one-hot (6)
//   [158:163] board texture: flush-draw, monotone, paired, two-paired, connected (5)
//   [163]     n_active (active+all_in) / N_PLAYERS
//   [164:308] action history 24 slots × 6 floats (seat, 4 action one-hot, amount/pot)
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

    // ── 4. Pot and stacks [110:117] ───────────────────────────────────────────
    vec[110] = (float)s.pot / INITIAL_STACK;
    for (const auto& p : s.players) {
        if (p.seat < N_PLAYERS)
            vec[111 + p.seat] = (float)p.stack / INITIAL_STACK;
    }

    // ── 5. Per-seat status masks [117:135] ────────────────────────────────────
    for (const auto& p : s.players) {
        if (p.seat >= N_PLAYERS) continue;
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

    // ── 8. Raises this street, hero bet, effective stack [142:145] ────────────
    vec[142] = (float)s.n_raises_this_street / std::max(MAX_RAISES_PER_STREET, 1);
    vec[143] = (float)s.your_bet_this_street / INITIAL_STACK;
    int max_opp_stack = 0;
    for (const auto& p : s.players) {
        if (p.seat == s.seat_to_act || p.is_folded) continue;
        max_opp_stack = std::max(max_opp_stack, p.stack);
    }
    int eff_stack = std::min(s.your_stack, max_opp_stack);
    vec[144] = (float)eff_stack / INITIAL_STACK;

    // ── 9. Last aggressor [145:158] ───────────────────────────────────────────
    // Defined as the most recent RAISE / ALL_IN / BIG_BLIND in action_log.
    // BIG_BLIND counts as the initial preflop aggressor; once a raise/all-in
    // happens it gets overridden. RAISE/ALL_IN entries on later streets
    // override the preflop big-blind aggressor.
    int last_agg_seat   = -1;
    int last_agg_amount = 0;
    for (const auto& e : s.action_log) {
        if (e.action == ActionType::RAISE ||
            e.action == ActionType::ALL_IN ||
            e.action == ActionType::BIG_BLIND) {
            last_agg_seat   = e.seat;
            last_agg_amount = e.amount;
        }
    }
    if (last_agg_seat >= 0 && last_agg_seat < N_PLAYERS) {
        vec[145 + last_agg_seat] = 1.0f;
        vec[151] = (float)last_agg_amount / std::max(pot, 1.0f);
        int rel_pos = ((last_agg_seat - s.seat_to_act) % n_in_game + n_in_game) % n_in_game;
        vec[152 + rel_pos] = 1.0f;
    }

    // ── 10. Board texture and active count [158:164] ─────────────────────────
    if (s.n_community > 0) {
        int suit_counts[4]  = {};
        int rank_counts[13] = {};
        for (int c = 0; c < s.n_community; c++) {
            suit_counts[card_suit(s.community_cards[c])]++;
            rank_counts[card_rank(s.community_cards[c])]++;
        }
        int max_suit = *std::max_element(suit_counts, suit_counts + 4);
        vec[158] = (max_suit >= 2) ? 1.0f : 0.0f;   // flush draw possible
        vec[159] = (max_suit >= 3) ? 1.0f : 0.0f;   // monotone board
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
    int n_active = 0;
    for (const auto& p : s.players) if (!p.is_folded) n_active++;
    vec[163] = (float)n_active / std::max(N_PLAYERS, 1);

    // ── 11. Action history [164:308] ──────────────────────────────────────────
    // Filter out blinds; take last 24 non-blind entries. 6 floats per slot:
    // seat (1) + action one-hot (4) + amount/pot (1).
    std::vector<const ActionEntry*> regular;
    regular.reserve(s.action_log.size());
    for (const auto& e : s.action_log) {
        if (e.action != ActionType::SMALL_BLIND && e.action != ActionType::BIG_BLIND)
            regular.push_back(&e);
    }
    int start = (int)regular.size() > 24 ? (int)regular.size() - 24 : 0;
    float pot_now = std::max(pot, 1.0f);

    for (int slot = 0; slot < (int)regular.size() - start; slot++) {
        const ActionEntry& e = *regular[start + slot];
        int base     = 164 + slot * 6;
        vec[base]    = (float)e.seat / std::max(N_PLAYERS - 1, 1);
        int atype    = action_onehot_idx(e.action);
        if (atype >= 0) vec[base + 1 + atype] = 1.0f;
        vec[base + 5]= (float)e.amount / pot_now;
    }

    return vec;
}
