#include "hand_eval.hpp"
#include <algorithm>
#include <cstring>

// All C(7,5)=21 index subsets
static const int SUBSETS_7C5[21][5] = {
    {0,1,2,3,4},{0,1,2,3,5},{0,1,2,3,6},{0,1,2,4,5},{0,1,2,4,6},
    {0,1,2,5,6},{0,1,3,4,5},{0,1,3,4,6},{0,1,3,5,6},{0,1,4,5,6},
    {0,2,3,4,5},{0,2,3,4,6},{0,2,3,5,6},{0,2,4,5,6},{0,3,4,5,6},
    {1,2,3,4,5},{1,2,3,4,6},{1,2,3,5,6},{1,2,4,5,6},{1,3,4,5,6},
    {2,3,4,5,6},
};

uint32_t evaluate_5(const Card hand[5]) {
    // Extract ranks and suits
    int ranks[5], suits[5];
    for (int i = 0; i < 5; i++) {
        ranks[i] = card_rank(hand[i]);
        suits[i] = card_suit(hand[i]);
    }

    // Sort ranks descending
    int sr[5];
    std::copy(ranks, ranks + 5, sr);
    std::sort(sr, sr + 5, std::greater<int>());

    // Flush check
    bool flush = (suits[0]==suits[1] && suits[1]==suits[2] &&
                  suits[2]==suits[3] && suits[3]==suits[4]);

    // Straight check (also handles A-low: A2345 where A=12)
    bool straight = false;
    int  straight_high = sr[0];
    if (sr[0]-sr[4] == 4 &&
        sr[0]!=sr[1] && sr[1]!=sr[2] && sr[2]!=sr[3] && sr[3]!=sr[4]) {
        straight = true;
    }
    // A-low straight: A(12),5(3),4(2),3(1),2(0)
    if (!straight && sr[0]==12 && sr[1]==3 && sr[2]==2 && sr[3]==1 && sr[4]==0) {
        straight = true;
        straight_high = 3; // 5-high
    }

    if (flush && straight)
        return (8u << 20) | (uint32_t)(straight_high << 16);

    // Count rank frequencies
    int freq[13] = {};
    for (int r : ranks) freq[r]++;

    // Collect groups: sorted by (freq desc, rank desc)
    int quads[4], trips[3], pairs[4], singletons[5];
    int nq=0, nt=0, np=0, ns=0;
    // Iterate ranks descending to prefer higher ranks
    for (int r = 12; r >= 0; r--) {
        if      (freq[r] == 4) quads[nq++]     = r;
        else if (freq[r] == 3) trips[nt++]     = r;
        else if (freq[r] == 2) pairs[np++]     = r;
        else if (freq[r] == 1) singletons[ns++]= r;
    }

    if (nq > 0) {
        // Quads: kicker is best non-quad card
        int kicker = (np > 0) ? pairs[0] : singletons[0];
        return (7u << 20) | ((uint32_t)quads[0] << 16) | ((uint32_t)kicker << 12);
    }

    if (nt > 0 && np > 0) {
        // Full house
        return (6u << 20) | ((uint32_t)trips[0] << 16) | ((uint32_t)pairs[0] << 12);
    }
    // Two trips (rare in 5-card but handle anyway: treat lower as pair)
    if (nt > 1) {
        return (6u << 20) | ((uint32_t)trips[0] << 16) | ((uint32_t)trips[1] << 12);
    }

    if (flush) {
        return (5u << 20) | ((uint32_t)sr[0]<<16)|((uint32_t)sr[1]<<12)|
               ((uint32_t)sr[2]<<8)|((uint32_t)sr[3]<<4)|(uint32_t)sr[4];
    }

    if (straight) {
        return (4u << 20) | ((uint32_t)straight_high << 16);
    }

    if (nt > 0) {
        // Trips + 2 kickers
        int k0 = (ns > 0) ? singletons[0] : -1;
        int k1 = (ns > 1) ? singletons[1] : -1;
        return (3u << 20) | ((uint32_t)trips[0]<<16) |
               ((uint32_t)(k0>=0?k0:0)<<12) | ((uint32_t)(k1>=0?k1:0)<<8);
    }

    if (np >= 2) {
        // Two pair + kicker
        int kicker = (ns > 0) ? singletons[0] : 0;
        return (2u << 20) | ((uint32_t)pairs[0]<<16) |
               ((uint32_t)pairs[1]<<12) | ((uint32_t)kicker<<8);
    }

    if (np == 1) {
        // One pair + 3 kickers
        int k0 = (ns > 0) ? singletons[0] : 0;
        int k1 = (ns > 1) ? singletons[1] : 0;
        int k2 = (ns > 2) ? singletons[2] : 0;
        return (1u << 20) | ((uint32_t)pairs[0]<<16) |
               ((uint32_t)k0<<12) | ((uint32_t)k1<<8) | ((uint32_t)k2<<4);
    }

    // High card
    return (0u << 20) | ((uint32_t)sr[0]<<16)|((uint32_t)sr[1]<<12)|
           ((uint32_t)sr[2]<<8)|((uint32_t)sr[3]<<4)|(uint32_t)sr[4];
}

uint32_t evaluate_7(const Card hand[7]) {
    uint32_t best = 0;
    for (const auto& idx : SUBSETS_7C5) {
        Card sub[5] = { hand[idx[0]], hand[idx[1]], hand[idx[2]],
                        hand[idx[3]], hand[idx[4]] };
        uint32_t score = evaluate_5(sub);
        if (score > best) best = score;
    }
    return best;
}
