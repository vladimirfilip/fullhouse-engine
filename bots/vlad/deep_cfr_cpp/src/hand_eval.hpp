#pragma once
#include <cstdint>
#include "card.hpp"

// Evaluate a 5-card hand. Higher score = better hand.
// Encoding: (category << 20) | (r1 << 16) | (r2 << 12) | (r3 << 8) | (r4 << 4) | r5
// Categories: 0=high card, 1=pair, 2=two pair, 3=trips, 4=straight,
//             5=flush, 6=full house, 7=quads, 8=straight flush
uint32_t evaluate_5(const Card hand[5]);

// Evaluate a 7-card hand by finding the best 5-card subset (max over C(7,5)=21).
uint32_t evaluate_7(const Card hand[7]);
