#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <random>
#include <stdexcept>

// Card = uint8_t, 0..51
// Encoding: rank = card / 4, suit = card % 4
// Ranks: 0=2, 1=3, ..., 12=A
// Suits: 0=s, 1=h, 2=d, 3=c
// Matches features.py: "2s"=0, "2h"=1, "2d"=2, "2c"=3, "3s"=4, ... "Ac"=51
using Card = uint8_t;
constexpr Card NULL_CARD = 0xFF;

inline int card_rank(Card c) { return c / 4; }
inline int card_suit(Card c) { return c % 4; }

inline Card card_from_str(std::string_view s) {
    static const char* RANKS = "23456789TJQKA";
    static const char* SUITS = "shdc";
    if (s.size() != 2) throw std::runtime_error("bad card: " + std::string(s));
    const char* rp = std::strchr(RANKS, s[0]);
    const char* sp = std::strchr(SUITS, s[1]);
    if (!rp || !sp) throw std::runtime_error("bad card: " + std::string(s));
    int ri = (int)(rp - RANKS);
    int si = (int)(sp - SUITS);
    return (Card)(ri * 4 + si);
}

inline std::string card_to_str(Card c) {
    static const char RANKS[] = "23456789TJQKA";
    static const char SUITS[] = "shdc";
    std::string s(2, ' ');
    s[0] = RANKS[card_rank(c)];
    s[1] = SUITS[card_suit(c)];
    return s;
}

// Build a full 52-card deck [0..51] and shuffle it in-place
inline void build_shuffled_deck(Card deck[52], std::mt19937& rng) {
    for (int i = 0; i < 52; ++i) deck[i] = (Card)i;
    std::shuffle(deck, deck + 52, rng);
}
