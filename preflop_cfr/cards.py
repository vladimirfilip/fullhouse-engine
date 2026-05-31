"""
169-bucket hand canonicalization for preflop CFR.

Rank order: 2=0 … A=12 (same mapping as bot.py _CARD_IDX // 4).
Bucket index: 0..168, ordered (pair > suited > offsuit), high-rank-major.
  [0..12]   pairs:   AA=0, KK=1, … 22=12
  [13..90]  suited:  AKs=13, AQs=14, … 23s=90
  [91..168] offsuit: AKo=91, AQo=92, … 32o=168

card format: eval7.Card objects or string "As", "Kh", etc.
"""

import itertools
import random

import eval7

RANKS  = "23456789TJQKA"
SUITS  = "shdc"
N_RANK = 13

# ── Rank character → index (2=0 … A=12) ──────────────────────────────────────
RANK_IDX: dict[str, int] = {r: i for i, r in enumerate(RANKS)}

# ── Precompute card-string → (rank_idx, suit_idx) ─────────────────────────────
_CARD_RANK: dict[str, int] = {r + s: RANK_IDX[r] for r in RANKS for s in SUITS}
_CARD_SUIT: dict[str, int] = {r + s: si for r in RANKS for si, s in enumerate(SUITS)}


def _card_str(c) -> str:
    """Accept eval7.Card or string, return canonical 2-char string."""
    return str(c)


def hand_to_bucket(c1, c2) -> int:
    """
    Map two hole cards to a 0..168 bucket index.

    Invariant: hand_to_bucket(c1, c2) == hand_to_bucket(c2, c1) for all suits.
    Suit isomorphism: only the suited/offsuit flag matters, not which suits.
    """
    s1, s2 = _card_str(c1), _card_str(c2)
    r1, r2 = _CARD_RANK[s1], _CARD_RANK[s2]
    suited = (_CARD_SUIT[s1] == _CARD_SUIT[s2])
    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)

    if hi == lo:
        # pair: AA→0, KK→1, ... 22→12
        return 12 - hi

    # non-pair: enumerate high-rank-major
    # number of non-pair combos with high rank > lo rank, high rank = hi:
    #   pairs idx 0..12
    #   suited: (hi, lo) for hi > lo → index = 13 + (12-hi)*hi//2 + ... computed directly
    # Simpler: count all (h,l) pairs with h>l in rank order, h=12 down to h=1.
    # Position within suited/offsuit block: hi_from_top = 12-hi (0=A, 1=K …)
    # combos per hi rank: hi combos (lo can be 0..hi-1)
    # sum_{r=hi+1}^{12} r = (12*(12+1)//2) - (hi*(hi+1)//2) = 78 - hi*(hi+1)//2
    offset = 78 - hi * (hi + 1) // 2 + lo  # 0-based within the block
    if suited:
        return 13 + offset
    else:
        return 91 + offset


# ── Inverse map: bucket → (hi_rank, lo_rank, suited) ─────────────────────────
def _build_bucket_info() -> list[tuple[int, int, bool]]:
    info = []
    # pairs
    for hi in range(12, -1, -1):   # AA first
        info.append((hi, hi, False))
    # suited
    for hi in range(12, 0, -1):
        for lo in range(hi - 1, -1, -1):
            info.append((hi, lo, True))
    # offsuit
    for hi in range(12, 0, -1):
        for lo in range(hi - 1, -1, -1):
            info.append((hi, lo, False))
    assert len(info) == 169
    return info


BUCKET_INFO: list[tuple[int, int, bool]] = _build_bucket_info()


# ── Deck helpers ──────────────────────────────────────────────────────────────

ALL_CARDS: list[eval7.Card] = [
    eval7.Card(r + s) for r in RANKS for s in SUITS
]


def fresh_deck() -> list[eval7.Card]:
    """Return a shuffled copy of the full 52-card deck."""
    deck = list(ALL_CARDS)
    random.shuffle(deck)
    return deck


def deal_hands(deck: list[eval7.Card], n: int) -> list[list[eval7.Card]]:
    """Deal n × 2-card hands from the front of deck (modifies deck in place)."""
    hands = []
    for _ in range(n):
        hands.append([deck.pop(), deck.pop()])
    return hands


# ── Suit isomorphism key for caching multiway leaves ─────────────────────────

def canonical_handset_key(hands: list[list[eval7.Card]]) -> tuple:
    """
    Return a suit-isomorphic canonical key for a set of concrete hole-card pairs.
    Used to cache multiway equity rollout results across suit permutations.
    """
    # Represent each hand as a sorted (rank, rank, suited) triple.
    triples = []
    for h in hands:
        s1, s2 = str(h[0]), str(h[1])
        r1, r2 = _CARD_RANK[s1], _CARD_RANK[s2]
        suited = (_CARD_SUIT[s1] == _CARD_SUIT[s2])
        hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
        triples.append((hi, lo, suited))
    return tuple(sorted(triples))
