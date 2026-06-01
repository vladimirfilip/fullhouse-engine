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

    # non-pair: enumerate high-rank-major, and within each hi block lo DESCENDING
    # (AKs=13, AQs=14, …, A2s=24, KQs=25, …) to match _build_bucket_info /
    # BUCKET_INFO and the module docstring.  The two encodings MUST agree: the HU
    # equity table is built indexed by BUCKET_INFO but queried via this function,
    # so any disagreement returns the wrong cell.
    #
    # block start = number of non-pair combos with high rank > hi
    #   sum_{r=hi+1}^{12} r = (12*(12+1)//2) - (hi*(hi+1)//2) = 78 - hi*(hi+1)//2
    # position within the block, lo descending = (hi-1) - lo
    offset = (78 - hi * (hi + 1) // 2) + (hi - 1 - lo)  # 0-based within the block
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
    Suit-isomorphism-invariant cache key for a list of concrete hole-card pairs,
    preserving the input hand ORDER.

    Two handsets share a key iff one is a global suit relabeling of the other
    with the hands in the same order.  Order is preserved on purpose: the cached
    equity vector is in input-hand order, so reusing it across handsets that
    share a key assigns each player's equity to the right seat.  (A previous
    version keyed on a *sorted* tuple of (hi, lo, suited) triples, which both
    dropped the hand→equity correspondence — returning per-seat equities in the
    wrong order on a cache hit — and discarded cross-hand suit coordination.)

    Suits are relabeled with a single global first-appearance map, so shared
    suits across hands (which drive flush/removal effects) yield a distinct key
    from non-shared ones, while a pure relabeling of all four suits collapses to
    the same key.
    """
    relabel: dict[int, int] = {}
    out: list[tuple] = []
    for h in hands:
        # Order the two cards by rank desc (suit-permutation invariant for
        # non-pairs since rank dominates); the global first-appearance relabel
        # below handles the suit labels.
        cards = sorted(
            ((_CARD_RANK[str(c)], _CARD_SUIT[str(c)]) for c in h),
            key=lambda rs: (-rs[0], rs[1]),
        )
        enc = []
        for rank, suit in cards:
            lbl = relabel.get(suit)
            if lbl is None:
                lbl = len(relabel)
                relabel[suit] = lbl
            enc.append((rank, lbl))
        # A hand is an unordered pair; sort the encoded cards so within-hand
        # ordering never produces spurious distinct keys (matters for pairs).
        out.append(tuple(sorted(enc)))
    return tuple(out)
