"""
Equity computation for preflop CFR leaf nodes.

HU (heads-up) leaves: precomputed 169×169 table built once at startup.
Multiway leaves: on-demand MC board rollout with LRU cache keyed on
    canonical_handset_key (suit-isomorphic).

All equity values are from the perspective of player 0 in the hand list.
equity_vector(hands) returns a list of floats summing to 1.
"""

from __future__ import annotations

import os
import random
import zlib

import numpy as np
import eval7

from preflop_cfr.cards import (
    ALL_CARDS, BUCKET_INFO, hand_to_bucket, canonical_handset_key, RANKS, SUITS
)
from preflop_cfr.config import HU_EQUITY_BOARDS, MULTIWAY_MC_BOARDS, HU_TABLE_PATH


# ── Board rollout ─────────────────────────────────────────────────────────────

def _rollout_equity(
    hands: list[list[eval7.Card]],
    known_board: list[eval7.Card],
    n_boards: int,
    rng: random.Random | None = None,
) -> list[float]:
    """
    Monte Carlo equity for n players, given their concrete hole cards and any
    known board cards.  Returns a list of fractional win+tie shares summing to 1.

    `rng`: when given, board sampling draws from this dedicated Random instead of
    the global RNG.  Multiway leaves seed it deterministically per matchup so
    every parallel worker freezes the SAME equity (CFR needs a stationary game;
    independent per-worker MC noise made workers optimise slightly different
    games — see plan §A.5).
    """
    n = len(hands)
    # Exclude dealt cards by VALUE, not object identity.  CFR-path hands are the
    # same Card objects as ALL_CARDS (random.sample), but build_hu_table feeds in
    # freshly-constructed eval7.Card objects whose id() is never in ALL_CARDS —
    # so an id()-based filter left the hole cards in the board deck and dealt
    # them onto the board (corrupting the HU table).  str(card) is stable.
    known = {str(c) for h in hands for c in h} | {str(c) for c in known_board}
    remaining = [c for c in ALL_CARDS if str(c) not in known]
    need = 5 - len(known_board)

    # Board sampling dominates this function (≈60% of total CFR runtime in the
    # profile), and random.sample's per-draw overhead (isinstance + _randbelow)
    # was the bulk of it.  Replace it with an inline partial Fisher–Yates: each
    # board swaps the first `need` slots of `remaining` with a random later slot
    # using a single random()*range multiply per card.  `remaining` is a private
    # throwaway list, so we shuffle it in place and never copy — a permuted prefix
    # is still a uniform draw, so successive boards stay correctly distributed.
    rnd      = rng.random if rng is not None else random.random
    evaluate = eval7.evaluate
    m        = len(remaining)
    tally    = [0.0] * n
    for _ in range(n_boards):
        for i in range(need):
            j = i + int(rnd() * (m - i))
            remaining[i], remaining[j] = remaining[j], remaining[i]
        board  = known_board + remaining[:need]
        scores = [evaluate(h + board) for h in hands]
        best   = max(scores)
        nwin   = scores.count(best)
        if nwin == 1:
            tally[scores.index(best)] += 1.0
        else:
            share = 1.0 / nwin
            for i, s in enumerate(scores):
                if s == best:
                    tally[i] += share

    # Each board contributes exactly 1.0 total, so the sum is n_boards (>0 here).
    if n_boards == 0:
        return [1.0 / n] * n
    return [t / n_boards for t in tally]


# ── 169×169 HU equity table ───────────────────────────────────────────────────

# Shape: [169, 169], entry [b1, b2] = P(hand-bucket-b1 beats hand-bucket-b2)
# when both players are all-in with no board cards.
# Built lazily on first access via build_hu_table().

_HU_TABLE: np.ndarray | None = None


def _sample_hand_for_bucket(bucket: int, exclude: set[str]) -> list[eval7.Card] | None:
    """Sample a concrete hand that falls in `bucket` avoiding `exclude` cards."""
    hi, lo, suited = BUCKET_INFO[bucket]
    hi_r, lo_r = RANKS[hi], RANKS[lo]
    if suited:
        # try all 4 suits
        suits = list(SUITS)
        random.shuffle(suits)
        for s in suits:
            c1s, c2s = hi_r + s, lo_r + s
            if c1s not in exclude and c2s not in exclude and c1s != c2s:
                return [eval7.Card(c1s), eval7.Card(c2s)]
        return None
    else:
        # offsuit or pair: try combinations
        combos = [(s1, s2) for s1 in SUITS for s2 in SUITS if s1 != s2 or hi == lo]
        if hi == lo:
            # pair: need two different suits
            combos = [(s1, s2) for i, s1 in enumerate(SUITS) for s2 in SUITS[i+1:]]
        else:
            combos = [(s1, s2) for s1 in SUITS for s2 in SUITS if s1 != s2]
        random.shuffle(combos)
        for s1, s2 in combos:
            c1s, c2s = hi_r + s1, lo_r + s2
            if c1s not in exclude and c2s not in exclude:
                return [eval7.Card(c1s), eval7.Card(c2s)]
        return None


def build_hu_table(n_boards: int = HU_EQUITY_BOARDS) -> np.ndarray:
    """
    Build and return the 169×169 HU equity table.
    entry [b1, b2] = P(b1 beats b2) estimated from n_boards MC rollouts per cell.
    Symmetric: table[b2, b1] = 1 - table[b1, b2].
    """
    table = np.full((169, 169), 0.5, dtype=np.float32)
    for b1 in range(169):
        for b2 in range(b1 + 1, 169):
            # sample concrete hands
            hand1 = _sample_hand_for_bucket(b1, set())
            if hand1 is None:
                continue
            exclude = {str(c) for c in hand1}
            hand2 = _sample_hand_for_bucket(b2, exclude)
            if hand2 is None:
                continue
            eq = _rollout_equity([hand1, hand2], [], n_boards)
            table[b1, b2] = eq[0]
            table[b2, b1] = eq[1]
    return table


def get_hu_table(path: str = HU_TABLE_PATH) -> np.ndarray:
    """
    Return the singleton 169×169 HU equity table.

    Resolution order: in-memory cache → on-disk cache → build (and persist).
    The table is deterministic up to MC noise, so caching it to disk lets every
    worker process (and every rerun) skip the multi-minute rebuild.
    """
    global _HU_TABLE
    if _HU_TABLE is not None:
        return _HU_TABLE

    if path and os.path.exists(path):
        try:
            data = np.load(path)
            if int(data["boards"]) >= HU_EQUITY_BOARDS:
                _HU_TABLE = data["table"].astype(np.float32)
                return _HU_TABLE
        except Exception:
            pass  # corrupt/old cache → fall through and rebuild

    _HU_TABLE = build_hu_table()
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, table=_HU_TABLE,
                     boards=np.array(HU_EQUITY_BOARDS))
        except Exception:
            pass  # best-effort cache; never fail training over it
    return _HU_TABLE


def hu_equity(b1: int, b2: int) -> float:
    """Return P(bucket b1 beats bucket b2) from the precomputed HU table."""
    return float(get_hu_table()[b1, b2])


# ── Multiway equity ────────────────────────────────────────────────────────────

# Cache keyed on canonical_handset_key (tuple) → list of floats
_multiway_cache: dict[tuple, list[float]] = {}


def multiway_equity(hands: list[list[eval7.Card]]) -> list[float]:
    """
    Equity vector for 2+ players with concrete hole cards, no board yet.
    Returns cached result when possible (suit-isomorphic key).
    """
    if len(hands) == 2:
        b1 = hand_to_bucket(hands[0][0], hands[0][1])
        b2 = hand_to_bucket(hands[1][0], hands[1][1])
        eq = hu_equity(b1, b2)
        return [eq, 1.0 - eq]

    key = canonical_handset_key(hands)
    if key in _multiway_cache:
        return _multiway_cache[key]

    # Seed a dedicated RNG from the (process-stable) matchup key so every worker
    # freezes the SAME equity for this leaf — a stationary game for CFR.  Python's
    # built-in hash() is salted per process, so derive a stable seed via crc32.
    seed = zlib.crc32(repr(key).encode())
    result = _rollout_equity(hands, [], MULTIWAY_MC_BOARDS,
                             rng=random.Random(seed))
    _multiway_cache[key] = result
    return result
