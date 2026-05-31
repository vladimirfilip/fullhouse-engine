"""Preflop policy — position-aware ranges informed by 6-max GTO charts.

Logic adapted from published RangeConverter / GTO Wizard 6-max ranges, hand-
encoded for memory efficiency. Three tiers: open-raise, call, 3-bet.
"""
from __future__ import annotations

from ..eval.equity import _hand_key

# Open-raise ranges by relative position (0=earliest, 1=BTN, 2=SB)
# pos_idx is normalised: 0=UTG/MP, 1=CO/HJ, 2=BTN, 3=SB, 4=BB
_OPEN_RANGES = {
    0: {  # UTG/MP — tight ~14%
        "AA","KK","QQ","JJ","TT","99","88","77",
        "AKs","AQs","AJs","ATs","KQs","KJs","QJs",
        "AKo","AQo",
    },
    1: {  # CO — wider ~22%
        "AA","KK","QQ","JJ","TT","99","88","77","66","55",
        "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A5s",
        "KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s",
        "AKo","AQo","AJo","KQo",
    },
    2: {  # BTN — wide ~40%
        "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
        "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
        "KQs","KJs","KTs","K9s","K8s","K7s","K6s","K5s",
        "QJs","QTs","Q9s","Q8s","JTs","J9s","J8s",
        "T9s","T8s","98s","97s","87s","86s","76s","65s","54s",
        "AKo","AQo","AJo","ATo","A9o","KQo","KJo","KTo","QJo","QTo","JTo",
    },
    3: {  # SB (vs unopened) — limp/raise mix; we just use BTN-style raises
        "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
        "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
        "KQs","KJs","KTs","K9s","K8s","K7s","Q9s","QJs","QTs","JTs","J9s",
        "T9s","T8s","98s","87s","76s","65s","54s",
        "AKo","AQo","AJo","ATo","KQo","KJo","KTo","QJo","JTo",
    },
}

# 3-bet (re-raise) ranges — polarised: value + bluff
_THREE_BET_VALUE = {"AA","KK","QQ","JJ","AKs","AKo"}
_THREE_BET_BLUFF = {"A5s","A4s","A3s","A2s","KQs","T9s","98s","76s"}
_THREE_BET_RANGE = _THREE_BET_VALUE | _THREE_BET_BLUFF

# Call (flat) ranges — middling hands that prefer pot-control to 3-betting
_CALL_VS_OPEN = {
    "TT","99","88","77","66","55","44","33","22",
    "AQs","AJs","ATs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s","76s",
    "AQo","AJo","KQo",
}


def position_tier(seat_to_act: int, n_players: int, dealer_seat: int) -> int:
    """Map absolute seat to a coarse position tier 0..4.

    Tier 0 = UTG/MP (early), 1 = CO/HJ (middle), 2 = BTN (late),
    3 = SB, 4 = BB.
    """
    rel = (seat_to_act - dealer_seat) % n_players
    if n_players == 2:
        # Heads-up: dealer is SB
        return 3 if rel == 0 else 4
    if rel == 0:
        return 2  # BTN
    if rel == 1:
        return 3  # SB
    if rel == 2:
        return 4  # BB
    if rel == n_players - 1:
        return 1  # CO
    return 0  # UTG/MP


def preflop_action(
    hole: list[str],
    seat_to_act: int,
    dealer_seat: int,
    n_players: int,
    has_aggressor: bool,  # someone has raised before us
    facing_3bet: bool,    # someone re-raised after preflop raise
) -> str:
    """Return 'fold' / 'call' / 'open_raise' / 'three_bet' / 'four_bet'.

    Caller maps these to engine actions with sizing.
    """
    key = _hand_key(hole)
    tier = position_tier(seat_to_act, n_players, dealer_seat)
    open_set = _OPEN_RANGES.get(min(tier, 3), _OPEN_RANGES[2])

    if facing_3bet:
        # 4-bet only with premium
        if key in {"AA", "KK", "QQ", "AKs", "AKo"}:
            return "four_bet"
        if key in _CALL_VS_OPEN:
            return "call"
        return "fold"

    if has_aggressor:
        # Facing an open: 3-bet or call or fold
        if key in _THREE_BET_VALUE:
            return "three_bet"
        if key in _CALL_VS_OPEN:
            return "call"
        # Position-dependent 3-bet bluff
        if tier >= 2 and key in _THREE_BET_BLUFF:
            return "three_bet"
        return "fold"

    # Unopened pot
    if key in open_set:
        return "open_raise"
    return "fold"
