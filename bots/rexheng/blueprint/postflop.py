"""Postflop policy — equity + board texture + SPR driven heuristic policy.

Kept simple by design: the blueprint is the *floor*. Exploit layer adds the edge.
"""
from __future__ import annotations

from typing import Sequence

from ..eval.equity import equity_vs_random


def board_texture(community: Sequence[str]) -> dict:
    """Classify the board for cbet sizing / bluff frequency.

    Returns dict with: dry/wet, paired, monotone, connected, high_card.
    """
    if not community:
        return {"dry": True, "paired": False, "monotone": False, "connected": False, "high": False}
    ranks = [c[0] for c in community]
    suits = [c[1] for c in community]
    rv = "23456789TJQKA"
    rank_idx = sorted([rv.index(r) for r in ranks])

    paired = len(set(ranks)) < len(ranks)
    monotone = len(set(suits)) == 1
    two_tone = len(set(suits)) == 2
    # Connected: gap between min and max <= 4 across 3-card flop, or any straight draw possible
    connected = (max(rank_idx) - min(rank_idx)) <= 4 if len(rank_idx) >= 3 else False
    high = max(rank_idx) >= rv.index("T")  # ten or higher

    wet = monotone or connected or (two_tone and high)
    return {
        "dry": not wet and not paired,
        "wet": wet,
        "paired": paired,
        "monotone": monotone,
        "two_tone": two_tone,
        "connected": connected,
        "high": high,
    }


def postflop_action(
    hole: list[str],
    community: list[str],
    pot: int,
    amount_owed: int,
    your_stack: int,
    can_check: bool,
    is_aggressor: bool,        # we were the preflop aggressor
    bluff_freq_mult: float = 1.0,
    value_thicker_mult: float = 1.0,
    iters: int = 120,
) -> dict:
    """Return decision dict: {kind: 'check'|'call'|'fold'|'bet', size_frac: float}.

    size_frac is fraction of pot for bets; caller converts to chip amount.
    """
    eq = equity_vs_random(hole, community, iters=iters)
    tex = board_texture(community)
    spr = your_stack / max(pot, 1)

    # Pot odds
    pot_odds = amount_owed / (pot + amount_owed) if amount_owed > 0 else 0.0

    # Bluff cbet frequency (when we're the aggressor and checked-to)
    cbet_freq = (0.75 if tex["dry"] else 0.45) * bluff_freq_mult
    bluff_size = 0.5 if tex["dry"] else 0.66
    value_size = 0.66 if tex["wet"] else 0.55

    if can_check:
        if is_aggressor:
            # Cbet logic: value + bluff
            if eq >= 0.62:
                return {"kind": "bet", "size_frac": value_size * value_thicker_mult, "intent": "value"}
            if eq >= 0.50 and tex["dry"]:
                return {"kind": "bet", "size_frac": 0.45, "intent": "thin_value"}
            # Bluff cbet
            import random
            if random.random() < cbet_freq * (0.6 if eq < 0.30 else 1.0):
                return {"kind": "bet", "size_frac": bluff_size, "intent": "bluff"}
            return {"kind": "check"}
        # Not aggressor: probe with strong hands; otherwise pot-control
        if eq >= 0.70:
            return {"kind": "bet", "size_frac": value_size * value_thicker_mult, "intent": "value"}
        if eq >= 0.55 and spr < 4:
            return {"kind": "bet", "size_frac": 0.5, "intent": "value"}
        return {"kind": "check"}

    # Facing a bet
    # Required equity to call = pot_odds; we add a safety margin via realisation factor
    realisation = 0.85
    needed = pot_odds / realisation
    if eq >= needed + 0.18:  # raise
        return {"kind": "raise", "size_frac": 1.0 * value_thicker_mult, "intent": "value"}
    if eq >= needed:
        return {"kind": "call"}
    # Bluff-raise frequency
    import random
    if eq >= 0.30 and random.random() < 0.10 * bluff_freq_mult and tex["wet"]:
        return {"kind": "raise", "size_frac": 1.1, "intent": "bluff"}
    return {"kind": "fold"}
