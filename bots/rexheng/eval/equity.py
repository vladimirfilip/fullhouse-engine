"""Hand equity primitives — Monte Carlo via eval7 (treys-backed shim)."""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Iterable

import eval7

_RANKS = "23456789TJQKA"
_SUITS = "shdc"
_FULL_DECK = [r + s for r in _RANKS for s in _SUITS]


def _to_eval7(cards: Iterable[str]):
    return [eval7.Card(c) for c in cards]


def equity_vs_random(hole: list[str], board: list[str], iters: int = 150) -> float:
    """Win probability vs one random opponent hand. Approximate (Monte Carlo)."""
    if len(hole) != 2:
        return 0.5
    used = set(hole) | set(board)
    deck = [c for c in _FULL_DECK if c not in used]
    needed_board = 5 - len(board)
    wins = 0.0
    for _ in range(iters):
        sample = random.sample(deck, 2 + needed_board)
        opp = sample[:2]
        future_board = list(board) + sample[2:2 + needed_board]
        my = eval7.evaluate(_to_eval7(hole) + _to_eval7(future_board))
        their = eval7.evaluate(_to_eval7(opp) + _to_eval7(future_board))
        if my < their:  # lower = stronger
            wins += 1.0
        elif my == their:
            wins += 0.5
    return wins / iters


_PREFLOP_EQUITY: dict[str, float] = {}


def _hand_key(hole: list[str]) -> str:
    """169-bucket key: 'AKs', 'AKo', 'TT', etc."""
    if len(hole) != 2:
        return "??"
    r1, s1 = hole[0][0], hole[0][1]
    r2, s2 = hole[1][0], hole[1][1]
    rv = "23456789TJQKA"
    if rv.index(r1) < rv.index(r2):
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def preflop_equity(hole: list[str]) -> float:
    """Cached approximate equity of the hand vs one random opponent preflop."""
    key = _hand_key(hole)
    if key in _PREFLOP_EQUITY:
        return _PREFLOP_EQUITY[key]
    val = equity_vs_random(hole, [], iters=500)
    _PREFLOP_EQUITY[key] = val
    return val


def hand_strength_label(hole: list[str]) -> str:
    """Coarse label: 'premium' / 'strong' / 'medium' / 'speculative' / 'trash'."""
    key = _hand_key(hole)
    PREMIUM = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
    STRONG  = {"TT", "99", "AQs", "AQo", "AJs", "AJo", "KQs", "KQo", "ATs", "KJs"}
    MEDIUM  = {"88", "77", "66", "55",
               "ATo", "A9s", "A8s", "KJo", "KTs", "QJs", "QTs", "JTs",
               "T9s", "98s", "87s", "76s", "65s",
               "A7s", "A6s", "A5s", "A4s", "A3s", "A2s"}
    if key in PREMIUM:
        return "premium"
    if key in STRONG:
        return "strong"
    if key in MEDIUM:
        return "medium"
    if key.endswith("s"):
        return "speculative"
    if key[0] == key[1]:
        return "medium"  # any pair
    return "trash"
