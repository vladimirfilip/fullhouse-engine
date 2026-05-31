"""
Pure Pot Odds — every decision is based purely on mathematical pot odds vs equity.

Core logic:
  - Calls whenever equity >= pot_odds (break-even threshold).
  - Bets/raises when equity is high (>55%) to build pot.
  - No position awareness, no hand-reading, no opponent modelling.
  - Bet sizing: always 50% pot (the "neutral" size that many GTO solvers start at).
  - A decent reference bot; exploitable by polarised opponents.
"""

import eval7
import random

BOT_NAME = "PurePotOdds"

RANKS = "23456789TJQKA"


def _equity(hole, board, n_sims=250):
    ALL = [r + s for r in RANKS for s in "shdc"]
    known = set(hole + board)
    deck  = [c for c in ALL if c not in known]
    wins  = 0
    need  = 5 - len(board)
    for _ in range(n_sims):
        random.shuffle(deck)
        run = board + deck[:need]
        opp = deck[need:need + 2]
        my  = eval7.evaluate([eval7.Card(c) for c in hole + run])
        op  = eval7.evaluate([eval7.Card(c) for c in opp  + run])
        wins += (1 if my > op else 0.5 if my == op else 0)
    return wins / n_sims


def _preflop_equity(cards):
    """Preflop equity estimation using simplified Chen formula proxy."""
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    suited = (s1 == s2)
    rv = {r: i for i, r in enumerate(RANKS)}
    hi = max(rv[r1], rv[r2])
    lo = min(rv[r1], rv[r2])
    gap = hi - lo

    if gap == 0:
        # Pocket pair: base equity ~50% for 22, ~85% for AA
        return 0.50 + hi * 0.027
    base = 0.35 + (hi * 0.02) + (lo * 0.01)
    if suited:
        base += 0.03
    base -= gap * 0.02
    return max(0.30, min(0.70, base))


def decide(game_state: dict) -> dict:
    street  = game_state["street"]
    hole    = game_state["your_cards"]
    board   = game_state["community_cards"]
    pot     = game_state["pot"]
    owed    = game_state["amount_owed"]
    can_chk = game_state["can_check"]
    stack   = game_state["your_stack"]
    min_r   = game_state["min_raise_to"]
    bet_in  = game_state["your_bet_this_street"]

    # Pot odds
    pot_odds = owed / (pot + owed) if owed > 0 else 0

    # Equity estimate
    if street == "preflop":
        eq = _preflop_equity(hole)
    else:
        eq = _equity(hole, board)

    def raise_to(mult):
        total = max(int(pot * mult) + game_state["current_bet"], min_r)
        total = min(total, stack + bet_in)
        if total - bet_in >= stack:
            return {"action": "all_in"}
        return {"action": "raise", "amount": total}

    if can_chk:
        # Bet 50% pot when equity justifies it
        if eq > 0.55:
            return raise_to(0.5)
        return {"action": "check"}

    # Call/fold decision
    if eq >= pot_odds:
        # Also raise when we have clear equity advantage and opponent bet small
        if eq > 0.65 and pot_odds < 0.35:
            return raise_to(0.75)
        return {"action": "call"}

    return {"action": "fold"}
