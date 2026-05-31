"""
Bluff Heavy — bets and raises at very high frequency regardless of hand strength.

Core logic:
  - Bets ~80% of the time when checked to, raises ~60% of bets faced.
  - Ignores hand strength; purely applies aggression.
  - Uses varying bet sizes (60-130% pot) to seem less mechanical.
  - Collapses hard against calling stations and trap players.
  - Highly profitable against tight-passive opponents; a disaster against loose-passive.
"""

import random

BOT_NAME = "BluffHeavy"


def decide(game_state: dict) -> dict:
    pot     = game_state["pot"]
    owed    = game_state["amount_owed"]
    can_chk = game_state["can_check"]
    stack   = game_state["your_stack"]
    min_r   = game_state["min_raise_to"]
    bet_in  = game_state["your_bet_this_street"]
    cur_bet = game_state["current_bet"]

    def raise_to(mult):
        total = max(int(pot * mult) + cur_bet, min_r)
        total = min(total, stack + bet_in)
        if total - bet_in >= stack:
            return {"action": "all_in"}
        return {"action": "raise", "amount": total}

    if can_chk:
        # Bet 80% of the time
        if random.random() < 0.80:
            mult = random.uniform(0.6, 1.3)
            return raise_to(mult)
        return {"action": "check"}

    # Facing a bet: raise 55%, call 20%, fold 25%
    r = random.random()
    if r < 0.55:
        mult = random.uniform(0.8, 1.5)
        return raise_to(mult)
    if r < 0.75:
        return {"action": "call"}
    return {"action": "fold"}
