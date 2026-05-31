"""
Trap / Slow Player — smooth-calls strong hands to induce bluffs; springs the trap late.

Core logic:
  - With very strong hands (two-pair+): checks or calls to keep opponent in; raises river.
  - With medium hands (top pair): bets normally for value.
  - On the river, or when opponent shows aggression into a slow-played hand, raises big.
  - Highly effective against bluff-heavy opponents; loses value against passive opponents.
"""

import eval7
import random

BOT_NAME = "TrapSlowPlayer"

RANKS = "23456789TJQKA"


def _hand_strength(hole, board):
    """Returns: 4=monster, 3=strong, 2=medium, 1=weak, 0=air."""
    if not board:
        return 0
    try:
        score = eval7.evaluate([eval7.Card(c) for c in hole + board])
        htype = eval7.handtype(score)
    except Exception:
        return 0

    if "Four" in htype or "Full" in htype or "Flush" in htype or "Straight" in htype:
        return 4
    if "Three" in htype or "Two" in htype:
        return 3
    if "Pair" in htype:
        # Top pair or overpair = medium; underpair = weak
        board_high = max(RANKS.index(c[0]) for c in board)
        hole_ranks = [RANKS.index(c[0]) for c in hole]
        if any(r > board_high for r in hole_ranks) or \
           any(RANKS.index(c[0]) == board_high and c[0] in [h[0] for h in hole] for c in board):
            return 2
        return 1
    return 0


def _preflop_equity(cards):
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    suited = (s1 == s2)
    rv = {r: i for i, r in enumerate(RANKS)}
    hi, lo = max(rv[r1], rv[r2]), min(rv[r1], rv[r2])
    if r1 == r2:
        return 0.50 + hi * 0.027
    base = 0.35 + hi * 0.02 + lo * 0.01 - (hi - lo) * 0.02
    if suited:
        base += 0.03
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

    strength = _hand_strength(hole, board)

    def raise_to(mult):
        total = max(int(pot * mult) + game_state["current_bet"], min_r)
        total = min(total, stack + bet_in)
        if total - bet_in >= stack:
            return {"action": "all_in"}
        return {"action": "raise", "amount": total}

    # ── Preflop ──────────────────────────────────────────────────────────────
    if street == "preflop":
        eq = _preflop_equity(hole)
        if eq >= 0.60:
            # Strong preflop hand — sometimes limp with the very best to trap
            if eq >= 0.72 and random.random() < 0.4:
                return {"action": "call"}   # limp with AA/KK occasionally
            return raise_to(3.0)
        if can_chk:
            return {"action": "check"}
        if eq >= 0.45 or owed <= 100:
            return {"action": "call"}
        return {"action": "fold"}

    # ── Postflop ─────────────────────────────────────────────────────────────
    is_river = (street == "river")

    if can_chk:
        if strength == 4:
            # Monster: always slow-play unless river (spring the trap)
            if is_river:
                return raise_to(1.0)      # big river bet with monster
            return {"action": "check"}    # check earlier streets

        if strength == 3:
            # Strong: slow-play on flop, start betting turn/river
            if street == "flop" and random.random() < 0.6:
                return {"action": "check"}
            return raise_to(0.65)

        if strength == 2:
            # Medium: bet for value normally
            return raise_to(0.55)

        return {"action": "check"}  # weak/air: check-fold

    # Facing a bet
    if strength == 4:
        # Monster facing a bet: call or raise (spring the trap on river)
        if is_river or random.random() < 0.4:
            return raise_to(1.2)   # raise
        return {"action": "call"}  # call earlier to keep in

    if strength == 3:
        # Strong: call most bets, raise big bets on river
        pot_fraction = owed / max(pot, 1)
        if is_river and pot_fraction > 0.5:
            return raise_to(1.0)
        if pot_fraction <= 1.0:
            return {"action": "call"}
        return {"action": "fold"}

    if strength == 2:
        # Medium: call reasonable bets
        pot_odds = owed / (pot + owed)
        return {"action": "call"} if pot_odds < 0.40 else {"action": "fold"}

    # Weak/air: fold
    return {"action": "fold"}
