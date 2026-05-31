"""
Tight Passive ("Rock") — plays only premium hands, limps rather than raises.

Core logic:
  - Preflop: AA/KK/QQ/JJ/AKs/AKo only. Calls blinds (limps), raises only with AA/KK.
  - Postflop: check-call with top pair or better; fold to aggression on weak boards.
  - Extremely exploitable by steal-heavy opponents but beats random/loose players.
"""

import eval7

BOT_NAME = "TightPassive"

RANKS = "23456789TJQKA"

# Only these hands are played preflop
PREMIUM = {"AA", "KK"}          # raise-worthy
STRONG  = {"QQ", "JJ", "AKs", "AKo"}   # just call/limp


def _hand_key(cards):
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    if RANKS.index(r1) < RANKS.index(r2):
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return r1 + r2
    suited = "s" if s1 == s2 else "o"
    return r1 + r2 + suited


def _top_pair_or_better(hole, board):
    """True if we have at least top pair on the board."""
    if not board:
        return False
    hole_ranks = {c[0] for c in hole}
    board_ranks = [c[0] for c in board]
    top_board_rank = max(board_ranks, key=lambda r: RANKS.index(r))
    # Pocket pair above board = overpair, counts
    if hole[0][0] == hole[1][0]:
        return RANKS.index(hole[0][0]) >= RANKS.index(top_board_rank)
    # Pair with top board card
    return top_board_rank in hole_ranks


def _made_hand_score(hole, board):
    """Quick score: sets > two_pair > top_pair > underpair > airball (0-4)."""
    if not board:
        return 0
    all7 = [eval7.Card(c) for c in hole + board]
    score = eval7.evaluate(all7)
    htype = eval7.handtype(score)
    if "Straight" in htype or "Flush" in htype or "Full" in htype or "Four" in htype:
        return 5
    if "Three" in htype or "Two" in htype:
        return 3
    if "Pair" in htype:
        # Is it top pair or better?
        return 2 if _top_pair_or_better(hole, board) else 1
    return 0


def decide(game_state: dict) -> dict:
    street   = game_state["street"]
    hole     = game_state["your_cards"]
    board    = game_state["community_cards"]
    pot      = game_state["pot"]
    owed     = game_state["amount_owed"]
    can_chk  = game_state["can_check"]
    stack    = game_state["your_stack"]
    min_r    = game_state["min_raise_to"]
    bet_in   = game_state["your_bet_this_street"]

    key = _hand_key(hole)

    # ── Preflop ──────────────────────────────────────────────────────────────
    if street == "preflop":
        if key in PREMIUM:
            # Raise big ~70% of the time, limp 30% to mix in a trap
            import random
            if random.random() < 0.7:
                raise_to = min(min_r * 3, stack + bet_in)
                raise_to = max(raise_to, min_r)
                return {"action": "raise", "amount": raise_to}
            return {"action": "call"}

        if key in STRONG:
            return {"action": "call"}

        # Trash — fold unless we can check for free
        if can_chk:
            return {"action": "check"}
        return {"action": "fold"}

    # ── Postflop ─────────────────────────────────────────────────────────────
    score = _made_hand_score(hole, board)

    if can_chk:
        return {"action": "check"}

    # Calling thresholds: stronger hand = call larger bets
    pot_fraction = owed / max(pot, 1)

    if score >= 5:    # monster — always call
        return {"action": "call"}
    if score >= 3:    # two pair / set — call up to pot-size bet
        return {"action": "call"} if pot_fraction <= 1.0 else {"action": "fold"}
    if score >= 2:    # top pair — call up to half pot
        return {"action": "call"} if pot_fraction <= 0.5 else {"action": "fold"}
    if score >= 1:    # underpair — call tiny bets only
        return {"action": "call"} if pot_fraction <= 0.2 else {"action": "fold"}

    return {"action": "fold"}
