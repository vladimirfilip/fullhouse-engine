"""
Loose Passive ("Calling Station") — calls almost everything, almost never raises.

Core logic:
  - Preflop: calls any pair, any broadway, any suited connector, any ace — ~60% of hands.
  - Postflop: calls with any made pair, any flush/straight draw; folds only on air.
  - Never bluffs. Extremely difficult to bluff off a hand.
  - Exploited by value-betting relentlessly; hard to exploit by bluffing.
"""

import eval7

BOT_NAME = "LoosePassive"

RANKS = "23456789TJQKA"


def _rank_val(r):
    return RANKS.index(r)


def _hand_key(cards):
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    if _rank_val(r1) < _rank_val(r2):
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return r1 + r2
    suited = "s" if s1 == s2 else "o"
    return r1 + r2 + suited


def _preflop_play(cards):
    """Return True if this hand is worth entering the pot."""
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    suited = (s1 == s2)
    ranks = sorted([_rank_val(r1), _rank_val(r2)], reverse=True)

    # Any pocket pair
    if r1 == r2:
        return True
    # Any ace
    if "A" in (r1, r2):
        return True
    # Any broadway (T+)
    if ranks[1] >= _rank_val("T"):
        return True
    # Any suited connector with gap <= 1 (e.g. 87s, 97s)
    if suited and abs(ranks[0] - ranks[1]) <= 2 and ranks[1] >= _rank_val("5"):
        return True
    # Any king
    if "K" in (r1, r2):
        return True

    return False


def _draw_outs(hole, board):
    """Estimate number of draw outs (flush draw or open-ended straight draw)."""
    if len(board) < 3:
        return 0

    suits = [c[1] for c in hole + board]
    hole_suits = [c[1] for c in hole]
    board_suits = [c[1] for c in board]

    # Flush draw: 4 to a flush
    for suit in "shdc":
        count = suits.count(suit)
        if count >= 4:
            return 9  # flush draw = ~9 outs

    # Straight draw
    all_ranks = sorted(set(_rank_val(c[0]) for c in hole + board))
    for i in range(len(all_ranks) - 3):
        window = all_ranks[i:i+4]
        if window[-1] - window[0] <= 4:
            return 8  # open-ended or gutshot

    return 0


def _postflop_worth_calling(hole, board, pot, owed):
    """True if we have enough to justify calling."""
    if not board:
        return True

    all_cards = [eval7.Card(c) for c in hole + board]
    score = eval7.evaluate(all_cards)
    htype = eval7.handtype(score)

    # Any made pair or better: call up to pot-sized bet
    if "Pair" in htype or "Two" in htype or "Three" in htype or \
       "Straight" in htype or "Flush" in htype or "Full" in htype or "Four" in htype:
        return owed <= pot * 1.2

    # Draw: call with pot odds better than ~4:1
    outs = _draw_outs(hole, board)
    if outs >= 8:
        return owed <= pot * 0.4
    if outs >= 4:
        return owed <= pot * 0.25

    # Air: call only tiny bets (< 15% pot)
    return owed <= pot * 0.15


def decide(game_state: dict) -> dict:
    street  = game_state["street"]
    hole    = game_state["your_cards"]
    board   = game_state["community_cards"]
    pot     = game_state["pot"]
    owed    = game_state["amount_owed"]
    can_chk = game_state["can_check"]
    stack   = game_state["your_stack"]

    if can_chk:
        return {"action": "check"}

    if street == "preflop":
        if _preflop_play(hole):
            return {"action": "call"}
        return {"action": "fold"}

    # Postflop
    if _postflop_worth_calling(hole, board, pot, owed):
        return {"action": "call"}
    return {"action": "fold"}
