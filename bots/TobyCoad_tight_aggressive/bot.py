"""
Tight Aggressive ("TAG") — the standard winning poker archetype.

Core logic:
  - Preflop: top ~15% of hands; raises 2.5-3x from position, 3x OOP.
  - Postflop: continuation-bets ~65% of the time; value-bets top pair+;
    folds draws to large bets unless getting good odds.
  - Position-aware: opens wider IP, tighter OOP; attacks checks IP.
  - Represents a GTO-adjacent "textbook" strategy that LLM bots tend to converge on.
"""

import eval7
import random

BOT_NAME = "TightAggressive"

RANKS = "23456789TJQKA"

# Top ~15% of hands
OPEN_RANGE = {
    "AA", "KK", "QQ", "JJ", "TT", "99", "88",
    "AKs", "AKo", "AQs", "AQo", "AJs", "AJo", "ATs",
    "KQs", "KQo", "KJs", "QJs", "JTs",
}
# Additional hands OK in position (IP adds ~5% more)
OPEN_IP_EXTRA = {"77", "66", "A9s", "KTs", "QTs", "J9s", "T9s", "98s"}


def _hand_key(cards):
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    if RANKS.index(r1) < RANKS.index(r2):
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def _sb_seat(state):
    for e in state["action_log"]:
        if e["action"] == "small_blind":
            return e["seat"]
    return None


def _in_position(state):
    my = state["seat_to_act"]
    sb = _sb_seat(state)
    n  = len(state["players"])
    if sb is None:
        return True
    if n == 2:
        return (my != sb) if state["street"] == "preflop" else (my == sb)
    # Multi: rough estimate — acting last = furthest from SB
    dealer = (sb - 1) % n
    active = [p["seat"] for p in state["players"] if not p["is_folded"]]
    ordered = [(dealer + 1 + i) % n for i in range(n) if (dealer + 1 + i) % n in active]
    return ordered and my == ordered[-1]


def _equity_fast(hole, board, n_sims=250):
    ALL = [r + s for r in RANKS for s in "shdc"]
    known = set(hole + board)
    deck  = [c for c in ALL if c not in known]
    wins  = 0
    need  = 5 - len(board)
    for _ in range(n_sims):
        random.shuffle(deck)
        run  = board + deck[:need]
        opp  = deck[need:need + 2]
        my   = eval7.evaluate([eval7.Card(c) for c in hole + run])
        op   = eval7.evaluate([eval7.Card(c) for c in opp  + run])
        wins += (1 if my > op else 0.5 if my == op else 0)
    return wins / n_sims


def _made_strong(hole, board):
    """True if we have two-pair or better."""
    if not board:
        return False
    s = eval7.evaluate([eval7.Card(c) for c in hole + board])
    return "Two" in eval7.handtype(s) or eval7.handtype(s) in (
        "Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
    )


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

    ip  = _in_position(game_state)
    key = _hand_key(hole)

    def raise_to(mult):
        total = max(int(pot * mult) + game_state["current_bet"], min_r)
        total = min(total, stack + bet_in)
        chips_need = total - bet_in
        if chips_need >= stack:
            return {"action": "all_in"}
        return {"action": "raise", "amount": total}

    # ── Preflop ──────────────────────────────────────────────────────────────
    if street == "preflop":
        in_range = key in OPEN_RANGE or (ip and key in OPEN_IP_EXTRA)

        if in_range:
            # Premium: raise; otherwise re-raise if facing a raise
            mult = 3 if not ip else 2.5
            return raise_to(mult) if owed == 0 or random.random() < 0.85 else {"action": "call"}

        # 3-bet squeeze: if already raised, fold marginal
        if owed > 0:
            return {"action": "fold"}
        if can_chk:
            return {"action": "check"}
        return {"action": "fold"}

    # ── Postflop ─────────────────────────────────────────────────────────────
    eq = _equity_fast(hole, board)
    pot_odds = owed / (pot + owed) if owed > 0 else 0

    if can_chk:
        # c-bet IP with top-pair+ equity, or as bluff 30% with draws
        if ip:
            if eq > 0.55:
                return raise_to(0.65)
            if eq > 0.35 and random.random() < 0.3:
                return raise_to(0.5)
        return {"action": "check"}

    # Facing a bet
    if eq > 0.65:
        # Strong hand: raise for value
        if random.random() < 0.4:
            return raise_to(0.8)
        return {"action": "call"}
    if eq > pot_odds + 0.05:
        return {"action": "call"}

    return {"action": "fold"}
