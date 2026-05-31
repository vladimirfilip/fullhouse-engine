"""
Loose Aggressive ("LAG") — wide range, relentless aggression.

Core logic:
  - Preflop: opens top ~35% of hands; attacks any limp with 4-5x; 3-bets polarised.
  - Postflop: always continuation-bets; fires second and third barrels at high frequency;
    applies maximum fold pressure with large sizings.
  - Breaks even or profits against tight/passive bots; gets stacked by calling stations.
  - Typical professional tournament player style.
"""

import eval7
import random

BOT_NAME = "LooseAggressive"

RANKS = "23456789TJQKA"

# Top ~35% of hands to open
LAG_OPEN = {
    "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
    "AKs","AKo","AQs","AQo","AJs","AJo","ATs","ATo","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
    "KQs","KQo","KJs","KJo","KTs","KTo","K9s",
    "QJs","QJo","QTs","QTo","Q9s",
    "JTs","JTo","J9s","J8s",
    "T9s","T9o","T8s",
    "98s","98o","97s",
    "87s","87o","86s",
    "76s","75s",
    "65s","64s",
    "54s",
}


def _hand_key(cards):
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    if RANKS.index(r1) < RANKS.index(r2):
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def _equity_fast(hole, board, n_sims=200):
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

    key = _hand_key(hole)

    def raise_to(mult):
        total = max(int(pot * mult) + game_state["current_bet"], min_r)
        total = min(total, stack + bet_in)
        if total - bet_in >= stack:
            return {"action": "all_in"}
        return {"action": "raise", "amount": total}

    # ── Preflop ──────────────────────────────────────────────────────────────
    if street == "preflop":
        if key not in LAG_OPEN:
            if can_chk:
                return {"action": "check"}
            # Limp-call wide; fold to big raises out of range
            if owed <= 200:   # <= 2BB, basically a limp
                return {"action": "call"}
            return {"action": "fold"}

        # In range: attack
        if owed == 0 or owed <= 100:   # no raise yet
            # Open raise or attack a limp
            mult = 4.0 if owed <= 100 else 2.5
            return raise_to(mult)

        # Facing a raise: 3-bet top 20% of our range, call middle, fold bottom
        top_of_range = {"AA","KK","QQ","JJ","TT","AKs","AKo","AQs","AQo","A5s","A4s"}
        if key in top_of_range:
            return raise_to(3.0)
        if RANKS.index(key[0]) >= RANKS.index("T"):
            return {"action": "call"}
        return {"action": "fold"}

    # ── Postflop ─────────────────────────────────────────────────────────────
    eq = _equity_fast(hole, board)
    pot_odds = owed / (pot + owed) if owed > 0 else 0

    # Street-based barrel frequency: flop=always, turn=75%, river=55%
    barrel_freq = {"flop": 1.0, "turn": 0.75, "river": 0.55}.get(street, 0.6)

    if can_chk:
        if eq > 0.3 and random.random() < barrel_freq:
            # Size up: 75-100% pot to maximise fold equity
            mult = random.uniform(0.75, 1.0)
            return raise_to(mult)
        return {"action": "check"}

    # Facing a bet
    if eq > 0.65:
        return raise_to(random.uniform(0.8, 1.2))   # raise for value/charge draws
    if eq > pot_odds + 0.02:
        return {"action": "call"}
    # Float IP on flop occasionally
    if eq > 0.30 and random.random() < 0.3:
        return {"action": "call"}
    return {"action": "fold"}
