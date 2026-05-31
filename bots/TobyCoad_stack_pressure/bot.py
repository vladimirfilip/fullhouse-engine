"""
Stack Pressure — plays based on Stack-to-Pot Ratio (SPR).

Core logic:
  - SPR = effective stack / pot.
  - Low SPR (<4): commit or fold; no half-measures. Jam with any reasonable equity.
  - Medium SPR (4-12): standard bet sizing, value-focused.
  - High SPR (>12): cautious; only commit with very strong hands; probe with small bets.
  - Preflop: adjusts aggressiveness based on effective stack depth vs blinds.
  - Exploits short-stack situations ruthlessly; avoids bloating pots without nutted hands
    at high SPR.
"""

import eval7
import random

BOT_NAME = "StackPressure"

RANKS = "23456789TJQKA"

PREMIUM = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
STRONG  = {"TT", "99", "88", "AQs", "AQo", "AJs", "KQs", "KQo"}


def _hand_key(cards):
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    if RANKS.index(r1) < RANKS.index(r2):
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def _equity_fast(hole, board, n_sims=220):
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


def _spr(state):
    my_stack = state["your_stack"]
    opp_stacks = [p["stack"] for p in state["players"]
                  if p["seat"] != state["seat_to_act"] and not p["is_folded"]]
    eff_stack = min(my_stack, min(opp_stacks)) if opp_stacks else my_stack
    return eff_stack / max(state["pot"], 1)


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
    spr = _spr(game_state)

    def raise_to(mult):
        total = max(int(pot * mult) + game_state["current_bet"], min_r)
        total = min(total, stack + bet_in)
        if total - bet_in >= stack:
            return {"action": "all_in"}
        return {"action": "raise", "amount": total}

    # ── Preflop ──────────────────────────────────────────────────────────────
    if street == "preflop":
        # Deep stacks: only premium; short stacks: widen to apply pressure
        effective_bb = stack / 100   # stack in big blinds
        if effective_bb < 20:
            # Short: push/fold with any pair or A-high
            tier = 3 if key in PREMIUM else (2 if key in STRONG else 0)
            if RANKS.index(key[0]) >= RANKS.index("T") or len(key) == 2:
                tier = max(tier, 1)
        else:
            tier = 3 if key in PREMIUM else (2 if key in STRONG else 0)

        if tier >= 2:
            return raise_to(2.5)
        if tier == 1 and effective_bb < 20:
            return {"action": "all_in"}   # short-stack shove
        if can_chk:
            return {"action": "check"}
        if owed <= 100:   # cheap limp-call
            return {"action": "call"}
        return {"action": "fold"}

    # ── Postflop ─────────────────────────────────────────────────────────────
    eq = _equity_fast(hole, board)
    pot_odds = owed / (pot + owed) if owed > 0 else 0

    # Low SPR: commit or fold
    if spr < 4:
        if can_chk:
            if eq > 0.40:
                return {"action": "all_in"}
            return {"action": "check"}
        if eq > pot_odds + 0.05:
            return {"action": "all_in"}
        return {"action": "fold"}

    # Medium SPR: standard value play
    if spr < 12:
        if can_chk:
            if eq > 0.55:
                return raise_to(0.65)
            return {"action": "check"}
        if eq > 0.65 and random.random() < 0.35:
            return raise_to(0.8)
        if eq > pot_odds:
            return {"action": "call"}
        return {"action": "fold"}

    # High SPR: tight, only commit with strong hands
    if can_chk:
        if eq > 0.70:
            return raise_to(0.33)   # small probe with nutted hand
        return {"action": "check"}

    if eq > 0.75:
        return raise_to(0.6)
    if eq > pot_odds + 0.10:   # need big edge to call at high SPR
        return {"action": "call"}
    return {"action": "fold"}
