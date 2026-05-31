"""Adversarial test opponent for equity/pot-odds bots.

This is not a submission candidate. It is a benchmark opponent designed to
stress bots that rely heavily on random-hand Monte Carlo equity and pot odds.
"""

import hashlib

BOT_NAME = "Anti Monte Carlo"

RANK_VALUE = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
PREMIUM = {"AA", "KK", "QQ", "JJ", "TT", "AK", "AQ"}
BROADWAY = {"AJ", "AT", "KQ", "KJ", "QJ", "JT"}


def decide(state):
    if state.get("type") == "warmup":
        return {"action": "check"}

    if state["street"] == "preflop":
        return _preflop(state)

    strength = _made_strength(state)
    owed = state["amount_owed"]
    pot = max(1, state["pot"])

    if state["can_check"]:
        if strength >= 3:
            return _raise_to(state, 1.15)
        if strength >= 2 and _roll(state, "thin") < 0.65:
            return _raise_to(state, 0.75)
        if _dry_board(state) and _roll(state, "dry_pressure") < 0.22:
            return _raise_to(state, 0.70)
        return {"action": "check"}

    pot_odds = owed / max(1, pot + owed)
    if strength >= 4:
        return _raise_to(state, 1.30)
    if strength >= 2 and pot_odds <= 0.32:
        return {"action": "call"}
    if owed <= max(20, pot * 0.04):
        return {"action": "call"}
    return {"action": "fold"}


def _preflop(state):
    cards = state["your_cards"]
    r1, r2 = cards[0][0], cards[1][0]
    high, low = sorted((r1, r2), key=lambda r: RANK_VALUE[r], reverse=True)
    label = high + low
    pair = high == low
    suited = cards[0][1] == cards[1][1]
    owed = state["amount_owed"]
    pot = max(1, state["pot"])

    if pair and high in {"A", "K", "Q", "J", "T"}:
        return _raise_to(state, 1.45)
    if label in PREMIUM:
        return _raise_to(state, 1.25)
    if suited and (label in BROADWAY or high == "A"):
        if owed <= pot * 0.20:
            return _raise_to(state, 0.70) if state["can_check"] else {"action": "call"}
    if pair and owed <= pot * 0.18:
        return {"action": "call"}
    if state["can_check"]:
        return {"action": "check"}
    if owed <= max(40, pot * 0.06):
        return {"action": "call"}
    return {"action": "fold"}


def _made_strength(state):
    ranks = [card[0] for card in state["your_cards"] + state.get("community_cards", [])]
    suits = [card[1] for card in state["your_cards"] + state.get("community_cards", [])]
    counts = sorted((ranks.count(rank) for rank in set(ranks)), reverse=True)
    flushish = max((suits.count(suit) for suit in set(suits)), default=0) >= 5
    if flushish:
        return 5
    if counts and counts[0] >= 3:
        return 4
    if len(counts) >= 2 and counts[0] == 2 and counts[1] == 2:
        return 3
    if counts and counts[0] == 2:
        pair_rank = max(RANK_VALUE[rank] for rank in set(ranks) if ranks.count(rank) == 2)
        return 2 if pair_rank >= 10 else 1
    high = max(RANK_VALUE[rank] for rank in ranks)
    return 1 if high >= 13 else 0


def _dry_board(state):
    board = state.get("community_cards", [])
    if len(board) < 3:
        return False
    ranks = [card[0] for card in board[:3]]
    suits = [card[1] for card in board[:3]]
    values = sorted(RANK_VALUE[rank] for rank in ranks)
    return len(set(suits)) == 3 and len(set(ranks)) == 3 and values[-1] - values[0] > 4


def _raise_to(state, fraction):
    current = state["your_bet_this_street"]
    target = int(state["current_bet"] + max(1, state["pot"]) * fraction)
    target = max(target, state["min_raise_to"])
    target = min(target, current + state["your_stack"])
    if target <= current:
        return {"action": "call"} if state["amount_owed"] else {"action": "check"}
    return {"action": "raise", "amount": target}


def _roll(state, salt):
    key = "|".join(
        [
            salt,
            str(state.get("hand_id", "")),
            str(state.get("seat_to_act", "")),
            state.get("street", ""),
            ",".join(state.get("your_cards", [])),
            ",".join(state.get("community_cards", [])),
            str(len(state.get("action_log", []))),
        ]
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)
