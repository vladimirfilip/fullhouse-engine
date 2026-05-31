"""Blueprint + exploit bot for Fullhouse Hackathon."""

import json
import math
import os
import random


BOT_NAME = "Dominic Blueprint"
BOT_AVATAR = "robot_1"

RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {rank: index + 2 for index, rank in enumerate(RANK_ORDER)}
PREMIUM_PAIRS = {"AA", "KK", "QQ", "JJ", "TT"}
PREMIUM_BROADWAYS = {"AKs", "AKo", "AQs", "AQo", "AJs", "KQs"}
STRONG_BROADWAYS = {"ATs", "KJs", "QJs", "KQo", "AJo"}
SPECULATIVE = {"T9s", "98s", "87s", "76s", "65s", "54s", "JTs", "QTs"}
STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def _load_blueprint():
    data_dir = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
    path = os.path.join(data_dir, "blueprint.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


BLUEPRINT = _load_blueprint()
RFI_EXPLOIT_FOLD_RATE = float(os.environ.get("DOMINIC_RFI_EXPLOIT_FOLD_RATE", "0.30"))
RIVER_PAIR_OVERBET_FOLD_AGGRESSION = float(os.environ.get("DOMINIC_RIVER_PAIR_OVERBET_FOLD_AGGRESSION", "0.70"))
RIVER_WEAK_PAIR_FOLD_AGGRESSION = float(os.environ.get("DOMINIC_RIVER_WEAK_PAIR_FOLD_AGGRESSION", "0.55"))


def decide(game_state):
    """Return one legal action within the Fullhouse sandbox rules."""
    try:
        if game_state.get("type") == "warmup":
            return {"action": "check"}
        profile = _build_opponent_profile(game_state)
        if game_state.get("street") == "preflop":
            action = _preflop_policy(game_state, profile)
        else:
            action = _postflop_policy(game_state, profile)
        return _legalize(action, game_state)
    except Exception:
        return _fallback_action(game_state)


def _fallback_action(state):
    if state.get("can_check"):
        return {"action": "check"}
    owed = int(state.get("amount_owed") or 0)
    pot = max(1, int(state.get("pot") or 1))
    if owed > 0 and owed <= pot * 0.08:
        return {"action": "call"}
    return {"action": "fold"}


def _preflop_policy(state, profile):
    hand = _starting_hand_key(state.get("your_cards", []))
    strength = _preflop_strength(hand)
    owed = int(state.get("amount_owed") or 0)
    pot = max(1, int(state.get("pot") or 1))
    stack = int(state.get("your_stack") or 0)
    active = _active_player_count(state)
    position_label = _position_label(state)
    position = _position_score(state)
    aggression = profile["aggression"]
    fold_rate = profile["fold_rate"]
    total_stack = stack + int(state.get("your_bet_this_street") or 0)

    if owed > 0 and total_stack <= 1200:
        if strength >= 0.86:
            return {"action": "all_in"} if stack <= owed * 1.5 else {"action": "call"}
        if owed <= pot * 0.08 and strength >= 0.62:
            return {"action": "call"}
        return _check_or_fold(state)

    if strength >= 0.88 and (hand in PREMIUM_PAIRS or owed <= pot * 0.55):
        return {"action": "raise", "amount": _raise_to(state, 3.2 if owed else 2.7)}

    if stack <= owed:
        if strength >= 0.68 or (strength >= 0.58 and pot >= owed * 2):
            return {"action": "all_in"}
        return _check_or_fold(state)

    if owed == 0:
        if position_label == "BB" and strength < 0.88:
            return {"action": "check"}
        in_rfi_range = _in_rfi_range(hand, position_label)
        open_cutoff = 0.47 - 0.10 * position - 0.08 * fold_rate
        if in_rfi_range or (fold_rate > RFI_EXPLOIT_FOLD_RATE and strength >= open_cutoff):
            size = 2.25 + min(1.0, active / 6.0)
            return {"action": "raise", "amount": _raise_to(state, size)}
        return {"action": "check"}

    call_price = owed / float(pot + owed)
    defend_bonus = 0.08 * position + 0.05 * fold_rate - 0.09 * aggression
    threshold = call_price + 0.24 - defend_bonus

    if owed > pot * 0.55 and hand not in PREMIUM_PAIRS and strength >= 0.76:
        return {"action": "call"}
    if strength >= 0.76 and owed <= pot * 1.6:
        return {"action": "raise", "amount": _raise_to(state, 3.0)}
    if strength >= threshold:
        return {"action": "call"}
    if state.get("can_check"):
        return {"action": "check"}
    return {"action": "fold"}


def _postflop_policy(state, profile):
    made = _made_hand_score(state.get("your_cards", []), state.get("community_cards", []))
    made_profile = _made_hand_profile(state.get("your_cards", []), state.get("community_cards", []))
    texture = _board_texture(state.get("community_cards", []))
    draw = _draw_score(state.get("your_cards", []), state.get("community_cards", []))
    owed = int(state.get("amount_owed") or 0)
    pot = max(1, int(state.get("pot") or 1))
    street = state.get("street", "flop")
    position = _position_score(state)
    aggression = profile["aggression"]
    fold_rate = profile["fold_rate"]
    stack = int(state.get("your_stack") or 0)
    spr = stack / float(max(1, pot))

    equity_proxy = made + draw * (0.55 if street != "river" else 0.05)
    equity_proxy -= texture["paired_penalty"]
    equity_proxy += 0.04 * position
    equity_proxy = _clamp(equity_proxy, 0.0, 1.0)

    if owed == 0 or state.get("can_check"):
        if equity_proxy >= 0.78:
            return {"action": "raise", "amount": _street_bet(state, 0.72 if spr > 2.0 else 0.95)}
        if equity_proxy >= 0.58 and profile["calling_rate"] > 0.38:
            return {"action": "raise", "amount": _street_bet(state, 0.55)}
        if equity_proxy < 0.34 and fold_rate > 0.42 and street in ("flop", "turn"):
            return {"action": "raise", "amount": _street_bet(state, 0.45)}
        return {"action": "check"}

    price = owed / float(pot + owed)
    risk_adjustment = 0.07 * aggression + (0.05 if street == "river" else 0.0)
    continue_threshold = price + 0.10 + risk_adjustment

    if street == "river" and made_profile["category"] == "one_pair":
        if owed >= pot * 0.75 and aggression < RIVER_PAIR_OVERBET_FOLD_AGGRESSION:
            return {"action": "fold"}
        if (
            made_profile["pair_rank"] < made_profile["top_board_rank"]
            and owed >= pot * 0.45
            and aggression < RIVER_WEAK_PAIR_FOLD_AGGRESSION
        ):
            return {"action": "fold"}

    if equity_proxy >= 0.84 and owed <= pot * 1.5:
        return {"action": "raise", "amount": _street_bet(state, 1.05)}
    if equity_proxy >= continue_threshold:
        return {"action": "call"}
    if draw >= 0.38 and street in ("flop", "turn") and owed <= pot * 0.45:
        return {"action": "call"}
    return {"action": "fold"}


def _legalize(action, state):
    act = str(action.get("action", "fold")).lower()
    if act not in {"fold", "check", "call", "raise", "all_in"}:
        return _fallback_action(state)

    can_check = bool(state.get("can_check"))
    owed = int(state.get("amount_owed") or 0)
    stack = max(0, int(state.get("your_stack") or 0))
    already_bet = max(0, int(state.get("your_bet_this_street") or 0))
    max_total = stack + already_bet
    min_raise_to = int(state.get("min_raise_to") or 0)

    if act == "fold" and can_check:
        return {"action": "check"}
    if act == "check":
        return {"action": "check"} if can_check else {"action": "call"}
    if act == "call":
        return {"action": "check"} if owed == 0 else {"action": "call"}
    if act == "all_in":
        return {"action": "all_in"} if stack > 0 else _check_or_fold(state)
    if act == "raise":
        amount = int(action.get("amount") or 0)
        if stack <= 0:
            return _check_or_fold(state)
        if max_total < min_raise_to:
            return {"action": "all_in"}
        amount = max(amount, min_raise_to)
        if amount >= max_total:
            return {"action": "all_in"}
        return {"action": "raise", "amount": amount}
    return _fallback_action(state)


def _check_or_fold(state):
    return {"action": "check"} if state.get("can_check") else {"action": "fold"}


def _raise_to(state, multiple):
    owed = int(state.get("amount_owed") or 0)
    current_bet = int(state.get("current_bet") or 0)
    min_raise = int(state.get("min_raise_to") or 0)
    pot = max(1, int(state.get("pot") or 1))
    stack = int(state.get("your_stack") or 0)
    already = int(state.get("your_bet_this_street") or 0)
    base = current_bet + owed + int(multiple * max(100, pot * 0.55))
    if owed == 0:
        base = int(max(min_raise, pot * multiple))
    return min(max(base, min_raise), stack + already)


def _street_bet(state, pot_fraction):
    pot = max(1, int(state.get("pot") or 1))
    min_raise = int(state.get("min_raise_to") or 0)
    stack = int(state.get("your_stack") or 0)
    already = int(state.get("your_bet_this_street") or 0)
    target = int(pot * pot_fraction)
    return min(max(target, min_raise), stack + already)


def _build_opponent_profile(state):
    log = list(state.get("match_action_log") or [])[-200:]
    hero_seat = state.get("seat_to_act")
    opponent_actions = [row for row in log if row.get("seat") != hero_seat]
    total = max(1, len(opponent_actions))
    raises = sum(1 for row in opponent_actions if row.get("action") in ("raise", "all_in"))
    calls = sum(1 for row in opponent_actions if row.get("action") == "call")
    folds = sum(1 for row in opponent_actions if row.get("action") == "fold")
    overbets = sum(
        1
        for row in opponent_actions
        if row.get("action") in ("raise", "all_in") and int(row.get("amount") or 0) >= 800
    )

    aggression = raises / float(total)
    fold_rate = folds / float(total)
    calling_rate = calls / float(total)
    overbet_rate = overbets / float(total)
    if aggression > 0.5 and fold_rate > 0.25:
        table_type = "volatile"
    elif aggression > 0.45:
        table_type = "maniac"
    elif calling_rate > 0.45:
        table_type = "sticky"
    elif fold_rate > 0.45:
        table_type = "overfolding"
    else:
        table_type = "balanced"

    return {
        "aggression": aggression,
        "fold_rate": fold_rate,
        "calling_rate": calling_rate,
        "overbet_rate": overbet_rate,
        "table_type": table_type,
    }


def _starting_hand_key(cards):
    if len(cards) != 2:
        return "72o"
    first, second = cards[0], cards[1]
    r1, r2 = first[0], second[0]
    suited = first[1] == second[1]
    ordered = sorted([r1, r2], key=lambda rank: RANK_VALUE.get(rank, 0), reverse=True)
    if ordered[0] == ordered[1]:
        return ordered[0] + ordered[1]
    return ordered[0] + ordered[1] + ("s" if suited else "o")


def _preflop_strength(hand):
    try:
        return float(BLUEPRINT.get("preflop", {}).get(hand, {}).get("strength"))
    except (TypeError, ValueError):
        pass
    if hand in PREMIUM_PAIRS:
        return {"AA": 1.0, "KK": 0.97, "QQ": 0.94, "JJ": 0.90, "TT": 0.86}[hand]
    if hand in PREMIUM_BROADWAYS:
        return 0.82
    if hand in STRONG_BROADWAYS:
        return 0.70
    if hand in SPECULATIVE:
        return 0.58
    if len(hand) == 2 and hand[0] == hand[1]:
        return 0.52 + RANK_VALUE.get(hand[0], 2) / 40.0
    high = RANK_VALUE.get(hand[0], 2)
    low = RANK_VALUE.get(hand[1], 2)
    suited_bonus = 0.08 if hand.endswith("s") else 0.0
    connected_bonus = 0.06 if abs(high - low) <= 1 else 0.0
    ace_bonus = 0.10 if hand[0] == "A" else 0.0
    return _clamp((high + low) / 30.0 + suited_bonus + connected_bonus + ace_bonus - 0.18, 0.05, 0.76)


def _made_hand_score(hole_cards, board_cards):
    cards = list(hole_cards or []) + list(board_cards or [])
    if len(cards) < 5:
        return 0.32 + min(0.30, _preflop_strength(_starting_hand_key(hole_cards or [])) * 0.35)

    ranks = [card[0] for card in cards]
    suits = [card[1] for card in cards]
    counts = sorted([ranks.count(rank) for rank in set(ranks)], reverse=True)
    is_flush = max(suits.count(suit) for suit in set(suits)) >= 5
    is_straight = _has_straight(ranks)

    if is_flush and is_straight:
        return 0.98
    if counts[0] == 4:
        return 0.96
    if counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
        return 0.92
    if is_flush:
        return 0.88
    if is_straight:
        return 0.84
    if counts[0] == 3:
        return 0.73
    if counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        return 0.62
    if counts[0] == 2:
        pair_rank = max(RANK_VALUE.get(rank, 0) for rank in set(ranks) if ranks.count(rank) == 2)
        return 0.42 + pair_rank / 42.0
    high = max(RANK_VALUE.get(rank, 0) for rank in ranks)
    return 0.16 + high / 70.0


def _made_hand_profile(hole_cards, board_cards):
    cards = list(hole_cards or []) + list(board_cards or [])
    if not cards:
        return {"category": "high_card", "pair_rank": 0, "top_board_rank": 0}
    ranks = [card[0] for card in cards]
    suits = [card[1] for card in cards]
    counts = sorted([ranks.count(rank) for rank in set(ranks)], reverse=True)
    board_ranks = [card[0] for card in board_cards or []]
    top_board_rank = max([RANK_VALUE.get(rank, 0) for rank in board_ranks] or [0])
    is_flush = bool(suits) and max(suits.count(suit) for suit in set(suits)) >= 5
    is_straight = _has_straight(ranks)
    if is_flush and is_straight:
        return {"category": "straight_flush", "pair_rank": 0, "top_board_rank": top_board_rank}
    if counts[0] == 4:
        return {"category": "quads", "pair_rank": 0, "top_board_rank": top_board_rank}
    if counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
        return {"category": "full_house", "pair_rank": 0, "top_board_rank": top_board_rank}
    if is_flush:
        return {"category": "flush", "pair_rank": 0, "top_board_rank": top_board_rank}
    if is_straight:
        return {"category": "straight", "pair_rank": 0, "top_board_rank": top_board_rank}
    if counts[0] == 3:
        return {"category": "trips", "pair_rank": 0, "top_board_rank": top_board_rank}
    pair_ranks = [RANK_VALUE.get(rank, 0) for rank in set(ranks) if ranks.count(rank) == 2]
    if len(pair_ranks) >= 2:
        return {"category": "two_pair", "pair_rank": max(pair_ranks), "top_board_rank": top_board_rank}
    if len(pair_ranks) == 1:
        return {"category": "one_pair", "pair_rank": pair_ranks[0], "top_board_rank": top_board_rank}
    return {"category": "high_card", "pair_rank": 0, "top_board_rank": top_board_rank}


def _draw_score(hole_cards, board_cards):
    if len(board_cards or []) >= 5:
        return 0.0
    cards = list(hole_cards or []) + list(board_cards or [])
    ranks = [card[0] for card in cards]
    suits = [card[1] for card in cards]
    flush_draw = 0.0
    if suits:
        flush_draw = 0.32 if max(suits.count(suit) for suit in set(suits)) == 4 else 0.0
    straight_draw = 0.26 if _near_straight(ranks) else 0.0
    overcards = sum(1 for card in hole_cards or [] if RANK_VALUE.get(card[0], 0) >= 12) * 0.05
    return _clamp(flush_draw + straight_draw + overcards, 0.0, 0.55)


def _board_texture(board_cards):
    board = list(board_cards or [])
    ranks = [card[0] for card in board]
    paired = any(ranks.count(rank) > 1 for rank in set(ranks))
    monotone = len(board) >= 3 and len({card[1] for card in board[:3]}) == 1
    return {
        "paired_penalty": 0.05 if paired else 0.0,
        "wet_bonus": 0.04 if monotone or _near_straight(ranks) else 0.0,
    }


def _has_straight(ranks):
    values = {RANK_VALUE.get(rank, 0) for rank in ranks}
    if 14 in values:
        values.add(1)
    for start in range(1, 11):
        if all(value in values for value in range(start, start + 5)):
            return True
    return False


def _near_straight(ranks):
    values = {RANK_VALUE.get(rank, 0) for rank in ranks}
    if 14 in values:
        values.add(1)
    for start in range(1, 11):
        if sum(1 for value in range(start, start + 5) if value in values) >= 4:
            return True
    return False


def _active_player_count(state):
    players = state.get("players") or []
    return max(2, sum(1 for player in players if not player.get("is_folded") and player.get("stack", 0) >= 0))


def _position_label(state):
    players = state.get("players") or []
    n_players = max(2, len(players))
    seat = int(state.get("seat_to_act") or 0)
    sb_seat, bb_seat = _blind_seats(state)
    if sb_seat is None or bb_seat is None:
        labels = ["LJ", "HJ", "CO", "BTN", "SB", "BB"][-n_players:]
        index = min(len(labels) - 1, max(0, int(_position_score_from_seat(seat, n_players) * (len(labels) - 1))))
        return labels[index]

    order = [((bb_seat + offset) % n_players) for offset in range(1, n_players + 1)]
    labels = ["LJ", "HJ", "CO", "BTN", "SB", "BB"][-n_players:]
    try:
        return labels[order.index(seat)]
    except ValueError:
        return "BTN"


def _blind_seats(state):
    sb_seat = None
    bb_seat = None
    for action in state.get("action_log") or []:
        if action.get("action") == "small_blind":
            sb_seat = int(action.get("seat"))
        elif action.get("action") == "big_blind":
            bb_seat = int(action.get("seat"))
    return sb_seat, bb_seat


def _position_score(state):
    label = _position_label(state)
    scores = {"LJ": 0.0, "HJ": 0.2, "CO": 0.45, "BTN": 0.75, "SB": 0.55, "BB": 0.35}
    if label in scores:
        return scores[label]
    return _position_score_from_seat(int(state.get("seat_to_act") or 0), max(2, len(state.get("players") or [])))


def _position_score_from_seat(seat, n_players):
    return _clamp(seat / float(max(1, n_players - 1)), 0.0, 1.0)


def _in_rfi_range(hand, position_label):
    return hand in set(BLUEPRINT.get("rfi_ranges", {}).get(position_label, {}).get("hands", []))


def _legacy_position_score(state):
    players = state.get("players") or []
    seat = int(state.get("seat_to_act") or 0)
    n_players = max(2, len(players))
    return _clamp(seat / float(max(1, n_players - 1)), 0.0, 1.0)


def _clamp(value, low, high):
    return max(low, min(high, value))
