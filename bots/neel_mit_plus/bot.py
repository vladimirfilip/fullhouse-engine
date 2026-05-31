"""Neel baseline bot.

Fast tournament-oriented strategy:
- table-driven preflop ranges
- Monte Carlo equity postflop when eval7 is available
- pot-odds aware calls
- value bets and occasional semi-bluffs from late position
"""

import hashlib
from collections import Counter

try:
    import eval7
except Exception:  # pragma: no cover - sandbox should provide eval7
    try:
        from treys import Card as _TreysCard
        from treys import Deck as _TreysDeck
        from treys import Evaluator as _TreysEvaluator

        class _CompatCard:
            def __init__(self, text):
                self.text = str(text)
                self._treys = _TreysCard.new(self.text[0] + self.text[1].lower())

            def __str__(self):
                return self.text

            def __eq__(self, other):
                return isinstance(other, _CompatCard) and self.text == other.text

            def __hash__(self):
                return hash(self.text)

        class _CompatDeck:
            def __init__(self):
                raw = _TreysDeck().cards
                self.cards = [_CompatCard(_TreysCard.int_to_str(c)[0].upper() + _TreysCard.int_to_str(c)[1]) for c in raw]

            def shuffle(self):
                import random

                random.shuffle(self.cards)

            def peek(self, n):
                return self.cards[:n]

        class _CompatEval7:
            Card = _CompatCard
            Deck = _CompatDeck
            _evaluator = _TreysEvaluator()
            _names = {
                1: "Straight Flush",
                2: "Four of a Kind",
                3: "Full House",
                4: "Flush",
                5: "Straight",
                6: "Trips",
                7: "Two Pair",
                8: "Pair",
                9: "High Card",
            }

            @classmethod
            def evaluate(cls, cards):
                treys_cards = [c._treys if isinstance(c, _CompatCard) else _CompatCard(c)._treys for c in cards]
                raw = cls._evaluator.evaluate(treys_cards[2:], treys_cards[:2])
                return 7463 - raw

            @classmethod
            def handtype(cls, score):
                raw = 7463 - score
                return cls._names.get(cls._evaluator.get_rank_class(raw), "Unknown")

        eval7 = _CompatEval7
    except Exception:
        eval7 = None


BOT_NAME = "Neel MIT Plus"

RANK_VALUE = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
PREMIUM_PAIRS = {"A", "K", "Q", "J", "T"}
STRONG_BROADWAY = {"AK", "AQ", "AJ", "KQ"}
PLAYABLE_BROADWAY = {"AT", "KJ", "KT", "QJ", "QT", "JT"}
SMALL_PAIRS = {"9", "8", "7", "6", "5", "4", "3", "2"}

CONFIG = {
    'value_bar_multi': 0.64,
    'value_bar_shorthanded': 0.55,
    'call_add_multi': 0.11,
    'call_add_shorthanded': 0.05,
    'check_bluff_equity': 0.45,
    'check_bluff_position': 0.62,
    'check_bluff_freq': 0.06,
    'check_bluff_bet': 0.45,
    'value_bet': 0.9,
    'facing_bet_raise': 1.05,
    'strong_pre_bet': 1.25,
    'broadway_pre_bet': 0.95,
    'steal_pre_bet': 0.55,
    'playable_late_position': 0.58,
    'playable_call_pot': 0.15,
    'dry_cbet_bonus': 1.45,
    'wet_cbet_penalty': 0.55,
    'station_value_discount': 0.05,
    'maniac_call_discount': 0.05,
    'nit_bluff_bonus': 1.8,
}

AX_SUITED = {"A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"}
LATE_OFFSUIT = {"A9", "KQ", "KJ", "QJ", "JT"}
SC_BROAD = {"T9", "98", "87", "76", "65", "54"}


def decide(state):
    if state.get("type") == "warmup":
        return {"action": "check"}

    street = state["street"]
    owed = state["amount_owed"]
    pot = max(0, state["pot"])
    stack = state["your_stack"]

    if stack <= 0:
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    if street == "preflop":
        return _preflop(state)

    equity = _postflop_equity(state)
    active_opponents = _active_opponents(state)
    position = _position_score(state)
    texture = _board_texture(state.get("community_cards", []))
    profile = _table_profile(state)
    was_aggressor = _was_preflop_aggressor(state)

    # Multi-way pots need tighter betting; heads-up can press thinner edges.
    value_bar = CONFIG["value_bar_multi"] if active_opponents >= 3 else CONFIG["value_bar_shorthanded"]
    call_bar = _pot_odds(owed, pot) + (
        CONFIG["call_add_multi"] if active_opponents >= 3 else CONFIG["call_add_shorthanded"]
    )
    bluff_freq = CONFIG["check_bluff_freq"]

    if profile == "station":
        value_bar -= CONFIG["station_value_discount"]
        bluff_freq *= 0.15
    elif profile == "maniac":
        call_bar -= CONFIG["maniac_call_discount"]
        bluff_freq *= 0.45
    elif profile == "nit":
        bluff_freq *= CONFIG["nit_bluff_bonus"]

    if texture == "dry":
        bluff_freq *= CONFIG["dry_cbet_bonus"]
    elif texture == "wet":
        bluff_freq *= CONFIG["wet_cbet_penalty"]
        value_bar += 0.03

    if was_aggressor and state["street"] == "flop" and texture in ("dry", "paired"):
        bluff_freq *= 1.35

    if state["can_check"]:
        if equity >= value_bar:
            return _raise_to_fraction(state, CONFIG["value_bet"])
        if (
            equity >= CONFIG["check_bluff_equity"]
            and position > CONFIG["check_bluff_position"]
            and _roll(state, "check_bluff") < CONFIG["check_bluff_freq"]
        ):
            return _raise_to_fraction(state, CONFIG["check_bluff_bet"])
        return {"action": "check"}

    if equity >= max(value_bar + 0.06, call_bar + 0.18):
        return _raise_to_fraction(state, CONFIG["facing_bet_raise"])

    if equity >= call_bar:
        return {"action": "call"}

    # Tournament chip-delta format rewards survival less than stack growth, but
    # tiny prices are still worth realizing with almost any live draw.
    if owed <= max(20, pot * 0.06):
        return {"action": "call"}

    return {"action": "fold"}


def _preflop(state):
    cards = state["your_cards"]
    r1, r2 = cards[0][0], cards[1][0]
    suited = cards[0][1] == cards[1][1]
    high, low = sorted((r1, r2), key=lambda r: RANK_VALUE[r], reverse=True)
    label = high + low
    pair = high == low
    owed = state["amount_owed"]
    pot = max(1, state["pot"])
    position = _position_score(state)
    stack = state["your_stack"]
    profile = _table_profile(state)
    is_late = position >= 0.70
    has_raise = _preflop_raise_count(state) > 0

    if pair and high in PREMIUM_PAIRS:
        return _raise_to_fraction(state, CONFIG["strong_pre_bet"])
    if label in STRONG_BROADWAY and (suited or position > 0.25):
        return _raise_to_fraction(state, CONFIG["broadway_pre_bet"])

    if pair and high in SMALL_PAIRS:
        if owed <= pot * (0.22 + 0.10 * position):
            return {"action": "call"}
        return {"action": "check"} if state["can_check"] else {"action": "fold"}

    playable = (
        label in PLAYABLE_BROADWAY
        or (suited and RANK_VALUE[high] >= 10 and RANK_VALUE[low] >= 8)
        or (suited and RANK_VALUE[high] - RANK_VALUE[low] <= 2 and RANK_VALUE[high] >= 9)
    )
    if suited:
        playable = playable or label in AX_SUITED or (is_late and label in SC_BROAD)
    if is_late and not has_raise:
        playable = playable or label in LATE_OFFSUIT

    if playable and position > CONFIG["playable_late_position"]:
        steal_freq = 0.45
        if profile == "nit":
            steal_freq = 0.62
        elif profile == "maniac":
            steal_freq = 0.32
        if owed == 0 and _roll(state, "preflop_steal") < steal_freq:
            return _raise_to_fraction(state, CONFIG["steal_pre_bet"])
        if owed <= pot * CONFIG["playable_call_pot"]:
            return {"action": "call"}

    if state["can_check"]:
        return {"action": "check"}

    if owed <= min(80, stack * 0.025):
        return {"action": "call"}

    return {"action": "fold"}


def _postflop_equity(state):
    if eval7 is None:
        return _fallback_strength(state)

    hole = [eval7.Card(c) for c in state["your_cards"]]
    board = [eval7.Card(c) for c in state["community_cards"]]
    opponents = _active_opponents(state)

    if len(board) == 5:
        hero = eval7.evaluate(hole + board)
        # River without opponent hole cards is unknowable; use made-hand class
        # as a conservative proxy.
        return _river_score_to_equity(hero)

    trials = 90 if len(board) == 3 else 120
    wins = 0.0
    dead = set(hole + board)

    for _ in range(trials):
        deck = eval7.Deck()
        deck.cards = [card for card in deck.cards if card not in dead]
        deck.shuffle()

        draw = deck.peek(2 * opponents + (5 - len(board)))
        opp_cards = [draw[i * 2 : i * 2 + 2] for i in range(opponents)]
        runout = board + draw[2 * opponents :]
        hero_score = eval7.evaluate(hole + runout)
        opp_scores = [eval7.evaluate(opp + runout) for opp in opp_cards]
        best_opp = max(opp_scores) if opp_scores else -1

        if hero_score > best_opp:
            wins += 1.0
        elif hero_score == best_opp:
            ties = 1 + sum(1 for score in opp_scores if score == hero_score)
            wins += 1.0 / ties

    return wins / trials


def _fallback_strength(state):
    ranks = [c[0] for c in state["your_cards"] + state["community_cards"]]
    counts = sorted((ranks.count(r) for r in set(ranks)), reverse=True)
    if counts and counts[0] >= 3:
        return 0.72
    if counts and counts[0] == 2:
        return 0.48
    high = max(RANK_VALUE[r] for r in ranks)
    return 0.38 + (high - 10) * 0.03


def _river_score_to_equity(score):
    hand_name = str(eval7.handtype(score)).lower()
    if "straight flush" in hand_name or "quads" in hand_name:
        return 0.96
    if "full house" in hand_name:
        return 0.88
    if "flush" in hand_name:
        return 0.78
    if "straight" in hand_name:
        return 0.72
    if "trips" in hand_name:
        return 0.62
    if "two pair" in hand_name:
        return 0.52
    if "pair" in hand_name:
        return 0.34
    return 0.18


def _active_opponents(state):
    my_seat = state["seat_to_act"]
    count = 0
    for player in state["players"]:
        if player["seat"] == my_seat:
            continue
        if not player.get("is_folded") and player.get("stack", 0) >= 0:
            count += 1
    return max(1, count)


def _position_score(state):
    seat = state["seat_to_act"]
    n = len(state["players"])
    sb = None
    bb = None
    for entry in state.get("action_log", []):
        if entry.get("action") == "small_blind":
            sb = entry.get("seat")
        elif entry.get("action") == "big_blind":
            bb = entry.get("seat")
    if bb is None:
        players = [p for p in state["players"] if not p.get("is_folded")]
        if len(players) <= 1:
            return 1.0
        seats = [p["seat"] for p in players]
        return seats.index(seat) / max(1, len(seats) - 1)
    if n == 2:
        return 1.0 if seat == sb else 0.0
    button = (bb - 2) % n
    offset = (seat - button) % n
    scores = {
        0: 1.0,   # BTN
        1: 0.28,  # SB
        2: 0.12,  # BB
        3: 0.00,  # UTG
        4: 0.38,  # MP/HJ
        5: 0.76,  # CO
    }
    if n == 5 and offset == 4:
        return 0.76
    if n == 4 and offset == 3:
        return 0.70
    return scores.get(offset, 0.40)


def _board_texture(cards):
    if len(cards) < 3:
        return "none"
    flop = cards[:3]
    ranks = [c[0] for c in flop]
    suits = [c[1] for c in flop]
    if len(set(ranks)) < 3:
        return "paired"
    suit_counts = Counter(suits)
    two_tone = max(suit_counts.values()) >= 2
    values = sorted(RANK_VALUE[r] for r in ranks)
    connected = values[-1] - values[0] <= 4
    if two_tone and connected:
        return "wet"
    if two_tone or connected:
        return "semi"
    return "dry"


def _preflop_raise_count(state):
    return sum(
        1
        for entry in state.get("action_log", [])
        if entry.get("action") in ("raise", "all_in") and entry.get("street", "preflop") == "preflop"
    )


def _was_preflop_aggressor(state):
    mine = state["seat_to_act"]
    last = None
    for entry in state.get("action_log", []):
        if entry.get("action") in ("raise", "all_in") and entry.get("street", "preflop") == "preflop":
            last = entry.get("seat")
    return last == mine


def _table_profile(state):
    counts = {}
    for entry in state.get("match_action_log", [])[-220:]:
        bot_id = entry.get("bot_id")
        action = entry.get("action")
        if not bot_id or action in ("small_blind", "big_blind"):
            continue
        row = counts.setdefault(bot_id, {"t": 0, "r": 0, "c": 0, "f": 0})
        row["t"] += 1
        if action in ("raise", "all_in"):
            row["r"] += 1
        elif action == "call":
            row["c"] += 1
        elif action == "fold":
            row["f"] += 1

    profiles = []
    my_seat = state["seat_to_act"]
    for player in state.get("players", []):
        if player.get("seat") == my_seat or player.get("is_folded"):
            continue
        row = counts.get(player.get("bot_id"), {})
        total = row.get("t", 0)
        if total < 18:
            continue
        raise_rate = row["r"] / total
        call_rate = row["c"] / total
        fold_rate = row["f"] / total
        if raise_rate > 0.36:
            profiles.append("maniac")
        elif call_rate > 0.44 and raise_rate < 0.20:
            profiles.append("station")
        elif fold_rate > 0.55 and raise_rate < 0.22:
            profiles.append("nit")

    if "maniac" in profiles:
        return "maniac"
    if profiles.count("station") >= 2:
        return "station"
    if profiles.count("nit") >= 2:
        return "nit"
    return "unknown"


def _pot_odds(owed, pot):
    if owed <= 0:
        return 0.0
    return owed / max(1, pot + owed)


def _raise_to_fraction(state, fraction):
    stack = state["your_stack"]
    current = state["your_bet_this_street"]
    pot = max(1, state["pot"])
    target = int(state["current_bet"] + pot * fraction)
    target = max(target, state["min_raise_to"])
    target = min(target, stack + current)
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
