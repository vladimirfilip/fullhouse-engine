"""Neel baseline bot.

Fast tournament-oriented strategy:
- table-driven preflop ranges
- Monte Carlo equity postflop when eval7 is available
- pot-odds aware calls
- value bets and occasional semi-bluffs from late position
"""

import hashlib

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


BOT_NAME = "Neel Baseline"

RANK_VALUE = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
PREMIUM_PAIRS = {"A", "K", "Q", "J", "T"}
STRONG_BROADWAY = {"AK", "AQ", "AJ", "KQ"}
PLAYABLE_BROADWAY = {"AT", "KJ", "KT", "QJ", "QT", "JT"}
SMALL_PAIRS = {"9", "8", "7", "6", "5", "4", "3", "2"}

CONFIG = {
    "value_bar_multi": 0.64,
    "value_bar_shorthanded": 0.55,
    "call_add_multi": 0.11,
    "call_add_shorthanded": 0.05,
    "check_bluff_equity": 0.45,
    "check_bluff_position": 0.62,
    "check_bluff_freq": 0.06,
    "check_bluff_bet": 0.45,
    "value_bet": 0.90,
    "facing_bet_raise": 1.05,
    "strong_pre_bet": 1.25,
    "broadway_pre_bet": 0.95,
    "steal_pre_bet": 0.55,
    "playable_late_position": 0.58,
    "playable_call_pot": 0.15,
}


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

    # Multi-way pots need tighter betting; heads-up can press thinner edges.
    value_bar = CONFIG["value_bar_multi"] if active_opponents >= 3 else CONFIG["value_bar_shorthanded"]
    call_bar = _pot_odds(owed, pot) + (
        CONFIG["call_add_multi"] if active_opponents >= 3 else CONFIG["call_add_shorthanded"]
    )

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

    if playable and position > CONFIG["playable_late_position"]:
        if owed == 0 and _roll(state, "preflop_steal") < 0.45:
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
    players = [p for p in state["players"] if not p.get("is_folded")]
    if len(players) <= 1:
        return 1.0
    seats = [p["seat"] for p in players]
    return seats.index(state["seat_to_act"]) / max(1, len(seats) - 1)


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
