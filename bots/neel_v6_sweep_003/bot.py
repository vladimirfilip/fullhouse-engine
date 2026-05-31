"""FullHouseOpenCFR: solver-inspired offline bot for the Fullhouse engine.

This is not a live-site automation bot. It is a single-file tournament agent
for the local Fullhouse sandbox, distilled from common open poker-AI ideas:
action abstraction, range-conditioned equity, opponent modelling, and a compact
postflop digest adapted from open-source Discounted CFR solver outputs.
"""

try:
    import eval7
except ModuleNotFoundError:
    from treys import Card as _TreysCard
    from treys import Deck as _TreysDeck
    from treys import Evaluator as _TreysEvaluator
    import random as _compat_random

    class _CompatCard:
        def __init__(self, text):
            self.text = str(text)
            self._treys = _TreysCard.new(self.text[0] + self.text[1].lower())

        def __str__(self):
            return self.text

        def __repr__(self):
            return "Card(%r)" % self.text

        def __eq__(self, other):
            return isinstance(other, _CompatCard) and self.text == other.text

        def __hash__(self):
            return hash(self.text)

    class _CompatDeck:
        def __init__(self):
            self.cards = [
                _CompatCard(_TreysCard.int_to_str(card)[0] + _TreysCard.int_to_str(card)[1].upper())
                for card in _TreysDeck().cards
            ]

        def shuffle(self):
            _compat_random.shuffle(self.cards)

        def peek(self, n):
            return self.cards[:n]

    _EVALUATOR = _TreysEvaluator()
    _CLASS_NAMES = {
        1: "Straight Flush",
        2: "Four of a Kind",
        3: "Full House",
        4: "Flush",
        5: "Straight",
        6: "Three of a Kind",
        7: "Two Pair",
        8: "Pair",
        9: "High Card",
    }

    class eval7:
        Card = _CompatCard
        Deck = _CompatDeck

        @staticmethod
        def evaluate(cards):
            treys_cards = [
                card._treys if isinstance(card, _CompatCard) else _CompatCard(card)._treys
                for card in cards
            ]
            return 7463 - _EVALUATOR.evaluate(treys_cards[2:], treys_cards[:2])

        @staticmethod
        def handtype(score):
            return _CLASS_NAMES.get(_EVALUATOR.get_rank_class(7463 - score), "Unknown")
import json
import os
import random

BOT_NAME = "FullHousePartitionV5"

# Pseudo-harmonic action translation (Ganzfried & Sandholm, IJCAI 2013).
# Abstract bet sizes are expressed as opponent-bet / pot. Observed bets that
# fall between adjacent buckets are sampled to A with probability
#     f(x) = ((B - x)(1 + A)) / ((B - A)(1 + x))
# and to B otherwise. This closes the exploit of betting just-above-threshold.
_PSEUDO_HARMONIC_BUCKETS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5)

RANKS = "23456789TJQKA"
RANK_VALUE = {rank: idx + 2 for idx, rank in enumerate(RANKS)}
DECK = [eval7.Card(rank + suit) for rank in RANKS for suit in "shdc"]
BIG_BLIND = 100

PREMIUM = {"AA", "KK", "QQ", "AKs", "AKo"}
VALUE_3BET = PREMIUM | {"JJ", "TT", "AQs", "AQo", "AJs", "KQs"}
LATE_STEALS = {
    "99", "88", "77", "66", "55", "ATs", "A9s", "A8s", "A7s", "A5s",
    "A4s", "A3s", "A2s", "KJs", "KTs", "K9s", "QJs", "QTs", "JTs",
    "T9s", "98s", "87s", "76s", "KQo", "KJo", "QJo",
}
BLIND_DEFENDS = LATE_STEALS | {
    "44", "33", "22", "AJo", "ATo", "A9o", "KTo", "Q9s", "J9s", "T8s",
    "97s", "86s", "75s", "65s", "54s",
}
BLOCKER_BLUFFS = {"A5s", "A4s", "A3s", "A2s", "KTs", "K9s", "QTs", "JTs", "T9s"}

_SEEN_ACTIONS = {}
_OPPONENTS = {}
_PREFLOP_CACHE = {}


def _load_blueprint():
    data_dir = os.environ.get("BOT_DATA_DIR")
    paths = []
    if data_dir:
        paths.append(os.path.join(data_dir, "preflop_blueprint.json"))
    paths.append(os.path.join(os.path.dirname(__file__), "data", "preflop_blueprint.json"))

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and isinstance(data.get("hands"), dict):
                return data
        except Exception:
            pass
    return {"version": "embedded-fallback", "hands": {}}


_BLUEPRINT = _load_blueprint()


def _load_postflop_digest():
    data_dir = os.environ.get("BOT_DATA_DIR")
    paths = []
    if data_dir:
        paths.append(os.path.join(data_dir, "postflop_solver_digest.json"))
    paths.append(os.path.join(os.path.dirname(__file__), "data", "postflop_solver_digest.json"))

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and isinstance(data.get("scenarios"), dict):
                return data
        except Exception:
            pass
    return {"version": "no-postflop-digest", "scenarios": {}}


_POSTFLOP_SOLVER = _load_postflop_digest()


def _load_cfr_profile():
    data_dir = os.environ.get("BOT_DATA_DIR")
    paths = []
    if data_dir:
        paths.append(os.path.join(data_dir, "cfr_action_profile.json"))
    paths.append(os.path.join(os.path.dirname(__file__), "data", "cfr_action_profile.json"))

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and isinstance(data.get("textures"), dict):
                return data
        except Exception:
            pass
    return {
        "version": "embedded-cfr-profile-fallback",
        "defaults": {
            "probe": {
                "solver_weight": 0.72,
                "baseline": 0.03,
                "min_bucket_n": 4,
                "air_equity_floor": 0.36,
                "draw_bonus": 0.09,
                "value_bonus": 0.13,
                "heads_up_bonus": 0.03,
                "two_way_penalty": 0.08,
            },
            "sizes": {"small": 0.34, "medium": 0.54, "polar": 0.72, "jam_spr": 0.82},
            "vs_bet": {
                "continue_equity": 0.18,
                "price_weight": 0.92,
                "made_discount": 0.048,
                "draw_discount": 0.028,
                "river_tax": 0.045,
                "multiway_tax": 0.048,
                "raise_value_made": 5,
                "raise_value_equity": 0.78,
                "raise_draw_min": 4,
                "raise_freq": 0.12,
                "draw_peel_price": 0.20,
                "pair_peel_price": 0.15,
                "fold_price": 0.36,
            },
        },
        "textures": {},
    }


_CFR_PROFILE = _load_cfr_profile()


def _load_risk_profile():
    data_dir = os.environ.get("BOT_DATA_DIR")
    paths = []
    if data_dir:
        paths.append(os.path.join(data_dir, "risk_profile.json"))
    paths.append(os.path.join(os.path.dirname(__file__), "data", "risk_profile.json"))

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and isinstance(data.get("defaults"), dict):
                return data
        except Exception:
            pass
    return {
        "version": "embedded-risk-fallback",
        "defaults": {
            "large_risk_frac": 0.22,
            "huge_risk_frac": 0.46,
            "lead_stack": 13000,
            "lead_extra_margin": 0.035,
            "unknown_extra_margin": 0.012,
            "multiway_margin": 0.035,
            "street_margin": {"flop": 0.015, "turn": 0.032, "river": 0.055},
            "made_relief": {"0": 0.0, "1": 0.018, "2": 0.045, "3": 0.075, "4": 0.12, "5": 0.16, "6": 0.24, "7": 0.30, "8": 0.35},
            "draw_relief": {"0": 0.0, "1": 0.005, "2": 0.020, "3": 0.045, "4": 0.065, "5": 0.080},
        },
        "bucket_rules": [],
    }


_RISK_PROFILE = _load_risk_profile()


def decide(state):
    """Return one legal action within the Fullhouse 2s decision budget."""
    try:
        _observe(state)
        if state.get("street") == "preflop":
            return _decide_preflop(state)
        return _decide_postflop(state)
    except Exception:
        # Surface to stderr for local debugging but keep production safe: a
        # check/fold default is better than a sandbox crash. Stderr is not
        # captured by the engine so it won't pollute opponent state.
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}


def _observe(state):
    hand_id = state.get("hand_id", "")
    log = state.get("action_log") or []
    start = _SEEN_ACTIONS.get(hand_id, 0)
    if start > len(log):
        start = 0

    mine = state.get("seat_to_act")
    seat_to_id = {p.get("seat"): p.get("bot_id") for p in state.get("players") or []}

    for entry in log[start:]:
        action = entry.get("action")
        if action in ("small_blind", "big_blind"):
            continue
        seat = entry.get("seat")
        if seat == mine:
            continue
        bot_id = seat_to_id.get(seat)
        if not bot_id:
            continue

        row = _OPPONENTS.setdefault(
            bot_id,
            {
                "actions": 0,
                "raises": 0,
                "calls": 0,
                "folds": 0,
                "checks": 0,
                "allins": 0,
            },
        )
        row["actions"] += 1
        if action == "raise":
            row["raises"] += 1
        elif action == "all_in":
            row["raises"] += 1
            row["allins"] += 1
        elif action == "call":
            row["calls"] += 1
        elif action == "fold":
            row["folds"] += 1
        elif action == "check":
            row["checks"] += 1

    _SEEN_ACTIONS[hand_id] = len(log)
    if len(_SEEN_ACTIONS) > 160:
        _SEEN_ACTIONS.clear()


def _decide_preflop(state):
    cards = state.get("your_cards") or []
    if len(cards) != 2:
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    key = _hand_key(cards)
    blueprint = _blueprint_entry(key)
    score = float(blueprint.get("score", _preflop_score(cards)))
    pos = _position_name(state)
    owed = int(state.get("amount_owed", 0))
    pot = max(1, int(state.get("pot", 0)))
    stack = int(state.get("your_stack", 0))
    my_bet = int(state.get("your_bet_this_street", 0))
    current = int(state.get("current_bet", 0))
    all_in_to = stack + my_bet
    raises = _raise_count(state)
    limpers = _limper_count(state)
    opponents = max(1, _opponents_in_hand(state))
    price = owed / max(1, pot + owed)
    table = _table_profile()
    pressure = table["raise_rate"]
    # Context-aware profiles (per Gemini review claim 5). Open decisions look
    # at who might 3-bet us; facing-raise decisions look at the raiser.
    # Falls back to table profile when sample size is small.
    behind = _remaining_villains_profile(state)
    raisers = _raisers_profile(state)

    if stack <= 0:
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    stack_bb = all_in_to / BIG_BLIND
    if stack_bb <= 7:
        if _blueprint_jam_ok(blueprint, stack_bb) or score >= 54 or key in {"22", "33", "44", "A2s", "A7o", "K9s", "KTo", "QJs"}:
            return {"action": "all_in"}
        if state.get("can_check"):
            return {"action": "check"}
        return {"action": "call"} if price <= 0.18 and score >= 43 else {"action": "fold"}

    if key in PREMIUM:
        if raises >= 2 or stack <= pot * 2.1:
            return {"action": "all_in"}
        target = max(current * 3 + pot // 2, BIG_BLIND * (4 + limpers))
        return _raise_to(state, target)

    if raises == 0:
        if pos == "bb" and state.get("can_check"):
            if score >= 78 or (score >= 62 and _roll(state, "bb_iso") < 0.32):
                return _raise_to(state, BIG_BLIND * (4 + limpers))
            return {"action": "check"}

        threshold = _open_threshold(pos, opponents)
        # Use players-behind-us profile when available; fall back to table avg.
        # These are the bots who might 3-bet our open — their tendency matters
        # more than the UTG maniac who already folded.
        ctx_open = behind if behind["actions"] >= 16 else table
        if ctx_open["actions"] >= 16 and ctx_open["fold_rate"] > 0.42:
            threshold -= 3
        if ctx_open["actions"] >= 16 and ctx_open["raise_rate"] > 0.52:
            threshold += 2

        open_freq = _blueprint_open_freq(blueprint, pos)
        if score >= threshold or _roll(state, "blueprint_open") < open_freq:
            size = 2.25 + min(limpers, 4) * 0.72
            if pos in ("early", "middle"):
                size += 0.35
            if pos in ("sb", "bb"):
                size += 0.25
            if score >= 85:
                size += 0.45
            if stack_bb <= 22:
                size -= 0.20
            return _raise_to(state, int(BIG_BLIND * max(2.0, size)))

        if owed > 0 and _blueprint_defend_ok(blueprint, pos, price, raises, opponents):
            return {"action": "call"}
        if owed > 0 and _preflop_call_ok(key, score, price, stack, current, opponents, pos):
            return {"action": "call"}
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    threebet_bar = 84 + max(0, raises - 1) * 5
    continue_bar = 63 + max(0, raises - 1) * 6 + max(0, opponents - 2) * 3
    if pos in ("button", "bb"):
        continue_bar -= 3
    if price <= 0.15:
        continue_bar -= 7
    elif price >= 0.30:
        continue_bar += 5
    # Facing a raise: use the raiser's specific profile (not table avg) when
    # we have enough observations. A nit's raise means strength; a maniac's
    # raise means little. Falls back to table profile when sample is small.
    ctx_face = raisers if raisers["actions"] >= 16 else table
    ctx_pressure = ctx_face["raise_rate"]
    if ctx_pressure > 0.54:
        threebet_bar -= 3
        continue_bar -= 2
    elif ctx_face["actions"] >= 16 and ctx_pressure < 0.24:
        threebet_bar += 4
        continue_bar += 3

    value_3bet = key in VALUE_3BET or blueprint.get("threebet") == "value"
    if value_3bet or score >= threebet_bar:
        if key in PREMIUM and (raises >= 2 or stack <= pot * 2.4):
            return {"action": "all_in"}
        if raises >= 2:
            if score >= 93:
                return {"action": "all_in"}
            if score >= 87 and price <= 0.28:
                return {"action": "call"}
            return {"action": "fold"}
        if stack <= pot * 1.9 and score >= 86:
            return {"action": "all_in"}
        if score >= 95:
            return {"action": "all_in"}
        return _raise_to(state, current * 3 + pot // 2)

    if (
        raises == 1
        and pos in ("button", "cutoff", "sb", "bb")
        and (key in BLOCKER_BLUFFS or blueprint.get("threebet") == "bluff")
        and stack_bb >= 24
        and opponents <= 2
        and _roll(state, "3bet_bluff") < (0.19 if ctx_pressure < 0.44 else 0.12)
    ):
        return _raise_to(state, current * 3 + pot // 3)

    if score >= continue_bar and price <= 0.34:
        return {"action": "call"}
    if _blueprint_defend_ok(blueprint, pos, price, raises, opponents):
        return {"action": "call"}
    if _preflop_call_ok(key, score, price, stack, current, opponents, pos):
        return {"action": "call"}
    return {"action": "check"} if state.get("can_check") else {"action": "fold"}


def _decide_postflop(state):
    owed = int(state.get("amount_owed", 0))
    pot = max(1, int(state.get("pot", 0)))
    stack = int(state.get("your_stack", 0))
    opponents = max(1, _opponents_in_hand(state))
    street = state.get("street")
    equity = _estimate_equity(state, opponents)
    made = _made_rank(state)
    draw = _draw_score(state)
    wet = _board_wetness(state.get("community_cards") or [])
    price = owed / max(1, pot + owed)
    spr = stack / max(1, pot)
    villain = _last_aggressor_profile(state)

    if stack <= 0:
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    risk_mode = "open"
    if not state.get("can_check"):
        risk_mode = _risk_gate_mode(state, made, draw, equity, opponents, stack, owed, price, spr, villain)
        if risk_mode == "fold":
            return {"action": "fold"}

    multiway_tax = 0.050 * max(0, opponents - 1)
    needed = price + 0.052 + multiway_tax
    if street == "river":
        needed += 0.035
    if villain["actions"] >= 8 and villain["raise_rate"] > 0.55:
        needed -= 0.030
    elif villain["actions"] >= 8 and villain["raise_rate"] < 0.22:
        needed += 0.060
    elif villain["actions"] < 8:
        # Was `< 6` — left a partition gap for villains with 6-7 actions
        # where neither the high-confidence branches (>=8) nor the
        # unknown-villain tax fired. Per code review, closes the gap.
        needed += 0.015

    if state.get("can_check"):
        if made >= 6 or equity >= 0.82:
            return _value_bet(state, pot, stack, equity, made, wet, opponents, spr)
        if made >= 4 and equity >= 0.67:
            return _value_bet(state, pot, stack, equity, made, wet, opponents, spr)
        if made == 3 and equity >= 0.60 and opponents <= 2:
            return _value_bet(state, pot, stack, equity, made, wet, opponents, spr)
        if made == 2 and equity >= 0.56 and opponents <= 2 and _roll(state, "thin_two_pair") < 0.62:
            return _value_bet(state, pot, stack, equity, made, wet, opponents, spr)
        if made == 1 and equity >= 0.63 and opponents == 1 and _roll(state, "thin_pair") < 0.34:
            return _small_bet(state, pot, stack)

        solver_action = _cfr_postflop_action(
            state, made, draw, equity, opponents, pot, stack, owed, price, spr, villain
        )
        if solver_action:
            return solver_action

        fold_equity = _fold_equity_hint(villain, opponents)
        if street != "river" and draw >= 3 and equity >= 0.34 and opponents <= 2:
            if _roll(state, "semi_bluff") < 0.24 + fold_equity:
                return _semi_bluff(state, pot, stack, wet, spr)
        if street != "river" and opponents == 1 and wet <= 1 and 0.31 <= equity <= 0.50:
            if _roll(state, "range_cbet") < 0.10 + fold_equity * 0.6:
                return _small_bet(state, pot, stack)
        # Safe river bluff (per Gemini): only HU, only with weak hands and low
        # equity, only against opponents with a strong observed fold tendency,
        # and only 35% of the time even when those gates are met. Closes the
        # documented "zero river bluff" leak without spewing into stations.
        if (
            street == "river"
            and opponents == 1
            and made <= 1
            and equity < 0.25
            and villain.get("actions", 0) > 10
            and villain.get("fold_rate", 0) > 0.60
            and _roll(state, "safe_river_bluff") < 0.35
        ):
            return _small_bet(state, pot, stack)
        return {"action": "check"}

    if street == "river":
        # Harmonic-translated bet fraction stops opponents exploiting the
        # 0.30-pot / 0.13-price boundaries with just-over sizings.
        harmonic_frac = _harmonic_bet_fraction(state, owed, pot, "harmonic_river_fold")
        harmonic_price_r = harmonic_frac / (1.0 + harmonic_frac) if harmonic_frac > 0 else 0.0
        if made <= 1 and harmonic_frac > 0.30 and villain["raise_rate"] < 0.45:
            return {"action": "fold"}
        if made == 1 and harmonic_price_r <= 0.13 and villain["raise_rate"] > 0.52:
            return {"action": "call"}

    if made >= 6:
        if stack <= pot * 1.25:
            return {"action": "call"} if risk_mode == "call_only" else {"action": "all_in"}
        if opponents <= 2 and _roll(state, "nut_raise") < 0.72:
            action = _raise_to(state, int(state.get("current_bet", 0)) + int(pot * 0.85))
            return _risk_limited_action(action, risk_mode)
        return {"action": "call"}

    if (made >= 4 and equity >= 0.76) or (made >= 3 and equity >= 0.82 and opponents <= 2):
        if stack <= pot * 1.05:
            return {"action": "call"} if risk_mode == "call_only" else {"action": "all_in"}
        if _roll(state, "value_raise") < (0.48 if opponents <= 2 else 0.24):
            action = _raise_to(state, int(state.get("current_bet", 0)) + int(pot * 0.68))
            return _risk_limited_action(action, risk_mode)
        return {"action": "call"}

    solver_action = _cfr_postflop_action(
        state, made, draw, equity, opponents, pot, stack, owed, price, spr, villain
    )
    if solver_action:
        return _risk_limited_action(solver_action, risk_mode)

    draw_bonus = 0.0 if street == "river" else min(0.14, draw * 0.035)
    if spr >= 2.2 and street != "river" and draw >= 3:
        draw_bonus += 0.025

    if equity + draw_bonus >= needed:
        if street != "river" and draw >= 3 and opponents <= 2 and stack > pot:
            if _roll(state, "raise_draw") < 0.16 + _fold_equity_hint(villain, opponents):
                action = _raise_to(state, int(state.get("current_bet", 0)) + int(pot * 0.58))
                return _risk_limited_action(action, risk_mode)
        return {"action": "call"}

    # Pseudo-harmonic translation of the opponent's bet for bucketed call
    # thresholds. Mitigates "just-over-threshold" exploit of these branches.
    harmonic_price = _harmonic_price(state, owed, pot, "harmonic_postflop_call")
    if street != "river" and draw >= 3 and harmonic_price <= 0.19 and owed <= stack * 0.18:
        return {"action": "call"}
    if made == 2 and harmonic_price <= 0.18 and opponents <= 2:
        return {"action": "call"}
    return {"action": "fold"}


def _estimate_equity(state, opponents):
    hero = [_card(c) for c in state.get("your_cards") or []]
    board = [_card(c) for c in state.get("community_cards") or []]
    if len(hero) != 2:
        return 0.50

    known = set((state.get("your_cards") or []) + (state.get("community_cards") or []))
    deck = [card for card in DECK if str(card) not in known]
    missing_board = 5 - len(board)
    seats = _active_opponent_seats(state)
    if not seats:
        seats = list(range(opponents))
    opponents = max(1, len(seats))

    street = state.get("street")
    if street == "flop":
        iterations = 210
    elif street == "turn":
        iterations = 240
    else:
        iterations = 280
    if opponents >= 3:
        iterations = max(95, iterations - 85)
    elif opponents == 1:
        iterations += 60

    floors = [_seat_range_floor(state, seat) for seat in seats[:opponents]]
    rng = random.Random(_stable_seed(state, "equity"))
    wins = 0.0

    for _ in range(iterations):
        used = set()
        opp_hands = []
        for floor in floors:
            hand = _sample_range_hand(deck, used, floor, rng)
            opp_hands.append(hand)
            used.add(str(hand[0]))
            used.add(str(hand[1]))

        remaining = [card for card in deck if str(card) not in used]
        if missing_board > 0:
            runout = rng.sample(remaining, missing_board)
        else:
            runout = []
        full_board = board + runout
        hero_score = eval7.evaluate(hero + full_board)

        tied = 1
        ahead = True
        for hand in opp_hands:
            opp_score = eval7.evaluate(list(hand) + full_board)
            if opp_score > hero_score:
                ahead = False
                break
            if opp_score == hero_score:
                tied += 1
        if ahead:
            wins += 1.0 / tied

    return wins / max(1, iterations)


def _sample_range_hand(deck, used, floor, rng):
    available = [card for card in deck if str(card) not in used]
    best = None
    best_score = -1
    for _ in range(24):
        c1, c2 = rng.sample(available, 2)
        score = _preflop_score_cards(c1, c2)
        if score > best_score:
            best_score = score
            best = (c1, c2)
        loosen = rng.random() * 14.0
        if score + loosen >= floor:
            return c1, c2
    return best


def _seat_range_floor(state, seat):
    floor = 19
    actions = [
        entry.get("action")
        for entry in state.get("action_log") or []
        if entry.get("seat") == seat and entry.get("action") not in ("small_blind", "big_blind")
    ]
    raises = sum(1 for action in actions if action in ("raise", "all_in"))
    calls = sum(1 for action in actions if action == "call")
    if raises:
        floor += 29 + max(0, raises - 1) * 8
    elif calls:
        floor += 10

    player = _player_by_seat(state, seat)
    if player:
        if int(player.get("bet_this_street") or 0) >= int(state.get("current_bet") or 0) and state.get("current_bet"):
            floor += 7
        if player.get("is_all_in"):
            floor += 12

        bot_id = player.get("bot_id")
        stats = _OPPONENTS.get(bot_id, {})
        actions_seen = stats.get("actions", 0)
        if actions_seen >= 10:
            raise_rate = stats.get("raises", 0) / max(1, actions_seen)
            call_rate = stats.get("calls", 0) / max(1, actions_seen)
            if raises and raise_rate < 0.23:
                floor += 6
            if raise_rate > 0.56:
                floor -= 5
            if calls and call_rate > 0.48:
                floor -= 4

    if state.get("street") == "river":
        floor += 3
    return max(0, min(84, floor))


def _preflop_call_ok(key, score, price, stack, current, opponents, pos):
    if price > 0.29:
        return False
    if key in BLIND_DEFENDS and pos in ("bb", "sb") and price <= 0.25:
        return True
    if score >= 58 and price <= 0.23:
        return True
    if key[0] == key[1] and price <= 0.20 and stack >= max(BIG_BLIND * 16, current * 9):
        return True
    if opponents >= 2 and _is_suited_connector_key(key) and price <= 0.22:
        return True
    return False


def _raise_to(state, target):
    stack = int(state.get("your_stack", 0))
    my_bet = int(state.get("your_bet_this_street", 0))
    all_in_to = stack + my_bet
    min_to = int(state.get("min_raise_to", 0))
    target = int(max(min_to, target))
    if all_in_to <= min_to or target >= all_in_to:
        return {"action": "all_in"}
    if target >= int(all_in_to * 0.92) and all_in_to <= int(max(1, state.get("pot", 0)) * 1.25):
        return {"action": "all_in"}
    return {"action": "raise", "amount": max(min_to, min(target, all_in_to - 1))}


def _value_bet(state, pot, stack, equity, made, wet, opponents, spr):
    frac = 0.52
    if made >= 6 or equity >= 0.83:
        frac = 0.88
    elif made >= 4 or equity >= 0.72:
        frac = 0.68
    if wet >= 3 and made >= 2:
        frac += 0.10
    if opponents >= 3:
        frac += 0.08
    if spr <= 1.05 and (made >= 4 or equity >= 0.78):
        return {"action": "all_in"}
    return _raise_to(state, int(pot * min(1.08, frac)))


def _small_bet(state, pot, stack):
    if stack <= pot * 0.45:
        return {"action": "all_in"}
    return _raise_to(state, int(pot * 0.34))


def _semi_bluff(state, pot, stack, wet, spr):
    if spr <= 0.90:
        return {"action": "all_in"}
    frac = 0.46 + 0.07 * min(3, wet)
    return _raise_to(state, int(pot * frac))


def _risk_gate_mode(state, made, draw, equity, opponents, stack, owed, price, spr, villain):
    if owed <= 0:
        return "open"

    defaults = _RISK_PROFILE.get("defaults") or {}
    risk_frac = owed / max(1, stack + owed)
    large_risk = float(defaults.get("large_risk_frac", 0.22))
    huge_risk = float(defaults.get("huge_risk_frac", 0.46))
    if risk_frac < large_risk and price <= 0.24:
        return "open"

    min_equity = _risk_min_equity(state, made, draw, equity, opponents, stack, risk_frac, villain)
    if equity < min_equity:
        return "fold"

    # Good enough to continue, but avoid turning marginal bluff-catchers and
    # draws into stack-off raises. Strong made hands remain free to value-raise.
    if risk_frac >= huge_risk and made < 4 and equity < min_equity + 0.075:
        return "call_only"
    if state.get("street") == "river" and made <= 2 and equity < min_equity + 0.045:
        return "call_only"
    if opponents >= 3 and made < 4 and equity < min_equity + 0.055:
        return "call_only"
    if stack >= int(defaults.get("lead_stack", 13000)) and made < 4 and equity < min_equity + 0.050:
        return "call_only"
    return "open"


def _risk_min_equity(state, made, draw, equity, opponents, stack, risk_frac, villain):
    defaults = _RISK_PROFILE.get("defaults") or {}
    rule = _risk_matching_rule(state, made, risk_frac, opponents)
    if rule:
        base = float(rule.get("min_equity", price_floor(state, risk_frac)))
    else:
        base = price_floor(state, risk_frac)

    street_margin = defaults.get("street_margin") or {}
    base += float(street_margin.get(state.get("street"), 0.030))
    base += max(0, opponents - 1) * float(defaults.get("multiway_margin", 0.035))

    if villain.get("actions", 0) < 8:
        base += float(defaults.get("unknown_extra_margin", 0.012))
    elif villain.get("raise_rate", 0.0) > 0.58:
        base -= 0.025
    elif villain.get("raise_rate", 0.0) < 0.22:
        base += 0.030

    if stack >= int(defaults.get("lead_stack", 13000)):
        base += float(defaults.get("lead_extra_margin", 0.035))

    made_relief = defaults.get("made_relief") or {}
    draw_relief = defaults.get("draw_relief") or {}
    base -= float(made_relief.get(str(min(8, made)), 0.0))
    if state.get("street") != "river":
        base -= float(draw_relief.get(str(min(5, draw)), 0.0))

    # Never demand impossible equity for monsters, but keep air/dominated pairs
    # honest when the call risks a meaningful fraction of the remaining stack.
    if made >= 6:
        return min(0.20, max(0.04, base))
    if made >= 4:
        return min(0.34, max(0.08, base))
    return max(0.10, min(0.78, base))


def price_floor(state, risk_frac):
    owed = int(state.get("amount_owed", 0))
    pot = max(1, int(state.get("pot", 0)))
    price = owed / max(1, pot + owed)
    return max(price + risk_frac * 0.18, price + 0.020)


def _risk_matching_rule(state, made, risk_frac, opponents):
    street = state.get("street")
    for rule in _RISK_PROFILE.get("bucket_rules") or []:
        if not isinstance(rule, dict):
            continue
        rule_street = rule.get("street")
        if rule_street not in (None, "*", street):
            continue
        if made < int(rule.get("min_made", 0)):
            continue
        if made > int(rule.get("max_made", 8)):
            continue
        if risk_frac < float(rule.get("min_risk_frac", 0.0)):
            continue
        if opponents < int(rule.get("min_opponents", 1)):
            continue
        if opponents > int(rule.get("max_opponents", 9)):
            continue
        return rule
    return None


def _risk_limited_action(action, risk_mode):
    if risk_mode != "call_only" or not isinstance(action, dict):
        return action
    if action.get("action") in ("raise", "all_in"):
        return {"action": "call"}
    return action


def _made_rank(state):
    cards = [_card(c) for c in (state.get("your_cards") or []) + (state.get("community_cards") or [])]
    if len(cards) < 5:
        return 0
    name = str(eval7.handtype(eval7.evaluate(cards))).lower()
    if "straight flush" in name:
        return 8
    if "four" in name:
        return 7
    if "full house" in name:
        return 6
    if "flush" in name:
        return 5
    if "straight" in name:
        return 4
    if "three" in name:
        return 3
    if "two pair" in name:
        return 2
    if "pair" in name:
        return 1
    return 0


def _draw_score(state):
    if state.get("street") == "river":
        return 0

    hero = state.get("your_cards") or []
    cards = hero + (state.get("community_cards") or [])
    suits = {}
    ranks = set()
    for card in cards:
        suits[card[1]] = suits.get(card[1], 0) + 1
        ranks.add(_rank_value(card[0]))

    score = 0
    for suit, count in suits.items():
        hero_suit_cards = [c for c in hero if c[1] == suit]
        if count >= 4 and hero_suit_cards:
            # Nut-aware: only count it as a full flush draw if our highest
            # card in that suit is King+ (else we're drawing to lose vs a
            # bigger flush). Per Gemini review, addresses reverse-implied
            # odds on non-nut flush draws.
            max_rank = max(_rank_value(c[0]) for c in hero_suit_cards)
            score += 2 if max_rank >= 13 else 1
        elif count == 3 and hero_suit_cards and state.get("street") == "flop":
            score += 1

    expanded = sorted(ranks | ({1} if 14 in ranks else set()))
    straight_draw = 0
    for start in range(1, 11):
        have = sum(1 for rank in range(start, start + 5) if rank in expanded)
        if have >= 4:
            straight_draw = 2
            break
        if have == 3:
            straight_draw = max(straight_draw, 1)
    score += straight_draw

    board_ranks = [_rank_value(card[0]) for card in state.get("community_cards") or []]
    hero_ranks = [_rank_value(card[0]) for card in hero]
    if board_ranks and state.get("street") == "flop":
        overcards = sum(1 for rank in hero_ranks if rank > max(board_ranks))
        if overcards == 2:
            score += 1

    return min(5, score)


def _board_wetness(board):
    if len(board) < 3:
        return 0
    suits = {}
    ranks = []
    for card in board:
        suits[card[1]] = suits.get(card[1], 0) + 1
        ranks.append(_rank_value(card[0]))

    wet = 0
    most_suit = max(suits.values())
    if most_suit >= 3:
        wet += 2
    elif most_suit == 2:
        wet += 1

    unique = sorted(set(ranks) | ({1} if 14 in ranks else set()))
    for start in range(1, 11):
        have = sum(1 for rank in range(start, start + 5) if rank in unique)
        if have >= 3:
            wet += 1
        if have >= 4:
            wet += 1
            break

    if len(set(ranks)) < len(ranks):
        wet += 1
    high_cards = sum(1 for rank in ranks if rank >= 11)
    if high_cards >= 2:
        wet += 1
    return min(6, wet)


def _preflop_score(cards):
    r1, r2 = _rank_value(cards[0][0]), _rank_value(cards[1][0])
    hi, lo = max(r1, r2), min(r1, r2)
    suited = cards[0][1] == cards[1][1]
    gap = hi - lo
    if hi == lo:
        return min(100, 47 + hi * 3.6)

    score = hi * 4.0 + lo * 2.0
    if suited:
        score += 5.5
    if gap == 1:
        score += 4.0
    elif gap == 2:
        score += 2.0
    elif gap >= 5:
        score -= 4.0
    if hi == 14:
        score += 5.5
    if hi >= 11 and lo >= 10:
        score += 5.0
    if lo <= 5 and hi < 10:
        score -= 3.0
    return max(0, min(100, score))


def _preflop_score_cards(c1, c2):
    a, b = str(c1), str(c2)
    cache_key = tuple(sorted((a, b)))
    cached = _PREFLOP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    score = _preflop_score([a, b])
    _PREFLOP_CACHE[cache_key] = score
    return score


def _open_threshold(pos, opponents):
    thresholds = {
        "early": 72,
        "middle": 66,
        "cutoff": 58,
        "button": 49,
        "sb": 53,
        "bb": 52,
    }
    base = thresholds.get(pos, 65)
    if opponents <= 2 and pos in ("button", "sb"):
        base -= 5
    if opponents >= 4 and pos in ("early", "middle"):
        base += 3
    return base


def _position_name(state):
    players = state.get("players") or []
    n = max(2, len(players))
    seat = int(state.get("seat_to_act", 0))
    sb, bb = _blind_seats(state)
    if n == 2:
        if seat == sb:
            return "sb" if state.get("street") == "preflop" else "button"
        return "bb"
    if bb is None:
        return "middle"
    dist = (seat - bb) % n
    if dist == 0:
        return "bb"
    if dist == n - 1:
        return "sb"
    if dist == n - 2:
        return "button"
    if dist == n - 3:
        return "cutoff"
    if dist <= 1:
        return "early"
    return "middle"


def _blind_seats(state):
    sb = None
    bb = None
    for entry in state.get("action_log") or []:
        if entry.get("action") == "small_blind":
            sb = entry.get("seat")
        elif entry.get("action") == "big_blind":
            bb = entry.get("seat")
        if sb is not None and bb is not None:
            return sb, bb
    return sb, bb


def _raise_count(state):
    return sum(1 for entry in state.get("action_log") or [] if entry.get("action") in ("raise", "all_in"))


def _limper_count(state):
    count = 0
    for entry in state.get("action_log") or []:
        action = entry.get("action")
        if action in ("raise", "all_in"):
            break
        if action == "call":
            count += 1
    return count


def _opponents_in_hand(state):
    mine = state.get("seat_to_act")
    return sum(
        1
        for player in state.get("players") or []
        if player.get("seat") != mine
        and not player.get("is_folded")
        and player.get("state") != "busted"
    )


def _active_opponent_seats(state):
    mine = state.get("seat_to_act")
    seats = []
    for player in state.get("players") or []:
        if player.get("seat") == mine:
            continue
        if not player.get("is_folded") and player.get("state") != "busted":
            seats.append(player.get("seat"))
    return seats


def _player_by_seat(state, seat):
    for player in state.get("players") or []:
        if player.get("seat") == seat:
            return player
    return None


def _table_profile():
    raises = 0
    calls = 0
    folds = 0
    passive = 0
    for row in _OPPONENTS.values():
        raises += row.get("raises", 0)
        calls += row.get("calls", 0)
        folds += row.get("folds", 0)
        passive += row.get("calls", 0) + row.get("checks", 0)
    actions = raises + passive + folds
    return {
        "actions": actions,
        "raise_rate": raises / max(1, raises + passive),
        "call_rate": calls / max(1, actions),
        "fold_rate": folds / max(1, actions),
    }


def _aggregate_profile(rows):
    """Aggregate _OPPONENTS rows into the same format as _table_profile.

    Mirrors _table_profile's denominator: raise_rate is aggression vs.
    non-fold actions (raises / (raises + passive)). Returns the same shape
    so callers can be drop-in replaced.
    """
    raises = calls = folds = passive = 0
    for row in rows:
        raises += row.get("raises", 0)
        calls += row.get("calls", 0)
        folds += row.get("folds", 0)
        passive += row.get("calls", 0) + row.get("checks", 0)
    actions = raises + passive + folds
    return {
        "actions": actions,
        "raise_rate": raises / max(1, raises + passive),
        "call_rate": calls / max(1, actions),
        "fold_rate": folds / max(1, actions),
    }


def _remaining_villains_profile(state):
    """Profile of opponents still in the hand (not folded, not us).

    Used in open-decision contexts where we care about who might call or
    3-bet our open raise — not the UTG maniac who already folded.
    """
    my_seat = state.get("seat_to_act")
    rows = []
    for p in state.get("players") or []:
        if p.get("seat") == my_seat:
            continue
        if p.get("is_folded"):
            continue
        bid = p.get("bot_id")
        if bid:
            row = _OPPONENTS.get(bid)
            if row:
                rows.append(row)
    return _aggregate_profile(rows)


def _raisers_profile(state):
    """Profile of opponents who've raised this hand.

    Used in facing-raise contexts where the aggressor's specific tendencies
    matter more than a table-wide average.
    """
    raiser_seats = set()
    my_seat = state.get("seat_to_act")
    for entry in state.get("action_log") or []:
        if entry.get("action") in ("raise", "all_in") and entry.get("seat") != my_seat:
            raiser_seats.add(entry.get("seat"))
    if not raiser_seats:
        return _aggregate_profile([])
    rows = []
    for p in state.get("players") or []:
        if p.get("seat") not in raiser_seats:
            continue
        bid = p.get("bot_id")
        if bid:
            row = _OPPONENTS.get(bid)
            if row:
                rows.append(row)
    return _aggregate_profile(rows)


def _last_aggressor_profile(state):
    seat_to_id = {p.get("seat"): p.get("bot_id") for p in state.get("players") or []}
    for entry in reversed(state.get("action_log") or []):
        if entry.get("action") in ("raise", "all_in"):
            row = _OPPONENTS.get(seat_to_id.get(entry.get("seat")), {})
            actions = row.get("actions", 0)
            return {
                "actions": actions,
                "raise_rate": row.get("raises", 0) / max(1, actions),
                "call_rate": row.get("calls", 0) / max(1, actions),
                "fold_rate": row.get("folds", 0) / max(1, actions),
            }
    return _table_profile()


def _fold_equity_hint(profile, opponents):
    if opponents > 2:
        return 0.0
    if profile["actions"] < 8:
        return 0.06
    return max(0.0, min(0.18, profile["fold_rate"] * 0.22 - profile["call_rate"] * 0.06))


def _cfr_postflop_action(state, made, draw, equity, opponents, pot, stack, owed, price, spr, villain):
    if opponents > 2 or state.get("street") not in ("flop", "turn", "river"):
        return None
    if state.get("can_check"):
        return _cfr_probe_action(state, made, draw, equity, opponents, pot, stack, spr)
    return _cfr_response_action(state, made, draw, equity, opponents, pot, stack, owed, price, villain)


def _cfr_probe_action(state, made, draw, equity, opponents, pot, stack, spr):
    if state.get("street") != "flop":
        return None

    board = state.get("community_cards") or []
    probe = _cfr_section(board, "probe")
    min_bucket_n = int(probe.get("min_bucket_n", 4))
    bucket = _solver_bucket(board, made, draw)
    if bucket and bucket.get("n", 0) >= min_bucket_n:
        base_freq = float(bucket.get("bet", 0.0)) + float(bucket.get("all_in", 0.0))
    else:
        base_freq = float(probe.get("fallback_frequency", 0.0))

    freq = base_freq * float(probe.get("solver_weight", 0.72))
    freq += float(probe.get("baseline", 0.03))
    if opponents == 1:
        freq += float(probe.get("heads_up_bonus", 0.03))
    else:
        freq -= float(probe.get("two_way_penalty", 0.08))
    if made >= 2:
        freq += float(probe.get("value_bonus", 0.13))
    elif made == 1 and equity >= 0.56:
        freq += float(probe.get("thin_pair_bonus", 0.04))
    if draw >= 3:
        freq += float(probe.get("draw_bonus", 0.09))
    elif draw == 0 and made == 0 and equity < float(probe.get("air_equity_floor", 0.36)):
        freq -= float(probe.get("air_penalty", 0.18))

    freq = _bounded(freq)
    if freq <= 0.0 or _roll(state, "open_cfr_probe") > freq:
        return None
    if made == 0 and draw == 0 and equity < float(probe.get("air_equity_floor", 0.36)):
        return None
    return _cfr_sized_bet(state, pot, stack, made, draw, equity, spr)


def _cfr_response_action(state, made, draw, equity, opponents, pot, stack, owed, price, villain):
    response = _cfr_section(state.get("community_cards") or [], "vs_bet")
    if not response:
        return None

    threshold = float(response.get("continue_equity", 0.18))
    threshold += price * float(response.get("price_weight", 0.92))
    threshold += max(0, opponents - 1) * float(response.get("multiway_tax", 0.048))
    if state.get("street") == "river":
        threshold += float(response.get("river_tax", 0.045))
    else:
        threshold -= min(5, draw) * float(response.get("draw_discount", 0.028))
    threshold -= min(5, made) * float(response.get("made_discount", 0.048))

    if villain["actions"] >= 8 and villain["raise_rate"] > 0.56:
        threshold -= 0.026
    elif villain["actions"] >= 8 and villain["raise_rate"] < 0.23:
        threshold += 0.030
    threshold = max(0.04, min(0.86, threshold))

    if equity >= threshold:
        if _cfr_raise_ok(state, response, made, draw, equity, price):
            return _cfr_response_raise(state, pot, stack, made, draw, equity)
        return {"action": "call"}

    # Pseudo-harmonic translation: when the opponent bet sits between our
    # abstract buckets, sample the bucket to use for the bucketed
    # call/fold thresholds. This stops "just-over-threshold" exploits.
    harmonic_price = _harmonic_price(state, owed, pot, "harmonic_cfr_vs_bet")
    if state.get("street") != "river" and draw >= 3 and harmonic_price <= float(response.get("draw_peel_price", 0.20)):
        return {"action": "call"}
    if made >= 1 and harmonic_price <= float(response.get("pair_peel_price", 0.15)):
        return {"action": "call"}
    if harmonic_price >= float(response.get("fold_price", 0.36)) and made <= 1 and draw <= 1:
        return {"action": "fold"}
    return None


def _cfr_raise_ok(state, response, made, draw, equity, price):
    if int(state.get("your_stack", 0)) <= 0:
        return False
    value_made = made >= int(response.get("raise_value_made", 5))
    value_equity = equity >= float(response.get("raise_value_equity", 0.78))
    draw_ok = state.get("street") != "river" and draw >= int(response.get("raise_draw_min", 4))
    if not (value_made or value_equity or draw_ok):
        return False
    if price > float(response.get("raise_max_price", 0.34)) and not (value_made or value_equity):
        return False
    freq = float(response.get("raise_freq", 0.12))
    if value_made or value_equity:
        freq += 0.22
    if draw_ok:
        freq += 0.08
    return _roll(state, "open_cfr_response_raise") < _bounded(freq)


def _cfr_response_raise(state, pot, stack, made, draw, equity):
    sizes = _cfr_section(state.get("community_cards") or [], "sizes")
    current = int(state.get("current_bet", 0))
    if stack <= pot * float(sizes.get("jam_spr", 0.82)) and (made >= 4 or equity >= 0.72):
        return {"action": "all_in"}
    if made >= 5 or equity >= 0.78:
        frac = float(sizes.get("polar", 0.72))
    elif draw >= 3:
        frac = float(sizes.get("medium", 0.54))
    else:
        frac = float(sizes.get("small", 0.34))
    return _raise_to(state, current + int(pot * frac))


def _cfr_sized_bet(state, pot, stack, made, draw, equity, spr):
    sizes = _cfr_section(state.get("community_cards") or [], "sizes")
    if spr <= float(sizes.get("jam_spr", 0.82)) and (made >= 3 or equity >= 0.70):
        return {"action": "all_in"}
    if made >= 4 or equity >= 0.74:
        frac = float(sizes.get("polar", 0.72))
    elif draw >= 3 or made >= 2:
        frac = float(sizes.get("medium", 0.54))
    else:
        frac = float(sizes.get("small", 0.34))
    return _raise_to(state, int(pot * frac))


def _cfr_section(board, name):
    defaults = _CFR_PROFILE.get("defaults") or {}
    base = defaults.get(name) if isinstance(defaults.get(name), dict) else {}
    textures = _CFR_PROFILE.get("textures") or {}
    row = textures.get(_board_texture_name(board))
    override = row.get(name) if isinstance(row, dict) and isinstance(row.get(name), dict) else {}
    merged = {}
    merged.update(base)
    merged.update(override)
    return merged


def _solver_postflop_action(state, made, draw, equity, opponents, pot, stack):
    if state.get("street") != "flop" or opponents > 2:
        return None

    bucket = _solver_bucket(state.get("community_cards") or [], made, draw)
    if not bucket or bucket.get("n", 0) < 4:
        return None

    bet_freq = float(bucket.get("bet", 0.0)) + float(bucket.get("all_in", 0.0))
    check_freq = float(bucket.get("check", 0.0))
    roll = _roll(state, "postflop_solver_digest")

    if bet_freq >= 0.74 and (made >= 1 or draw >= 1 or equity >= 0.36):
        return _solver_sized_bet(state, pot, stack, made, draw, equity)
    if bet_freq >= 0.58 and (made >= 2 or draw >= 2 or equity >= 0.45) and roll < (bet_freq - 0.45):
        return _solver_sized_bet(state, pot, stack, made, draw, equity)
    if check_freq >= 0.78 and made <= 1 and draw <= 1:
        return {"action": "check"}
    return None


def _solver_sized_bet(state, pot, stack, made, draw, equity):
    if stack <= pot * 0.55 and (made >= 3 or equity >= 0.70):
        return {"action": "all_in"}
    if made >= 4 or equity >= 0.72:
        return _raise_to(state, int(pot * 0.64))
    if draw >= 3:
        return _raise_to(state, int(pot * 0.54))
    return _small_bet(state, pot, stack)


def _solver_bucket(board, made, draw):
    texture = _board_texture_name(board)
    scenarios = _POSTFLOP_SOLVER.get("scenarios") or {}
    scenario = scenarios.get(texture)
    if not isinstance(scenario, dict):
        return None
    buckets = scenario.get("buckets")
    if not isinstance(buckets, dict):
        return None

    for d in range(min(5, draw), -1, -1):
        row = buckets.get(f"{made}:{d}")
        if row:
            return row
    for m in range(max(0, made - 1), -1, -1):
        row = buckets.get(f"{m}:{min(5, draw)}") or buckets.get(f"{m}:0")
        if row:
            return row
    return None


def _board_texture_name(board):
    flop = list(board[:3])
    if len(flop) < 3:
        return "rainbow_mid"

    suits = {}
    ranks = []
    for card in flop:
        suits[card[1]] = suits.get(card[1], 0) + 1
        ranks.append(_rank_value(card[0]))

    if len(set(ranks)) < len(ranks):
        return "paired_mid"
    if max(suits.values()) >= 3:
        return "monotone_high"

    expanded = set(ranks)
    if 14 in expanded:
        expanded.add(1)
    straightiness = 0
    for start in range(1, 11):
        straightiness = max(straightiness, sum(1 for rank in range(start, start + 5) if rank in expanded))
    high_cards = sum(1 for rank in ranks if rank >= 10)

    if straightiness >= 3 and high_cards >= 2:
        return "wet_broadway"
    if straightiness >= 3:
        return "low_connected"
    if max(ranks) >= 13:
        return "dry_high"
    return "rainbow_mid"


def _is_suited_connector_key(key):
    if len(key) != 3 or key[2] != "s":
        return False
    hi = _rank_value(key[0])
    lo = _rank_value(key[1])
    return hi - lo <= 2 and lo >= 5


def _blueprint_entry(key):
    hands = _BLUEPRINT.get("hands") or {}
    row = hands.get(key)
    return row if isinstance(row, dict) else {}


def _blueprint_open_freq(row, pos):
    table = row.get("open") if isinstance(row, dict) else None
    if not isinstance(table, dict):
        return 0.0
    try:
        return max(0.0, min(1.0, float(table.get(pos, 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _blueprint_defend_ok(row, pos, price, raises, opponents):
    if not isinstance(row, dict) or raises > 1:
        return False
    table = row.get("defend")
    if not isinstance(table, dict):
        return False
    try:
        freq = float(table.get(pos, 0.0))
        cap = float(row.get("max_call_price", 0.0))
    except (TypeError, ValueError):
        return False
    if opponents >= 3:
        cap -= 0.03
        freq -= 0.15
    return price <= cap and _bounded(freq) >= 0.50


def _blueprint_jam_ok(row, stack_bb):
    if not isinstance(row, dict):
        return False
    try:
        return stack_bb <= float(row.get("jam_bb", 0.0))
    except (TypeError, ValueError):
        return False


def _bounded(value):
    return max(0.0, min(1.0, value))


def _harmonic_bet_fraction(state, owed, pot, salt):
    """Apply pseudo-harmonic translation to an observed opponent bet.

    Maps the observed pot-fraction x = owed / pot to one of the abstract
    bucket sizes in ``_PSEUDO_HARMONIC_BUCKETS`` by sampling between the
    two adjacent buckets A < x <= B with probability
        f(x) = ((B - x) * (1 + A)) / ((B - A) * (1 + x))
    of choosing A (the smaller bucket). Returns the chosen bucket as a
    pot-fraction. Deterministic via ``_roll``.
    """
    pot_pos = max(1, int(pot))
    owed = max(0, int(owed))
    if owed <= 0:
        return 0.0
    x = owed / pot_pos
    buckets = _PSEUDO_HARMONIC_BUCKETS
    if x <= buckets[0]:
        return buckets[0]
    if x >= buckets[-1]:
        return buckets[-1]
    a = buckets[0]
    b = buckets[-1]
    for i in range(len(buckets) - 1):
        if buckets[i] <= x <= buckets[i + 1]:
            a = buckets[i]
            b = buckets[i + 1]
            break
    if b <= a:
        return a
    if x == a:
        return a
    if x == b:
        return b
    denom = (b - a) * (1.0 + x)
    if denom <= 0:
        return a
    f_a = ((b - x) * (1.0 + a)) / denom
    f_a = _bounded(f_a)
    if _roll(state, salt) < f_a:
        return a
    return b


def _harmonic_price(state, owed, pot, salt):
    """Pseudo-harmonic-translated price (= owed / (pot + owed)).

    Converts the observed bet to one of the abstract pot-fraction buckets
    via :func:`_harmonic_bet_fraction`, then re-derives the price metric
    the bot uses for threshold comparisons.
    """
    frac = _harmonic_bet_fraction(state, owed, pot, salt)
    if frac <= 0.0:
        return 0.0
    return frac / (1.0 + frac)


def _hand_key(cards):
    r1, r2 = cards[0][0], cards[1][0]
    v1, v2 = _rank_value(r1), _rank_value(r2)
    if v1 == v2:
        return r1 + r2
    hi, lo = (r1, r2) if v1 > v2 else (r2, r1)
    return hi + lo + ("s" if cards[0][1] == cards[1][1] else "o")


def _rank_value(rank):
    return RANK_VALUE.get(rank, 0)


def _card(text):
    return eval7.Card(text)


def _stable_seed(state, salt):
    text = "%s:%s:%s:%s" % (
        state.get("hand_id", ""),
        state.get("seat_to_act", 0),
        len(state.get("action_log") or []),
        salt,
    )
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _roll(state, salt):
    return random.Random(_stable_seed(state, salt)).random()
