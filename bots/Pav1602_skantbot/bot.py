"""
SkantBot v0.2 - Fullhouse Hackathon entry
============================================

Architecture:
  Layer 1: Hardcoded GTO preflop ranges from PokerCoaching's "Implementable GTO" charts.
           These act as our anchor point. Every range deviation goes through Config.
  Layer 2: Equity-based postflop decisions via eval7 Monte Carlo.
           Adaptive sample count: more sims on big decisions, fewer on small.
  Layer 3: Bayesian opponent modelling. Tracks VPIP/PFR/3bet%/fold-to-3bet/
           postflop aggression with population priors. Extreme reads trigger
           exploitative deviations (3-bet wider vs nits, value-bet vs stations).
  Layer 4: Heads-up branch when len(active_players)==2 - completely different
           ranges, since 6-max BTN of 43% becomes HU SB of 81%.
  Layer 5: Defensive try/except wrapping. Never crash, never time out.

Position derivation: dealer button is NOT in game_state. Recovered by parsing
action_log for the small_blind/big_blind entries, then computing button =
(BB_seat - 2) mod n.

Config dataclass exposes ~30 tunable parameters for Optuna. Defaults are
solver-derived starting points.
"""

import random
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

try:
    import eval7
    HAVE_EVAL7 = True
except ImportError:
    HAVE_EVAL7 = False

BOT_NAME = "SkantBot"
BOT_AVATAR = "robot_1"

# =============================================================================
# CONSTANTS
# =============================================================================

BIG_BLIND = 100
SMALL_BLIND = 50
RANKS = "23456789TJQKA"
RANK_IDX = {r: i for i, r in enumerate(RANKS)}

# =============================================================================
# CONFIG - every tunable parameter exposed for Optuna sweeps
# =============================================================================

@dataclass
class Config:
    # Preflop tightness offsets (1.0 = solver default, >1 = looser, <1 = tighter)
    rfi_tightness: float = 1.0
    threebet_tightness: float = 1.0
    fourbet_tightness: float = 1.0

    # Position-specific aggression multipliers
    pos_aggression_lj: float = 1.0
    pos_aggression_hj: float = 1.0
    pos_aggression_co: float = 1.0
    pos_aggression_btn: float = 1.0
    pos_aggression_sb: float = 1.0
    pos_aggression_bb: float = 1.0

    # Postflop equity thresholds
    equity_value_bet: float = 0.62      # bet for value above this
    equity_thin_value: float = 0.52     # thin value-bet IP only
    equity_call_threshold: float = 0.42 # call above this if pot odds work
    equity_raise_threshold: float = 0.72  # raise instead of call above this

    # C-bet (continuation bet as preflop raiser) frequency
    cbet_freq_dry: float = 0.75          # dry boards: bet 75% of the time
    cbet_freq_wet: float = 0.50          # wet boards
    cbet_size_pct: float = 0.50          # bet size as fraction of pot

    # Bluff frequency (semi-bluffs and pure bluffs)
    bluff_freq_ip: float = 0.20          # in-position bluff freq
    bluff_freq_oop: float = 0.10         # out-of-position bluff freq

    # Bet sizing presets (fractions of pot)
    sizing_value: float = 0.66
    sizing_polarised: float = 1.00
    sizing_thin: float = 0.40

    # 4-bet/5-bet sizing (multipliers of opponent's bet)
    threebet_size_ip: float = 3.5        # IP 3-bet = 3.5x the open
    threebet_size_oop: float = 4.0       # OOP 3-bet = 4x the open
    fourbet_size_ip: float = 2.3
    fourbet_size_oop: float = 2.5

    # Opponent modelling triggers
    min_hands_for_exploit: int = 25      # don't trust stats below this
    fold_to_3bet_exploit_threshold: float = 0.70
    vpip_station_threshold: float = 0.50
    vpip_nit_threshold: float = 0.15

    # Time/sim budget
    mc_sims_flop: int = 300
    mc_sims_turn: int = 400
    mc_sims_river: int = 600
    time_budget_sec: float = 1.6


CONFIG = Config()  # mutable global, can be replaced by tuned values

# =============================================================================
# PREFLOP RANGES (PokerCoaching Implementable GTO, 100bb cash, no rake)
# =============================================================================

# RFI: hands we open when folded to us. Position keys map to range strings.
RFI_RANGES_TXT = {
    "LJ":  "66+,A3s+,K8s+,Q9s+,J9s+,T9s,ATo+,KJo+,QJo",
    "HJ":  "55+,A2s+,K6s+,Q9s+,J9s+,T9s,98s,87s,76s,ATo+,KTo+,QTo+",
    "CO":  "33+,A2s+,K3s+,Q6s+,J8s+,T7s+,97s+,87s,76s,A8o+,KTo+,QTo+,JTo",
    "BTN": "22+,A2s+,K2s+,Q3s+,J4s+,T6s+,96s+,85s+,75s+,64s+,53s+,A4o+,K8o+,Q9o+,J9o+,T8o+,98o",
    # SB has a mixed limp/raise strategy. Combined VPIP ~62.3%.
    # We collapse to "raise this range or fold". Limping is the topic of v0.3.
    "SB":  "22+,A2s+,K2s+,Q4s+,J6s+,T7s+,96s+,85s+,75s+,64s+,54s,A2o+,K7o+,Q8o+,J8o+,T8o+,98o",
}

# 3-bet ranges by (my_position, raiser_position). Derived from the chart images.
# Format: "3bet_range_str" or ("3bet_range_str", "call_range_str")
THREEBET_RANGES = {
    # In-position (HJ, CO, BTN) facing earlier raiser
    ("HJ", "LJ"): "TT+,AKs,AQs,AJs,A5s,KQs,KJs,KTs,AKo,AQo",
    ("CO", "LJ"): "TT+,AKs,AQs,AJs,A5s,A4s,KQs,KJs,KTs,AKo,AQo",
    ("CO", "HJ"): "TT+,AKs,AQs,AJs,A5s,A4s,KQs,KJs,KTs,AKo,AQo,AJo",

    # BTN has both 3-bet and call ranges (mixed strategy)
    ("BTN", "LJ"): ("JJ+,AKs,AQs,AJs,A5s,A4s,KQs,AKo,AQo",
                    "TT,99,88,77,76s,65s,54s,JTs,T9s,98s,QJs,KJs,KTs,ATs,QTs"),
    ("BTN", "HJ"): ("QQ+,AKs,AQs,AJs,A5s,A4s,KQs,KJs,AKo,AQo",
                    "JJ-77,T9s,98s,87s,76s,JTs,QJs,KJs,KTs,ATs"),
    ("BTN", "CO"): ("TT+,AKs,AQs,AJs,A5s,A4s,KQs,KJs,AKo,AQo,AJo",
                    "99-66,T9s,98s,87s,76s,JTs,QJs,KTs"),

    # Out-of-position (SB) facing raiser - pure 3-bet or fold
    ("SB", "LJ"): "TT+,AKs,AQs,AJs,A5s,KQs,AKo,AQo",
    ("SB", "HJ"): "TT+,AKs,AQs,AJs,A5s,A4s,KQs,AKo,AQo,AJo",
    ("SB", "CO"): "99+,AKs,AQs,AJs,ATs,A5s,A4s,KQs,KJs,AKo,AQo,AJo",
    ("SB", "BTN"): "88+,A9s+,A5s,A4s,KTs+,QTs+,JTs,AKo,AQo,AJo,KQo",

    # BB facing raiser - 3-bet AND wide calling range
    ("BB", "LJ"): ("QQ+,AKs,A5s,AKo,AQs",
                   "22-JJ,A2s-AJs,K9s-KQs,Q9s+,J9s+,T9s,98s,87s,76s,65s,ATo+,KTo+,QTo+,JTo"),
    ("BB", "HJ"): ("JJ+,AKs,AQs,A5s,A4s,KQs,AKo,AQo",
                   "22-TT,A2s-AJs,K8s-KJs,Q8s+,J8s+,T8s,97s+,86s+,75s+,65s,54s,A9o+,KTo+,QTo+,JTo"),
    ("BB", "CO"): ("TT+,AKs,AQs,AJs,A5s,A4s,KQs,AKo,AQo",
                   "22-99,A2s-ATs,K6s-KJs,Q6s+,J7s+,T7s+,96s+,85s+,75s+,65s,54s,43s,A7o+,K9o+,Q9o+,J9o+,T9o"),
    ("BB", "BTN"): ("99+,AKs,AQs,AJs,ATs,A5s,A4s,A3s,KQs,KJs,AKo,AQo,AJo,KQo",
                    "22-88,A2s-A9s,K2s-KTs,Q2s+,J5s+,T5s+,95s+,84s+,74s+,63s+,53s+,42s+,32s,A2o-ATo,K6o-KJo,Q8o+,J8o+,T8o,97o+,87o,76o,65o"),
}

# 4-bet ranges (facing a 3-bet, we're the original raiser)
# At 100bb, very value-heavy with minimal blocker bluffs (A5s)
# Source: extrapolated from HU 5.1% 4-bet (KK+/AK + small A5s) tightened for 6-max
FOURBET_RANGE_VALUE = "QQ+,AKs,AKo"  # near-pure value
FOURBET_RANGE_BLUFF = "A5s,A4s"      # blocker bluffs at low frequency

# 5-bet shove range (facing a 4-bet) - top of range only
FIVEBET_RANGE = "KK+,AKs"  # AA, KK, AKs only - everything else folds to 4-bet


# =============================================================================
# HEADS-UP RANGES (when only 2 players left, e.g. late bracket)
# Source: PokerCoaching HUNL 100bb chart
# =============================================================================

HU_BTN_OPEN_TXT = "22+,A2s+,K2s+,Q2s+,J2s+,T2s+,93s+,82s+,72s+,62s+,52s+,42s+,32s,A2o+,K2o+,Q4o+,J5o+,T6o+,97o+,87o,76o,65o,54o"
# 81% range - everything except deep junk

HU_BB_3BET_TXT = "TT+,A9s+,A5s,A4s,KQs,KJs,QJs,JTs,AJo+,KQo"
# ~20% range, top of range plus suited connectors

HU_BB_CALL_VS_BTN_TXT = "22-99,A2s-A8s,K2s-KJs,Q2s-QTs,J3s-JTs,T3s-T9s,93s-98s,83s-87s,73s-76s,63s-65s,53s-54s,A2o-ATo,K2o-KJo,Q4o-QTo,J5o-J9o,T6o-T9o,97o-98o,87o,76o"

HU_BTN_4BET_TXT = "KK+,AKs,AKo"  # 5.1% = KK+, AKs, AKo, plus AQs sometimes
HU_BB_5BET_TXT = "KK+,AKs"        # 3.7% range - effectively KK+ AKs only


# =============================================================================
# EXPAND RANGES INTO SETS (one-time at module load)
# =============================================================================

def expand_range(range_str: str) -> set:
    """Parse range string -> set of canonical hand keys ('AKs', 'TT', 'AKo')."""
    hands = set()
    if not range_str or not range_str.strip():
        return hands

    for raw in range_str.split(","):
        part = raw.strip()
        if not part:
            continue

        try:
            # Pocket pairs
            if len(part) >= 2 and part[0] == part[1]:
                r = part[0]
                if len(part) == 2:
                    hands.add(r + r)
                elif part[2:] == "+":
                    idx = RANK_IDX[r]
                    for i in range(idx, len(RANKS)):
                        hands.add(RANKS[i] + RANKS[i])
                elif "-" in part:
                    bits = part.split("-")
                    hi = bits[0][0]
                    lo = bits[1][0]
                    for i in range(RANK_IDX[lo], RANK_IDX[hi] + 1):
                        hands.add(RANKS[i] + RANKS[i])
                continue

            # Non-pairs
            if len(part) >= 3:
                r1, r2, suit = part[0], part[1], part[2]
                if suit not in ("s", "o"):
                    continue

                if len(part) == 3:
                    hands.add(r1 + r2 + suit)
                elif part[3:] == "+":
                    # "A5s+" → A5s through AKs
                    idx2 = RANK_IDX[r2]
                    idx1 = RANK_IDX[r1]
                    for i in range(idx2, idx1):
                        hands.add(r1 + RANKS[i] + suit)
                elif "-" in part[3:]:
                    # "A5s-A2s" → A5s, A4s, A3s, A2s
                    bits = part.split("-")
                    hi_card = bits[0][1]
                    lo_card = bits[1][1]
                    for i in range(RANK_IDX[lo_card], RANK_IDX[hi_card] + 1):
                        hands.add(r1 + RANKS[i] + suit)
        except Exception:
            continue
    return hands


# Pre-expand all ranges at module load
RFI_SETS = {pos: expand_range(r) for pos, r in RFI_RANGES_TXT.items()}

THREEBET_SETS = {}
THREEBET_CALL_SETS = {}
for key, rng in THREEBET_RANGES.items():
    if isinstance(rng, tuple):
        THREEBET_SETS[key] = expand_range(rng[0])
        THREEBET_CALL_SETS[key] = expand_range(rng[1])
    else:
        THREEBET_SETS[key] = expand_range(rng)
        THREEBET_CALL_SETS[key] = set()

FOURBET_VALUE_SET = expand_range(FOURBET_RANGE_VALUE)
FOURBET_BLUFF_SET = expand_range(FOURBET_RANGE_BLUFF)
FIVEBET_SET = expand_range(FIVEBET_RANGE)

HU_BTN_OPEN = expand_range(HU_BTN_OPEN_TXT)
HU_BB_3BET = expand_range(HU_BB_3BET_TXT)
HU_BB_CALL_VS_BTN = expand_range(HU_BB_CALL_VS_BTN_TXT)
HU_BTN_4BET = expand_range(HU_BTN_4BET_TXT)
HU_BB_5BET = expand_range(HU_BB_5BET_TXT)


# =============================================================================
# OPPONENT MODELLING - module-level state, persists across hands within a match
# =============================================================================

# Population priors (Bayesian baseline)
PRIOR_VPIP = 0.30        # average opponent plays 30% of hands
PRIOR_PFR = 0.20         # average opponent raises preflop 20%
PRIOR_3BET = 0.08        # 8% 3-bet frequency
PRIOR_FOLD_TO_3BET = 0.55
PRIOR_FOLD_TO_CBET = 0.50
PRIOR_AGGRESSION = 0.25  # postflop bet/raise frequency
PRIOR_WEIGHT = 15        # how many "hands" of prior weight (low = adapt fast)


def _new_opponent():
    return {
        "hands_observed": 0,
        "vpip_actions": 0, "vpip_chances": 0,
        "pfr_actions": 0, "pfr_chances": 0,
        "threebet_actions": 0, "threebet_chances": 0,
        "fold_to_3bet": 0, "faced_3bet": 0,
        "fold_to_cbet": 0, "faced_cbet": 0,
        "postflop_aggression": 0, "postflop_actions": 0,
        "raise_preflop_seen": False,
    }


OPPONENTS: Dict[str, dict] = defaultdict(_new_opponent)
PROCESSED_HANDS: set = set()  # hand_ids whose action_log has been processed


def opp_stat(bot_id: str, stat: str, prior: float) -> float:
    """Compute Bayesian-smoothed stat for an opponent. Returns prior if no data."""
    if bot_id not in OPPONENTS:
        return prior
    opp = OPPONENTS[bot_id]
    actions_key = stat + "_actions" if stat + "_actions" in opp else stat
    chances_key = stat + "_chances" if stat + "_chances" in opp else "hands_observed"
    actions = opp.get(actions_key, 0)
    chances = opp.get(chances_key, 0)
    return (actions + prior * PRIOR_WEIGHT) / max(chances + PRIOR_WEIGHT, 1)


def update_opponents_from_log(state: dict):
    """Parse action_log to update per-opponent counters. Idempotent per hand."""
    hand_id = state.get("hand_id", "")
    log = state.get("action_log", [])
    players_by_seat = {p["seat"]: p["bot_id"] for p in state["players"]}

    # Find blind seats so we can ignore them in VPIP calc
    sb_seat = bb_seat = None
    for entry in log:
        if entry.get("action") == "small_blind":
            sb_seat = entry["seat"]
        elif entry.get("action") == "big_blind":
            bb_seat = entry["seat"]
            break

    # Track first-action-per-player to determine VPIP/PFR contribution
    first_action_seen = set()
    raised_already = set()  # seats that have raised this hand
    for entry in log:
        seat = entry.get("seat")
        action = entry.get("action")
        if seat is None or action in ("small_blind", "big_blind"):
            continue
        bot_id = players_by_seat.get(seat)
        if not bot_id:
            continue

        opp = OPPONENTS[bot_id]

        # First voluntary action this hand: counts toward VPIP/PFR/3bet
        if seat not in first_action_seen:
            first_action_seen.add(seat)
            if action != "fold":
                opp["vpip_actions"] += 1
            opp["vpip_chances"] += 1

            if action in ("raise", "all_in"):
                if not raised_already:
                    # First aggressor: PFR
                    opp["pfr_actions"] += 1
                else:
                    # 3-bet (or worse)
                    opp["threebet_actions"] += 1
                opp["pfr_chances"] += 1
                opp["threebet_chances"] += 1
            else:
                opp["pfr_chances"] += 1
                if raised_already:
                    opp["threebet_chances"] += 1
                    if action == "fold":
                        opp["fold_to_3bet"] += 1
                    opp["faced_3bet"] += 1

        # Track raises for next iteration
        if action in ("raise", "all_in"):
            raised_already.add(seat)

    # Mark processed at end of hand only
    if state.get("type") == "hand_complete":
        PROCESSED_HANDS.add(hand_id)
        for bot_id in players_by_seat.values():
            OPPONENTS[bot_id]["hands_observed"] += 1


# =============================================================================
# POSITION DERIVATION (recover dealer from action_log)
# =============================================================================

def get_position_label(state: dict) -> str:
    """Compute position label: UTG/LJ/HJ/MP/CO/BTN/SB/BB."""
    n = len(state["players"])
    my_seat = state["seat_to_act"]
    log = state.get("action_log", [])

    # Find BB seat from log
    bb_seat = None
    sb_seat = None
    for entry in log:
        if entry.get("action") == "big_blind":
            bb_seat = entry["seat"]
        elif entry.get("action") == "small_blind":
            sb_seat = entry["seat"]

    if bb_seat is None:
        return "MP"  # safe fallback

    if n == 2:
        # Heads-up: SB seat = button = "BTN-SB", BB seat = "BB"
        return "BTN" if my_seat == sb_seat else "BB"

    btn_seat = (bb_seat - 2) % n
    offset = (my_seat - btn_seat) % n

    # Standard 6-max position labels by offset from button
    if n >= 6:
        labels_6max = {0: "BTN", 1: "SB", 2: "BB", 3: "LJ", 4: "HJ", 5: "CO"}
        return labels_6max.get(offset, "LJ")
    elif n == 5:
        return {0: "BTN", 1: "SB", 2: "BB", 3: "HJ", 4: "CO"}.get(offset, "HJ")
    elif n == 4:
        return {0: "BTN", 1: "SB", 2: "BB", 3: "CO"}.get(offset, "CO")
    elif n == 3:
        return {0: "BTN", 1: "SB", 2: "BB"}.get(offset, "BTN")
    return "MP"


# =============================================================================
# HAND CANONICALISATION
# =============================================================================

def hand_str(hole_cards: List[str]) -> str:
    """Convert ['As', 'Kh'] -> 'AKo'. Pairs -> 'AA'. High card first."""
    if len(hole_cards) != 2:
        return ""
    r1, s1 = hole_cards[0][0], hole_cards[0][1]
    r2, s2 = hole_cards[1][0], hole_cards[1][1]
    if r1 == r2:
        return r1 + r2
    if RANK_IDX[r1] < RANK_IDX[r2]:
        r1, r2, s1, s2 = r2, r1, s2, s1
    suit = "s" if s1 == s2 else "o"
    return r1 + r2 + suit


# =============================================================================
# EQUITY CALCULATION (eval7 Monte Carlo)
# =============================================================================

_EQUITY_CACHE = {}

def equity_vs_random(hole_cards: List[str], community_cards: List[str],
                     n_sims: int = 300, n_opp: int = 1) -> float:
    """Monte Carlo equity vs random opponent hands. Returns float [0, 1]."""
    if not HAVE_EVAL7:
        return _equity_heuristic(hole_cards)

    cache_key = (tuple(hole_cards), tuple(community_cards), n_opp)
    if cache_key in _EQUITY_CACHE:
        return _EQUITY_CACHE[cache_key]

    try:
        my_cards = [eval7.Card(c) for c in hole_cards]
        board = [eval7.Card(c) for c in community_cards]
        used = set(str(c) for c in my_cards + board)
        deck = [eval7.Card(r + s) for r in RANKS for s in "shdc" if (r + s) not in used]

        wins = ties = 0
        needed = 5 - len(board)
        for _ in range(n_sims):
            sample = random.sample(deck, 2 * n_opp + needed)
            opp_hands = [sample[i:i+2] for i in range(0, 2 * n_opp, 2)]
            full_board = board + sample[2 * n_opp:]
            my_score = eval7.evaluate(my_cards + full_board)
            opp_scores = [eval7.evaluate(h + full_board) for h in opp_hands]
            best_opp = max(opp_scores)
            if my_score > best_opp:
                wins += 1
            elif my_score == best_opp:
                ties += 1
        eq = (wins + ties / 2) / n_sims
    except Exception:
        eq = _equity_heuristic(hole_cards)

    _EQUITY_CACHE[cache_key] = eq
    return eq


def equity_vs_range(hole_cards: List[str], community_cards: List[str],
                    villain_range: set, n_sims: int = 300) -> float:
    """Monte Carlo equity vs a specific assumed range. More accurate than vs random."""
    if not HAVE_EVAL7 or not villain_range:
        return equity_vs_random(hole_cards, community_cards, n_sims)

    try:
        my_cards = [eval7.Card(c) for c in hole_cards]
        board = [eval7.Card(c) for c in community_cards]
        used_str = set(str(c) for c in my_cards + board)
        deck_strs = [r + s for r in RANKS for s in "shdc" if (r + s) not in used_str]

        # Build all combos of villain hands consistent with the range
        villain_combos = []
        for hand_class in villain_range:
            combos = _hand_class_to_combos(hand_class, used_str)
            villain_combos.extend(combos)

        if not villain_combos:
            return equity_vs_random(hole_cards, community_cards, n_sims)

        wins = ties = 0
        needed = 5 - len(board)
        for _ in range(n_sims):
            # Pick a random villain combo
            v_combo = random.choice(villain_combos)
            v_str = {v_combo[0], v_combo[1]}
            avail = [c for c in deck_strs if c not in v_str]
            if len(avail) < needed:
                continue
            extra = random.sample(avail, needed) if needed > 0 else []
            full_board = board + [eval7.Card(c) for c in extra]
            v_cards = [eval7.Card(c) for c in v_combo]
            my_score = eval7.evaluate(my_cards + full_board)
            v_score = eval7.evaluate(v_cards + full_board)
            if my_score > v_score:
                wins += 1
            elif my_score == v_score:
                ties += 1
        return (wins + ties / 2) / max(n_sims, 1)
    except Exception:
        return equity_vs_random(hole_cards, community_cards, n_sims)


def _hand_class_to_combos(hand_class: str, used: set) -> List[Tuple[str, str]]:
    """Generate all card-combo pairs for a hand class like 'AKs' or 'TT'."""
    combos = []
    if len(hand_class) == 2:  # pair
        r = hand_class[0]
        suits = "shdc"
        for i in range(4):
            for j in range(i + 1, 4):
                c1, c2 = r + suits[i], r + suits[j]
                if c1 not in used and c2 not in used:
                    combos.append((c1, c2))
    elif len(hand_class) == 3:
        r1, r2, suit = hand_class[0], hand_class[1], hand_class[2]
        if suit == "s":
            for s in "shdc":
                c1, c2 = r1 + s, r2 + s
                if c1 not in used and c2 not in used:
                    combos.append((c1, c2))
        else:  # offsuit
            suits = "shdc"
            for s1 in suits:
                for s2 in suits:
                    if s1 == s2:
                        continue
                    c1, c2 = r1 + s1, r2 + s2
                    if c1 not in used and c2 not in used:
                        combos.append((c1, c2))
    return combos


def _equity_heuristic(hole: List[str]) -> float:
    """Crude equity heuristic when eval7 unavailable."""
    r1 = RANK_IDX.get(hole[0][0], 0)
    r2 = RANK_IDX.get(hole[1][0], 0)
    pair = hole[0][0] == hole[1][0]
    suited = hole[0][1] == hole[1][1]
    high = max(r1, r2)
    if pair:
        return 0.50 + (high - 0) * 0.025
    base = 0.30 + (high - 0) * 0.012
    if suited:
        base += 0.04
    if abs(r1 - r2) <= 4:
        base += 0.02
    return min(base, 0.85)


# =============================================================================
# BOARD TEXTURE
# =============================================================================

def board_texture(board: List[str]) -> str:
    """Classify board: 'dry', 'wet', 'medium'."""
    if len(board) < 3:
        return "dry"
    ranks = [RANK_IDX[c[0]] for c in board]
    suits = [c[1] for c in board]

    # Flush draws / monotone
    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit = max(suit_counts.values())

    # Connectedness
    sorted_r = sorted(ranks)
    gap = max(sorted_r) - min(sorted_r)
    has_pair = len(set(ranks)) < len(ranks)

    if max_suit >= 3:
        return "wet"  # monotone
    if max_suit == 2 and gap <= 4:
        return "wet"  # FD + connected
    if gap <= 4 and not has_pair:
        return "wet"  # straight-y
    if has_pair:
        return "dry"  # paired
    if gap >= 6 and max_suit < 2:
        return "dry"
    return "medium"


# =============================================================================
# SAFE ACTION HELPERS
# =============================================================================

def safe_raise_amount(state: dict, target: int) -> int:
    """Clamp raise to legal bounds."""
    stack = state["your_stack"]
    bet_so_far = state["your_bet_this_street"]
    max_raise = stack + bet_so_far
    target = max(int(target), state["min_raise_to"])
    target = min(target, max_raise)
    return target


def count_aggressors(state: dict) -> int:
    """Count voluntary raisers/all-ins before us this hand (excluding us)."""
    log = state.get("action_log", [])
    me = state["seat_to_act"]
    return sum(1 for e in log
               if e.get("action") in ("raise", "all_in") and e.get("seat") != me)


def count_my_raises(state: dict) -> int:
    """Count how many times WE have raised in this hand."""
    log = state.get("action_log", [])
    me = state["seat_to_act"]
    return sum(1 for e in log
               if e.get("action") in ("raise", "all_in") and e.get("seat") == me)


def preflop_scenario(state: dict) -> str:
    """Classify the preflop situation we face.
    Returns: 'open' | 'face_open' | 'face_3bet_as_raiser' | 'face_3bet_cold'
           | 'face_4bet_as_raiser' | 'face_5bet_as_raiser'."""
    aggressors = count_aggressors(state)
    my_raises = count_my_raises(state)
    if aggressors == 0:
        return "open"
    if aggressors == 1 and my_raises == 0:
        return "face_open"
    if aggressors == 1 and my_raises == 1:
        return "face_3bet_as_raiser"
    if aggressors == 2 and my_raises == 0:
        return "face_3bet_cold"
    if aggressors == 2 and my_raises == 1:
        return "face_4bet_as_raiser"
    if my_raises >= 2:
        return "face_5bet_as_raiser"
    return "face_3bet_cold"


def find_aggressor_seat(state: dict) -> Optional[int]:
    """Return seat of last aggressor (raiser/all-in) before us, if any."""
    log = state.get("action_log", [])
    me = state["seat_to_act"]
    for e in reversed(log):
        if e.get("action") in ("raise", "all_in") and e.get("seat") != me:
            return e["seat"]
    return None


def get_opp_position(state: dict, opp_seat: int) -> str:
    """Compute opponent's position label from their seat."""
    n = len(state["players"])
    log = state.get("action_log", [])
    bb_seat = None
    sb_seat = None
    for entry in log:
        if entry.get("action") == "big_blind":
            bb_seat = entry["seat"]
        elif entry.get("action") == "small_blind":
            sb_seat = entry["seat"]
    if bb_seat is None:
        return "MP"
    if n == 2:
        return "BTN" if opp_seat == sb_seat else "BB"
    btn_seat = (bb_seat - 2) % n
    offset = (opp_seat - btn_seat) % n
    if n >= 6:
        labels = {0: "BTN", 1: "SB", 2: "BB", 3: "LJ", 4: "HJ", 5: "CO"}
        return labels.get(offset, "LJ")
    elif n == 5:
        return {0: "BTN", 1: "SB", 2: "BB", 3: "HJ", 4: "CO"}.get(offset, "HJ")
    elif n == 4:
        return {0: "BTN", 1: "SB", 2: "BB", 3: "CO"}.get(offset, "CO")
    elif n == 3:
        return {0: "BTN", 1: "SB", 2: "BB"}.get(offset, "BTN")
    return "MP"


def aggressor_likely_range(state: dict, agg_seat: int) -> set:
    """Estimate aggressor's likely range based on their position and action history."""
    agg_pos = get_opp_position(state, agg_seat)
    aggressors = count_aggressors(state)
    # Default: their RFI range
    if aggressors == 1 and agg_pos in RFI_SETS:
        return RFI_SETS[agg_pos]
    if aggressors == 2:
        # They 3-bet: tight value range
        return expand_range("QQ+,AKs,AKo,A5s")
    return RFI_SETS.get(agg_pos, RFI_SETS["LJ"])


# =============================================================================
# PREFLOP DECISION
# =============================================================================

def decide_preflop_6max(state: dict, position: str, hand: str, cfg: Config) -> dict:
    pot = state["pot"]
    owed = state["amount_owed"]
    can_check = state["can_check"]
    bb = BIG_BLIND
    log = state.get("action_log", [])

    # === HEADS-UP BRANCH ===
    n_active = sum(1 for p in state["players"] if not p.get("is_folded"))
    if n_active == 2:
        return decide_preflop_hu(state, position, hand, cfg)

    scenario = preflop_scenario(state)

    # === SCENARIO: Open or check (no aggressor) ===
    if scenario == "open":
        rfi_set = RFI_SETS.get(position, set())
        if hand in rfi_set:
            tightness = cfg.rfi_tightness * getattr(cfg, f"pos_aggression_{position.lower()}", 1.0)
            if random.random() < tightness:
                limpers = sum(1 for e in log if e.get("action") == "call")
                target = int(bb * (2.5 + limpers))
                return {"action": "raise", "amount": safe_raise_amount(state, target)}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # === SCENARIO: Facing a single open raise ===
    if scenario == "face_open":
        agg_seat = find_aggressor_seat(state)
        agg_pos = get_opp_position(state, agg_seat) if agg_seat is not None else "LJ"

        threebet_set = THREEBET_SETS.get((position, agg_pos), set())
        call_set = THREEBET_CALL_SETS.get((position, agg_pos), set())

        # Exploit: opponent folds to 3-bets too often → 3-bet wider with A-x-suited
        if agg_seat is not None:
            opp_id = next((p["bot_id"] for p in state["players"] if p["seat"] == agg_seat), None)
            if opp_id and OPPONENTS[opp_id].get("faced_3bet", 0) >= cfg.min_hands_for_exploit:
                fold_to_3bet = (OPPONENTS[opp_id]["fold_to_3bet"] /
                                max(OPPONENTS[opp_id]["faced_3bet"], 1))
                if fold_to_3bet >= cfg.fold_to_3bet_exploit_threshold:
                    if len(hand) == 3 and hand[0] == "A" and hand[2] == "s":
                        threebet_set = threebet_set | {hand}

        if hand in threebet_set:
            ip = position in ("CO", "BTN")
            current = state["current_bet"]
            mult = cfg.threebet_size_ip if ip else cfg.threebet_size_oop
            target = int(current * mult)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}

        if hand in call_set and owed <= state["your_stack"] * 0.15:
            return {"action": "call"}

        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # === SCENARIO: We opened, opp 3-bet us ===
    if scenario == "face_3bet_as_raiser":
        # 4-bet our value range
        if hand in FOURBET_VALUE_SET:
            ip = position in ("CO", "BTN")
            current = state["current_bet"]
            mult = cfg.fourbet_size_ip if ip else cfg.fourbet_size_oop
            target = int(current * mult)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}

        # 4-bet bluff with blockers (low frequency, IP only)
        if hand in FOURBET_BLUFF_SET and position in ("CO", "BTN"):
            if random.random() < 0.30:  # 30% bluff frequency to keep range balanced
                ip = True
                current = state["current_bet"]
                target = int(current * cfg.fourbet_size_ip)
                return {"action": "raise", "amount": safe_raise_amount(state, target)}

        # Value-call with strong-but-not-4-bet hands (keeps us honest)
        if hand in {"JJ", "TT", "AKo", "AQs"} and owed <= state["your_stack"] * 0.20:
            return {"action": "call"}

        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # === SCENARIO: We didn't open, two raisers in front (3-bet cold) ===
    if scenario == "face_3bet_cold":
        # Squeeze/cold-4-bet very rare; only top of range
        if hand in {"AA", "KK", "QQ", "AKs"}:
            return {"action": "all_in"}
        if hand in {"JJ", "AKo", "AQs"} and owed <= state["your_stack"] * 0.15:
            return {"action": "call"}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # === SCENARIO: We 4-bet, opp 5-bet (or we opened, opp 3-bet, someone 4-bet) ===
    if scenario in ("face_4bet_as_raiser", "face_5bet_as_raiser"):
        # 5-bet shove with KK+/AKs only; AKo and QQ fold (correct 100bb play)
        if hand in FIVEBET_SET or hand in {"AA", "KK"}:
            return {"action": "all_in"}
        # AA/KK already handled by FIVEBET_SET ("KK+,AKs"). Other premium folds.
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # Default fallback
    if can_check:
        return {"action": "check"}
    return {"action": "fold"}


def decide_preflop_hu(state: dict, position: str, hand: str, cfg: Config) -> dict:
    """Heads-up preflop - much wider ranges."""
    aggressors = count_aggressors(state)
    can_check = state["can_check"]
    log = state.get("action_log", [])
    bb = BIG_BLIND

    if aggressors == 0:
        # We're SB/BTN, open or check
        if position == "BTN":
            # Open 81% of hands
            if hand in HU_BTN_OPEN:
                target = int(bb * 2.5)
                return {"action": "raise", "amount": safe_raise_amount(state, target)}
            if can_check:
                return {"action": "check"}
            return {"action": "fold"}
        else:
            # We're BB and got a free check (SB limped)
            if can_check:
                return {"action": "check"}
            return {"action": "fold"}

    if aggressors == 1:
        # We're BB facing BTN open
        if hand in HU_BB_3BET:
            current = state["current_bet"]
            target = int(current * 4.0)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}
        if hand in HU_BB_CALL_VS_BTN:
            if state["amount_owed"] <= state["your_stack"] * 0.12:
                return {"action": "call"}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    if aggressors == 2:
        # We opened, BB 3-bet us (we're BTN)
        if hand in HU_BTN_4BET:
            current = state["current_bet"]
            target = int(current * 2.3)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}
        if hand in {"QQ", "JJ", "AKo", "AQs"}:
            return {"action": "call"}
        return {"action": "fold"}

    # aggressors >= 3 - 5-bet pot, jam top of range
    if hand in HU_BB_5BET or hand in {"AA", "KK"}:
        return {"action": "all_in"}
    return {"action": "fold"}


# =============================================================================
# POSTFLOP DECISION
# =============================================================================

def decide_postflop(state: dict, position: str, cfg: Config) -> dict:
    hole = state["your_cards"]
    board = state["community_cards"]
    pot = state["pot"]
    owed = state["amount_owed"]
    can_check = state["can_check"]
    stack = state["your_stack"]
    street = state["street"]
    log = state.get("action_log", [])

    # Were we the preflop aggressor?
    pf_log = [e for e in log if e.get("action") in ("raise", "all_in") and
              e.get("action") not in ("small_blind", "big_blind")]
    me = state["seat_to_act"]
    was_pf_aggressor = bool(pf_log) and pf_log[0].get("seat") == me

    # Active opponent count
    n_opp = sum(1 for p in state["players"]
                if not p.get("is_folded") and p.get("seat") != me and not p.get("is_all_in"))
    n_opp = max(1, n_opp)

    # Choose sim count by street
    if street == "flop":
        n_sims = cfg.mc_sims_flop
    elif street == "turn":
        n_sims = cfg.mc_sims_turn
    else:
        n_sims = cfg.mc_sims_river

    # Equity: vs villain range if we can identify a clear aggressor
    agg_seat = find_aggressor_seat(state)
    if agg_seat is not None and len(log) > 4:
        v_range = aggressor_likely_range(state, agg_seat)
        eq = equity_vs_range(hole, board, v_range, n_sims=n_sims)
    else:
        eq = equity_vs_random(hole, board, n_sims=n_sims, n_opp=min(n_opp, 3))

    texture = board_texture(board)
    in_position = position in ("CO", "BTN")

    # === Free check option ===
    if can_check:
        # Strong: bet for value
        if eq >= cfg.equity_value_bet:
            target = int(state["current_bet"] + pot * cfg.sizing_value)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}

        # PFR continuation bet
        if was_pf_aggressor and street == "flop":
            cbet_freq = cfg.cbet_freq_dry if texture == "dry" else cfg.cbet_freq_wet
            if random.random() < cbet_freq:
                target = int(state["current_bet"] + pot * cfg.cbet_size_pct)
                return {"action": "raise", "amount": safe_raise_amount(state, target)}

        # Thin value IP
        if eq >= cfg.equity_thin_value and in_position:
            target = int(state["current_bet"] + pot * cfg.sizing_thin)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}

        # Bluff frequency
        bluff_freq = cfg.bluff_freq_ip if in_position else cfg.bluff_freq_oop
        if eq < cfg.equity_thin_value and random.random() < bluff_freq:
            target = int(state["current_bet"] + pot * cfg.sizing_thin)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}

        return {"action": "check"}

    # === Facing a bet ===
    if owed <= 0:
        return {"action": "check"}

    pot_odds = owed / (pot + owed) if (pot + owed) > 0 else 1.0

    # Big hand: raise for value (IP or OOP)
    if eq >= cfg.equity_raise_threshold:
        target = int(state["current_bet"] + (pot + owed) * cfg.sizing_value)
        return {"action": "raise", "amount": safe_raise_amount(state, target)}

    # Strong enough for value-call
    if eq >= cfg.equity_value_bet:
        return {"action": "call"}

    # Decent equity vs pot odds
    if eq >= pot_odds + 0.05:
        return {"action": "call"}

    # Marginal: only call if very cheap
    if eq >= cfg.equity_call_threshold and owed <= pot * 0.25:
        return {"action": "call"}

    return {"action": "fold"}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def decide(game_state: dict) -> dict:
    """Called by engine for every action. Must return within 2 seconds."""
    global _EQUITY_CACHE
    _EQUITY_CACHE = {}
    t0 = time.time()

    try:
        # Update opponent stats (cheap)
        try:
            update_opponents_from_log(game_state)
        except Exception:
            pass

        position = get_position_label(game_state)
        hand = hand_str(game_state["your_cards"])
        street = game_state["street"]

        if street == "preflop":
            action = decide_preflop_6max(game_state, position, hand, CONFIG)
        else:
            action = decide_postflop(game_state, position, CONFIG)

        # Hard time-budget guard
        if time.time() - t0 > CONFIG.time_budget_sec:
            return {"action": "check"} if game_state.get("can_check") else {"action": "fold"}

        return action

    except Exception:
        # Total fallback
        if game_state.get("can_check"):
            return {"action": "check"}
        if game_state.get("amount_owed", 999) <= game_state.get("pot", 0) * 0.10:
            return {"action": "call"}
        return {"action": "fold"}
