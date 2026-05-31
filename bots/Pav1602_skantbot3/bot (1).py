"""
SkantBot v0.3 - Fullhouse Hackathon entry
==========================================

Refactor of v0.2.1 into a fully parametric architecture suitable for
Optuna sweeps via Guneet's harness.

Key architectural changes from v0.2.1:
  - Single Config dataclass; every threshold/frequency/trigger on it.
  - Environment-variable loading (SKANT_<FIELD>) for harness injection.
  - Mixed-strategy preflop ranges (dict[hand, freq]) instead of sets.
  - Stack-aware tightness as a first-class multiplier.
  - Multiway c-bet penalty.
  - Field-shrink range widening (single parameter, interpolated).
  - Cold-start caution (shifts equity thresholds when opponent unknown).
  - Deterministic per-hand RNG (initialized from hand_id) for CRN compat.
  - All literals from v0.2.1 stack-preservation guards now Config fields.

Code structure (strict order, per spec):
  1. Imports
  2. Engine constants (immutable, given by rules)
  3. @dataclass Config and load_config_from_env()
  4. Range data (preflop charts as freq dicts)
  5. Helper functions (pure utilities, no decision logic)
  6. Opponent modelling (stat tracking + queries)
  7. Position derivation
  8. Equity calculation
  9. Preflop decision
  10. Postflop decision
  11. decide() entry point
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================

import os
import random
import time
from dataclasses import dataclass, fields, asdict
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

try:
    import eval7
    HAVE_EVAL7 = True
except ImportError:
    HAVE_EVAL7 = False


BOT_NAME = "SkantBot"
BOT_AVATAR = "robot_1"


# ============================================================================
# 2. ENGINE CONSTANTS (rules-given, never tuned)
# ============================================================================

BIG_BLIND = 100
SMALL_BLIND = 50
STARTING_STACK = 10000
RANKS = "23456789TJQKA"
RANK_IDX = {r: i for i, r in enumerate(RANKS)}


# ============================================================================
# 3. CONFIG DATACLASS + ENV LOADER
# ============================================================================

@dataclass
class Config:
    # --- Preflop tightness offsets ---
    # Multiplied into raise frequencies. 1.0 = chart values, >1 = looser.
    rfi_tightness: float = 1.0
    threebet_tightness: float = 1.0
    fourbet_tightness: float = 1.0

    # --- Position-specific aggression multipliers ---
    pos_aggression_lj: float = 1.0
    pos_aggression_hj: float = 1.0
    pos_aggression_co: float = 1.0
    pos_aggression_btn: float = 1.0
    pos_aggression_sb: float = 1.0
    pos_aggression_bb: float = 1.0

    # --- Stack-aware tightness curve ---
    # Default OFF (1.0) - exposed for Optuna to tune.
    # Empirically, aggressive tightening at shallow depths hurt more than helped
    # against the reference field. Optuna can find the right curve per opponent pool.
    stack_full_threshold_bb: float = 80.0
    stack_short_threshold_bb: float = 30.0
    stack_short_tightness: float = 1.0

    # --- Field-shrink widening (4-handed and below) ---
    # When n_active < 6, we widen ranges proportionally toward HU.
    # widening_factor = 1.0 + shrink_widening_factor * (6 - n_active)
    shrink_widening_factor: float = 0.10

    # --- Cold-start caution ---
    # Adds to call thresholds when we don't have enough hands on opponent.
    # Default 0 (off) - exposed for Optuna to tune. Setting >0 makes us
    # tighter when calling against unknowns, but can leak EV by missing calls.
    cold_start_caution: float = 0.0
    cold_start_threshold_hands: int = 6

    # --- Postflop equity thresholds ---
    equity_value_bet: float = 0.62        # bet for value above this
    equity_thin_value: float = 0.52       # thin value-bet IP only
    equity_call_threshold: float = 0.42   # marginal call threshold
    equity_raise_threshold: float = 0.72  # raise instead of just call
    pot_odds_buffer_normal: float = 0.08  # extra equity required vs pot odds
    pot_odds_buffer_marginal: float = 0.20 # how much pot we'll call vs marginal eq

    # --- Stack preservation guard ---
    # When facing a bet, the % of stack at risk triggers different thresholds.
    stack_risk_high_threshold: float = 0.30        # 30%+ risk = high
    stack_risk_medium_threshold: float = 0.15      # 15-30% risk = medium
    stack_risk_high_eq_normal: float = 0.72        # equity needed if high risk, normal opp
    stack_risk_high_eq_maniac: float = 0.78        # equity needed if high risk, vs maniac
    stack_risk_med_eq_normal: float = 0.58
    stack_risk_med_eq_maniac: float = 0.65

    # --- Jam-or-fold logic ---
    fourbet_commit_threshold: float = 0.25         # if 4-bet would commit >25% of stack, jam-or-fold
    shallow_jam_threshold_bb: float = 40.0         # if stack <40bb facing 3-bet, jam-or-fold
    fourbet_call_threshold_pct: float = 0.15       # cap on calling 3-bets out-of-position
    threebet_call_threshold_pct: float = 0.15      # cap on calling raises with weaker hands

    # --- C-bet (continuation bet) ---
    cbet_freq_dry: float = 0.75
    cbet_freq_wet: float = 0.50
    cbet_size_pct: float = 0.50
    # Multiway penalty: cbet_freq *= cbet_multiway_penalty ^ (n_opp - 1).
    # Default 0.75 = mild penalty (cbet 56% of normal vs 2 opps, 42% vs 3 opps).
    # Optuna can tune lower if pool tends to be sticky multiway.
    cbet_multiway_penalty: float = 0.75

    # --- Bluff frequencies ---
    bluff_freq_ip: float = 0.20
    bluff_freq_oop: float = 0.10
    fourbet_bluff_freq: float = 0.30

    # --- Bet sizing presets (fractions of pot) ---
    sizing_value: float = 0.66
    sizing_polarised: float = 1.00
    sizing_thin: float = 0.40

    # --- Preflop sizing multipliers ---
    open_size_bb: float = 2.5
    threebet_size_ip: float = 3.5
    threebet_size_oop: float = 4.0
    fourbet_size_ip: float = 2.3
    fourbet_size_oop: float = 2.5

    # --- Opponent modelling ---
    prior_weight: float = 15.0                     # Bayesian prior strength
    min_hands_for_exploit: int = 25
    fold_to_3bet_exploit_threshold: float = 0.70
    maniac_min_sample: int = 6
    maniac_vpip_threshold: float = 0.50
    maniac_pfr_threshold: float = 0.40
    station_min_sample: int = 8
    station_vpip_threshold: float = 0.45
    station_pfr_threshold: float = 0.15

    # --- Time/sim budget ---
    mc_sims_flop: int = 300
    mc_sims_turn: int = 400
    mc_sims_river: int = 600
    time_budget_sec: float = 1.6


def load_config_from_env() -> Config:
    """Load Config, overriding any field for which SKANT_<FIELDNAME_UPPER> is set.
    This is how Guneet's harness injects parameter values into trial runs."""
    cfg = Config()
    for f in fields(cfg):
        env_key = "SKANT_" + f.name.upper()
        env_val = os.environ.get(env_key)
        if env_val is None:
            continue
        try:
            # Cast to the field's declared type
            if f.type == int or f.type is int:
                setattr(cfg, f.name, int(env_val))
            elif f.type == float or f.type is float:
                setattr(cfg, f.name, float(env_val))
            elif f.type == bool or f.type is bool:
                setattr(cfg, f.name, env_val.lower() in ("1", "true", "yes"))
            else:
                setattr(cfg, f.name, env_val)
        except (ValueError, TypeError):
            pass  # silently keep default on bad input
    return cfg


CONFIG = load_config_from_env()


# ============================================================================
# 4. RANGE DATA (preflop charts as freq dicts)
# ============================================================================

# Source: PokerCoaching "Implementable GTO" charts, 100bb cash, no rake.
# Format: dict[hand_class] -> raise_frequency in [0.0, 1.0]
# Defaults are 1.0 for hands in chart, 0.0 implied for hands not present.
# This structure supports mixed strategies (e.g. K7s with 0.6 raise freq).

def _expand_to_freq_dict(range_str: str, freq: float = 1.0) -> Dict[str, float]:
    """Parse a range string and return {hand: freq} for every hand in it."""
    result = {}
    if not range_str or not range_str.strip():
        return result

    for raw in range_str.split(","):
        part = raw.strip()
        if not part:
            continue
        try:
            # Pocket pairs
            if len(part) >= 2 and part[0] == part[1]:
                r = part[0]
                if len(part) == 2:
                    result[r + r] = freq
                elif part[2:] == "+":
                    idx = RANK_IDX[r]
                    for i in range(idx, len(RANKS)):
                        result[RANKS[i] + RANKS[i]] = freq
                elif "-" in part:
                    bits = part.split("-")
                    hi = bits[0][0]
                    lo = bits[1][0]
                    for i in range(RANK_IDX[lo], RANK_IDX[hi] + 1):
                        result[RANKS[i] + RANKS[i]] = freq
                continue
            # Non-pairs
            if len(part) >= 3:
                r1, r2, suit = part[0], part[1], part[2]
                if suit not in ("s", "o"):
                    continue
                if len(part) == 3:
                    result[r1 + r2 + suit] = freq
                elif part[3:] == "+":
                    idx2 = RANK_IDX[r2]
                    idx1 = RANK_IDX[r1]
                    for i in range(idx2, idx1):
                        result[r1 + RANKS[i] + suit] = freq
                elif "-" in part[3:]:
                    bits = part.split("-")
                    hi_card = bits[0][1]
                    lo_card = bits[1][1]
                    for i in range(RANK_IDX[lo_card], RANK_IDX[hi_card] + 1):
                        result[r1 + RANKS[i] + suit] = freq
        except Exception:
            continue
    return result


# === RFI (open) ranges by position ===
# Default frequencies are pure (1.0). Optuna can shift via tightness/aggression.
RFI_FREQS: Dict[str, Dict[str, float]] = {
    "LJ":  _expand_to_freq_dict("66+,A3s+,K8s+,Q9s+,J9s+,T9s,ATo+,KJo+,QJo"),
    "HJ":  _expand_to_freq_dict("55+,A2s+,K6s+,Q9s+,J9s+,T9s,98s,87s,76s,ATo+,KTo+,QTo+"),
    "CO":  _expand_to_freq_dict("33+,A2s+,K3s+,Q6s+,J8s+,T7s+,97s+,87s,76s,A8o+,KTo+,QTo+,JTo"),
    "BTN": _expand_to_freq_dict("22+,A2s+,K2s+,Q3s+,J4s+,T6s+,96s+,85s+,75s+,64s+,53s+,A4o+,K8o+,Q9o+,J9o+,T8o+,98o"),
    "SB":  _expand_to_freq_dict("22+,A2s+,K2s+,Q4s+,J6s+,T7s+,96s+,85s+,75s+,64s+,54s,A2o+,K7o+,Q8o+,J8o+,T8o+,98o"),
}


# === 3-bet & call ranges by (my_position, raiser_position) ===
THREEBET_FREQS: Dict[Tuple[str, str], Dict[str, float]] = {
    # In-position
    ("HJ", "LJ"): _expand_to_freq_dict("TT+,AKs,AQs,AJs,A5s,KQs,KJs,KTs,AKo,AQo"),
    ("CO", "LJ"): _expand_to_freq_dict("TT+,AKs,AQs,AJs,A5s,A4s,KQs,KJs,KTs,AKo,AQo"),
    ("CO", "HJ"): _expand_to_freq_dict("TT+,AKs,AQs,AJs,A5s,A4s,KQs,KJs,KTs,AKo,AQo,AJo"),
    ("BTN", "LJ"): _expand_to_freq_dict("JJ+,AKs,AQs,AJs,A5s,A4s,KQs,AKo,AQo"),
    ("BTN", "HJ"): _expand_to_freq_dict("QQ+,AKs,AQs,AJs,A5s,A4s,KQs,KJs,AKo,AQo"),
    ("BTN", "CO"): _expand_to_freq_dict("TT+,AKs,AQs,AJs,A5s,A4s,KQs,KJs,AKo,AQo,AJo"),
    # Out-of-position (SB)
    ("SB", "LJ"): _expand_to_freq_dict("TT+,AKs,AQs,AJs,A5s,KQs,AKo,AQo"),
    ("SB", "HJ"): _expand_to_freq_dict("TT+,AKs,AQs,AJs,A5s,A4s,KQs,AKo,AQo,AJo"),
    ("SB", "CO"): _expand_to_freq_dict("99+,AKs,AQs,AJs,ATs,A5s,A4s,KQs,KJs,AKo,AQo,AJo"),
    ("SB", "BTN"): _expand_to_freq_dict("88+,A9s+,A5s,A4s,KTs+,QTs+,JTs,AKo,AQo,AJo,KQo"),
    # BB
    ("BB", "LJ"): _expand_to_freq_dict("QQ+,AKs,A5s,AKo,AQs"),
    ("BB", "HJ"): _expand_to_freq_dict("JJ+,AKs,AQs,A5s,A4s,KQs,AKo,AQo"),
    ("BB", "CO"): _expand_to_freq_dict("TT+,AKs,AQs,AJs,A5s,A4s,KQs,AKo,AQo"),
    ("BB", "BTN"): _expand_to_freq_dict("99+,AKs,AQs,AJs,ATs,A5s,A4s,A3s,KQs,KJs,AKo,AQo,AJo,KQo"),
}


# === 3-bet calling ranges (when we don't 3-bet but defend) ===
THREEBET_CALL_FREQS: Dict[Tuple[str, str], Dict[str, float]] = {
    # BTN call ranges
    ("BTN", "LJ"): _expand_to_freq_dict("TT,99,88,77,76s,65s,54s,JTs,T9s,98s,QJs,KJs,KTs,ATs,QTs"),
    ("BTN", "HJ"): _expand_to_freq_dict("JJ-77,T9s,98s,87s,76s,JTs,QJs,KJs,KTs,ATs"),
    ("BTN", "CO"): _expand_to_freq_dict("99-66,T9s,98s,87s,76s,JTs,QJs,KTs"),
    # BB calling ranges (much wider - they get a discount)
    ("BB", "LJ"): _expand_to_freq_dict("22-JJ,A2s-AJs,K9s-KQs,Q9s+,J9s+,T9s,98s,87s,76s,65s,ATo+,KTo+,QTo+,JTo"),
    ("BB", "HJ"): _expand_to_freq_dict("22-TT,A2s-AJs,K8s-KJs,Q8s+,J8s+,T8s,97s+,86s+,75s+,65s,54s,A9o+,KTo+,QTo+,JTo"),
    ("BB", "CO"): _expand_to_freq_dict("22-99,A2s-ATs,K6s-KJs,Q6s+,J7s+,T7s+,96s+,85s+,75s+,65s,54s,43s,A7o+,K9o+,Q9o+,J9o+,T9o"),
    ("BB", "BTN"): _expand_to_freq_dict("22-88,A2s-A9s,K2s-KTs,Q2s+,J5s+,T5s+,95s+,84s+,74s+,63s+,53s+,42s+,32s,A2o-ATo,K6o-KJo,Q8o+,J8o+,T8o,97o+,87o,76o,65o"),
}


# === 4-bet and 5-bet ranges (value-heavy at 100bb) ===
FOURBET_VALUE_FREQS: Dict[str, float] = _expand_to_freq_dict("QQ+,AKs,AKo")
FOURBET_BLUFF_FREQS: Dict[str, float] = _expand_to_freq_dict("A5s,A4s")
FIVEBET_FREQS: Dict[str, float] = _expand_to_freq_dict("KK+,AKs")


# === Heads-up ranges (separate chart, big difference from 6-max) ===
HU_BTN_OPEN_FREQS: Dict[str, float] = _expand_to_freq_dict(
    "22+,A2s+,K2s+,Q2s+,J2s+,T2s+,93s+,82s+,72s+,62s+,52s+,42s+,32s,"
    "A2o+,K2o+,Q4o+,J5o+,T6o+,97o+,87o,76o,65o,54o"
)
HU_BB_3BET_FREQS: Dict[str, float] = _expand_to_freq_dict(
    "TT+,A9s+,A5s,A4s,KQs,KJs,QJs,JTs,AJo+,KQo"
)
HU_BB_CALL_FREQS: Dict[str, float] = _expand_to_freq_dict(
    "22-99,A2s-A8s,K2s-KJs,Q2s-QTs,J3s-JTs,T3s-T9s,93s-98s,83s-87s,73s-76s,"
    "63s-65s,53s-54s,A2o-ATo,K2o-KJo,Q4o-QTo,J5o-J9o,T6o-T9o,97o-98o,87o,76o"
)
HU_BTN_4BET_FREQS: Dict[str, float] = _expand_to_freq_dict("KK+,AKs,AKo")
HU_BB_5BET_FREQS: Dict[str, float] = _expand_to_freq_dict("KK+,AKs")


# Cached set of "tight monster" hands used in jam-or-fold spots.
TIGHT_MONSTERS = {"AA", "KK", "QQ", "AKs"}
PURE_MONSTERS = {"AA", "KK"}


# ============================================================================
# 5. HELPER FUNCTIONS (pure utilities, no decision logic)
# ============================================================================

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


def board_texture(board: List[str]) -> str:
    """Classify board: 'dry', 'wet', 'medium'."""
    if len(board) < 3:
        return "dry"
    ranks = [RANK_IDX[c[0]] for c in board]
    suits = [c[1] for c in board]
    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit = max(suit_counts.values())
    sorted_r = sorted(ranks)
    gap = max(sorted_r) - min(sorted_r)
    has_pair = len(set(ranks)) < len(ranks)
    if max_suit >= 3:
        return "wet"
    if max_suit == 2 and gap <= 4:
        return "wet"
    if gap <= 4 and not has_pair:
        return "wet"
    if has_pair:
        return "dry"
    if gap >= 6 and max_suit < 2:
        return "dry"
    return "medium"


def safe_raise_amount(state: dict, target: int) -> int:
    """Clamp raise to legal bounds."""
    stack = state["your_stack"]
    bet_so_far = state["your_bet_this_street"]
    max_raise = stack + bet_so_far
    target = max(int(target), state["min_raise_to"])
    target = min(target, max_raise)
    return target


def stack_tightness(stack_bb: float, cfg: Config) -> float:
    """Returns multiplier for opening/3-bet frequencies based on stack depth.
    Smoothly interpolates from 1.0 at full stack to cfg.stack_short_tightness at shallow."""
    if stack_bb >= cfg.stack_full_threshold_bb:
        return 1.0
    if stack_bb <= cfg.stack_short_threshold_bb:
        return cfg.stack_short_tightness
    range_bb = cfg.stack_full_threshold_bb - cfg.stack_short_threshold_bb
    if range_bb <= 0:
        return 1.0
    ratio = (stack_bb - cfg.stack_short_threshold_bb) / range_bb
    return cfg.stack_short_tightness + ratio * (1.0 - cfg.stack_short_tightness)


def field_widening(n_active: int, cfg: Config) -> float:
    """Multiplier that widens ranges when fewer than 6 players are active.
    Returns 1.0 at full ring, scales up as field shrinks."""
    if n_active >= 6:
        return 1.0
    return 1.0 + cfg.shrink_widening_factor * (6 - n_active)


def stack_risked_pct(state: dict, owed: int) -> float:
    """Fraction of effective stack at risk if we call this bet."""
    stack = state["your_stack"]
    bet_so_far = state["your_bet_this_street"]
    total_invested_if_call = bet_so_far + owed
    starting_stack_estimate = stack + bet_so_far
    if starting_stack_estimate <= 0:
        return 1.0
    return min(1.0, total_invested_if_call / starting_stack_estimate)


def get_hand_rng(state: dict) -> random.Random:
    """Deterministic per-hand RNG. Same hand_id + same matchup = same decisions.
    This is what lets the harness's CRN testing cancel out our randomness."""
    hand_id = state.get("hand_id", "")
    seat = state.get("seat_to_act", 0)
    # Mix hand_id and seat so different seats see different randomness within a hand
    seed_str = f"{hand_id}:{seat}:{len(state.get('action_log', []))}"
    return random.Random(hash(seed_str) & 0xFFFFFFFF)


def lookup_freq(freq_dict: Dict[str, float], hand: str) -> float:
    """Get raise frequency for hand. Returns 0.0 if not in chart."""
    return freq_dict.get(hand, 0.0)


# ============================================================================
# 6. OPPONENT MODELLING
# ============================================================================

# Population priors (Bayesian baseline)
PRIOR_VPIP = 0.30
PRIOR_PFR = 0.20
PRIOR_3BET = 0.08
PRIOR_FOLD_TO_3BET = 0.55
PRIOR_FOLD_TO_CBET = 0.50
PRIOR_AGGRESSION = 0.25


def _new_opponent():
    return {
        "hands_observed": 0,
        "vpip_actions": 0, "vpip_chances": 0,
        "pfr_actions": 0, "pfr_chances": 0,
        "threebet_actions": 0, "threebet_chances": 0,
        "fold_to_3bet": 0, "faced_3bet": 0,
        "fold_to_cbet": 0, "faced_cbet": 0,
        "postflop_aggression": 0, "postflop_actions": 0,
    }


OPPONENTS: Dict[str, dict] = defaultdict(_new_opponent)
PROCESSED_HANDS: set = set()


def update_opponents_from_log(state: dict):
    """Parse action_log to update per-opponent counters. Idempotent per hand."""
    hand_id = state.get("hand_id", "")
    log = state.get("action_log", [])
    players_by_seat = {p["seat"]: p["bot_id"] for p in state["players"]}

    sb_seat = bb_seat = None
    for entry in log:
        if entry.get("action") == "small_blind":
            sb_seat = entry["seat"]
        elif entry.get("action") == "big_blind":
            bb_seat = entry["seat"]
            break

    first_action_seen = set()
    raised_already = set()
    for entry in log:
        seat = entry.get("seat")
        action = entry.get("action")
        if seat is None or action in ("small_blind", "big_blind"):
            continue
        bot_id = players_by_seat.get(seat)
        if not bot_id:
            continue
        opp = OPPONENTS[bot_id]

        if seat not in first_action_seen:
            first_action_seen.add(seat)
            if action != "fold":
                opp["vpip_actions"] += 1
            opp["vpip_chances"] += 1

            if action in ("raise", "all_in"):
                if not raised_already:
                    opp["pfr_actions"] += 1
                else:
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

        if action in ("raise", "all_in"):
            raised_already.add(seat)

    if state.get("type") == "hand_complete":
        PROCESSED_HANDS.add(hand_id)
        for bot_id in players_by_seat.values():
            OPPONENTS[bot_id]["hands_observed"] += 1


def is_maniac(bot_id: str, cfg: Config) -> bool:
    """Opponent shows random/aggressive pattern - don't call their shoves."""
    if bot_id not in OPPONENTS:
        return False
    opp = OPPONENTS[bot_id]
    if opp.get("vpip_chances", 0) < cfg.maniac_min_sample:
        return False
    vpip = opp["vpip_actions"] / max(opp["vpip_chances"], 1)
    pfr = opp["pfr_actions"] / max(opp["pfr_chances"], 1)
    return vpip > cfg.maniac_vpip_threshold and pfr > cfg.maniac_pfr_threshold


def is_calling_station(bot_id: str, cfg: Config) -> bool:
    """Opponent calls too much, never bluff them."""
    if bot_id not in OPPONENTS:
        return False
    opp = OPPONENTS[bot_id]
    if opp.get("vpip_chances", 0) < cfg.station_min_sample:
        return False
    vpip = opp["vpip_actions"] / max(opp["vpip_chances"], 1)
    pfr = opp["pfr_actions"] / max(opp["pfr_chances"], 1)
    return vpip > cfg.station_vpip_threshold and pfr < cfg.station_pfr_threshold


def is_unknown(bot_id: str, cfg: Config) -> bool:
    """Not enough hands observed to trust any read."""
    if bot_id not in OPPONENTS:
        return True
    return OPPONENTS[bot_id].get("hands_observed", 0) < cfg.cold_start_threshold_hands


def any_active_maniac(state: dict, cfg: Config) -> bool:
    """Any aggressor in this hand flagged as maniac."""
    log = state.get("action_log", [])
    me = state["seat_to_act"]
    for e in log:
        if e.get("action") in ("raise", "all_in") and e.get("seat") != me:
            seat = e["seat"]
            bot_id = next((p["bot_id"] for p in state["players"] if p["seat"] == seat), None)
            if bot_id and is_maniac(bot_id, cfg):
                return True
    return False


def any_active_unknown(state: dict, cfg: Config) -> bool:
    """Any opponent in the hand we don't have enough data on yet."""
    me = state["seat_to_act"]
    for p in state["players"]:
        if p.get("seat") == me or p.get("is_folded"):
            continue
        if is_unknown(p["bot_id"], cfg):
            return True
    return False


# ============================================================================
# 7. POSITION DERIVATION
# ============================================================================

def get_position_label(state: dict) -> str:
    """Compute position label: LJ/HJ/CO/BTN/SB/BB based on dealer derived from log."""
    n = len(state["players"])
    my_seat = state["seat_to_act"]
    log = state.get("action_log", [])

    bb_seat = sb_seat = None
    for entry in log:
        if entry.get("action") == "big_blind":
            bb_seat = entry["seat"]
        elif entry.get("action") == "small_blind":
            sb_seat = entry["seat"]

    if bb_seat is None:
        return "MP"

    if n == 2:
        return "BTN" if my_seat == sb_seat else "BB"

    btn_seat = (bb_seat - 2) % n
    offset = (my_seat - btn_seat) % n

    if n >= 6:
        return {0: "BTN", 1: "SB", 2: "BB", 3: "LJ", 4: "HJ", 5: "CO"}.get(offset, "LJ")
    elif n == 5:
        return {0: "BTN", 1: "SB", 2: "BB", 3: "HJ", 4: "CO"}.get(offset, "HJ")
    elif n == 4:
        return {0: "BTN", 1: "SB", 2: "BB", 3: "CO"}.get(offset, "CO")
    elif n == 3:
        return {0: "BTN", 1: "SB", 2: "BB"}.get(offset, "BTN")
    return "MP"


def get_opp_position(state: dict, opp_seat: int) -> str:
    """Compute opponent's position label from their seat."""
    n = len(state["players"])
    log = state.get("action_log", [])
    bb_seat = sb_seat = None
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
        return {0: "BTN", 1: "SB", 2: "BB", 3: "LJ", 4: "HJ", 5: "CO"}.get(offset, "LJ")
    elif n == 5:
        return {0: "BTN", 1: "SB", 2: "BB", 3: "HJ", 4: "CO"}.get(offset, "HJ")
    elif n == 4:
        return {0: "BTN", 1: "SB", 2: "BB", 3: "CO"}.get(offset, "CO")
    elif n == 3:
        return {0: "BTN", 1: "SB", 2: "BB"}.get(offset, "BTN")
    return "MP"


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


def find_aggressor_seat(state: dict) -> Optional[int]:
    """Return seat of last aggressor before us, if any."""
    log = state.get("action_log", [])
    me = state["seat_to_act"]
    for e in reversed(log):
        if e.get("action") in ("raise", "all_in") and e.get("seat") != me:
            return e["seat"]
    return None


def preflop_scenario(state: dict) -> str:
    """Classify the preflop situation."""
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


# ============================================================================
# 8. EQUITY CALCULATION
# ============================================================================

_EQUITY_CACHE: Dict[tuple, float] = {}


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


def equity_vs_random(hole_cards: List[str], community_cards: List[str],
                     n_sims: int = 300, n_opp: int = 1) -> float:
    """Monte Carlo equity vs random opponent hand(s)."""
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


def _hand_class_to_combos(hand_class: str, used: set) -> List[Tuple[str, str]]:
    """All card-combo pairs for a hand class like 'AKs' or 'TT'."""
    combos = []
    if len(hand_class) == 2:
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
        else:
            suits = "shdc"
            for s1 in suits:
                for s2 in suits:
                    if s1 == s2:
                        continue
                    c1, c2 = r1 + s1, r2 + s2
                    if c1 not in used and c2 not in used:
                        combos.append((c1, c2))
    return combos


def equity_vs_range(hole_cards: List[str], community_cards: List[str],
                    villain_range: Dict[str, float], n_sims: int = 300) -> float:
    """Monte Carlo equity vs a frequency-weighted range."""
    if not HAVE_EVAL7 or not villain_range:
        return equity_vs_random(hole_cards, community_cards, n_sims)

    try:
        my_cards = [eval7.Card(c) for c in hole_cards]
        board = [eval7.Card(c) for c in community_cards]
        used_str = set(str(c) for c in my_cards + board)
        deck_strs = [r + s for r in RANKS for s in "shdc" if (r + s) not in used_str]

        # Weight combos by frequency
        weighted_combos = []
        for hand_class, freq in villain_range.items():
            if freq <= 0:
                continue
            combos = _hand_class_to_combos(hand_class, used_str)
            for combo in combos:
                weighted_combos.append((combo, freq))

        if not weighted_combos:
            return equity_vs_random(hole_cards, community_cards, n_sims)

        wins = ties = 0
        needed = 5 - len(board)
        # Build weighted choice list
        weights = [w for _, w in weighted_combos]
        for _ in range(n_sims):
            v_combo, _ = random.choices(weighted_combos, weights=weights, k=1)[0]
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


def aggressor_likely_range(state: dict, agg_seat: int) -> Dict[str, float]:
    """Estimate aggressor's likely range based on position and action history."""
    agg_pos = get_opp_position(state, agg_seat)
    aggressors = count_aggressors(state)
    if aggressors == 1 and agg_pos in RFI_FREQS:
        return RFI_FREQS[agg_pos]
    if aggressors == 2:
        # 3-bet range: tight value
        return _expand_to_freq_dict("QQ+,AKs,AKo,A5s")
    return RFI_FREQS.get(agg_pos, RFI_FREQS["LJ"])


# ============================================================================
# 9. PREFLOP DECISION
# ============================================================================

def _effective_freq(base_freq: float, position: str, scenario: str,
                    stack_bb: float, n_active: int, cfg: Config) -> float:
    """Apply tightness/aggression/stack/field adjustments to a base chart frequency."""
    if base_freq <= 0:
        return 0.0

    pos_mult = getattr(cfg, f"pos_aggression_{position.lower()}", 1.0)
    stack_mult = stack_tightness(stack_bb, cfg)
    field_mult = field_widening(n_active, cfg)

    if scenario == "open":
        scenario_mult = cfg.rfi_tightness
    elif scenario == "threebet":
        scenario_mult = cfg.threebet_tightness
    elif scenario == "fourbet":
        scenario_mult = cfg.fourbet_tightness
    else:
        scenario_mult = 1.0

    return min(1.0, base_freq * pos_mult * stack_mult * field_mult * scenario_mult)


def decide_preflop_6max(state: dict, position: str, hand: str, cfg: Config,
                        rng: random.Random) -> dict:
    pot = state["pot"]
    owed = state["amount_owed"]
    can_check = state["can_check"]
    log = state.get("action_log", [])
    stack_bb = state["your_stack"] / BIG_BLIND
    n_active = sum(1 for p in state["players"] if not p.get("is_folded"))

    # === HEADS-UP BRANCH ===
    if n_active == 2:
        return decide_preflop_hu(state, position, hand, cfg, rng)

    scenario = preflop_scenario(state)
    facing_maniac = any_active_maniac(state, cfg)

    # === SCENARIO: Open or check ===
    if scenario == "open":
        # If maniac at table, restrict to top of range
        if facing_maniac:
            tight_set = _expand_to_freq_dict("66+,AJs+,KQs,AQo+,AKo")
            base_freq = tight_set.get(hand, 0.0)
        else:
            base_freq = lookup_freq(RFI_FREQS.get(position, {}), hand)

        eff_freq = _effective_freq(base_freq, position, "open", stack_bb, n_active, cfg)
        if eff_freq > 0 and rng.random() < eff_freq:
            limpers = sum(1 for e in log if e.get("action") == "call")
            target = int(BIG_BLIND * (cfg.open_size_bb + limpers))
            return {"action": "raise", "amount": safe_raise_amount(state, target)}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # === SCENARIO: Facing a single open raise ===
    if scenario == "face_open":
        agg_seat = find_aggressor_seat(state)
        agg_pos = get_opp_position(state, agg_seat) if agg_seat is not None else "LJ"

        threebet_range = THREEBET_FREQS.get((position, agg_pos), {})
        call_range = THREEBET_CALL_FREQS.get((position, agg_pos), {})

        # Exploit: opponent folds to 3-bets too often → 3-bet wider
        if agg_seat is not None:
            opp_id = next((p["bot_id"] for p in state["players"] if p["seat"] == agg_seat), None)
            if opp_id and OPPONENTS[opp_id].get("faced_3bet", 0) >= cfg.min_hands_for_exploit:
                fold_to_3bet = (OPPONENTS[opp_id]["fold_to_3bet"] /
                                max(OPPONENTS[opp_id]["faced_3bet"], 1))
                if fold_to_3bet >= cfg.fold_to_3bet_exploit_threshold:
                    if len(hand) == 3 and hand[0] == "A" and hand[2] == "s":
                        threebet_range = dict(threebet_range)
                        threebet_range[hand] = 1.0

        threebet_freq = lookup_freq(threebet_range, hand)
        eff_3bet = _effective_freq(threebet_freq, position, "threebet", stack_bb, n_active, cfg)

        if eff_3bet > 0 and rng.random() < eff_3bet:
            ip = position in ("CO", "BTN")
            current = state["current_bet"]
            mult = cfg.threebet_size_ip if ip else cfg.threebet_size_oop
            target = int(current * mult)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}

        call_freq = lookup_freq(call_range, hand)
        if call_freq > 0 and owed <= state["your_stack"] * cfg.threebet_call_threshold_pct:
            return {"action": "call"}

        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # === SCENARIO: We opened, opp 3-bet us ===
    if scenario == "face_3bet_as_raiser":
        risk_pct = stack_risked_pct(state, owed)

        if facing_maniac and risk_pct >= cfg.stack_risk_high_threshold:
            if hand in TIGHT_MONSTERS:
                return {"action": "all_in"}
            return {"action": "fold"}

        # Compute proposed 4-bet to check commit
        ip = position in ("CO", "BTN")
        current = state["current_bet"]
        mult = cfg.fourbet_size_ip if ip else cfg.fourbet_size_oop
        proposed_4bet = int(current * mult)
        chips_to_4bet = proposed_4bet - state["your_bet_this_street"]
        fourbet_risk = chips_to_4bet / max(state["your_stack"] + state["your_bet_this_street"], 1)

        # Jam-or-fold if 4-bet commits too much
        if fourbet_risk >= cfg.fourbet_commit_threshold:
            fivebet_freq = lookup_freq(FIVEBET_FREQS, hand)
            if fivebet_freq > 0 or hand in TIGHT_MONSTERS:
                return {"action": "all_in"}
            return {"action": "fold"}

        # Shallow stack jam-or-fold
        if stack_bb < cfg.shallow_jam_threshold_bb:
            fivebet_freq = lookup_freq(FIVEBET_FREQS, hand)
            if fivebet_freq > 0 or hand in TIGHT_MONSTERS:
                return {"action": "all_in"}
            return {"action": "fold"}

        # Standard 4-bet for value
        value_freq = lookup_freq(FOURBET_VALUE_FREQS, hand)
        eff_4bet = _effective_freq(value_freq, position, "fourbet", stack_bb, n_active, cfg)
        if eff_4bet > 0 and rng.random() < eff_4bet:
            return {"action": "raise", "amount": safe_raise_amount(state, proposed_4bet)}

        # 4-bet bluff with blockers - not vs maniacs
        if not facing_maniac and ip:
            bluff_freq = lookup_freq(FOURBET_BLUFF_FREQS, hand)
            if bluff_freq > 0 and rng.random() < cfg.fourbet_bluff_freq:
                return {"action": "raise", "amount": safe_raise_amount(state, proposed_4bet)}

        # Value-call with strong-but-not-4-bet hands
        if hand in {"JJ", "TT", "AKo", "AQs"} and owed <= state["your_stack"] * cfg.fourbet_call_threshold_pct:
            return {"action": "call"}

        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # === SCENARIO: We didn't open, two raisers in front (3-bet cold) ===
    if scenario == "face_3bet_cold":
        risk_pct = stack_risked_pct(state, owed)
        if facing_maniac and risk_pct >= cfg.stack_risk_high_threshold:
            if hand in PURE_MONSTERS:
                return {"action": "all_in"}
            return {"action": "fold"}

        if hand in TIGHT_MONSTERS:
            return {"action": "all_in"}
        if hand in {"JJ", "AKo", "AQs"} and owed <= state["your_stack"] * cfg.threebet_call_threshold_pct:
            return {"action": "call"}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # === SCENARIO: Facing a 4-bet or 5-bet ===
    if scenario in ("face_4bet_as_raiser", "face_5bet_as_raiser"):
        risk_pct = stack_risked_pct(state, owed)
        if facing_maniac and risk_pct >= cfg.stack_risk_high_threshold:
            if hand == "AA":
                return {"action": "all_in"}
            return {"action": "fold"}
        if lookup_freq(FIVEBET_FREQS, hand) > 0 or hand in PURE_MONSTERS:
            return {"action": "all_in"}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    if can_check:
        return {"action": "check"}
    return {"action": "fold"}


def decide_preflop_hu(state: dict, position: str, hand: str, cfg: Config,
                      rng: random.Random) -> dict:
    """Heads-up preflop with separate ranges."""
    aggressors = count_aggressors(state)
    can_check = state["can_check"]
    stack_bb = state["your_stack"] / BIG_BLIND

    if aggressors == 0:
        if position == "BTN":
            base_freq = lookup_freq(HU_BTN_OPEN_FREQS, hand)
            eff_freq = _effective_freq(base_freq, "BTN", "open", stack_bb, 2, cfg)
            if eff_freq > 0 and rng.random() < eff_freq:
                target = int(BIG_BLIND * cfg.open_size_bb)
                return {"action": "raise", "amount": safe_raise_amount(state, target)}
            if can_check:
                return {"action": "check"}
            return {"action": "fold"}
        else:
            if can_check:
                return {"action": "check"}
            return {"action": "fold"}

    if aggressors == 1:
        # We're BB facing BTN open
        threebet_freq = lookup_freq(HU_BB_3BET_FREQS, hand)
        eff_3bet = _effective_freq(threebet_freq, "BB", "threebet", stack_bb, 2, cfg)
        if eff_3bet > 0 and rng.random() < eff_3bet:
            current = state["current_bet"]
            target = int(current * cfg.threebet_size_oop)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}
        call_freq = lookup_freq(HU_BB_CALL_FREQS, hand)
        if call_freq > 0 and state["amount_owed"] <= state["your_stack"] * cfg.threebet_call_threshold_pct:
            return {"action": "call"}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    if aggressors == 2:
        fourbet_freq = lookup_freq(HU_BTN_4BET_FREQS, hand)
        eff_4bet = _effective_freq(fourbet_freq, "BTN", "fourbet", stack_bb, 2, cfg)
        if eff_4bet > 0 and rng.random() < eff_4bet:
            current = state["current_bet"]
            target = int(current * cfg.fourbet_size_ip)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}
        if hand in {"QQ", "JJ", "AKo", "AQs"}:
            return {"action": "call"}
        return {"action": "fold"}

    if lookup_freq(HU_BB_5BET_FREQS, hand) > 0 or hand in PURE_MONSTERS:
        return {"action": "all_in"}
    return {"action": "fold"}


# ============================================================================
# 10. POSTFLOP DECISION
# ============================================================================

def decide_postflop(state: dict, position: str, cfg: Config,
                    rng: random.Random) -> dict:
    hole = state["your_cards"]
    board = state["community_cards"]
    pot = state["pot"]
    owed = state["amount_owed"]
    can_check = state["can_check"]
    stack = state["your_stack"]
    street = state["street"]
    log = state.get("action_log", [])
    me = state["seat_to_act"]

    # Were we the preflop aggressor?
    pf_log = [e for e in log if e.get("action") in ("raise", "all_in")]
    was_pf_aggressor = bool(pf_log) and pf_log[0].get("seat") == me

    # Active opponent count
    n_opp = sum(1 for p in state["players"]
                if not p.get("is_folded") and p.get("seat") != me and not p.get("is_all_in"))
    n_opp = max(1, n_opp)

    # Sim count by street
    if street == "flop":
        n_sims = cfg.mc_sims_flop
    elif street == "turn":
        n_sims = cfg.mc_sims_turn
    else:
        n_sims = cfg.mc_sims_river

    # Equity
    agg_seat = find_aggressor_seat(state)
    if agg_seat is not None and len(log) > 4:
        v_range = aggressor_likely_range(state, agg_seat)
        eq = equity_vs_range(hole, board, v_range, n_sims=n_sims)
    else:
        eq = equity_vs_random(hole, board, n_sims=n_sims, n_opp=min(n_opp, 3))

    texture = board_texture(board)
    in_position = position in ("CO", "BTN")

    # Opponent profile
    facing_maniac = any_active_maniac(state, cfg)
    facing_station = False
    if agg_seat is not None:
        agg_id = next((p["bot_id"] for p in state["players"] if p["seat"] == agg_seat), None)
        if agg_id and is_calling_station(agg_id, cfg):
            facing_station = True

    # Cold-start caution: only applied when FACING aggression, not when we initiate.
    # Adding it to value-betting thresholds was making us too passive in early hands.
    cold_caution_call = cfg.cold_start_caution if any_active_unknown(state, cfg) else 0.0

    # === Free check option ===
    if can_check:
        # Strong: bet for value (no cold caution - we're initiating, not calling)
        if eq >= cfg.equity_value_bet:
            target = int(state["current_bet"] + pot * cfg.sizing_value)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}

        # PFR continuation bet (with multiway penalty)
        if was_pf_aggressor and street == "flop":
            cbet_freq = cfg.cbet_freq_dry if texture == "dry" else cfg.cbet_freq_wet
            # Multiway penalty
            cbet_freq *= cfg.cbet_multiway_penalty ** (n_opp - 1)
            # No bluff c-bets vs station
            if facing_station and eq < cfg.equity_thin_value:
                cbet_freq = 0.0
            if rng.random() < cbet_freq:
                target = int(state["current_bet"] + pot * cfg.cbet_size_pct)
                return {"action": "raise", "amount": safe_raise_amount(state, target)}

        # Thin value IP - not vs station with weak hand
        if eq >= cfg.equity_thin_value and in_position:
            target = int(state["current_bet"] + pot * cfg.sizing_thin)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}

        # Bluff - never vs station
        if facing_station:
            return {"action": "check"}
        bluff_freq = cfg.bluff_freq_ip if in_position else cfg.bluff_freq_oop
        if eq < cfg.equity_thin_value and rng.random() < bluff_freq:
            target = int(state["current_bet"] + pot * cfg.sizing_thin)
            return {"action": "raise", "amount": safe_raise_amount(state, target)}

        return {"action": "check"}

    # === Facing a bet ===
    if owed <= 0:
        return {"action": "check"}

    pot_odds = owed / (pot + owed) if (pot + owed) > 0 else 1.0
    risk_pct = stack_risked_pct(state, owed)

    # === STACK PRESERVATION GUARD (cold caution applies here - we're calling unknowns) ===
    if risk_pct >= cfg.stack_risk_high_threshold:
        threshold = (cfg.stack_risk_high_eq_maniac if facing_maniac
                     else cfg.stack_risk_high_eq_normal)
        threshold += cold_caution_call
        if eq >= threshold:
            return {"action": "call"}
        return {"action": "fold"}

    if risk_pct >= cfg.stack_risk_medium_threshold:
        threshold = (cfg.stack_risk_med_eq_maniac if facing_maniac
                     else cfg.stack_risk_med_eq_normal)
        threshold += cold_caution_call
        if eq >= threshold:
            if eq >= cfg.equity_raise_threshold and not facing_maniac:
                target = int(state["current_bet"] + (pot + owed) * cfg.sizing_value)
                return {"action": "raise", "amount": safe_raise_amount(state, target)}
            return {"action": "call"}
        return {"action": "fold"}

    # === Normal-sized bet (cold caution applied to call thresholds, not raises) ===
    if eq >= cfg.equity_raise_threshold:
        target = int(state["current_bet"] + (pot + owed) * cfg.sizing_value)
        return {"action": "raise", "amount": safe_raise_amount(state, target)}

    if eq >= cfg.equity_value_bet:
        return {"action": "call"}

    if eq >= pot_odds + cfg.pot_odds_buffer_normal + cold_caution_call:
        return {"action": "call"}

    if eq >= cfg.equity_call_threshold and owed <= pot * cfg.pot_odds_buffer_marginal:
        return {"action": "call"}

    return {"action": "fold"}


# ============================================================================
# 11. MAIN ENTRY POINT
# ============================================================================

def decide(game_state: dict) -> dict:
    """Engine entry. Must return within 2 seconds."""
    global _EQUITY_CACHE
    _EQUITY_CACHE = {}
    t0 = time.time()

    try:
        try:
            update_opponents_from_log(game_state)
        except Exception:
            pass

        position = get_position_label(game_state)
        hand = hand_str(game_state["your_cards"])
        street = game_state["street"]
        rng = get_hand_rng(game_state)

        if street == "preflop":
            action = decide_preflop_6max(game_state, position, hand, CONFIG, rng)
        else:
            action = decide_postflop(game_state, position, CONFIG, rng)

        if time.time() - t0 > CONFIG.time_budget_sec:
            return {"action": "check"} if game_state.get("can_check") else {"action": "fold"}

        return action

    except Exception:
        if game_state.get("can_check"):
            return {"action": "check"}
        if game_state.get("amount_owed", 999) <= game_state.get("pot", 0) * 0.10:
            return {"action": "call"}
        return {"action": "fold"}
