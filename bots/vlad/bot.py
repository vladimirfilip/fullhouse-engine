"""Vlad's NLHE bot — GTO network + preflop CFR table + MC fallback.

decide() is called once per action and must return within 2 seconds.
See CLAUDE.md for architecture details and submission format.
"""

import math
import os
import random
import time
from bisect import bisect_left as _bisect_left

import numpy as np
from eval7 import Card, evaluate, handtype

BOT_NAME   = "The House"
BOT_AVATAR = "robot_1"

RANKS   = "23456789TJQKA"
SUITS   = "shdc"
ALL_CARDS = [Card(r + s) for r in RANKS for s in SUITS]

_N_PLAYERS    = 6
_INITIAL_STACK = 10_000
_SMALL_BLIND   = 50
_BIG_BLIND     = 100
_MAX_RAISES_PER_STREET = 8   # must match deep_cfr/config.py and config.hpp

_INPUT_DIM = 308   # must match deep_cfr/config.py and deep_cfr_cpp/src/config.hpp


def _envf(name: str, default: float) -> float:
    """Read a float tuning knob from the environment, falling back to `default`.

    Used so tools/tune_*.py can sweep parameters without editing this file. The
    baked-in defaults are the production values; os is already imported and env
    reads are validator-safe. Submission uses the defaults (no env set).
    """
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


# ── Latency guard ───────────────────────────────────────────────────────────
# The action timeout is 2 s on a 0.5-core box; a self-timeout auto-folds. Each
# decide() sets a wall-clock deadline well under the cap, and the Monte-Carlo
# loop (the only unbounded cost) stops once it's hit — degrading to a noisier
# equity estimate instead of ever blowing the budget. Overridable for tuning.
_TIME_BUDGET = _envf("VLAD_TIME_BUDGET", 1.5)
_MC_MIN_ITERS = 64      # floor so a deadline-truncated rollout stays meaningful
_DECIDE_DEADLINE: float | None = None   # absolute deadline (time.time()); per decide()

# ── Encodings (must match deep_cfr_cpp/src/features.cpp) ─────────────────
_CARD_IDX: dict[str, int] = {
    r + s: ri * 4 + si
    for ri, r in enumerate(RANKS)
    for si, s in enumerate(SUITS)
}
_ACTION_ONEHOT: dict[str, int] = {
    "fold": 0, "check": 1, "call": 1, "raise": 2, "all_in": 3,
}
_STREET_IDX: dict[str, int] = {
    "preflop": 0, "flop": 1, "turn": 2, "river": 3,
}

# Abstract action indices (must match deep_cfr/config.py)
_FOLD, _CHECK_CALL = 0, 1
_0_27X, _THIRD, _HALF, _FULL, _1_72X, _2X, _ALL_IN = 2, 3, 4, 5, 6, 7, 8

# ── Decision-engine configuration ──────────────────────────────────────────
# Disabled: the shipped preflop_strategy.npz was trained+keyed with the old
# bucket encoding (preflop_cfr/cards.py hand_to_bucket had its within-block lo
# ordering reversed, and the HU equity table was built with hole cards leaking
# onto the board).  Those bugs are now fixed, so the bot's bucket keys no longer
# match this stale table (non-pair hands would read the wrong hand's row).
# Preflop runs entirely on the chart/Chen heuristic until the table is retrained
# with the corrected solver.  See preflop_cfr/{cards,equity}.py.
_USE_PREFLOP_TABLE = False

# Postflop engine: "net" = trained GTO strategy net (balanced ranges), "mc" =
# Monte-Carlo equity + pot-odds.  Toggle is read by _postflop_decide so the two
# can be A/B-compared in a local tournament.  Overridable via env for sweeps.
_POSTFLOP_ENGINE = os.environ.get("VLAD_POSTFLOP_ENGINE", "mc")

# Risk gate: never commit more than this fraction of the effective stack on a
# single street without clearing the street/commitment equity floor (see
# _risk_gate).  Guards against stacking off light to pressure bots.
_RISK_COMMIT_FRAC = 0.35

# Raise ladder: (action_index, rb_fn) where rb_fn(eff_pot) -> raise-by amount.
# Single source of truth for _abstract_to_raw and _legal_actions.
# Rounding per entry matches the original C++ implementation exactly.
_RAISE_LADDER = [
    (_0_27X, lambda p: round(p * 0.27)),
    (_THIRD, lambda p: p // 3),
    (_HALF,  lambda p: p // 2),
    (_FULL,  lambda p: p),
    (_1_72X, lambda p: round(p * 1.72)),
    (_2X,    lambda p: p * 2),
]


def _hero_position(gs: dict) -> tuple[int, int]:
    """(hero_pos, n_in_game): hero's seat offset from the dealer and table size.

    The frozen engine doesn't surface dealer_seat, so derive it from the SB
    action: heads-up the dealer IS the SB; 3+-handed the SB sits one left of the
    dealer. Single source of truth for every position-dependent computation
    (feature vector, preflop info-set key, preflop heuristic).
    """
    seat      = gs["seat_to_act"]
    al        = gs["action_log"]
    n_in_game = max(len(gs["players"]), 1)
    dealer    = 0
    if al and al[0].get("action") == "small_blind":
        sb_seat = al[0]["seat"]
        dealer  = sb_seat if n_in_game == 2 else (sb_seat - 1) % n_in_game
    return (seat - dealer) % n_in_game, n_in_game


def _in_position(gs: dict) -> bool:
    """True if hero acts last among the still-live players this (postflop) street.

    Postflop the betting order runs clockwise from the small blind (offset 1 from
    the dealer) and the button (offset 0) acts last, so a player's "order rank" is
    (offset - 1) mod n. Hero is in position when no live opponent has a higher
    rank. Equity realises better in position (you see opponents act, can check
    back, bluff, and control the pot), so the continue threshold is relaxed IP and
    tightened OOP in _mc_postflop.
    """
    seat = gs["seat_to_act"]
    hero_off, n = _hero_position(gs)
    dealer = (seat - hero_off) % n

    def rank(off: int) -> int:
        return (off - 1) % n

    hero_rank = rank(hero_off)
    for p in gs["players"]:
        if p["seat"] == seat or p.get("is_folded") or p.get("state") in ("folded", "busted"):
            continue
        if rank((p["seat"] - dealer) % n) > hero_rank:
            return False
    return True


# ── Feature extraction (mirrors deep_cfr_cpp/src/features.cpp) ─────────────
# Layout — see deep_cfr/config.py docstring for the full byte-for-byte map.

def _derive_n_raises_this_street(action_log: list, n_seats: int) -> int:
    """Replicate the engine's per-street raise counter from the public action_log.

    The frozen engine resets n_raises_this_street to 0 at every street change
    and increments inside handle_aggression() when raise_size >= last_aggression_size.
    The dict view of the action log loses the per-street boundaries, so we walk
    forward simulating who still needs to act and detect each round close.
    Cheap (~150 actions per hand, single pass).
    """
    if not action_log:
        return 0

    # Initial set of seated players seen in the log.
    seats = sorted({e["seat"] for e in action_log})
    if not seats:
        return 0
    n = max(n_seats, len(seats))

    active        = set(seats)               # not folded
    all_in_set    = set()
    bet_this      = {s: 0 for s in seats}    # chips committed this street
    current_bet   = 0
    last_agg_size = 0
    n_raises      = 0
    to_act        = set()                    # who still needs to act this round
    street_after_blinds = False              # blinds set up preflop initial state

    def reset_street():
        nonlocal current_bet, last_agg_size, n_raises
        for s in seats:
            bet_this[s] = 0
        current_bet   = 0
        last_agg_size = _BIG_BLIND   # mirrors engine.cpp advance_street()
        n_raises      = 0

    def open_round(initiating_seat):
        # Everyone still capable of acting needs to act once.
        return {s for s in active if s not in all_in_set and s != initiating_seat}

    for e in action_log:
        seat = e["seat"]
        act  = e.get("action")
        amt  = e.get("amount", 0)

        if act == "small_blind":
            bet_this[seat] = amt
            current_bet = max(current_bet, amt)
            # Player posted less than the full blind → all-in from blind post.
            if amt < _SMALL_BLIND:
                all_in_set.add(seat)
            continue
        if act == "big_blind":
            bet_this[seat] = amt
            current_bet = max(current_bet, amt)
            last_agg_size = _BIG_BLIND
            if amt < _BIG_BLIND:
                all_in_set.add(seat)
            # Preflop: everyone except BB needs to act (BB is included so it
            # can exercise its option — mirrors engine start_hand semantics).
            to_act = {s for s in active if s not in all_in_set}
            street_after_blinds = True
            continue

        # Non-blind action; start tracking if we haven't yet.
        if not street_after_blinds:
            # Should not happen in a well-formed log, but guard anyway.
            street_after_blinds = True
            to_act = set(active) - all_in_set

        to_act.discard(seat)

        if act == "fold":
            active.discard(seat)
        elif act == "check":
            pass
        elif act == "call":
            bet_this[seat] = current_bet
        elif act in ("raise", "all_in"):
            new_bet  = amt
            raise_sz = new_bet - current_bet
            if raise_sz >= last_agg_size and raise_sz > 0:
                n_raises     += 1
                last_agg_size = raise_sz
                # Reopen action for everyone except the aggressor.
                to_act = {s for s in active if s not in all_in_set and s != seat}
            bet_this[seat] = new_bet
            current_bet    = max(current_bet, new_bet)
            if act == "all_in":
                all_in_set.add(seat)

        # Round closes when no one needs to act (everyone matched current_bet
        # or is folded/all-in). Advance street.
        round_over = (
            not to_act and
            all(bet_this[s] == current_bet for s in active if s not in all_in_set)
        )
        if round_over:
            reset_street()
            # Next street: postflop, first to act is left of dealer; we don't
            # need that — we just need to be ready to count raises again.
            to_act = set(active) - all_in_set

    return n_raises


def _build_feature_vector(gs: dict) -> np.ndarray:
    """Convert game-state dict → INPUT_DIM-float numpy array."""
    vec = np.zeros(_INPUT_DIM, dtype=np.float32)

    # ── 1. Hole cards [0:52] ────────────────────────────────────────────────
    for card in gs["your_cards"]:
        vec[_CARD_IDX[card]] = 1.0

    # ── 2. Board cards [52:104] ─────────────────────────────────────────────
    for card in gs["community_cards"]:
        vec[52 + _CARD_IDX[card]] = 1.0

    # ── 3. Hero position relative to dealer [104:110] ───────────────────────
    seat = gs["seat_to_act"]
    al   = gs["action_log"]
    hero_pos, n_in_game = _hero_position(gs)
    vec[104 + hero_pos] = 1.0

    # ── 4. Pot and stacks [110:117] ─────────────────────────────────────────
    pot = gs["pot"]
    vec[110] = pot / _INITIAL_STACK
    for p in gs["players"][:_N_PLAYERS]:
        vec[111 + p["seat"]] = p["stack"] / _INITIAL_STACK

    # ── 5. Per-seat status masks [117:135] ──────────────────────────────────
    for p in gs["players"][:_N_PLAYERS]:
        ps = p["seat"]
        if p.get("is_folded"):
            vec[117 + ps] = 1.0
        if p.get("is_all_in"):
            vec[123 + ps] = 1.0
        vec[129 + ps] = p.get("bet_this_street", 0) / _INITIAL_STACK

    # ── 6. Street one-hot [135:139] ─────────────────────────────────────────
    vec[135 + _STREET_IDX.get(gs.get("street", "preflop"), 0)] = 1.0

    # ── 7. Pot odds, SPR (log-scaled), owed [139:142] ───────────────────────
    owed     = gs["amount_owed"]
    vec[139] = owed / max(pot + owed, 1)
    spr      = gs["your_stack"] / max(pot, 1)
    vec[140] = min(math.log10(spr + 1.0) / math.log10(101.0), 1.0)
    vec[141] = owed / _INITIAL_STACK

    # ── 8. n_raises_this_street, hero bet, eff stack [142:145] ──────────────
    n_raises = _derive_n_raises_this_street(al, len(gs["players"]))
    vec[142] = min(n_raises, _MAX_RAISES_PER_STREET) / _MAX_RAISES_PER_STREET
    vec[143] = gs.get("your_bet_this_street", 0) / _INITIAL_STACK
    my_stack    = gs["your_stack"]
    max_opp     = 0
    for p in gs["players"]:
        if p["seat"] == seat or p.get("is_folded"):
            continue
        max_opp = max(max_opp, p.get("stack", 0))
    eff_stack   = min(my_stack, max_opp)
    vec[144] = eff_stack / _INITIAL_STACK

    # ── 9. Last aggressor [145:158] ─────────────────────────────────────────
    # Most recent RAISE / ALL_IN / BIG_BLIND in action_log (same definition
    # as features.cpp — bias-free across the train/inference boundary).
    last_agg_seat   = -1
    last_agg_amount = 0
    for e in al:
        if e.get("action") in ("raise", "all_in", "big_blind"):
            last_agg_seat   = e["seat"]
            last_agg_amount = e.get("amount", 0)
    if 0 <= last_agg_seat < _N_PLAYERS:
        vec[145 + last_agg_seat] = 1.0
        vec[151] = last_agg_amount / _INITIAL_STACK
        rel = (last_agg_seat - seat) % n_in_game
        vec[152 + rel] = 1.0

    # ── 10. Board texture [158:163], n_active [163] ─────────────────────────
    board = gs["community_cards"]
    if board:
        bidx     = [_CARD_IDX[c] for c in board]
        b_suits  = [c % 4 for c in bidx]
        b_ranks  = [c // 4 for c in bidx]
        suit_counts = [b_suits.count(s) for s in range(4)]
        max_suit = max(suit_counts)
        rank_cnt: dict[int, int] = {}
        for r in b_ranks:
            rank_cnt[r] = rank_cnt.get(r, 0) + 1
        pairs = sum(1 for cnt in rank_cnt.values() if cnt >= 2)
        rank_set = sorted(set(b_ranks))
        connected = any(rank_set[i + 1] - rank_set[i] == 1
                        for i in range(len(rank_set) - 1))
        vec[158] = 1.0 if max_suit >= 2 else 0.0
        vec[159] = 1.0 if max_suit >= 3 else 0.0
        vec[160] = 1.0 if pairs >= 1 else 0.0
        vec[161] = 1.0 if pairs >= 2 else 0.0
        vec[162] = 1.0 if connected else 0.0
    n_active = sum(1 for p in gs["players"]
                   if not p.get("is_folded") and p.get("state") != "busted")
    vec[163] = n_active / _N_PLAYERS

    # ── 11. Action history [164:308] — 24 slots × 6 floats ──────────────────
    regular = [e for e in al if e.get("action") not in ("small_blind", "big_blind")]
    last24  = regular[-24:]
    for slot, e in enumerate(last24):
        base = 164 + slot * 6
        vec[base] = e["seat"] / max(_N_PLAYERS - 1, 1)
        atype = _ACTION_ONEHOT.get(e.get("action", ""), -1)
        if 0 <= atype <= 3:
            vec[base + 1 + atype] = 1.0
        # Normalise by INITIAL_STACK so early bets are not compressed by
        # the large current pot (mirrors features.cpp).
        vec[base + 5] = e.get("amount", 0) / _INITIAL_STACK

    return vec


# ── Pure-numpy forward pass ────────────────────────────────────────────────

def _numpy_forward(
    layers: list[tuple[np.ndarray, np.ndarray]],
    x: np.ndarray,
) -> np.ndarray:
    """LeakyReLU MLP → softmax probability vector."""
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1:
            x = np.where(x > 0, x, 0.01 * x)   # LeakyReLU
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ── Model loading (at import / warmup time) ────────────────────────────────

DATA_DIR   = os.environ.get("BOT_DATA_DIR",
             os.path.join(os.path.dirname(__file__), "data"))
_MODEL_PATH = os.path.join(DATA_DIR, "better_gto", "gto_strategy.npz")

_GTO_LAYERS: list[tuple[np.ndarray, np.ndarray]] | None = None

_N_ACTIONS = 9   # must match deep_cfr/config.py N_ACTIONS

try:
    _data = np.load(_MODEL_PATH)
    _n    = int(_data["n_layers"])
    _layers_tmp = [(_data[f"layer{i}_w"], _data[f"layer{i}_b"]) for i in range(_n)]
    _in_dim  = _layers_tmp[0][0].shape[1]
    _out_dim = _layers_tmp[-1][0].shape[0]
    if _out_dim != _N_ACTIONS:
        raise ValueError(
            f"model output dim {_out_dim} != N_ACTIONS {_N_ACTIONS}; retrain required"
        )
    if _in_dim != _INPUT_DIM:
        raise ValueError(
            f"model input dim {_in_dim} != INPUT_DIM {_INPUT_DIM}; "
            f"retrain required (feature vector has changed)"
        )
    _GTO_LAYERS = _layers_tmp
    # print(f"[bot] Loaded GTO model ({_n} layers, in={_in_dim}, out={_out_dim}) "
    #       f"from {_MODEL_PATH}", flush=True)
except Exception as _e:
    pass
    # print(f"[bot] GTO model not available ({_e}); using Monte Carlo fallback.", flush=True)


# ── Preflop CFR table (import-time load) ──────────────────────────────────
# Mirrored from preflop_cfr/cards.py and preflop_cfr/abstraction.py.
# bot.py cannot import from preflop_cfr/ at runtime; keep this block
# byte-for-byte consistent with those modules.

_PREFLOP_TABLE_PATH = os.path.join(DATA_DIR, "preflop_cfr", "preflop_strategy.npz")
_PREFLOP_TABLE: dict[int, np.ndarray] | None = None

# -- mirrored: hand_to_bucket (preflop_cfr/cards.py) --
_PF_RANK_IDX: dict[str, int] = {r: i for i, r in enumerate(RANKS)}
_PF_CARD_RANK: dict[str, int] = {r + s: _PF_RANK_IDX[r] for r in RANKS for s in SUITS}
_PF_CARD_SUIT: dict[str, int] = {r + s: si for r in RANKS for si, s in enumerate(SUITS)}


def _preflop_bucket(c1: str, c2: str) -> int:
    r1, r2 = _PF_CARD_RANK[c1], _PF_CARD_RANK[c2]
    suited  = (_PF_CARD_SUIT[c1] == _PF_CARD_SUIT[c2])
    hi, lo  = (r1, r2) if r1 >= r2 else (r2, r1)
    if hi == lo:
        return 12 - hi
    # lo descending within each hi block (AKs=13, …, A2s=24): must match
    # preflop_cfr/cards.py hand_to_bucket exactly or table lookups miss.
    offset = (78 - hi * (hi + 1) // 2) + (hi - 1 - lo)
    return 13 + offset if suited else 91 + offset


# -- mirrored: FNV-1a 64-bit hash (preflop_cfr/abstraction.py) --
_PF_FNV_OFFSET = 14695981039346656037
_PF_FNV_PRIME  = 1099511628211


def _pf_fnv1a(data: bytes) -> int:
    import struct
    h = _PF_FNV_OFFSET
    for byte in data:
        h ^= byte
        h  = (h * _PF_FNV_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    return struct.unpack("q", struct.pack("Q", h))[0]


# -- mirrored: amount_to_abstract (preflop_cfr/abstraction.py) --
# All raise fractions, keyed by abstract action index.
_PF_RAISE_FRACS = {
    _0_27X: 0.27,
    _THIRD: 1.0 / 3.0,
    _HALF:  0.50,
    _FULL:  1.00,
    _1_72X: 1.72,
    _2X:    2.00,
}
# BUG-FIX (key mismatch): training (abstraction.py) only maps observed bets to
# raise actions in config.PREFLOP_ACTIONS.  The shipped table uses
# [HALF, FULL, 2X] (npz "actions" field).  Including the off-grid sizes here
# would emit history tokens that never exist in the table → guaranteed info-set
# misses on every raised pot.  Default to the training subset; the loader below
# overrides this from the table's own "actions" metadata.
_PF_TABLE_RAISE_ACTIONS = [_HALF, _FULL, _2X]
_PF_ACTIVE_RAISES = [(a, _PF_RAISE_FRACS[a]) for a in _PF_TABLE_RAISE_ACTIONS]


def _pf_amount_to_abstract(raise_to: int, pot: int,
                            current_bet: int, bet_this_street: int) -> int:
    eff_pot    = pot + max(0, current_bet - bet_this_street)
    raise_size = raise_to - current_bet
    if raise_size <= 0 or eff_pot <= 0:
        return _CHECK_CALL
    frac = raise_size / eff_pot
    best_idx, best_dist = _CHECK_CALL, float("inf")
    for a_idx, target_frac in _PF_ACTIVE_RAISES:
        dist = abs(frac - target_frac)
        if dist < best_dist:
            best_dist, best_idx = dist, a_idx
    return best_idx


# -- mirrored: infoset_key (preflop_cfr/abstraction.py) --
def _preflop_infoset_key(gs: dict) -> int:
    """Compute the preflop info-set hash for the current game state."""
    al          = gs["action_log"]
    hero_pos, _ = _hero_position(gs)
    bucket      = _preflop_bucket(gs["your_cards"][0], gs["your_cards"][1])

    # Replay action log to build history (non-blind actions only)
    # Track pot/current_bet so we can translate raise amounts.
    pot         = 0
    current_bet = _BIG_BLIND
    bets        = {}     # seat -> chips committed this street
    history     = []

    for e in al:
        act  = e.get("action", "")
        eseat = e["seat"]
        amt  = e.get("amount", 0)

        if act == "small_blind":
            bets[eseat]  = amt
            pot         += amt
            current_bet  = max(current_bet, amt)
            continue
        if act == "big_blind":
            bets[eseat]  = amt
            pot         += amt
            current_bet  = max(current_bet, amt)
            continue

        bst = bets.get(eseat, 0)

        if act == "fold":
            abstract = _FOLD
        elif act in ("check", "call"):
            abstract = _CHECK_CALL
            bets[eseat]  = current_bet
            pot         += max(0, current_bet - bst)
        elif act in ("raise", "all_in"):
            abstract     = _pf_amount_to_abstract(amt, pot, current_bet, bst)
            pot         += amt - bst
            current_bet  = max(current_bet, amt)
            bets[eseat]  = amt
        else:
            continue

        history.append((eseat, abstract))

    hist_tuple = tuple(a for _, a in history)
    raw = f"{hero_pos}|{'_'.join(map(str, hist_tuple))}|{bucket}"
    return _pf_fnv1a(raw.encode())


# Preflop table applicability: only use when near canonical 100bb 6-max config
_PF_STACK_TOL = 0.25   # allow stacks within ±25% of 100bb

if _USE_PREFLOP_TABLE:
    try:
        _pf_data = np.load(_PREFLOP_TABLE_PATH)
        _pf_n_players = int(_pf_data["n_players"])
        _pf_stack_bb  = int(_pf_data["stack_bb"])
        if _pf_n_players == _N_PLAYERS and _pf_stack_bb == _INITIAL_STACK // _BIG_BLIND:
            _pf_keys = _pf_data["keys"]
            _pf_strat = _pf_data["strategy"]
            _PREFLOP_TABLE = {int(k): _pf_strat[i] for i, k in enumerate(_pf_keys)}
            # Rebuild the raise-action subset from the table's own metadata so
            # inference histories match exactly what was trained (Bug B fix).
            if "actions" in _pf_data:
                _acts = [int(x) for x in str(_pf_data["actions"]).strip("[] ").split(",") if x.strip()]
                _raises = [a for a in _acts if a in _PF_RAISE_FRACS]
                if _raises:
                    _PF_ACTIVE_RAISES = [(a, _PF_RAISE_FRACS[a]) for a in _raises]
    except Exception:
        pass


def _preflop_applicable(gs: dict) -> bool:
    """Check whether the live game config is close enough to the solved 100bb 6-max."""
    if gs.get("street") != "preflop":
        return False
    players = gs["players"]
    active  = [p for p in players
               if not p.get("is_folded") and p.get("state") != "busted"]
    if len(active) != _N_PLAYERS:
        return False
    target = _INITIAL_STACK
    for p in active:
        stack = p["stack"]
        if p["seat"] == gs["seat_to_act"]:
            stack = gs["your_stack"]
        if abs(stack - target) > target * _PF_STACK_TOL:
            return False
    return True


# ── Abstract action → engine dict ─────────────────────────────────────────

def _jitter_raise(target: int, min_r: int, all_tot: int, jitter: float = 0.05) -> dict:
    """Apply +-jitter% uniform noise to a raise amount, then clamp to legal range."""
    jittered = int(target * random.uniform(1 - jitter, 1 + jitter))
    jittered = max(jittered, min_r)
    if jittered >= all_tot:
        return {"action": "all_in"}
    return {"action": "raise", "amount": jittered}


def _abstract_to_raw(action_idx: int, gs: dict) -> dict:
    owed    = gs["amount_owed"]
    cur_bet = gs["current_bet"]
    min_r   = gs["min_raise_to"]
    stack   = gs["your_stack"]
    my_bet  = gs["your_bet_this_street"]
    all_tot = my_bet + stack
    eff_pot = gs["pot"] + owed

    if action_idx == _FOLD:
        return {"action": "fold"}
    if action_idx == _CHECK_CALL:
        return {"action": "check" if owed == 0 else "call"}
    if action_idx == _ALL_IN:
        return {"action": "all_in"}
    for a_idx, rb_fn in _RAISE_LADDER:
        if action_idx == a_idx:
            target = cur_bet + max(rb_fn(eff_pot), min_r - cur_bet)
            return _jitter_raise(target, min_r, all_tot)
    return {"action": "fold"}


# ── Shared decision helpers ────────────────────────────────────────────────


def _legal_index(legal: list, action: int):
    """Position of `action` within the legal list, or None if it isn't legal."""
    return next((i for i, a in enumerate(legal) if a == action), None)


def _mask_probs(full_probs: np.ndarray, legal: list) -> np.ndarray:
    """Slice full_probs to legal indices, clip negatives, normalize."""
    arr = np.maximum(
        np.array([full_probs[a] for a in legal], dtype=np.float64), 0.0
    )
    s = arr.sum()
    if s < 1e-12:
        arr[:] = 1.0 / len(arr)
    else:
        arr /= s
    return arr


def _realtime_search(gs: dict, legal: list, gto_probs: np.ndarray,
                     equity: float | None = None,
                     exploit: tuple = (0.0, 0.0)) -> int:
    """
    GTO network with a targeted MC equity correction plus a read-derived exploit
    bias (Module D1).

    Raise *sizing* is still owned by the GTO network. We only adjust action-class
    probabilities:

    - Facing a bet (owed > 0): shift between FOLD and CHECK_CALL by the MC equity
      edge over pot odds, PLUS `callfold_shift` (>0 = call lighter vs over-bluffers
      like maniacs; <0 = fold more vs value-heavy nits/passive lines).
    - Able to bet (owed == 0): shift mass from CHECK_CALL into the bet actions by
      `bet_shift` (>0) to c-bet/bluff harder vs an over-folding field, preserving
      the network's bet-size preference.

    `exploit = (callfold_shift, bet_shift)` is read-gated upstream, so it is (0, 0)
    with no confident read and the network's policy is used unchanged.
    """
    gto_arr = _mask_probs(gto_probs, legal)
    callfold_shift, bet_shift = exploit

    owed = gs["amount_owed"]
    if owed <= 0:
        blended = gto_arr.copy()
        call_idx = _legal_index(legal, _CHECK_CALL)
        bet_idxs = [i for i, a in enumerate(legal) if a not in (_FOLD, _CHECK_CALL)]
        if bet_shift > 0.0 and call_idx is not None and bet_idxs:
            move = min(bet_shift, float(blended[call_idx]))
            blended[call_idx] -= move
            bet_mass = float(sum(blended[i] for i in bet_idxs))
            if bet_mass > 1e-9:
                for i in bet_idxs:           # keep the net's size distribution
                    blended[i] += move * blended[i] / bet_mass
            else:                            # net gives ~0 to bets → use half-pot
                half = next((i for i, a in enumerate(legal) if a == _HALF), bet_idxs[0])
                blended[half] += move
            blended /= blended.sum()
        return legal[int(np.random.choice(len(legal), p=blended))]

    pot = gs["pot"]
    if equity is None:
        equity = _run_mc(gs, max_iters=2_000)

    pot_odds     = owed / (pot + owed)          # minimum equity to break even on a call
    equity_edge  = equity - pot_odds            # + = call is profitable; − = fold preferred

    fold_idx = _legal_index(legal, _FOLD)
    call_idx = _legal_index(legal, _CHECK_CALL)

    if fold_idx is None or call_idx is None:
        return legal[int(np.random.choice(len(legal), p=gto_arr))]

    # Equity edge (±20 pp) + read-derived exploit, capped at ±35 pp total.
    shift   = float(np.clip(equity_edge * 0.8, -0.20, 0.20)) + callfold_shift
    shift   = float(np.clip(shift, -0.35, 0.35))
    blended = gto_arr.copy()
    blended[fold_idx] = max(blended[fold_idx] - shift, 0.0)
    blended[call_idx] = max(blended[call_idx] + shift, 0.0)
    blended /= blended.sum()
    return legal[int(np.random.choice(len(legal), p=blended))]


def _legal_actions(gs: dict, n_raises: int) -> list:
    """Return list of abstract action indices legal in the current state."""
    owed    = gs["amount_owed"]
    stack   = gs["your_stack"]
    cur     = gs["current_bet"]
    min_r   = gs["min_raise_to"]
    my_bet  = gs.get("your_bet_this_street", 0)
    all_tot = my_bet + stack
    eff_pot = gs["pot"] + owed
    min_rb  = min_r - cur

    legal = [_CHECK_CALL]
    if owed > 0:
        legal.append(_FOLD)
    if stack > 0:
        if n_raises < _MAX_RAISES_PER_STREET:
            last_tgt = -1
            for a_idx, rb_fn in _RAISE_LADDER:
                rb  = max(rb_fn(eff_pot), min_rb)
                tgt = cur + rb
                if tgt < all_tot and tgt != last_tgt:
                    last_tgt = tgt
                    legal.append(a_idx)
        legal.append(_ALL_IN)
    return legal


def _preflop_table_decide(gs: dict) -> dict | None:
    """
    Attempt a preflop decision from the tabular CFR table.
    Returns None when the table is unavailable or the live config is out of scope.
    """
    if _PREFLOP_TABLE is None:
        return None
    if not _preflop_applicable(gs):
        return None

    key = _preflop_infoset_key(gs)
    if key not in _PREFLOP_TABLE:
        return None

    probs = _PREFLOP_TABLE[key]   # float32[9]

    n_raises = _derive_n_raises_this_street(gs["action_log"], len(gs["players"]))
    legal = _legal_actions(gs, n_raises)

    arr = _mask_probs(probs, legal)
    action_idx = legal[int(np.random.choice(len(legal), p=arr))]
    return _abstract_to_raw(action_idx, gs)


# ── Heuristic preflop engine (replaces the broken CFR table) ───────────────
# Hand strength via the Chen formula; position- and action-aware thresholds.
# Tuned to open ~15% UTG widening to ~45% on the button, 3-bet a premium core,
# flat speculative hands at a price, and fold the rest.

_RANK_ORDER = {r: i for i, r in enumerate(RANKS)}   # '2'->0 .. 'A'->12
_CHEN_BASE  = {"A": 10.0, "K": 8.0, "Q": 7.0, "J": 6.0, "T": 5.0, "9": 4.5,
               "8": 4.0, "7": 3.5, "6": 3.0, "5": 2.5, "4": 2.0, "3": 1.5, "2": 1.0}
_PREMIUM_3BET = {"AA", "KK", "QQ", "AKs", "AKo"}   # always 3-bet / 4-bet-jam core

# Open-raise Chen thresholds by position; lower = wider.
# Used only as a fallback (BB iso-raises, profile-driven widening) when a
# position has no explicit chart range below.
_OPEN_THRESH = {"UTG": 8.5, "MP": 7.5, "CO": 6.5, "BTN": 5.0, "SB": 6.0, "BB": 8.0}


def _expand_range(tokens) -> frozenset:
    """Expand poker range notation into a frozenset of canonical hand labels.

    Labels match _preflop_context: pair "AA", suited "AKs", offsuit "AKo",
    higher rank first.  Supported tokens (standard chart notation):
        "AA"     exact pair
        "TT+"    that pair and every higher pair (TT,JJ,QQ,KK,AA)
        "AKs"    exact suited combo            "AKo"   exact offsuit combo
        "ATs+"   fixed high card, lower card from the named rank up to one
                 below the high card, suited   (ATs,AJs,AQs,AKs)
        "AQo+"   same, offsuit                 (AQo,AKo)
    """
    out: set[str] = set()
    for tok in tokens:
        t = tok.strip()
        plus = t.endswith("+")
        if plus:
            t = t[:-1]
        suit = ""
        if t and t[-1] in "so":
            suit, t = t[-1], t[:-1]
        a, b = t[0].upper(), t[1].upper()
        ia, ib = _RANK_ORDER[a], _RANK_ORDER[b]
        if ia < ib:                                   # normalise: high card first
            ia, ib = ib, ia
        if ia == ib:                                  # pair
            top = 12 if plus else ia                  # 'A' index == 12
            for r in range(ia, top + 1):
                out.add(RANKS[r] * 2)
        else:                                         # non-pair
            top = ia - 1 if plus else ib              # vary lower card up to a-1
            for r in range(ib, top + 1):
                out.add(f"{RANKS[ia]}{RANKS[r]}{suit}")
    return frozenset(out)


# Chart-derived 6-max 100bb GTO ranges (open frequencies ~16% UTG → ~48% BTN),
# approximated from standard free solver charts (RangeConverter / PokerCoaching).
# These cover the high-frequency root nodes; deeper/short-stack spots fall back
# to the Chen-formula logic below.
_OPEN_RANGE = {
    "UTG": _expand_range([
        "22+",
        "A2s+", "K9s+", "Q9s+", "J9s+", "T8s+", "97s+", "86s+", "75s+", "65s", "54s",
        "ATo+", "KJo+", "QJo",
    ]),
    "MP": _expand_range([
        "22+",
        "A2s+", "K8s+", "Q9s+", "J8s+", "T8s+", "97s+", "86s+", "75s+", "64s+", "54s",
        "A9o+", "KTo+", "QTo+", "JTo",
    ]),
    "CO": _expand_range([
        "22+",
        "A2s+", "K6s+", "Q8s+", "J8s+", "T7s+", "96s+", "86s+", "75s+", "64s+", "54s", "53s",
        "A8o+", "KTo+", "QTo+", "JTo", "T9o",
    ]),
    "BTN": _expand_range([
        "22+",
        "A2s+", "K2s+", "Q4s+", "J6s+", "T6s+", "96s+", "85s+", "74s+", "63s+", "53s+", "43s",
        "A2o+", "K8o+", "Q9o+", "J9o+", "T8o+", "98o", "87o",
    ]),
    "SB": _expand_range([                              # raise-first-in (limp handled separately)
        "22+",
        "A2s+", "K5s+", "Q6s+", "J7s+", "T7s+", "96s+", "85s+", "74s+", "64s+", "53s+",
        "A7o+", "K9o+", "Q9o+", "J9o+", "T9o", "98o",
    ]),
}

# 3-bet ranges when facing a single open (value core + position-appropriate
# bluffs).  Premium hands in _PREMIUM_3BET always re-raise on top of these.
_THREEBET_RANGE = {
    "UTG": _expand_range(["QQ+", "AKs", "AKo", "AQs", "A5s", "A4s"]),
    "MP":  _expand_range(["JJ+", "AQs+", "AKo", "AJs", "KQs", "A5s", "A4s"]),
    "CO":  _expand_range(["TT+", "AJs+", "AQo+", "KQs", "KJs", "A5s", "A4s", "A3s", "76s"]),
    "BTN": _expand_range(["99+", "ATs+", "AQo+", "KTs+", "QJs", "JTs",
                          "A5s", "A4s", "A3s", "A2s", "76s", "65s"]),
    "SB":  _expand_range(["TT+", "AJs+", "AKo", "AQo", "KQs", "KJs",
                          "A5s", "A4s", "A3s", "76s", "65s"]),
    "BB":  _expand_range(["99+", "AJs+", "AQo+", "KQs", "KJs", "QJs",
                          "A5s", "A4s", "A3s", "A2s", "76s", "65s", "54s"]),
}


def _chen_score(hi: str, lo: str, suited: bool) -> float:
    """Chen preflop hand-strength score (≈ -1.5 for 72o, 20 for AA)."""
    if hi == lo:                                    # pocket pair
        return max(_CHEN_BASE[hi] * 2.0, 5.0)
    score = _CHEN_BASE[hi]
    if suited:
        score += 2.0
    gap = _RANK_ORDER[hi] - _RANK_ORDER[lo] - 1     # ranks strictly between
    score -= {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}.get(gap, 5.0)
    if gap <= 1 and _RANK_ORDER[hi] < _RANK_ORDER["Q"]:
        score += 1.0                                # straight-y bonus
    return score


def _pos_category(off: int, n: int) -> str:
    """Map seat offset from dealer (0=BTN,1=SB,2=BB,3=UTG,...) to a category."""
    if n <= 2:
        return "BTN" if off == 0 else "BB"
    if off == 0:
        return "BTN"
    if off == 1:
        return "SB"
    if off == 2:
        return "BB"
    if off == n - 1:
        return "CO"
    if off == 3:
        return "UTG"
    return "MP"


def _preflop_context(gs: dict) -> dict:
    """Derive hand + position + action features for the preflop decision."""
    c1, c2 = gs["your_cards"][0], gs["your_cards"][1]
    r1, r2 = c1[0], c2[0]
    suited = c1[1] == c2[1]
    if _RANK_ORDER[r1] >= _RANK_ORDER[r2]:
        hi, lo = r1, r2
    else:
        hi, lo = r2, r1
    label = f"{hi}{lo}" + ("s" if suited and hi != lo else "" if hi == lo else "o")

    al     = gs["action_log"]
    off, n = _hero_position(gs)

    n_raises = sum(1 for e in al if e.get("action") in ("raise", "all_in"))
    limpers  = sum(1 for e in al if e.get("action") == "call") if n_raises == 0 else 0

    eff_bb = min(gs["your_stack"], _max_opponent_stack(gs)) / _BIG_BLIND
    return {
        "hi": hi, "lo": lo, "suited": suited, "pair": hi == lo, "label": label,
        "chen": _chen_score(hi, lo, suited),
        "pos": _pos_category(off, n), "n": n,
        "n_raises": n_raises, "limpers": limpers, "eff_bb": eff_bb,
    }


def _preflop_callers(gs: dict) -> int:
    """Count cold-calls of a raise so far this (preflop) street.

    A caller between us and a re-raiser means a multiway pot with a stronger
    combined range, so we should commit a deep stack more cautiously (see
    _preflop_decide's facing-a-re-raise logic).
    """
    seen_raise = False
    callers = 0
    for e in gs.get("action_log", []):
        act = e.get("action")
        if act in ("raise", "all_in"):
            seen_raise = True
        elif act == "call" and seen_raise:
            callers += 1
    return callers


def _max_opponent_stack(gs: dict) -> int:
    seat = gs["seat_to_act"]
    m = 0
    for p in gs["players"]:
        if p["seat"] == seat or p.get("is_folded"):
            continue
        m = max(m, p.get("stack", 0))
    return m or gs["your_stack"]


def _raise_to_amount(gs: dict, to_amount: int) -> dict:
    """Build a raise to `to_amount` total chips, clamped to legal range."""
    min_r   = gs["min_raise_to"]
    all_tot = gs["your_bet_this_street"] + gs["your_stack"]
    to_amount = max(int(to_amount), min_r)
    if to_amount >= all_tot:
        return {"action": "all_in"}
    return {"action": "raise", "amount": to_amount}


def _preflop_decide(gs: dict, profile_counts: tuple = (0, 0, 0)) -> dict:
    """Position-and-strength-aware preflop policy. Self-contained, no model."""
    ctx   = _preflop_context(gs)
    chen  = ctx["chen"]
    pos   = ctx["pos"]
    label = ctx["label"]
    owed  = gs["amount_owed"]
    pot   = max(1, gs["pot"])
    can_check = gs.get("can_check", owed == 0)
    eff_bb = ctx["eff_bb"]
    maniac, station, nit = profile_counts

    # Profile nudges: steal wider vs nits, tighten vs maniacs (they 3-bet light).
    open_adj = -0.6 * nit + 0.6 * maniac
    open_thr = _OPEN_THRESH[pos] + open_adj

    # ── Short stack: jam-or-fold (≤ 14 bb effective) ───────────────────────
    if eff_bb <= 14.0:
        jam_thr = 7.5 if ctx["n_raises"] == 0 else 9.5
        if label in _PREMIUM_3BET or chen >= jam_thr:
            return {"action": "all_in"}
        if can_check:
            return {"action": "check"}
        if owed <= _BIG_BLIND and chen >= 6.0:
            return {"action": "call"}
        return {"action": "fold"}

    # ── Unopened pot (no prior raise) ──────────────────────────────────────
    if ctx["n_raises"] == 0:
        if pos in _OPEN_RANGE:
            should_open = label in _OPEN_RANGE[pos]
            # vs passive nits, steal a touch wider; vs maniacs hold to the chart.
            if not should_open and nit > 0 and maniac == 0:
                should_open = chen >= (open_thr - 1.0)
        else:
            should_open = chen >= open_thr      # BB iso-raise / fallback
        if should_open:
            to = 3 * _BIG_BLIND + ctx["limpers"] * _BIG_BLIND      # ~3bb + 1bb/limper
            return _raise_to_amount(gs, to)
        if can_check:
            return {"action": "check"}   # BB checks its option
        # Limp only from late position (CO/BTN) or SB completing.
        # UTG/MP never limp — bad OOP, easily re-raised off equity.
        late = pos in ("SB", "CO", "BTN")
        spec = late and (ctx["pair"] or (ctx["suited"] and chen >= 5.5))
        if spec and owed <= _BIG_BLIND:
            return {"action": "call"}
        return {"action": "fold"}

    # ── Facing a raise ─────────────────────────────────────────────────────
    facing_3bet = ctx["n_raises"] >= 2
    cur = gs["current_bet"]

    # Chart-driven re-raise: premium core always; otherwise the position's
    # 3-bet range (only when facing a single open — vs a 3-bet we 4-bet the
    # premium core only).
    threebet_set = _THREEBET_RANGE.get(pos, _PREMIUM_3BET)
    want_reraise = label in _PREMIUM_3BET or (not facing_3bet and label in threebet_set)

    if want_reraise:
        # Shallow (≤ 35 bb eff): we're committed — jam the premium/3-bet core.
        if eff_bb <= 35:
            return {"action": "all_in"}

        # Deep, facing a single open: 3-bet (value + chart bluffs) to a size.
        if not facing_3bet:
            return _raise_to_amount(gs, cur * 3)

        # Deep, facing a RE-raise of our line (only premiums reach here). Blanket-
        # jamming 100 bb is the biggest leak: value-heavy 4-bet ranges crush QQ/AK,
        # and a cold-caller in between makes it worse. Commit by hand strength.
        callers   = _preflop_callers(gs)
        four_bet  = ctx["n_raises"] >= 3        # we 3-bet and got 4-bet (or more)

        if four_bet:
            # vs a 4-bet at depth only the very top stacks off.
            if label in ("AA", "KK"):
                return {"action": "all_in"}
            return {"action": "fold"}           # QQ/AK dominated by 4-bet value ranges

        # Facing a single 3-bet while deep.
        if label in ("AA", "KK"):
            return _raise_to_amount(gs, round(cur * 2.3))   # 4-bet value, non-committing
        if label == "QQ" and callers >= 1:
            return {"action": "fold"}           # multiway 3-bet pot: QQ is dominated
        # QQ / AKs / AKo: flat to realize equity rather than stack off 100 bb.
        if owed <= pot * 0.90:
            return {"action": "call"}
        return {"action": "fold"}

    if not facing_3bet:
        # Flat strong hands and set-mine speculative hands at the right price.
        call_thr = 9.5 if pos in ("CO", "BTN", "BB") else 10.5
        if chen >= call_thr and owed <= pot * 0.60:
            return {"action": "call"}
        set_mine = ctx["pair"] or (ctx["suited"] and chen >= 7.0)
        if set_mine and owed <= pot * 0.33 and eff_bb >= 25:
            return {"action": "call"}

    if can_check:
        return {"action": "check"}
    return {"action": "fold"}


# ── Postflop engine (net or MC) + risk gate ────────────────────────────────


_EXPLOIT_CONF = _envf("VLAD_EXPLOIT_CONF", 0.35)      # min read conf for D1 bias
_EXPLOIT_CALL_MAX = _envf("VLAD_EXPLOIT_CALL_MAX", 0.15)  # max fold/call shift
_EXPLOIT_BET_MAX = _envf("VLAD_EXPLOIT_BET_MAX", 0.15)    # max check→bet shift


def _exploit_bias(gs: dict, profiles: dict) -> tuple:
    """Read-derived (callfold_shift, bet_shift) for _realtime_search (Module D1).

    Returns (0, 0) without a confident read so the GTO policy is used unchanged.
    """
    owed = gs["amount_owed"]
    if owed > 0:
        # Facing a bet: lean to call vs over-bluffers, fold vs value-heavy lines.
        tag, conf = _last_aggressor_read(gs, profiles)
        if conf < _EXPLOIT_CONF:
            return 0.0, 0.0
        if tag == "maniac":
            return _EXPLOIT_CALL_MAX * conf, 0.0
        if tag in ("nit", "station"):       # nits/stations rarely bluff-raise
            return -_EXPLOIT_CALL_MAX * conf, 0.0
        return 0.0, 0.0

    # Able to bet: c-bet/bluff harder when the live field over-folds (nit-like).
    best = 0.0
    for p in gs["players"]:
        if p["seat"] == gs["seat_to_act"] or p.get("state") not in ("active", "all_in"):
            continue
        tag, conf = _classify_opponent(profiles.get(p.get("bot_id")) or {})
        if conf >= _EXPLOIT_CONF and tag == "nit":
            best = max(best, conf)
    return 0.0, _EXPLOIT_BET_MAX * best


def _net_postflop(gs: dict, equity: float | None, profiles: dict | None = None) -> dict:
    """GTO strategy net with the precomputed-equity fold/call correction and the
    read-derived exploit bias (Module D1)."""
    vec   = _build_feature_vector(gs)
    probs = _numpy_forward(_GTO_LAYERS, vec)  # type: ignore[arg-type]

    stack    = gs["your_stack"]
    n_raises = _derive_n_raises_this_street(gs["action_log"], len(gs["players"]))
    legal    = _legal_actions(gs, n_raises)

    # Dampen ALL_IN and large overbets in raise wars when stack is still deep.
    # The training cap (4 raises/street) forces {FOLD,CALL,ALL_IN} at n_raises=4,
    # making the network assign inflated ALL_IN mass in re-raise spots.  Suppress
    # it proportionally when there's still meaningful stack to play (SPR > 2).
    if n_raises >= 2 and stack > 0:
        spr = stack / max(gs["pot"], 1)
        if spr > 2.0:
            scale = min((spr - 2.0) / 4.0, 1.0)   # ramps 0→1 from SPR 2 to 6
            dampen = 1.0 - 0.75 * scale             # 1.0 at SPR=2, 0.25 at SPR≥6
            for _a in (_ALL_IN, _2X, _1_72X):
                probs[_a] = probs[_a] * dampen

    exploit = _exploit_bias(gs, profiles) if profiles else (0.0, 0.0)
    action_idx = _realtime_search(gs, legal, probs, equity=equity, exploit=exploit)
    return _abstract_to_raw(action_idx, gs)


# Offensive flop c-bet is OFF by default: it's the highest-variance, least-
# validated change (head-to-head tournaments were inconclusive: +5k seed 1,
# −37k seed 2). Enable with VLAD_CBET_BLUFF=1 for multi-seed validation before
# shipping it on. The rest of the postflop improvements remain active.
_CBET_BLUFF_ENABLED = _envf("VLAD_CBET_BLUFF", 0.0) >= 0.5


def _has_initiative(gs: dict) -> bool:
    """True if hero was the last preflop aggressor (took the betting lead)."""
    log = gs.get("action_log", [])
    if not log:
        return False
    annot = _reconstruct_streets(log)
    last_pf = None
    for entry, a in zip(log, annot):
        if a["street"] == "preflop" and \
                str(entry.get("action", "")).lower() in ("raise", "all_in"):
            last_pf = entry.get("seat")
    return last_pf == gs["seat_to_act"]


def _should_cbet_bluff(gs: dict, equity: float, profile_counts: tuple,
                       texture: str, n_active: int) -> bool:
    """Disciplined flop c-bet/bluff when we can check (Module D1, MC path).

    Mirrors cfr_equity_v28 (#4): as the preflop aggressor, HU/3-way, on a non-wet
    board, with some backup equity, when the field is not a station/maniac (they
    call). Fixes vlad's over-passive 'check when equity < call_thr' leak. The RNG
    is seeded per action in decide(), so this is replay-deterministic.
    """
    if not _CBET_BLUFF_ENABLED:
        return False
    maniac, station, nit = profile_counts
    if gs.get("street") != "flop":
        return False
    if n_active > 2 or equity < 0.28:
        return False
    if station > 0 or maniac > 0:            # they don't fold — don't bluff
        return False
    if texture == "wet":                     # opponents hold draws on wet boards
        return False
    if not _has_initiative(gs):
        return False
    return random.random() < 0.55


def _mc_postflop(gs: dict, equity: float, profile_counts: tuple) -> dict:
    """Monte-Carlo equity + pot-odds engine with pot-fraction sizing."""
    maniac, station, nit = profile_counts
    owed    = gs["amount_owed"]
    pot     = max(1, gs["pot"])
    stack   = gs["your_stack"]
    cur     = gs["current_bet"]
    min_r   = gs["min_raise_to"]
    texture = _board_texture(gs.get("community_cards", []))
    n_active = sum(1 for p in gs["players"]
                   if p.get("state") == "active" and p["seat"] != gs["seat_to_act"])

    # Multiway + commitment-scaled call threshold (Module B3, equity realization:
    # equity realizes worse multiway and when a big bet commits us — cf.
    # cfr_equity_v28's `required += (n_opps-1)*0.015` and stack-fraction bumps).
    pot_odds  = owed / (pot + owed) if owed > 0 else 0.0
    multi_adj = min(0.08 + 0.06 * max(0, n_active - 1), 0.26)
    commit_adj = 0.0
    if owed >= stack * 0.40:
        commit_adj += 0.04
    if owed >= stack * 0.75:
        commit_adj += 0.06
    call_thr  = pot_odds + multi_adj + commit_adj - 0.03 * maniac + 0.04 * nit

    if equity < call_thr:
        # Too weak to call/value-bet — but with initiative on a foldable board,
        # c-bet for fold equity rather than meekly checking (Module D1, MC path).
        if owed == 0 and _should_cbet_bluff(gs, equity, profile_counts,
                                            texture, n_active):
            return _raise_to_amount(gs, round(pot * 0.5))
        return {"action": "check"} if owed == 0 else {"action": "fold"}

    # Decide whether to raise/bet vs call/check.
    equity_edge = equity - call_thr
    should_bet  = equity_edge > 0.18 or (owed == 0 and equity > 0.62)

    if not should_bet:
        return {"action": "check"} if owed == 0 else {"action": "call"}

    # Pot-fraction sizing (correct postflop bet sizing, not min-raise).
    if equity > 0.85:
        bet_frac = 0.90
    elif equity > 0.72:
        bet_frac = 0.70
    elif equity > 0.60:
        bet_frac = 0.50
    else:
        bet_frac = 0.35

    if texture == "wet":
        bet_frac += 0.15          # protect / deny equity on draws
    elif texture == "dry" and owed == 0:
        bet_frac -= 0.10          # small bets work on dry boards
    # D2 inelastic value sizing: stations pay off oversized value bets (they call
    # too wide), so bet bigger — cfr_equity_v28 uses +0.25 and ranks #4.
    if station > 0 and maniac == 0:
        bet_frac += 0.25
    if maniac > 0 and owed > 0:
        bet_frac += 0.15          # re-raise big vs maniacs

    return _raise_to_amount(gs, cur + max(round(pot * bet_frac), min_r - cur))


def _board_features(board: list) -> dict:
    """Full-board texture analysis (Module E). Considers every dealt card, not
    just the flop, so turn/river runouts (completed flushes, four-to-straight,
    pairing of the board) are visible to sizing / risk / anti-punt logic.

    Returns structured flags consumed by range-conditioned equity (B), the
    anti-punt layer (C), and exploit sizing (D), plus the coarse wetness used by
    the existing MC sizing and risk gate.
    """
    n = len(board)
    base = {
        "n": n, "paired": False, "two_pair": False, "trips": False,
        "two_tone": False, "flush_possible": False, "four_flush": False,
        "monotone": False, "max_suit": 0,
        "connected": False, "straight_possible": False, "four_straight": False,
        "straight_cards": 0,
    }
    if n < 3:
        return base

    ranks = [_RANK_ORDER[c[0]] for c in board]
    suits = [c[1] for c in board]

    rank_counts: dict = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    pairs = sum(1 for v in rank_counts.values() if v >= 2)
    suit_counts: dict = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_suit = max(suit_counts.values())

    # Straightness: most board cards inside any 5-rank window (Ace counts high
    # and low). 3+ => a straight is makeable with two hole cards; 4 => one card
    # completes it (four-to-straight); span<=4 of distinct ranks => connected.
    rank_set = set(ranks)
    if 12 in rank_set:               # Ace also low (wheel)
        rank_set = rank_set | {-1}
    best = 0
    for lo in range(-1, 9):          # window low edge from A-low(-1) up to T
        cnt = sum(1 for r in rank_set if lo <= r <= lo + 4)
        best = max(best, cnt)

    distinct = sorted(set(ranks))
    span = distinct[-1] - distinct[0]

    base.update({
        "paired": pairs >= 1 or any(v >= 2 for v in rank_counts.values()),
        "two_pair": pairs >= 2,
        "trips": any(v >= 3 for v in rank_counts.values()),
        "max_suit": max_suit,
        "two_tone": max_suit >= 2,
        "flush_possible": max_suit >= 3,
        "four_flush": max_suit >= 4,
        "monotone": max_suit == n,
        "straight_cards": best,
        "straight_possible": best >= 3,
        "four_straight": best >= 4,
        "connected": len(distinct) >= 3 and span <= 4,
    })
    return base


def _board_texture(board: list) -> str:
    """Coarse texture category: none / dry / semi / wet / paired.

    Flop (3 cards) classification is preserved byte-for-byte from the original
    implementation; turn/river now read the full board (Module E) so a paired
    turn, completed flush, or four-to-straight is reflected.
    """
    n = len(board)
    if n < 3:
        return "none"

    f = _board_features(board)

    if n == 3:
        # Original flop semantics, unchanged.
        ranks = [_RANK_ORDER[c[0]] for c in board]
        suits = [c[1] for c in board]
        if len(set(ranks)) < 3:
            return "paired"
        two_tone = max(suits.count(s) for s in set(suits)) >= 2
        connected = max(ranks) - min(ranks) <= 4
        if two_tone and connected:
            return "wet"
        if two_tone or connected:
            return "semi"
        return "dry"

    # Turn / river: full-board aware.
    if f["paired"]:
        return "paired"
    wet = f["flush_possible"] or f["four_straight"]
    semi = f["two_tone"] or f["straight_possible"] or f["connected"]
    if wet:
        return "wet"
    if semi:
        return "semi"
    return "dry"


def _risk_gate(action: dict, gs: dict, equity: float) -> dict:
    """Block large stack commitments that equity doesn't justify.

    Counters pressure bots (e.g. anti_monte_carlo): we never stack off a big
    fraction of the effective stack unless equity clears a street- and
    commitment-scaled floor.  Cheap/medium actions pass through untouched.
    """
    act = action.get("action")
    if act in ("fold", "check"):
        return action

    stack   = gs["your_stack"]
    my_bet  = gs["your_bet_this_street"]
    all_tot = my_bet + stack
    owed    = gs["amount_owed"]
    if act == "all_in":
        commit = stack
    elif act == "call":
        commit = min(owed, stack)
    else:  # raise
        commit = min(action.get("amount", 0) - my_bet, stack)

    frac = commit / max(all_tot, 1)
    if frac < _RISK_COMMIT_FRAC:
        return action

    street = gs.get("street", "flop")
    base   = {"flop": 0.50, "turn": 0.55, "river": 0.60}.get(street, 0.52)
    if _board_texture(gs.get("community_cards", [])) == "wet":
        base += 0.03
    floor = base + 0.18 * (frac - _RISK_COMMIT_FRAC)
    if equity >= floor:
        return action

    # Equity doesn't justify a big commitment. Fall back to the cheapest sane line.
    if owed == 0:
        return {"action": "check"}
    pot      = gs["pot"]
    pot_odds = owed / (pot + owed)
    if equity >= pot_odds and act in ("raise", "all_in"):
        return {"action": "call"}      # don't bloat the pot, but the call prices in
    if act == "call" and equity >= pot_odds:
        return action
    return {"action": "fold"}


# ── Anti-punt override layer (Module C) ─────────────────────────────────────
#
# Final guardrail after the engine + risk gate: catches the specific recurring
# leaks that bleed chips, but ONLY when we have a confident opponent read, so it
# never fires vs unknown / strong bots (and so leaves the golden behavior intact
# until reads accumulate). Reverts the offending aggressive/thin line to the
# cheapest sane alternative.

_HAND_TIER = {
    "High Card": "weak", "Pair": "pair", "Two Pair": "medium",
    "Trips": "strong", "Three of a Kind": "strong", "Straight": "strong",
    "Flush": "strong", "Full House": "very_strong", "Quads": "very_strong",
    "Four of a Kind": "very_strong", "Straight Flush": "very_strong",
}
_ANTI_PUNT_CONF = _envf("VLAD_ANTIPUNT_CONF", 0.35)   # min read conf to fire


def _made_hand_tier(gs: dict) -> str:
    """Hero's current made-hand tier: weak / pair / medium / strong / very_strong."""
    board = gs.get("community_cards", [])
    if len(board) < 3:
        return "weak"
    score = evaluate([Card(c) for c in gs["your_cards"]] + [Card(c) for c in board])
    return _HAND_TIER.get(str(handtype(score)), "weak")


def _last_aggressor_read(gs: dict, profiles: dict) -> tuple:
    """(tag, conf) for the player we're facing a bet from. When owed > 0 the most
    recent raise/all-in in the hand is, by definition, the live aggressor (bets
    reset each street), so the last aggressive entry suffices — no street
    reconstruction needed."""
    seats_by = {p["seat"]: p.get("bot_id") for p in gs["players"]}
    last_seat = None
    for entry in gs.get("action_log", []):
        if str(entry.get("action", "")).lower() in ("raise", "all_in"):
            last_seat = entry.get("seat")
    if last_seat is None or last_seat == gs["seat_to_act"]:
        return "unknown", 0.0
    return _classify_opponent(profiles.get(seats_by.get(last_seat)) or {})


def _field_station_read(gs: dict, profiles: dict) -> tuple:
    """(tag, conf) of the most station-like active opponent — worst to bluff into."""
    best = ("unknown", 0.0)
    for p in gs["players"]:
        if p["seat"] == gs["seat_to_act"] or p.get("state") not in ("active", "all_in"):
            continue
        tag, conf = _classify_opponent(profiles.get(p.get("bot_id")) or {})
        if tag == "station" and conf > best[1]:
            best = (tag, conf)
    return best


def _n_active_opponents(gs: dict) -> int:
    return sum(1 for p in gs["players"]
               if p["seat"] != gs["seat_to_act"] and p.get("state") in ("active", "all_in"))


def _anti_punt(action: dict, gs: dict, equity: float, profiles: dict) -> dict:
    """Revert known-bad aggressive/thin lines when a confident read says so."""
    act = action.get("action")
    if act in ("fold", "check"):
        return action

    street = gs.get("street", "")
    owed = gs["amount_owed"]
    pot = max(1, gs["pot"])
    can_check = gs.get("can_check", owed == 0)
    aggressive = act in ("raise", "all_in")
    tier = _made_hand_tier(gs)

    def _passive():
        return {"action": "check"} if can_check else {"action": "call"}

    # 1. River air-bluff into a station (no draws exist on the river).
    if street == "river" and aggressive and tier in ("weak", "pair"):
        _, conf = _field_station_read(gs, profiles)
        if conf >= _ANTI_PUNT_CONF:
            return {"action": "check"} if can_check else {"action": "fold"}

    # 2. Oversized river bluff-catch vs a nit/passive line.
    if street == "river" and act == "call" and owed > 0.55 * pot \
            and tier in ("weak", "pair"):
        tag, conf = _last_aggressor_read(gs, profiles)
        if tag == "nit" and conf >= _ANTI_PUNT_CONF:
            return {"action": "fold"}

    # 3. Low-equity multiway c-bet/raise on a wet board (draws keep equity up,
    #    so the equity floor naturally spares semi-bluffs).
    if street in ("flop", "turn") and aggressive and _n_active_opponents(gs) >= 2 \
            and tier in ("weak", "pair") and equity < 0.45:
        if _board_texture(gs.get("community_cards", [])) == "wet":
            return _passive()

    return action


def _postflop_decide(gs: dict, profile_counts: tuple = (0, 0, 0),
                     profiles: dict | None = None) -> dict:
    """Route the postflop decision through the configured engine, risk gate, and
    the anti-punt override layer."""
    if profiles is None:
        my_bot_id = next((p["bot_id"] for p in gs["players"]
                          if p["seat"] == gs["seat_to_act"]), None)
        profiles = _build_opponent_profiles(gs.get("match_action_log", []), my_bot_id)

    equity = _run_mc(gs, max_iters=1_200, profiles=profiles)

    if _POSTFLOP_ENGINE == "mc" or _GTO_LAYERS is None:
        action = _mc_postflop(gs, equity, profile_counts)
    else:
        action = _net_postflop(gs, equity, profiles)

    action = _risk_gate(action, gs, equity)
    return _anti_punt(action, gs, equity, profiles)


# ── Range-conditioned equity (Module B) ────────────────────────────────────
#
# Uniform-random opponent hands overvalue marginal holdings vs tight ranges and
# undervalue them vs loose ones. We bias each opponent's sampled hole cards by a
# per-opponent strength floor (a starting-hand percentile), derived from their
# archetype tag + confidence. Cheap rejection sampling on a Chen-score percentile
# keeps the rollout fast.

def _build_hand_pctl():
    """Precompute starting-hand strength percentiles by Chen score, weighted by
    combo counts (pair=6, suited=4, offsuit=12). Returns a sorted list of the
    1326 combos' Chen scores for bisect-based percentile lookup."""
    scores = []
    for i, hi in enumerate(RANKS):
        for j in range(0, i + 1):
            lo = RANKS[j]
            if i == j:
                scores.extend([_chen_score(hi, lo, False)] * 6)
            else:
                scores.extend([_chen_score(hi, lo, True)] * 4)
                scores.extend([_chen_score(hi, lo, False)] * 12)
    scores.sort()
    return scores


_HAND_PCTL_SORTED = _build_hand_pctl()
_N_COMBOS = len(_HAND_PCTL_SORTED)   # 1326


def _hand_pctl(c1: str, c2: str) -> float:
    """Fraction of starting combos weaker than this hand (0..1), by Chen score."""
    r1, r2 = c1[0], c2[0]
    if _RANK_ORDER[r1] >= _RANK_ORDER[r2]:
        hi, lo = r1, r2
    else:
        hi, lo = r2, r1
    suited = c1[1] == c2[1]
    chen = _chen_score(hi, lo, suited)
    return _bisect_left(_HAND_PCTL_SORTED, chen) / _N_COMBOS


# Base continuing-range strength floors (percentile) by archetype. Higher = the
# opponent only continues with stronger hands, so we sample them tighter.
_RANGE_FLOOR_BASE = {
    "nit": _envf("VLAD_FLOOR_NIT", 0.78), "tag": 0.52, "normal": 0.28,
    "unknown": 0.0, "station": _envf("VLAD_FLOOR_STATION", 0.08), "maniac": 0.04,
}
_FLOOR_TURN  = _envf("VLAD_FLOOR_TURN", 0.05)   # street bump: opp still in on turn
_FLOOR_RIVER = _envf("VLAD_FLOOR_RIVER", 0.10)  # ...and on the river
# conf=0 anchor. Kept at 0 so an unknown opponent stays uniform on the flop
# (unbiased); the street bump below still reflects "still in the hand" postflop.
_RANGE_FLOOR_NEUTRAL = 0.0


def _seat_range_floor(tag: str, conf: float, street: str) -> float:
    """Per-opponent starting-hand percentile floor for MC rejection sampling.

    Blends the archetype base floor toward neutral by confidence (weak reads stay
    near uniform), then nudges up postflop: an opponent still in the hand on later
    streets holds a stronger range on average.
    """
    base = _RANGE_FLOOR_BASE.get(tag, _RANGE_FLOOR_NEUTRAL)
    floor = _RANGE_FLOOR_NEUTRAL + (base - _RANGE_FLOOR_NEUTRAL) * max(0.0, min(1.0, conf))
    floor += {"flop": 0.0, "turn": _FLOOR_TURN, "river": _FLOOR_RIVER}.get(street, 0.0)
    return max(0.0, min(0.92, floor))


def _hand_aggression_bump(gs: dict) -> float:
    """Range-floor bump from aggression shown THIS hand (Module B, action-
    conditioned — cfr_equity_v28's `opp_floor` idea). 3-bets+ and postflop barrels
    mean the live range is stronger than the archetype alone implies. Bounded.
    """
    log = gs.get("action_log", [])
    if not log:
        return 0.0
    annot = _reconstruct_streets(log)
    pf_raises = post_raises = 0
    for entry, a in zip(log, annot):
        if str(entry.get("action", "")).lower() not in ("raise", "all_in"):
            continue
        if a["street"] == "preflop":
            pf_raises += 1
        else:
            post_raises += 1
    # Open raise (1 preflop raise) is baseline; 3-bets+ signal strength.
    bump = 0.06 * max(0, pf_raises - 1) + 0.06 * post_raises
    pot = max(1, gs.get("pot", 1))
    owed = gs.get("amount_owed", 0)
    if owed >= 0.5 * pot:
        bump += 0.04
    return min(0.25, bump)


# ── Monte Carlo equity ──────────────────────────────────────────────────────

def monte_carlo_equity(
    hole_cards: list,
    board_cards: list,
    remaining_cards: list,
    num_opponents: int = 1,
    time_limit: float = 0.5,
    max_iters: int | None = None,
    opp_floors: list | None = None,
    deadline: float | None = None,
) -> float:
    """Monte-Carlo equity vs `num_opponents`. If `opp_floors` is given (one
    strength percentile per opponent), each opponent's hole cards are rejection-
    sampled to sit at/above that floor — range-conditioned equity (Module B).

    `deadline` (a time.time() value) is a hard wall-clock cap from the latency
    guard: once past it we stop early (keeping ≥ _MC_MIN_ITERS samples so the
    estimate stays usable) rather than risk the 2 s action timeout.
    """
    start = time.time()
    wins, iters = 0, 0
    need_board = 5 - len(board_cards)
    n_rem = len(remaining_cards)
    floors = opp_floors if opp_floors else None
    while (max_iters is None and time.time() - start < time_limit) or \
          (max_iters is not None and iters < max_iters):
        if deadline is not None and iters >= _MC_MIN_ITERS and time.time() > deadline:
            break
        random.shuffle(remaining_cards)
        if floors is None:
            opp_hands = [remaining_cards[i * 2: i * 2 + 2] for i in range(num_opponents)]
            idx = 2 * num_opponents
        else:
            opp_hands = []
            idx = 0
            for f in floors:
                hand = remaining_cards[idx:idx + 2]
                idx += 2
                tries = 0
                # Resample (bounded) until the hand clears the opponent's floor.
                while (f > 0.0 and tries < 3 and idx + 2 <= n_rem
                       and _hand_pctl(str(hand[0]), str(hand[1])) < f):
                    hand = remaining_cards[idx:idx + 2]
                    idx += 2
                    tries += 1
                opp_hands.append(hand)
        board = list(board_cards)
        board.extend(remaining_cards[idx:idx + need_board])
        my_score   = evaluate(hole_cards + board)
        opp_scores = [evaluate(oh + board) for oh in opp_hands]
        best       = max(my_score, *opp_scores)
        if my_score == best:
            n_tied  = opp_scores.count(best)
            wins   += 1 / (n_tied + 1)
        iters += 1
    return wins / max(iters, 1)


# ── Opponent modeling: street reconstruction (Module A1) ───────────────────
#
# Neither the cross-hand match_action_log nor the within-hand action_log carries
# a `street` field (see engine: only {hand_num,seat,bot_id,action,amount} /
# {seat,action,amount}). We reconstruct street boundaries by replaying the
# betting round structurally: a round closes when every live, non-all-in seat
# has acted and matched the last aggression. Crucially this is driven by action
# *type*, not amount — the cross-hand log records the bot's raw return (amounts
# are often None and may differ from what the engine executed), so amounts are
# unreliable, but the action sequence and ordering are exactly the engine's.
#
# Limitation: short all-ins that don't reopen action cannot be distinguished
# without chip accounting, so every raise/all_in is treated as reopening. This
# can over-extend a street in the rare short-all-in case; the 4-street clamp and
# per-hand reset bound the error, and aggregate opponent stats wash it out.

_STREET_NAMES = ("preflop", "flop", "turn", "river")
_BLIND_ACTIONS = frozenset(("small_blind", "big_blind"))
_AGGRESSIVE_ACTIONS = frozenset(("raise", "all_in", "bet"))
_PASSIVE_ACTIONS = frozenset(("check", "call"))


def _reconstruct_streets(actions: list, dealt_seats=None) -> list:
    """Annotate each action of a SINGLE hand with the street it occurred on.

    Works for both log formats:
      - within-hand action_log: includes leading small_blind/big_blind entries.
      - cross-hand match_action_log slice (one hand_num): no blind entries,
        bot-raw action names, amounts possibly missing.

    Args:
        actions: ordered list of dicts, each with at least "seat" and "action".
        dealt_seats: optional iterable of seats dealt into the hand. If omitted,
            inferred as every distinct seat appearing in `actions`.

    Returns a list aligned 1:1 with `actions`; each element is a dict
    ``{"street": str, "street_aggr_before": int}`` where street_aggr_before is
    the count of aggressive actions already taken on that street before this
    entry (blinds excluded). Blind entries are labeled preflop with 0.
    """
    if dealt_seats is None:
        live = {a.get("seat") for a in actions if a.get("seat") is not None}
    else:
        live = set(dealt_seats)

    folded: set = set()
    all_in: set = set()

    def _fresh_need() -> set:
        return set(live) - folded - all_in

    need = _fresh_need()
    street_idx = 0
    aggr_count = 0  # aggressive actions on the current street (blinds excluded)
    out = []

    for entry in actions:
        act = str(entry.get("action", "")).lower().strip()

        if act in _BLIND_ACTIONS:
            # Blinds are forced posts, not voluntary actions: they neither
            # consume a seat's obligation to act (BB keeps the option) nor count
            # as aggression. They do confirm the poster is in the hand.
            out.append({"street": "preflop", "street_aggr_before": 0})
            continue

        out.append({"street": _STREET_NAMES[street_idx],
                    "street_aggr_before": aggr_count})

        seat = entry.get("seat")
        need.discard(seat)

        if act == "fold":
            folded.add(seat)
        elif act == "all_in":
            all_in.add(seat)
            aggr_count += 1
            need = (live - folded - all_in) - {seat}
        elif act in _AGGRESSIVE_ACTIONS:  # raise / bet
            aggr_count += 1
            need = (live - folded - all_in) - {seat}
        # passive (check/call) or unknown → only consumes this seat's turn.

        if not need:
            # Betting round closed → subsequent entries belong to the next street.
            if street_idx < len(_STREET_NAMES) - 1:
                street_idx += 1
            aggr_count = 0
            need = _fresh_need()

    return out


# ── Opponent modeling: leak profiles + archetype (Modules A2 / A3) ──────────
#
# Built on _reconstruct_streets. We aggregate per-bot_id leak counters across
# the rolling cross-hand match_action_log (keyed on bot_id, NOT seat: seats are
# re-indexed to the alive set every hand). All stats here are position-
# independent and reconstructable from {hand_num,seat,bot_id,action,street,
# street_aggr_before} alone.
#
# Deferred (needs per-hand blind/position inference, which the cross-hand log
# does not carry): fold_to_steal, positional VPIP/PFR splits. Tracked in
# IMPROVEMENT_PLAN.md.
#
# Empirical-Bayes blend: each rate is shrunk toward a population prior with
# weight min(1, opportunities / target), so sparse reads stay near baseline and
# the confidence rises with sample size.

# (prior_mean, shrinkage_target) per leak. Priors are 6-max population defaults.
_LEAK_PRIORS = {
    "vpip":              (0.24, 12),
    "pfr":               (0.17, 12),
    "pf_reraise":        (0.07, 8),    # 3-bet-ish: reraise when facing a raise
    "fold_to_flop_cbet": (0.50, 8),
    "fold_to_turn":      (0.45, 6),
    "river_call":        (0.50, 6),
}
_AGGR_PRIOR = 1.4          # aggression factor (aggr actions / calls) prior
_MIN_HANDS_FOR_TAG = 8     # below this, opponent stays "unknown"
_PROFILE_ACT_CONF = 0.30   # min classification confidence before we exploit a read


def _fresh_opp_counters() -> dict:
    return {
        "hands": 0, "actions": 0,
        "vpip": 0, "pfr": 0,
        "pf_reraise_opp": 0, "pf_reraise": 0,
        "flop_cbet_faced": 0, "flop_cbet_fold": 0,
        "turn_faced": 0, "turn_fold": 0,
        "river_faced": 0, "river_call": 0,
        "n_aggr": 0, "n_call": 0,
        "saw_flop": 0, "saw_river": 0,
    }


def _iter_hands(match_log: list):
    """Yield (hand_num, [entries]) groups from a chronological match log.

    Entries for a hand are contiguous (the engine appends in play order), so we
    split on hand_num changes without sorting.
    """
    cur_id = _SENTINEL = object()
    bucket: list = []
    for entry in match_log:
        hid = entry.get("hand_num")
        if hid != cur_id and bucket:
            yield cur_id, bucket
            bucket = []
        cur_id = hid
        bucket.append(entry)
    if bucket:
        yield cur_id, bucket


def _build_opponent_profiles(match_log: list, my_bot_id=None) -> dict:
    """Aggregate per-bot_id leak counters from the rolling match_action_log.

    Returns ``{bot_id: counters}``. Pass ``my_bot_id`` to exclude hero.
    """
    profiles: dict = {}
    _AGGR = ("raise", "all_in")
    _VOL = ("call", "raise", "all_in")

    for _hand_num, entries in _iter_hands(match_log):
        annot = _reconstruct_streets(entries)
        seen_vol: dict = {}      # bot_id -> voluntarily entered preflop
        seen_raise: dict = {}    # bot_id -> raised preflop
        present: set = set()
        saw_flop: set = set()
        saw_river: set = set()

        for entry, a in zip(entries, annot):
            bid = entry.get("bot_id")
            if bid is None or bid == my_bot_id:
                continue
            act = str(entry.get("action", "")).lower().strip()
            if act in _BLIND_ACTIONS:
                continue
            street = a["street"]
            faced = a["street_aggr_before"] >= 1

            prof = profiles.setdefault(bid, _fresh_opp_counters())
            present.add(bid)
            prof["actions"] += 1

            if street == "preflop":
                if act in _VOL:
                    seen_vol[bid] = True
                if act in _AGGR:
                    seen_raise[bid] = True
                if faced:
                    prof["pf_reraise_opp"] += 1
                    if act in _AGGR:
                        prof["pf_reraise"] += 1
            else:
                saw_flop.add(bid)
                if street == "river":
                    saw_river.add(bid)
                if street == "flop" and faced and act in ("fold", "call") + _AGGR:
                    prof["flop_cbet_faced"] += 1
                    if act == "fold":
                        prof["flop_cbet_fold"] += 1
                elif street == "turn" and faced and act in ("fold", "call") + _AGGR:
                    prof["turn_faced"] += 1
                    if act == "fold":
                        prof["turn_fold"] += 1
                elif street == "river" and faced and act in ("fold", "call") + _AGGR:
                    prof["river_faced"] += 1
                    if act == "call":
                        prof["river_call"] += 1

            if act in _AGGR:
                prof["n_aggr"] += 1
            elif act == "call":
                prof["n_call"] += 1

        for bid in present:
            prof = profiles[bid]
            prof["hands"] += 1
            if seen_vol.get(bid):
                prof["vpip"] += 1
            if seen_raise.get(bid):
                prof["pfr"] += 1
            if bid in saw_flop:
                prof["saw_flop"] += 1
            if bid in saw_river:
                prof["saw_river"] += 1

    return profiles


def _shrunk_rate(succ: int, opp: int, key: str) -> tuple[float, float]:
    """Empirical-Bayes rate + confidence (0..1) for a leak counter."""
    prior, target = _LEAK_PRIORS[key]
    if opp <= 0:
        return prior, 0.0
    w = min(1.0, opp / target)
    return prior * (1.0 - w) + (succ / opp) * w, w


def _opp_leaks(counters: dict) -> dict:
    """Derive blended leak rates + confidences from raw counters."""
    hands = counters.get("hands", 0)
    vpip, vpip_c = _shrunk_rate(counters["vpip"], hands, "vpip")
    pfr, pfr_c = _shrunk_rate(counters["pfr"], hands, "pfr")
    rer, rer_c = _shrunk_rate(counters["pf_reraise"], counters["pf_reraise_opp"], "pf_reraise")
    fc, fc_c = _shrunk_rate(counters["flop_cbet_fold"], counters["flop_cbet_faced"], "fold_to_flop_cbet")
    tf, tf_c = _shrunk_rate(counters["turn_fold"], counters["turn_faced"], "fold_to_turn")
    rc, rc_c = _shrunk_rate(counters["river_call"], counters["river_faced"], "river_call")
    af = counters["n_aggr"] / counters["n_call"] if counters["n_call"] > 0 else (
        _AGGR_PRIOR if counters["n_aggr"] == 0 else float(counters["n_aggr"]))
    return {
        "hands": hands,
        "vpip": vpip, "vpip_conf": vpip_c,
        "pfr": pfr, "pfr_conf": pfr_c,
        "pf_reraise": rer, "pf_reraise_conf": rer_c,
        "fold_to_flop_cbet": fc, "fold_to_flop_cbet_conf": fc_c,
        "fold_to_turn": tf, "fold_to_turn_conf": tf_c,
        "river_call": rc, "river_call_conf": rc_c,
        "aggression_factor": af,
    }


def _classify_opponent(counters: dict) -> tuple[str, float]:
    """Map a profile to (archetype, confidence) in {nit,station,maniac,tag,normal,unknown}.

    Confidence ramps with hands observed and gates downstream exploits.
    """
    hands = counters.get("hands", 0)
    if hands < _MIN_HANDS_FOR_TAG:
        return "unknown", 0.0

    L = _opp_leaks(counters)
    vpip, pfr, af = L["vpip"], L["pfr"], L["aggression_factor"]
    river_call = L["river_call"]
    confidence = min(0.9, 0.25 + hands / 50.0)

    if vpip < 0.16 and pfr < 0.12:
        tag = "nit"          # tight: a preflop-folder never reaches a flop, so
        #                      VPIP/PFR define the nit; fold_cbet only refines.
    elif vpip > 0.42 and (pfr > 0.30 or af > 2.5):
        tag = "maniac"
    elif vpip > 0.30 and af < 1.0 and river_call > 0.55:
        tag = "station"
    elif 0.18 <= vpip <= 0.30 and pfr >= 0.14 and af >= 1.3:
        tag = "tag"
    else:
        tag = "normal"
    return tag, confidence


def _opponent_profile_counts(
    profiles: dict, players: list, my_seat: int
) -> tuple[int, int, int]:
    """Return (maniac_count, calling_station_count, nit_count) among active opponents.

    `profiles` is the prebuilt per-bot_id leak table from the street-aware
    profiler (Modules A2/A3). Only reads with confidence ≥ _PROFILE_ACT_CONF
    count, so sparse/uncertain opponents fall through to neutral (no nudge).
    """
    maniac = station = nit = 0
    for p in players:
        if p["seat"] == my_seat or p["state"] in ("folded", "busted"):
            continue
        prof = profiles.get(p["bot_id"])
        if prof is None:
            continue
        tag, conf = _classify_opponent(prof)
        if conf < _PROFILE_ACT_CONF:
            continue
        if tag == "maniac":
            maniac += 1
        elif tag == "station":
            station += 1
        elif tag == "nit":
            nit += 1
    return maniac, station, nit


def _run_mc(game_state: dict, time_limit: float = 0.5, max_iters: int | None = None,
            range_conditioned: bool = True, profiles: dict | None = None,
            deadline: float | None = None) -> float:
    if deadline is None:
        deadline = _DECIDE_DEADLINE
    my_cards    = list(map(Card, game_state["your_cards"]))
    board_cards = list(map(Card, game_state["community_cards"]))
    rest_cards  = [c for c in ALL_CARDS if c not in my_cards and c not in board_cards]
    my_seat = game_state["seat_to_act"]
    active_opps = [p for p in game_state["players"]
                   if p["seat"] != my_seat and p["state"] in ("active", "all_in")]
    n_opp = max(len(active_opps), 1)

    # Range-conditioned sampling (Module B): bias each live opponent's hole cards
    # by a strength floor derived from their archetype read. Self-contained so the
    # GTO path, MC path, and fallback all share it.
    opp_floors = None
    if range_conditioned and active_opps:
        if profiles is None:
            my_bot_id = next((p["bot_id"] for p in game_state["players"]
                              if p["seat"] == my_seat), None)
            profiles = _build_opponent_profiles(
                game_state.get("match_action_log", []), my_bot_id)
        street = game_state.get("street", "flop")
        # Action-conditioned bump (cfr_equity_v28 #4): aggression shown THIS hand
        # (3-bets+, postflop barrels, big bets) means a stronger live range than
        # the archetype alone implies. Applied to every still-live opponent.
        aggr_bump = _hand_aggression_bump(game_state)
        opp_floors = []
        for p in active_opps:
            tag, conf = _classify_opponent(profiles.get(p["bot_id"]) or {})
            # A maniac's aggression is bluffy, not strong — do NOT tighten their
            # assumed range when they bet (cf. cfr_equity_v28's maniac floor cut).
            bump = 0.0 if (tag == "maniac" and conf >= 0.35) else aggr_bump
            floor = _seat_range_floor(tag, conf, street) + bump
            opp_floors.append(min(0.92, floor))

    return monte_carlo_equity(my_cards, board_cards, rest_cards, n_opp,
                              time_limit, max_iters, opp_floors=opp_floors,
                              deadline=deadline)


# ── Main entry point ───────────────────────────────────────────────────────

def _action_seed(game_state: dict) -> int:
    """Stable integer seed derived entirely from deterministic game-state fields."""
    hand_id = game_state.get("hand_id", "")
    # hand_id format: "<match_id>_h<NNNN>" — extract the hand number
    try:
        hand_num = int(hand_id.rsplit("_h", 1)[-1])
    except (ValueError, IndexError):
        hand_num = 0
    action_count = len(game_state.get("action_log", []))
    seat = game_state.get("seat_to_act", 0)
    return (hand_num * 10_000 + action_count * 10 + seat) % (2 ** 31)


def decide(game_state: dict) -> dict:
    """Called once per action. Must return within 2 seconds."""

    if game_state.get("type") == "warmup":
        return {"action": "check"}

    # Latency guard: hard wall-clock cap for this action, read by the MC loop.
    global _DECIDE_DEADLINE
    _DECIDE_DEADLINE = time.time() + _TIME_BUDGET

    # Seed both RNGs from game state so replays with the same match seed are identical.
    _seed = _action_seed(game_state)
    random.seed(_seed)
    np.random.seed(_seed)

    my_seat = game_state["seat_to_act"]

    # Build opponent profiles once per decision; the same dict feeds the preflop
    # range nudges, postflop sizing, range-conditioned equity, and the exploit /
    # anti-punt layers (previously rebuilt 3× per postflop action).
    my_bot_id = next((p["bot_id"] for p in game_state["players"]
                      if p["seat"] == my_seat), None)
    profiles = _build_opponent_profiles(
        game_state.get("match_action_log", []), my_bot_id)
    profile_counts = _opponent_profile_counts(profiles, game_state["players"], my_seat)

    try:
        if game_state.get("street", "preflop") == "preflop":
            # Chart-derived heuristic preflop policy. The trained CFR table is
            # left loaded but unused: at 400k traversals over ~4.9M info sets it
            # never converged and open-jams premiums at 100bb. Re-enable via
            # _preflop_table_decide() once the table is retrained to convergence.
            return _preflop_decide(game_state, profile_counts)
        # Postflop: net or MC engine, behind the equity risk gate.
        return _postflop_decide(game_state, profile_counts, profiles)
    except Exception:
        # Last-ditch safe fallback: never crash → never auto-fold from an error.
        owed = game_state.get("amount_owed", 0)
        if owed == 0:
            return {"action": "check"}
        try:
            equity = _run_mc(game_state, max_iters=400)
            pot    = game_state["pot"]
            pot_odds = owed / (pot + owed)
            if equity >= pot_odds:
                return {"action": "call"}
        except Exception:
            pass
        return {"action": "fold"}
