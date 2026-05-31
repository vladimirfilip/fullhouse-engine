"""Vlad's NLHE bot — GTO network + preflop CFR table + MC fallback.

decide() is called once per action and must return within 2 seconds.
See CLAUDE.md for architecture details and submission format.
"""

import math
import os
import random
import time

import numpy as np
from eval7 import Card, evaluate

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
# Preflop is handled by a hand-strength heuristic (see _preflop_decide).  The
# tabular CFR preflop table is OFF: the shipped table is a 5k-iteration smoke
# run that confidently jams trash (98s/J2s) — re-enable only after a full
# retrain (see RETRAINING.md).
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
    # Frozen engine doesn't surface dealer_seat, so derive from the SB action.
    # Heads-up: dealer IS the SB. For 3+ players, SB is one left of dealer.
    seat      = gs["seat_to_act"]
    al        = gs["action_log"]
    n_in_game = max(len(gs["players"]), 1)
    dealer    = 0
    if al and al[0].get("action") == "small_blind":
        sb_seat = al[0]["seat"]
        dealer  = sb_seat if n_in_game == 2 else (sb_seat - 1) % n_in_game
    hero_pos = (seat - dealer) % n_in_game
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
    offset = 78 - hi * (hi + 1) // 2 + lo
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
    al         = gs["action_log"]
    seat       = gs["seat_to_act"]
    n_in_game  = max(len(gs["players"]), 1)
    dealer     = 0
    if al and al[0].get("action") == "small_blind":
        sb_seat = al[0]["seat"]
        dealer  = sb_seat if n_in_game == 2 else (sb_seat - 1) % n_in_game

    hero_pos = (seat - dealer) % n_in_game
    bucket   = _preflop_bucket(gs["your_cards"][0], gs["your_cards"][1])

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
                     equity: float | None = None) -> int:
    """
    GTO network with a targeted MC equity correction on the fold/call decision.

    Raise sizing is owned entirely by the GTO network — immediate chip EV and
    long-run strategy EV diverge for raise sizing (range balancing, multi-street
    value, etc.), so a crude 1-level model only corrupts those decisions.

    The only EV correction applied: when facing a bet (owed > 0), shift
    probability between FOLD and CHECK_CALL in proportion to how far our MC
    equity sits above or below the pot odds break-even.  The shift is bounded
    so the network's read on the full situation still dominates.

    `equity` may be supplied precomputed (shared with the risk gate) to avoid a
    second Monte-Carlo rollout.
    """
    gto_arr = _mask_probs(gto_probs, legal)

    owed = gs["amount_owed"]
    if owed <= 0:
        # Check or bet — no fold/call tension, trust the network fully.
        return legal[int(np.random.choice(len(legal), p=gto_arr))]

    pot = gs["pot"]
    if equity is None:
        equity = _run_mc(gs, max_iters=2_000)

    pot_odds     = owed / (pot + owed)          # minimum equity to break even on a call
    equity_edge  = equity - pot_odds            # + = call is profitable; − = fold preferred

    fold_idx = next((i for i, a in enumerate(legal) if a == _FOLD),       None)
    call_idx = next((i for i, a in enumerate(legal) if a == _CHECK_CALL), None)

    if fold_idx is None or call_idx is None:
        return legal[int(np.random.choice(len(legal), p=gto_arr))]

    # Shift up to 20 pp between fold and call.  Positive edge → move mass from
    # fold to call; negative edge → move mass from call to fold.
    shift   = float(np.clip(equity_edge * 0.8, -0.20, 0.20))
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
_OPEN_THRESH = {"UTG": 8.5, "MP": 7.5, "CO": 6.5, "BTN": 5.0, "SB": 6.0, "BB": 8.0}


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

    seat = gs["seat_to_act"]
    al   = gs["action_log"]
    n    = max(len(gs["players"]), 1)
    dealer = 0
    if al and al[0].get("action") == "small_blind":
        sb = al[0]["seat"]
        dealer = sb if n == 2 else (sb - 1) % n
    off = (seat - dealer) % n

    n_raises = sum(1 for e in al if e.get("action") in ("raise", "all_in"))
    limpers  = sum(1 for e in al if e.get("action") == "call") if n_raises == 0 else 0

    eff_bb = min(gs["your_stack"], _max_opponent_stack(gs)) / _BIG_BLIND
    return {
        "hi": hi, "lo": lo, "suited": suited, "pair": hi == lo, "label": label,
        "chen": _chen_score(hi, lo, suited),
        "pos": _pos_category(off, n), "n": n,
        "n_raises": n_raises, "limpers": limpers, "eff_bb": eff_bb,
    }


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
        if chen >= open_thr:
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

    if label in _PREMIUM_3BET or chen >= (15.0 if facing_3bet else 13.0):
        # 3-bet/4-bet for value.
        if facing_3bet:
            # Jam all premium hands vs a 3-bet: no fold equity for a 4-bet-fold
            # and opponents' 3-bet ranges are strong enough to call jams with QQ+/AK.
            return {"action": "all_in"}
        if eff_bb <= 35:
            return {"action": "all_in"}
        return _raise_to_amount(gs, cur * 3)

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


def _net_postflop(gs: dict, equity: float | None) -> dict:
    """GTO strategy net with the precomputed-equity fold/call correction."""
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

    action_idx = _realtime_search(gs, legal, probs, equity=equity)
    return _abstract_to_raw(action_idx, gs)


def _mc_postflop(gs: dict, equity: float, profile_counts: tuple) -> dict:
    """Monte-Carlo equity + pot-odds engine with pot-fraction sizing."""
    maniac, station, nit = profile_counts
    owed    = gs["amount_owed"]
    pot     = max(1, gs["pot"])
    stack   = gs["your_stack"]
    my_bet  = gs["your_bet_this_street"]
    cur     = gs["current_bet"]
    min_r   = gs["min_raise_to"]
    all_tot = my_bet + stack
    texture = _board_texture(gs.get("community_cards", []))
    n_active = sum(1 for p in gs["players"]
                   if p.get("state") == "active" and p["seat"] != gs["seat_to_act"])

    pot_odds  = owed / (pot + owed) if owed > 0 else 0.0
    multi_adj = 0.20 if n_active >= 3 else 0.08
    call_thr  = pot_odds + multi_adj - 0.03 * maniac + 0.04 * nit

    if equity < call_thr:
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
    if station > 0 and maniac == 0:
        bet_frac += 0.15          # size up vs calling stations
    if maniac > 0 and owed > 0:
        bet_frac += 0.15          # re-raise big vs maniacs

    target = cur + max(round(pot * bet_frac), min_r - cur)
    target = max(target, min_r)
    if target >= all_tot:
        return {"action": "all_in"}
    return {"action": "raise", "amount": target}


def _board_texture(board: list) -> str:
    """Classify the flop texture: dry / semi / wet / paired."""
    if len(board) < 3:
        return "none"
    flop = board[:3]
    ranks = [_RANK_ORDER[c[0]] for c in flop]
    suits = [c[1] for c in flop]
    if len(set(ranks)) < 3:
        return "paired"
    two_tone = max(suits.count(s) for s in set(suits)) >= 2
    connected = max(ranks) - min(ranks) <= 4
    if two_tone and connected:
        return "wet"
    if two_tone or connected:
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


def _postflop_decide(gs: dict, profile_counts: tuple = (0, 0, 0)) -> dict:
    """Route the postflop decision through the configured engine + risk gate."""
    equity = _run_mc(gs, max_iters=1_200)

    if _POSTFLOP_ENGINE == "mc" or _GTO_LAYERS is None:
        action = _mc_postflop(gs, equity, profile_counts)
    else:
        action = _net_postflop(gs, equity)

    return _risk_gate(action, gs, equity)


# ── Monte Carlo equity (fallback) ──────────────────────────────────────────

def monte_carlo_equity(
    hole_cards: list,
    board_cards: list,
    remaining_cards: list,
    num_opponents: int = 1,
    time_limit: float = 0.5,
    max_iters: int | None = None,
) -> float:
    start = time.time()
    wins, iters = 0, 0
    while (max_iters is None and time.time() - start < time_limit) or \
          (max_iters is not None and iters < max_iters):
        random.shuffle(remaining_cards)
        opp_hands = [remaining_cards[i * 2: i * 2 + 2] for i in range(num_opponents)]
        board     = list(board_cards)
        for i in range(5 - len(board_cards)):
            board.append(remaining_cards[2 * num_opponents + i])
        my_score   = evaluate(hole_cards + board)
        opp_scores = [evaluate(oh + board) for oh in opp_hands]
        best       = max(my_score, *opp_scores)
        if my_score == best:
            n_tied  = opp_scores.count(best)
            wins   += 1 / (n_tied + 1)
        iters += 1
    return wins / max(iters, 1)


def choose_action(mc_equity, pot, amount_owed, already_bet, min_raise_to, your_stack, n_players,
                  maniac_count=0, calling_station_count=0, nit_count=0):
    if maniac_count > 0:
        buffer = 0.02
    elif n_players == 2:
        buffer = 0.10
    elif n_players <= 4:
        buffer = 0.15
    else:
        buffer = 0.30
    buffer = max(0.0, buffer - 0.05 * nit_count)
    required_equity = amount_owed / (pot + amount_owed) if amount_owed > 0 else 0

    if mc_equity < required_equity + buffer:
        return {"action": "check"} if amount_owed == 0 else {"action": "fold"}

    raise_threshold = 0.60 if (calling_station_count > 0 and maniac_count == 0) else 0.80
    if mc_equity > raise_threshold or mc_equity > required_equity + buffer + 0.15:
        all_chips    = already_bet + your_stack
        if all_chips < pot * 2:
            return {"action": "all_in"}
        raise_amount = max(min_raise_to, already_bet + amount_owed)
        raise_amount = min(raise_amount, all_chips)
        if raise_amount == all_chips:
            return {"action": "all_in"}
        return _jitter_raise(raise_amount, min_raise_to, all_chips)

    return {"action": "check"} if amount_owed == 0 else {"action": "call"}


def _opponent_profile_counts(
    match_log: list, players: list, my_seat: int
) -> tuple[int, int, int]:
    """Return (maniac_count, calling_station_count, nit_count) among active opponents."""
    my_bot_id = next((p["bot_id"] for p in players if p["seat"] == my_seat), None)
    counts: dict[str, dict] = {}
    for entry in match_log:
        bid = entry.get("bot_id")
        if bid is None or bid == my_bot_id:
            continue
        act = entry.get("action", "")
        if bid not in counts:
            counts[bid] = {
                "fold": 0, "check": 0, "call": 0, "raise": 0, "all_in": 0, "total": 0
            }
        if act in counts[bid]:
            counts[bid][act] += 1
        counts[bid]["total"] += 1

    profiles: dict[str, str] = {}
    for bid, c in counts.items():
        t = c["total"]
        if t < 5:
            profiles[bid] = "unknown"
        elif (c["all_in"] + c["raise"]) / t > 0.50:
            profiles[bid] = "maniac"
        elif c["call"] / t > 0.50:
            profiles[bid] = "calling_station"
        elif c["fold"] / t > 0.60:
            profiles[bid] = "nit"
        else:
            profiles[bid] = "normal"

    maniac = station = nit = 0
    for p in players:
        if p["seat"] == my_seat or p["state"] in ("folded", "busted"):
            continue
        label = profiles.get(p["bot_id"], "unknown")
        if label == "maniac":
            maniac += 1
        elif label == "calling_station":
            station += 1
        elif label == "nit":
            nit += 1
    return maniac, station, nit


def _run_mc(game_state: dict, time_limit: float = 0.5, max_iters: int | None = None) -> float:
    my_cards    = list(map(Card, game_state["your_cards"]))
    board_cards = list(map(Card, game_state["community_cards"]))
    rest_cards  = [c for c in ALL_CARDS if c not in my_cards and c not in board_cards]
    n_opp = max(sum(
        p["state"] in ("active", "all_in")
        for p in game_state["players"]
        if p["seat"] != game_state["seat_to_act"]
    ), 1)
    return monte_carlo_equity(my_cards, board_cards, rest_cards, n_opp, time_limit, max_iters)


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

    # Seed both RNGs from game state so replays with the same match seed are identical.
    _seed = _action_seed(game_state)
    random.seed(_seed)
    np.random.seed(_seed)

    my_seat = game_state["seat_to_act"]

    # Opponent profiling drives both preflop range nudges and postflop sizing.
    profile_counts = _opponent_profile_counts(
        game_state.get("match_action_log", []), game_state["players"], my_seat
    )

    try:
        if game_state.get("street", "preflop") == "preflop":
            # Heuristic preflop policy (the trained table is disabled — it shipped
            # a smoke-run that jams trash; see RETRAINING.md).
            return _preflop_decide(game_state, profile_counts)
        # Postflop: net or MC engine, behind the equity risk gate.
        return _postflop_decide(game_state, profile_counts)
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
