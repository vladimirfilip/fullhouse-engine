"""

RULES:
  - Implement the decide() function below. That's it.
  - You may import any stdlib module and any library in requirements.txt
  - You have 2 seconds to return an action or you auto-fold
  - If your function crashes, it auto-folds for that hand

NOT ALLOWED (will DQ your bot):
  - External API calls: no Claude/OpenAI/Anthropic/Google/any HTTP. Network is
    blocked at the container level; trying anyway is a DQ.
  - File writes during gameplay; data/ is read-only and only at import time.
  - subprocess / os.system / shell commands.
  - Threading or async tricks to dodge the 2s/action signal timer.
  - Reflection: __import__('socket'), getattr(__builtins__, 'open'),
    eval(), exec(), compile() — do all flagged by the validator.
  - Collusion between bots you've registered with friends — bots must play
    independently; coordinated soft-play or chip-dumping = both DQ'd.
  - Reading other bots' code or hole cards (you can't anyway, but trying = DQ).

OPTIONAL DATA FILES (NEW):
  Submit a .zip archive containing:
    bot.py        (this file, required at root)
    data/         (optional directory with .npz, .pkl, .bin, etc.)

  At module-import time only, you can read from a sibling 'data/' directory:

      import os
      DATA_DIR = os.environ.get("BOT_DATA_DIR",
                                os.path.join(os.path.dirname(__file__), "data"))
      with open(os.path.join(DATA_DIR, "blueprint.npz"), "rb") as f:
          BLUEPRINT = ...load(f)

  Limits:
    - Total submission (bot.py + data/) <= 250 MB
    - data/ alone <= 200 MB
    - bot.py <= 5 MB
    - File access during decide() is blocked at the OS level

CARD FORMAT:
  Cards are strings like "As" (Ace of spades), "Td" (Ten of diamonds)
  Ranks: 2 3 4 5 6 7 8 9 T J Q K A
  Suits: s (spades) h (hearts) d (diamonds) c (clubs)

RETURN FORMAT:
  {"action": "fold"}
  {"action": "check"}          # only valid when amount_owed == 0
  {"action": "call"}
  {"action": "raise", "amount": 1200}   # amount = TOTAL bet, not raise-by
  {"action": "all_in"}

  Invalid actions default to fold. Raises below min_raise_to are snapped up.
"""

# ── Imports ───────────────────────────────────────────────────────────────
import os
import random
import time

import math

import numpy as np
from eval7 import Card, evaluate

# ─────────────────────────────────────────────────────────────────────────

BOT_NAME   = "The House"
BOT_AVATAR = "robot_1"

RANKS   = "23456789TJQKA"
SUITS   = "shdc"
ALL_CARDS = [Card(r + s) for r in RANKS for s in SUITS]

_N_PLAYERS    = 6
_INITIAL_STACK = 10_000
_SMALL_BLIND   = 50
_BIG_BLIND     = 100
_MAX_RAISES_PER_STREET = 4

_INPUT_DIM = 308   # must match deep_cfr/config.py and deep_cfr_cpp/src/config.hpp

# ── Card encoding (must match deep_cfr/features.py) ───────────────────────
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
_PF_ACTIVE_RAISES = [
    (_HALF,  0.50),
    (_FULL,  1.00),
    (_2X,    2.00),
    (_0_27X, 0.27),
    (_THIRD, 1.0 / 3.0),
    (_1_72X, 1.72),
]


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

try:
    _pf_data = np.load(_PREFLOP_TABLE_PATH)
    _pf_n_players = int(_pf_data["n_players"])
    _pf_stack_bb  = int(_pf_data["stack_bb"])
    if _pf_n_players == _N_PLAYERS and _pf_stack_bb == _INITIAL_STACK // _BIG_BLIND:
        _pf_keys = _pf_data["keys"]
        _pf_strat = _pf_data["strategy"]
        _PREFLOP_TABLE = {int(k): _pf_strat[i] for i, k in enumerate(_pf_keys)}
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
    pot     = gs["pot"]
    owed    = gs["amount_owed"]
    cur_bet = gs["current_bet"]
    min_r   = gs["min_raise_to"]
    stack   = gs["your_stack"]
    my_bet  = gs["your_bet_this_street"]
    all_tot = my_bet + stack
    eff_pot = pot + owed

    if action_idx == _FOLD:
        return {"action": "fold"}
    if action_idx == _CHECK_CALL:
        return {"action": "check" if owed == 0 else "call"}
    if action_idx == _0_27X:
        target = cur_bet + max(round(eff_pot * 0.27), min_r - cur_bet)
        return _jitter_raise(target, min_r, all_tot)
    if action_idx == _THIRD:
        target = cur_bet + max(eff_pot // 3, min_r - cur_bet)
        return _jitter_raise(target, min_r, all_tot)
    if action_idx == _HALF:
        target = cur_bet + max(eff_pot // 2, min_r - cur_bet)
        return _jitter_raise(target, min_r, all_tot)
    if action_idx == _FULL:
        target = cur_bet + max(eff_pot, min_r - cur_bet)
        return _jitter_raise(target, min_r, all_tot)
    if action_idx == _1_72X:
        target = cur_bet + max(round(eff_pot * 1.72), min_r - cur_bet)
        return _jitter_raise(target, min_r, all_tot)
    if action_idx == _2X:
        target = cur_bet + max(eff_pot * 2, min_r - cur_bet)
        return _jitter_raise(target, min_r, all_tot)
    if action_idx == _ALL_IN:
        return {"action": "all_in"}
    return {"action": "fold"}


# ── GTO decision ───────────────────────────────────────────────────────────


def _realtime_search(gs: dict, legal: list, gto_probs: np.ndarray) -> int:
    """
    GTO network with a targeted MC equity correction on the fold/call decision.

    Raise sizing is owned entirely by the GTO network — immediate chip EV and
    long-run strategy EV diverge for raise sizing (range balancing, multi-street
    value, etc.), so a crude 1-level model only corrupts those decisions.

    The only EV correction applied: when facing a bet (owed > 0), shift
    probability between FOLD and CHECK_CALL in proportion to how far our MC
    equity sits above or below the pot odds break-even.  The shift is bounded
    so the network's read on the full situation still dominates.
    """
    # GTO distribution over legal actions.
    gto_arr = np.array([gto_probs[a] for a in legal], dtype=np.float64)
    gto_arr = np.maximum(gto_arr, 0.0)
    if gto_arr.sum() < 1e-12:
        gto_arr[:] = 1.0 / len(legal)
    else:
        gto_arr /= gto_arr.sum()

    owed = gs["amount_owed"]
    if owed <= 0:
        # Check or bet — no fold/call tension, trust the network fully.
        return legal[int(np.random.choice(len(legal), p=gto_arr))]

    pot = gs["pot"]
    equity = _run_mc(gs, max_iters=3_000)

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

    # Build legal set (same logic as _gto_decide)
    owed  = gs["amount_owed"]
    stack = gs["your_stack"]
    pot   = gs["pot"]
    eff_pot = pot + owed
    cur   = gs["current_bet"]
    min_r = gs["min_raise_to"]
    my_bet = gs.get("your_bet_this_street", 0)
    all_tot = my_bet + stack
    n_raises = _derive_n_raises_this_street(gs["action_log"], len(gs["players"]))

    legal = [_CHECK_CALL]
    if owed > 0:
        legal.append(_FOLD)
    if stack > 0:
        if n_raises < _MAX_RAISES_PER_STREET:
            last_tgt = -1
            for a_idx, rb_raw in (
                (_0_27X, round(eff_pot * 0.27)),
                (_THIRD, eff_pot // 3),
                (_HALF,  eff_pot // 2),
                (_FULL,  eff_pot),
                (_1_72X, round(eff_pot * 1.72)),
                (_2X,    eff_pot * 2),
            ):
                rb  = max(rb_raw, min_r - cur)
                tgt = cur + rb
                if tgt < all_tot and tgt != last_tgt:
                    last_tgt = tgt
                    legal.append(a_idx)
        legal.append(_ALL_IN)

    # Restrict to legal actions and renormalise
    legal_probs = np.array([max(probs[a], 0.0) for a in legal], dtype=np.float64)
    total = legal_probs.sum()
    if total < 1e-12:
        legal_probs[:] = 1.0 / len(legal)
    else:
        legal_probs /= total

    action_idx = legal[int(np.random.choice(len(legal), p=legal_probs))]
    return _abstract_to_raw(action_idx, gs)


def _gto_decide(gs: dict) -> dict:
    """Run the GTO strategy net with real-time EV blending."""
    # Preflop tabular CFR table takes priority when applicable.
    pf = _preflop_table_decide(gs)
    if pf is not None:
        return pf

    vec   = _build_feature_vector(gs)
    probs = _numpy_forward(_GTO_LAYERS, vec)  # type: ignore[arg-type]

    owed  = gs["amount_owed"]
    stack = gs["your_stack"]
    # n_raises is already encoded in vec[142]; read it back to avoid
    # a second walk of the action log.
    n_raises = round(vec[142] * _MAX_RAISES_PER_STREET)

    legal = [_CHECK_CALL]
    if owed > 0:
        legal.append(_FOLD)
    if stack > 0:
        if n_raises < _MAX_RAISES_PER_STREET:
            # Mirror the C++ deduplication in get_legal_actions():
            #   • drop any bet whose target >= all_in_tot (would collapse to
            #     all-in, duplicating the explicit ALL_IN entry below)
            #   • skip duplicate targets (two fractions can clamp to the same
            #     min-raise at shallow effective stacks)
            pot      = gs["pot"]
            eff_pot  = pot + owed
            cur      = gs["current_bet"]
            min_r    = gs["min_raise_to"]
            my_bet   = gs.get("your_bet_this_street", 0)
            all_tot  = my_bet + stack
            min_rb   = min_r - cur
            last_tgt = -1
            for a_idx, rb_raw in (
                (_0_27X, round(eff_pot * 0.27)),
                (_THIRD, eff_pot // 3),
                (_HALF,  eff_pot // 2),
                (_FULL,  eff_pot),
                (_1_72X, round(eff_pot * 1.72)),
                (_2X,    eff_pot * 2),
            ):
                rb  = max(rb_raw, min_rb)
                tgt = cur + rb
                if tgt < all_tot and tgt != last_tgt:
                    last_tgt = tgt
                    legal.append(a_idx)
        # ALL_IN is always legal when the player has chips, even past the
        # raise cap — committing all chips is not a standard re-raise.
        legal.append(_ALL_IN)

    action_idx = _realtime_search(gs, legal, probs)
    return _abstract_to_raw(action_idx, gs)


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


def _profile_opponents(match_log: list, players: list, my_seat: int) -> dict:
    my_bot_id = next((p["bot_id"] for p in players if p["seat"] == my_seat), None)
    counts: dict[str, dict] = {}
    for entry in match_log:
        bid = entry.get("bot_id")
        if bid is None or bid == my_bot_id:
            continue
        act = entry.get("action", "")
        if bid not in counts:
            counts[bid] = {"fold": 0, "check": 0, "call": 0, "raise": 0, "all_in": 0, "total": 0}
        if act in counts[bid]:
            counts[bid][act] += 1
        counts[bid]["total"] += 1

    profiles: dict[str, str] = {}
    for bid, c in counts.items():
        t = c["total"]
        if t < 5:
            profiles[bid] = "unknown"
            continue
        if (c["all_in"] + c["raise"]) / t > 0.50:
            profiles[bid] = "maniac"
        elif c["call"] / t > 0.50:
            profiles[bid] = "calling_station"
        elif c["fold"] / t > 0.60:
            profiles[bid] = "nit"
        else:
            profiles[bid] = "normal"
    return profiles


def _count_active_profiles(profiles: dict, players: list, my_seat: int) -> tuple[int, int, int]:
    maniac_count = station_count = nit_count = 0
    for p in players:
        if p["seat"] == my_seat or p["state"] in ("folded", "busted"):
            continue
        label = profiles.get(p["bot_id"], "unknown")
        if label == "maniac":
            maniac_count += 1
        elif label == "calling_station":
            station_count += 1
        elif label == "nit":
            nit_count += 1
    return maniac_count, station_count, nit_count


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
    profiles = _profile_opponents(
        game_state.get("match_action_log", []), game_state["players"], my_seat
    )
    maniac_count, station_count, nit_count = _count_active_profiles(
        profiles, game_state["players"], my_seat
    )

    # ── GTO path ──────────────────────────────────────────────────────────
    if _GTO_LAYERS is not None:
        try:
            return _gto_decide(game_state)
        except Exception:
            pass

    # ── Monte Carlo fallback ───────────────────────────────────────────────
    equity = _run_mc(game_state, time_limit=0.5)
    pot    = game_state["pot"]
    active = sum(p["state"] == "active" for p in game_state["players"])
    return choose_action(
        equity, pot,
        game_state["amount_owed"],
        game_state["your_bet_this_street"],
        game_state["min_raise_to"],
        game_state["your_stack"],
        active,
        maniac_count=maniac_count,
        calling_station_count=station_count,
        nit_count=nit_count,
    )
