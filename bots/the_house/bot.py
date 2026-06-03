"""The House - a lean, resilient NLHE bot.

A clean rewrite. Three layers, in priority order:

  1. POLICY    - a trained GTO strategy net (gto_strategy.npz) proposes an
                 action distribution over the 9-action abstraction. This is the
                 game-theoretic baseline that keeps us hard to exploit.
  2. EQUITY    - a Monte-Carlo eval7 equity estimate (range-conditioned by the
                 opponents' archetypes) is the resilience layer: it overrides the
                 net to hard-fold clear losers, cap commitment when we're behind,
                 and force value when we hold the near-nuts. The net never gets to
                 punt our stack.
  3. EXPLOIT   - lightweight per-opponent profiling (VPIP/PFR/aggression/river
                 calls) shifts the policy to farm weak field: bluff/steal more vs
                 over-folders, never bluff and size up value vs calling stations,
                 call lighter vs maniacs.

If the net is unavailable, layer 2 alone makes every decision (pure equity +
pot-odds). decide() must return within 2 s; a wall-clock deadline truncates the
only unbounded cost (the MC loop).

See CLAUDE.md for the engine protocol and submission limits.
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

# -- Table / engine constants ------------------------------------------------
RANKS = "23456789TJQKA"
SUITS = "shdc"
ALL_CARDS = [Card(r + s) for r in RANKS for s in SUITS]

_N_PLAYERS     = 6
_INITIAL_STACK = 10_000
_SMALL_BLIND   = 50
_BIG_BLIND     = 100
_MAX_RAISES_PER_STREET = 4
_INPUT_DIM     = 252
_N_ACTIONS     = 9

# -- Encodings (must match the training feature pipeline byte-for-byte) -------
_CARD_IDX = {r + s: ri * 4 + si for ri, r in enumerate(RANKS) for si, s in enumerate(SUITS)}
_ACTION_ONEHOT = {"fold": 0, "check": 1, "call": 1, "raise": 2, "all_in": 3}
_STREET_IDX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_RANK_ORDER = {r: i for i, r in enumerate(RANKS)}

# Abstract action indices (must match the trained net's output layout).
_FOLD, _CHECK_CALL = 0, 1
_0_27X, _THIRD, _HALF, _FULL, _1_72X, _2X, _ALL_IN = 2, 3, 4, 5, 6, 7, 8

# Raise ladder: (abstract index, raise-BY amount from effective pot). Single
# source of truth for both legality and amount mapping.
_RAISE_LADDER = [
    (_0_27X, lambda p: round(p * 0.27)),
    (_THIRD, lambda p: p // 3),
    (_HALF,  lambda p: p // 2),
    (_FULL,  lambda p: p),
    (_1_72X, lambda p: round(p * 1.72)),
    (_2X,    lambda p: p * 2),
]
_ACTION_POT_FRAC = {_0_27X: 0.27, _THIRD: 0.33, _HALF: 0.5, _FULL: 1.0,
                    _1_72X: 1.72, _2X: 2.0, _ALL_IN: 10.0}
_BET_ORDER = [_0_27X, _THIRD, _HALF, _FULL, _1_72X, _2X, _ALL_IN]

# -- Latency guard ------------------------------------------------------------
_TIME_BUDGET   = 1.4               # wall-clock budget per decide() (cap is 2 s)
_MC_MIN_ITERS  = 80                # floor so a truncated rollout stays usable
_DEADLINE: float | None = None     # absolute time.time() deadline, per decide()


# ==========================================================================
# -  Position helpers                                                          -
# ==========================================================================

def _hero_position(gs: dict) -> tuple[int, int]:
    """(hero offset from dealer, table size). Dealer derived from the SB post."""
    seat = gs["seat_to_act"]
    al = gs["action_log"]
    n = max(len(gs["players"]), 1)
    dealer = 0
    if al and al[0].get("action") == "small_blind":
        sb = al[0]["seat"]
        dealer = sb if n == 2 else (sb - 1) % n
    return (seat - dealer) % n, n


def _in_position(gs: dict) -> bool:
    """True if hero acts last among live players this street (button = last)."""
    seat = gs["seat_to_act"]
    hero_off, n = _hero_position(gs)
    dealer = (seat - hero_off) % n
    hero_rank = (hero_off - 1) % n
    for p in gs["players"]:
        if p["seat"] == seat or p.get("is_folded") or p.get("state") in ("folded", "busted"):
            continue
        if ((p["seat"] - dealer) % n - 1) % n > hero_rank:
            return False
    return True


def _n_active_opponents(gs: dict) -> int:
    seat = gs["seat_to_act"]
    return sum(1 for p in gs["players"]
               if p["seat"] != seat and p.get("state") in ("active", "all_in"))


def _derive_n_raises_this_street(action_log: list, n_seats: int) -> int:
    """Replay the public action log to recover the engine's per-street raise count."""
    if not action_log:
        return 0
    seats = sorted({e["seat"] for e in action_log})
    if not seats:
        return 0
    active = set(seats)
    all_in: set = set()
    bet_this = {s: 0 for s in seats}
    current_bet = 0
    last_agg = 0
    n_raises = 0
    to_act: set = set()
    started = False

    def reset_street():
        nonlocal current_bet, last_agg, n_raises
        for s in seats:
            bet_this[s] = 0
        current_bet = 0
        last_agg = _BIG_BLIND
        n_raises = 0

    for e in action_log:
        seat = e["seat"]
        act = e.get("action")
        amt = e.get("amount", 0) or 0
        if act == "small_blind":
            bet_this[seat] = amt
            current_bet = max(current_bet, amt)
            if amt < _SMALL_BLIND:
                all_in.add(seat)
            continue
        if act == "big_blind":
            bet_this[seat] = amt
            current_bet = max(current_bet, amt)
            last_agg = _BIG_BLIND
            if amt < _BIG_BLIND:
                all_in.add(seat)
            to_act = {s for s in active if s not in all_in}
            started = True
            continue
        if not started:
            started = True
            to_act = set(active) - all_in
        to_act.discard(seat)
        if act == "fold":
            active.discard(seat)
        elif act == "call":
            bet_this[seat] = current_bet
        elif act in ("raise", "all_in"):
            raise_sz = amt - current_bet
            if raise_sz >= last_agg and raise_sz > 0:
                n_raises += 1
                last_agg = raise_sz
                to_act = {s for s in active if s not in all_in and s != seat}
            bet_this[seat] = amt
            current_bet = max(current_bet, amt)
            if act == "all_in":
                all_in.add(seat)
        if not to_act and all(bet_this[s] == current_bet
                              for s in active if s not in all_in):
            reset_street()
            to_act = set(active) - all_in
    return n_raises


# ==========================================================================
# -  Feature vector + net forward (mirrors the training pipeline exactly)      -
# ==========================================================================

def _build_feature_vector(gs: dict) -> np.ndarray:
    vec = np.zeros(_INPUT_DIM, dtype=np.float32)
    for card in gs["your_cards"]:
        vec[_CARD_IDX[card]] = 1.0
    for card in gs["community_cards"]:
        vec[52 + _CARD_IDX[card]] = 1.0

    seat = gs["seat_to_act"]
    al = gs["action_log"]
    hero_pos, n_in_game = _hero_position(gs)
    vec[104 + hero_pos] = 1.0

    pot = gs["pot"]
    vec[110] = pot / _INITIAL_STACK
    for p in gs["players"][:_N_PLAYERS]:
        vec[111 + p["seat"]] = p["stack"] / _INITIAL_STACK
        ps = p["seat"]
        if p.get("is_folded"):
            vec[117 + ps] = 1.0
        if p.get("is_all_in"):
            vec[123 + ps] = 1.0
        vec[129 + ps] = p.get("bet_this_street", 0) / _INITIAL_STACK

    vec[135 + _STREET_IDX.get(gs.get("street", "preflop"), 0)] = 1.0

    owed = gs["amount_owed"]
    vec[139] = owed / max(pot + owed, 1)
    spr = gs["your_stack"] / max(pot, 1)
    vec[140] = min(math.log10(spr + 1.0) / math.log10(101.0), 1.0)
    vec[141] = owed / _INITIAL_STACK

    n_raises = _derive_n_raises_this_street(al, len(gs["players"]))
    vec[142] = min(n_raises, _MAX_RAISES_PER_STREET) / _MAX_RAISES_PER_STREET
    vec[143] = gs.get("your_bet_this_street", 0) / _INITIAL_STACK
    my_stack = gs["your_stack"]
    max_opp = 0
    for p in gs["players"]:
        if p["seat"] == seat or p.get("is_folded"):
            continue
        max_opp = max(max_opp, p.get("stack", 0))
    vec[144] = min(my_stack, max_opp) / _INITIAL_STACK

    last_seat, last_amt = -1, 0
    for e in al:
        if e.get("action") in ("raise", "all_in", "big_blind"):
            last_seat = e["seat"]
            last_amt = e.get("amount", 0) or 0
    if 0 <= last_seat < _N_PLAYERS:
        vec[145 + last_seat] = 1.0
        vec[151] = last_amt / _INITIAL_STACK
        # (rel-pos one-hot dropped in Tier-2b: redundant with seat one-hot + hero pos)

    board = gs["community_cards"]
    if board:
        bidx = [_CARD_IDX[c] for c in board]
        suits = [c % 4 for c in bidx]
        ranks = [c // 4 for c in bidx]
        max_suit = max(suits.count(s) for s in range(4))
        rank_cnt: dict[int, int] = {}
        for r in ranks:
            rank_cnt[r] = rank_cnt.get(r, 0) + 1
        pairs = sum(1 for c in rank_cnt.values() if c >= 2)
        rs = sorted(set(ranks))
        connected = any(rs[i + 1] - rs[i] == 1 for i in range(len(rs) - 1))
        vec[152] = 1.0 if max_suit >= 2 else 0.0   # flush-draw
        vec[153] = 1.0 if pairs >= 1 else 0.0       # paired
        vec[154] = 1.0 if connected else 0.0        # connected
    n_active = sum(1 for p in gs["players"]
                   if not p.get("is_folded") and p.get("state") != "busted")
    vec[155] = n_active / _N_PLAYERS

    regular = [e for e in al if e.get("action") not in ("small_blind", "big_blind")]
    for slot, e in enumerate(regular[-16:]):
        base = 156 + slot * 6
        vec[base] = e["seat"] / max(_N_PLAYERS - 1, 1)
        atype = _ACTION_ONEHOT.get(e.get("action", ""), -1)
        if 0 <= atype <= 3:
            vec[base + 1 + atype] = 1.0
        vec[base + 5] = (e.get("amount", 0) or 0) / _INITIAL_STACK
    return vec


def _numpy_forward(layers, x: np.ndarray) -> np.ndarray:
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1:
            x = np.where(x > 0, x, 0.01 * x)   # LeakyReLU
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# -- Load the GTO net at import / warmup --------------------------------------
_DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
_GTO_LAYERS = None
try:
    _d = np.load(os.path.join(_DATA_DIR, "gto_strategy.npz"))
    _n = int(_d["n_layers"])
    _tmp = [(_d[f"layer{i}_w"], _d[f"layer{i}_b"]) for i in range(_n)]
    if _tmp[0][0].shape[1] == _INPUT_DIM and _tmp[-1][0].shape[0] == _N_ACTIONS:
        _GTO_LAYERS = _tmp
except Exception:
    _GTO_LAYERS = None


# ==========================================================================
# -  Preflop CFR blueprint (Tier 1): GTO 6-max 100bb ranges                    -
# ==========================================================================
# A solved tabular preflop strategy keyed by a compact betting-context info-set.
# This is a byte-for-byte mirror of preflop_cfr/abstraction.py + cfr.py - any
# change to the key encoding there MUST be reflected here or the lookups miss.
_PF_KEY_VERSION = "betting_context_v1"
_PF_TABLE: dict[int, np.ndarray] | None = None
try:
    _pf_path = os.path.join(_DATA_DIR, "preflop_strategy.npz")
    if not os.path.exists(_pf_path):
        _pf_path = os.path.join(_DATA_DIR, "preflop_cfr", "preflop_strategy.npz")
    _pf = np.load(_pf_path, allow_pickle=False)
    if (str(_pf["key_version"]) == _PF_KEY_VERSION
            and int(_pf["n_players"]) == _N_PLAYERS
            and int(_pf["stack_bb"]) == _INITIAL_STACK // _BIG_BLIND):
        _ks, _ss = _pf["keys"], _pf["strategy"]
        _PF_TABLE = {int(k): _ss[i] for i, k in enumerate(_ks)}
except Exception:
    _PF_TABLE = None

_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211


def _pf_fnv1a(data: bytes) -> int:
    """FNV-1a 64-bit -> signed int64 (mirror of abstraction.fnv1a_64)."""
    h = _FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    return h - 2 ** 64 if h >= 2 ** 63 else h


def _pf_bucket(c1: str, c2: str) -> int:
    """169-bucket hand canonicalization (mirror of cards._hand_to_bucket_compute)."""
    r1, r2 = _RANK_ORDER[c1[0]], _RANK_ORDER[c2[0]]
    suited = c1[1] == c2[1]
    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    if hi == lo:
        return 12 - hi
    offset = (78 - hi * (hi + 1) // 2) + (hi - 1 - lo)
    return (13 + offset) if suited else (91 + offset)


def _pf_facing_bucket(owed: int, pot: int) -> int:
    """Mirror of abstraction.facing_bucket."""
    if owed <= 0:
        return 0
    r = owed / pot if pot > 0 else 0.0
    if r <= 0.40:
        return 1
    if r <= 0.85:
        return 2
    return 3


def _pf_infoset_key(gs: dict, n_raises: int) -> int | None:
    """Mirror of cfr._infoset_key derived from the live game state. Returns None
    if the spot is out of the solved abstraction (not full-ring 6-max)."""
    players = gs["players"]
    if len(players) != _N_PLAYERS:
        return None                       # solve is strictly 6-max
    seat = gs["seat_to_act"]
    hero_off, n = _hero_position(gs)
    if n != _N_PLAYERS:
        return None
    hero_pos = hero_off
    nr = n_raises if n_raises < 3 else 3
    facing = _pf_facing_bucket(gs["amount_owed"], gs["pot"])
    n_live = sum(1 for p in players if not p.get("is_folded"))
    # committed: hero has voluntarily invested beyond its blind.
    blind = _SMALL_BLIND if hero_pos == 1 else (_BIG_BLIND if hero_pos == 2 else 0)
    committed = 1 if gs.get("your_bet_this_street", 0) > blind else 0
    # last aggressor relative to hero (raise/all_in only; blinds excluded).
    last_rel = 6
    for e in reversed(gs["action_log"]):
        if e.get("action") in ("raise", "all_in"):
            last_rel = (e["seat"] - seat) % _N_PLAYERS
            break
    bucket = _pf_bucket(gs["your_cards"][0], gs["your_cards"][1])
    raw = f"{hero_pos}|{nr}|{facing}|{n_live}|{committed}|{last_rel}|{bucket}"
    return _pf_fnv1a(raw.encode())


def _preflop_raise_count(gs: dict) -> int:
    """Voluntary raises/all-ins so far this (preflop) hand - mirrors the solver's
    n_raises increment (every non-fold/call action)."""
    return sum(1 for e in gs["action_log"] if e.get("action") in ("raise", "all_in"))


def _pf_table_decide(gs: dict, legal: list) -> dict | None:
    """Tier 1: consult the solved preflop blueprint. Returns a raw action, or None
    to fall back to the heuristic chart (out of abstraction / unvisited info-set /
    effective stack far from the 100bb solve depth)."""
    if _PF_TABLE is None:
        return None
    eff_bb = _eff_stack(gs) / _BIG_BLIND
    if not (80.0 <= eff_bb <= 140.0):     # blueprint is a 100bb solve
        return None
    n_raises = _preflop_raise_count(gs)
    key = _pf_infoset_key(gs, n_raises)
    if key is None:
        return None
    strat = _PF_TABLE.get(key)
    if strat is None:
        return None
    # Sample over the legal & in-table (>0) actions to preserve the GTO mixture.
    probs = np.array([max(strat[a], 0.0) if a in legal else 0.0
                      for a in range(_N_ACTIONS)], dtype=np.float64)
    s = probs.sum()
    if s < 1e-9:
        return None
    probs /= s
    idx = int(np.random.choice(_N_ACTIONS, p=probs))
    if idx == _FOLD and gs["amount_owed"] == 0:
        return {"action": "check"}        # never fold a free option
    # Deep-stack jam guard: while the blueprint is mid-training it over-uses
    # ALL_IN; at >25bb a 100bb open/3-bet-jam just punts. Keep the range and
    # aggression but use a pot-sized raise instead. (Remove once the table
    # converges to sized raises.)
    if idx == _ALL_IN and eff_bb > 25.0 and _FULL in legal:
        return _abstract_to_raw(_FULL, gs)
    return _abstract_to_raw(idx, gs)


# ==========================================================================
# -  Abstract-action - raw-action mapping                                      -
# ==========================================================================

def _legal_actions(gs: dict, n_raises: int) -> list:
    """Abstract action indices legal in the current state."""
    owed = gs["amount_owed"]
    stack = gs["your_stack"]
    cur = gs["current_bet"]
    min_r = gs["min_raise_to"]
    my_bet = gs.get("your_bet_this_street", 0)
    all_tot = my_bet + stack
    eff_pot = gs["pot"] + owed
    min_rb = min_r - cur
    legal = [_CHECK_CALL]
    if owed > 0:
        legal.append(_FOLD)
    if stack > 0:
        if n_raises < _MAX_RAISES_PER_STREET:
            last_tgt = -1
            for a_idx, rb_fn in _RAISE_LADDER:
                tgt = cur + max(rb_fn(eff_pot), min_rb)
                if tgt < all_tot and tgt != last_tgt:
                    last_tgt = tgt
                    legal.append(a_idx)
        legal.append(_ALL_IN)
    return legal


def _jitter_raise(target: int, min_r: int, all_tot: int, jitter: float = 0.06) -> dict:
    j = int(target * random.uniform(1 - jitter, 1 + jitter))
    j = max(j, min_r)
    if j >= all_tot:
        return {"action": "all_in"}
    return {"action": "raise", "amount": j}


def _abstract_to_raw(idx: int, gs: dict) -> dict:
    owed = gs["amount_owed"]
    cur = gs["current_bet"]
    min_r = gs["min_raise_to"]
    stack = gs["your_stack"]
    my_bet = gs["your_bet_this_street"]
    all_tot = my_bet + stack
    eff_pot = gs["pot"] + owed
    if idx == _FOLD:
        return {"action": "fold"}
    if idx == _CHECK_CALL:
        return {"action": "check" if owed == 0 else "call"}
    if idx == _ALL_IN:
        return {"action": "all_in"}
    for a_idx, rb_fn in _RAISE_LADDER:
        if idx == a_idx:
            target = cur + max(rb_fn(eff_pot), min_r - cur)
            return _jitter_raise(target, min_r, all_tot)
    return {"action": "fold"}


def _mask_probs(full: np.ndarray, legal: list) -> np.ndarray:
    arr = np.maximum(np.array([full[a] for a in legal], dtype=np.float64), 0.0)
    s = arr.sum()
    if s < 1e-12:
        arr[:] = 1.0 / len(arr)
    else:
        arr /= s
    return arr


def _idx_of(legal: list, action: int):
    return next((i for i, a in enumerate(legal) if a == action), None)


# ==========================================================================
# -  Equity (Monte-Carlo, range-conditioned)                                   -
# ==========================================================================

def _chen_score(hi: str, lo: str, suited: bool) -> float:
    _base = {"A": 10.0, "K": 8.0, "Q": 7.0, "J": 6.0, "T": 5.0, "9": 4.5,
             "8": 4.0, "7": 3.5, "6": 3.0, "5": 2.5, "4": 2.0, "3": 1.5, "2": 1.0}
    if hi == lo:
        return max(_base[hi] * 2.0, 5.0)
    score = _base[hi] + (2.0 if suited else 0.0)
    gap = _RANK_ORDER[hi] - _RANK_ORDER[lo] - 1
    score -= {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}.get(gap, 5.0)
    if gap <= 1 and _RANK_ORDER[hi] < _RANK_ORDER["Q"]:
        score += 1.0
    return score


def _build_pctl():
    scores = []
    for i, hi in enumerate(RANKS):
        for j in range(i + 1):
            lo = RANKS[j]
            if i == j:
                scores.extend([_chen_score(hi, lo, False)] * 6)
            else:
                scores.extend([_chen_score(hi, lo, True)] * 4)
                scores.extend([_chen_score(hi, lo, False)] * 12)
    scores.sort()
    return scores


_PCTL_SORTED = _build_pctl()
_N_COMBOS = len(_PCTL_SORTED)


def _hand_pctl(c1: str, c2: str) -> float:
    r1, r2 = c1[0], c2[0]
    hi, lo = (r1, r2) if _RANK_ORDER[r1] >= _RANK_ORDER[r2] else (r2, r1)
    chen = _chen_score(hi, lo, c1[1] == c2[1])
    return _bisect_left(_PCTL_SORTED, chen) / _N_COMBOS


def _mc_equity(hole, board, rest, n_opp, opp_floors, deadline):
    """MC win probability vs n_opp. opp_floors (per-opp Chen percentile) biases
    each opponent's sampled hand to sit at/above that strength floor."""
    start = time.time()
    wins = 0.0
    iters = 0
    need = 5 - len(board)
    n_rem = len(rest)
    floors = opp_floors or None
    time_limit = 0.45
    while time.time() - start < time_limit:
        if deadline is not None and iters >= _MC_MIN_ITERS and time.time() > deadline:
            break
        random.shuffle(rest)
        if floors is None:
            opp_hands = [rest[i * 2:i * 2 + 2] for i in range(n_opp)]
            idx = 2 * n_opp
        else:
            opp_hands = []
            idx = 0
            for f in floors:
                hand = rest[idx:idx + 2]
                idx += 2
                tries = 0
                while (f > 0.0 and tries < 3 and idx + 2 <= n_rem
                       and _hand_pctl(str(hand[0]), str(hand[1])) < f):
                    hand = rest[idx:idx + 2]
                    idx += 2
                    tries += 1
                opp_hands.append(hand)
        full_board = board + rest[idx:idx + need]
        my = evaluate(hole + full_board)
        opp = [evaluate(h + full_board) for h in opp_hands]
        best = max(my, *opp)
        if my == best:
            wins += 1.0 / (opp.count(best) + 1)
        iters += 1
    return wins / max(iters, 1)


# Per-archetype shift to the opponent range floor, applied only with a confident
# read. A maniac's aggression is bluffy (widen their range); a nit's is nutted.
_TAG_FLOOR_ADJ = {"maniac": -0.16, "station": -0.07, "nit": +0.07, "tag": +0.02}


def _street_raise_counts(gs: dict) -> tuple[int, int]:
    """(preflop raises, postflop raises) in THIS hand, blinds excluded."""
    al = gs["action_log"]
    annot = _reconstruct_streets(al)
    pre = post = 0
    for e, a in zip(al, annot):
        if e.get("action") in ("raise", "all_in"):
            if a["street"] == "preflop":
                pre += 1
            else:
                post += 1
    return pre, post


def _base_floor(gs: dict, street: str) -> float:
    """Opponent continuing-range floor (Chen percentile), before per-opp reads.

    The resilience core (ported from cfr_equity_v28): anyone still contesting a
    postflop pot holds a non-random range, and every raise / barrel / large bet
    narrows that range toward the nuts. Computing equity against this tightened
    range is exactly what lets us fold marginal hands to a multi-street barrel /
    jam instead of paying it off."""
    pre, post = _street_raise_counts(gs)
    floor = 0.40 + pre * 0.07 + post * 0.07
    owed, pot, stack = gs["amount_owed"], gs["pot"], gs["your_stack"]
    if owed > 0:
        if owed >= 0.5 * pot:
            floor += 0.05
        if owed >= 0.85 * stack:
            floor += 0.12          # near all-in: range is nuts-weighted
        elif owed >= 0.40 * stack:
            floor += 0.06
    if street == "preflop":
        floor -= 0.18              # preflop ranges are far wider
    return floor


def _equity(gs: dict, profiles: dict, deadline) -> float:
    my = list(map(Card, gs["your_cards"]))
    board = list(map(Card, gs["community_cards"]))
    rest = [c for c in ALL_CARDS if c not in my and c not in board]
    seat = gs["seat_to_act"]
    active = [p for p in gs["players"]
              if p["seat"] != seat and p["state"] in ("active", "all_in")]
    n_opp = max(len(active), 1)
    street = gs.get("street", "flop")
    base = _base_floor(gs, street)
    floors = []
    for p in active:
        tag, conf = _classify(profiles.get(p["bot_id"]) or {})
        adj = _TAG_FLOOR_ADJ.get(tag, 0.0) if conf >= 0.3 else 0.0
        floors.append(max(0.0, min(0.80, base + adj)))
    return _mc_equity(my, board, rest, n_opp, floors or None, deadline)


# ==========================================================================
# -  Opponent profiling (lean)                                                 -
# ==========================================================================

_BLINDS = frozenset(("small_blind", "big_blind"))
_AGGR = frozenset(("raise", "all_in", "bet"))
_STREETS = ("preflop", "flop", "turn", "river")
_MIN_HANDS_FOR_TAG = 8


def _reconstruct_streets(actions: list) -> list:
    """Annotate each action of a single hand with its street + aggression-before."""
    live = {a.get("seat") for a in actions if a.get("seat") is not None}
    folded: set = set()
    all_in: set = set()
    need = set(live)
    street_idx = 0
    aggr = 0
    out = []
    for e in actions:
        act = str(e.get("action", "")).lower().strip()
        if act in _BLINDS:
            out.append({"street": "preflop", "aggr_before": 0})
            continue
        out.append({"street": _STREETS[street_idx], "aggr_before": aggr})
        seat = e.get("seat")
        need.discard(seat)
        if act == "fold":
            folded.add(seat)
        elif act == "all_in":
            all_in.add(seat)
            aggr += 1
            need = (live - folded - all_in) - {seat}
        elif act in _AGGR:
            aggr += 1
            need = (live - folded - all_in) - {seat}
        if not need:
            if street_idx < len(_STREETS) - 1:
                street_idx += 1
            aggr = 0
            need = live - folded - all_in
    return out


def _iter_hands(match_log: list):
    cur = object()
    bucket: list = []
    for e in match_log:
        hid = e.get("hand_num")
        if hid != cur and bucket:
            yield bucket
            bucket = []
        cur = hid
        bucket.append(e)
    if bucket:
        yield bucket


def _build_profiles(match_log: list, my_id) -> dict:
    """{bot_id: counters} from the rolling cross-hand log (hero excluded)."""
    prof: dict = {}
    for entries in _iter_hands(match_log):
        annot = _reconstruct_streets(entries)
        vol: set = set()
        raised: set = set()
        present: set = set()
        for e, a in zip(entries, annot):
            bid = e.get("bot_id")
            if bid is None or bid == my_id:
                continue
            act = str(e.get("action", "")).lower().strip()
            if act in _BLINDS:
                continue
            c = prof.setdefault(bid, {"hands": 0, "vpip": 0, "pfr": 0,
                                      "n_aggr": 0, "n_call": 0,
                                      "pf_faced": 0, "pf_fold": 0,
                                      "flop_faced": 0, "flop_fold": 0,
                                      "turn_faced": 0, "turn_fold": 0,
                                      "river_faced": 0, "river_call": 0,
                                      "river_fold": 0})
            present.add(bid)
            st = a["street"]
            faced = a["aggr_before"] >= 1
            cont = act in ("fold", "call", "raise", "all_in")
            if st == "preflop":
                if act in ("call", "raise", "all_in"):
                    vol.add(bid)
                if act in ("raise", "all_in"):
                    raised.add(bid)
                if faced and cont:                 # facing a raise/3-bet preflop
                    c["pf_faced"] += 1
                    if act == "fold":
                        c["pf_fold"] += 1
            elif faced and cont:                   # facing a bet/raise postflop
                c[st + "_faced"] += 1
                if st == "river" and act == "call":
                    c["river_call"] += 1
                if act == "fold":
                    c[st + "_fold"] += 1
            if act in ("raise", "all_in"):
                c["n_aggr"] += 1
            elif act == "call":
                c["n_call"] += 1
        for bid in present:
            prof[bid]["hands"] += 1
            if bid in vol:
                prof[bid]["vpip"] += 1
            if bid in raised:
                prof[bid]["pfr"] += 1
    return prof


def _classify(c: dict) -> tuple[str, float]:
    """(archetype, confidence) in {nit, station, maniac, tag, normal, unknown}."""
    hands = c.get("hands", 0)
    if hands < _MIN_HANDS_FOR_TAG:
        return "unknown", 0.0
    vpip = c["vpip"] / hands
    pfr = c["pfr"] / hands
    af = c["n_aggr"] / c["n_call"] if c["n_call"] > 0 else (1.4 if c["n_aggr"] == 0 else 3.0)
    rc = c["river_call"] / c["river_faced"] if c["river_faced"] > 0 else 0.5
    conf = min(0.9, 0.25 + hands / 50.0)
    if vpip < 0.16 and pfr < 0.12:
        tag = "nit"
    elif vpip > 0.42 and (pfr > 0.30 or af > 2.5):
        tag = "maniac"
    elif vpip > 0.30 and af < 1.0 and rc > 0.55:
        tag = "station"
    elif 0.18 <= vpip <= 0.30 and pfr >= 0.14 and af >= 1.3:
        tag = "tag"
    else:
        tag = "normal"
    return tag, conf


# Empirical-Bayes priors (mean, shrinkage target) for each leak rate. Sparse
# reads stay near the population prior; confidence (the weight) rises with samples.
_LEAK_PRI = {
    "pf_fold": (0.55, 6), "flop_fold": (0.45, 6), "turn_fold": (0.45, 5),
    "river_fold": (0.45, 5), "river_call": (0.45, 5),
}


def _rate(succ: int, opp: int, key: str) -> tuple[float, float]:
    """Shrunk leak rate + confidence weight (0..1)."""
    prior, target = _LEAK_PRI[key]
    if opp <= 0:
        return prior, 0.0
    w = min(1.0, opp / target)
    return prior * (1.0 - w) + (succ / opp) * w, w


def _street_fold_key(street: str) -> str | None:
    return {"flop": "flop_fold", "turn": "turn_fold", "river": "river_fold"}.get(street)


def _exploit(gs: dict, profiles: dict) -> tuple[float, float, bool, float]:
    """Measured, per-street read-derived shifts:
        (callfold_shift, bet_shift, allow_bluff, value_upsize)

    callfold_shift > 0 - call lighter when facing a bet; < 0 - fold more.
    bet_shift      > 0 - bet/bluff more when checked to (scales with how often the
                         stickiest live opponent folds on THIS street).
    allow_bluff         - may fire bluffs at all (off vs a sticky/station field).
    value_upsize        - extra pot fraction on value bets vs callers.
    """
    seat = gs["seat_to_act"]
    street = gs.get("street", "flop")
    callfold = 0.0
    bet = 0.0
    allow_bluff = True
    upsize = 0.0

    live = [p for p in gs["players"]
            if p["seat"] != seat and p.get("state") in ("active", "all_in")]

    # --- Betting side: barrel into a folding field, value-size vs callers -------
    fkey = _street_fold_key(street)
    if fkey and live:
        fold_rates = []          # (rate, conf) of each live opp folding this street
        call_rates = []          # river-call propensity (station detector)
        for p in live:
            c = profiles.get(p["bot_id"]) or {}
            faced = c.get(fkey.replace("_fold", "_faced"), 0)
            fr, fw = _rate(c.get(fkey, 0), faced, fkey)
            fold_rates.append((fr, fw))
            rc, rw = _rate(c.get("river_call", 0), c.get("river_faced", 0), "river_call")
            call_rates.append((rc, rw))
        # The stickiest opponent governs our fold equity (anyone who calls kills a bluff).
        stick_fold, stick_w = min(fold_rates, key=lambda x: x[0])
        bet = float(np.clip((stick_fold - 0.45) * 0.9, -0.10, 0.35)) * stick_w
        if stick_fold < 0.30 and stick_w >= 0.5:
            allow_bluff = False              # someone here never folds - don't bluff
        # Value up-sizing vs confident callers (inelastic stations).
        best_call, best_w = max(call_rates, key=lambda x: x[0])
        if best_w >= 0.4 and best_call > 0.55:
            upsize += min(0.5, (best_call - 0.45) * 1.6)
            allow_bluff = False

    # --- Facing side: read the aggressor's honesty ----------------------------
    last_seat = -1
    for e in gs["action_log"]:
        if e.get("action") in ("raise", "all_in"):
            last_seat = e["seat"]
    if last_seat >= 0:
        bid = next((p["bot_id"] for p in gs["players"] if p["seat"] == last_seat), None)
        c = profiles.get(bid) or {}
        if c.get("hands", 0) >= _MIN_HANDS_FOR_TAG:
            af = c["n_aggr"] / c["n_call"] if c["n_call"] > 0 else (
                1.4 if c["n_aggr"] == 0 else 3.0)
            tag, conf = _classify(c)
            # Bluffy aggressor (high AF / maniac) -> call lighter; honest, value-
            # heavy aggressor (passive AF / nit / tag) -> fold more.
            callfold = float(np.clip((af - 1.6) * 0.07, -0.04, 0.16))
            if tag in ("nit", "tag"):
                callfold = min(callfold, -0.10)
            elif tag == "maniac":
                callfold = max(callfold, 0.14)
            callfold *= conf
    return callfold, bet, allow_bluff, upsize


# ==========================================================================
# -  Decision engine                                                           -
# ==========================================================================

# Equity needed to bet for value when checked to (heads-up baseline; relaxed
# in position, tightened multiway).
_VALUE_BET_EQ = {"preflop": 0.55, "flop": 0.56, "turn": 0.60, "river": 0.64}
# Equity above which we never just check/call - we bet/raise for value.
_STRONG_EQ = 0.80
# Max fraction of the effective stack we'll commit on one street without the
# equity to back it (risk gate).
_RISK_COMMIT_FRAC = 0.30


def _eff_stack(gs: dict) -> int:
    seat = gs["seat_to_act"]
    mine = gs["your_stack"] + gs["your_bet_this_street"]
    opp = max((p["stack"] + p.get("bet_this_street", 0)
               for p in gs["players"]
               if p["seat"] != seat and p.get("state") in ("active", "all_in")),
              default=mine)
    return min(mine, opp)


def _best_bet_action(legal: list, gs: dict, target_frac: float, net_probs):
    """Pick the legal bet abstract action whose pot-fraction is closest to
    target_frac; let the net break ties only among similarly-sized options."""
    bets = [a for a in legal if a in _BET_ORDER]
    if not bets:
        return None
    best = min(bets, key=lambda a: abs(_ACTION_POT_FRAC[a] - target_frac))
    if net_probs is not None:
        ref = _ACTION_POT_FRAC[best]
        near = [a for a in bets if abs(_ACTION_POT_FRAC[a] - ref) <= 0.18]
        if len(near) > 1:
            best = max(near, key=lambda a: net_probs[a])
    return best


def _raise_to(gs: dict, total: int, jitter: float = 0.06) -> dict:
    """Build a legal raise to ~`total` (a total bet amount), or all-in if it
    reaches the stack."""
    min_r = gs["min_raise_to"]
    all_tot = gs["your_bet_this_street"] + gs["your_stack"]
    total = max(total, min_r)
    j = int(total * random.uniform(1 - jitter, 1 + jitter))
    j = max(j, min_r)
    if j >= all_tot:
        return {"action": "all_in"}
    return {"action": "raise", "amount": j}


# Position-based first-in opening thresholds (Chen percentile). Keyed by offset
# from the dealer (0 = button, 1 = SB, 2 = BB, -). Looser late, tighter early.
_OPEN_THR = {0: 0.55, 1: 0.60, 3: 0.84, 4: 0.80, 5: 0.72}   # 2 (BB) handled inline
_PREMIUM = {"AA", "KK", "QQ", "AKs", "AKo"}
_STRONG_PF = {"JJ", "TT", "AQs", "AQo", "AJs", "KQs"}


def _combo(gs: dict) -> tuple[str, float, bool]:
    """(canonical combo string, Chen percentile, is_pair)."""
    c1, c2 = gs["your_cards"]
    r1, r2 = c1[0], c2[0]
    hi, lo = (r1, r2) if _RANK_ORDER[r1] >= _RANK_ORDER[r2] else (r2, r1)
    pair = hi == lo
    suited = c1[1] == c2[1]
    combo = hi + lo + ("" if pair else ("s" if suited else "o"))
    return combo, _hand_pctl(c1, c2), pair


def _preflop_decide(gs, legal, profiles, equity) -> dict:
    combo, strength, pair = _combo(gs)
    hero_off, n = _hero_position(gs)
    owed = gs["amount_owed"]
    pot = gs["pot"]
    n_raises = _derive_n_raises_this_street(gs["action_log"], len(gs["players"]))
    callfold, _, _, _ = _exploit(gs, profiles)
    bb = _BIG_BLIND
    can_raise = any(a in _BET_ORDER or a == _ALL_IN for a in legal)

    # -- Unopened pot (no raise yet - possibly limpers) --------------------
    if n_raises == 0:
        thr = _OPEN_THR.get(hero_off, 0.78)
        if hero_off == 2:        # BB: only raise the strongest vs limpers
            thr = 0.86
        open_it = combo in _PREMIUM or strength >= thr
        if open_it and can_raise:
            limpers = max(0, (pot - (_SMALL_BLIND + _BIG_BLIND)) // bb)
            return _raise_to(gs, (3 + limpers) * bb)
        if owed <= 0:
            return {"action": "check"}
        # Cheap completion from the SB with a semi-playable hand; else fold.
        if owed <= bb and strength >= 0.45:
            return {"action": "call"}
        return {"action": "fold"}

    # -- Facing a raise / 3-bet --------------------------------------------
    pot_odds = owed / (pot + owed) if owed > 0 else 0.0
    # Value 3-bet/4-bet core: always reraise the premiums for value.
    if combo in _PREMIUM and can_raise:
        # vs a 3-bet+ jam the nut hands; otherwise reraise ~3x.
        if n_raises >= 2 and combo in ("AA", "KK"):
            return {"action": "all_in"}
        return _raise_to(gs, 3 * gs["current_bet"])

    # Calling range: strong broadways / pairs vs a single raise at a fair price.
    call_thr = 0.70 if n_raises == 1 else 0.90
    if combo in _STRONG_PF:
        call_thr = min(call_thr, 0.55)
    # Pocket pairs get a set-mining discount when stacks are deep enough.
    if pair and _eff_stack(gs) >= 15 * owed:
        call_thr = min(call_thr, 0.50)
    # Out of position the hand realises equity worse and gets barrelled off more,
    # so demand a stronger continuing range when not last to act.
    if not _in_position(gs):
        call_thr += 0.07
    edge = equity - pot_odds + callfold
    if (strength >= call_thr or edge > 0.10) and pot_odds <= 0.5:
        return {"action": "call"}
    return {"action": "fold"} if owed > 0 else {"action": "check"}


def _net_distribution(gs: dict, legal: list):
    if _GTO_LAYERS is None:
        return None, None
    try:
        full = _numpy_forward(_GTO_LAYERS, _build_feature_vector(gs))
    except Exception:
        return None, None
    return full, _mask_probs(full, legal)


_HT_RANK = {"High Card": 0, "Pair": 1, "Two Pair": 2, "Trips": 3, "Straight": 4,
            "Flush": 5, "Full House": 6, "Quads": 7, "Straight Flush": 8}


def _contributes(cards, board, cat) -> bool:
    """Do the hero's hole cards actually improve on what the board already shows?

    On paired / double-paired boards the equity sim credits hero with the board's
    pair(s) even when the hole cards add nothing (e.g. Q6 on KKTT). This catches
    that: a holding only counts as 'made' if a hole card pairs the board, it is a
    pocket pair, or it contributes to a flush/straight."""
    hr = [c[0] for c in cards]
    hs = [c[1] for c in cards]
    branks = [c[0] for c in board]
    bsuits = [c[1] for c in board]
    if hr[0] == hr[1]:                                  # pocket pair (pair or set)
        return True
    if any(r in branks for r in hr):                   # pairs / trips the board
        return True
    for s in set(hs):                                  # contributes to a flush
        if bsuits.count(s) + hs.count(s) >= 5 and bsuits.count(s) < 5:
            return True
    if cat >= 4:                                       # straight+ : assume hole used
        return True
    return False


def _made_class(gs) -> int:
    """Board-relative made-hand strength of the hero's holding:
        0 = air / hero only plays the board (contributes nothing)
        1 = a real but weak pair (middle/bottom pair, weak top pair)
        2 = strong one pair (overpair, or top pair with a big kicker)
        3 = two pair or better, with a real hole-card contribution

    The board-relative view (not raw equity) is what lets us fold to barrels:
    ace-high on QTT or Q6 on KKTT both return 0, not a 'pair'/'two pair'."""
    board = gs["community_cards"]
    if len(board) < 3:
        return 3  # preflop: not gated here
    try:
        cat = _HT_RANK.get(handtype(evaluate([Card(c) for c in gs["your_cards"]]
                                             + [Card(c) for c in board])), 0)
    except Exception:
        return 3
    cards = gs["your_cards"]
    if not _contributes(cards, board, cat):
        return 0                                       # hero only plays the board
    if cat >= 2:
        return 3
    branks = [_RANK_ORDER[c[0]] for c in board]
    hr = [_RANK_ORDER[c[0]] for c in cards]
    topb = max(branks)
    pocket = hr[0] == hr[1]
    if pocket:
        return 2 if hr[0] > topb else 1                # overpair vs underpair
    paired = [r for r in hr if r in branks]
    prank = max(paired) if paired else 0
    kick = max((r for r in hr if r != prank), default=0)
    if prank == topb:
        return 2 if kick >= _RANK_ORDER["Q"] else 1    # top pair good / weak kicker
    return 1                                            # middle / bottom pair


def _has_initiative(gs) -> bool:
    """Hero made the most recent aggressive action in the hand (is the bettor /
    preflop raiser whose initiative carries to a checked-to street)."""
    seat = gs["seat_to_act"]
    for e in reversed(gs["action_log"]):
        if e.get("action") in ("raise", "all_in"):
            return e["seat"] == seat
    return False


def _board_wet(board) -> bool:
    """Coarse wetness: a flush-draw (3+ one suit) or straight-y texture (3 ranks
    within a 4-span). Dry boards favour small range c-bets; wet boards polarize."""
    if len(board) < 3:
        return False
    suits = [c[1] for c in board]
    if max(suits.count(s) for s in set(suits)) >= 3:
        return True
    ranks = sorted(set(_RANK_ORDER[c[0]] for c in board))
    return any(sum(1 for r in ranks if lo <= r <= lo + 4) >= 3 for lo in ranks)


def _choose(gs, legal, equity, profiles, deadline) -> dict:
    owed = gs["amount_owed"]
    pot = gs["pot"]
    street = gs.get("street", "flop")
    n_opp = _n_active_opponents(gs)
    ip = _in_position(gs)
    callfold, bet_shift, allow_bluff, upsize = _exploit(gs, profiles)

    full_probs, net_arr = _net_distribution(gs, legal)

    # Multiway penalty: each extra opponent demands more equity to continue.
    mw = max(0.0, (n_opp - 1) * 0.05)

    # -- Facing a bet ------------------------------------------------------
    if owed > 0:
        stack = gs["your_stack"]
        pot_odds = owed / (pot + owed)

        # Made-hand discipline: do NOT pay off a real barrel / jam with air or a
        # board-only pair. This is the single biggest bust-saver vs strong bots -
        # equity over-credits board pairs, so gate on board-relative made strength.
        _, post_aggr = _street_raise_counts(gs)
        facing_jam = any(p.get("is_all_in") and p["seat"] != gs["seat_to_act"]
                         for p in gs["players"]) or owed >= 0.95 * stack
        # pot_odds >= 0.28  <=>  a bet of >= ~0.4x the pot (a real barrel, not a stab)
        big_bet = pot_odds >= 0.28 or facing_jam
        is_maniac = callfold > 0.10   # _exploit loosens us vs confident maniacs
        if (street in ("turn", "river") and big_bet and post_aggr >= 1
                and not is_maniac and _made_class(gs) <= 1
                and _idx_of(legal, _FOLD) is not None):
            # weak/board-only made hand vs a sized barrel: fold unless the price
            # is tiny (drawing-odds territory we'd take anyway).
            if pot_odds > 0.12:
                return {"action": "fold"}

        # Value raise: equity is now measured vs the aggression-tightened range,
        # so >=0.80 is a genuinely strong hand. Size ~ pot, upsized vs stations.
        if equity >= 0.80 and n_opp <= 2:
            tgt = 1.0 + upsize
            a = _best_bet_action(legal, gs, tgt, full_probs)
            if a is not None and equity >= 0.84:
                return _abstract_to_raw(a, gs)
            if a is not None and net_arr is not None and net_arr[_idx_of(legal, a)] > 0.20:
                return _abstract_to_raw(a, gs)

        # Required equity to continue = pot odds + margins that ESCALATE with the
        # bet size (cfr_equity_v28's model). Big bets / near-jams demand much more
        # equity, so marginal hands fold to barrels instead of paying them off.
        required = pot_odds + 0.04 + (n_opp - 1) * 0.02
        if owed >= 0.40 * stack:
            required += 0.04
        if owed >= 0.75 * stack:
            required += 0.08
        required -= callfold       # read-aware: call lighter vs maniacs, tighter vs nits
        if net_arr is not None:    # the net may only TIGHTEN (ask for more)
            fi = _idx_of(legal, _FOLD)
            if fi is not None:
                required += 0.05 * float(net_arr[fi])
        if equity >= required:
            return {"action": "call"}
        return {"action": "fold"} if _idx_of(legal, _FOLD) is not None else {"action": "check"}

    # -- Checked to (owed == 0): bet for value / bluff, or check -----------
    vthr = _VALUE_BET_EQ.get(street, 0.58) + mw - (0.04 if ip else 0.0)

    # Don't value-bet into a prior aggressor (or postflop) with only a board pair
    # or air - equity over-credits it and we just bloat a pot we can't continue.
    pre_r, post_r = _street_raise_counts(gs)
    weak_made = _made_class(gs) == 0
    if equity >= vthr and not (weak_made and (post_r >= 1 or pre_r >= 2)):
        frac = 0.5 + max(0.0, (equity - vthr)) * 1.3 + upsize
        frac = min(frac, 1.5)
        a = _best_bet_action(legal, gs, frac, full_probs)
        if a is not None:
            return _abstract_to_raw(a, gs)
        return {"action": "check"}

    # Bluff conservatively. Fire only when (a) we hold a genuine draw, so a fold
    # is a free win and a call still has outs, or our range/initiative + fold
    # equity makes a balanced bet +EV. allow_bluff is already off vs stations /
    # maniacs, and we never fire pure air into 3+ ways.
    if allow_bluff and n_opp <= 2:
        init = _has_initiative(gs)
        wet = _board_wet(gs["community_cards"])
        freq = 0.0
        size = 0.5
        if street in ("flop", "turn"):
            if init and 0.22 <= equity < vthr:
                # c-bet / barrel the non-value part of our range. Small on dry
                # boards (range bet), bigger & rarer on wet boards (polarized).
                if street == "flop":
                    freq, size = (0.55, 0.34) if not wet else (0.34, 0.6)
                else:                                   # turn: barrel more selectively
                    freq, size = (0.36, 0.5) if not wet else (0.22, 0.66)
                freq += bet_shift
            elif ip and equity >= 0.40:                  # pure semi-bluff w/ a draw, no initiative
                freq, size = 0.20 + bet_shift, 0.5
        elif street == "river" and n_opp == 1:
            # Polarized river: bluff missed hands when we held initiative, so our
            # value bets get paid (balance). Never bluff a hand with showdown value.
            if init and _made_class(gs) == 0 and equity < 0.30:
                freq, size = 0.32 + bet_shift, 0.66
        if freq > 0 and random.random() < freq:
            a = _best_bet_action(legal, gs, size, full_probs)
            if a is not None:
                return _abstract_to_raw(a, gs)
    return {"action": "check"}


def _risk_gate(action: dict, gs: dict, equity: float) -> dict:
    """Never put a large fraction of the effective stack in without the equity to
    back it. Downgrades over-commits to a call (facing a bet) or check."""
    if action.get("action") not in ("raise", "all_in"):
        return action
    eff = max(_eff_stack(gs), 1)
    my_bet = gs["your_bet_this_street"]
    if action["action"] == "all_in":
        commit = gs["your_stack"]
    else:
        commit = action["amount"] - my_bet
    frac = commit / eff
    if frac <= _RISK_COMMIT_FRAC:
        return action
    # Preflop multiway MC understates premium equity, and the preflop chart has
    # already gated to a strong range — only veto a clearly-bad shove (<0.50).
    if gs.get("street") == "preflop":
        return action if equity >= 0.50 else (
            {"action": "call"} if gs["amount_owed"] > 0 else {"action": "check"})
    # Postflop: commitment escalates the equity bar with size; all-in needs the most.
    floor = 0.62 + 0.22 * min(1.0, frac)
    if equity >= floor:
        return action
    # Pull back.
    if gs["amount_owed"] > 0:
        return {"action": "call"}
    return {"action": "check"}


# ==========================================================================
# -  Short-stack push/fold (Tier 4)                                            -
# ==========================================================================
# Below ~12bb effective the 100bb ranges/blueprint no longer apply: the correct
# game is jam-or-fold (open-shove your range, call-off by equity). Sized opens
# just commit you and get re-shoved on. Thresholds are Chen-percentile shove
# ranges by position, widening as the stack shortens.

_SHOVE_BASE = {0: 0.50, 5: 0.58, 4: 0.68, 3: 0.74, 1: 0.55, 2: 0.66}


def _shortstack_preflop(gs: dict, equity: float, legal: list) -> dict:
    combo, strength, _ = _combo(gs)
    hero_off, _ = _hero_position(gs)
    owed, pot = gs["amount_owed"], gs["pot"]
    eff_bb = _eff_stack(gs) / _BIG_BLIND
    n_raises = _preflop_raise_count(gs)
    can_jam = _ALL_IN in legal

    if n_raises == 0:
        # Unopened: jam or fold (no limping when short).
        thr = _SHOVE_BASE.get(hero_off, 0.70) - max(0.0, (12.0 - eff_bb)) * 0.02
        if can_jam and (strength >= thr or combo in _PREMIUM):
            return {"action": "all_in"}
        return {"action": "check"} if owed == 0 else {"action": "fold"}

    # Facing a raise/shove: get premiums in; otherwise call off by equity vs the
    # (aggression-tightened) range at the price offered.
    if combo in _PREMIUM:
        if can_jam and gs["your_stack"] > owed:
            return {"action": "all_in"}
        return {"action": "call"}
    pot_odds = owed / (pot + owed) if owed > 0 else 0.0
    if strength >= 0.80 or equity >= pot_odds + 0.02:
        return {"action": "call"}
    return {"action": "fold"} if owed > 0 else {"action": "check"}


# ==========================================================================
# -  Entry point                                                               -
# ==========================================================================

def _action_seed(gs: dict) -> int:
    hid = gs.get("hand_id", "")
    try:
        hand_num = int(hid.rsplit("_h", 1)[-1])
    except (ValueError, IndexError):
        hand_num = 0
    return (hand_num * 10_000 + len(gs.get("action_log", [])) * 10
            + gs.get("seat_to_act", 0)) % (2 ** 31)


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        return {"action": "check"}

    global _DEADLINE
    _DEADLINE = time.time() + _TIME_BUDGET
    seed = _action_seed(game_state)
    random.seed(seed)
    np.random.seed(seed)

    try:
        my_id = next((p["bot_id"] for p in game_state["players"]
                      if p["seat"] == game_state["seat_to_act"]), None)
        profiles = _build_profiles(game_state.get("match_action_log", []), my_id)

        n_raises = _derive_n_raises_this_street(
            game_state["action_log"], len(game_state["players"]))
        legal = _legal_actions(game_state, n_raises)

        equity = _equity(game_state, profiles, _DEADLINE)

        if game_state.get("street") == "preflop":
            if _eff_stack(game_state) / _BIG_BLIND <= 12.0:
                action = _shortstack_preflop(game_state, equity, legal)   # Tier 4
            else:
                # Tier 1: solved blueprint when in abstraction, else heuristic chart.
                action = (_pf_table_decide(game_state, legal)
                          or _preflop_decide(game_state, legal, profiles, equity))
        else:
            action = _choose(game_state, legal, equity, profiles, _DEADLINE)
        action = _risk_gate(action, game_state, equity)
        return action
    except Exception:
        # Absolute resilience floor: never crash. Check if free, else fold.
        if game_state.get("can_check") or game_state.get("amount_owed", 1) == 0:
            return {"action": "check"}
        return {"action": "fold"}
