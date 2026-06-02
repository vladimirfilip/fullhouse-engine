"""
Info-set key encoding and action-amount translation for preflop CFR.

The canonical info-set key is a compact **betting-context** tuple:
    (hero_pos, n_raises, facing, n_live, hero_committed, bucket)

encoded as a UTF-8 string, then hashed to a 64-bit signed integer via FNV-1a.

This replaces the old last-N-actions recency truncation, which aliased
genuinely different spots (it forgot how many players folded and who was still
live), producing the SB-90% / limp-heavy pathologies.  The new key is an
imperfect-recall abstraction on the decision-relevant public state instead:
position, how many raises have gone in, the size hero faces, how many players
remain, and whether hero has already voluntarily invested.

This module is the SINGLE SOURCE OF TRUTH for this encoding.
bot.py contains a mirrored copy (_preflop_bucket, _preflop_infoset_key,
_pf_facing_bucket) — any change here must be reflected there identically.
"""

from __future__ import annotations

import struct

from preflop_cfr import config


# ── FNV-1a 64-bit ─────────────────────────────────────────────────────────────
_FNV_OFFSET = 14695981039346656037  # 64-bit
_FNV_PRIME  = 1099511628211


def fnv1a_64(data: bytes) -> int:
    """FNV-1a 64-bit hash, returned as a signed Python int."""
    h = _FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    # interpret as signed int64
    return struct.unpack("q", struct.pack("Q", h))[0]


# ── Action-amount translation ─────────────────────────────────────────────────
# Convert a real chip amount (raise-to total) into the nearest abstract action
# index.  Used in bot.py to translate observed opponent bets into the history
# sequence the solver was trained on.

def amount_to_abstract(raise_to: int, pot: int, current_bet: int,
                        your_bet_this_street: int) -> int:
    """
    Map a 'raise to X chips' amount to the closest *sized* PREFLOP_ACTIONS raise
    index.  Returns CHECK_CALL if the amount is <= current_bet (call/check).

    NOTE: this never returns ALL_IN — it only ranks the sized-raise fractions and
    has no stack information to detect a shove.  The solver tree records ALL_IN as
    a distinct action (game.legal_actions/apply_action), so callers replaying an
    action log MUST map an "all_in" action straight to config.ALL_IN and only
    route genuine "raise" actions through this function.  The engine normalises a
    full-stack "raise" to an "all_in" action (engine/game.py:_validate), so the
    action label alone is sufficient to tell them apart.  bot.py's mirror
    (_pf_amount_to_abstract / _preflop_infoset_key) follows the same rule.
    """
    eff_pot = pot + max(0, current_bet - your_bet_this_street)
    if eff_pot <= 0:
        return config.CHECK_CALL

    raise_size = raise_to - current_bet   # extra chips above current bet
    if raise_size <= 0:
        return config.CHECK_CALL

    # pot-fraction of raise (relative to effective pot)
    frac = raise_size / eff_pot

    # fractions for the active raise actions (same as _abstract_to_raw in bot.py)
    _FRACS = [
        (config.BET_0_27X_POT, 0.27),
        (config.BET_THIRD_POT, 0.333),
        (config.BET_HALF_POT,  0.50),
        (config.BET_FULL_POT,  1.00),
        (config.BET_1_72X_POT, 1.72),
        (config.BET_2X_POT,    2.00),
    ]

    best_idx, best_dist = config.CHECK_CALL, float("inf")
    for action_idx, target_frac in _FRACS:
        if action_idx not in config.PREFLOP_ACTIONS:
            continue
        dist = abs(frac - target_frac)
        if dist < best_dist:
            best_dist = dist
            best_idx  = action_idx
    return best_idx


# ── Facing-size bucket ────────────────────────────────────────────────────────
# Coarse bucket of the bet hero must call, relative to the pot.  Mirrored in
# bot.py (_pf_facing_bucket).  Boundaries are deliberately coarse so tiny
# pot-accounting differences between solver and bot rarely cross a boundary.

def facing_bucket(owed: int, pot: int) -> int:
    """0 = nothing owed (can check); 1 = ≤0.40·pot; 2 = ≤0.85·pot; 3 = >0.85."""
    if owed <= 0:
        return 0
    r = owed / pot if pot > 0 else 0.0
    if r <= 0.40:
        return 1
    if r <= 0.85:
        return 2
    return 3


# ── Info-set key ──────────────────────────────────────────────────────────────

def infoset_key(hero_pos: int, n_raises: int, facing: int, n_live: int,
                hero_committed: int, last_aggr_rel: int, bucket: int) -> int:
    """
    Encode a preflop info-set as a 64-bit signed int from its betting context.

    hero_pos:       seat index relative to dealer (0=dealer/BTN in 6-max).
    n_raises:       raises so far this hand, capped at 3 (4-bet+ collapse).
    facing:         facing_bucket(owed, pot) — size hero faces (0..3).
    n_live:         players not yet folded (2..6).
    hero_committed: 1 if hero has voluntarily invested beyond the blind, else 0.
    last_aggr_rel:  (aggressor_seat - hero_seat) % n_players, in 1..5; 6 = no
                    raise yet (unopened/limped).  Lets hero's range respond to
                    WHO raised (e.g. a UTG open vs a CO open), which the other
                    fields can't distinguish.
    bucket:         0..168 hand bucket from cards.hand_to_bucket.

    SINGLE SOURCE OF TRUTH — bot.py._preflop_infoset_key mirrors this byte-for-byte.
    """
    nr = n_raises if n_raises < 3 else 3
    raw = (f"{hero_pos}|{nr}|{facing}|{n_live}|{hero_committed}"
           f"|{last_aggr_rel}|{bucket}")
    return fnv1a_64(raw.encode())
