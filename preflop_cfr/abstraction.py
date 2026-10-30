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
