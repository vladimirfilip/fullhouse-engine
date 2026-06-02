"""
Generate a *rudimentary* preflop strategy table from known 6-max 100bb GTO charts.

This is NOT a CFR solve.  It enumerates the full reachable preflop betting tree
(deterministic given the betting abstraction — cards don't affect legal_actions)
and, at every decision node × 169 hand buckets, writes a chart-derived action
distribution.  The charts (open ranges, 3-bet ranges, premium core) are taken
verbatim from bots/vlad/bot.py so the table agrees with the shipped heuristic.

The output is written through the canonical preflop_cfr.export.export_strategy,
so the .npz format, keys (FNV-1a over `pos|history|bucket`), and metadata are
guaranteed identical to a real solve — bot.py can load it unchanged.

Run:
    python -m preflop_cfr.gen_rudimentary
"""
from __future__ import annotations

import os
import sys

import numpy as np

from preflop_cfr import config
from preflop_cfr.cards import BUCKET_INFO, RANKS
from preflop_cfr.cfr import _infoset_key as _canon_key
from preflop_cfr.export import export_strategy
from preflop_cfr.game import (
    make_initial_state, is_terminal, legal_actions, apply_action,
)

# ── Pull the canonical GTO charts straight from the shipped bot ───────────────
_BOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "bots", "vlad")
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)
import bot as _bot  # noqa: E402  (the production bot module)

_OPEN_RANGE     = _bot._OPEN_RANGE
_THREEBET_RANGE = _bot._THREEBET_RANGE
_PREMIUM_3BET   = _bot._PREMIUM_3BET
_chen_score     = _bot._chen_score
_pos_category   = _bot._pos_category

FOLD, CC, THIRD, FULL, ALLIN = (
    config.FOLD, config.CHECK_CALL, config.BET_THIRD_POT,
    config.BET_FULL_POT, config.ALL_IN,
)

# ── Per-bucket hand descriptors (label / chen / pair / suited) ────────────────
def _bucket_label(hi: int, lo: int, suited: bool) -> str:
    hc, lc = RANKS[hi], RANKS[lo]
    if hi == lo:
        return hc + lc
    return f"{hc}{lc}{'s' if suited else 'o'}"


_HAND: list[dict] = []
for _b in range(169):
    _hi, _lo, _su = BUCKET_INFO[_b]
    _hc, _lc = RANKS[_hi], RANKS[_lo]
    _HAND.append({
        "label":  _bucket_label(_hi, _lo, _su),
        "chen":   _chen_score(_hc, _lc, _su),
        "pair":   _hi == _lo,
        "suited": _su,
    })

# BB has no explicit open chart; iso-raise threshold fallback (mirror bot.py).
_OPEN_THRESH_BB = 8.0


def _first_legal(legal: set, prefer: tuple) -> int | None:
    for a in prefer:
        if a in legal:
            return a
    return None


def _strategy(pos: str, n_raises: int, bucket: int, legal_set: frozenset) -> np.ndarray:
    """Chart-derived action distribution (length-9) for one info set.

    Mirrors the deep-stack (>35bb) branches of bot.py `_preflop_decide`; the
    table is only consulted at ~100bb so the short-stack jam logic never applies.
    """
    h = _HAND[bucket]
    label, chen, pair, suited = h["label"], h["chen"], h["pair"], h["suited"]

    v = np.zeros(config.N_ACTIONS, dtype=np.float64)
    can_fold = FOLD in legal_set
    open_act = _first_legal(legal_set, (THIRD, FULL, ALLIN))
    rer_act  = _first_legal(legal_set, (FULL, THIRD, ALLIN))

    # ── Unopened pot (RFI / BB option / iso vs limpers) ──────────────────────
    if n_raises == 0:
        if pos in _OPEN_RANGE:
            should_open = label in _OPEN_RANGE[pos]
        else:                                   # BB iso-raise fallback
            should_open = chen >= _OPEN_THRESH_BB
        if should_open and open_act is not None:
            v[open_act] += 1.0
        elif not can_fold:                      # owed == 0 → check the option
            v[CC] += 1.0
        else:
            v[FOLD] += 1.0
        return _finish(v, legal_set)

    facing_3bet  = n_raises >= 2
    threebet_set = _THREEBET_RANGE.get(pos, _PREMIUM_3BET)

    # ── Facing a single raise ────────────────────────────────────────────────
    if not facing_3bet:
        if label in _PREMIUM_3BET and rer_act is not None:
            v[rer_act] += 0.85          # value 3-bet, occasional flat-trap
            v[CC]      += 0.15
        elif label in threebet_set and rer_act is not None:
            v[rer_act] += 0.55          # 3-bet bluff region
            v[FOLD]    += 0.30
            v[CC]      += 0.15
        else:
            call_thr = 9.5 if pos in ("CO", "BTN", "BB") else 10.5
            set_mine = pair or (suited and chen >= 7.0)
            if chen >= call_thr:
                v[CC] += 1.0
            elif set_mine:
                v[CC]   += 0.55         # set-mine / flat at a price
                v[FOLD] += 0.45
            elif not can_fold:
                v[CC] += 1.0
            else:
                v[FOLD] += 1.0
        return _finish(v, legal_set)

    # ── Facing a 3-bet (or more), deep ───────────────────────────────────────
    if label in ("AA", "KK"):
        if rer_act is not None:
            v[rer_act] += 0.80          # 4-bet value
            v[CC]      += 0.20
        else:
            v[CC] += 1.0
    elif label in ("QQ", "AKs", "AKo"):
        v[CC] += 0.65                   # flat to realise equity
        if rer_act is not None:
            v[rer_act] += 0.15
        else:
            v[CC] += 0.15
        v[FOLD] += 0.20
    elif label in ("JJ", "TT", "AQs", "AQo", "AJs", "KQs"):
        v[CC]   += 0.45
        v[FOLD] += 0.55
    else:
        if can_fold:
            v[FOLD] += 1.0
        else:
            v[CC] += 1.0
    return _finish(v, legal_set)


def _finish(v: np.ndarray, legal_set: frozenset) -> np.ndarray:
    """Zero any mass on illegal actions and renormalise (uniform if all zero)."""
    for a in range(config.N_ACTIONS):
        if a not in legal_set:
            v[a] = 0.0
    s = v.sum()
    if s <= 0.0:
        for a in legal_set:
            v[a] = 1.0 / len(legal_set)
    else:
        v /= s
    return v


# ── Tree enumeration ──────────────────────────────────────────────────────────
def generate() -> tuple[dict, dict]:
    strategy_sum: dict[int, np.ndarray] = {}
    visit_sum:    dict[int, float]      = {}
    stats = {"nodes": 0}

    # Cache the per-bucket chart distribution, which depends only on
    # (pos, n_raises, legal_set, bucket).  (_canon_key memoises keys itself.)
    strat_cache: dict[tuple, np.ndarray] = {}

    def visit(state):
        if is_terminal(state) or state.to_act == -1:
            return
        stats["nodes"] += 1

        seat     = state.to_act
        pos      = _pos_category(seat, config.N_PLAYERS)   # dealer_seat == 0
        legal    = legal_actions(state)
        legal_set = frozenset(legal)

        # n_raises straight from the live state — the betting-context key carries
        # it directly now (no history truncation), so the chart distribution and
        # the key agree on how many raises have gone in.
        n_raises = state.n_raises

        for bucket in range(169):
            key = _canon_key(state, seat, bucket)

            sc = (pos, n_raises if n_raises < 2 else 2, legal_set, bucket)
            dist = strat_cache.get(sc)
            if dist is None:
                dist = _strategy(pos, n_raises, bucket, legal_set)
                strat_cache[sc] = dist

            acc = strategy_sum.get(key)
            if acc is None:
                strategy_sum[key] = dist.copy()
            else:
                acc += dist
            visit_sum[key] = config.PRUNE_MIN_VISITS  # keep every chart info set

        for a in legal:
            visit(apply_action(state, a))

    # Dealer at seat 0 fixes pos == seat; cards are irrelevant to the tree shape.
    visit(make_initial_state(dealer_seat=0))
    print(f"[gen] betting decision nodes visited: {stats['nodes']:,}")
    print(f"[gen] distinct info-set keys:          {len(strategy_sum):,}")
    return strategy_sum, visit_sum


def main() -> None:
    strategy_sum, visit_sum = generate()
    n = export_strategy(strategy_sum, visit_sum)
    print(f"[gen] exported {n:,} info sets -> {config.EXPORT_PATH}")


if __name__ == "__main__":
    main()
