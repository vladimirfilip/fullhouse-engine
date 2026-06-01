"""Unit tests for Module B: range-conditioned equity.

Covers the hand-strength percentile table, the per-archetype range floor, and
the rejection-sampling behavior of monte_carlo_equity (tighter opponent floors
must lower a marginal hand's equity, looser floors must not).
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_spec = importlib.util.spec_from_file_location(
    "bot_rangeeq",
    os.path.join(os.path.dirname(__file__), "..", "bots", "vlad", "bot.py"),
)
_bot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bot)  # type: ignore

Card = _bot.Card


# ── hand percentile ─────────────────────────────────────────────────────────


def test_pctl_ordering():
    aa = _bot._hand_pctl("As", "Ad")
    kk = _bot._hand_pctl("Ks", "Kd")
    aks = _bot._hand_pctl("As", "Ks")
    trash = _bot._hand_pctl("7s", "2d")
    assert aa > kk > aks
    assert aa > 0.95
    assert trash < 0.10
    assert 0.0 <= trash <= aa <= 1.0


def test_pctl_suited_beats_offsuit():
    assert _bot._hand_pctl("As", "Ts") > _bot._hand_pctl("As", "Td")


def test_pctl_order_independent():
    assert _bot._hand_pctl("As", "Kd") == _bot._hand_pctl("Kd", "As")


# ── range floor ─────────────────────────────────────────────────────────────


def test_floor_unknown_is_uniform_on_flop():
    # No read → uniform (0) on the flop; mild bump on later streets.
    assert _bot._seat_range_floor("unknown", 0.0, "flop") == 0.0
    assert _bot._seat_range_floor("unknown", 0.0, "turn") > 0.0


def test_floor_nit_tighter_than_station():
    nit = _bot._seat_range_floor("nit", 0.9, "flop")
    sta = _bot._seat_range_floor("station", 0.9, "flop")
    assert nit > 0.5 > sta


def test_floor_scales_with_confidence():
    lo = _bot._seat_range_floor("nit", 0.2, "flop")
    hi = _bot._seat_range_floor("nit", 0.9, "flop")
    assert hi > lo


# ── rejection sampling in monte_carlo_equity ────────────────────────────────


def _equity(hole, board, floors, iters=4000, seed=0):
    import random
    random.seed(seed)
    hc = [Card(c) for c in hole]
    bd = [Card(c) for c in board]
    rest = [c for c in _bot.ALL_CARDS if c not in hc and c not in bd]
    return _bot.monte_carlo_equity(hc, bd, rest, num_opponents=len(floors),
                                   max_iters=iters, opp_floors=floors)


def test_tight_floor_lowers_marginal_equity():
    # A marginal made hand (middle pair) loses equity when the lone opponent is
    # forced to a strong (nit) range vs an unconstrained one.
    hole, board = ["9c", "8d"], ["9h", "5s", "2c"]
    uniform = _equity(hole, board, [0.0])
    tight = _equity(hole, board, [0.80])
    assert tight < uniform - 0.02


def test_loose_floor_close_to_uniform():
    hole, board = ["9c", "8d"], ["9h", "5s", "2c"]
    uniform = _equity(hole, board, [0.0])
    loose = _equity(hole, board, [0.05])
    assert abs(loose - uniform) < 0.05


def test_nuts_unaffected_by_floor():
    # With the nut flush, opponent range barely matters.
    hole, board = ["Ah", "Kh"], ["Qh", "Jh", "2h"]
    uniform = _equity(hole, board, [0.0])
    tight = _equity(hole, board, [0.80])
    assert uniform > 0.9 and tight > 0.85


def test_multiway_floors_run_without_error():
    hole, board = ["As", "Ad"], ["Kh", "7d", "2c"]
    eq = _equity(hole, board, [0.5, 0.3, 0.0], iters=2000)
    assert 0.0 <= eq <= 1.0
