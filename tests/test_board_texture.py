"""Unit tests for Module E: full-board texture (_board_features / _board_texture).

Verifies (a) the rich feature extraction over flop/turn/river, and (b) that the
coarse _board_texture category preserves the original flop semantics exactly
while becoming full-board aware on the turn and river.
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_spec = importlib.util.spec_from_file_location(
    "bot_texture",
    os.path.join(os.path.dirname(__file__), "..", "bots", "vlad", "bot.py"),
)
_bot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bot)  # type: ignore

feats = _bot._board_features
tex = _bot._board_texture


# ── _board_features ─────────────────────────────────────────────────────────


def test_preflop_empty():
    assert feats([])["n"] == 0
    assert tex([]) == "none"
    assert tex(["As"]) == "none"


def test_dry_rainbow_flop():
    f = feats(["Ah", "7d", "2c"])
    assert not f["paired"] and not f["two_tone"]
    assert not f["flush_possible"] and not f["connected"]


def test_paired_flop():
    f = feats(["Th", "Td", "2c"])
    assert f["paired"] and not f["two_pair"] and not f["trips"]


def test_monotone_flop_is_flush_possible():
    f = feats(["Ah", "9h", "2h"])
    assert f["monotone"] and f["flush_possible"] and f["max_suit"] == 3
    assert not f["four_flush"]


def test_connected_flop():
    f = feats(["9h", "8d", "7c"])
    assert f["connected"] and f["straight_possible"]
    assert f["straight_cards"] == 3


def test_turn_pairs_board():
    # Flop unpaired, turn pairs it → paired must surface on the full board.
    f = feats(["Th", "7d", "2c", "7s"])
    assert f["paired"]
    assert tex(["Th", "7d", "2c", "7s"]) == "paired"


def test_river_completes_flush():
    f = feats(["Ah", "9h", "2c", "5d", "Kh"])
    assert f["flush_possible"] and f["max_suit"] == 3
    assert tex(["Ah", "9h", "2c", "5d", "Kh"]) == "wet"


def test_four_flush_board():
    f = feats(["Ah", "9h", "2h", "5h", "Kd"])
    assert f["four_flush"] and f["max_suit"] == 4


def test_four_to_straight():
    # J-T-9-8 on board → one card completes a straight.
    f = feats(["Jh", "Tc", "9d", "8s"])
    assert f["four_straight"] and f["straight_cards"] == 4
    assert tex(["Jh", "Tc", "9d", "8s"]) == "wet"


def test_wheel_straight_uses_ace_low():
    # A-2-3-4 → four to the wheel (ace counts low).
    f = feats(["Ah", "2c", "3d", "4s"])
    assert f["four_straight"]


def test_broadway_straight_on_board():
    f = feats(["Ah", "Kc", "Qd", "Js", "Th"])
    assert f["straight_cards"] == 5


def test_two_pair_board():
    f = feats(["Th", "Td", "7c", "7s", "2h"])
    assert f["two_pair"] and f["paired"]


def test_trips_board():
    f = feats(["Th", "Td", "Tc", "7s", "2h"])
    assert f["trips"] and f["paired"]


# ── _board_texture: flop semantics preserved ────────────────────────────────


def _legacy_flop_texture(board):
    """The original pre-Module-E implementation, for equivalence checking."""
    if len(board) < 3:
        return "none"
    flop = board[:3]
    ranks = [_bot._RANK_ORDER[c[0]] for c in flop]
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


def test_flop_category_matches_legacy_exhaustively():
    # Every 3-card combination must classify identically to the old code.
    ranks = "23456789TJQKA"
    suits = "shdc"
    deck = [r + s for r in ranks for s in suits]
    checked = 0
    for i in range(len(deck)):
        for j in range(i + 1, len(deck)):
            for k in range(j + 1, len(deck)):
                board = [deck[i], deck[j], deck[k]]
                assert tex(board) == _legacy_flop_texture(board), board
                checked += 1
    assert checked == 22100  # C(52,3)
