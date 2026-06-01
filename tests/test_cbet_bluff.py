"""Tests for the MC-path offensive c-bet (Phase 2 #1, cfr_equity-inspired).

Verifies initiative detection and that the c-bet fires only under the disciplined
conditions (flop, HU/3-way, dry/semi board, has initiative, no station/maniac,
some backup equity) and is otherwise a no-op (check).
"""

import importlib.util
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_spec = importlib.util.spec_from_file_location(
    "bot_cbet",
    os.path.join(os.path.dirname(__file__), "..", "bots", "vlad", "bot.py"),
)
_bot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bot)  # type: ignore

# The offensive c-bet ships OFF by default (VLAD_CBET_BLUFF); enable it here so
# these tests exercise the logic. Production keeps it gated off pending
# multi-seed tournament validation.
_bot._CBET_BLUFF_ENABLED = True


def _players(states):
    return [{"seat": s, "bot_id": b, "state": st} for s, (b, st) in states.items()]


# preflop: seat 0 (hero) raises, seat 1 calls → hero has initiative.
_PF_LOG = [
    {"seat": 1, "action": "small_blind", "amount": 50},
    {"seat": 2, "action": "big_blind", "amount": 100},
    {"seat": 0, "action": "raise", "amount": 300},
    {"seat": 1, "action": "fold"},
    {"seat": 2, "action": "call", "amount": 200},
]


def _flop_gs(board=("Kh", "7d", "2c"), log=None):
    return {
        "seat_to_act": 0, "street": "flop",
        "your_cards": ["Ad", "Qs"], "community_cards": list(board),
        "amount_owed": 0, "pot": 650, "current_bet": 0, "min_raise_to": 100,
        "your_stack": 9700, "your_bet_this_street": 0,
        "action_log": log if log is not None else _PF_LOG,
        "players": _players({0: ("hero", "active"), 2: ("v", "active")}),
    }


def test_initiative_detection():
    assert _bot._has_initiative(_flop_gs()) is True
    # If villain (seat 2) was the last pf raiser, hero has no initiative.
    log = _PF_LOG[:2] + [{"seat": 2, "action": "raise", "amount": 300},
                         {"seat": 0, "action": "call", "amount": 200}]
    assert _bot._has_initiative(_flop_gs(log=log)) is False


def test_cbet_fires_with_initiative_dry_board_no_station():
    random.seed(0)
    fired = sum(1 for _ in range(400)
                if _bot._should_cbet_bluff(_flop_gs(), 0.40, (0, 0, 0), "dry", 1))
    assert 150 < fired < 280       # ≈ 55% frequency


def test_cbet_blocked_without_initiative():
    log = _PF_LOG[:2] + [{"seat": 2, "action": "raise", "amount": 300},
                         {"seat": 0, "action": "call", "amount": 200}]
    gs = _flop_gs(log=log)
    assert not any(_bot._should_cbet_bluff(gs, 0.40, (0, 0, 0), "dry", 1)
                   for _ in range(50))


def test_cbet_blocked_vs_station():
    assert not any(_bot._should_cbet_bluff(_flop_gs(), 0.40, (0, 1, 0), "dry", 1)
                   for _ in range(50))


def test_cbet_blocked_on_wet_board():
    assert not any(_bot._should_cbet_bluff(_flop_gs(), 0.40, (0, 0, 0), "wet", 1)
                   for _ in range(50))


def test_cbet_blocked_multiway():
    assert not any(_bot._should_cbet_bluff(_flop_gs(), 0.40, (0, 0, 0), "dry", 3)
                   for _ in range(50))


def test_cbet_blocked_with_no_backup_equity():
    assert not any(_bot._should_cbet_bluff(_flop_gs(), 0.10, (0, 0, 0), "dry", 1)
                   for _ in range(50))


def test_cbet_blocked_on_turn():
    gs = _flop_gs(board=("Kh", "7d", "2c", "5s"))
    gs["street"] = "turn"
    assert not any(_bot._should_cbet_bluff(gs, 0.40, (0, 0, 0), "dry", 1)
                   for _ in range(50))


def test_mc_postflop_cbets_weak_hand_with_initiative():
    # Weak hand (low equity) but initiative + dry + HU + no station → should bet,
    # not check. Seed so the 55% roll fires.
    random.seed(0)
    got_bet = False
    for seed in range(20):
        random.seed(seed)
        act = _bot._mc_postflop(_flop_gs(), 0.32, (0, 0, 0))
        if act["action"] == "raise":
            got_bet = True
            break
    assert got_bet
