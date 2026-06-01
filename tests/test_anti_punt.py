"""Unit tests for Module C: the anti-punt override layer.

Each rule must (a) fire and revert the bad line when a confident read supports
it, and (b) be a strict no-op when the read is absent/low-confidence — the
property that keeps it safe vs unknown/strong opponents.
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_spec = importlib.util.spec_from_file_location(
    "bot_antipunt",
    os.path.join(os.path.dirname(__file__), "..", "bots", "vlad", "bot.py"),
)
_bot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bot)  # type: ignore


def _players(states_by_seat):
    return [{"seat": s, "bot_id": b, "state": st}
            for s, (b, st) in states_by_seat.items()]


def _station_profile(hands=40):
    """Counters that classify as a confident station."""
    c = _bot._fresh_opp_counters()
    c.update({"hands": hands, "vpip": int(hands * 0.7), "pfr": int(hands * 0.05),
              "n_call": 60, "n_aggr": 3,
              "river_faced": 20, "river_call": 16})
    return c


def _nit_profile(hands=40):
    c = _bot._fresh_opp_counters()
    c.update({"hands": hands, "vpip": int(hands * 0.10), "pfr": int(hands * 0.07),
              "n_call": 5, "n_aggr": 1})
    return c


def test_station_and_nit_profiles_classify():
    assert _bot._classify_opponent(_station_profile())[0] == "station"
    assert _bot._classify_opponent(_nit_profile())[0] == "nit"


# ── Rule 1: river air-bluff into station ────────────────────────────────────

def _river_bluff_gs():
    return {
        "seat_to_act": 0, "street": "river",
        "your_cards": ["7c", "2d"], "community_cards": ["Ah", "Kd", "9s", "5c", "3h"],
        "amount_owed": 0, "pot": 1000, "can_check": True,
        "players": _players({0: ("hero", "active"), 1: ("villain", "active")}),
    }


def test_rule1_blocks_river_bluff_into_station():
    gs = _river_bluff_gs()
    profiles = {"villain": _station_profile()}
    action = _bot._anti_punt({"action": "raise", "amount": 800}, gs, 0.05, profiles)
    assert action["action"] == "check"


def test_rule1_noop_without_read():
    gs = _river_bluff_gs()
    action = _bot._anti_punt({"action": "raise", "amount": 800}, gs, 0.05, {})
    assert action == {"action": "raise", "amount": 800}


def test_rule1_noop_with_strong_hand():
    gs = _river_bluff_gs()
    gs["your_cards"] = ["Ac", "Ad"]   # set of aces -> strong, value bet is fine
    gs["community_cards"] = ["As", "Kd", "9s", "5c", "3h"]
    profiles = {"villain": _station_profile()}
    action = _bot._anti_punt({"action": "raise", "amount": 800}, gs, 0.9, profiles)
    assert action["action"] == "raise"


# ── Rule 2: oversized river bluff-catch vs nit ──────────────────────────────

def _river_call_gs(owed):
    return {
        "seat_to_act": 0, "street": "river",
        "your_cards": ["Qc", "Jd"], "community_cards": ["Ah", "Kd", "9s", "5c", "3h"],
        "amount_owed": owed, "pot": 1000, "can_check": False,
        # villain (seat 1) raised on the river -> last aggressor.
        "action_log": [{"seat": 1, "action": "raise", "amount": owed}],
        "players": _players({0: ("hero", "active"), 1: ("villain", "active")}),
    }


def test_rule2_folds_big_river_call_vs_nit():
    gs = _river_call_gs(owed=800)   # > 0.55 * pot
    profiles = {"villain": _nit_profile()}
    action = _bot._anti_punt({"action": "call"}, gs, 0.40, profiles)
    assert action["action"] == "fold"


def test_rule2_noop_small_bet():
    gs = _river_call_gs(owed=200)   # < 0.55 * pot, price is fine
    profiles = {"villain": _nit_profile()}
    action = _bot._anti_punt({"action": "call"}, gs, 0.40, profiles)
    assert action["action"] == "call"


def test_rule2_noop_vs_station():
    gs = _river_call_gs(owed=800)
    profiles = {"villain": _station_profile()}    # not a nit -> call stands
    action = _bot._anti_punt({"action": "call"}, gs, 0.40, profiles)
    assert action["action"] == "call"


# ── Rule 3: low-equity multiway c-bet on wet board ──────────────────────────

def _multiway_wet_gs():
    return {
        "seat_to_act": 0, "street": "flop",
        "your_cards": ["Ac", "2d"], "community_cards": ["Kh", "Qh", "9h"],
        "amount_owed": 0, "pot": 600, "can_check": True,
        "players": _players({0: ("hero", "active"), 1: ("v1", "active"),
                             2: ("v2", "active")}),
    }


def test_rule3_blocks_lowequity_multiway_wet_cbet():
    gs = _multiway_wet_gs()
    # Needs at least one station in the field for the read to bite? No — rule 3
    # is field-size + texture + equity based, not read-gated. Verify it fires.
    action = _bot._anti_punt({"action": "raise", "amount": 300}, gs, 0.20, {})
    assert action["action"] == "check"


def test_rule3_noop_heads_up():
    gs = _multiway_wet_gs()
    gs["players"] = _players({0: ("hero", "active"), 1: ("v1", "active")})
    action = _bot._anti_punt({"action": "raise", "amount": 300}, gs, 0.20, {})
    assert action["action"] == "raise"


def test_rule3_noop_with_equity():
    gs = _multiway_wet_gs()
    action = _bot._anti_punt({"action": "raise", "amount": 300}, gs, 0.70, {})
    assert action["action"] == "raise"


def test_fold_and_check_pass_through():
    gs = _river_bluff_gs()
    assert _bot._anti_punt({"action": "fold"}, gs, 0.0, {}) == {"action": "fold"}
    assert _bot._anti_punt({"action": "check"}, gs, 0.0, {}) == {"action": "check"}
