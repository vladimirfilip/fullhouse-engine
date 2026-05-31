"""Golden-output regression test for bots/vlad/bot.py.

Run once with --regen to write tests/golden_bot.json, then pytest enforces it.
All scenarios use deterministic RNG (decide() seeds from game-state fields).
"""

import importlib.util
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_bot.json")

_bot_spec = importlib.util.spec_from_file_location(
    "bot_golden",
    os.path.join(os.path.dirname(__file__), "..", "bots", "vlad", "bot.py"),
)  # noqa: E501
_bot_mod = importlib.util.module_from_spec(_bot_spec)
_bot_spec.loader.exec_module(_bot_mod)  # type: ignore


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _players(n=6, stack=9900, overrides=None):
    ps = [
        {"seat": i, "stack": stack, "state": "active",
         "is_folded": False, "is_all_in": False,
         "bet_this_street": 0, "bot_id": f"p{i}"}
        for i in range(n)
    ]
    if overrides:
        for seat, kv in overrides.items():
            ps[seat].update(kv)
    return ps


SCENARIOS = {
    # 1. Preflop CFR table path: UTG facing BB, 6-max 100bb
    "preflop_utg_vs_bb": {
        "hand_id": "m1_h0001",
        "your_cards": ["Ah", "Kh"],
        "community_cards": [],
        "street": "preflop",
        "seat_to_act": 3,
        "pot": 150,
        "your_stack": 9900,
        "amount_owed": 100,
        "can_check": False,
        "current_bet": 100,
        "min_raise_to": 200,
        "your_bet_this_street": 0,
        "players": _players(6, 9900, {
            1: {"bet_this_street": 50},
            2: {"bet_this_street": 100},
        }),
        "action_log": [
            {"seat": 1, "action": "small_blind", "amount": 50},
            {"seat": 2, "action": "big_blind",   "amount": 100},
        ],
        "match_action_log": [],
    },

    # 2. Preflop 3-bet scenario, BB facing UTG open + BTN 3bet
    "preflop_bb_vs_3bet": {
        "hand_id": "m1_h0002",
        "your_cards": ["Qh", "Qs"],
        "community_cards": [],
        "street": "preflop",
        "seat_to_act": 2,
        "pot": 1050,
        "your_stack": 9900,
        "amount_owed": 800,
        "can_check": False,
        "current_bet": 900,
        "min_raise_to": 1700,
        "your_bet_this_street": 100,
        "players": _players(6, 9100, {
            1: {"stack": 9950, "bet_this_street": 50},
            2: {"stack": 9900, "bet_this_street": 100},
            3: {"stack": 9700, "bet_this_street": 300},
            0: {"stack": 9100, "bet_this_street": 900},
        }),
        "action_log": [
            {"seat": 1, "action": "small_blind", "amount": 50},
            {"seat": 2, "action": "big_blind",   "amount": 100},
            {"seat": 3, "action": "raise",        "amount": 300},
            {"seat": 4, "action": "fold",         "amount": 0},
            {"seat": 5, "action": "fold",         "amount": 0},
            {"seat": 0, "action": "raise",        "amount": 900},
            {"seat": 1, "action": "fold",         "amount": 0},
        ],
        "match_action_log": [],
    },

    # 3. Preflop, 4 players — preflop table inapplicable, falls to GTO net
    "preflop_4handed": {
        "hand_id": "m1_h0003",
        "your_cards": ["7d", "6d"],
        "community_cards": [],
        "street": "preflop",
        "seat_to_act": 3,
        "pot": 150,
        "your_stack": 9900,
        "amount_owed": 100,
        "can_check": False,
        "current_bet": 100,
        "min_raise_to": 200,
        "your_bet_this_street": 0,
        "players": _players(4, 9900, {
            1: {"bet_this_street": 50},
            2: {"bet_this_street": 100},
        }),
        "action_log": [
            {"seat": 1, "action": "small_blind", "amount": 50},
            {"seat": 2, "action": "big_blind",   "amount": 100},
        ],
        "match_action_log": [],
    },

    # 4. Flop, facing half-pot bet — GTO path with realtime equity search
    "flop_facing_bet": {
        "hand_id": "m1_h0004",
        "your_cards": ["As", "Qs"],
        "community_cards": ["Th", "7d", "2c"],
        "street": "flop",
        "seat_to_act": 4,
        "pot": 600,
        "your_stack": 9700,
        "amount_owed": 300,
        "can_check": False,
        "current_bet": 300,
        "min_raise_to": 600,
        "your_bet_this_street": 0,
        "players": _players(6, 9700, {
            3: {"stack": 9700, "bet_this_street": 300},
        }),
        "action_log": [
            {"seat": 1, "action": "small_blind", "amount": 50},
            {"seat": 2, "action": "big_blind",   "amount": 100},
            {"seat": 3, "action": "raise",        "amount": 300},
            {"seat": 4, "action": "call",         "amount": 300},
            {"seat": 5, "action": "fold"},
            {"seat": 0, "action": "fold"},
            {"seat": 1, "action": "fold"},
            {"seat": 2, "action": "fold"},
            {"seat": 3, "action": "raise",        "amount": 300},
        ],
        "match_action_log": [],
    },

    # 5. Turn, check-to — GTO path, no fold/call tension
    "turn_check_to": {
        "hand_id": "m1_h0005",
        "your_cards": ["Kd", "Qd"],
        "community_cards": ["Jh", "Tc", "9d", "2s"],
        "street": "turn",
        "seat_to_act": 2,
        "pot": 400,
        "your_stack": 9800,
        "amount_owed": 0,
        "can_check": True,
        "current_bet": 0,
        "min_raise_to": 200,
        "your_bet_this_street": 0,
        "players": _players(6, 9800),
        "action_log": [
            {"seat": 1, "action": "small_blind", "amount": 50},
            {"seat": 2, "action": "big_blind",   "amount": 100},
            {"seat": 3, "action": "call",         "amount": 100},
            {"seat": 4, "action": "call",         "amount": 100},
            {"seat": 5, "action": "fold"},
            {"seat": 0, "action": "fold"},
            {"seat": 1, "action": "call",         "amount": 100},
            {"seat": 2, "action": "check"},
            {"seat": 1, "action": "check"},
            {"seat": 3, "action": "check"},
            {"seat": 4, "action": "check"},
            {"seat": 2, "action": "check"},
        ],
        "match_action_log": [],
    },

    # 6. River, facing large overbet
    "river_facing_overbet": {
        "hand_id": "m1_h0006",
        "your_cards": ["Jc", "Js"],
        "community_cards": ["Jh", "Th", "9h", "8c", "2d"],
        "street": "river",
        "seat_to_act": 1,
        "pot": 4000,
        "your_stack": 6000,
        "amount_owed": 5000,
        "can_check": False,
        "current_bet": 5000,
        "min_raise_to": 10000,
        "your_bet_this_street": 0,
        "players": _players(6, 6000, {
            2: {"stack": 5000, "bet_this_street": 5000},
        }),
        "action_log": [
            {"seat": 1, "action": "small_blind", "amount": 50},
            {"seat": 2, "action": "big_blind",   "amount": 100},
            {"seat": 3, "action": "fold"},
            {"seat": 4, "action": "fold"},
            {"seat": 5, "action": "fold"},
            {"seat": 0, "action": "fold"},
            {"seat": 1, "action": "call",         "amount": 100},
            {"seat": 2, "action": "check"},
            {"seat": 1, "action": "check"},
            {"seat": 2, "action": "raise",        "amount": 5000},
        ],
        "match_action_log": [],
    },
}


def _run(name: str) -> dict:
    return _bot_mod.decide(SCENARIOS[name])


def _collect_golden() -> dict:
    return {name: _run(name) for name in SCENARIOS}


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def golden():
    if not os.path.exists(_GOLDEN_PATH):
        pytest.skip(
            "Golden file missing — run: "
            "python tests/test_bot_golden.py --regen"
        )
    with open(_GOLDEN_PATH) as f:
        return json.load(f)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_golden_scenario(name, golden):  # noqa: F811
    result = _run(name)
    assert result == golden[name], (
        f"Scenario '{name}': got {result}, expected {golden[name]}"
    )


def test_mc_fallback_smoke(monkeypatch):
    """MC fallback must return a valid action when GTO layers unavailable."""
    monkeypatch.setattr(_bot_mod, "_GTO_LAYERS", None)
    result = _run("flop_facing_bet")
    assert "action" in result
    assert result["action"] in ("fold", "check", "call", "raise", "all_in")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="Regenerate golden file")
    args = ap.parse_args()
    if args.regen:
        data = _collect_golden()
        with open(_GOLDEN_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Golden file written to {_GOLDEN_PATH}")
        for name, out in data.items():
            print(f"  {name}: {out}")
    else:
        print("Run with --regen to generate the golden file.")
