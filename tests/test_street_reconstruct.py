"""Unit tests for Module A1: _reconstruct_streets in bots/vlad/bot.py.

Street labels are reconstructed structurally (by action type, not amount) so
the same logic works on both the within-hand action_log (which carries
small_blind/big_blind entries) and a single-hand slice of the cross-hand
match_action_log (no blinds, bot-raw action names, amounts often missing).

Two layers of validation:
  1. Hand-crafted scenarios with street boundaries known from poker rules.
  2. An engine-driven cross-check: play real hands, compare the reconstructed
     streets against the engine's own street_start events (ground truth).
     Skipped automatically if eval7 is unavailable.
"""

import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_bot_spec = importlib.util.spec_from_file_location(
    "bot_street",
    os.path.join(os.path.dirname(__file__), "..", "bots", "vlad", "bot.py"),
)
_bot = importlib.util.module_from_spec(_bot_spec)
_bot_spec.loader.exec_module(_bot)  # type: ignore

reconstruct = _bot._reconstruct_streets


def _streets(actions, **kw):
    return [r["street"] for r in reconstruct(actions, **kw)]


def _aggr(actions, **kw):
    return [r["street_aggr_before"] for r in reconstruct(actions, **kw)]


def A(seat, action):
    return {"seat": seat, "action": action}


# ── 1. Hand-crafted scenarios ───────────────────────────────────────────────


def test_empty():
    assert reconstruct([]) == []


def test_preflop_fold_around_to_bb():
    # 6-max, seats 0..5. Blinds posted by SB=1, BB=2 (within-hand format).
    # UTG..BTN all fold, SB folds, BB wins uncontested → hand ends preflop.
    actions = [
        A(1, "small_blind"), A(2, "big_blind"),
        A(3, "fold"), A(4, "fold"), A(5, "fold"), A(0, "fold"), A(1, "fold"),
    ]
    assert _streets(actions) == ["preflop"] * 7


def test_limped_pot_bb_checks_option_then_flop():
    # SB completes, everyone calls, BB checks option → preflop closes.
    # Then a flop where action checks through.
    actions = [
        A(1, "small_blind"), A(2, "big_blind"),
        A(3, "call"), A(4, "call"), A(5, "call"), A(0, "call"),
        A(1, "call"),    # SB completes
        A(2, "check"),   # BB option → preflop closes
        # flop: first actor is SB(1), check around
        A(1, "check"), A(2, "check"), A(3, "check"), A(4, "check"),
        A(5, "check"), A(0, "check"),
    ]
    s = _streets(actions)
    assert s[:8] == ["preflop"] * 8          # 2 blinds + 4 calls + SB + BB option
    assert s[8:] == ["flop"] * 6


def test_preflop_raise_reopens_then_call_closes():
    # BB option does NOT close while a raise is outstanding.
    actions = [
        A(1, "small_blind"), A(2, "big_blind"),
        A(3, "raise"),       # open
        A(4, "fold"), A(5, "fold"), A(0, "fold"), A(1, "fold"),
        A(2, "call"),        # BB calls the raise → preflop closes
        A(2, "check"), A(3, "check"),   # flop, heads-up BB vs opener
    ]
    s = _streets(actions)
    assert s == ["preflop"] * 8 + ["flop", "flop"]


def test_three_bet_aggression_counts():
    # open, 3-bet, call. street_aggr_before should track 0,1,2 on the raises.
    actions = [
        A(1, "small_blind"), A(2, "big_blind"),
        A(3, "raise"),   # open  (aggr_before = 0)
        A(4, "raise"),   # 3-bet (aggr_before = 1)
        A(3, "call"),    # call  (aggr_before = 2) → closes
    ]
    aggr = _aggr(actions)
    # blinds are 0,0; then open=0, 3bet=1, call=2
    assert aggr == [0, 0, 0, 1, 2]
    assert _streets(actions) == ["preflop"] * 5


def test_all_streets_with_postflop_betting():
    actions = [
        A(1, "small_blind"), A(2, "big_blind"),
        A(3, "raise"), A(4, "fold"), A(5, "fold"), A(0, "fold"),
        A(1, "fold"), A(2, "call"),                       # preflop closes (HU: 2 vs 3)
        A(2, "check"), A(3, "raise"), A(2, "call"),       # flop closes
        A(2, "check"), A(3, "raise"), A(2, "call"),       # turn closes
        A(2, "check"), A(3, "check"),                     # river closes
    ]
    s = _streets(actions)
    assert s == (["preflop"] * 8) + (["flop"] * 3) + (["turn"] * 3) + (["river"] * 2)


def test_fold_to_flop_cbet_is_attributed_to_flop():
    actions = [
        A(1, "small_blind"), A(2, "big_blind"),
        A(3, "raise"), A(2, "call"), A(0, "fold"), A(1, "fold"),
        A(4, "fold"), A(5, "fold"),                       # preflop closes
        A(2, "check"), A(3, "raise"), A(2, "fold"),       # flop c-bet, BB folds
    ]
    res = reconstruct(actions)
    fold_entry = res[-1]
    assert fold_entry["street"] == "flop"
    assert fold_entry["street_aggr_before"] == 1          # faced the c-bet


def test_cross_hand_format_no_blinds():
    # match_action_log slice: no blind entries, starts at UTG. Provide the
    # dealt-in seats explicitly (cross-hand log can't always infer walks).
    actions = [
        A(3, "raise"), A(4, "fold"), A(5, "fold"), A(0, "fold"),
        A(1, "fold"), A(2, "call"),                       # preflop closes
        A(2, "check"), A(3, "check"),                     # flop checks through
        A(2, "raise"), A(3, "call"),                      # turn closes
        A(2, "raise"), A(3, "fold"),                      # river: bet/fold
    ]
    s = _streets(actions, dealt_seats=[0, 1, 2, 3, 4, 5])
    assert s == (["preflop"] * 6) + (["flop"] * 2) + (["turn"] * 2) + (["river"] * 2)


def test_heads_up_postflop_first_actor():
    # HU cross-hand: 2 seats, no blinds in log. preflop raise/call, then 3 streets.
    actions = [
        A(0, "raise"), A(1, "call"),       # preflop
        A(1, "check"), A(0, "check"),      # flop
        A(1, "check"), A(0, "check"),      # turn
        A(1, "check"), A(0, "check"),      # river
    ]
    s = _streets(actions, dealt_seats=[0, 1])
    assert s == (["preflop"] * 2) + (["flop"] * 2) + (["turn"] * 2) + (["river"] * 2)


def test_street_clamps_at_river():
    # Pathological over-long sequence must never exceed "river".
    actions = [A(0, "raise"), A(1, "call")] * 10
    s = _streets(actions, dealt_seats=[0, 1])
    assert set(s) <= set(_bot._STREET_NAMES)
    assert s[-1] == "river"


# ── 2. Engine-driven cross-check (ground truth) ─────────────────────────────
#
# Play real hands through the frozen engine and compare reconstructed streets
# against the engine's own per-action events. Each "action" event already
# carries the street it occurred on (game.py _emit), captured before the street
# advances — so it is authoritative ground truth.

pytest.importorskip("eval7", reason="engine cross-check needs eval7")


def _play(seed, n, policy):
    from engine.game import PokerEngine
    eng = PokerEngine(hand_id="t_h0001", bot_ids=list("abcdef")[:n],
                      dealer_seat=seed % n, seed=seed)
    state = eng.start_hand()
    while state.get("type") == "action_request":
        state = eng.apply_action(state["seat_to_act"], policy(state, seed))
    return eng.action_log, eng.events


def _engine_action_streets(events):
    """Street active for each engine 'action' event, in order."""
    return [ev["street"] for ev in events
            if isinstance(ev, dict) and ev.get("type") == "action"]


def _crosscheck(action_log, events):
    recon = reconstruct(action_log)
    recon_nonblind = [r["street"] for r, a in zip(recon, action_log)
                      if str(a["action"]).lower() not in _bot._BLIND_ACTIONS]
    return recon_nonblind, _engine_action_streets(events)


def _calldown(state, seed):
    return {"action": "call"} if state["amount_owed"] > 0 else {"action": "check"}


def _varied(state, seed):
    # Deterministic mix of raise / call / fold / check driven by game-state.
    h = (state["seat_to_act"] * 31 + len(state["action_log"]) * 7 + seed) % 10
    owed = state["amount_owed"]
    if owed == 0:
        return {"action": "raise", "amount": state["min_raise_to"]} if h < 3 else {"action": "check"}
    if h < 5:
        return {"action": "call"}
    if h < 8:
        return {"action": "fold"}
    return {"action": "raise", "amount": state["min_raise_to"]}


@pytest.mark.parametrize("seed", [1, 2, 7, 13, 42, 99])
@pytest.mark.parametrize("n", [2, 3, 6])
def test_engine_crosscheck_calldown(seed, n):
    recon, gt = _crosscheck(*_play(seed, n, _calldown))
    assert recon == gt


@pytest.mark.parametrize("seed", [1, 2, 7, 13, 42, 99, 123, 2024])
@pytest.mark.parametrize("n", [2, 3, 6])
def test_engine_crosscheck_varied(seed, n):
    recon, gt = _crosscheck(*_play(seed, n, _varied))
    assert recon == gt
