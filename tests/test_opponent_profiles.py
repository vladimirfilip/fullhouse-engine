"""Unit tests for Modules A2/A3: opponent leak profiles + archetype tagging.

Two layers:
  1. Synthetic match logs with exact, hand-counted expectations.
  2. Engine-driven: scripted archetype bots play real multi-hand matches; the
     classifier must recover the archetype from the resulting match_action_log.
"""

import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_spec = importlib.util.spec_from_file_location(
    "bot_oppprof",
    os.path.join(os.path.dirname(__file__), "..", "bots", "vlad", "bot.py"),
)
_bot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bot)  # type: ignore


def M(hand, seat, bot, action, amount=None):
    return {"hand_num": hand, "seat": seat, "bot_id": bot,
            "action": action, "amount": amount}


# ── 1. Synthetic-log unit tests ─────────────────────────────────────────────


def test_iter_hands_groups_contiguous():
    log = [M(0, 0, "a", "fold"), M(0, 1, "b", "call"),
           M(1, 0, "a", "raise"), M(2, 1, "b", "fold")]
    groups = list(_bot._iter_hands(log))
    assert [h for h, _ in groups] == [0, 1, 2]
    assert len(groups[0][1]) == 2 and len(groups[1][1]) == 1


def test_self_excluded():
    log = [M(0, 0, "hero", "raise"), M(0, 1, "villain", "fold")]
    profs = _bot._build_opponent_profiles(log, my_bot_id="hero")
    assert "hero" not in profs
    assert "villain" in profs


def test_vpip_pfr_counts():
    # 4 hands. villain: fold, call, raise, fold → vpip 2/4, pfr 1/4.
    log = []
    for h, act in enumerate(["fold", "call", "raise", "fold"]):
        log.append(M(h, 0, "hero", "raise" if act != "fold" else "call"))
        log.append(M(h, 1, "villain", act))
    profs = _bot._build_opponent_profiles(log, my_bot_id="hero")
    v = profs["villain"]
    assert v["hands"] == 4
    assert v["vpip"] == 2      # call + raise
    assert v["pfr"] == 1       # raise only


def test_reraise_opportunity_and_action():
    # hero opens (aggr_before 0), villain 3-bets (aggr_before 1).
    log = [M(0, 0, "hero", "raise"), M(0, 1, "villain", "raise"),
           M(0, 0, "hero", "call")]
    v = _bot._build_opponent_profiles(log, my_bot_id="hero")["villain"]
    assert v["pf_reraise_opp"] == 1
    assert v["pf_reraise"] == 1


def test_fold_to_flop_cbet_attribution():
    # preflop: hero raise, villain call (both reach flop).
    # flop: hero bets (raise), villain folds → fold_to_flop_cbet 1/1.
    log = [
        M(0, 0, "hero", "raise"), M(0, 1, "villain", "call"),
        M(0, 0, "hero", "raise"), M(0, 1, "villain", "fold"),
    ]
    v = _bot._build_opponent_profiles(log, my_bot_id="hero")["villain"]
    assert v["flop_cbet_faced"] == 1
    assert v["flop_cbet_fold"] == 1
    assert v["saw_flop"] == 1
    assert v["saw_river"] == 0


def test_river_call_attribution():
    # Walk all the way to a river call by villain.
    log = [
        M(0, 0, "hero", "raise"), M(0, 1, "villain", "call"),     # preflop
        M(0, 0, "hero", "raise"), M(0, 1, "villain", "call"),     # flop
        M(0, 0, "hero", "raise"), M(0, 1, "villain", "call"),     # turn
        M(0, 0, "hero", "raise"), M(0, 1, "villain", "call"),     # river
    ]
    v = _bot._build_opponent_profiles(log, my_bot_id="hero")["villain"]
    assert v["saw_river"] == 1
    assert v["river_faced"] == 1
    assert v["river_call"] == 1


def test_shrunk_rate_toward_prior_when_sparse():
    prior, _ = _bot._LEAK_PRIORS["vpip"]
    val, conf = _bot._shrunk_rate(1, 1, "vpip")     # 1 obs only
    assert conf < 1.0
    # blended value sits between observed (1.0) and prior
    assert prior < val < 1.0


def test_unknown_below_min_hands():
    log = [M(0, 1, "villain", "fold")]
    v = _bot._build_opponent_profiles(log)["villain"]
    assert _bot._classify_opponent(v)[0] == "unknown"


# ── 2. Engine-driven archetype recovery ─────────────────────────────────────

pytest.importorskip("eval7", reason="engine archetype test needs eval7")


def _run_match_log(policies, n_hands=60, seed=1):
    """Play real hands and accumulate a match_action_log exactly as match.py does
    (only bot-returned action/amount, no blinds)."""
    from engine.game import PokerEngine
    bot_ids = list(policies.keys())
    match_log = []
    for hand_num in range(n_hands):
        eng = PokerEngine(hand_id=f"t_h{hand_num:04d}", bot_ids=bot_ids,
                          dealer_seat=hand_num % len(bot_ids), seed=seed * 1000 + hand_num)
        state = eng.start_hand()
        while state.get("type") == "action_request":
            seat = state["seat_to_act"]
            bid = bot_ids[seat]
            action = policies[bid](state)
            match_log.append({"hand_num": hand_num, "seat": seat, "bot_id": bid,
                              "action": action.get("action"), "amount": action.get("amount")})
            state = eng.apply_action(seat, action)
    return match_log


def _p_nit(state):
    # Folds everything except when it can check for free.
    return {"action": "check"} if state["can_check"] else {"action": "fold"}


def _p_station(state):
    # Calls anything, never raises.
    return {"action": "check"} if state["can_check"] else {"action": "call"}


def _p_maniac(state):
    # Raises whenever legal, else calls.
    if state["your_stack"] > 0 and state["min_raise_to"] < state["your_stack"] + state["your_bet_this_street"]:
        return {"action": "raise", "amount": state["min_raise_to"]}
    return {"action": "call"}


def _p_tag(state):
    # Tight-aggressive-ish: raise ~ a fifth of the time, otherwise check/fold.
    h = (state["seat_to_act"] + len(state["action_log"])) % 5
    if h == 0 and state["your_stack"] > 0:
        return {"action": "raise", "amount": state["min_raise_to"]}
    return {"action": "check"} if state["can_check"] else {"action": "fold"}


def test_engine_recovers_nit():
    log = _run_match_log({"hero": _p_station, "nit": _p_nit, "x": _p_station})
    v = _bot._build_opponent_profiles(log, my_bot_id="hero")["nit"]
    tag, conf = _bot._classify_opponent(v)
    assert tag == "nit"
    assert conf > 0.4


def test_engine_recovers_maniac():
    log = _run_match_log({"hero": _p_station, "maniac": _p_maniac, "x": _p_station})
    v = _bot._build_opponent_profiles(log, my_bot_id="hero")["maniac"]
    tag, _ = _bot._classify_opponent(v)
    assert tag == "maniac"


def test_engine_recovers_station():
    # Maniac drives betting so the station faces (and calls) lots of rivers.
    log = _run_match_log({"hero": _p_maniac, "station": _p_station, "x": _p_maniac})
    v = _bot._build_opponent_profiles(log, my_bot_id="hero")["station"]
    L = _bot._opp_leaks(v)
    # Defining station traits: loose, passive (low aggression factor).
    assert L["vpip"] > 0.30
    assert L["aggression_factor"] < 1.0


def test_profiles_keyed_on_bot_id_survive_seat_reindex():
    # bot_ids stable; seats differ per hand by rotating dealer. Ensure the same
    # villain accumulates across hands rather than splitting by seat.
    log = _run_match_log({"hero": _p_station, "v": _p_maniac, "w": _p_nit})
    profs = _bot._build_opponent_profiles(log, my_bot_id="hero")
    assert set(profs.keys()) == {"v", "w"}
    assert profs["v"]["hands"] > 30 and profs["w"]["hands"] > 30
