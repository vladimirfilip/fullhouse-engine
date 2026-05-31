"""Submission entrypoint — def decide(game_state) -> dict.

Three-layer architecture (see docs/specs/2026-04-25-poker-bot-design.md):
  Layer 1 (blueprint): preflop ranges + postflop heuristic policy
  Layer 2 (opponent model): per-seat stats + archetype + bot-origin fingerprint
  Layer 3 (decision): combine, sample, enforce legality

Everything wrapped in try/except so the bot never crashes the match.
"""
from __future__ import annotations

import time
import random

from .params import PARAMS
from .eval.equity import preflop_equity, equity_vs_random, hand_strength_label
from .blueprint.preflop import preflop_action, position_tier
from .blueprint.postflop import postflop_action, board_texture
from .opponent.tracker import OpponentModel, archetype, fingerprint
from .opponent.exploit import deviation

BOT_NAME = "RexFold-v1"
BOT_AVATAR = "robot_1"

# --- Per-process state ---------------------------------------------------------
_OPP = OpponentModel()
_LAST_HAND_ID = None


def _legal_raise(amount: int, state: dict) -> dict:
    """Snap raise to legal bounds: >=min_raise_to, <=stack -> all_in fallback."""
    your_stack = state["your_stack"]
    your_in = state.get("your_bet_this_street", 0)
    min_total = state["min_raise_to"]
    max_total = your_stack + your_in
    if amount >= max_total:
        return {"action": "all_in"}
    if amount < min_total:
        amount = min_total
    if amount >= max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": int(amount)}


def _has_aggressor(state: dict) -> bool:
    """Did anyone raise/all_in preflop before our action this hand?"""
    log = state.get("action_log", [])
    for a in log:
        if a.get("street") == "preflop" and a.get("action") in ("raise", "all_in"):
            return True
    # Fallback: amount_owed > BB implies someone raised
    return state.get("amount_owed", 0) > 100


def _facing_3bet(state: dict) -> bool:
    log = state.get("action_log", [])
    raises = [a for a in log if a.get("street") == "preflop" and a.get("action") in ("raise", "all_in")]
    return len(raises) >= 2


def _was_preflop_aggressor(state: dict, my_seat: int) -> bool:
    log = state.get("action_log", [])
    last_pf_aggr = None
    for a in log:
        if a.get("street") == "preflop" and a.get("action") in ("raise", "all_in"):
            last_pf_aggr = a.get("seat")
    return last_pf_aggr == my_seat


def _safety_action(state: dict) -> dict:
    """Emergency fallback — used when time budget tight."""
    if state.get("can_check"):
        return {"action": "check"}
    pot = state.get("pot", 0)
    owed = state.get("amount_owed", 0)
    if pot > 0 and owed / max(pot, 1) <= 0.2:
        return {"action": "call"}
    return {"action": "fold"}


def _opponent_seat_for_exploit(state: dict, my_seat: int) -> int | None:
    """Pick the most recent aggressor or first opponent for exploit lookup."""
    log = state.get("action_log", [])
    for a in reversed(log):
        if a.get("seat") != my_seat and a.get("action") in ("raise", "bet", "all_in", "call"):
            return a.get("seat")
    n = len(state.get("players", []))
    for s in range(n):
        if s != my_seat:
            return s
    return None


def _bb(state: dict) -> int:
    """Approximate big blind from the action log."""
    for a in state.get("action_log", []):
        if a.get("action") == "blind" and a.get("kind") == "big":
            return a.get("amount", 100)
    return 100


def decide(game_state: dict) -> dict:
    t0 = time.perf_counter()
    try:
        state = game_state
        my_seat = state["seat_to_act"]
        n = len(state.get("players", []))
        dealer_seat = 0
        # Infer dealer from action log: SB poster == dealer in HU else dealer+1
        for a in state.get("action_log", []):
            if a.get("action") == "blind" and a.get("kind") == "small":
                dealer_seat = a.get("seat", 0) if n == 2 else (a.get("seat", 0) - 1) % n
                break

        # --- Update opponent model ---
        try:
            _OPP.observe_state(state, my_seat)
        except Exception:
            pass  # never fail on stats

        opp_seat = _opponent_seat_for_exploit(state, my_seat)
        opp_stats = _OPP.stats.get(opp_seat) if opp_seat is not None else None
        arche = archetype(opp_stats) if opp_stats else "default"
        origin = fingerprint(opp_stats) if opp_stats else "unknown"
        if PARAMS.get("exploit_enabled"):
            mults = deviation(arche, origin, bracket_mode=bool(PARAMS.get("bracket_mode")))
        else:
            mults = {"bluff_freq": 1.0, "value_thicker": 1.0, "three_bet_light": 1.0, "hero_call": 1.0}

        street = state["street"]
        hole = state["your_cards"]
        community = state.get("community_cards", [])
        pot = state["pot"]
        owed = state["amount_owed"]
        stack = state["your_stack"]
        my_in = state.get("your_bet_this_street", 0)
        can_check = state["can_check"]
        min_raise_to = state["min_raise_to"]
        bb_size = _bb(state)

        # --- Preflop ---
        if street == "preflop":
            has_aggr = _has_aggressor(state)
            facing_3 = _facing_3bet(state)
            decision = preflop_action(
                hole=hole,
                seat_to_act=my_seat,
                dealer_seat=dealer_seat,
                n_players=n,
                has_aggressor=has_aggr,
                facing_3bet=facing_3,
            )

            # Big-bet preflop guard: if call costs >25% of stack, narrow to premium only.
            # Defends against maniac-style 4xBB+ raises before opponent model converges.
            cost_ratio = owed / max(stack + my_in, 1)
            if owed > 0 and cost_ratio > 0.25:
                from .eval.equity import _hand_key as _hk
                key = _hk(hole)
                premium_only = {"AA","KK","QQ","JJ","TT","AKs","AKo","AQs"}
                if key not in premium_only:
                    if can_check:
                        return {"action": "check"}
                    return {"action": "fold"}

            # Apply 3-bet light boost from exploit layer
            if decision == "fold" and not facing_3 and has_aggr and mults["three_bet_light"] > 1.05:
                eq = preflop_equity(hole)
                if eq >= 0.42 and random.random() < (mults["three_bet_light"] - 1.0) * 0.6:
                    decision = "three_bet"

            # Hero-call boost
            if decision == "fold" and has_aggr and not facing_3 and mults["hero_call"] > 1.1:
                eq = preflop_equity(hole)
                if eq >= PARAMS["preflop_call_eq_floor"]:
                    decision = "call"

            if decision == "fold":
                if can_check:
                    return {"action": "check"}
                return {"action": "fold"}
            if decision == "call":
                if can_check:
                    return {"action": "check"}
                return {"action": "call"}
            if decision == "open_raise":
                target = int(bb_size * float(PARAMS["open_raise_bb"]))
                target = max(target, min_raise_to)
                return _legal_raise(target, state)
            if decision == "three_bet":
                target = int(state.get("current_bet", bb_size) * float(PARAMS["three_bet_mult"]))
                target = max(target, min_raise_to)
                return _legal_raise(target, state)
            if decision == "four_bet":
                target = int(state.get("current_bet", bb_size) * float(PARAMS["four_bet_mult"]))
                target = max(target, min_raise_to)
                return _legal_raise(target, state)

            # Fallback
            return {"action": "fold"} if not can_check else {"action": "check"}

        # --- Postflop ---
        # Map street to equity iter budget
        iters_map = {
            "flop":  int(PARAMS["equity_iters_flop"]),
            "turn":  int(PARAMS["equity_iters_turn"]),
            "river": int(PARAMS["equity_iters_river"]),
        }
        iters = iters_map.get(street, 80)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        budget = float(PARAMS["decision_budget_ms"])
        if elapsed_ms > budget * 0.5:
            iters = max(20, int(iters * 0.5))

        is_aggr = _was_preflop_aggressor(state, my_seat)
        post = postflop_action(
            hole=hole,
            community=community,
            pot=pot,
            amount_owed=owed,
            your_stack=stack,
            can_check=can_check,
            is_aggressor=is_aggr,
            bluff_freq_mult=float(mults["bluff_freq"]),
            value_thicker_mult=float(mults["value_thicker"]),
            iters=iters,
        )

        kind = post["kind"]
        if kind == "check":
            return {"action": "check"}
        if kind == "fold":
            # Hero-call boost: if exploit says hero-call, lift fold to call when eq close
            if mults["hero_call"] > 1.1 and pot > 0 and owed / pot < 0.5:
                eq2 = equity_vs_random(hole, community, iters=max(40, iters // 2))
                pot_odds = owed / (pot + owed)
                needed = pot_odds / float(PARAMS["realisation_factor"])
                if eq2 >= needed - 0.05 * (mults["hero_call"] - 1.0):
                    return {"action": "call"}
            return {"action": "fold"}
        if kind == "call":
            return {"action": "call"}
        if kind == "bet":
            size = int(pot * float(post["size_frac"]))
            size = max(size + my_in, min_raise_to)
            return _legal_raise(size, state)
        if kind == "raise":
            size = int((pot + owed) * float(post["size_frac"])) + owed + my_in
            size = max(size, min_raise_to)
            return _legal_raise(size, state)

        return _safety_action(state)

    except BaseException:
        return _safety_action(game_state) if isinstance(game_state, dict) else {"action": "fold"}
