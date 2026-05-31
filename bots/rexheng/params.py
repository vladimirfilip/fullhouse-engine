"""Tunable parameters — the evolution surface.

Ralph Loop mutates these. Bot reads from a `PARAMS` dict so an evolved bot is
just this file with different numbers.
"""
from __future__ import annotations

PARAMS: dict[str, float | bool] = {
    # --- Preflop ---
    "open_raise_bb": 2.5,         # open size in BB
    "three_bet_mult": 3.0,        # 3-bet size as multiple of last raise
    "four_bet_mult": 2.3,         # 4-bet size as multiple of 3-bet
    "preflop_call_eq_floor": 0.42, # min equity vs random to call a raise

    # --- Postflop ---
    "cbet_freq_dry": 0.75,
    "cbet_freq_wet": 0.45,
    "cbet_size_dry_frac": 0.5,
    "cbet_size_wet_frac": 0.66,
    "value_size_frac": 0.66,
    "thin_value_eq_floor": 0.55,
    "value_eq_floor": 0.62,
    "raise_eq_margin": 0.18,      # extra equity over needed to raise vs call
    "bluff_raise_freq": 0.10,
    "realisation_factor": 0.85,

    # --- Aggression / safety ---
    "all_in_eq_floor": 0.60,      # only shove if eq above this
    "fold_to_3bet_eq_floor": 0.45,

    # --- Exploit ---
    "exploit_enabled": True,
    "fingerprint_min_hands": 20,
    "archetype_min_hands": 12,

    # --- Bracket mode (clamps exploit) ---
    "bracket_mode": False,

    # --- Time / safety ---
    "decision_budget_ms": 800.0,  # internal soft target (engine allows 2000)
    "equity_iters_preflop": 0,    # 0 = use cached
    "equity_iters_flop": 120,
    "equity_iters_turn": 90,
    "equity_iters_river": 60,
}
