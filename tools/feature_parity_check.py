#!/usr/bin/env python3
"""feature_parity_check.py — assert the C++ and Python feature builders agree.

The 252-dim feature vector must be byte-for-byte identical across
deep_cfr_cpp/src/features.cpp and bots/the_house/bot.py `_build_feature_vector`,
or training fits one layout and inference reads another (silent corruption — the
bot loads weight SHAPES from the .npz, so a layout drift is invisible at load).

Run AFTER rebuilding the C++ extension (so deep_cfr_gen.build_features exists and
deep_cfr_gen.INPUT_DIM matches):

    cmake --build deep_cfr_cpp/build --config Release     # (Linux: make build-cpp)
    python tools/feature_parity_check.py

Exits non-zero on any mismatch.
"""
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# locate the built extension
for sub in ("deep_cfr_cpp/build/Release", "deep_cfr_cpp/build"):
    p = ROOT / sub
    if p.exists():
        sys.path.insert(0, str(p))
import deep_cfr_gen  # noqa: E402

# load the production mirror
_spec = importlib.util.spec_from_file_location(
    "_house", str(ROOT / "bots" / "the_house" / "bot.py"))
_house = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_house)

RANKS, SUITS = "23456789TJQKA", "shdc"
DECK = [r + s for r in RANKS for s in SUITS]
DEALER = 0  # SB=seat1, BB=seat2 -> bot derives the same dealer from the SB post


def _players(folds, stacks=None, bets=None, allin=()):
    out = []
    for i in range(6):
        out.append({
            "seat": i, "bot_id": f"b{i}",
            "stack": (stacks or {}).get(i, 9000),
            "state": "folded" if i in folds else ("all_in" if i in allin else "active"),
            "is_folded": i in folds, "is_all_in": i in allin,
            "bet_this_street": (bets or {}).get(i, 0),
        })
    return out


def _gs(cards, board, street, seat, owed, cur, pot, action_log, folds=(), allin=()):
    return {
        "street": street, "seat_to_act": seat, "pot": pot,
        "community_cards": board, "current_bet": cur,
        "min_raise_to": cur + 100, "amount_owed": owed, "can_check": owed == 0,
        "your_cards": cards, "your_stack": 9000, "your_bet_this_street": 0,
        "players": _players(folds, allin=allin), "hand_id": "parity_h1",
        "action_log": action_log, "match_action_log": [],
    }


def _compare(name, gs):
    bot_vec = _house._build_feature_vector(gs).astype(np.float32)
    n_raises = _house._derive_n_raises_this_street(gs["action_log"], len(gs["players"]))
    cpp_in = dict(gs)
    cpp_in["dealer_seat"] = DEALER
    cpp_in["n_raises_this_street"] = int(n_raises)
    cpp_vec = np.asarray(deep_cfr_gen.build_features(cpp_in), dtype=np.float32)
    if bot_vec.shape != cpp_vec.shape:
        print(f"FAIL {name}: shape {bot_vec.shape} (bot) vs {cpp_vec.shape} (cpp)")
        return False
    # Compare with a tolerance: C++ (Eigen float) and numpy round the few divided
    # / log-scaled features differently in the last float32 bit (~1e-7). A real
    # LAYOUT error flips a one-hot 0<->1 (diff = 1.0), far above this tolerance.
    d = np.abs(bot_vec.astype(np.float64) - cpp_vec.astype(np.float64))
    maxdiff = float(d.max())
    TOL = 1e-4
    if maxdiff > TOL:
        idx = np.where(d > TOL)[0]
        print(f"FAIL {name}: {len(idx)} differing indices (max |Δ|={maxdiff:.2e}):")
        for i in idx[:8]:
            print(f"    [{i}] bot={bot_vec[i]:.5f}  cpp={cpp_vec[i]:.5f}")
        return False
    print(f"ok   {name}  (max |Δ|={maxdiff:.1e})")
    return True


def main():
    assert deep_cfr_gen.INPUT_DIM == _house._INPUT_DIM == 252, (
        f"INPUT_DIM mismatch: cpp={deep_cfr_gen.INPUT_DIM} "
        f"bot={_house._INPUT_DIM} (expected 252) — rebuild the extension")
    bl = [{"seat": 1, "action": "small_blind", "amount": 50},
          {"seat": 2, "action": "big_blind", "amount": 100}]
    spots = {
        "preflop_utg_unopened":
            _gs(["As", "Kh"], [], "preflop", 3, 100, 100, 150, list(bl)),
        "preflop_facing_3bet":
            _gs(["Qd", "Qc"], [], "preflop", 5, 900, 900, 1400,
                bl + [{"seat": 3, "action": "raise", "amount": 300},
                      {"seat": 4, "action": "raise", "amount": 900}]),
        "flop_checked_to":
            _gs(["Ah", "Kd"], ["Qh", "7c", "2d"], "flop", 3, 0, 0, 700,
                bl + [{"seat": 3, "action": "raise", "amount": 300},
                      {"seat": 0, "action": "call", "amount": 300}],
                folds=(1, 2, 4, 5)),
        "turn_facing_bet":
            _gs(["Th", "Td"], ["Qh", "7h", "2h", "9c"], "turn", 3, 800, 800, 2400,
                bl + [{"seat": 3, "action": "raise", "amount": 300},
                      {"seat": 5, "action": "call", "amount": 300},
                      {"seat": 5, "action": "raise", "amount": 800}],
                folds=(0, 1, 2, 4)),
        "river_paired_board":
            _gs(["Ah", "Jd"], ["Td", "4d", "Qs", "Tc", "2c"], "river", 3,
                0, 0, 3000,
                bl + [{"seat": 3, "action": "raise", "amount": 300},
                      {"seat": 0, "action": "call", "amount": 300},
                      {"seat": 3, "action": "raise", "amount": 900},
                      {"seat": 0, "action": "call", "amount": 900}],
                folds=(1, 2, 4, 5)),
    }
    ok = all(_compare(n, g) for n, g in spots.items())

    # Randomised fuzz: long varied action logs to exercise the 16-slot ring,
    # last-aggressor, texture, and all renumbered indices.
    rng = random.Random(20260603)
    for t in range(200):
        deck = DECK[:]
        rng.shuffle(deck)
        nb = rng.choice([0, 3, 4, 5])
        cards = deck[:2]
        board = deck[2:2 + nb]
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[nb]
        al = list(bl)
        for _ in range(rng.randint(0, 22)):   # may exceed 16 -> tests ring wrap
            al.append({"seat": rng.randint(0, 5),
                       "action": rng.choice(["fold", "call", "check", "raise", "all_in"]),
                       "amount": rng.choice([0, 100, 300, 800, 2000])})
        owed = rng.choice([0, 200, 800])
        cur = owed
        gs = _gs(cards, board, street, rng.randint(0, 5), owed, cur,
                 rng.randint(150, 6000), al)
        if not _compare(f"fuzz_{t}", gs):
            ok = False
            break

    print("\n" + ("ALL PARITY CHECKS PASSED" if ok else "PARITY FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
