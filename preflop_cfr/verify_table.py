"""
Sanity-check the exported preflop strategy table on known situations.

For each scenario we drive preflop_cfr.game to the exact decision node (the tree
is card-independent), compute the canonical info-set key, look up the strategy,
mask it to the node's legal actions, and print the distribution.  We then assert
the well-known facts (premiums never fold, trash open-folds, etc.) so a
"ridiculous" distribution fails loudly.

Also checks that bot.py's mirrored key encoding produces the SAME key for a spot,
i.e. the table the bot would actually query end-to-end.

Run:
    python -m preflop_cfr.verify_table
"""
from __future__ import annotations

import os
import sys

import eval7
import numpy as np

from preflop_cfr import config
from preflop_cfr.cards import hand_to_bucket
from preflop_cfr.cfr import _infoset_key as _canon_key
from preflop_cfr.export import load_strategy
from preflop_cfr.game import (
    make_initial_state, legal_actions, apply_action, is_terminal,
)

_ACT_NAME = {0: "FOLD", 1: "CHK/CALL", 3: "RAISE-1/3", 5: "RAISE-pot", 8: "ALL-IN"}


def _cards(label: str) -> list:
    """Concrete eval7 cards for a hand label (AA / AKs / AKo)."""
    r1, r2 = label[0], label[1]
    if len(label) == 2:                       # pair
        return [eval7.Card(r1 + "s"), eval7.Card(r2 + "h")]
    if label[2] == "s":
        return [eval7.Card(r1 + "s"), eval7.Card(r2 + "s")]
    return [eval7.Card(r1 + "s"), eval7.Card(r2 + "h")]


def _drive(actions: list[int]):
    """Apply a sequence of abstract actions from a fresh 6-max hand (dealer=0)."""
    state = make_initial_state(dealer_seat=0)
    for a in actions:
        assert not is_terminal(state) and state.to_act != -1, "line ended early"
        state = apply_action(state, a)
    assert not is_terminal(state) and state.to_act != -1, "reached terminal node"
    return state


def _lookup(table, state, label):
    bucket = hand_to_bucket(*_cards(label))
    key    = _canon_key(state, state.to_act, bucket)   # same builder as training
    legal  = legal_actions(state)
    probs  = table.get(key)
    if probs is None:
        return None, legal, key
    masked = np.array([max(probs[a], 0.0) for a in legal], dtype=np.float64)
    s = masked.sum()
    masked = masked / s if s > 0 else np.full(len(legal), 1.0 / len(legal))
    return dict(zip(legal, masked)), legal, key


def _fmt(dist) -> str:
    return "  ".join(f"{_ACT_NAME[a]} {p*100:4.1f}%" for a, p in dist.items())


# (description, line-to-reach-node, hand, assertion(dist)->bool, why)
F, C, T, P, A = 0, 1, 3, 5, 8  # FOLD, CALL, THIRD, FULL(pot), ALLIN

SCENARIOS = [
    ("UTG RFI  AA",  [],            "AA",  lambda d: d.get(F, 0) < .01 and (d.get(T,0)+d.get(P,0)+d.get(A,0)) > .9, "premium must open, never fold"),
    ("UTG RFI  72o", [],            "72o", lambda d: d.get(F, 0) > .9, "trash open-folds UTG"),
    ("UTG RFI  KJo", [],            "KJo", lambda d: d.get(T,0)+d.get(P,0) > .9, "KJo opens UTG (in chart)"),
    ("UTG RFI  K9o", [],            "K9o", lambda d: d.get(F, 0) > .9, "K9o folds UTG (out of chart)"),
    ("BTN RFI  A5s", [F, F, F],     "A5s", lambda d: d.get(T,0)+d.get(P,0) > .9, "BTN opens wide incl A5s"),
    ("BTN RFI  J2o", [F, F, F],     "J2o", lambda d: d.get(F, 0) > .9, "J2o folds even on BTN"),
    ("BB vs BTN open  QQ",  [F, F, F, T, F], "QQ",  lambda d: d.get(P,0)+d.get(A,0) > .6, "QQ 3-bets vs steal"),
    ("BB vs BTN open  72o", [F, F, F, T, F], "72o", lambda d: d.get(F, 0) > .8, "72o folds vs open"),
    ("BB vs BTN open  T9s", [F, F, F, T, F], "T9s", lambda d: d.get(C, 0) > .5, "T9s flat-defends BB"),
    ("CO vs UTG open  AKs", [T, F], "AKs", lambda d: d.get(P,0)+d.get(A,0) > .6, "AKs 3-bets vs UTG open"),
    ("CO vs UTG open  KQo", [T, F], "KQo", lambda d: d.get(F, 0) > .6, "KQo folds vs UTG open (deep)"),
    # UTG opens, MP 3-bets (pot); CO faces the 3-bet (within the 4-action window):
    ("CO vs 3bet  AA",  [T, P], "AA",  lambda d: d.get(A,0)+d.get(P,0) > .6, "AA 4-bets/jams vs 3-bet"),
    ("CO vs 3bet  QQ",  [T, P], "QQ",  lambda d: d.get(C,0) > .4 or d.get(F,0) > .4, "QQ flats or folds, not auto-jam"),
    ("CO vs 3bet  AJo", [T, P], "AJo", lambda d: d.get(F, 0) > .8, "AJo folds vs a 3-bet deep"),
]


def main() -> None:
    table = load_strategy()
    print(f"loaded {len(table):,} info sets from {config.EXPORT_PATH}\n")

    failures = 0
    misses = 0
    for desc, line, hand, check, why in SCENARIOS:
        state = _drive(line)
        dist, legal, key = _lookup(table, state, hand)
        if dist is None:
            print(f"  MISS  {desc:24s}  (no key in table; legal={legal})")
            misses += 1
            continue
        ok = check(dist)
        flag = "  ok " if ok else "FAIL "
        if not ok:
            failures += 1
        # sums to 1?
        assert abs(sum(dist.values()) - 1.0) < 1e-6, f"{desc}: probs don't sum to 1"
        print(f"{flag} {desc:24s}  {_fmt(dist)}")
        if not ok:
            print(f"        ^ expected: {why}")

    print()
    _parity_check()

    print()
    if failures or misses:
        print(f"RESULT: {failures} bad distribution(s), {misses} miss(es) — NOT clean")
        sys.exit(1)
    print("RESULT: all known situations sane, all keys present — clean")


def _parity_check() -> None:
    """Confirm bot.py's mirrored key encoding matches the canonical one."""
    bot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "bots", "vlad")
    if bot_dir not in sys.path:
        sys.path.insert(0, bot_dir)
    import bot as _bot  # noqa: E402

    # UTG RFI with AA: action_log has only the posted blinds.
    gs = {
        "your_cards": ["As", "Ah"],
        "seat_to_act": 3,                       # UTG (dealer=0, SB=1, BB=2)
        "action_log": [
            {"seat": 1, "action": "small_blind", "amount": config.SMALL_BLIND},
            {"seat": 2, "action": "big_blind",   "amount": config.BIG_BLIND},
        ],
        "players": [{"seat": i, "stack": config.INITIAL_STACK} for i in range(6)],
    }
    bot_key = _bot._preflop_infoset_key(gs)
    state = _drive([])                       # UTG to act, no prior actions
    canon_key = _canon_key(state, state.to_act, hand_to_bucket(*_cards("AA")))
    match = bot_key == canon_key
    print(f"key parity (UTG AA RFI): bot={bot_key}  canonical={canon_key}  "
          f"{'MATCH' if match else 'MISMATCH'}")
    if not match:
        raise SystemExit("bot.py key encoding diverged from the table — lookups would miss")


if __name__ == "__main__":
    main()
