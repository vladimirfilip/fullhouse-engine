#!/usr/bin/env python3
"""Low-variance A/B: bb/100 with per-hand stack reset + common random numbers.

Why this exists: match-end chip delta is dominated by all-or-nothing bust/snowball
events (±50k swings), so it cannot resolve per-decision strategy edges. Here we:

  * reset every player to STARTING_STACK each hand (no busts, no snowballing) —
    each hand is an independent cash-game sample;
  * accumulate match_action_log across hands so vlad still builds opponent reads
    (essential to test the profiling / range / anti-punt modules);
  * pair the two variants by hand seed (common random numbers): hand i uses the
    same deck + dealer for both new and base, so shared card luck cancels in the
    per-hand difference.

Reports each variant's bb/100 and the paired new-base difference with a 95% CI.

Usage:
  .venv/Scripts/python.exe tools/eval_bb100.py --hands 1500 --field weak
"""

import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.game import PokerEngine, STARTING_STACK, BIG_BLIND  # noqa: E402
from sandbox.match import BotProcess, _inject_match_log  # noqa: E402

FIELDS = {
    # Weak/exploitable: base already crushes this; tests offensive value.
    "weak": {
        "maniac":     "bots/Linglingletsgo_maniac/bot.py",
        "nit":        "bots/Linglingletsgo_nit/bot.py",
        "overfolder": "bots/Linglingletsgo_overfolder/bot.py",
        "tag":        "bots/TobyCoad_tight_aggressive/bot.py",
        "station":    "bots/Linglingletsgo_calling_station/bot.py",
    },
    # Strong: where defensive refinements (B/C) should earn their keep.
    "strong": {
        "saroop":  "bots/saroopjagdev_mybot/bot.py",
        "neel":    "bots/neel_v6_oppprofile/bot.py",
        "skant":   "bots/Pav1602_skantbot4/bot.py",
        "tag":     "bots/TobyCoad_tight_aggressive/bot.py",
        "dominic": "bots/Linglingletsgo_dominic/bot.py",
    },
}
VARIANT_PATHS = {"new": "_ab/new/bot.py", "base": "_ab/base/bot.py"}


def _play_hand(engine, procs, bot_ids, match_log, hand_num):
    state = _inject_match_log(engine.start_hand(), match_log)
    steps = 0
    while state.get("type") == "action_request":
        seat = state["seat_to_act"]
        bid = bot_ids[seat]
        action = procs[bid].act(state)
        match_log.append({"hand_num": hand_num, "seat": seat, "bot_id": bid,
                          "action": action.get("action"), "amount": action.get("amount")})
        state = _inject_match_log(engine.apply_action(seat, action), match_log)
        steps += 1
        if steps > 1000:
            break
    return state.get("final_stacks", {})


def _run_variant(vlad_path, field, hands, base_seed):
    """Return list of vlad's per-hand chip deltas (stacks reset each hand)."""
    bot_paths = {"vlad": os.path.join(ROOT, vlad_path)}
    bot_paths.update({b: os.path.join(ROOT, p) for b, p in field.items()})
    bot_ids = list(bot_paths)
    n = len(bot_ids)

    procs = {bid: BotProcess(bid, path) for bid, path in bot_paths.items()}
    for p in procs.values():
        p.warmup()

    match_log = []
    deltas = []
    try:
        for i in range(hands):
            engine = PokerEngine(hand_id=f"e_h{i:05d}", bot_ids=bot_ids,
                                 dealer_seat=i % n, seed=base_seed + i)
            final = _play_hand(engine, procs, bot_ids, match_log, i)
            deltas.append(final.get("vlad", STARTING_STACK) - STARTING_STACK)
    finally:
        errs = {b: pr.errors[:3] for b, pr in procs.items() if pr.errors}
        for p in procs.values():
            p.stop()
    if errs:
        print(f"  [{vlad_path}] bot errors: {errs}", file=sys.stderr)
    return deltas


def _bb100(deltas):
    m = sum(deltas) / len(deltas)
    return m / BIG_BLIND * 100.0  # = mean chips/hand (BB=100), in bb/100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--field", choices=list(FIELDS), default="weak")
    ap.add_argument("--seed", type=int, default=10_000)
    args = ap.parse_args()
    field = FIELDS[args.field]

    print(f"Field: {args.field} ({', '.join(field)})  hands={args.hands}")
    new = _run_variant(VARIANT_PATHS["new"], field, args.hands, args.seed)
    base = _run_variant(VARIANT_PATHS["base"], field, args.hands, args.seed)

    diffs = [n - b for n, b in zip(new, base)]
    md = sum(diffs) / len(diffs)
    var = sum((d - md) ** 2 for d in diffs) / max(1, len(diffs) - 1)
    se = math.sqrt(var / len(diffs))
    ci = 1.96 * se
    t = md / se if se > 0 else 0.0

    print(f"\n{'variant':<8} {'bb/100':>10}")
    print("-" * 20)
    print(f"{'new':<8} {_bb100(new):>10.2f}")
    print(f"{'base':<8} {_bb100(base):>10.2f}")
    print(f"\nPaired new-base: {md / BIG_BLIND * 100:+.2f} bb/100  "
          f"(95% CI ±{ci / BIG_BLIND * 100:.2f}, t={t:+.2f}, n={len(diffs)})")
    sig = "SIGNIFICANT" if abs(t) > 1.96 else "not significant"
    print(f"Verdict: {sig} at 95%.")


if __name__ == "__main__":
    main()
