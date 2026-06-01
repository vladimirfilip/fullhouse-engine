#!/usr/bin/env python3
"""A/B harness: vlad_new (A2/A3 profiling wired) vs vlad_base (HEAD, pre-wiring).

Paired / common-random-number design. For each seed we run TWO matches with the
SAME seed and the SAME 5 opponents; only the vlad seat differs (base vs new).
Same seed => same shuffled deck, so opponent holdings start identical and the
only causal difference is vlad's strategy. This isolates the profiling change
far better than putting both variants at one table (where they cannibalise each
other's chips). We compare vlad's chip delta across the two runs.

Usage:
  .venv/Scripts/python.exe tools/ab_profiling.py --seeds 20 --hands 400
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sandbox.match import run_match  # noqa: E402

# 5 fixed opponents; vlad occupies the 6th seat. Field chosen to expose the
# preflop profiling effect (maniac/nit detection drive the open-threshold nudge).
OPPONENTS = {
    "maniac":     "bots/Linglingletsgo_maniac/bot.py",
    "nit":        "bots/Linglingletsgo_nit/bot.py",
    "overfolder": "bots/Linglingletsgo_overfolder/bot.py",
    "tag":        "bots/TobyCoad_tight_aggressive/bot.py",
    "station":    "bots/Linglingletsgo_calling_station/bot.py",
}
VARIANT_PATHS = {
    "new":  "_ab/new/bot.py",
    "base": "_ab/base/bot.py",
}


def _run(variant_path, seed, hands):
    paths = {"vlad": os.path.join(ROOT, variant_path)}
    paths.update({b: os.path.join(ROOT, p) for b, p in OPPONENTS.items()})
    res = run_match(match_id=f"ab_{seed}", bot_paths=paths, n_hands=hands, seed=seed)
    errs = {b: e for b, e in res["bot_errors"].items() if e}
    return res["chip_delta"]["vlad"], errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--hands", type=int, default=400)
    args = ap.parse_args()

    rows = []
    new_t = base_t = 0
    wins = 0
    for seed in range(1, args.seeds + 1):
        new_d, e1 = _run(VARIANT_PATHS["new"], seed, args.hands)
        base_d, e2 = _run(VARIANT_PATHS["base"], seed, args.hands)
        new_t += new_d
        base_t += base_d
        wins += 1 if new_d > base_d else 0
        rows.append((seed, new_d, base_d))
        if e1 or e2:
            print(f"  [seed {seed}] ERRORS new={e1} base={e2}", file=sys.stderr)

    print(f"\n{'seed':>5}  {'new':>10}  {'base':>10}  {'new-base':>10}")
    print("-" * 42)
    for seed, n, b in rows:
        print(f"{seed:>5}  {n:>+10}  {b:>+10}  {n - b:>+10}")
    print("-" * 42)
    n = args.seeds
    print(f"{'TOT':>5}  {new_t:>+10}  {base_t:>+10}  {new_t - base_t:>+10}")
    print(f"\nPer-match avg delta: new {new_t / n:+.0f}   base {base_t / n:+.0f}"
          f"   edge {(new_t - base_t) / n:+.0f} chips/match")
    print(f"new beat base in {wins}/{n} seeds")


if __name__ == "__main__":
    main()
