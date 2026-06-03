#!/usr/bin/env python3
"""rep_field_test.py - representative-field harness.

The all-strong top-12 table is an unrepresentative worst case (the weakest of 6
elite bots always busts). The real 86-bot Swiss exposes a bot to a MIX of
strong / mid / weak opponents, where chips are won by farming the weak/mid field
and not catastrophically busting to the strong one. This harness approximates
that exposure: it runs many random 6-bot tables drawn from a representative pool
(the test bot always seated), in parallel, and reports the test bot's mean chip
delta, mean rank, and how often it finishes top-half / busts.

Usage:
    python tools/rep_field_test.py --bot the_house --tables 48 --hands 400
    python tools/rep_field_test.py --bot the_house --bot2 vlad_base   # A/B in same tables
"""
import argparse
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sandbox.match import run_match  # noqa: E402

BOTS_DIR = ROOT / "bots"

# Representative pool spanning the archetype space (flood-free: no neel_v6_sweep_*
# / neel_v*_profile bots, which can deadlock the headless runner via stderr flood).
POOL = {
    "strong": ["cfr_equity_v28", "neel_v2_harmonic", "Pav1602_skantbot4",
               "neel_robust_hybrid", "saroopjagdev_mybot", "neel_v2"],
    "mid":    ["shark", "mathematician", "rexheng", "Littleguygabe_mybot",
               "anti_monte_carlo", "Pav1602_skantbot2"],
    "weak":   ["Linglingletsgo_calling_station", "Linglingletsgo_maniac",
               "Linglingletsgo_nit", "Linglingletsgo_overfolder",
               "Linglingletsgo_pot_bluffer", "Linglingletsgo_balanced_shark"],
}


def _path(name):
    return str(BOTS_DIR / name / "bot.py")


def _draw_table(rng, test_bots):
    """test_bots + 1 strong + 1 mid + (6-len-2) weak/mixed, shuffled to 6 seats."""
    others = []
    others.append(rng.choice(POOL["strong"]))
    others.append(rng.choice(POOL["mid"]))
    rest_pool = POOL["strong"] + POOL["mid"] + POOL["weak"]
    while len(test_bots) + len(others) < 6:
        c = rng.choice(rest_pool)
        if c not in others:
            others.append(c)
    return others


def _run_one(args):
    idx, seed, test_bots = args
    rng = random.Random(seed)
    others = _draw_table(rng, test_bots)
    names = list(test_bots) + others
    bot_paths = {n: _path(n) for n in names}
    res = run_match(f"rep{idx}", bot_paths, n_hands=HANDS, verbose=False, seed=seed)
    deltas = res["chip_delta"]
    ranked = sorted(names, key=lambda b: -deltas[b])
    out = {}
    for tb in test_bots:
        out[tb] = (deltas[tb], ranked.index(tb) + 1, len(names))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="the_house")
    ap.add_argument("--bot2", default=None, help="second bot seated in the SAME tables (A/B)")
    ap.add_argument("--tables", type=int, default=48)
    ap.add_argument("--hands", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    global HANDS
    HANDS = a.hands

    test_bots = [a.bot] + ([a.bot2] if a.bot2 else [])
    seeds = [random.randint(1, 10 ** 9) for _ in range(a.tables)]
    jobs = [(i, s, test_bots) for i, s in enumerate(seeds)]

    agg = {b: {"delta": [], "rank": []} for b in test_bots}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_run_one, jobs):
            for b, (d, r, n) in out.items():
                agg[b]["delta"].append(d)
                agg[b]["rank"].append(r)

    print(f"REPRESENTATIVE FIELD: {a.tables} random 6-bot tables x {a.hands} hands")
    print("-" * 64)
    for b in test_bots:
        ds, rs = agg[b]["delta"], agg[b]["rank"]
        tot = sum(ds)
        top_half = sum(1 for r in rs if r <= 3) / len(rs)
        busts = sum(1 for d in ds if d <= -10000) / len(ds)
        wins = sum(1 for r in rs if r == 1) / len(rs)
        print(f"{b:12} total {tot:+9d}  mean/table {statistics.mean(ds):+8.0f}  "
              f"mean rank {statistics.mean(rs):.2f}/6  win {wins:.0%}  "
              f"top3 {top_half:.0%}  bust {busts:.0%}")


if __name__ == "__main__":
    main()
