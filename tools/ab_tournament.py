#!/usr/bin/env python3
"""ab_tournament.py — paired A/B on the real tournament metric (cum chip delta).

The plan's hard-won validation lesson (IMPROVEMENT_PLAN.md): small-field bb/100
is unreliable (opponents call `random` unseeded → broken CRN), so the ONLY metric
that predicts placement is a representative Swiss tournament's cumulative chip
delta. This harness runs that tournament with BOTH the candidate (`bots/vlad`) and
a frozen baseline (`bots/vlad_base`) in the SAME field, over N seeds, and reports
whether the candidate out-places the baseline.

Workflow:
    # 1. Freeze the current bot as the baseline (do this ONCE, before tuning):
    python tools/ab_tournament.py --snapshot --seeds 1

    # 2. Edit bots/vlad/bot.py, then A/B it against the frozen baseline:
    python tools/ab_tournament.py --seeds 1 2 3

Per-bot env tuning is NOT possible (sandbox bots inherit the parent env, so every
vlad copy would see the same VLAD_* knobs). Tune by editing bots/vlad/bot.py; the
frozen bots/vlad_base stays fixed as the comparison floor and submission baseline.

Reuses run_tournament's match primitives, so the seed→deck mapping and Swiss
pairing are identical to a real `python run_tournament.py` run.
"""

import argparse
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.tournament import compute_standings, swiss_pairing  # noqa: E402
from run_tournament import (  # noqa: E402
    TABLE_SIZE,
    _discover_bots,
    _run_one_match,
)

BOTS_DIR = ROOT / "bots"

# Representative fields. "rep" spans the archetype space the way the real 86-bot
# Swiss does: tuned leaders (neel sweeps), strong non-neel (skant / cfr_equity),
# mid bots (saroop / older neel), and the explicit weak archetypes. Keeping it
# ~15 opponents means new + base + field ≈ 17 bots → 3 tables/round, like the
# real event. Edit freely; any name must be a directory under bots/.
FIELDS = {
    "rep": [
        # tuned leaders
        "neel_v6_sweep_000", "neel_v6_sweep_002", "neel_v6_sweep_004",
        # strong non-neel
        "Pav1602_skantbot4", "cfr_equity_v28", "neel_mit_plus", "neel_v2_harmonic",
        # mid
        "saroopjagdev_mybot", "Pav1602_skantbot2", "neel_v6_sweep_005",
        # weak archetypes (offense test)
        "Linglingletsgo_calling_station", "Linglingletsgo_maniac",
        "Linglingletsgo_nit", "Linglingletsgo_overfolder",
        "Linglingletsgo_balanced_shark",
    ],
    # Flood-free mirror of "rep": same strong+weak archetype spread, but excludes
    # the neel_v6_sweep_* / neel_v*_profile bots, which dump tracebacks to an
    # undrained stderr pipe on their exception path and intermittently deadlock the
    # match runner (sandbox/match.py never reads the bot's stderr in headless mode,
    # so a flood fills the 64KB pipe and readline() hangs forever). Use this for
    # reliable multi-seed A/Bs.
    "stable": [
        # strong leaders (flood-free)
        "cfr_equity_v28", "neel_mit_plus", "neel_v2_harmonic",
        # strong non-neel
        "Pav1602_skantbot4", "Pav1602_skantbot2", "saroopjagdev_mybot",
        "shark", "mathematician",
        # mid / varied strong
        "rexheng", "Littleguygabe_mybot", "anti_monte_carlo",
        # weak archetypes (offense test)
        "Linglingletsgo_calling_station", "Linglingletsgo_maniac",
        "Linglingletsgo_nit", "Linglingletsgo_overfolder",
    ],
    # Top 12 of the current pool: the distinct strong archetypes that ranked
    # high in BOTH finished full-field (86-bot) runs. Derived from the raw top
    # 12 of tournament_20260601_000617, but the 8 near-duplicate neel_v6_sweep_*
    # variants (which swap ranks by variance) are collapsed to the single best
    # sweep that appears in both runs, and the freed slots are filled with the
    # next stable distinct bots. Use as the sparring field for the real metric.
    "top12": [
        "neel_v6_sweep_002", "Pav1602_skantbot4", "neel_v6_sweep_004",
        "neel_v2_harmonic", "neel_v5_partition", "neel_range_tracker",
        "neel_v2_riskgate", "cfr_equity_v28", "neel_v3_gemini",
        "neel_v2", "neel_robust_hybrid", "saroopjagdev_mybot",
    ],
    # Strong-only: where defensive modules should show. Faster (fewer bots).
    "strong": [
        "neel_v6_sweep_000", "neel_v6_sweep_004", "Pav1602_skantbot4",
        "cfr_equity_v28", "neel_mit_plus", "neel_v2_harmonic",
    ],
    # Weak-only: offense / value-extraction test.
    "weak": [
        "Linglingletsgo_calling_station", "Linglingletsgo_maniac",
        "Linglingletsgo_nit", "Linglingletsgo_overfolder",
        "Linglingletsgo_balanced_shark", "Linglingletsgo_pot_bluffer",
    ],
}


def _snapshot_baseline(new_name: str, base_name: str) -> None:
    """Copy bots/<new_name>/{bot.py,data} → bots/<base_name>/ (overwrites)."""
    src = BOTS_DIR / new_name
    dst = BOTS_DIR / base_name
    if not (src / "bot.py").exists():
        sys.exit(f"ERROR: {src/'bot.py'} not found — nothing to snapshot.")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copy2(src / "bot.py", dst / "bot.py")
    if (src / "data").exists():
        shutil.copytree(src / "data", dst / "data")
    print(f"Froze baseline: {src} -> {dst}")


def _run_seed(bots: dict, rounds: int, hands: int, seed: int, workers: int) -> list:
    """Run one headless Swiss tournament; return final standings (ranked list).

    Mirrors run_tournament._run_tournament minus all console output. Match seeds
    come from _run_one_match, so the deck stream matches a real run at this seed.
    """
    standings = [
        {"bot_id": bid, "bot_path": path, "cumulative_delta": 0,
         "matches_played": 0, "best_match_delta": 0}
        for bid, path in bots.items()
    ]
    all_results: list = []

    for round_num in range(1, rounds + 1):
        tables = swiss_pairing(standings, table_size=TABLE_SIZE)
        with ThreadPoolExecutor(max_workers=min(workers, len(tables))) as pool:
            futures = [
                pool.submit(_run_one_match, round_num, ti, table, hands, False, seed)
                for ti, table in enumerate(tables, 1)
            ]
            for fut in as_completed(futures):
                _match_id, _ti, bot_paths, result = fut.result()
                for bid in result["bot_ids"]:
                    all_results.append({
                        "bot_id": bid,
                        "bot_path": bot_paths[bid],
                        "chip_delta": result["chip_delta"][bid],
                    })
        standings = compute_standings(all_results)

    return standings


def _placement(standings: list, bot_id: str) -> tuple[int, int]:
    """(rank, cumulative_delta) for bot_id; rank is 1-based."""
    for i, b in enumerate(standings, 1):
        if b["bot_id"] == bot_id:
            return i, b["cumulative_delta"]
    return len(standings) + 1, 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paired A/B (candidate vs frozen baseline) on the Swiss "
                    "cum-delta metric.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--new", default="vlad", help="Candidate bot dir name.")
    ap.add_argument("--base", default="vlad_base", help="Frozen baseline bot dir name.")
    ap.add_argument("--snapshot", action="store_true",
                    help="Freeze --new → --base before running, then run.")
    ap.add_argument("--field", default="rep", choices=list(FIELDS),
                    help="Opponent field preset.")
    ap.add_argument("--extra-bots", nargs="+", default=[], metavar="BOT",
                    help="Extra opponent dir names to append to the field.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="Tournament seeds (each is one paired run).")
    ap.add_argument("--rounds", type=int, default=6, help="Swiss rounds per seed.")
    ap.add_argument("--hands", type=int, default=400, help="Hands per match.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel tables within a round.")
    args = ap.parse_args()

    if args.snapshot:
        _snapshot_baseline(args.new, args.base)

    if not (BOTS_DIR / args.base / "bot.py").exists():
        sys.exit(f"ERROR: baseline {args.base!r} not found. Run once with "
                 f"--snapshot to freeze the current bot first.")

    field = FIELDS[args.field] + list(args.extra_bots)
    want = [args.new, args.base] + field
    bots = _discover_bots(str(BOTS_DIR), include=set(want))
    missing = [b for b in want if b not in bots]
    if missing:
        sys.exit(f"ERROR: field bots not found under bots/: {missing}")

    print("=" * 68)
    print(f"A/B: {args.new}  vs  {args.base}")
    print(f"Field '{args.field}': {len(field)} opponents  "
          f"(total {len(bots)} bots / table {TABLE_SIZE})")
    print(f"Seeds {args.seeds}   rounds {args.rounds}   hands {args.hands}")
    print("=" * 68)

    rows = []
    for seed in args.seeds:
        t0 = time.time()
        standings = _run_seed(bots, args.rounds, args.hands, seed, args.workers)
        n = len(standings)
        nr, nd = _placement(standings, args.new)
        br, bd = _placement(standings, args.base)
        rows.append((seed, nr, nd, br, bd))
        print(f"seed {seed:>3}  |  NEW #{nr}/{n} {nd:>+8}   "
              f"BASE #{br}/{n} {bd:>+8}   "
              f"d(new-base) {nd - bd:>+8}   ({time.time() - t0:.0f}s)")

    print("-" * 68)
    new_ranks = [r[1] for r in rows]
    new_deltas = [r[2] for r in rows]
    base_ranks = [r[3] for r in rows]
    base_deltas = [r[4] for r in rows]
    paired = [r[2] - r[4] for r in rows]
    wins = sum(1 for p in paired if p > 0)

    def _mean(xs):
        return statistics.mean(xs) if xs else 0.0

    print(f"NEW   mean rank {_mean(new_ranks):.2f}   mean delta {_mean(new_deltas):>+10.0f}")
    print(f"BASE  mean rank {_mean(base_ranks):.2f}   mean delta {_mean(base_deltas):>+10.0f}")
    print(f"PAIRED new-base: mean {_mean(paired):>+10.0f}   "
          f"new out-places base on {wins}/{len(rows)} seeds")
    if len(paired) >= 2:
        sd = statistics.pstdev(paired)
        print(f"  paired spread (pstdev) {sd:>.0f}   "
              f"per-seed d {[f'{p:+}' for p in paired]}")

    better_rank = _mean(new_ranks) < _mean(base_ranks)
    verdict = ("SHIP - new out-places base" if (wins > len(rows) / 2 and better_rank)
               else "INCONCLUSIVE - needs more seeds" if wins == len(rows) / 2
               else "REGRESSION - keep baseline")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
