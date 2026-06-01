#!/usr/bin/env python3
"""
run_tournament.py — Local Swiss-system tournament runner.

Discovers all bots in bots/, runs a Swiss tournament (6-bot tables,
400 hands/match by default), and writes all output to
tournament_logs/tournament_<timestamp>.txt.

Rounds are sequential (Swiss pairing depends on prior results), but
tables within a round run concurrently via ThreadPoolExecutor.

Usage:
  python run_tournament.py
  python run_tournament.py --rounds 3 --hands 400 --verbose
  python run_tournament.py --bots aggressor shark vlad
  python run_tournament.py --seed 42
  python run_tournament.py --workers 4
"""

import argparse
import datetime
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.tournament import (  # noqa: E402
    compute_standings,
    select_finalists,
    swiss_pairing,
)
from sandbox.match import run_match  # noqa: E402

BOTS_DIR = ROOT / "bots"
LOGS_DIR = ROOT / "tournament_logs"
TABLE_SIZE = 6

# Named opponent fields, selectable with --field. Kept in sync with the same
# presets in tools/ab_tournament.py.
FIELDS = {
    # Top 12 of the current pool: the distinct strong archetypes that ranked
    # high in BOTH finished full-field (86-bot) runs. The near-duplicate
    # neel_v6_sweep_* variants (which swap ranks by variance) are collapsed to
    # the single best sweep present in both runs; freed slots go to the next
    # stable distinct bots.
    "top12": [
        "neel_v6_sweep_002", "Pav1602_skantbot4", "neel_v6_sweep_004",
        "neel_v2_harmonic", "neel_v5_partition", "neel_range_tracker",
        "neel_v2_riskgate", "cfr_equity_v28", "neel_v3_gemini",
        "neel_v2", "neel_robust_hybrid", "saroopjagdev_mybot",
    ],
}
DEFAULT_ROUNDS = 3
DEFAULT_HANDS = 400
DEFAULT_WORKERS = os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Tee: write to both an original stream and a log file
# ---------------------------------------------------------------------------

class _TeeStream:
    def __init__(self, original, logfile):
        self._orig = original
        self._log = logfile

    def write(self, data):
        self._orig.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._orig.flush()
        self._log.flush()

    def fileno(self):
        return self._orig.fileno()

    def isatty(self):
        try:
            return self._orig.isatty()
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Bot discovery
# ---------------------------------------------------------------------------

def _discover_bots(bots_dir, include=None):
    """Return {bot_id: bot_path} for every bot.py found under bots_dir."""
    bots = {}
    for subdir in sorted(Path(bots_dir).iterdir()):
        if not subdir.is_dir():
            continue
        bot_py = subdir / "bot.py"
        if not bot_py.exists():
            continue
        name = subdir.name
        if include and name not in include:
            continue
        bots[name] = str(bot_py)
    return bots


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sep(ch="=", width=72):
    return ch * width


def _print_standings(standings):
    print()
    print("{:>4}  {:<24} {:>12}  {:>8}  {:>12}".format(
        "Rank", "Bot", "Cum Delta", "Matches", "Best Match",
    ))
    print(_sep("-", 66))
    for i, b in enumerate(standings, 1):
        print("{:>4}. {:<24} {:>+12}  {:>8}  {:>+12}".format(
            i,
            b["bot_id"],
            b["cumulative_delta"],
            b["matches_played"],
            b["best_match_delta"],
        ))
    print()


def _print_match_result(result):
    print()
    print("  {:<24} {:>12}  {:>10}".format("Bot", "Final Stack", "Delta"))
    print("  " + _sep("-", 50))
    for bid in sorted(
        result["bot_ids"], key=lambda b: -result["chip_delta"][b]
    ):
        print("  {:<24} {:>12}  {:>+10}".format(
            bid,
            result["final_stacks"][bid],
            result["chip_delta"][bid],
        ))
    errs = {b: e for b, e in result["bot_errors"].items() if e}
    if errs:
        print()
        print("  Bot errors:")
        for bid, msgs in errs.items():
            for m in msgs:
                print(f"    [{bid}] {m}")
    print()


# ---------------------------------------------------------------------------
# Core tournament logic
# ---------------------------------------------------------------------------

def _run_one_match(round_num, ti, table, hands, verbose, seed):
    """Run a single table match; called from a worker thread."""
    match_id = f"swiss_r{round_num}_t{ti}_{uuid.uuid4().hex[:6]}"
    bot_paths = {b["bot_id"]: b["bot_path"] for b in table}
    match_seed = (seed * 999983 + round_num * 1009 + ti) if seed is not None else None
    result = run_match(
        match_id=match_id,
        bot_paths=bot_paths,
        n_hands=hands,
        verbose=verbose,
        seed=match_seed,
    )
    return match_id, ti, bot_paths, result


def _run_tournament(bots, rounds, hands, verbose, seed, workers):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(_sep())
    print("FULLHOUSE SWISS TOURNAMENT")
    print(f"Started : {now}")
    print(
        f"Rounds  : {rounds}   Hands/match : {hands}"
        f"   Table size : {TABLE_SIZE}   Workers : {workers}"
    )
    if seed is not None:
        print(f"Seed    : {seed}")
    print(_sep())

    print(f"\nDiscovered {len(bots)} bot(s):")
    for name, path in bots.items():
        print(f"  {name:<24}  {path}")

    standings = [
        {
            "bot_id": bid,
            "bot_path": path,
            "cumulative_delta": 0,
            "matches_played": 0,
            "best_match_delta": 0,
        }
        for bid, path in bots.items()
    ]
    all_results = []
    match_counter = 0

    for round_num in range(1, rounds + 1):
        print()
        print(_sep())
        print(f"ROUND {round_num} / {rounds}")
        print(_sep())

        tables = swiss_pairing(standings, table_size=TABLE_SIZE)
        print(f"\n{len(tables)} table(s) this round:")
        for ti, table in enumerate(tables, 1):
            names = ", ".join(b["bot_id"] for b in table)
            print(f"  Table {ti} ({len(table)} bots): {names}")

        # Tables within a round are independent — run them concurrently.
        # ThreadPoolExecutor is sufficient: bot CPU work happens in their
        # own subprocesses, so threads spend most time on subprocess I/O
        # (which releases the GIL).
        round_results = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(tables))) as pool:
            futures = {
                pool.submit(
                    _run_one_match, round_num, ti, table, hands, verbose, seed
                ): ti
                for ti, table in enumerate(tables, 1)
            }
            for fut in as_completed(futures):
                match_id, ti, bot_paths, result = fut.result()
                round_results[ti] = (match_id, bot_paths, result)

        for ti in sorted(round_results):
            match_counter += 1
            match_id, bot_paths, result = round_results[ti]
            print()
            print(f"--- Match {match_counter}  (Round {round_num}, Table {ti}) ---")
            print(f"  Bots: {', '.join(bot_paths)}")
            print(f"  ID  : {match_id}")
            print(
                f"\n  Completed in {result['duration_s']}s"
                f"  ({result['n_hands']} hands played)"
            )
            _print_match_result(result)

            for bid in result["bot_ids"]:
                all_results.append({
                    "bot_id": bid,
                    "bot_path": bot_paths[bid],
                    "chip_delta": result["chip_delta"][bid],
                })

        standings = compute_standings(all_results)
        print(f"--- Standings after Round {round_num} ---")
        _print_standings(standings)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(_sep())
    print("FINAL TOURNAMENT RESULTS")
    print(_sep())
    _print_standings(standings)

    n_finalists = min(64, len(standings))
    finalists = select_finalists(standings, n=n_finalists)
    print(f"Top {n_finalists} qualifier cut:")
    for i, b in enumerate(finalists, 1):
        print(
            f"  {i:>3}. {b['bot_id']:<24}  {b['cumulative_delta']:>+12}"
            f"  ({b['matches_played']} match(es))"
        )

    now_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print(f"Tournament finished : {now_end}")
    print(f"Total matches       : {match_counter}")
    print(_sep())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a Fullhouse Swiss tournament locally",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bots-dir", default=str(BOTS_DIR),
        help="Directory containing bot subdirectories",
    )
    parser.add_argument(
        "--bots", nargs="+", metavar="BOT",
        help="Restrict to these bot names (default: all)",
    )
    parser.add_argument(
        "--field", choices=list(FIELDS),
        help="Use a named opponent field preset (combines with --bots)",
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS,
        help="Number of Swiss rounds",
    )
    parser.add_argument(
        "--hands", type=int, default=DEFAULT_HANDS,
        help="Hands per match",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-action logs (very verbose)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Deterministic seed for reproducible matches",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Max tables to run in parallel within a round",
    )
    args = parser.parse_args()

    include = None
    if args.field or args.bots:
        include = set(args.bots or [])
        if args.field:
            include |= set(FIELDS[args.field])
    bots = _discover_bots(args.bots_dir, include=include)
    if len(bots) < 2:
        sys.exit(
            f"ERROR: Need at least 2 bots in {args.bots_dir}, found {len(bots)}"
        )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"tournament_{timestamp}.txt"

    print(f"Writing log to: {log_path}\n")

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    with open(log_path, "w", encoding="utf-8") as lf:
        sys.stdout = _TeeStream(orig_stdout, lf)
        sys.stderr = _TeeStream(orig_stderr, lf)
        try:
            _run_tournament(
                bots, args.rounds, args.hands, args.verbose, args.seed, args.workers
            )
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
