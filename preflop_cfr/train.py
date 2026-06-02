"""
Preflop CFR training loop.

Usage:
    python -m preflop_cfr.train              # full run
    python -m preflop_cfr.train --quick      # smoke test (5k iters)
    python -m preflop_cfr.train --iters 50000 --workers 8
    python -m preflop_cfr.train --resume     # resume from checkpoint

Convergence model
-----------------
CFR converges because each player's *current* strategy is driven by regrets
accumulated over ALL prior iterations, and the average strategy is the running
mean of those current strategies.  This only works if every iteration builds on
the accumulated tables.

Single-process mode does exactly that: one persistent ``regret_sum`` /
``strategy_sum`` pair, mutated in place.

Parallel mode keeps it correct via *warm-start + delta merge*:
  - At the start of each sync round every worker is given a snapshot of the
    current shared tables and runs its slice of iterations on a private copy.
  - Each worker returns only the *delta* it produced (changed/new info sets).
  - The parent sums the deltas back into the shared tables.
This is the standard synchronous parallel-CFR approximation: workers don't see
each other's updates *within* a round, but every round starts from the merged
state, so the strategy keeps improving across the whole run.  (The previous
implementation started every worker from empty tables and summed independent
from-scratch runs — because regret matching is scale-invariant that caps quality
at one round's length no matter how many total iterations are requested.)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time

import numpy as np

from preflop_cfr import config
from preflop_cfr.cfr import run_iteration
from preflop_cfr.equity import get_hu_table
from preflop_cfr.export import (
    export_strategy, save_checkpoint, load_checkpoint,
)

N_PLAYERS = config.N_PLAYERS


# ── Table merge ───────────────────────────────────────────────────────────────

def _merge(target: dict[int, np.ndarray], source: dict[int, np.ndarray]):
    """Add source values into target in place (creating missing keys)."""
    for k, v in source.items():
        cur = target.get(k)
        if cur is None:
            target[k] = v.copy()
        else:
            cur += v


def _merge_scalar(target: dict[int, float], source: dict[int, float]):
    """Add scalar source values (visit counts) into target in place."""
    for k, v in source.items():
        target[k] = target.get(k, 0.0) + v


def _floor_regrets(regret_sum: dict[int, np.ndarray]):
    """Re-apply the RM+ non-negativity floor to the merged regret table.

    Single-process CFR+ keeps regrets ≥0 inline (see cfr._traverse), but the
    parallel delta-merge sums per-worker deltas of independently-floored regrets,
    which can dip a cell below 0.  Clamp here so the shared table preserves the
    RM+ invariant before the next round warm-starts from it.
    """
    for v in regret_sum.values():
        np.maximum(v, 0.0, out=v)


# ── Single-process chunk (authoritative, exact CFR+) ──────────────────────────

def _run_chunk(n_iters: int, regret_sum: dict, strategy_sum: dict,
               visit_sum: dict, start_t: int, regret_base: dict | None = None):
    """Run n_iters CFR+ traversals directly on the shared tables (no reset).

    `start_t` is the global iteration index of the first traversal in this chunk;
    the linear-CFR average-strategy weight is (start_t + i + 1) so it increases
    monotonically across the whole run, not just within a chunk.

    `regret_base` (parallel workers only): the shared warm-start snapshot regrets
    are lazily copied from on first touch.  None in single-process.
    """
    randrange = random.randrange
    for i in range(n_iters):
        t = start_t + i
        run_iteration(t % N_PLAYERS, regret_sum, strategy_sum, visit_sum,
                      float(t + 1), dealer_seat=randrange(N_PLAYERS),
                      regret_base=regret_base)


# ── Parallel worker (warm-start + delta) ──────────────────────────────────────

def _worker_init():
    """Pool initializer: load the (disk-cached) HU table once per process."""
    get_hu_table()


def _worker_delta(args: tuple) -> tuple[dict, dict]:
    """
    Warm-start regrets from the shared snapshot, run a slice of iterations on a
    private copy, and return the deltas to merge back.

    Memory note: only ``regret_sum`` is shipped to the worker.  ``strategy_sum``
    is *write-only* during a traversal (see cfr._traverse — it is never read to
    make a decision), so a worker can accumulate it from empty and the result is
    exactly the delta to add back.  Not shipping the base strategy roughly halves
    per-worker RAM and the pickle traffic.

    The warm-start regrets are NOT copied up front.  ``local_r`` starts empty and
    cfr._traverse lazily copies a row out of ``base_r`` the first time the chunk
    touches it (regret_base=base_r).  A chunk that visits only part of the tree
    therefore materialises only that part, so peak per-worker RAM is "info sets
    touched this chunk" rather than "the whole table" — which is what lets more
    workers fit under the RAM budget (see _safe_workers).
    """
    n_iters, seed, base_r, base_t = args
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    # local_r evolves from the warm-start via lazy copy-on-touch out of base_r;
    # strategy and visit tables start empty so their from-empty totals ARE the
    # deltas.
    local_r: dict[int, np.ndarray] = {}
    local_s: dict[int, np.ndarray] = {}
    local_v: dict[int, float]      = {}

    _run_chunk(n_iters, local_r, local_s, local_v, base_t, regret_base=base_r)

    # local_r holds only the rows this chunk touched (each = base value + chunk
    # updates).  Convert in place to the delta to merge back.
    for k in list(local_r.keys()):
        b = base_r.get(k)
        if b is None:
            continue            # new info set: full value is the delta
        d = local_r[k] - b
        if d.any():
            local_r[k] = d
        else:
            del local_r[k]
    return local_r, local_s, local_v


def _table_nbytes(table: dict[int, np.ndarray]) -> int:
    """
    Rough resident size of an info-set table in bytes (array data + Python
    object/dict/key overhead).  Used to size the per-round worker count.
    """
    if not table:
        return 0
    # 9×float64 data (72B) + ndarray object (~112B) + dict slot (~100B) + key (~28B).
    return len(table) * (72 + 112 + 100 + 28)


# Per-worker peak RAM as a multiple of the live regret-table size.  With lazy
# copy-on-touch warm-start (see _worker_delta) a worker holds: the received base
# snapshot (1×) + only the rows its chunk touches + its strategy delta.  Rounds
# are split across workers, so each chunk touches a fraction of the tree and the
# old up-front full copy (which made it ~3×) is gone — ~2× is now a safe estimate.
# Raise this if you still see OOM (it trades worker count for headroom); lowering
# it past real usage risks an OOM-kill.  Overridable via PREFLOP_WORKER_TABLE_MULT.
_WORKER_TABLE_MULT = float(os.environ.get("PREFLOP_WORKER_TABLE_MULT", "2"))


def _safe_workers(n_workers: int, regret_sum: dict, budget_gb: float) -> int:
    """
    Cap concurrent workers so peak RAM stays under budget.  Each worker holds
    roughly _WORKER_TABLE_MULT × table_size (see that constant).  As the table
    grows over the run this returns a smaller number, which is what keeps the
    full run from being OOM-killed.
    """
    table = _table_nbytes(regret_sum)
    if table == 0:
        return n_workers
    budget = budget_gb * (1024 ** 3)
    fit = int(budget // (_WORKER_TABLE_MULT * table))
    return max(1, min(n_workers, fit))


def _split(total: int, parts: int) -> list[int]:
    """Split `total` iterations across `parts` workers as evenly as possible."""
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    total_iters:      int  = config.ITERATIONS,
    n_workers:        int  = 1,
    checkpoint_every: int  = config.CHECKPOINT_EVERY,
    sync_every:       int  = config.SYNC_EVERY,
    resume:           bool = False,
    verbose:          bool = True,
    mem_budget_gb:    float = config.MEM_BUDGET_GB,
):
    regret_sum:   dict[int, np.ndarray] = {}
    strategy_sum: dict[int, np.ndarray] = {}
    visit_sum:    dict[int, float]      = {}
    iter_done = 0

    if resume and os.path.exists(config.CHECKPOINT_PATH):
        regret_sum, strategy_sum, visit_sum, iter_done = load_checkpoint()
        if verbose:
            print(f"[preflop_cfr] Resumed from checkpoint at iter {iter_done}",
                  flush=True)

    if verbose:
        print(f"[preflop_cfr] Preparing HU equity table "
              f"({config.HU_EQUITY_BOARDS} boards) ...", flush=True)
    get_hu_table()   # build + cache to disk before workers spawn (they load it)
    if verbose:
        print("[preflop_cfr] HU table ready.", flush=True)

    pool = None
    if n_workers > 1:
        pool = mp.Pool(processes=n_workers, initializer=_worker_init)

    # Round size: a checkpoint interval single-process; a (smaller) sync interval
    # in parallel mode to bound worker divergence between merges.
    round_iters = checkpoint_every if pool is None else min(sync_every,
                                                            checkpoint_every)
    next_ckpt = (iter_done // checkpoint_every + 1) * checkpoint_every
    prev_premium: dict[str, np.ndarray] | None = None
    stable_ckpts = 0   # consecutive checkpoints with premium drift < eps
    t0 = time.time()

    try:
        while iter_done < total_iters:
            n = min(round_iters, total_iters - iter_done)

            if pool is None:
                _run_chunk(n, regret_sum, strategy_sum, visit_sum, iter_done)
            else:
                # Cap concurrent workers to the live table size so peak RAM
                # stays under budget — the tables grow over the run, so this
                # scales down automatically and prevents the OOM-kill.
                safe = _safe_workers(n_workers, regret_sum, mem_budget_gb)
                if verbose and safe < n_workers:
                    print(f"[preflop_cfr] table ~{_table_nbytes(regret_sum)/1e6:.0f}MB"
                          f"  -> capping workers {n_workers}->{safe} "
                          f"(budget {mem_budget_gb:g}GB)", flush=True)
                counts = [c for c in _split(n, safe) if c > 0]
                # Each worker warm-starts its linear-CFR weight from the global
                # iter_done so weights stay monotonic across the whole run.
                tasks  = [(c, random.randrange(2**31), regret_sum, iter_done)
                          for c in counts]
                for dr, ds, dv in pool.map(_worker_delta, tasks):
                    _merge(regret_sum,   dr)
                    _merge(strategy_sum, ds)
                    _merge_scalar(visit_sum, dv)
                # Restore the RM+ floor after summing independently-floored
                # per-worker regret deltas (see _floor_regrets).
                _floor_regrets(regret_sum)
                del tasks

            iter_done += n
            elapsed = time.time() - t0
            rate    = iter_done / max(elapsed, 1e-6)
            if verbose:
                print(f"[preflop_cfr] iter={iter_done}/{total_iters}  "
                      f"info_sets={len(strategy_sum):,}  "
                      f"{rate:.0f} iter/s  elapsed={elapsed:.1f}s", flush=True)

            if iter_done >= next_ckpt or iter_done >= total_iters:
                save_checkpoint(regret_sum, strategy_sum, visit_sum, iter_done)
                next_ckpt += checkpoint_every

                # Convergence gate (runs regardless of verbosity): track how much
                # the premium UTG-open mix has moved since the last checkpoint.
                cur_premium = _premium_snapshot(strategy_sum)
                drift = _max_premium_drift(prev_premium, cur_premium)

                if verbose:
                    print(f"[preflop_cfr] Checkpoint saved at iter {iter_done}.",
                          flush=True)
                    _visit_histogram(visit_sum)
                    _print_premium_drift(prev_premium, cur_premium)

                prev_premium = cur_premium

                if drift is not None and drift < config.CONVERGENCE_DRIFT_EPS:
                    stable_ckpts += 1
                    if verbose:
                        print(f"[preflop_cfr] premium drift {drift:.4f} < eps "
                              f"{config.CONVERGENCE_DRIFT_EPS} "
                              f"({stable_ckpts}/{config.CONVERGENCE_PATIENCE})",
                              flush=True)
                    if stable_ckpts >= config.CONVERGENCE_PATIENCE:
                        print(f"[preflop_cfr] Converged: premium drift below "
                              f"{config.CONVERGENCE_DRIFT_EPS} for "
                              f"{stable_ckpts} consecutive checkpoints at iter "
                              f"{iter_done}. Stopping early.", flush=True)
                        break
                else:
                    if drift is not None and verbose:
                        print(f"[preflop_cfr] premium drift {drift:.4f} "
                              f"(>= eps {config.CONVERGENCE_DRIFT_EPS}); "
                              f"streak reset", flush=True)
                    stable_ckpts = 0
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    n_exported = export_strategy(strategy_sum, visit_sum)
    if verbose:
        print(f"[preflop_cfr] Exported {n_exported:,} info sets -> "
              f"{config.EXPORT_PATH}", flush=True)

    _print_spot_check(strategy_sum, verbose)
    return regret_sum, strategy_sum


# ── Convergence diagnostics ─────────────────────────────────────────────────

# Per-info-set visits come from the dedicated visit_sum table (one +1 per
# opponent-node visit), which is exactly the quantity export.py thresholds with
# PRUNE_MIN_VISITS — keeping the diagnostic and the prune metric consistent.
# (The average-strategy table is now iteration-weighted, so it can no longer
# double as the visit proxy.)
_VISIT_BINS = [0, 1, 10, 100, 1_000, 10_000]

# Premium UTG-open hands to watch for strategy drift (the ones the old
# unconverged table mangled — see preflop-table-disabled memory).
_PREMIUM_HANDS = ("AA", "KK", "QQ", "JJ", "AKs", "AKo")


def _visit_histogram(visit_sum: dict[int, float]) -> None:
    """Print a visits/info-set histogram + summary against TARGET_VISITS_PER_SET.

    Reads the dedicated visit-count table (true per-info-set visit counts).  The
    average-strategy table can no longer serve as the visit proxy now that it is
    iteration-weighted (linear CFR) — its row sums scale with t, not visits.
    """
    if not visit_sum:
        print("[preflop_cfr] no info sets yet.", flush=True)
        return

    counts = np.fromiter(visit_sum.values(),
                         dtype=np.float64, count=len(visit_sum))
    target = config.TARGET_VISITS_PER_SET
    edges  = _VISIT_BINS + [float("inf")]
    print(f"[preflop_cfr] visits/info-set over {len(counts):,} sets "
          f"(mean={counts.mean():.0f} median={np.median(counts):.0f}):", flush=True)
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(((counts >= lo) & (counts < hi)).sum())
        hi_label = "inf" if hi == float("inf") else f"{int(hi)}"
        bar = "#" * min(40, int(40 * n / len(counts)))
        print(f"    [{int(lo):>6}, {hi_label:>6})  {n:>9,}  {bar}", flush=True)
    met = int((counts >= target).sum())
    print(f"[preflop_cfr] {met:,}/{len(counts):,} "
          f"({100*met/len(counts):.1f}%) sets >= target {target:,} visits",
          flush=True)


def _premium_snapshot(strategy_sum: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Average-strategy vectors for the premium UTG-open hands (history empty)."""
    from preflop_cfr.cards import BUCKET_INFO, RANKS
    from preflop_cfr.abstraction import infoset_key

    name_to_bucket: dict[str, int] = {}
    for bucket, (hi, lo, suited) in enumerate(BUCKET_INFO):
        suit_char = "s" if suited else "o" if hi != lo else ""
        name = f"{RANKS[hi]}{RANKS[lo]}{suit_char}"
        name_to_bucket[name] = bucket

    utg_pos = 3  # (UTG_seat - dealer) % 6 in 6-max
    snap: dict[str, np.ndarray] = {}
    for name in _PREMIUM_HANDS:
        bucket = name_to_bucket.get(name)
        if bucket is None:
            continue
        ssum = strategy_sum.get(infoset_key(utg_pos, (), bucket))
        if ssum is None:
            continue
        total = ssum.sum()
        if total > 0:
            snap[name] = ssum / total
    return snap


def _max_premium_drift(prev: dict[str, np.ndarray] | None,
                       cur: dict[str, np.ndarray]) -> float | None:
    """Largest |Δ probability| across all premium hands vs the last checkpoint.

    Returns None (don't gate) until both snapshots contain every premium hand —
    early on, some premium info sets may not have been visited yet, and we must
    not declare convergence before the strategy they track even exists.
    """
    if not prev or not cur:
        return None
    worst = 0.0
    seen = 0
    for name in _PREMIUM_HANDS:
        a, b = cur.get(name), prev.get(name)
        if a is None or b is None:
            continue
        d = float(np.abs(a - b).max())
        if d > worst:
            worst = d
        seen += 1
    if seen < len(_PREMIUM_HANDS):
        return None
    return worst


def _print_premium_drift(prev: dict[str, np.ndarray] | None,
                         cur: dict[str, np.ndarray]) -> None:
    """Print premium-hand fold/call/raise/jam mix and max drift vs last checkpoint."""
    if not cur:
        return
    print("[preflop_cfr] premium UTG-open mix "
          "(fold/call/raise/jam) | drift:", flush=True)
    for name in _PREMIUM_HANDS:
        v = cur.get(name)
        if v is None:
            continue
        fold = v[config.FOLD]
        call = v[config.CHECK_CALL]
        jam  = v[config.ALL_IN]
        raise_ = max(0.0, 1.0 - fold - call - jam)
        if prev and name in prev:
            drift = float(np.abs(v - prev[name]).max())
            drift_s = f"max|d|={drift:.3f}"
        else:
            drift_s = "  -  "
        print(f"    {name:4s}  {fold:.2f}/{call:.2f}/{raise_:.2f}/{jam:.2f}"
              f"   {drift_s}", flush=True)


def _print_spot_check(strategy_sum: dict[int, np.ndarray], verbose: bool):
    """Print UTG open-raise range as a quick convergence sanity check."""
    if not verbose:
        return

    from preflop_cfr.cards import BUCKET_INFO, RANKS
    from preflop_cfr.abstraction import infoset_key

    # UTG in 6-max: hero_position = (UTG_seat - dealer) % 6 = 3, history empty.
    utg_pos = 3
    open_pct = {}
    for bucket, (hi, lo, suited) in enumerate(BUCKET_INFO):
        key = infoset_key(utg_pos, (), bucket)
        ssum = strategy_sum.get(key)
        if ssum is None:
            continue
        total = ssum.sum()
        if total == 0:
            continue
        avg = ssum / total
        raise_prob = 1.0 - avg[config.FOLD] - avg[config.CHECK_CALL]
        suit_char  = "s" if suited else "o" if hi != lo else ""
        hand_name  = (f"{RANKS[hi]}{RANKS[lo]}{suit_char}" if hi != lo
                      else f"{RANKS[hi]}{RANKS[lo]}")
        open_pct[hand_name] = raise_prob

    if open_pct:
        top = sorted(open_pct.items(), key=lambda x: -x[1])[:20]
        print("\n[preflop_cfr] UTG open-raise rates (top 20):")
        for hand, pct in top:
            bar = "#" * int(pct * 20)
            print(f"  {hand:5s}  {pct:.2f}  {bar}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train preflop 6-max tabular CFR")
    parser.add_argument("--quick",   action="store_true",
                        help=f"Smoke test ({config.QUICK_ITERATIONS} iters)")
    parser.add_argument("--iters",   type=int, default=None,
                        help="Override total iterations")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) // 2),
                        help="Parallel worker processes. Default ~ physical core "
                             "count (logical/2): measured throughput scales well "
                             "to ~physical cores, then flattens (the equity work "
                             "is compute-bound, so hyperthreads add little and "
                             "each extra process duplicates the equity cache). "
                             "Override for a different box.")
    parser.add_argument("--sync",    type=int, default=config.SYNC_EVERY,
                        help="Iterations between cross-worker merges (parallel)")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--mem-budget-gb", type=float, default=config.MEM_BUDGET_GB,
                        help="RAM budget for table copies; caps concurrent "
                             "workers as tables grow (default: %(default)g, "
                             "or $PREFLOP_MEM_BUDGET_GB)")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    n_iters = config.QUICK_ITERATIONS if args.quick else (args.iters or config.ITERATIONS)

    train(
        total_iters      = n_iters,
        n_workers        = args.workers,
        checkpoint_every = min(config.CHECKPOINT_EVERY, n_iters),
        sync_every       = args.sync,
        resume           = args.resume,
        verbose          = not args.quiet,
        mem_budget_gb    = args.mem_budget_gb,
    )
