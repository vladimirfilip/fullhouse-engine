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


# ── Single-process chunk (authoritative, exact CFR) ───────────────────────────

def _run_chunk(n_iters: int, regret_sum: dict, strategy_sum: dict):
    """Run n_iters traversals directly on the shared tables (no reset)."""
    randrange = random.randrange
    for i in range(n_iters):
        run_iteration(i % N_PLAYERS, regret_sum, strategy_sum,
                      dealer_seat=randrange(N_PLAYERS))


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
    per-worker RAM and the pickle traffic.  Regrets are turned into a delta in
    place to avoid holding a third full table.
    """
    n_iters, seed, base_r = args
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    # Warm-started regret copy (evolves from the merged state); strategy starts
    # empty so its from-empty total IS the delta.
    local_r = {k: v.copy() for k, v in base_r.items()}
    local_s: dict[int, np.ndarray] = {}

    _run_chunk(n_iters, local_r, local_s)

    # Convert regrets to a delta in place: reuse local_r as the output so we
    # never hold base + working + delta simultaneously.
    for k in list(local_r.keys()):
        b = base_r.get(k)
        if b is None:
            continue            # new info set: full value is the delta
        d = local_r[k] - b
        if d.any():
            local_r[k] = d
        else:
            del local_r[k]
    return local_r, local_s


def _table_nbytes(table: dict[int, np.ndarray]) -> int:
    """
    Rough resident size of an info-set table in bytes (array data + Python
    object/dict/key overhead).  Used to size the per-round worker count.
    """
    if not table:
        return 0
    # 9×float64 data (72B) + ndarray object (~112B) + dict slot (~100B) + key (~28B).
    return len(table) * (72 + 112 + 100 + 28)


def _safe_workers(n_workers: int, regret_sum: dict, budget_gb: float) -> int:
    """
    Cap concurrent workers so peak RAM stays under budget.  Each worker holds
    roughly: warm-started regrets + accumulated strategy + the regret pickle
    in flight ≈ 3 × table_size.  As the tables grow over the run this returns a
    smaller number, which is what keeps the full run from being OOM-killed.
    """
    table = _table_nbytes(regret_sum)
    if table == 0:
        return n_workers
    budget = budget_gb * (1024 ** 3)
    fit = int(budget // (3 * table))
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
    iter_done = 0

    if resume and os.path.exists(config.CHECKPOINT_PATH):
        regret_sum, strategy_sum, iter_done = load_checkpoint()
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
    t0 = time.time()

    try:
        while iter_done < total_iters:
            n = min(round_iters, total_iters - iter_done)

            if pool is None:
                _run_chunk(n, regret_sum, strategy_sum)
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
                tasks  = [(c, random.randrange(2**31), regret_sum)
                          for c in counts]
                for dr, ds in pool.map(_worker_delta, tasks):
                    _merge(regret_sum,   dr)
                    _merge(strategy_sum, ds)
                del tasks

            iter_done += n
            elapsed = time.time() - t0
            rate    = iter_done / max(elapsed, 1e-6)
            if verbose:
                print(f"[preflop_cfr] iter={iter_done}/{total_iters}  "
                      f"info_sets={len(strategy_sum):,}  "
                      f"{rate:.0f} iter/s  elapsed={elapsed:.1f}s", flush=True)

            if iter_done >= next_ckpt or iter_done >= total_iters:
                save_checkpoint(regret_sum, strategy_sum, iter_done)
                next_ckpt += checkpoint_every
                if verbose:
                    print(f"[preflop_cfr] Checkpoint saved at iter {iter_done}.",
                          flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    n_exported = export_strategy(strategy_sum)
    if verbose:
        print(f"[preflop_cfr] Exported {n_exported:,} info sets -> "
              f"{config.EXPORT_PATH}", flush=True)

    _print_spot_check(strategy_sum, verbose)
    return regret_sum, strategy_sum


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
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)),
                        help="Parallel worker processes (default: CPU count)")
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
