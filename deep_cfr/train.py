"""
Deep CFR training orchestrator.

Usage (from repo root):
    python -m deep_cfr.train [--iters N] [--games N] [--quick] [--workers N]

Outputs:
    bots/vlad/data/gto_strategy.npz   -- numpy weights for the production bot

Data generation is parallelised across N_WORKERS C++ std::threads (one per CPU
core) inside the deep_cfr_gen extension.  Each thread runs independent MCCFR
traversals and writes directly into the shared C++ reservoir buffers via a
thread-safe add_batch().  The GIL is released for the entire generate_and_add()
call, so Python training and C++ data-gen can overlap via ThreadPoolExecutor.

Training (regret / strategy net SGD steps) remains on the main process CPU.
"""

from __future__ import annotations
import argparse
import glob
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    K_ITERATIONS, GAMES_PER_ITER,
    BATCH_SIZE, LEARNING_RATE,
    REGRET_TRAIN_STEPS, STRATEGY_TRAIN_STEPS,
    REGRET_BUF_CAP, STRATEGY_BUF_CAP,
    MODEL_FILENAME, N_WORKERS,
    INPUT_DIM, N_ACTIONS,
)

# Mid-run strategy snapshots exist only so an interrupted run still leaves a
# usable model; they don't feed back into training. At 300 iters the old
# every-5-iters × 2 000-step cadence spent a large slice of wall-clock retraining
# a throwaway net. Snapshot less often and for fewer steps (the bigger batch also
# covers the buffer faster) — the budget goes to data-gen and the final net.
STRATEGY_CKPT_EVERY  = 25    # train+export strategy snapshot every N iterations
STRATEGY_CKPT_STEPS  = 500   # quick mid-run training steps (vs the final pass)
from .networks import make_regret_net, make_strategy_net
from .export import export_net, load_net

# Checkpoints are written here by the training loop (regret_net_iter_{t}.npz).
CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))


def _latest_checkpoint(ckpt_dir: str = CKPT_DIR):
    """Return (path, iteration) of the most recently modified regret-net
    checkpoint, or (None, 0) if none exist."""
    paths = glob.glob(os.path.join(ckpt_dir, "regret_net_iter_*.npz"))
    if not paths:
        return None, 0
    latest = max(paths, key=os.path.getmtime)
    m = re.search(r"regret_net_iter_(\d+)\.npz$", os.path.basename(latest))
    iteration = int(m.group(1)) if m else 0
    return latest, iteration

# ── Load C++ data-generation extension ────────────────────────────────────────
# MSVC puts the .pyd in build/Release/; GCC puts the .so directly in build/.
_HERE = os.path.dirname(__file__)
_CPP_ROOT = os.path.abspath(os.path.join(_HERE, '..', 'deep_cfr_cpp', 'build'))
sys.path.insert(0, os.path.join(_CPP_ROOT, 'Release'))
sys.path.insert(0, _CPP_ROOT)
try:
    import deep_cfr_gen
    _USE_CPP = True
    print(f"[deep_cfr] C++ data-gen loaded (INPUT_DIM={deep_cfr_gen.INPUT_DIM}, "
          f"N_ACTIONS={deep_cfr_gen.N_ACTIONS}, HIDDEN_DIM={deep_cfr_gen.HIDDEN_DIM})")
except ImportError:
    _USE_CPP = False
    print("[deep_cfr] WARNING: deep_cfr_gen not found — falling back to Python MCCFR")


# ── Parallel data generation ───────────────────────────────────────────────

def _parallel_generate(
    n_games: int,
    regret_net: nn.Module,
    cpp_buffers,
    iteration_t: int,
    n_workers: int,
) -> None:
    """
    Distribute `n_games` MCCFR traversals across `n_workers` C++ threads and
    add samples directly to the C++ reservoir buffers (no intermediate allocation).
    """
    if not _USE_CPP:
        raise RuntimeError(
            "deep_cfr_gen C++ extension not found. "
            "Build it: cmake --build deep_cfr_cpp/build --target deep_cfr_gen"
        )

    weights = [p.detach().cpu().numpy() for _, p in regret_net.named_parameters()]
    cpp_buffers.generate_and_add(n_games, n_workers, iteration_t, weights)


# ── Training helpers ───────────────────────────────────────────────────────

def _make_io_buffers(device: torch.device):
    """Two reusable (state, target, weight) tensor triples for double-buffered
    prefetch, plus their numpy views the C++ sampler fills in place.

    Pinned host memory on CUDA so `.to(device, non_blocking=True)` is a genuinely
    async H2D copy (non_blocking is a no-op from pageable memory). Plain tensors
    on CPU, where pin_memory requires CUDA and isn't needed.
    """
    pin = device.type == "cuda"
    s = [torch.empty((BATCH_SIZE, INPUT_DIM), dtype=torch.float32, pin_memory=pin)
         for _ in range(2)]
    t = [torch.empty((BATCH_SIZE, N_ACTIONS), dtype=torch.float32, pin_memory=pin)
         for _ in range(2)]
    w = [torch.empty((BATCH_SIZE,), dtype=torch.float32, pin_memory=pin)
         for _ in range(2)]
    return (s, t, w,
            [b.numpy() for b in s], [b.numpy() for b in t], [b.numpy() for b in w])


def _make_adam(net: nn.Module, device: torch.device) -> optim.Adam:
    # fused=True collapses the many tiny per-step optimizer kernel launches that
    # dominate wall-time on this small MLP; only supported on CUDA.
    return optim.Adam(net.parameters(), lr=LEARNING_RATE,
                      fused=(device.type == "cuda"))


def _train_regret(net: nn.Module, cpp_buffers, n_steps: int,
                  device: torch.device) -> float:
    """Iteration-weighted MSE on (state, regret_vector, weight) triples.

    Double-buffered prefetch: a background thread fills buffer B via GIL-free
    sample_regret_into() while PyTorch trains on buffer A, then they swap.

    Perf: the loss is kept on-device and only synced to host at the print cadence
    (a per-step loss.item() forces a CUDA sync every step, which serialises the
    GPU and defeats the prefetch). Buffers are pinned (see _make_io_buffers) and
    the optimizer is fused (see _make_adam).
    """
    net.train()
    opt     = _make_adam(net, device)
    loss_fn = nn.MSELoss(reduction="none")
    s_tens, t_tens, w_tens, s_bufs, t_bufs, w_bufs = _make_io_buffers(device)
    last = None

    cur = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(cpp_buffers.sample_regret_into,
                          s_bufs[cur], t_bufs[cur], w_bufs[cur])
        for step in range(n_steps):
            k   = fut.result()          # wait for current buffer to be ready
            nxt = 1 - cur
            if step < n_steps - 1:     # pre-fetch into the other buffer
                fut = pool.submit(cpp_buffers.sample_regret_into,
                                  s_bufs[nxt], t_bufs[nxt], w_bufs[nxt])

            states  = s_tens[cur][:k].to(device, non_blocking=True)
            targets = t_tens[cur][:k].to(device, non_blocking=True)
            weights = w_tens[cur][:k].to(device, non_blocking=True)

            preds    = net(states)
            per_samp = loss_fn(preds, targets).mean(dim=1)   # [B]
            # Linear CFR weighting (true weighted mean); do NOT normalise to
            # mean=1 within the batch (that undoes the time-averaging).
            loss     = (per_samp * weights).sum() / (weights.sum() + 1e-8)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = loss.detach()        # keep on-device; no host sync here
            if step % 200 == 0:
                print(f"    regret step {step:4d}  loss={last.item():.4f}")
            cur = nxt

    return last.item() if last is not None else 0.0


def _train_strategy(net: nn.Module, cpp_buffers, n_steps: int,
                    device: torch.device) -> float:
    """Iteration-weighted MSE on (state, strategy_vector, weight) triples.

    Same double-buffered prefetch pattern as _train_regret.
    Cosine LR decay (1e-3 → 1e-5) over the full step budget: the strategy net
    trains once and ships, so it benefits from a finer final pass that a
    constant LR would oscillate through.
    """
    net.train()
    opt      = _make_adam(net, device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=1e-5
    )
    loss_fn = nn.MSELoss(reduction="none")
    s_tens, t_tens, w_tens, s_bufs, t_bufs, w_bufs = _make_io_buffers(device)
    last = None

    cur = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(cpp_buffers.sample_strategy_into,
                          s_bufs[cur], t_bufs[cur], w_bufs[cur])
        for step in range(n_steps):
            k   = fut.result()
            nxt = 1 - cur
            if step < n_steps - 1:
                fut = pool.submit(cpp_buffers.sample_strategy_into,
                                  s_bufs[nxt], t_bufs[nxt], w_bufs[nxt])

            states  = s_tens[cur][:k].to(device, non_blocking=True)
            targets = t_tens[cur][:k].to(device, non_blocking=True)
            weights = w_tens[cur][:k].to(device, non_blocking=True)

            preds    = net(states)
            per_samp = loss_fn(preds, targets).mean(dim=1)   # [B]
            # Linear CFR weighting (true weighted mean); do NOT normalise to
            # mean=1 within the batch — that undoes the time-averaging Deep CFR
            # depends on.
            loss     = (per_samp * weights).sum() / (weights.sum() + 1e-8)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
            last = loss.detach()        # keep on-device; no host sync here
            if step % 200 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"    strategy step {step:4d}  loss={last.item():.4f}  lr={lr:.2e}")
            cur = nxt

    return last.item() if last is not None else 0.0


# ── Main training loop ─────────────────────────────────────────────────────

def train(
    k_iterations: int = K_ITERATIONS,
    games_per_iter: int = GAMES_PER_ITER,
    n_workers: int = N_WORKERS,
    resume_from: str | None = None,
    max_hours: float = 6.5,
) -> None:
    # Device order: CUDA (NVIDIA) > DirectML (AMD/Intel on Windows) > CPU.
    # DirectML is opt-in via `pip install torch-directml`; absence is silent.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml  # type: ignore
            device = torch_directml.device()
        except ImportError:
            device = torch.device("cpu")
    print(f"Workers: {n_workers}  |  Iterations: {k_iterations}  |  Games/iter: {games_per_iter}  |  Device: {device}")

    if not _USE_CPP:
        raise RuntimeError(
            "deep_cfr_gen C++ extension not found. "
            "Build it: cmake --build deep_cfr_cpp/build --target deep_cfr_gen"
        )

    cpp_buffers  = deep_cfr_gen.DeepCFRBuffers(REGRET_BUF_CAP, STRATEGY_BUF_CAP)
    strategy_net = make_strategy_net().to(device)
    regret_net   = make_regret_net().to(device)

    # ── Resume: warm-start the regret net from a checkpoint ────────────────────
    # Only the regret net is checkpointed (not the reservoir buffers), so we
    # continue data-gen with the loaded policy and rebuild buffers from scratch.
    # Iteration numbering continues past the checkpoint so Linear-CFR weighting
    # stays monotonic and new checkpoints don't clobber the old ones.
    start_iter = 0
    if resume_from is not None:
        load_net(regret_net, resume_from)
        m = re.search(r"regret_net_iter_(\d+)\.npz$", os.path.basename(resume_from))
        start_iter = int(m.group(1)) if m else 0
        print(f"Resuming from {os.path.abspath(resume_from)} "
              f"(iteration {start_iter}); training through {k_iterations}.")
        if start_iter >= k_iterations:
            print(f"Checkpoint iteration {start_iter} >= target {k_iterations}; "
                  f"nothing to do. Raise --iters to continue training.")

    # ── Budget state (wall-clock completion guarantee) ─────────────────────
    run_start  = time.perf_counter()
    budget_s   = max_hours * 3600.0
    iter_ewma: float | None = None        # smoothed iteration wall-time
    stop_reason = f"reached target {k_iterations} iterations"

    for t in range(start_iter + 1, k_iterations + 1):
        # ── Wall-clock cap: stop BEFORE an iteration that would overrun, so the
        # final strategy fit always runs and we ship a converged-as-possible net
        # (the prior run had no cap and a hard kill shipped an undertrained net).
        elapsed_total = time.perf_counter() - run_start
        if iter_ewma is not None and elapsed_total + iter_ewma > budget_s:
            stop_reason = (f"wall-clock cap ({max_hours}h): "
                           f"{elapsed_total / 3600:.2f}h elapsed, "
                           f"next iter ~{iter_ewma:.0f}s would overrun")
            print(f"\n[budget] {stop_reason} — stopping data-gen, going to final fit.")
            break

        t0 = time.perf_counter()
        print(f"\n=== Iteration {t}/{k_iterations} "
              f"({elapsed_total / 3600:.2f}h / {max_hours}h) ===")

        # ── Parallel data generation (uses net trained in previous iteration) ─
        print(f"  Generating {games_per_iter} games across {n_workers} workers…",
              flush=True)
        t_gen = time.perf_counter()
        _parallel_generate(games_per_iter, regret_net, cpp_buffers, t, n_workers)
        print(f"  gen done in {time.perf_counter() - t_gen:.1f}s  "
              f"regret_buf={cpp_buffers.regret_size():,}  "
              f"strategy_buf={cpp_buffers.strategy_size():,}", flush=True)

        # ── Train fresh regret net on full accumulated buffer ─────────────
        if cpp_buffers.regret_ready(BATCH_SIZE):
            print(f"  Training regret net ({REGRET_TRAIN_STEPS} steps)…")
            # Explicitly release the old net before allocating the new one so
            # Python's reference-counter frees its tensors + Adam moments
            # immediately rather than waiting for GC on the next cycle.
            del regret_net
            regret_net = make_regret_net().to(device)
            _train_regret(regret_net, cpp_buffers, REGRET_TRAIN_STEPS, device)
        else:
            print(f"  Regret buffer too small ({cpp_buffers.regret_size()}), skipping train.")

        elapsed = time.perf_counter() - t0
        iter_ewma = elapsed if iter_ewma is None else 0.6 * iter_ewma + 0.4 * elapsed
        print(f"  Iteration done in {elapsed:.1f}s")

        # ── Checkpoint: save outside data/ (keep last 3 to cap disk use) ────────
        ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"regret_net_iter_{t}.npz")
        export_net(regret_net, ckpt_path)
        print(f"  Checkpoint -> {os.path.abspath(ckpt_path)}")
        # Delete checkpoints older than the last 3 to avoid unbounded disk use.
        old = os.path.join(ckpt_dir, f"regret_net_iter_{t - 3}.npz")
        if os.path.exists(old):
            os.remove(old)

        # ── Periodic strategy net snapshot ────────────────────────────────────
        # Train a fresh strategy net on the accumulated buffer and export it to
        # gto_strategy.npz every STRATEGY_CKPT_EVERY iterations so the bot has
        # a usable model even if the run is interrupted before completion.
        if t % STRATEGY_CKPT_EVERY == 0 and cpp_buffers.strategy_ready(BATCH_SIZE):
            print(f"  Strategy snapshot (iter {t}, {STRATEGY_CKPT_STEPS} steps)…")
            snap_net = make_strategy_net().to(device)
            _train_strategy(snap_net, cpp_buffers, STRATEGY_CKPT_STEPS, device)
            out_dir = os.path.join(os.path.dirname(__file__), "..", "bots", "vlad", "data")
            os.makedirs(out_dir, exist_ok=True)
            snap_path = os.path.join(out_dir, MODEL_FILENAME + ".npz")
            export_net(snap_net, snap_path)
            print(f"  Strategy snapshot -> {os.path.abspath(snap_path)}")
            del snap_net

    # ── Final strategy net training ────────────────────────────────────────
    print(f"\n[stop] {stop_reason}")
    if cpp_buffers.strategy_ready(BATCH_SIZE):
        print(f"\nTraining strategy net ({STRATEGY_TRAIN_STEPS} steps)…")
        _train_strategy(strategy_net, cpp_buffers, STRATEGY_TRAIN_STEPS, device)
    else:
        print("Strategy buffer too small; saving untrained strategy net.")

    # ── Export ────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "bots", "vlad", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, MODEL_FILENAME + ".npz")
    export_net(strategy_net, out_path)
    print(f"\nSaved -> {os.path.abspath(out_path)}")


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Deep CFR 6-max NLHE trainer")
    ap.add_argument("--iters",   type=int, default=K_ITERATIONS,
                    help="Number of CFR outer iterations")
    ap.add_argument("--games",   type=int, default=GAMES_PER_ITER,
                    help="Games (traversals) per iteration")
    ap.add_argument("--workers", type=int, default=N_WORKERS,
                    help="Parallel worker processes for data generation")
    ap.add_argument("--quick",   action="store_true",
                    help="Smoke-test: 5 iterations x 200 games")
    ap.add_argument("--resume",  action="store_true",
                    help="Continue from the last-modified regret-net checkpoint "
                         "in ./checkpoints/")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Continue from a specific regret-net checkpoint .npz")
    ap.add_argument("--max-hours", type=float, default=6.5,
                    help="Wall-clock cap: stop data-gen before overrun and run the "
                         "final strategy fit (reserve ~1h of the 7h for it)")
    args = ap.parse_args()

    resume_from = args.resume_from
    if args.resume and resume_from is None:
        resume_from, it = _latest_checkpoint()
        if resume_from is None:
            print(f"--resume: no checkpoints found in {CKPT_DIR}; starting fresh.")
        else:
            print(f"--resume: latest checkpoint is {resume_from} (iteration {it}).")

    common = dict(n_workers=args.workers, resume_from=resume_from,
                  max_hours=args.max_hours)
    if args.quick:
        train(k_iterations=5, games_per_iter=200, **common)
    else:
        train(k_iterations=args.iters, games_per_iter=args.games, **common)
