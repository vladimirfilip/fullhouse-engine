"""
Deep CFR training orchestrator.

Usage (from repo root):
    python -m bots.vlad.deep_cfr.train [--iters N] [--games N] [--quick] [--workers N]

Outputs:
    bots/vlad/data/gto_strategy.npz   -- numpy weights for the production bot

Data generation is parallelised across N_WORKERS processes (one per CPU core).
Each worker runs a batch of independent MCCFR traversals and returns raw numpy
arrays; the main process aggregates them into the reservoir buffers.  The GIL
does not limit throughput because each worker is a separate OS process.

Training (regret / strategy net SGD steps) remains on the main process CPU.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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
from .networks import make_regret_net, make_strategy_net
from .export import export_net

# ── Load C++ data-generation extension ────────────────────────────────────────
_HERE = os.path.dirname(__file__)
_CPP_BUILD = os.path.join(_HERE, '..', 'deep_cfr_cpp', 'build', 'Release')
sys.path.insert(0, os.path.abspath(_CPP_BUILD))
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
            "Build it: cmake --build bots/vlad/deep_cfr_cpp/build --target deep_cfr_gen"
        )

    weights = [p.detach().cpu().numpy() for _, p in regret_net.named_parameters()]
    cpp_buffers.generate_and_add(n_games, n_workers, iteration_t, weights)


# ── Training helpers ───────────────────────────────────────────────────────

def _train_regret(net: nn.Module, cpp_buffers, n_steps: int,
                  device: torch.device) -> float:
    """MSE training on (state, regret_vector) pairs.

    Uses double-buffered prefetch: a background thread fills numpy buffer B via
    GIL-free sample_regret_into() while PyTorch trains on buffer A, then they
    swap. This hides the ~10–20 ms C++ sampling cost behind forward/backward.
    """
    net.train()
    opt     = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    last_loss = 0.0

    # Two buffer pairs — training uses one while C++ fills the other.
    s_bufs = [np.empty((BATCH_SIZE, INPUT_DIM), dtype=np.float32) for _ in range(2)]
    t_bufs = [np.empty((BATCH_SIZE, N_ACTIONS), dtype=np.float32) for _ in range(2)]
    # torch.from_numpy shares memory (zero-copy); kept alive for the run.
    s_tens = [torch.from_numpy(b) for b in s_bufs]
    t_tens = [torch.from_numpy(b) for b in t_bufs]

    cur = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(cpp_buffers.sample_regret_into, s_bufs[cur], t_bufs[cur])
        for step in range(n_steps):
            k   = fut.result()          # wait for current buffer to be ready
            nxt = 1 - cur
            if step < n_steps - 1:     # pre-fetch into the other buffer
                fut = pool.submit(cpp_buffers.sample_regret_into, s_bufs[nxt], t_bufs[nxt])

            states  = s_tens[cur][:k].to(device, non_blocking=True)
            targets = t_tens[cur][:k].to(device, non_blocking=True)
            preds   = net(states)
            loss    = loss_fn(preds, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
            if step % 200 == 0:
                print(f"    regret step {step:4d}  loss={loss.item():.4f}")
            cur = nxt

    return last_loss


def _train_strategy(net: nn.Module, cpp_buffers, n_steps: int,
                    device: torch.device) -> float:
    """Iteration-weighted MSE on (state, strategy_vector, weight) triples.

    Same double-buffered prefetch pattern as _train_regret.
    """
    net.train()
    opt     = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss(reduction="none")
    last_loss = 0.0

    s_bufs = [np.empty((BATCH_SIZE, INPUT_DIM), dtype=np.float32) for _ in range(2)]
    t_bufs = [np.empty((BATCH_SIZE, N_ACTIONS), dtype=np.float32) for _ in range(2)]
    w_bufs = [np.empty((BATCH_SIZE,),           dtype=np.float32) for _ in range(2)]
    s_tens = [torch.from_numpy(b) for b in s_bufs]
    t_tens = [torch.from_numpy(b) for b in t_bufs]
    w_tens = [torch.from_numpy(b) for b in w_bufs]

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
            per_elem = loss_fn(preds, targets)          # [B, N_ACTIONS]
            per_samp = per_elem.mean(dim=1)             # [B]
            # Linear CFR weighting: weight = iteration_t, applied as a true
            # weighted mean. Do NOT normalise to mean=1 within the batch — that
            # collapses late-iteration samples to the same effective influence
            # as early ones and undoes the time-averaging Deep CFR depends on.
            loss     = (per_samp * weights).sum() / (weights.sum() + 1e-8)

            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
            if step % 200 == 0:
                print(f"    strategy step {step:4d}  loss={loss.item():.4f}")
            cur = nxt

    return last_loss


# ── Main training loop ─────────────────────────────────────────────────────

def train(
    k_iterations: int = K_ITERATIONS,
    games_per_iter: int = GAMES_PER_ITER,
    n_workers: int = N_WORKERS,
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
            "Build it: cmake --build bots/vlad/deep_cfr_cpp/build --target deep_cfr_gen"
        )

    cpp_buffers  = deep_cfr_gen.DeepCFRBuffers(REGRET_BUF_CAP, STRATEGY_BUF_CAP)
    strategy_net = make_strategy_net().to(device)
    regret_net   = make_regret_net().to(device)

    for t in range(1, k_iterations + 1):
        t0 = time.perf_counter()
        print(f"\n=== Iteration {t}/{k_iterations} ===")

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
            regret_net = make_regret_net().to(device)
            _train_regret(regret_net, cpp_buffers, REGRET_TRAIN_STEPS, device)
        else:
            print(f"  Regret buffer too small ({cpp_buffers.regret_size()}), skipping train.")

        elapsed = time.perf_counter() - t0
        print(f"  Iteration done in {elapsed:.1f}s")

        # ── Checkpoint: save the freshly-trained regret net ───────────────────
        out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, f"regret_net_iter_{t}.npz")
        export_net(regret_net, ckpt_path)
        print(f"  Checkpoint -> {os.path.abspath(ckpt_path)}")

    # ── Final strategy net training ────────────────────────────────────────
    if cpp_buffers.strategy_ready(BATCH_SIZE):
        print(f"\nTraining strategy net ({STRATEGY_TRAIN_STEPS} steps)…")
        _train_strategy(strategy_net, cpp_buffers, STRATEGY_TRAIN_STEPS, device)
    else:
        print("Strategy buffer too small; saving untrained strategy net.")

    # ── Export ────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
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
    args = ap.parse_args()

    if args.quick:
        train(k_iterations=5, games_per_iter=200, n_workers=args.workers)
    else:
        train(k_iterations=args.iters, games_per_iter=args.games, n_workers=args.workers)
