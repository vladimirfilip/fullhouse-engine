#!/usr/bin/env python3
"""bench_feed.py — isolate where a training step's wall-time goes.

Pure-GPU compute is known (~16 ms/step). This times the two other components of
a real _train_regret step separately so we stop guessing:
  (A) C++ reservoir sampling  (sample_regret_into into a pinned buffer)
  (B) H2D transfer            (pinned host -> CUDA)
Run on the VM after `make build-cpp`:
    .venv/bin/python tools/bench_feed.py
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for sub in ("deep_cfr_cpp/build/Release", "deep_cfr_cpp/build"):
    if (ROOT / sub).exists():
        sys.path.insert(0, str(ROOT / sub))
import deep_cfr_gen  # noqa: E402
from deep_cfr.config import (BATCH_SIZE, INPUT_DIM, N_ACTIONS,  # noqa: E402
                             REGRET_BUF_CAP, STRATEGY_BUF_CAP)
from deep_cfr.networks import make_regret_net  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={dev}  BATCH_SIZE={BATCH_SIZE}  INPUT_DIM={INPUT_DIM}  "
      f"hw_threads={torch.get_num_threads()}")

# 1) Populate the regret buffer with a quick data-gen pass.
bufs = deep_cfr_gen.DeepCFRBuffers(REGRET_BUF_CAP, STRATEGY_BUF_CAP)
net = make_regret_net()
weights = [p.detach().cpu().numpy() for _, p in net.named_parameters()]
print("generating data to fill the reservoir…", flush=True)
t = time.time()
bufs.generate_and_add(4000, max(1, torch.get_num_threads()), 1, weights)
print(f"  gen 4000 games in {time.time()-t:.1f}s; regret_buf={bufs.regret_size():,}")
if bufs.regret_size() < BATCH_SIZE:
    print("buffer too small; raise games"); sys.exit(1)

# pinned host buffers (what _train_regret uses on CUDA)
pin = dev.type == "cuda"
s = torch.empty((BATCH_SIZE, INPUT_DIM), dtype=torch.float32, pin_memory=pin)
ta = torch.empty((BATCH_SIZE, N_ACTIONS), dtype=torch.float32, pin_memory=pin)
w = torch.empty((BATCH_SIZE,), dtype=torch.float32, pin_memory=pin)
sn, tn, wn = s.numpy(), ta.numpy(), w.numpy()

# (A) sampling only
N = 50
for _ in range(3):
    bufs.sample_regret_into(sn, tn, wn)          # warmup
t = time.time()
for _ in range(N):
    bufs.sample_regret_into(sn, tn, wn)
samp_ms = 1000 * (time.time() - t) / N
print(f"(A) sample_regret_into : {samp_ms:6.1f} ms/call")

# (B) H2D transfer only
if dev.type == "cuda":
    for _ in range(3):
        s.to(dev, non_blocking=True); torch.cuda.synchronize()
    t = time.time()
    for _ in range(N):
        s.to(dev, non_blocking=True)
        ta.to(dev, non_blocking=True)
        w.to(dev, non_blocking=True)
    torch.cuda.synchronize()
    h2d_ms = 1000 * (time.time() - t) / N
    print(f"(B) H2D transfer       : {h2d_ms:6.1f} ms/call  "
          f"({BATCH_SIZE*INPUT_DIM*4/1e6:.0f} MB states)")

print("\ncompare to ~16 ms pure-GPU compute/step. The biggest of these is the "
      "real bottleneck.")
