"""Parity test: CUDA cuBLAS forward pass vs numpy reference.

Skipped automatically when deep_cfr_cuda is not built.

To build and run:
    make build-cpp-cuda
    python -m pytest tests/test_cuda_forward.py -v
"""

import os
import sys
import numpy as np
import pytest

# Add both the flat build dir and the MSVC Release subdir to sys.path
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BUILD = os.path.join(_REPO, "deep_cfr_cpp", "build_cuda")
sys.path.insert(0, os.path.join(_BUILD, "Release"))
sys.path.insert(0, _BUILD)

deep_cfr_cuda = pytest.importorskip(
    "deep_cfr_cuda",
    reason="deep_cfr_cuda not built — run: make build-cpp-cuda",
)

INPUT_DIM   = deep_cfr_cuda.INPUT_DIM    # 308
N_ACTIONS   = deep_cfr_cuda.N_ACTIONS    # 9
HIDDEN_DIM  = deep_cfr_cuda.HIDDEN_DIM   # 512
N_LAYERS    = deep_cfr_cuda.N_LAYERS     # 4 hidden layers
LEAKY_ALPHA = 0.01

# Architecture: 4 hidden + 1 output, matching the regret net
_LAYER_DIMS = (
    [(HIDDEN_DIM, INPUT_DIM)]
    + [(HIDDEN_DIM, HIDDEN_DIM)] * (N_LAYERS - 1)
    + [(N_ACTIONS, HIDDEN_DIM)]
)


def _make_weights(rng):
    """Return [W0,b0,...,W4,b4] matching the regret net (linear output)."""
    weights = []
    for out_d, in_d in _LAYER_DIMS:
        W = rng.standard_normal((out_d, in_d)).astype(np.float32)
        b = rng.standard_normal(out_d).astype(np.float32)
        weights.extend([W, b])
    return weights


def _numpy_forward(weights, x):
    """Reference forward: same math as Eigen forward_single, linear output.
    x: [batch x in_dim]  →  returns [batch x out_dim]
    """
    n = len(weights) // 2
    h = x.copy()
    for i in range(n):
        W = weights[i * 2]
        b = weights[i * 2 + 1]
        h = h @ W.T + b
        if i < n - 1:
            h = np.where(h < 0, LEAKY_ALPHA * h, h)
    return h


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_shapes():
    rng = np.random.default_rng(0)
    weights = _make_weights(rng)
    x = rng.standard_normal((32, INPUT_DIM)).astype(np.float32)
    out = deep_cfr_cuda.forward_batch(weights, x)
    assert out.shape == (32, N_ACTIONS), f"Expected (32, {N_ACTIONS}), got {out.shape}"
    assert out.dtype == np.float32


@pytest.mark.parametrize("batch", [1, 16, 512, 4096])
def test_parity(batch):
    """GPU forward must agree with numpy reference to within 1e-3 (float32 rounding)."""
    rng = np.random.default_rng(42)
    weights = _make_weights(rng)
    x = rng.standard_normal((batch, INPUT_DIM)).astype(np.float32)

    ref = _numpy_forward(weights, x)
    gpu = deep_cfr_cuda.forward_batch(weights, x)

    max_diff = float(np.abs(ref - gpu).max())
    mean_diff = float(np.abs(ref - gpu).mean())
    print(f"  batch={batch:5d}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")
    assert max_diff < 1e-3, (
        f"Max abs diff {max_diff:.2e} >= 1e-3 at batch={batch} — "
        "likely cuBLAS GEMM order differs from numpy; investigate if > 1e-2"
    )


def test_deterministic():
    """Two calls with identical inputs must produce bit-identical outputs."""
    rng = np.random.default_rng(7)
    weights = _make_weights(rng)
    x = rng.standard_normal((64, INPUT_DIM)).astype(np.float32)

    out1 = deep_cfr_cuda.forward_batch(weights, x)
    out2 = deep_cfr_cuda.forward_batch(weights, x)
    assert np.array_equal(out1, out2), "forward_batch is not deterministic"


def test_single_sample_matches_batch():
    """forward_batch(x[0:1]) must match forward_batch(x)[0]."""
    rng = np.random.default_rng(99)
    weights = _make_weights(rng)
    x = rng.standard_normal((8, INPUT_DIM)).astype(np.float32)

    batch_out = deep_cfr_cuda.forward_batch(weights, x)
    for i in range(8):
        single_out = deep_cfr_cuda.forward_batch(weights, x[i : i + 1])
        diff = float(np.abs(batch_out[i] - single_out[0]).max())
        assert diff < 1e-4, (
            f"Sample {i}: batch result differs from single-sample result by {diff:.2e}"
        )


def test_zero_input():
    """All-zero input should give bias-only output through LeakyReLU layers."""
    rng = np.random.default_rng(5)
    weights = _make_weights(rng)
    x = np.zeros((4, INPUT_DIM), dtype=np.float32)

    ref = _numpy_forward(weights, x)
    gpu = deep_cfr_cuda.forward_batch(weights, x)
    max_diff = float(np.abs(ref - gpu).max())
    assert max_diff < 1e-5, f"Zero-input diff {max_diff:.2e} exceeds 1e-5"
