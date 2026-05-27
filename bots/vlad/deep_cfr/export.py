"""
Export a trained PokerNet to a numpy .npz file for production inference.

The .npz file contains paired weight/bias arrays for each layer:
    layer0_w, layer0_b, layer1_w, layer1_b, …

The production bot (bot.py) loads this file and runs a pure-numpy
forward pass — no PyTorch dependency required at inference time.
"""

from __future__ import annotations
import os
import numpy as np
import torch.nn as nn

from .networks import PokerNet


def export_net(net: PokerNet, path: str) -> None:
    """
    Save PokerNet weights to a .npz archive.

    File layout:
        layer{i}_w   float32 [out_dim, in_dim]  – weight matrix
        layer{i}_b   float32 [out_dim]           – bias vector
        n_layers     scalar int                  – number of saved layers
    """
    net.eval()
    arrays: dict[str, np.ndarray] = {}
    layer_idx = 0

    for module in net.net:
        if isinstance(module, nn.Linear):
            w = module.weight.detach().cpu().numpy()   # [out, in]
            b = module.bias.detach().cpu().numpy()     # [out]
            arrays[f"layer{layer_idx}_w"] = w
            arrays[f"layer{layer_idx}_b"] = b
            layer_idx += 1

    arrays["n_layers"] = np.array(layer_idx, dtype=np.int32)
    np.savez(path, **arrays)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Exported {layer_idx} layers to {path} ({size_mb:.1f} MB)")
