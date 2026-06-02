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


def load_net(net: PokerNet, path: str) -> PokerNet:
    """
    Load weights saved by export_net back into a PokerNet (in place).

    Reads the layer{i}_w / layer{i}_b arrays and copies them into the Linear
    modules of `net` in order. The architecture (INPUT_DIM/HIDDEN_DIM/N_LAYERS)
    must match the one the checkpoint was exported from. Returns `net`.
    """
    import torch

    data = np.load(path)
    linears = [m for m in net.net if isinstance(m, nn.Linear)]
    n_saved = int(data["n_layers"])
    if n_saved != len(linears):
        raise ValueError(
            f"Checkpoint {path} has {n_saved} layers but the network has "
            f"{len(linears)} — architecture mismatch."
        )
    with torch.no_grad():
        for i, layer in enumerate(linears):
            w = torch.from_numpy(data[f"layer{i}_w"])
            b = torch.from_numpy(data[f"layer{i}_b"])
            if tuple(layer.weight.shape) != tuple(w.shape):
                raise ValueError(
                    f"Checkpoint {path} layer{i} weight shape {tuple(w.shape)} "
                    f"!= network shape {tuple(layer.weight.shape)}."
                )
            layer.weight.copy_(w)
            layer.bias.copy_(b)
    print(f"Loaded {n_saved} layers from {path}")
    return net
