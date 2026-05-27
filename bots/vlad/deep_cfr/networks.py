"""
PyTorch MLP definitions for Deep CFR.

RegretNet  – linear output  (regrets can be negative or positive)
StrategyNet – softmax output (action probabilities summing to 1)

Both share the same MLP backbone (4 hidden layers × 512 nodes, LeakyReLU).
A 512-dim, 4-layer MLP has ≈1 M parameters → ≈4 MB on disk.
"""

import torch
import torch.nn as nn
from .config import INPUT_DIM, HIDDEN_DIM, N_LAYERS, N_ACTIONS


class PokerNet(nn.Module):
    """
    Generic MLP for poker regret / strategy estimation.

    Args:
        output_activation: "linear" for RegretNet, "softmax" for StrategyNet.
    """

    def __init__(self, output_activation: str = "linear") -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = INPUT_DIM
        for _ in range(N_LAYERS):
            layers.append(nn.Linear(in_dim, HIDDEN_DIM))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            in_dim = HIDDEN_DIM
        layers.append(nn.Linear(HIDDEN_DIM, N_ACTIONS))
        if output_activation == "softmax":
            layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_regret_net() -> PokerNet:
    return PokerNet(output_activation="linear")


def make_strategy_net() -> PokerNet:
    return PokerNet(output_activation="softmax")
