"""
Export the average strategy table to bots/vlad/data/preflop_cfr/preflop_strategy.npz.

Format:
    keys      int64[N]    — FNV-1a 64-bit info-set hashes
    strategy  float32[N,9] — average strategy probability vectors (length-9)
    version   scalar str
    n_players scalar int
    stack_bb  scalar int   — initial stack in big blinds
    actions   str          — JSON list of active PREFLOP_ACTIONS indices

No allow_pickle / object arrays — passes the sandbox validator.
"""

from __future__ import annotations

import json
import os

import numpy as np

from preflop_cfr import config


VERSION = "1"


def export_strategy(
    strategy_sum: dict[int, np.ndarray],
    path: str = config.EXPORT_PATH,
    min_visits: int = config.PRUNE_MIN_VISITS,
) -> int:
    """
    Normalise strategy_sum into average strategy, prune low-visit info sets,
    and write an .npz file.  Returns the number of info sets exported.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    keys_list:  list[int]        = []
    strats_list: list[np.ndarray] = []

    for key, ssum in strategy_sum.items():
        total = ssum.sum()
        if total < min_visits:
            continue
        avg = (ssum / total).astype(np.float32)
        keys_list.append(key)
        strats_list.append(avg)

    n = len(keys_list)
    if n == 0:
        raise ValueError("No info sets to export (all below min_visits threshold)")

    keys_arr     = np.array(keys_list,  dtype=np.int64)
    strategy_arr = np.stack(strats_list, axis=0).astype(np.float32)

    np.savez(
        path,
        keys      = keys_arr,
        strategy  = strategy_arr,
        version   = np.array(VERSION),
        n_players = np.array(config.N_PLAYERS),
        stack_bb  = np.array(config.INITIAL_STACK // config.BIG_BLIND),
        actions   = np.array(json.dumps(config.PREFLOP_ACTIONS)),
    )
    return n


def load_strategy(path: str = config.EXPORT_PATH) -> dict[int, np.ndarray]:
    """
    Load exported strategy table into a dict keyed by int64 hash.
    Validates metadata for compatibility.
    """
    data = np.load(path)

    n_players = int(data["n_players"])
    stack_bb  = int(data["stack_bb"])
    expected_bb = config.INITIAL_STACK // config.BIG_BLIND

    if n_players != config.N_PLAYERS:
        raise ValueError(
            f"Strategy table n_players={n_players} != config {config.N_PLAYERS}"
        )
    if stack_bb != expected_bb:
        raise ValueError(
            f"Strategy table stack_bb={stack_bb} != config {expected_bb}"
        )

    keys     = data["keys"]
    strategy = data["strategy"]
    return {int(k): strategy[i] for i, k in enumerate(keys)}


def save_checkpoint(
    regret_sum:   dict[int, np.ndarray],
    strategy_sum: dict[int, np.ndarray],
    iteration:    int,
    path: str = config.CHECKPOINT_PATH,
):
    """Save raw regret/strategy sums for resuming training."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not regret_sum:
        return

    r_keys   = np.array(list(regret_sum.keys()),   dtype=np.int64)
    r_values = np.stack(list(regret_sum.values()),  axis=0)
    s_keys   = np.array(list(strategy_sum.keys()),  dtype=np.int64)
    s_values = np.stack(list(strategy_sum.values()), axis=0)

    np.savez(
        path,
        r_keys     = r_keys,
        r_values   = r_values,
        s_keys     = s_keys,
        s_values   = s_values,
        iteration  = np.array(iteration),
    )


def load_checkpoint(
    path: str = config.CHECKPOINT_PATH,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]:
    """Load regret/strategy sums. Returns (regret_sum, strategy_sum, iteration)."""
    data = np.load(path)

    regret_sum:   dict[int, np.ndarray] = {}
    strategy_sum: dict[int, np.ndarray] = {}

    for k, v in zip(data["r_keys"], data["r_values"]):
        regret_sum[int(k)] = v.copy()
    for k, v in zip(data["s_keys"], data["s_values"]):
        strategy_sum[int(k)] = v.copy()

    iteration = int(data["iteration"])
    return regret_sum, strategy_sum, iteration
