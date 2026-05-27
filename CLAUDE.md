# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This is a fork of the Fullhouse poker bot competition engine. The competition runs 1–5 June 2026. Work happens in `bots/vlad/` — a No-Limit Texas Hold'em bot competing in a Swiss-system tournament. All other engine files (`engine/`, `sandbox/`, `demo.py`) are frozen and should not be modified.

## Commands

```bash
# Install dependencies (eval7 requires Cython<3 + no-build-isolation)
make install

# Run tests
make test
# or directly:
python -m pytest tests/ -q

# Run a single test file
python -m pytest tests/test_game.py -q

# Validate bot before submission
make validate BOT=bots/vlad/bot.py

# Run interactive demo UI (localhost:5000)
make demo

# Clean compiled artifacts
make clean
```

## Architecture

### Engine (frozen)

**`engine/game.py`** — `PokerEngine` class: pure-Python NLHE rules for 2–9 players. Constants: `SMALL_BLIND=50`, `BIG_BLIND=100`, `STARTING_STACK=10_000`. Handles betting rounds, side pots, showdowns via eval7. Emits a structured event log per hand.

**`engine/tournament.py`** — Swiss-system pairing and standings. Ranking tiebreakers: cumulative chip delta → matches played → best match delta → bot_id.

**`sandbox/runner.py`** — Bot process executor. Loads `bot.py`, communicates via newline-delimited JSON on stdin/stdout. Enforces **2s per-action timeout** (threading-based for Windows compat) and **30s warmup timeout**.

**`sandbox/match.py`** — Orchestrates a full 400-hand match. Injects a rolling `match_action_log` (up to 200 entries) so bots can model opponents across hands. Returns chip deltas and hand history.

**`sandbox/validator.py`** — Static + dynamic pre-submission checks. Forbidden imports: `socket`, `requests`, `subprocess`, `multiprocessing`, `pickle`, `eval`, `exec`, `compile`.

### Bot Protocol

Each action, `runner.py` calls `decide(game_state: dict) -> dict` in the bot. The game state includes:

- `your_cards`: `["As", "Kh"]` — hole cards
- `community_cards`: board cards (empty preflop)
- `street`: `"preflop" | "flop" | "turn" | "river"`
- `pot`, `your_stack`, `amount_owed`, `can_check`, `current_bet`, `min_raise_to`
- `players`: list of opponent info (stack, state, bet_this_street)
- `action_log`: all actions this hand
- `match_action_log`: rolling cross-hand history (up to 200 entries)

Valid return values:
```python
{"action": "fold"}
{"action": "check"}
{"action": "call"}
{"action": "raise", "amount": <int>}  # total bet amount, >= min_raise_to
{"action": "all_in"}
```

### `bots/vlad/bot.py` — Current Implementation

Loads a pre-trained GTO strategy network at import time (`bots/vlad/data/gto_strategy.npz`). Falls back to Monte Carlo equity if the model is unavailable.

- **GTO path**: runs a pure-numpy forward pass through the strategy network, picks the highest-probability legal action.
- **MC fallback**: `monte_carlo_equity(hole_cards, board_cards, remaining_deck, num_opponents, time_limit=0.5)` — shuffles deck, evaluates with eval7, returns win probability. `choose_action(...)` compares equity to pot-odds threshold plus a buffer (10% HU, 15% for 3–4, 30% for 5+).

### `bots/vlad/deep_cfr/` — Deep CFR Training System

Offline training pipeline (not used at runtime). Produces `bots/vlad/data/gto_strategy.npz`.

**Key files:**

- **`config.py`** — all hyperparameters: `N_ACTIONS=5`, `INPUT_DIM=189`, `HIDDEN_DIM=512`, `N_LAYERS=4`, `K_ITERATIONS=100`, `GAMES_PER_ITER=10_000`, `BATCH_SIZE=4_096`, `N_WORKERS` (auto = CPU count).
- **`env.py`** — `GameState` wrapping the engine. Implements `is_terminal()`, `is_chance_node()`, `get_legal_actions()`, `apply_action()`, `get_payoffs()`, `sample_chance_event()`, `current_player()`.
- **`features.py`** — `build_feature_vector(state_dict)` → 189-float numpy array (52 hole-card one-hot + 52 board one-hot + 6 position one-hot + 7 pot/stack scalars + 72 action-history).
- **`networks.py`** — `make_regret_net()` and `make_strategy_net()` (4-layer MLP, ReLU, 512 hidden). Output dim = 5 (one per abstract action).
- **`memory.py`** — `ReservoirBuffer` with reservoir sampling. `.add()`, `.sample(n)`, `.is_ready(n)`, `len()`.
- **`mccfr.py`** — External Sampling MCCFR. Opponent nodes sample one action and add `(state_vec, strategy, t)` to strategy memory. Traverser nodes enumerate all actions and add `(state_vec, regret_vec)` to regret memory. Buffers store raw numpy arrays (not tensors) for cheap IPC pickling.
- **`export.py`** — `export_strategy_net(net, path)` serialises weights as numpy `.npz` for sandbox inference.
- **`train.py`** — Orchestrator. Parallelises data generation across `N_WORKERS` OS processes via `ProcessPoolExecutor` (bypasses GIL). Worker function `_worker_batch` is module-level (required for Windows spawn pickling). Weights shipped to workers as numpy dicts; converted back to torch inside workers. `n_workers=1` runs in-process (no subprocess overhead).

**Abstract action space** (index → meaning):

| Index | Name | Description |
|---|---|---|
| 0 | FOLD | Fold |
| 1 | CHECK_CALL | Check or call |
| 2 | BET_HALF_POT | Bet/raise to 0.5× pot |
| 3 | BET_FULL_POT | Bet/raise to 1× pot |
| 4 | ALL_IN | All-in |

**Training commands:**

```bash
# Smoke test (5 iters × 200 games, ~15 min on 16 cores)
python -m bots.vlad.deep_cfr.train --quick

# Full training (100 iters × 10,000 games, several hours)
python -m bots.vlad.deep_cfr.train

# Control worker count explicitly
python -m bots.vlad.deep_cfr.train --workers 8
```

Output: `bots/vlad/data/gto_strategy.npz` (3.6 MB, read-only at runtime).

**Important constraints:** PyTorch is only needed for training (offline). The sandbox forbids subprocess/multiprocessing, so `train.py` must never be imported by `bot.py`. The numpy-only forward pass in `bot.py` is what runs in production.

## Hardware Constraints (bots/vlad/)

All code in `bots/vlad/` must operate within these production sandbox limits:

| Resource | Limit |
|---|---|
| RAM | 768 MB |
| CPU | 0.5 cores |
| Action timeout | 2 seconds |
| Warmup timeout | 30 seconds |
| `/tmp` | 20 MB |
| `data/` directory | 200 MB (read-only) |
| Total submission | 250 MB |

Heavy resources (model weights, lookup tables) must be loaded once at module import time (warmup), not inside `decide()`. The filesystem is read-only at runtime — no file writes. No network access. No subprocess/threading spawning inside bots.

## Submission Format

A valid submission is either:
- `bots/vlad/bot.py` (single file, ≤ 5 MB)
- `bots/vlad/bot.zip` containing `bot.py` at root + optional `data/` subdirectory (≤ 200 MB)

Data files in `data/` are mounted read-only at `/data` inside the sandbox.
