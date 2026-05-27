# Multi-way Convergence Improvements

Implement two targeted improvements to the Deep CFR training pipeline to increase
CFR convergence in 4+-way pot situations. The model is currently undertrained in
these spots because External Sampling MCCFR generates samples proportional to how
often each situation arises in a GTO game — multi-way flops are rare, so their
regret estimates are thin.

## Overview

**Option 1 — Buffer oversampling**: tag every regret/strategy sample with
`n_active` (active player count at that state). In `_parallel_generate`, repeat
samples from 4+-way states `MULTIWAY_WEIGHT=4` times before feeding them into the
reservoir buffers. No architecture changes; works with the existing 189-dim network
if Option 2 is not implemented at the same time.

**Option 2 — Explicit `n_active` feature**: insert a single `n_active` scalar at
index 117 in the feature vector, shifting the action-history block from `[117:189]`
to `[118:190]`. Expand `INPUT_DIM` from 189 to 190. Must be applied atomically
across `config.py`, `features.py`, and `bot.py` — mismatches will silently corrupt
inference.

Both options can be implemented together (recommended) or independently.

---

## Files to change

```
bots/vlad/deep_cfr/config.py     Option 2 only  — INPUT_DIM 189 → 190
bots/vlad/deep_cfr/features.py   Option 2 only  — insert n_active at [117], shift history
bots/vlad/deep_cfr/mccfr.py      Option 1       — attach n_active to sample tuples
bots/vlad/deep_cfr/train.py      Option 1       — repeat multi-way samples in aggregation
bots/vlad/bot.py                 Option 2 only  — mirror feature vector change
```

---

## Option 2 — Feature vector expansion

### `bots/vlad/deep_cfr/config.py`

Change `INPUT_DIM`:
```python
INPUT_DIM = 190   # was 189; +1 for explicit n_active feature at index 117
```

### `bots/vlad/deep_cfr/features.py`

Current layout in `build_feature_vector`:
- `[110]` pot, `[111:117]` stacks, history loop starts at `117 + slot * 3`

After change:
- `[110]` pot, `[111:117]` stacks, **`[117]` n_active**, history loop starts at `118 + slot * 3`

Specific edits:

1. Update the module docstring to reflect the new layout:
   ```
   [117]      – normalised active player count  (n_active / (N_PLAYERS - 1))
   [118:190]  – last 24 regular actions × 3 floats each: (was [117:189])
   ```

2. After the stacks block (after `for p in players[:N_PLAYERS]:`), insert:
   ```python
   # ── 4b. Active player count ───────────────────────────────────────────────
   n_active = sum(1 for p in players if p["state"] in ("active", "all_in"))
   vec[117] = n_active / max(N_PLAYERS - 1, 1)
   ```

3. In the betting history loop, change the base index:
   ```python
   base = 118 + slot * 3   # was 117 + slot * 3
   ```

### `bots/vlad/bot.py`

`bot.py` contains `_build_feature_vector` which mirrors `features.py` exactly.
Apply the same two edits:

1. After the stacks block (`for p in gs["players"][:_N_PLAYERS]:`), insert:
   ```python
   # Active player count [117]
   n_active = sum(1 for p in gs["players"] if p["state"] in ("active", "all_in"))
   vec[117] = n_active / max(_N_PLAYERS - 1, 1)
   ```
   Note: bot.py uses `gs["players"]` and `_N_PLAYERS`, not the training-side names.

2. In the action-history loop, change the base index:
   ```python
   base = 118 + slot * 3   # was 117 + slot * 3
   ```

---

## Option 1 — Buffer oversampling

### `bots/vlad/deep_cfr/mccfr.py`

At both sample-collection sites, compute `n_active` and append it to the tuple.
The training code strips the tag before inserting into the reservoir.

**Opponent node** (currently around line 82–86 — the `strategy_memory.add(...)` call):
```python
n_active = sum(
    1 for p in state._state_dict["players"]
    if p["state"] in ("active", "all_in")
)
strategy_memory.add((
    state_vec,
    strategy,
    float(iteration_t),
    n_active,           # new: tag for multi-way weighting
))
```

**Traverser node** (currently around line 117–120 — the `regret_memory.add(...)` call):
```python
n_active = sum(
    1 for p in state._state_dict["players"]
    if p["state"] in ("active", "all_in")
)
regret_memory.add((
    state_vec,
    regret_vec,
    n_active,           # new: tag for multi-way weighting
))
```

### `bots/vlad/deep_cfr/train.py`

Add the weight constant near the top of the file (after imports):
```python
MULTIWAY_WEIGHT = 4   # multi-way samples (4+ active players) repeated this many times
```

In `_parallel_generate`, replace the aggregation loop:
```python
for r_data, s_data in results:
    for item in r_data:
        reps = MULTIWAY_WEIGHT if item[2] >= 4 else 1  # item[2] is n_active
        for _ in range(reps):
            regret_buf.add((item[0], item[1]))          # strip tag before storing
    for item in s_data:
        reps = MULTIWAY_WEIGHT if item[3] >= 4 else 1  # item[3] is n_active
        for _ in range(reps):
            strategy_buf.add((item[0], item[1], item[2]))  # strip tag, keep weight
```

The existing `_train_regret` and `_train_strategy` functions are unchanged — they
already expect 2-tuples and 3-tuples respectively, which is what they'll receive
after the tag is stripped above.

---

## Verification

After making all changes, run the test suite and a quick smoke-train to confirm
nothing is broken before committing to a full run:

```bash
make test
python -m bots.vlad.deep_cfr.train --quick
```

`--quick` runs 5 iterations × 200 games (~15 min on 16 cores). Check that:
- No shape mismatch errors in the network forward pass
- `regret_buf` and `strategy_buf` grow as expected
- The exported `gto_strategy.npz` loads correctly in `bot.py` at import time (`make validate BOT=bots/vlad/bot.py`)

Full training command after smoke test passes:
```bash
python -m bots.vlad.deep_cfr.train
```
