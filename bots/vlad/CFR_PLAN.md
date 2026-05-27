Here is a complete, technical Markdown specification designed to be fed directly into an autonomous coding agent like Claude Code or GitHub Copilot Workspaces. It provides the architectural blueprint, data structures, tensor shapes, and exact algorithmic steps needed to implement Deep CFR using PyTorch.

***

# Deep CFR 6-Max Poker Bot Specification

## 1. Overview and Architecture
This system implements Deep Counterfactual Regret Minimization (Deep CFR) for 6-max No-Limit Texas Hold'em. To fit within a 150MB memory footprint for the final bot, it uses Neural Networks to approximate regrets and strategies instead of tabular lookup dictionaries.

### Proposed File Structure
```text
deep_cfr/
│
├── config.py         # Hyperparameters and action abstractions
├── env.py            # Poker environment / Game state representation
├── features.py       # Converts Game State -> PyTorch Tensors
├── networks.py       # PyTorch model definitions (RegretNet, StrategyNet)
├── memory.py         # Reservoir Sampling Buffers
├── mccfr.py          # Core tree traversal logic (External Sampling)
├── train.py          # Orchestrates data generation and network training
└── export.py         # Exports final StrategyNet for the production bot
```

---

## 2. Environment & Action Abstraction (`env.py`, `config.py`)

### Action Abstraction
To keep the action space tractable, continuous bet sizes are discretized.
**Output Action Space (`N_ACTIONS = 5`)**:
0. `FOLD`
1. `CHECK_CALL`
2. `BET_HALF_POT`
3. `BET_FULL_POT`
4. `ALL_IN`

### Required `GameState` Interface
The environment must provide a `GameState` object with the following methods for the MCCFR traversal:
* `is_terminal() -> bool`: Returns true if the hand is over.
* `get_payoffs() -> list[float]`: Returns a 6-element list of chip changes for each player.
* `is_chance_node() -> bool`: Returns true if cards need to be dealt.
* `sample_chance_event() -> GameState`: Returns a new state with cards dealt.
* `current_player() -> int`: Returns the index (0-5) of the acting player.
* `get_legal_actions() -> list[int]`: Returns a list of valid action indices from the action space.
* `apply_action(action_idx: int) -> GameState`: Returns a new state after applying the action.

---

## 3. State Representation / Tensorization (`features.py`)

The neural networks require a fixed-size 1D float tensor representing the current game state. 

**Input Vector Shape: `[189]`**
* **Hole Cards (52 floats):** 1.0 if in hand, 0.0 otherwise.
* **Board Cards (52 floats):** 1.0 if on board, 0.0 otherwise.
* **Player Position (6 floats):** One-hot encoded (e.g., UTG = `[1, 0, 0, 0, 0, 0]`).
* **Pot & Stack Sizes (7 floats):** 
    * Normalized Pot Size (`Pot / Initial Buy-in`)
    * Normalized Stack Sizes for all 6 players (`Stack / Initial Buy-in`)
* **Betting History (72 floats):** 
    * 4 betting rounds (Pre-flop, Flop, Turn, River)
    * Maximum 6 actions allowed per round = 24 action slots.
    * Each slot is 3 floats: `[Player_ID_Normalized (0-1), Action_Type_Normalized (0-1), Bet_Fraction_Of_Pot]`.
    * Unused slots remain `0.0`.

*Agent instruction: Implement a `StateToTensor(state: GameState) -> torch.Tensor` function based on this schema.*

---

## 4. Neural Network Architectures (`networks.py`)

Implement two identical Multi-Layer Perceptron (MLP) architectures using PyTorch. 

**Architecture Specifications:**
* **Input Layer:** 189 nodes
* **Hidden Layers:** 4 to 6 layers of 512 nodes, using `LeakyReLU` activations.
* **Output Layer:** 5 nodes (one for each action in the abstraction).
* **Final Activation:** 
  * `RegretNet`: No activation (Linear output, as regrets can be negative or positive).
  * `StrategyNet`: `Softmax` (Outputs must sum to 1.0).

```python
# Pseudo-code representation
class PokerNet(nn.Module):
    # Linear(189, 512) -> LeakyReLU -> Linear(512, 512) ... -> Linear(512, 5)
```
*Note: A 5x512 MLP has ~1 million parameters, taking up ~4MB of disk space.*

---

## 5. Memory Buffers (`memory.py`)

Due to the massive amount of data generated, standard lists will cause Out-Of-Memory (OOM) errors. Implement **Reservoir Sampling** to maintain a fixed-capacity memory that represents a uniform sample of all historical data.

* **Regret Buffer Capacity:** ~4,000,000 entries.
* **Strategy Buffer Capacity:** ~4,000,000 entries.

**Data Structures:**
* Regret Memory Entry: `Tuple[torch.Tensor (state), torch.Tensor (5 instantaneous regrets)]`
* Strategy Memory Entry: `Tuple[torch.Tensor (state), torch.Tensor (5 action probabilities), float (iteration weight)]`

---

## 6. MCCFR Algorithm (`mccfr.py`)

Implement the External Sampling MCCFR traversal recursive function.

**Inputs:**
* `state`: Current `GameState`
* `traverser`: The player ID (0-5) for whom we are calculating regrets.
* `regret_net`: The current value network.
* `regret_memory`, `strategy_memory`: The reservoir buffers.
* `iteration_t`: Current training iteration (used for weighting).

**Logic Flow:**
1. **Terminal Node:** Return `state.get_payoffs()[traverser]`.
2. **Chance Node:** Call `sample_chance_event()` and recurse.
3. **Get Strategy:**
   * Convert `state` to tensor.
   * Pass through `regret_net` to get predicted regrets.
   * Apply Regret Matching: $P(a) = \frac{\max(R(a), 0)}{\sum \max(R(a), 0)}$. (If all $\le 0$, use uniform distribution over legal actions).
4. **Opponent Node (`state.current_player() != traverser`):**
   * Append `(tensor, strategy, iteration_t)` to `strategy_memory`.
   * Sample *one* action $a$ according to $P(a)$.
   * Recurse on that action: return `mccfr(state.apply_action(a), ...)`.
5. **Traverser Node (`state.current_player() == traverser`):**
   * Initialize `action_evs = []`.
   * For *every* legal action $a$:
     * $EV_a = $ `mccfr(state.apply_action(a), ...)`
     * Append $EV_a$ to `action_evs`.
   * Calculate node EV: $V = \sum (P(a) \times EV_a)$.
   * Calculate instantaneous regret for each action: $R_a = EV_a - V$. (If action is illegal, $R_a = 0$).
   * Append `(tensor, [R_0, R_1, R_2, R_3, R_4])` to `regret_memory`.
   * Return $V$.

---

## 7. The Training Loop (`train.py`)

Orchestrate the Deep CFR pipeline. 

**Hyperparameters:**
* `K_ITERATIONS = 100` (Outer loop sweeps)
* `GAMES_PER_ITERATION = 10_000`
* `BATCH_SIZE = 4096`
* `LEARNING_RATE = 0.001` (Adam Optimizer)

**Pipeline:**
1. Initialize `regret_net` and `strategy_net` (and their respective buffers).
2. Loop `t` from 1 to `K_ITERATIONS`:
   * **Data Generation:**
     * Loop 0 to `GAMES_PER_ITERATION`:
       * Randomly select `traverser` (0 to 5).
       * Create fresh `GameState`.
       * Call `mccfr(state, traverser, regret_net, buffers, t)`.
   * **Train Regret Network:**
     * Re-initialize `regret_net` weights from scratch (optional but recommended in some Deep CFR variants, or just train on the reservoir).
     * Train for `E` epochs on `regret_memory` using Mean Squared Error (MSE) loss predicting the instantaneous regrets.
3. **Train Strategy Network (Final Step):**
   * Train `strategy_net` on `strategy_memory`.
   * Loss Function: Cross-Entropy Loss (or MSE on probabilities), weighted by the `iteration_t` stored in the buffer.
4. **Export:**
   * Save `strategy_net.state_dict()` to `gto_6max_bot.pth`.

---

## 8. Production Bot Execution (`export.py`)

The final agent script.

1. Load `strategy_net.pth` (discard the regret net and all memory buffers).
2. Observe table state, format into the 189-float tensor.
3. Run forward pass through `strategy_net`.
4. Output is a probability distribution over the 5 abstract actions.
5. Filter out illegal actions, re-normalize probabilities to 1.0.
6. Sample a random number and execute the chosen action.