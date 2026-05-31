# CUDA Data Generation for Deep CFR — Implementation Plan

> **Audience:** a Claude Code session running on a Linux Cloud VM with an NVIDIA GPU.
> **Goal:** accelerate Deep CFR data generation by moving the per-node neural-net
> forward pass (the bottleneck) from CPU/Eigen onto the GPU as batched cuBLAS GEMMs,
> while keeping the entire downstream training pipeline (`DeepCFRBuffers`, the
> reservoirs, `deep_cfr/train.py`) **byte-for-byte unchanged**.
>
> **Status when this doc was written:** Phase 0 (GPU MLP forward + parity test) is
> already coded and committed. It has **not yet been built or run on a GPU** — the
> author's dev machine has no NVIDIA card. Your first job is to validate Phase 0,
> then implement Phases 1–4.

---

## 0. Orientation — read this first

### Repository facts

- Repo root contains: `engine/` (frozen), `sandbox/` (frozen), `bots/vlad/` (the bot),
  `deep_cfr/` (Python training orchestration), `deep_cfr_cpp/` (C++/CUDA data gen),
  `preflop_cfr/` (unrelated tabular solver), `tests/`.
- **Only touch `deep_cfr_cpp/` and, if needed, `deep_cfr/train.py` + `Makefile` + `tests/`.**
  Everything under `engine/`, `sandbox/`, `demo.py` is frozen — do not modify.
- This is offline training code. The sandbox restrictions that apply to
  `bots/vlad/bot.py` (no threads, no subprocess, 768 MB RAM, 2 s timeout) **do NOT
  apply here**. Use all the cores, RAM, and GPU you want.

### Environment setup on the VM

```bash
# One-shot bootstrap: Python 3.10 venv + CUDA torch + CPU C++ build
make install-vm
source .venv/bin/activate

# Sanity-check the GPU toolchain
nvcc --version
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `nvcc` is missing, install the CUDA toolkit (matching the driver shown by
`nvidia-smi`). The cuBLAS dev headers (`cublas_v2.h`) ship with the toolkit.

### The data-generation pipeline as it exists today (CPU)

The canonical training loop is **Python** (`deep_cfr/train.py`). Each of K=100
iterations:

1. Serialise the current PyTorch **regret net** weights to a list of numpy arrays.
2. Call `cpp_buffers.generate_and_add(n_games, n_workers, iteration_t, weights)`
   — this is the C++ `DeepCFRBuffers` method (in `deep_cfr_cpp/src/binding.cpp`).
   It spawns `n_workers` `std::thread`s, each running External-Sampling MCCFR
   traversals, writing `RegretSample` and `StrategySample` records into two shared
   thread-safe `ReservoirBuffer`s.
3. Train a fresh regret net on samples drawn from the regret reservoir.
4. Periodically train + export the strategy net to `bots/vlad/data/gto_strategy.npz`.

**The bottleneck is step 2**, specifically `forward_single()` in
`deep_cfr_cpp/src/network.cpp` — a 308→512→512→512→512→9 MLP evaluated once at
*every* decision node, tens of millions of times per iteration, as tiny Eigen
GEMVs. Batching these into GPU GEMMs is the entire point of this project.

### Critical insight: data gen only uses the REGRET net, with LINEAR output

Look at `binding.cpp::mlp_from_weights` → `make_inference_net(false)`:
`softmax_output = false`. The strategy net is trained *offline* from collected
`StrategySample`s and **never runs during traversal**. Therefore the GPU forward
pass is a pure **linear-output** MLP — **no softmax kernel needed**.

### Second insight: the only chance node is the root

Community cards are dealt *inside* `apply_action` when a street advances (see
`engine.cpp`). So `GameState::is_chance_node()` is true **only at the initial
deal**. `train.cpp::run_worker` deals once (`sample_chance_event`), then runs
`mccfr` for each of the 6 traversers on that already-dealt state. Every interior
node is a decision node needing exactly one forward pass. The tree is a clean
alternating structure — no interior chance handling required in the GPU port.

### File inventory of `deep_cfr_cpp/src/`

| File | Role | Touch in this project? |
|---|---|---|
| `config.hpp` | All constants: `INPUT_DIM=308`, `N_ACTIONS=9`, `HIDDEN_DIM=512`, `N_LAYERS=4`, action indices, `MAX_DEPTH=200`, `MAX_RAISES_PER_STREET=8`, `LEAKY_ALPHA=0.01` | Read-only reference |
| `card.hpp` | Card type + suit/rank helpers | No |
| `engine.hpp/.cpp` | `PokerEngine`: fixed-size, trivially copyable NLHE rules | Reuse as-is |
| `hand_eval.hpp/.cpp` | Showdown hand strength | Reuse as-is |
| `features.hpp/.cpp` | `build_feature_vector(StateDict)` → 308 floats. **Layout must stay byte-for-byte identical to bot.py.** | Reuse as-is (Phase 2); port to device (Phase 4) |
| `network.hpp/.cpp` | Eigen MLP, `forward_single` (CPU reference) | Reuse as the parity oracle |
| `mccfr.hpp/.cpp` | `GameState` wrapper + recursive `mccfr()` traversal | **Convert to state machine (Phase 1)** |
| `reservoir.hpp` | Thread-safe `ReservoirBuffer<T>` (Vitter Algorithm R) | Reuse as-is |
| `train.hpp/.cpp` | `parallel_generate` — spawns CPU worker threads | Reference; CUDA gets its own driver |
| `binding.cpp` | pybind11 module `deep_cfr_gen` → `DeepCFRBuffers` | Reference; keep working as CPU fallback |
| `network_cuda.cuh/.cu` | **Phase 0 — DONE.** cuBLAS batched MLP forward (opaque handle) | Extend in Phase 2/3 |
| `binding_cuda.cpp` | **Phase 0 — DONE.** pybind11 module `deep_cfr_cuda` → `forward_batch` | Extend in Phase 2/3 |

### Invariants you must NOT break

1. **The 308-dim feature layout** is identical across `features.cpp`, `config.py`,
   and `bots/vlad/bot.py`. Any drift silently corrupts the trained model. If you
   port features to device code in Phase 4, add a parity test against the CPU
   `build_feature_vector`.
2. **The `DeepCFRBuffers` Python API** (`generate_and_add`, `sample_regret_into`,
   `sample_strategy_into`, `regret_ready`, etc. — see `binding.cpp`) must remain
   call-compatible so `deep_cfr/train.py` needs minimal or zero changes.
3. **Reservoir semantics** (Vitter Algorithm R, capacity 4M each). Samples must
   end up in `ReservoirBuffer` via `add_batch`. Easiest path: collect samples
   host-side, push through the existing `ReservoirBuffer` unchanged.
4. **Linear-CFR weighting:** every sample carries `weight = iteration_t`.
5. **MCCFR correctness details** — preserve exactly (see Phase 1 checklist).
6. Keep the CPU `deep_cfr_gen` extension building and working as the reference
   oracle and fallback. CUDA is opt-in via `-DENABLE_CUDA=ON`.

---

## Phase 0 — GPU MLP forward pass (ALREADY CODED; you must build + validate)

### What exists

- `network_cuda.cuh` — opaque interface. No CUDA types leak to callers, so
  `binding_cuda.cpp` is compiled by the host compiler, not nvcc.
- `network_cuda.cu` — `cuda_mlp_create / forward / destroy`. Uses cuBLAS `Sgemm`
  with `CUBLAS_OP_T` on W and `CUBLAS_OP_N` on X so the column-major result
  `[out×batch]` shares memory layout with the desired row-major `[batch×out]` (no
  explicit transpose). A `k_bias_leaky_relu` kernel fuses bias + LeakyReLU on
  hidden layers; `k_bias` does bias-only on the linear output layer.
- `binding_cuda.cpp` — pybind11 module `deep_cfr_cuda` exposing
  `forward_batch(weights, inputs, max_batch=-1) -> [batch×N_ACTIONS]`.
- `CMakeLists.txt` — `option(ENABLE_CUDA OFF)`; when ON, `enable_language(CUDA)`,
  `find_package(CUDAToolkit)`, builds `deep_cfr_cuda`, links `CUDA::cublas`.
- `Makefile` — `build-cpp-cuda` target (configures into `deep_cfr_cpp/build_cuda`).
- `tests/test_cuda_forward.py` — parity vs a numpy reference (`test_parity` at
  batches 1/16/512/4096, tol 1e-3), determinism, single-vs-batch, zero-input.

### Build + run

```bash
make build-cpp-cuda
python -m pytest tests/test_cuda_forward.py -v
```

### Likely issues to fix (you have a GPU; the author did not)

- **`CMAKE_CUDA_ARCHITECTURES` not set** on newer CMake → configure error. Fix by
  passing e.g. `-DCMAKE_CUDA_ARCHITECTURES=native` (CMake ≥3.24) or the specific
  SM (e.g. `86` for A10/A40/3090, `89` for L4/4090, `90` for H100) in the
  `build-cpp-cuda` cmake invocation in the `Makefile`.
- **`cublas_v2.h` not found** → CUDA toolkit dev package missing, or
  `CUDAToolkit_ROOT` needs setting.
- **pybind11 + nvcc host-compiler friction** → `binding_cuda.cpp` is deliberately
  pure host code (opaque pointer) to avoid this. If nvcc still tries to touch
  pybind headers, confirm `binding_cuda.cpp` is in the host-compiled source list,
  not flagged as CUDA.
- **`max_diff` between 1e-3 and 1e-2** at large batch: acceptable float reduction
  reordering. If it *exceeds* 1e-2, the GEMM orientation is wrong — re-derive the
  `OP_T`/`OP_N`/`lda` choices against the comment block in `network_cuda.cu`.

**Definition of done for Phase 0:** all of `tests/test_cuda_forward.py` passes on
the GPU.

---

## Phase 1 — Recursion → explicit state machine (CPU only, no GPU needed)

**Why:** The current `mccfr()` is recursive and evaluates one forward pass inline
per node. To batch forwards across thousands of concurrent traversals (Phase 2),
the traversal must be reworked into an explicit, resumable state machine that can
**suspend at each point it needs a forward pass** and resume once the regret
vector is supplied externally.

This phase is pure CPU refactoring. It introduces **no GPU dependency** and can be
developed and tested without CUDA. The deliverable is a state-machine traversal
that is provably equivalent to the recursive `mccfr()`.

### The recursion to replicate (`mccfr.cpp::mccfr`)

```
mccfr(state, traverser):
  if terminal or depth>=MAX_DEPTH: return payoff(traverser)
  if chance: return mccfr(sample_chance(), traverser)        # only at root
  legal   = state.get_legal_actions()
  fvec    = build_feature_vector(state)
  regrets = forward_single(regret_net, fvec)                 # <-- THE BATCH POINT
  strategy = regret_match(regrets, legal)                    # max(r,0) norm; uniform fallback
  if current_player != traverser:                            # opponent node
      append StrategySample{fvec, strategy, weight=t}
      a = sample_from(strategy)                               # cumulative + uniform draw
      return mccfr(state.apply_action(a), traverser)
  else:                                                       # traverser node
      for a in legal: action_evs[a] = mccfr(state.apply_action(a), traverser)
      node_ev = sum(strategy[a] * action_evs[a])
      append RegretSample{fvec, regrets = action_evs[a]-node_ev for a in legal, weight=t}
      return node_ev
```

### State-machine design

Define a per-traversal **frame stack**. Each `Frame` holds a `GameState` (value
type — copyable) plus the bookkeeping to resume after a forward pass:

```cpp
enum class Phase { NEED_FORWARD, HAVE_REGRETS, EXPAND, RETURN_CHILD };

struct Frame {
    GameState   state;
    FeatureVec  fvec;              // filled when entering NEED_FORWARD
    LegalActions legal;
    float       regrets[N_ACTIONS];   // written by the (batched) forward
    float       strategy[N_ACTIONS];
    float       action_evs[N_ACTIONS];
    Phase       phase;
    int         child_idx;         // traverser: which legal action we're evaluating
    bool        is_traverser;      // current_player == traverser
    float       ret_ev;            // value this frame returns to its parent
};

struct Traversal {
    int                traverser;
    std::vector<Frame> stack;      // DFS stack; top = stack.back()
    bool               done = false;
    float              result;     // EV of the root once done
    // sample sinks — local, merged into the reservoir per traversal/game
    std::vector<RegretSample>*   regret_out;
    std::vector<StrategySample>* strategy_out;
    int iteration_t;
};
```

`advance(Traversal&)` runs the traversal forward **until it must block on a
forward pass** (top frame enters `NEED_FORWARD` with its `fvec` filled), or until
the traversal completes:

```
advance(tr):
  loop:
    if tr.stack.empty(): tr.done = true; return  // (root popped → tr.result set on pop)
    Frame& f = tr.stack.back();
    switch f.phase:

      NEED_FORWARD:
        // fvec already built when this frame was pushed.
        return;                 // BLOCK — caller will run forward, fill f.regrets,
                                //         set f.phase = HAVE_REGRETS, then re-advance.

      HAVE_REGRETS:
        f.strategy = regret_match(f.regrets, f.legal);
        if (!f.is_traverser) {                       // opponent node
            append StrategySample{f.fvec, f.strategy, t} -> tr.strategy_out;
            int a = sample_from(f.strategy, f.legal, rng);
            f.phase = RETURN_CHILD;                  // this node returns child's ev
            push_child(tr, f.state.apply_action(a)); // new frame, NEED_FORWARD
            continue;            // loop processes child; it will BLOCK on its forward
        } else {                                     // traverser node
            f.child_idx = 0;
            f.phase = EXPAND;
            push_child(tr, f.state.apply_action(f.legal[0]));
            continue;
        }

      EXPAND:                    // a child just returned; its ev is in <returned>
        f.action_evs[f.legal[f.child_idx]] = <ev returned by popped child>;
        f.child_idx++;
        if (f.child_idx < f.legal.n) {
            push_child(tr, f.state.apply_action(f.legal[f.child_idx]));
            continue;
        }
        // all children done → compute node_ev + regret sample, then pop
        node_ev = sum(f.strategy[a]*f.action_evs[a] for a in legal);
        append RegretSample{f.fvec, action_evs[a]-node_ev for a in legal, t};
        pop_returning(tr, node_ev);  // pop f, deliver node_ev to parent (see below)
        continue;

      RETURN_CHILD:              // opponent node: forward the child's ev upward
        pop_returning(tr, <ev returned by popped child>);
        continue;
```

`push_child(tr, childState)`:
- If `childState.is_terminal() || depth>=MAX_DEPTH`: don't push a frame; instead
  immediately deliver `childState.get_payoff(traverser)` to the current top frame
  (same mechanism as a pop return). This keeps terminal handling out of the stack.
- Else: build a `Frame`, set `is_traverser = (childState.current_player()==traverser)`,
  build its `fvec`, set `phase = NEED_FORWARD`, push.

`pop_returning(tr, ev)`:
- Pop the top frame. If the stack is now empty, `tr.result = ev; tr.done = true`.
- Otherwise stash `ev` so the new top frame's next `advance` step (its `EXPAND` or
  `RETURN_CHILD` case) consumes it. (Implement with a small `pending_child_ev`
  field on `Traversal`, or pass via a return slot.)

`regret_match` and `sample_from` are lifted verbatim from `mccfr.cpp` lines
191–223 — keep the uniform fallback and the cumulative-draw tie behaviour
identical.

### Single-traversal driver (Phase 1 deliverable)

Wire the state machine to the **existing CPU `forward_single`** so it has no GPU
dependency yet:

```cpp
float mccfr_sm(const GameState& dealt, int traverser, const MLP& net,
               std::vector<RegretSample>& rout,
               std::vector<StrategySample>& sout, int t, std::mt19937& rng) {
    Traversal tr = init(dealt, traverser, &rout, &sout, t);
    while (!tr.done) {
        advance(tr);
        if (tr.done) break;
        Frame& f = tr.stack.back();      // blocked on NEED_FORWARD
        auto r = forward_single(net, f.fvec);
        std::copy(r.begin(), r.end(), f.regrets);
        f.phase = Phase::HAVE_REGRETS;
    }
    return tr.result;
}
```

### Equivalence test (the gate for Phase 1)

Add a C++ test (or a pybind-exposed debug entry point) that, for a fixed RNG seed
and a fixed network, runs **both** `mccfr` and `mccfr_sm` on the same dealt state
and asserts:
- identical returned EV (to ~1e-5),
- identical multiset of `RegretSample`s and `StrategySample`s (same fvecs, same
  regret/strategy vectors, same count).

Because both use the same `forward_single` and the same RNG stream **in the same
order**, results should match almost exactly. Watch the RNG consumption order: the
recursive version draws for opponent sampling in DFS order; replicate that order
in the state machine (opponent draw happens in `HAVE_REGRETS` before pushing the
child — same as recursion).

> If exact RNG-order parity proves fiddly, fall back to a **statistical**
> equivalence test: run both over many deals with independent RNG and compare
> aggregate action frequencies / mean regrets within tolerance. Document whichever
> you chose.

**Definition of done for Phase 1:** `mccfr_sm` passes the equivalence test against
recursive `mccfr`; CPU `deep_cfr_gen` still builds and `make train-quick` still runs.

---

## Phase 2 — Batched orchestration: M concurrent traversals + GPU inference

**Why:** With the state machine, you can keep **M traversals in flight** (M ≈
8k–32k). Each tick, advance every live traversal until it blocks on a forward;
collect all blocked frames' `fvec`s into one `[M×308]` batch; run a single GPU
`forward_batch`; scatter the `[M×9]` regrets back; resume. The GPU does dense
GEMMs instead of millions of GEMVs.

### Driver loop

```
generate_cuda(n_games, iteration_t, regret_net_weights):
  upload weights to a persistent CudaMLP (created once per generate call)
  pool = []                      # up to M live Traversals
  pending_deals = a generator of (dealt_state, traverser) work items
                  # remember: deal once, enqueue 6 traversers per deal
  refill(pool, pending_deals)    # fill empty slots

  batch_fvec = float[M * 308]
  batch_out  = float[M * 9]

  while pool not all done:
    blocked = []
    for tr in pool:
      if tr.done:
        merge tr.regret_out / tr.strategy_out into reservoirs (add_batch)
        replace tr with a fresh traversal from pending_deals (or mark slot empty)
        continue
      advance(tr)
      if tr.done: (handle as above)
      else: blocked.append(tr)   # top frame is NEED_FORWARD

    if blocked empty: break
    # gather
    for i, tr in enumerate(blocked):
        memcpy(batch_fvec + i*308, tr.stack.back().fvec, 308 floats)
    # one GPU call
    cuda_mlp_forward(mlp, batch_fvec, batch_out, blocked.size())
    # scatter + resume
    for i, tr in enumerate(blocked):
        copy batch_out + i*9 -> tr.stack.back().regrets
        tr.stack.back().phase = HAVE_REGRETS
```

Notes:
- **Batch density:** until traversals start finishing, every live traversal blocks
  on exactly one forward per tick → batch ≈ M. Refill finished slots immediately to
  keep batches full. Near the tail, batches shrink — that's fine.
- **Feature building** (`build_feature_vector`) now runs on CPU for M frames per
  tick. Once the GEMM is on GPU this becomes the next hottest path. **Parallelise
  it with OpenMP** across the blocked list (`#pragma omp parallel for`). Defer
  full GPU feature offload to Phase 4.
- **Host memory:** each `Frame` holds a `GameState` (~2–3 KB). Stack depth is
  modest in practice (~20–40; `MAX_DEPTH=200` is a guard). At M=8k × depth 30 ×
  2.5 KB ≈ 600 MB. **M is the key tuning knob** — balance GPU batch efficiency vs
  host RAM. Make it configurable.
- **RNG:** each traversal gets its own `std::mt19937` seeded from a master seed
  stream (mirror `train.cpp::parallel_generate` seeding). Sampling stays on CPU.
- **Samples → reservoir:** collect per-traversal into local `std::vector`s
  (like `run_worker`'s `local_regret`/`local_strategy`), then `add_batch` into the
  existing shared `ReservoirBuffer`s. **No reservoir changes.**

### Persistent CudaMLP across ticks

Phase 0's `forward_batch` creates+destroys a `CudaMLP` per call — fine for tests,
wasteful per tick. Extend `network_cuda.cuh` with a reusable handle:
- Create the `CudaMLP` once per `generate_cuda` call (weights uploaded once).
- Reuse its device ping-pong buffers across all ticks (size them to M at create).
- Destroy at the end of the call.

Optionally keep the device input buffer resident and expose a variant of
`cuda_mlp_forward` that takes a pre-sized host batch and a row count `< max_batch`.

### Integration point

Add a CUDA `DeepCFRBuffers`-equivalent to `binding_cuda.cpp` mirroring
`binding.cpp`'s class — same method names (`generate_and_add`, `sample_*_into`,
`regret_ready`, ...). The `n_workers` argument is repurposed as **M** (concurrent
traversals) or ignored; document the choice. Then in `deep_cfr/train.py`, select
the module:

```python
try:
    import deep_cfr_cuda as _gen   # GPU path
except ImportError:
    import deep_cfr_gen as _gen    # CPU fallback
```

Keep the `DeepCFRBuffers` construction and all `sample_*_into` / prefetch logic in
`train.py` unchanged — only the module providing the class swaps.

### Validation (Phase 2 gate)

1. **End-to-end smoke:** `make train-quick` (5 iters × 200 games) on the CUDA path
   completes and produces a `gto_strategy.npz`.
2. **Statistical equivalence vs CPU:** run a few iterations on both backends with
   matched seeds; compare aggregate regret distributions and action frequencies
   (not per-sample — cuRAND/threading differences make exact parity impossible).
   Reuse/extend the harness from Phase 1.
3. **Reservoir sanity:** `regret_size()` / `strategy_size()` grow as expected.

**Definition of done for Phase 2:** CUDA `generate_and_add` populates the
reservoirs; `make train-quick` runs end-to-end on GPU; statistical equivalence
holds; CPU path still works.

---

## Phase 3 — Tuning, profiling, and production wiring

1. **Tune M.** Sweep M ∈ {4k, 8k, 16k, 32k}. Measure games/sec and peak host+device
   RAM. Pick the knee. Add it as a config constant (and/or CLI flag).
2. **Profile the tick.** Use Nsight Systems (`nsys profile python -m deep_cfr.train
   --quick`) to confirm the GPU GEMM is now a large fraction of time and identify
   the next bottleneck (expected: CPU feature building or the gather/scatter
   memcpys).
3. **Overlap.** If gather/scatter dominates, use pinned host memory
   (`cudaHostAlloc`) for `batch_fvec`/`batch_out` and a CUDA stream so the H2D copy
   of the next batch overlaps compute. (cuBLAS on a non-default stream.)
4. **Compare wall-clock** end-to-end: full or near-full iteration on CPU vs CUDA.
   Record the speedup. Target 10–30× on data gen.
5. **Wire the Makefile/train.py autoselect cleanly** and document GPU build in
   `CLAUDE.md` (add a `build-cpp-cuda` mention near `build-cpp`). Update
   `install-vm` if a CUDA build should be the default on GPU VMs.
6. **Determinism knob.** Provide a fixed-seed mode for reproducible debugging runs.

**Definition of done for Phase 3:** documented speedup number, chosen M, profile
showing GEMM-dominated time, clean autoselect, updated docs.

---

## Phase 4 (optional) — On-GPU features + frontier wavefront

The ceiling after Phase 3 is the CPU work per tick (feature building + engine
transitions + gather/scatter + PCIe round-trips). Eliminate it by moving state
on-device.

1. **Port `build_feature_vector` to a `__device__` kernel.** Operate directly on a
   device-resident `StateDict`-equivalent (fixed-size; the engine is already
   fixed-array friendly, but `StateDict::action_log` is a `std::vector` — replace
   with a fixed array, like `PokerEngine::action_log_[MAX_HAND_ACTIONS]`). **Add a
   parity test** against the CPU `build_feature_vector` over many random states —
   the 308-layout invariant is sacred.
2. **Port the engine transitions** (`apply_action`, `get_legal_actions`,
   `abstract_to_raw`) to device code. The engine is value-type and uses no STL
   containers except that one vector — feasible but the biggest lift.
3. **Frontier/wavefront traversal.** Keep all in-flight nodes in device memory;
   advance the frontier with a kernel; inference via cuBLAS batched GEMM; per-
   traversal DFS stacks in global memory; RNG via cuRAND; samples appended to a
   device buffer then reservoir-sampled.
4. **Validate** against Phases 1–3 statistically; keep CPU as oracle.

This is a research-grade lift. Only pursue it if Phase 3 profiling shows CPU/PCIe
as the dominant cost and the speedup is worth the complexity and validation burden.

---

## Quick command reference

```bash
# Setup
make install-vm && source .venv/bin/activate

# CPU reference build (always keep working)
make build-cpp
python -m pytest tests/ -q

# CUDA build + Phase 0 parity
make build-cpp-cuda
python -m pytest tests/test_cuda_forward.py -v

# Smoke-test training (CPU or CUDA depending on which extension is importable)
make train-quick

# Profile
nsys profile -o cuda_gen python -m deep_cfr.train --quick
```

## Definition of done (whole project)

- Phases 0–3 complete; Phase 4 optional.
- CUDA path produces a `gto_strategy.npz` indistinguishable in quality from the CPU
  path (statistical equivalence + comparable training loss curves).
- Documented data-gen speedup.
- CPU `deep_cfr_gen` still builds and runs as the reference/fallback.
- The 308-dim feature layout and `DeepCFRBuffers` API are unchanged.
```
