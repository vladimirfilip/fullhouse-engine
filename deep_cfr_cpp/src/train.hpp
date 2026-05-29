#pragma once
#include "config.hpp"
#include "network.hpp"
#include "reservoir.hpp"
#include "mccfr.hpp"

// Parallel data generation: run n_games MCCFR traversals across n_workers threads,
// merge results into shared reservoir buffers. The full training loop lives in
// Python (bots/vlad/deep_cfr/train.py) — this is the only entry point the
// pybind layer needs.
void parallel_generate(int n_games,
                       const MLP& regret_net,
                       ReservoirBuffer<RegretSample>&   regret_buf,
                       ReservoirBuffer<StrategySample>& strategy_buf,
                       int iteration_t,
                       int n_workers);
