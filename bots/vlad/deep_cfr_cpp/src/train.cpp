#define _CRT_SECURE_NO_WARNINGS
#include "train.hpp"
#include <thread>
#include <vector>
#include <random>

// ── Worker ────────────────────────────────────────────────────────────────────

// Player-count distribution matching train.py (weights [5,10,15,20,50])
static int sample_n_players(std::mt19937& rng) {
    static const int counts[]  = {2,3,4,5,6};
    static const int weights[] = {5,10,15,20,50};
    static const int total     = 5+10+15+20+50; // 100
    int r = std::uniform_int_distribution<int>(0, total - 1)(rng);
    int cumul = 0;
    for (int i = 0; i < 5; i++) {
        cumul += weights[i];
        if (r < cumul) return counts[i];
    }
    return 6;
}

// Workers write directly into shared reservoir buffers (thread-safe via ReservoirBuffer::add).
// Local vectors are per-game and cleared after each game, keeping peak memory proportional
// to one game's worth of samples (a few MB) rather than all n_games (potentially GBs).
static void run_worker(int n_games, uint32_t seed, int iteration_t, MLP regret_net,
                       ReservoirBuffer<RegretSample>&   regret_buf,
                       ReservoirBuffer<StrategySample>& strategy_buf) {
    std::mt19937 rng(seed);
    std::vector<RegretSample>   local_regret;
    std::vector<StrategySample> local_strategy;
    local_regret.reserve(2000);
    local_strategy.reserve(8000);

    for (int g = 0; g < n_games; g++) {
        int n_p = sample_n_players(rng);
        // Deal once; traverse for every seat (same optimization as train.py)
        GameState initial(n_p);
        GameState dealt = initial.sample_chance_event(rng);
        for (int traverser = 0; traverser < n_p; traverser++) {
            mccfr(dealt, traverser, regret_net,
                  local_regret, local_strategy,
                  iteration_t, 0, rng);
        }
        // Merge this game's samples into the shared reservoir — one lock per
        // game per buffer, instead of one lock per sample (~100x fewer acquires).
        regret_buf.add_batch(local_regret);
        strategy_buf.add_batch(local_strategy);
        local_regret.clear();
        local_strategy.clear();
    }
}

// ── Parallel data generation ──────────────────────────────────────────────────

void parallel_generate(int n_games,
                       const MLP& regret_net,
                       ReservoirBuffer<RegretSample>&   regret_buf,
                       ReservoirBuffer<StrategySample>& strategy_buf,
                       int iteration_t,
                       int n_workers) {
    // Distribute games evenly across workers
    std::vector<int> game_counts(n_workers);
    int base  = n_games / n_workers;
    int extra = n_games % n_workers;
    for (int i = 0; i < n_workers; i++)
        game_counts[i] = base + (i < extra ? 1 : 0);

    // Generate seeds
    std::mt19937 seed_rng(std::random_device{}());
    std::vector<uint32_t> seeds(n_workers);
    for (auto& s : seeds) s = seed_rng();

    if (n_workers == 1) {
        run_worker(game_counts[0], seeds[0], iteration_t, regret_net,
                   regret_buf, strategy_buf);
        return;
    }

    // Multi-threaded: workers write directly into shared buffers via thread-safe add().
    // No intermediate WorkerResult vectors — peak memory is O(one game) per worker.
    std::vector<std::thread> threads;
    threads.reserve(n_workers);
    for (int i = 0; i < n_workers; i++) {
        threads.emplace_back([&, i]() {
            run_worker(game_counts[i], seeds[i], iteration_t, regret_net,
                       regret_buf, strategy_buf);
        });
    }
    for (auto& t : threads) t.join();
}
