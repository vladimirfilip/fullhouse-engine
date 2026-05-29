#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <mutex>

// Reservoir sampling buffer using Vitter's Algorithm R.
// Exactly mirrors memory.py ReservoirBuffer.
// add() is thread-safe: multiple worker threads may call it concurrently.
// sample*() and clear() must only be called single-threaded (training phase).
template<typename T>
class ReservoirBuffer {
public:
    explicit ReservoirBuffer(int capacity) : capacity_(capacity), n_seen_(0) {
        // Reserve the full capacity upfront so std::vector never reallocates.
        // Without this, the geometric doubling strategy of libstdc++ would
        // transiently allocate ~2× capacity (e.g. 8 M slots when cap=4 M),
        // causing ~20 GB peak RSS on Linux during the first fill.
        data_.reserve(capacity);
    }

    void add(const T& item) {
        std::lock_guard<std::mutex> lk(mtx_);
        add_locked(item);
    }

    // Add many items under a single lock. Eliminates the per-sample mutex
    // contention that dominates the merge phase in train.cpp when running with
    // 8+ workers and tens of millions of samples per iteration.
    void add_batch(const std::vector<T>& items) {
        if (items.empty()) return;
        std::lock_guard<std::mutex> lk(mtx_);
        for (const auto& it : items) add_locked(it);
    }

    // Sample up to n items uniformly at random without replacement.
    // O(k) when k << n (hash-set rejection); O(n) partial Fisher-Yates when k >= n/4.
    std::vector<T> sample(int n) {
        int sz = (int)data_.size();
        int k  = std::min(n, sz);
        if (k >= sz) return data_;

        std::vector<T> result;
        result.reserve(k);

        if (k * 4 <= sz) {
            // k much smaller than sz: rejection sampling with hash set, O(k) expected
            std::unordered_set<int> chosen;
            chosen.reserve(k * 2);
            std::uniform_int_distribution<int> dist(0, sz - 1);
            while ((int)result.size() < k) {
                int r = dist(rng_);
                if (chosen.insert(r).second)
                    result.push_back(data_[r]);
            }
        } else {
            // k close to sz: partial Fisher-Yates on index array, O(k)
            std::vector<int> idx(sz);
            std::iota(idx.begin(), idx.end(), 0);
            for (int i = 0; i < k; i++) {
                int j = std::uniform_int_distribution<int>(i, sz - 1)(rng_);
                std::swap(idx[i], idx[j]);
                result.push_back(data_[idx[i]]);
            }
        }
        return result;
    }

    bool is_ready(int min_size) const { return (int)data_.size() >= min_size; }
    int  size() const { return (int)data_.size(); }
    void clear() { data_.clear(); n_seen_ = 0; }

    void seed(uint64_t s) { rng_.seed(s); }

private:
    // Caller must hold mtx_. Same logic as the original add() — extracted so
    // add_batch() can reuse it under a single lock.
    void add_locked(const T& item) {
        n_seen_++;
        if ((int)data_.size() < capacity_) {
            data_.push_back(item);
        } else {
            int idx = std::uniform_int_distribution<int>(0, n_seen_ - 1)(rng_);
            if (idx < capacity_) data_[idx] = item;
        }
    }

    int capacity_;
    int n_seen_;
    std::vector<T> data_;
    std::mt19937 rng_{std::random_device{}()};
    std::mutex mtx_;
};
