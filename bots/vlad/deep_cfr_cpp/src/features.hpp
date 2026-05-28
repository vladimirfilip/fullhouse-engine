#pragma once
#include <array>
#include "config.hpp"
#include "engine.hpp"

using FeatureVec = std::array<float, INPUT_DIM>;

// Build a 308-float feature vector from a StateDict.
// Layout must match bot.py _build_feature_vector() exactly.
FeatureVec build_feature_vector(const StateDict& s);
