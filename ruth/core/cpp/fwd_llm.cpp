#include "fwd_llm.h"
#include <cmath>

namespace ruth {

// --- Internal PRNG Implementation (xoshiro256**) ---

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

struct Xoshiro256StarStarState {
  uint64_t s[4];
};

static uint64_t next(Xoshiro256StarStarState &state) {
  const uint64_t result = rotl(state.s[1] * 5, 7) * 9;

  const uint64_t t = state.s[1] << 17;

  state.s[2] ^= state.s[0];
  state.s[3] ^= state.s[1];
  state.s[1] ^= state.s[2];
  state.s[0] ^= state.s[3];

  state.s[2] ^= t;

  state.s[3] = rotl(state.s[3], 45);

  return result;
}

// SplitMix64 for seeding
static uint64_t splitmix64(uint64_t &x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

// --- RuthTrainer Implementation ---

void RuthTrainer::generate_perturbation(uint64_t seed,
                                        std::vector<float> &buffer) {
  // 1. Initialize PRNG state
  Xoshiro256StarStarState state;
  uint64_t sm64_state = seed;
  state.s[0] = splitmix64(sm64_state);
  state.s[1] = splitmix64(sm64_state);
  state.s[2] = splitmix64(sm64_state);
  state.s[3] = splitmix64(sm64_state);

  size_t n = buffer.size();
  size_t i = 0;

  // 2. Generate Standard Normal Noise (Box-Muller)
  // We generate pairs of normal variables from pairs of uniform variables.

  // Constants for conversion
  // 2^-53
  const double scale = 1.1102230246251565e-16;

  while (i + 1 < n) {
    // Generate two uniform doubles in [0, 1)
    // xoshiro returns uint64. We take top 53 bits.
    uint64_t u1_raw = next(state) >> 11;
    uint64_t u2_raw = next(state) >> 11;

    double u1 = u1_raw * scale;
    double u2 = u2_raw * scale;

    // Avoid log(0)
    if (u1 <= 0.0)
      u1 = 1.0e-10;

    // Box-Muller
    double mag = std::sqrt(-2.0 * std::log(u1));
    double z0 = mag * std::cos(2.0 * M_PI * u2);
    double z1 = mag * std::sin(2.0 * M_PI * u2);

    buffer[i] = static_cast<float>(z0);
    buffer[i + 1] = static_cast<float>(z1);
    i += 2;
  }

  // Handle odd last element
  if (i < n) {
    uint64_t u1_raw = next(state) >> 11;
    uint64_t u2_raw = next(state) >> 11;
    double u1 = u1_raw * scale;
    double u2 = u2_raw * scale;
    if (u1 <= 0.0)
      u1 = 1.0e-10;
    double mag = std::sqrt(-2.0 * std::log(u1));
    double z0 = mag * std::cos(2.0 * M_PI * u2);
    buffer[i] = static_cast<float>(z0);
  }
}

float RuthTrainer::compute_update(float loss_plus, float loss_minus,
                                  float epsilon, float baseline, float cap) {
  // rho = (L+ - L-) / (2 * epsilon)
  float rho = (loss_plus - loss_minus) / (2.0f * epsilon);

  // Subtract baseline
  float rho_adj = rho - baseline;

  // Clip
  return std::clamp(rho_adj, -cap, cap);
}

} // namespace ruth
