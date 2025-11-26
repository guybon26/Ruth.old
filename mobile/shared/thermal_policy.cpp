#include "thermal_policy.h"
#include <algorithm>

namespace ruth {

ThermalPolicy::ThermalPolicy()
    : state_(IDLE), failures_(0), next_allowed_run_(0) {}

bool ThermalPolicy::should_run(float temp_c, float battery_percent,
                               bool is_charging, uint64_t now_ts) {
  // 1. Battery Check (Hard Stop)
  if (battery_percent < BATTERY_MIN_THRESHOLD) {
    return false;
  }

  // 2. Backoff Check
  if (now_ts < next_allowed_run_) {
    return false;
  }

  // 3. Hysteresis Logic
  if (state_ == IDLE) {
    if (temp_c > TEMP_HIGH_THRESHOLD) {
      state_ = COOLDOWN;
      return false;
    }
    return true;
  } else { // COOLDOWN
    if (temp_c < TEMP_LOW_THRESHOLD) {
      state_ = IDLE;
      return true;
    }
    return false;
  }
}

void ThermalPolicy::report_success() {
  failures_ = 0;
  next_allowed_run_ = 0;
}

void ThermalPolicy::report_failure(uint64_t now_ts) {
  failures_++;
  // Exponential backoff: 10min * 2^failures
  // Cap at some reasonable limit if needed, but for now standard exp backoff
  // failures starts at 1 after increment.
  // 1 failure -> 10 * 2 = 20 mins? Or 10 * 2^(failures-1)?
  // Prompt says: 10min * 2^failures.
  // So 1 failure -> 20 mins. 2 failures -> 40 mins.

  // Let's stick to the prompt exactly: 10min * 2^failures
  uint64_t backoff = BASE_BACKOFF_SECONDS * (1ULL << failures_);
  next_allowed_run_ = now_ts + backoff;
}

void ThermalPolicy::reset() {
  state_ = IDLE;
  failures_ = 0;
  next_allowed_run_ = 0;
}

} // namespace ruth
