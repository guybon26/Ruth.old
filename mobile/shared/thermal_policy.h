#ifndef RUTH_THERMAL_POLICY_H
#define RUTH_THERMAL_POLICY_H

#include <cstdint>

namespace ruth {

class ThermalPolicy {
public:
  enum State { IDLE, COOLDOWN };

  ThermalPolicy();

  /**
   * Checks if the training process should run based on thermal and battery
   * conditions.
   *
   * @param temp_c Current device temperature in Celsius.
   * @param battery_percent Current battery level (0.0 to 100.0).
   * @param is_charging Whether the device is currently charging.
   * @param now_ts Current timestamp in seconds.
   * @return True if allowed to run, False otherwise.
   */
  bool should_run(float temp_c, float battery_percent, bool is_charging,
                  uint64_t now_ts);

  /**
   * Reports a successful run. Resets failure count and backoff.
   */
  void report_success();

  /**
   * Reports a failure (e.g., thermal event during run). Triggers exponential
   * backoff.
   *
   * @param now_ts Current timestamp in seconds.
   */
  void report_failure(uint64_t now_ts);

  /**
   * Resets the policy state.
   */
  void reset();

  // Getters for testing
  State get_state() const { return state_; }
  uint64_t get_next_allowed_run() const { return next_allowed_run_; }
  int get_failures() const { return failures_; }

private:
  State state_;
  int failures_;
  uint64_t next_allowed_run_;

  // Constants
  static constexpr float TEMP_HIGH_THRESHOLD = 38.0f;
  static constexpr float TEMP_LOW_THRESHOLD = 35.0f;
  static constexpr float BATTERY_MIN_THRESHOLD = 20.0f;
  static constexpr uint64_t BASE_BACKOFF_SECONDS = 600; // 10 minutes
};

} // namespace ruth

#endif // RUTH_THERMAL_POLICY_H
