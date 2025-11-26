#include "thermal_policy.h"
#include <cassert>
#include <iostream>

using namespace ruth;

void test_battery_cutoff() {
  std::cout << "Testing Battery Cutoff..." << std::endl;
  ThermalPolicy policy;
  uint64_t now = 1000;

  // Battery < 20% -> Should fail
  assert(policy.should_run(30.0f, 19.0f, false, now) == false);

  // Battery >= 20% -> Should pass (if temp ok)
  assert(policy.should_run(30.0f, 20.0f, false, now) == true);

  std::cout << "Passed." << std::endl;
}

void test_hysteresis() {
  std::cout << "Testing Hysteresis..." << std::endl;
  ThermalPolicy policy;
  uint64_t now = 1000;

  // 1. Start IDLE, Temp OK
  assert(policy.should_run(37.0f, 50.0f, false, now) == true);
  assert(policy.get_state() == ThermalPolicy::IDLE);

  // 2. Temp goes High (> 38) -> Switch to COOLDOWN
  assert(policy.should_run(38.1f, 50.0f, false, now) == false);
  assert(policy.get_state() == ThermalPolicy::COOLDOWN);

  // 3. Temp drops but still > 35 -> Stay in COOLDOWN
  assert(policy.should_run(36.0f, 50.0f, false, now) == false);
  assert(policy.get_state() == ThermalPolicy::COOLDOWN);

  // 4. Temp drops < 35 -> Switch to IDLE
  assert(policy.should_run(34.9f, 50.0f, false, now) == true);
  assert(policy.get_state() == ThermalPolicy::IDLE);

  std::cout << "Passed." << std::endl;
}

void test_exponential_backoff() {
  std::cout << "Testing Exponential Backoff..." << std::endl;
  ThermalPolicy policy;
  uint64_t now = 1000;

  // 1. Report Failure
  policy.report_failure(now);
  assert(policy.get_failures() == 1);

  // Backoff should be 10min * 2^1 = 20 mins = 1200 seconds
  uint64_t expected_next = now + 1200;
  assert(policy.get_next_allowed_run() == expected_next);

  // 2. Try running before backoff -> Should fail
  assert(policy.should_run(30.0f, 50.0f, false, now + 100) == false);

  // 3. Try running after backoff -> Should pass
  assert(policy.should_run(30.0f, 50.0f, false, expected_next + 1) == true);

  // 4. Report another failure
  now = expected_next + 100;
  policy.report_failure(now);
  assert(policy.get_failures() == 2);

  // Backoff should be 10min * 2^2 = 40 mins = 2400 seconds
  expected_next = now + 2400;
  assert(policy.get_next_allowed_run() == expected_next);

  // 5. Report Success -> Reset
  policy.report_success();
  assert(policy.get_failures() == 0);
  assert(policy.get_next_allowed_run() == 0);

  std::cout << "Passed." << std::endl;
}

int main() {
  test_battery_cutoff();
  test_hysteresis();
  test_exponential_backoff();
  std::cout << "All Thermal Policy tests passed!" << std::endl;
  return 0;
}
