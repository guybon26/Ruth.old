#include "fwd_llm.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace ruth;

void test_reproducibility() {
  std::cout << "Testing Reproducibility..." << std::endl;
  std::vector<float> v1(100);
  std::vector<float> v2(100);

  RuthTrainer::generate_perturbation(42, v1);
  RuthTrainer::generate_perturbation(42, v2);

  for (size_t i = 0; i < v1.size(); ++i) {
    assert(v1[i] == v2[i]);
  }

  std::vector<float> v3(100);
  RuthTrainer::generate_perturbation(123, v3);
  assert(v1[0] != v3[0]); // Should be different
  std::cout << "Passed." << std::endl;
}

void test_distribution() {
  std::cout << "Testing Distribution..." << std::endl;
  size_t n = 10000;
  std::vector<float> v(n);
  RuthTrainer::generate_perturbation(1, v);

  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / n;

  double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / n - mean * mean);

  std::cout << "Mean: " << mean << " (Expected ~0)" << std::endl;
  std::cout << "Std: " << stdev << " (Expected ~1)" << std::endl;

  assert(std::abs(mean) < 0.05);
  assert(std::abs(stdev - 1.0) < 0.05);
  std::cout << "Passed." << std::endl;
}

void test_compute_update() {
  std::cout << "Testing Compute Update..." << std::endl;
  float l_plus = 10.0f;
  float l_minus = 8.0f;
  float eps = 0.1f;
  float baseline = 0.0f;
  float cap = 100.0f;

  // rho = (10 - 8) / 0.2 = 10.0
  float rho = RuthTrainer::compute_update(l_plus, l_minus, eps, baseline, cap);
  assert(std::abs(rho - 10.0f) < 1e-5);

  // Test Baseline
  baseline = 2.0f;
  // rho = 10.0 - 2.0 = 8.0
  rho = RuthTrainer::compute_update(l_plus, l_minus, eps, baseline, cap);
  assert(std::abs(rho - 8.0f) < 1e-5);

  // Test Clipping
  cap = 5.0f;
  // rho = 8.0, clipped to 5.0
  rho = RuthTrainer::compute_update(l_plus, l_minus, eps, baseline, cap);
  assert(std::abs(rho - 5.0f) < 1e-5);

  std::cout << "Passed." << std::endl;
}

int main() {
  test_reproducibility();
  test_distribution();
  test_compute_update();
  std::cout << "All C++ tests passed!" << std::endl;
  return 0;
}
