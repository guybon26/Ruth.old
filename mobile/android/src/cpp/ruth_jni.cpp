#include "../../../ruth/core/cpp/fwd_llm.h" // Relative path to header
#include <cmath>
#include <jni.h>
#include <numeric>
#include <vector>

// Mock ExecuTorch Forward Pass
// In reality, this would call the ExecuTorch runtime
float mock_executorch_forward(const std::vector<float> &input,
                              const std::vector<float> &weights_perturbation) {
  // Simulate loss calculation: MSE(input) + perturbation_effect
  // This is just a dummy function to simulate work
  float loss = 0.0f;
  for (float val : input) {
    loss += val * val;
  }

  // Simulate perturbation effect
  if (!weights_perturbation.empty()) {
    float perturbation_mag = 0.0f;
    for (float p : weights_perturbation) {
      perturbation_mag += p * p;
    }
    loss += std::sqrt(perturbation_mag) * 0.01f;
  }

  return loss;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_ruth_Native_step(JNIEnv *env, jobject thiz, jfloatArray input,
                          jfloatArray target, jlong seed, jfloat epsilon) {

  // 1. Convert JNI arrays to C++
  jfloat *input_ptr = env->GetFloatArrayElements(input, nullptr);
  jfloat *target_ptr = env->GetFloatArrayElements(target, nullptr);
  jsize input_len = env->GetArrayLength(input);

  std::vector<float> input_vec(input_ptr, input_ptr + input_len);

  // 2. Mock Inference (Loss 0)
  // We pass empty perturbation for base loss
  float loss0 = mock_executorch_forward(input_vec, {});

  // 3. Generate Perturbation
  // Assume model size is same as input size for this mock
  std::vector<float> perturbation(input_len);
  ruth::RuthTrainer::generate_perturbation(static_cast<uint64_t>(seed),
                                           perturbation);

  // 4. Mock Perturbed Inference
  // Loss Plus
  float loss_plus = mock_executorch_forward(input_vec, perturbation);

  // Loss Minus (Simulated by subtracting perturbation effect in mock)
  // In real ExecuTorch, we'd apply -epsilon * perturbation
  float loss_minus = loss0 - (loss_plus - loss0);

  // 5. Compute Update
  float baseline = 0.0f; // Could be passed in or managed in C++ state
  float cap = 5.0f;
  float scalar = ruth::RuthTrainer::compute_update(loss_plus, loss_minus,
                                                   epsilon, baseline, cap);

  // 6. Release JNI arrays
  env->ReleaseFloatArrayElements(
      input, input_ptr, JNI_ABORT); // JNI_ABORT because we didn't modify input
  env->ReleaseFloatArrayElements(target, target_ptr, JNI_ABORT);

  // 7. Return Result {scalar, loss}
  jfloatArray result = env->NewFloatArray(2);
  if (result == nullptr) {
    return nullptr; // Out of memory
  }

  jfloat result_values[2];
  result_values[0] = scalar;
  result_values[1] = loss0;

  env->SetFloatArrayRegion(result, 0, 2, result_values);

  return result;
}
