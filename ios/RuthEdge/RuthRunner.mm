#import "RuthRunner.h"
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace executorch::extension;
using namespace executorch::runtime;

@interface RuthRunner () {
  std::unique_ptr<Module> _module;
  std::mt19937 _rng;
}
@end

@implementation RuthRunner

- (instancetype)initWithModelPath:(NSString *)modelPath {
  self = [super init];
  if (self) {
    // Initialize PRNG
    std::random_device rd;
    _rng = std::mt19937(rd());

    // Load Module
    try {
      _module = std::make_unique<Module>(modelPath.UTF8String,
                                         Module::LoadMode::MmapUseMlock);
      NSLog(@"[RuthRunner] Model loaded successfully from %@", modelPath);
    } catch (const std::exception &e) {
      NSLog(@"[RuthRunner] Failed to load model: %s", e.what());
      return nil;
    }
  }
  return self;
}

- (float)trainStepWithInput:(NSString *)input {
  if (!_module) {
    NSLog(@"[RuthRunner] Module not initialized.");
    return 0.0f;
  }

  @autoreleasepool {
    // 1. Tokenize (Simple ASCII for demo)
    // In production, use a real tokenizer (BPE/SentencePiece)
    std::vector<int64_t> token_ids;
    const char *utf8Input = [input UTF8String];
    size_t len = strlen(utf8Input);
    for (size_t i = 0; i < len; ++i) {
      token_ids.push_back((int64_t)utf8Input[i]);
    }

    // Ensure we have enough tokens for the model (e.g. pad or truncate)
    // For Llama-1B export, we used seq_len=128. Let's truncate/pad.
    size_t seq_len = 128;
    if (token_ids.size() > seq_len) {
      token_ids.resize(seq_len);
    } else {
      while (token_ids.size() < seq_len) {
        token_ids.push_back(0); // Pad with 0
      }
    }

    // 2. Create Tensors
    // Input IDs: [1, seq_len]
    std::vector<int64_t> input_shape = {1, (int64_t)seq_len};
    auto input_tensor = from_blob(token_ids.data(), input_shape);

    // Labels: Same as input for this demo
    auto labels_tensor = from_blob(token_ids.data(), input_shape);

    // Seed: [1]
    int64_t seed_val = _rng();
    std::vector<int64_t> seed_data = {seed_val};
    std::vector<int64_t> seed_shape = {1};
    auto seed_tensor = from_blob(seed_data.data(), seed_shape);

    // Epsilon: Scalar (float)
    // Note: ExecuTorch might expect a tensor for float inputs if the graph was
    // exported that way. The export script used `epsilon = 0.1` (float) in
    // example_inputs. If exported as a Tensor, we need a Tensor. If as a float,
    // we pass EValue(double). Let's assume it was exported as a Tensor for
    // consistency with typical PyTorch export behavior for "inputs". But
    // `torch.export` usually handles primitives. Let's try passing as
    // EValue(double) first, or check the export script. Export script:
    // `example_inputs = (input_ids, labels, seed, epsilon)` where epsilon is
    // float. So it's likely a scalar input.

    // 3. Run Inference 0 (Baseline, epsilon=0)
    float loss_0 = 0.0f;
    try {
      // Inputs: input_ids, labels, seed, epsilon
      std::vector<EValue> inputs_0;
      inputs_0.emplace_back(input_tensor);
      inputs_0.emplace_back(labels_tensor);
      inputs_0.emplace_back(seed_tensor);
      inputs_0.emplace_back(0.0); // epsilon = 0

      auto result_0 = _module->execute("forward_perturb", inputs_0);

      if (result_0.ok()) {
        // Extract scalar loss
        // Result should be a Tensor (scalar)
        auto outputs = result_0.get();
        if (outputs.size() > 0 && outputs[0].isTensor()) {
          loss_0 = outputs[0].toTensor().const_data_ptr<float>()[0];
        }
      } else {
        NSLog(@"[RuthRunner] Forward 0 failed: %d", (int)result_0.error());
        return 0.0f;
      }
    } catch (const std::exception &e) {
      NSLog(@"[RuthRunner] Exception in Forward 0: %s", e.what());
      return 0.0f;
    }

    // 4. Run Inference P (Perturbed, epsilon=0.1)
    float epsilon = 0.1f;
    float loss_p = 0.0f;
    try {
      std::vector<EValue> inputs_p;
      inputs_p.emplace_back(input_tensor);
      inputs_p.emplace_back(labels_tensor);
      inputs_p.emplace_back(seed_tensor);
      inputs_p.emplace_back((double)epsilon);

      auto result_p = _module->execute("forward_perturb", inputs_p);

      if (result_p.ok()) {
        auto outputs = result_p.get();
        if (outputs.size() > 0 && outputs[0].isTensor()) {
          loss_p = outputs[0].toTensor().const_data_ptr<float>()[0];
        }
      } else {
        NSLog(@"[RuthRunner] Forward P failed: %d", (int)result_p.error());
        return 0.0f;
      }
    } catch (const std::exception &e) {
      NSLog(@"[RuthRunner] Exception in Forward P: %s", e.what());
      return 0.0f;
    }

    // 5. Calculate Gradient Estimate
    float estimated_grad = (loss_p - loss_0) / epsilon;

    NSLog(@"[RuthRunner] Step Complete. Loss0: %.4f, LossP: %.4f, Grad: %.4f",
          loss_0, loss_p, estimated_grad);

    return estimated_grad;
  }
}

@end
