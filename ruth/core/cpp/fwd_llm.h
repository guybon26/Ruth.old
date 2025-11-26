#ifndef RUTH_FWD_LLM_H
#define RUTH_FWD_LLM_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace ruth {

class RuthTrainer {
public:
    /**
     * Generates a perturbation vector using a deterministic PRNG seeded with `seed`.
     * Fills the provided buffer with standard normal random numbers.
     * 
     * @param seed The 64-bit seed for the PRNG.
     * @param buffer The buffer to fill with noise. Must be pre-allocated.
     */
    static void generate_perturbation(uint64_t seed, std::vector<float>& buffer);

    /**
     * Computes the scalar update rho for Forward Gradient.
     * rho = (loss_plus - loss_minus) / (2 * epsilon) - baseline
     * The result is clipped to [-cap, cap].
     * 
     * @param loss_plus Loss with +epsilon perturbation.
     * @param loss_minus Loss with -epsilon perturbation.
     * @param epsilon The perturbation magnitude.
     * @param baseline The control variate baseline to subtract.
     * @param cap The maximum absolute value for the update.
     * @return The computed scalar update.
     */
    static float compute_update(float loss_plus, float loss_minus, float epsilon, float baseline, float cap);
};

} // namespace ruth

#endif // RUTH_FWD_LLM_H
