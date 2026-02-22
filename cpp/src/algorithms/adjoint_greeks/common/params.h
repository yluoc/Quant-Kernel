#ifndef QK_AGM_PARAMS_H
#define QK_AGM_PARAMS_H

#include <cstdint>

namespace qk::agm {

struct PathwiseDerivativeParams {
    int32_t paths;
    uint64_t seed;
};

struct LikelihoodRatioParams {
    int32_t paths;
    uint64_t seed;
    double weight_clip;
};

// Runtime configuration for the BSM hand-adjoint delta estimator.
//
// NOTE: Despite the "Aad" prefix, the current implementation is a
// hand-differentiated BSM closed-form delta with Tikhonov regularization,
// not a tape-based AAD engine.  Field names are preserved for ABI stability.
struct AadParams {
    // Controls regularization decay: lambda_eff = regularization / sqrt(tape_steps).
    // Higher values weaken regularization.  Must be >= 4.
    int32_t tape_steps;
    // Tikhonov regularization strength toward the ATM delta prior.
    // Set to 0.0 to disable regularization.  Must be >= 0.
    double regularization;
};

} // namespace qk::agm

#endif /* QK_AGM_PARAMS_H */
