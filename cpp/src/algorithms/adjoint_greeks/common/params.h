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

struct AadParams {
    int32_t tape_steps;
    double regularization;
};

} // namespace qk::agm

#endif /* QK_AGM_PARAMS_H */
