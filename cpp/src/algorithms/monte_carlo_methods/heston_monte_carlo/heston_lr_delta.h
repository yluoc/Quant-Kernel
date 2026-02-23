#ifndef QK_MCM_HESTON_LR_DELTA_H
#define QK_MCM_HESTON_LR_DELTA_H

#include <cstdint>

namespace qk::mcm {

double heston_likelihood_ratio_delta(
    double spot, double strike, double t,
    double r, double q,
    double v0, double kappa, double theta,
    double sigma, double rho,
    int32_t option_type,
    int32_t paths, int32_t steps, uint64_t seed,
    double weight_clip);

} // namespace qk::mcm

#endif /* QK_MCM_HESTON_LR_DELTA_H */
