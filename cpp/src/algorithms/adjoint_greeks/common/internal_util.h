#ifndef QK_AGM_INTERNAL_UTIL_H
#define QK_AGM_INTERNAL_UTIL_H

#include "common/option_util.h"

#include <algorithm>
#include <cmath>

namespace qk::agm::detail {

constexpr double kEps = 1e-12;

inline double nan_value() { return qk::nan_value(); }

inline bool valid_option_type(int32_t option_type) { return qk::valid_option_type(option_type); }

inline bool valid_common_inputs(double spot, double strike, double t, double vol,
                                double r, double q, int32_t option_type) {
    return qk::valid_common_inputs(spot, strike, t, vol, r, q, option_type);
}

inline double deterministic_delta(double spot, double strike, double t, double r, double q,
                                  int32_t option_type) {
    const double forward = spot * std::exp((r - q) * t);
    const double qf = std::exp(-q * t);
    if (std::fabs(forward - strike) < kEps * strike) {
        return (option_type == QK_CALL) ? 0.5 * qf : -0.5 * qf;
    }
    if (option_type == QK_CALL) return (forward > strike) ? qf : 0.0;
    return (forward < strike) ? -qf : 0.0;
}

inline double bsm_delta(double spot, double strike, double t, double vol, double r, double q,
                        int32_t option_type) {
    if (t <= kEps || vol <= kEps) {
        return deterministic_delta(spot, strike, t, r, q, option_type);
    }

    const double sqrt_t = std::sqrt(t);
    const double d1 = (std::log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t);
    const double qf = std::exp(-q * t);
    if (option_type == QK_CALL) return qf * norm_cdf(d1);
    return qf * (norm_cdf(d1) - 1.0);
}

inline double clamp_delta(double delta, double t, double q) {
    const double qf = std::exp(-q * t);
    return std::min(qf, std::max(-qf, delta));
}

inline double splitmix64_hash(uint64_t x) { return qk::splitmix64_hash(x); }

} // namespace qk::agm::detail

#endif /* QK_AGM_INTERNAL_UTIL_H */
