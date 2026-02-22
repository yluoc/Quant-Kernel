#ifndef QK_OPTION_UTIL_H
#define QK_OPTION_UTIL_H

#include "common/math_util.h"

#include <quantkernel/qk_abi.h>

#include <algorithm>
#include <cstdint>

namespace qk {

// Produces a quiet NaN through the shared math helper to keep invalid-result signaling consistent.
inline double nan_value() {
    double out = 0.0;
    write_nan(&out);
    return out;
}

// Valid option side encoding is restricted to ABI constants QK_CALL/QK_PUT.
inline bool valid_option_type(int32_t option_type) {
    return option_type == QK_CALL || option_type == QK_PUT;
}

// Generic intrinsic payoff helper: max(x - y, 0) for calls and max(y - x, 0) for puts.
// Returns NaN when option type is invalid so callers can propagate input errors.
inline double intrinsic_value(double x, double y, int32_t option_type) {
    if (option_type == QK_CALL) return std::max(0.0, x - y);
    if (option_type == QK_PUT)  return std::max(0.0, y - x);
    return nan_value();
}

// Shared scalar-domain validation used across pricing models:
// finite inputs, positive spot/strike, and non-negative maturity/volatility.
inline bool valid_common_inputs(double spot, double strike, double t, double vol,
                                double r, double q, int32_t option_type) {
    if (!valid_option_type(option_type)) return false;
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q)) {
        return false;
    }
    return spot > 0.0 && strike > 0.0 && t >= 0.0 && vol >= 0.0;
}

// SplitMix64-style bit mixer for deterministic pseudo-randomization.
// Output is normalized to [-1, 1] for downstream perturbation/sampling use.
inline double splitmix64_hash(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    // Map to [-1, 1]
    return static_cast<double>(static_cast<int64_t>(x)) / static_cast<double>(INT64_MAX);
}

} // namespace qk

#endif /* QK_OPTION_UTIL_H */
