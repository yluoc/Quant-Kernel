#ifndef QK_MCM_INTERNAL_UTIL_H
#define QK_MCM_INTERNAL_UTIL_H

#include "common/math_util.h"
#include <quantkernel/qk_abi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace qk::mcm::detail {

constexpr double kEps = 1e-12;

inline double nan_value() {
    double out = 0.0;
    write_nan(&out);
    return out;
}

inline bool valid_option_type(int32_t option_type) {
    return option_type == QK_CALL || option_type == QK_PUT;
}

inline bool valid_common_inputs(double spot, double strike, double t, double vol,
                                double r, double q, int32_t option_type) {
    if (!valid_option_type(option_type)) return false;
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q)) {
        return false;
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return false;
    return true;
}

inline bool valid_mc_counts(int32_t paths, int32_t steps) {
    return paths > 1 && steps > 0;
}

inline double intrinsic_value(double s, double k, int32_t option_type) {
    if (option_type == QK_CALL) return std::max(0.0, s - k);
    if (option_type == QK_PUT) return std::max(0.0, k - s);
    return nan_value();
}

inline double gbm_terminal(double spot, double t, double r, double q, double vol, double z) {
    double drift = (r - q - 0.5 * vol * vol) * t;
    double diffusion = vol * std::sqrt(std::max(t, 0.0)) * z;
    return spot * std::exp(drift + diffusion);
}

inline double payoff(double spot_t, double strike, int32_t option_type) {
    return intrinsic_value(spot_t, strike, option_type);
}

// Acklam inverse-normal approximation.
inline double inv_norm_cdf(double p) {
    if (p <= 0.0 || p >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    static constexpr double a1 = -3.969683028665376e+01;
    static constexpr double a2 = 2.209460984245205e+02;
    static constexpr double a3 = -2.759285104469687e+02;
    static constexpr double a4 = 1.383577518672690e+02;
    static constexpr double a5 = -3.066479806614716e+01;
    static constexpr double a6 = 2.506628277459239e+00;

    static constexpr double b1 = -5.447609879822406e+01;
    static constexpr double b2 = 1.615858368580409e+02;
    static constexpr double b3 = -1.556989798598866e+02;
    static constexpr double b4 = 6.680131188771972e+01;
    static constexpr double b5 = -1.328068155288572e+01;

    static constexpr double c1 = -7.784894002430293e-03;
    static constexpr double c2 = -3.223964580411365e-01;
    static constexpr double c3 = -2.400758277161838e+00;
    static constexpr double c4 = -2.549732539343734e+00;
    static constexpr double c5 = 4.374664141464968e+00;
    static constexpr double c6 = 2.938163982698783e+00;

    static constexpr double d1 = 7.784695709041462e-03;
    static constexpr double d2 = 3.224671290700398e-01;
    static constexpr double d3 = 2.445134137142996e+00;
    static constexpr double d4 = 3.754408661907416e+00;

    static constexpr double p_low = 0.02425;
    static constexpr double p_high = 1.0 - p_low;

    if (p < p_low) {
        double q = std::sqrt(-2.0 * std::log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    if (p > p_high) {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    double q = p - 0.5;
    double r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

inline int trailing_zero_count(uint32_t x) {
    if (x == 0U) return 0;
    int c = 0;
    while ((x & 1U) == 0U) {
        ++c;
        x >>= 1U;
    }
    return c;
}

inline double sobol_1d(uint32_t index) {
    static thread_local uint32_t x = 0U;
    static thread_local uint32_t prev = 0U;

    if (index == 0U) {
        x = 0U;
        prev = 0U;
        return 0.0;
    }

    if (index != prev + 1U) {
        x = 0U;
        prev = 0U;
        for (uint32_t i = 1U; i <= index; ++i) {
            uint32_t c = static_cast<uint32_t>(trailing_zero_count(i));
            x ^= (1U << (31U - c));
            prev = i;
        }
    } else {
        uint32_t c = static_cast<uint32_t>(trailing_zero_count(index));
        x ^= (1U << (31U - c));
        prev = index;
    }

    return (static_cast<double>(x) + 0.5) / 4294967296.0;
}

inline double halton(uint32_t index, uint32_t base) {
    double f = 1.0;
    double r = 0.0;
    uint32_t i = index;
    while (i > 0U) {
        f /= static_cast<double>(base);
        r += f * static_cast<double>(i % base);
        i /= base;
    }
    return r;
}

} // namespace qk::mcm::detail

#endif /* QK_MCM_INTERNAL_UTIL_H */
