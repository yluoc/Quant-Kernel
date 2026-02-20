#ifndef QK_MATH_UTIL_H
#define QK_MATH_UTIL_H

#include <cmath>
#include <cstdint>
#include <cstring>

namespace qk {


/* Write a NaN into *dst using memcpy to avoid -ffast-math optimizing it away */
inline void write_nan(double* dst) {
    static const uint64_t nan_bits = UINT64_C(0x7FF8000000000000);
    std::memcpy(dst, &nan_bits, sizeof(double));
}

inline bool is_finite_safe(double x) {
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(double));
    uint64_t exp_mask = UINT64_C(0x7FF0000000000000);
    return (bits & exp_mask) != exp_mask;
}


// Fast norm_cdf using Abramowitz & Stegun rational approximation (7.1.26)
// Max error ~1.5e-7, much faster than std::erfc
inline double norm_cdf(double x) {
    static constexpr double a1 = 0.254829592;
    static constexpr double a2 = -0.284496736;
    static constexpr double a3 = 1.421413741;
    static constexpr double a4 = -1.453152027;
    static constexpr double a5 = 1.061405429;
    static constexpr double p  = 0.3275911;

    int sign = (x < 0.0) ? -1 : 1;
    double ax = std::fabs(x) * M_SQRT1_2;

    double t = 1.0 / (1.0 + p * ax);
    double t2 = t * t;
    double t3 = t2 * t;
    double t4 = t3 * t;
    double t5 = t4 * t;
    double y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * std::exp(-ax * ax);

    return 0.5 * (1.0 + sign * y);
}

inline double norm_pdf(double x) {
    static constexpr double INV_SQRT_2PI = 0.3989422804014327;
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

} // namespace qk

#endif /* QK_MATH_UTIL_H */
