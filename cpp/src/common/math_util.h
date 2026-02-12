#ifndef QK_MATH_UTIL_H
#define QK_MATH_UTIL_H

#include <cmath>
#include <cstdint>
#include <cstring>

namespace qk {

/* --- NaN-safe helpers (safe under -ffast-math) --- */

/* Write a NaN into *dst using memcpy to avoid -ffast-math optimizing it away */
inline void write_nan(double* dst) {
    static const uint64_t nan_bits = UINT64_C(0x7FF8000000000000);
    std::memcpy(dst, &nan_bits, sizeof(double));
}

/* Check finiteness in integer domain (works under -ffast-math) */
inline bool is_finite_safe(double x) {
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(double));
    uint64_t exp_mask = UINT64_C(0x7FF0000000000000);
    return (bits & exp_mask) != exp_mask;
}

/* --- Standard normal distribution --- */

inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

inline double norm_pdf(double x) {
    static constexpr double INV_SQRT_2PI = 0.3989422804014327;
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

} // namespace qk

#endif /* QK_MATH_UTIL_H */
