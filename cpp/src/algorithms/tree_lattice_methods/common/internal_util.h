#ifndef QK_TLM_INTERNAL_UTIL_H
#define QK_TLM_INTERNAL_UTIL_H

#include "common/math_util.h"
#include <quantkernel/qk_abi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace qk::tlm::detail {

// Small-buffer optimization: use stack for small arrays, heap for large
static constexpr int32_t kStackThreshold = 1025; // steps+1 <= 1025 uses stack

constexpr double kEps = 1e-12;

inline double nan_value() {
    double out = 0.0;
    write_nan(&out);
    return out;
}

inline bool valid_option_type(int32_t option_type) {
    return option_type == QK_CALL || option_type == QK_PUT;
}

inline bool valid_inputs(double spot, double strike, double t, double vol,
                         int32_t steps, int32_t option_type) {
    if (!valid_option_type(option_type)) return false;
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) || !is_finite_safe(vol)) {
        return false;
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0 || steps <= 0) return false;
    return true;
}

inline double intrinsic_value(double x, double y, int32_t option_type) {
    if (option_type == QK_CALL) return std::max(0.0, x - y);
    if (option_type == QK_PUT) return std::max(0.0, y - x);
    return nan_value();
}

inline double clamp_probability(double p) {
    return std::min(1.0 - 1e-12, std::max(1e-12, p));
}

inline double binomial_price(double spot, double strike, double t, double r, double q,
                             int32_t option_type, int32_t steps, bool american_style,
                             double up, double down, double prob_up) {
    if (!is_finite_safe(r) || !is_finite_safe(q) || !is_finite_safe(up) ||
        !is_finite_safe(down) || !is_finite_safe(prob_up)) {
        return nan_value();
    }
    if (up <= 0.0 || down <= 0.0 || up <= down) return nan_value();

    double dt = t / static_cast<double>(steps);
    double disc = std::exp(-r * dt);
    double p = clamp_probability(prob_up);

    int32_t buf_sz = steps + 1;
    double stack_buf[kStackThreshold];
    std::unique_ptr<double[]> heap_buf;
    double* __restrict__ v;
    if (buf_sz <= kStackThreshold) {
        v = stack_buf;
    } else {
        heap_buf = std::make_unique<double[]>(static_cast<std::size_t>(buf_sz));
        v = heap_buf.get();
    }

    double ud = up / down;
    double stock = spot * std::pow(down, static_cast<double>(steps));
    for (int32_t i = 0; i <= steps; ++i) {
        v[i] = intrinsic_value(stock, strike, option_type);
        stock *= ud;
    }

    double one_minus_p = 1.0 - p;
    if (!american_style) {
        for (int32_t n = steps - 1; n >= 0; --n) {
            for (int32_t i = 0; i <= n; ++i) {
                v[i] = disc * (p * v[i + 1] + one_minus_p * v[i]);
            }
        }
    } else {
        for (int32_t n = steps - 1; n >= 0; --n) {
            double stock_n = spot * std::pow(down, static_cast<double>(n));
            for (int32_t i = 0; i <= n; ++i) {
                double cont = disc * (p * v[i + 1] + one_minus_p * v[i]);
                v[i] = std::max(cont, intrinsic_value(stock_n, strike, option_type));
                stock_n *= ud;
            }
        }
    }

    return v[0];
}

} // namespace qk::tlm::detail

#endif /* QK_TLM_INTERNAL_UTIL_H */
