#ifndef QK_TLM_INTERNAL_UTIL_H
#define QK_TLM_INTERNAL_UTIL_H

#include "common/math_util.h"
#include <quantkernel/qk_abi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace qk::tlm::detail {

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

    std::vector<double> values(static_cast<std::size_t>(steps + 1), 0.0);
    double stock = spot * std::pow(down, static_cast<double>(steps));
    double ud = up / down;
    for (int32_t i = 0; i <= steps; ++i) {
        values[static_cast<std::size_t>(i)] = intrinsic_value(stock, strike, option_type);
        stock *= ud;
    }

    for (int32_t n = steps - 1; n >= 0; --n) {
        double stock_n = spot * std::pow(down, static_cast<double>(n));
        for (int32_t i = 0; i <= n; ++i) {
            double cont = disc * (p * values[static_cast<std::size_t>(i + 1)] +
                                  (1.0 - p) * values[static_cast<std::size_t>(i)]);
            if (american_style) {
                double exercise = intrinsic_value(stock_n, strike, option_type);
                values[static_cast<std::size_t>(i)] = std::max(cont, exercise);
            } else {
                values[static_cast<std::size_t>(i)] = cont;
            }
            stock_n *= ud;
        }
    }

    return values[0];
}

} // namespace qk::tlm::detail

#endif /* QK_TLM_INTERNAL_UTIL_H */
