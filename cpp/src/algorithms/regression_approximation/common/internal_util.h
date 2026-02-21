#ifndef QK_RAM_INTERNAL_UTIL_H
#define QK_RAM_INTERNAL_UTIL_H

#include "algorithms/integral_quadrature/common/internal_util.h"
#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "common/option_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qk::ram::detail {

constexpr double kEps = 1e-12;

inline double nan_value() { return qk::nan_value(); }

inline bool valid_option_type(int32_t option_type) { return qk::valid_option_type(option_type); }

inline bool valid_common_inputs(double spot, double strike, double t, double vol,
                                double r, double q, int32_t option_type) {
    return qk::valid_common_inputs(spot, strike, t, vol, r, q, option_type);
}

inline double call_from_bsm(double spot, double strike, double t, double vol, double r, double q) {
    return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, QK_CALL);
}

inline double call_put_from_call_parity(double call_price, double spot, double strike,
                                        double t, double r, double q, int32_t option_type) {
    if (option_type == QK_CALL) return call_price;
    if (option_type == QK_PUT) {
        return call_price - spot * std::exp(-q * t) + strike * std::exp(-r * t);
    }
    return nan_value();
}

inline double no_arb_call_lower(double spot, double strike, double t, double r, double q) {
    return std::max(0.0, spot * std::exp(-q * t) - strike * std::exp(-r * t));
}

inline double no_arb_call_upper(double spot, double t, double q) {
    return spot * std::exp(-q * t);
}

inline double stabilized_call_price(
    double call_estimate, double fallback_call, double spot, double strike, double t, double r, double q
) {
    if (!is_finite_safe(fallback_call)) return call_estimate;
    if (!is_finite_safe(call_estimate)) return fallback_call;

    const double lb = no_arb_call_lower(spot, strike, t, r, q);
    const double ub = no_arb_call_upper(spot, t, q);
    if (call_estimate < lb - 1e-10 || call_estimate > ub + 1e-10) {
        return fallback_call;
    }
    return std::min(ub, std::max(lb, call_estimate));
}

inline double clamp_unit_interval(double x) {
    return std::min(1.0, std::max(0.0, x));
}

inline bool solve_linear_system(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int col = 0; col < n; ++col) {
        int max_row = col;
        double max_val = std::fabs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double v = std::fabs(A[row * n + col]);
            if (v > max_val) { max_val = v; max_row = row; }
        }
        if (max_val < 1e-15) return false;
        if (max_row != col) {
            for (int j = col; j < n; ++j) std::swap(A[col * n + j], A[max_row * n + j]);
            std::swap(b[col], b[max_row]);
        }
        const double pivot = A[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            const double factor = A[row * n + col] / pivot;
            for (int j = col; j < n; ++j) {
                A[row * n + j] -= factor * A[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }
    for (int row = n - 1; row >= 0; --row) {
        for (int j = row + 1; j < n; ++j) {
            b[row] -= A[row * n + j] * b[j];
        }
        b[row] /= A[row * n + row];
    }
    return true;
}

inline void gauss_hermite_nodes(int n, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.clear();
    weights.clear();
    if (n <= 0) return;

    if (!qk::iqm::detail::gauss_hermite_rule(n, nodes, weights)) {
        nodes.clear();
        weights.clear();
    }
}

} // namespace qk::ram::detail

#endif /* QK_RAM_INTERNAL_UTIL_H */
