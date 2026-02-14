#ifndef QK_CFA_INTERNAL_UTIL_H
#define QK_CFA_INTERNAL_UTIL_H

#include "common/math_util.h"
#include <quantkernel/qk_abi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>

namespace qk::cfa::detail {

constexpr double kPi = 3.1415926535897932384626433832795;
constexpr double kEps = 1e-12;
const std::complex<double> kI(0.0, 1.0);

inline double nan_value() {
    double out = 0.0;
    write_nan(&out);
    return out;
}

inline bool valid_option_type(int32_t option_type) {
    return option_type == QK_CALL || option_type == QK_PUT;
}

inline double intrinsic_value(double x, double y, int32_t option_type) {
    if (option_type == QK_CALL) return std::max(0.0, x - y);
    if (option_type == QK_PUT)  return std::max(0.0, y - x);
    return nan_value();
}

inline double clamp01(double x) {
    return std::min(1.0, std::max(0.0, x));
}

inline double integrate_simpson(const std::function<double(double)>& f,
                                double a, double b, int32_t steps) {
    if (steps < 64) steps = 64;
    if (steps % 2 != 0) ++steps;
    if (!(b > a)) return 0.0;

    double h = (b - a) / static_cast<double>(steps);
    double sum = f(a) + f(b);

    for (int32_t i = 1; i < steps; ++i) {
        double x = a + static_cast<double>(i) * h;
        double w = (i % 2 == 0) ? 2.0 : 4.0;
        sum += w * f(x);
    }
    return (h / 3.0) * sum;
}

template <typename CfFn>
inline double probability_p2(const CfFn& cf, double log_strike,
                             int32_t steps, double integration_limit) {
    auto integrand = [&](double u) -> double {
        std::complex<double> z(u, 0.0);
        std::complex<double> num = std::exp(-kI * z * log_strike) * cf(z);
        std::complex<double> den = kI * z;
        return std::real(num / den);
    };
    double p2 = 0.5 + (1.0 / kPi) * integrate_simpson(integrand, 1e-8, integration_limit, steps);
    return clamp01(p2);
}

template <typename CfFn>
inline double probability_p1(const CfFn& cf, double log_strike,
                             int32_t steps, double integration_limit) {
    std::complex<double> phi_minus_i = cf(std::complex<double>(0.0, -1.0));
    if (std::abs(phi_minus_i) < 1e-14) return nan_value();

    auto integrand = [&](double u) -> double {
        std::complex<double> z(u, -1.0);
        std::complex<double> num = std::exp(-kI * std::complex<double>(u, 0.0) * log_strike) * cf(z);
        std::complex<double> den = kI * std::complex<double>(u, 0.0) * phi_minus_i;
        return std::real(num / den);
    };
    double p1 = 0.5 + (1.0 / kPi) * integrate_simpson(integrand, 1e-8, integration_limit, steps);
    return clamp01(p1);
}

inline double call_put_from_call_parity(double call_price, double spot, double strike,
                                        double t, double r, double q, int32_t option_type) {
    if (option_type == QK_CALL) return call_price;
    if (option_type == QK_PUT) {
        return call_price - spot * std::exp(-q * t) + strike * std::exp(-r * t);
    }
    return nan_value();
}

} // namespace qk::cfa::detail

#endif /* QK_CFA_INTERNAL_UTIL_H */
