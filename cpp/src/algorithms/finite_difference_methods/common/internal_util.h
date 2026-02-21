#ifndef QK_FDM_INTERNAL_UTIL_H
#define QK_FDM_INTERNAL_UTIL_H

#include "common/option_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace qk::fdm::detail {

constexpr double kEps = 1e-12;

inline double nan_value() { return qk::nan_value(); }

inline bool valid_option_type(int32_t option_type) { return qk::valid_option_type(option_type); }

inline bool valid_inputs(double spot, double strike, double t, double vol,
                         int32_t option_type) {
    if (!valid_option_type(option_type)) return false;
    if (!is_finite_safe(spot) || !is_finite_safe(strike) ||
        !is_finite_safe(t) || !is_finite_safe(vol))
        return false;
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return false;
    return true;
}

inline bool valid_grid_params(int32_t time_steps, int32_t spot_steps) {
    return time_steps > 0 && spot_steps > 1;
}

inline double intrinsic_value(double s, double k, int32_t option_type) {
    return qk::intrinsic_value(s, k, option_type);
}

inline void thomas_solve(double* __restrict__ a, double* __restrict__ b,
                         double* __restrict__ c, double* __restrict__ d, int32_t n) {
    for (int32_t i = 1; i < n; ++i) {
        double m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] -= m * d[i - 1];
    }
    d[n - 1] /= b[n - 1];
    for (int32_t i = n - 2; i >= 0; --i) {
        d[i] = (d[i] - c[i] * d[i + 1]) / b[i];
    }
}

inline std::vector<double> build_spot_grid(double spot, double strike, double vol, double t,
                                           int32_t spot_steps, double& ds) {
    // Extremely large vol*sqrt(t) can otherwise create unusably coarse grids.
    const double sigma_sqrt_t = vol * std::sqrt(t);
    const double spread = std::clamp(4.0 * sigma_sqrt_t + 0.25, 0.35, 2.50);
    const double buffer = std::exp(spread);

    const double base_min = std::min(spot, strike);
    const double base_max = std::max(spot, strike);
    double s_min = base_min / buffer;
    double s_max = base_max * buffer;
    if (s_min < 1e-8) s_min = 1e-8;
    if (s_max <= s_min + 1e-8) s_max = s_min + 1.0;

    ds = (s_max - s_min) / static_cast<double>(spot_steps);
    std::vector<double> S(spot_steps + 1);
    for (int32_t i = 0; i <= spot_steps; ++i) {
        S[i] = s_min + i * ds;
    }
    return S;
}

inline double interpolate_price(const std::vector<double>& S,
                                const std::vector<double>& V,
                                double spot) {
    int32_t n = static_cast<int32_t>(S.size());
    if (spot <= S[0]) return V[0];
    if (spot >= S[n - 1]) return V[n - 1];
    for (int32_t i = 0; i < n - 1; ++i) {
        if (S[i] <= spot && spot <= S[i + 1]) {
            double w = (spot - S[i]) / (S[i + 1] - S[i]);
            return V[i] * (1.0 - w) + V[i + 1] * w;
        }
    }
    return V[n / 2];
}

inline double upper_boundary(double s_max, double strike, double r, double q,
                             double tau, int32_t option_type) {
    if (option_type == QK_CALL)
        return s_max * std::exp(-q * tau) - strike * std::exp(-r * tau);
    return 0.0; // put value at S_max
}

inline double lower_boundary(double s_min, double strike, double r, double q,
                             double tau, int32_t option_type) {
    if (option_type == QK_PUT)
        return strike * std::exp(-r * tau) - s_min * std::exp(-q * tau);
    return 0.0; // call value at S_min
}

} // namespace qk::fdm::detail

#endif /* QK_FDM_INTERNAL_UTIL_H */
