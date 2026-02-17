#include "algorithms/monte_carlo_methods/quasi_monte_carlo/quasi_monte_carlo.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"

#include <algorithm>
#include <cmath>

namespace qk::mcm {

double quasi_monte_carlo_sobol_price(double spot, double strike, double t, double vol,
                                     double r, double q, int32_t option_type,
                                     int32_t paths) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        paths <= 1) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    const double disc = std::exp(-r * t);
    double sum = 0.0;

    for (int32_t i = 1; i <= paths; ++i) {
        double u = detail::sobol_1d(static_cast<uint32_t>(i));
        u = std::clamp(u, 1e-12, 1.0 - 1e-12);
        double z = detail::inv_norm_cdf(u);
        double st = detail::gbm_terminal(spot, t, r, q, vol, z);
        sum += detail::payoff(st, strike, option_type);
    }

    return disc * (sum / static_cast<double>(paths));
}

double quasi_monte_carlo_halton_price(double spot, double strike, double t, double vol,
                                      double r, double q, int32_t option_type,
                                      int32_t paths) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        paths <= 1) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    const double disc = std::exp(-r * t);
    double sum = 0.0;

    for (int32_t i = 1; i <= paths; ++i) {
        double u = detail::halton(static_cast<uint32_t>(i), 2U);
        u = std::clamp(u, 1e-12, 1.0 - 1e-12);
        double z = detail::inv_norm_cdf(u);
        double st = detail::gbm_terminal(spot, t, r, q, vol, z);
        sum += detail::payoff(st, strike, option_type);
    }

    return disc * (sum / static_cast<double>(paths));
}

} // namespace qk::mcm
