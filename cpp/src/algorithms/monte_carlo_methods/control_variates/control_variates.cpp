#include "algorithms/monte_carlo_methods/control_variates/control_variates.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"

#include <cmath>
#include <random>

namespace qk::mcm {

double control_variates_price(double spot, double strike, double t, double vol,
                              double r, double q, int32_t option_type,
                              int32_t paths, uint64_t seed) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        paths <= 2) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double disc = std::exp(-r * t);
    const double expected_x = spot * std::exp(-q * t);

    double sum_y = 0.0;
    double sum_x = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;

    for (int32_t i = 0; i < paths; ++i) {
        double z = normal(rng);
        double st = detail::gbm_terminal(spot, t, r, q, vol, z);
        double x = disc * st;
        double y = disc * detail::payoff(st, strike, option_type);

        sum_y += y;
        sum_x += x;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    double mean_y = sum_y / static_cast<double>(paths);
    double mean_x = sum_x / static_cast<double>(paths);
    double var_x = (sum_xx / static_cast<double>(paths)) - mean_x * mean_x;
    if (var_x <= detail::kEps) return mean_y;

    double cov_xy = (sum_xy / static_cast<double>(paths)) - mean_x * mean_y;
    double beta = cov_xy / var_x;

    return mean_y - beta * (mean_x - expected_x);
}

} // namespace qk::mcm
