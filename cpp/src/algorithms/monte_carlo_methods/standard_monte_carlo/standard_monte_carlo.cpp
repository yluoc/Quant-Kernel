#include "algorithms/monte_carlo_methods/standard_monte_carlo/standard_monte_carlo.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"

#include <cmath>
#include <random>

namespace qk::mcm {

double standard_monte_carlo_price(double spot, double strike, double t, double vol,
                                  double r, double q, int32_t option_type,
                                  int32_t paths, uint64_t seed) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        paths <= 1) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) {
        const double fwd = spot * std::exp((r - q) * t);
        return std::exp(-r * t) * detail::intrinsic_value(fwd, strike, option_type);
    }

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double disc = std::exp(-r * t);

    double sum = 0.0;
    const int32_t n_pairs = paths / 2;
    for (int32_t i = 0; i < n_pairs; ++i) {
        const double z = normal(rng);
        const double st_up = detail::gbm_terminal(spot, t, r, q, vol, z);
        const double st_dn = detail::gbm_terminal(spot, t, r, q, vol, -z);
        sum += detail::payoff(st_up, strike, option_type);
        sum += detail::payoff(st_dn, strike, option_type);
    }
    if ((paths & 1) != 0) {
        const double z = normal(rng);
        const double st = detail::gbm_terminal(spot, t, r, q, vol, z);
        sum += detail::payoff(st, strike, option_type);
    }

    return disc * (sum / static_cast<double>(paths));
}

} // namespace qk::mcm
