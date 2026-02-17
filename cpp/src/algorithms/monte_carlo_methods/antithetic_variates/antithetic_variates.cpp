#include "algorithms/monte_carlo_methods/antithetic_variates/antithetic_variates.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"

#include <cmath>
#include <random>

namespace qk::mcm {

double antithetic_variates_price(double spot, double strike, double t, double vol,
                                 double r, double q, int32_t option_type,
                                 int32_t paths, uint64_t seed) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        paths <= 1) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double disc = std::exp(-r * t);

    int32_t pairs = std::max(1, paths / 2);
    double sum = 0.0;

    for (int32_t i = 0; i < pairs; ++i) {
        double z = normal(rng);
        double st1 = detail::gbm_terminal(spot, t, r, q, vol, z);
        double st2 = detail::gbm_terminal(spot, t, r, q, vol, -z);
        double p1 = detail::payoff(st1, strike, option_type);
        double p2 = detail::payoff(st2, strike, option_type);
        sum += 0.5 * (p1 + p2);
    }

    return disc * (sum / static_cast<double>(pairs));
}

} // namespace qk::mcm
