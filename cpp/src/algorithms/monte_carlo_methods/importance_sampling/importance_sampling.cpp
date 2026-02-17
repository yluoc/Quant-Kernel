#include "algorithms/monte_carlo_methods/importance_sampling/importance_sampling.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"

#include <cmath>
#include <random>

namespace qk::mcm {

double importance_sampling_price(double spot, double strike, double t, double vol,
                                 double r, double q, int32_t option_type,
                                 int32_t paths, double shift, uint64_t seed) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        paths <= 1 || !is_finite_safe(shift)) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(shift, 1.0);
    const double disc = std::exp(-r * t);

    double sum = 0.0;
    for (int32_t i = 0; i < paths; ++i) {
        double y = normal(rng);
        double st = detail::gbm_terminal(spot, t, r, q, vol, y);
        double weight = std::exp(-shift * y + 0.5 * shift * shift);
        sum += detail::payoff(st, strike, option_type) * weight;
    }

    return disc * (sum / static_cast<double>(paths));
}

} // namespace qk::mcm
