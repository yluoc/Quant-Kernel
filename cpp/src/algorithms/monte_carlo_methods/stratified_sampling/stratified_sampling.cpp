#include "algorithms/monte_carlo_methods/stratified_sampling/stratified_sampling.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace qk::mcm {

double stratified_sampling_price(double spot, double strike, double t, double vol,
                                 double r, double q, int32_t option_type,
                                 int32_t paths, uint64_t seed) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        paths <= 1) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const double disc = std::exp(-r * t);

    double sum = 0.0;
    for (int32_t i = 0; i < paths; ++i) {
        double u = (static_cast<double>(i) + unif(rng)) / static_cast<double>(paths);
        u = std::clamp(u, 1e-12, 1.0 - 1e-12);
        double z = detail::inv_norm_cdf(u);
        double st = detail::gbm_terminal(spot, t, r, q, vol, z);
        sum += detail::payoff(st, strike, option_type);
    }

    return disc * (sum / static_cast<double>(paths));
}

} // namespace qk::mcm
