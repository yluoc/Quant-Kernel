#include "algorithms/monte_carlo_methods/milstein/milstein.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"
#include "common/mc_engine.h"
#include "common/model_concepts.h"

#include <cmath>

namespace qk::mcm {

double milstein_price(double spot, double strike, double t, double vol,
                      double r, double q, int32_t option_type,
                      int32_t paths, int32_t steps, uint64_t seed) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        !detail::valid_mc_counts(paths, steps)) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    const double disc = std::exp(-r * t);
    const double drift = r - q;
    auto gen = mc::make_mt19937_normal(seed);
    auto step = models::make_bsm_milstein_step(vol, drift);
    auto accum = [&](double S_T, int) { return detail::payoff(S_T, strike, option_type); };
    double mean = mc::estimate_stepwise(spot, t, paths, steps, gen, step, accum);
    return disc * mean;
}

} // namespace qk::mcm
