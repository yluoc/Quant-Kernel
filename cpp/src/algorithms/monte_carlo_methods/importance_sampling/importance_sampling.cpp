#include "algorithms/monte_carlo_methods/importance_sampling/importance_sampling.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"
#include "common/mc_engine.h"
#include "common/model_concepts.h"

#include <cmath>

namespace qk::mcm {

double importance_sampling_price(double spot, double strike, double t, double vol,
                                 double r, double q, int32_t option_type,
                                 int32_t paths, double shift, uint64_t seed) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        paths <= 1 || !is_finite_safe(shift)) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    const double disc = std::exp(-r * t);
    auto gen = mc::make_mt19937_normal(seed);
    auto terminal = models::make_bsm_terminal(vol, r, q);
    auto accum = [&](double S_T, double z, int) {
        double y = z + shift;
        double weight = std::exp(-shift * y + 0.5 * shift * shift);
        return detail::payoff(S_T, strike, option_type) * weight;
    };

    auto shifted_model = [&](double sp, double tt, double z) {
        return terminal(sp, tt, z + shift);
    };

    double mean = mc::estimate_terminal(spot, t, paths, gen, shifted_model, accum);
    return disc * mean;
}

} // namespace qk::mcm
