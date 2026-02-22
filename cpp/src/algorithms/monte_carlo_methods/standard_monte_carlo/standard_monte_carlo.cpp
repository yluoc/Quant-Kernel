#include "algorithms/monte_carlo_methods/standard_monte_carlo/standard_monte_carlo.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"
#include "common/mc_engine.h"
#include "common/model_concepts.h"

#include <cmath>

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

    const double disc = std::exp(-r * t);
    auto gen = mc::make_mt19937_normal(seed);
    auto model = models::make_bsm_terminal(vol, r, q);
    auto accum = [&](double S_T, double /*z*/, int) {
        return detail::payoff(S_T, strike, option_type);
    };
    double mean = mc::estimate_terminal_antithetic(spot, t, paths, gen, model, accum);
    return disc * mean;
}

} // namespace qk::mcm
