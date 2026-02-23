#include "algorithms/monte_carlo_methods/heston_monte_carlo/heston_monte_carlo.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"
#include "common/mc_engine.h"
#include "common/model_concepts.h"

#include <cmath>

namespace qk::mcm {

double heston_monte_carlo_price(double spot, double strike, double t,
                                double r, double q,
                                double v0, double kappa, double theta,
                                double sigma, double rho,
                                int32_t option_type,
                                int32_t paths, int32_t steps, uint64_t seed) {
    // Validate common inputs (pass vol=1.0 placeholder — Heston has no scalar vol).
    if (!detail::valid_option_type(option_type) ||
        spot <= 0.0 || strike <= 0.0 || t < 0.0 ||
        v0 < 0.0 || kappa < 0.0 || theta < 0.0 || sigma < 0.0 ||
        rho < -1.0 || rho > 1.0 ||
        !detail::valid_mc_counts(paths, steps)) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    const double disc = std::exp(-r * t);
    auto gen = mc::make_mt19937_normal(seed);
    auto step = models::make_heston_euler_step(r, q, kappa, theta, sigma, rho);
    auto accum = [&](double S_T, int) {
        return detail::payoff(S_T, strike, option_type);
    };
    double mean = mc::estimate_stepwise_2d(spot, v0, t, paths, steps, gen, step, accum);
    return disc * mean;
}

} // namespace qk::mcm
