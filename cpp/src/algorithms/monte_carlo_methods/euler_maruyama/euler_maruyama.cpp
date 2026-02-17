#include "algorithms/monte_carlo_methods/euler_maruyama/euler_maruyama.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"

#include <cmath>
#include <random>

namespace qk::mcm {

namespace {

double simulate_euler_path(double spot, double t, double vol, double drift,
                           int32_t steps, std::mt19937_64& rng) {
    const double dt = t / static_cast<double>(steps);
    const double sqrt_dt = std::sqrt(dt);
    std::normal_distribution<double> normal(0.0, 1.0);

    double s = spot;
    for (int32_t j = 0; j < steps; ++j) {
        double dw = sqrt_dt * normal(rng);
        s += drift * s * dt + vol * s * dw;
        s = std::max(1e-12, s);
    }
    return s;
}

} // namespace

double euler_maruyama_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            int32_t paths, int32_t steps, uint64_t seed) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        !detail::valid_mc_counts(paths, steps)) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    std::mt19937_64 rng(seed);
    const double disc = std::exp(-r * t);
    const double drift = r - q;
    double sum = 0.0;

    for (int32_t i = 0; i < paths; ++i) {
        double st = simulate_euler_path(spot, t, vol, drift, steps, rng);
        sum += detail::payoff(st, strike, option_type);
    }

    return disc * (sum / static_cast<double>(paths));
}

} // namespace qk::mcm
