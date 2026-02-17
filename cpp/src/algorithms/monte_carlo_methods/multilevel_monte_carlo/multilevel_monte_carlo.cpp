#include "algorithms/monte_carlo_methods/multilevel_monte_carlo/multilevel_monte_carlo.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"

#include <cmath>
#include <random>

namespace qk::mcm {

double multilevel_monte_carlo_price(double spot, double strike, double t, double vol,
                                    double r, double q, int32_t option_type,
                                    int32_t base_paths, int32_t levels, int32_t base_steps,
                                    uint64_t seed) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        base_paths <= 1 || levels <= 0 || base_steps <= 0) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double estimate = 0.0;

    for (int32_t level = 0; level < levels; ++level) {
        int32_t n_level = std::max(16, base_paths >> level);
        int32_t fine_steps = base_steps << level;
        double dt_fine = t / static_cast<double>(fine_steps);
        double sqrt_dt_fine = std::sqrt(dt_fine);
        double disc = std::exp(-r * t);

        double level_sum = 0.0;

        for (int32_t i = 0; i < n_level; ++i) {
            if (level == 0) {
                double s = spot;
                for (int32_t j = 0; j < fine_steps; ++j) {
                    double z = normal(rng);
                    s += (r - q) * s * dt_fine + vol * s * sqrt_dt_fine * z;
                    s = std::max(1e-12, s);
                }
                level_sum += detail::payoff(s, strike, option_type);
            } else {
                double s_fine = spot;
                double s_coarse = spot;
                for (int32_t coarse_step = 0; coarse_step < (fine_steps / 2); ++coarse_step) {
                    double z1 = normal(rng);
                    double z2 = normal(rng);

                    s_fine += (r - q) * s_fine * dt_fine + vol * s_fine * sqrt_dt_fine * z1;
                    s_fine = std::max(1e-12, s_fine);
                    s_fine += (r - q) * s_fine * dt_fine + vol * s_fine * sqrt_dt_fine * z2;
                    s_fine = std::max(1e-12, s_fine);

                    double zc = (z1 + z2) * M_SQRT1_2;
                    double dt_coarse = 2.0 * dt_fine;
                    s_coarse += (r - q) * s_coarse * dt_coarse + vol * s_coarse * std::sqrt(dt_coarse) * zc;
                    s_coarse = std::max(1e-12, s_coarse);
                }

                double pf = detail::payoff(s_fine, strike, option_type);
                double pc = detail::payoff(s_coarse, strike, option_type);
                level_sum += (pf - pc);
            }
        }

        estimate += disc * (level_sum / static_cast<double>(n_level));
    }

    return estimate;
}

} // namespace qk::mcm
