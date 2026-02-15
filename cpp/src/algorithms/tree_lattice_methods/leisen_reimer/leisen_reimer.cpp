#include "algorithms/tree_lattice_methods/leisen_reimer/leisen_reimer.h"

#include "algorithms/tree_lattice_methods/common/internal_util.h"

#include <cmath>

namespace qk::tlm {
namespace {

double peizer_pratt_inverse(double z, int32_t n) {
    double n_adj = static_cast<double>(n) + (1.0 / 3.0);
    double n_den = n_adj + 0.1 / (static_cast<double>(n) + 1.0);
    double expo = -((z * z) / (n_den * n_den)) * (static_cast<double>(n) + 1.0 / 6.0);
    double term = std::sqrt(std::max(0.0, 1.0 - std::exp(expo)));
    return 0.5 + std::copysign(0.5 * term, z);
}

} // namespace

double leisen_reimer_price(double spot, double strike, double t, double vol, double r, double q,
                           int32_t option_type, int32_t steps, bool american_style) {
    if (!detail::valid_inputs(spot, strike, t, vol, steps, option_type) ||
        !is_finite_safe(r) || !is_finite_safe(q)) {
        return detail::nan_value();
    }
    if (steps % 2 == 0) ++steps;
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) {
        double fwd = spot * std::exp((r - q) * t);
        return std::exp(-r * t) * detail::intrinsic_value(fwd, strike, option_type);
    }

    double sqrt_t = std::sqrt(t);
    double d1 = (std::log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t);
    double d2 = d1 - vol * sqrt_t;

    double p = detail::clamp_probability(peizer_pratt_inverse(d2, steps));
    double p_tilde = detail::clamp_probability(peizer_pratt_inverse(d1, steps));
    double dt = t / static_cast<double>(steps);
    double growth = std::exp((r - q) * dt);
    double up = growth * (p_tilde / p);
    double down = (growth - p * up) / (1.0 - p);

    return detail::binomial_price(spot, strike, t, r, q, option_type, steps, american_style,
                                  up, down, p);
}

} // namespace qk::tlm
