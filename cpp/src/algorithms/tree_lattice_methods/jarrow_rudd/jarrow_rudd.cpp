#include "algorithms/tree_lattice_methods/jarrow_rudd/jarrow_rudd.h"

#include "algorithms/tree_lattice_methods/common/internal_util.h"

#include <cmath>

namespace qk::tlm {

double jarrow_rudd_price(double spot, double strike, double t, double vol, double r, double q,
                         int32_t option_type, int32_t steps, bool american_style) {
    if (!detail::valid_inputs(spot, strike, t, vol, steps, option_type) ||
        !is_finite_safe(r) || !is_finite_safe(q)) {
        return detail::nan_value();
    }
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) {
        double fwd = spot * std::exp((r - q) * t);
        return std::exp(-r * t) * detail::intrinsic_value(fwd, strike, option_type);
    }

    double dt = t / static_cast<double>(steps);
    double drift = (r - q - 0.5 * vol * vol) * dt;
    double vol_term = vol * std::sqrt(dt);
    double up = std::exp(drift + vol_term);
    double down = std::exp(drift - vol_term);
    double p = 0.5;
    return detail::binomial_price(spot, strike, t, r, q, option_type, steps, american_style,
                                  up, down, p);
}

} // namespace qk::tlm
