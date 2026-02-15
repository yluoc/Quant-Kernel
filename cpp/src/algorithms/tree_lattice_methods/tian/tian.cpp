#include "algorithms/tree_lattice_methods/tian/tian.h"

#include "algorithms/tree_lattice_methods/common/internal_util.h"

#include <cmath>

namespace qk::tlm {

double tian_price(double spot, double strike, double t, double vol, double r, double q,
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
    double rdt = std::exp((r - q) * dt);
    double v = std::exp(vol * vol * dt);
    double sqrt_term = std::sqrt(std::max(0.0, v * v + 2.0 * v - 3.0));
    double up = 0.5 * rdt * v * (v + 1.0 + sqrt_term);
    double down = 0.5 * rdt * v * (v + 1.0 - sqrt_term);
    double p = (rdt - down) / (up - down);
    return detail::binomial_price(spot, strike, t, r, q, option_type, steps, american_style,
                                  up, down, p);
}

} // namespace qk::tlm
