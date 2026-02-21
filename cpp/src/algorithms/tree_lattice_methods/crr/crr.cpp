#include "algorithms/tree_lattice_methods/crr/crr.h"

#include "algorithms/tree_lattice_methods/common/internal_util.h"

#include <algorithm>
#include <cmath>

namespace qk::tlm {

double crr_price(double spot, double strike, double t, double vol, double r, double q,
                 int32_t option_type, int32_t steps, bool american_style) {
    if (!detail::valid_inputs(spot, strike, t, vol, steps, option_type) ||
        !is_finite_safe(r) || !is_finite_safe(q)) {
        return detail::nan_value();
    }
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps || vol * std::sqrt(t) <= 1e-3) {
        return detail::deterministic_limit_price(spot, strike, t, r, q, option_type, american_style);
    }

    double dt = t / static_cast<double>(steps);
    double up = std::exp(vol * std::sqrt(dt));
    double down = 1.0 / up;
    double growth = std::exp((r - q) * dt);
    if (!(down < growth && growth < up)) {
        return detail::deterministic_limit_price(spot, strike, t, r, q, option_type, american_style);
    }
    double p = (growth - down) / (up - down);
    return detail::binomial_price(spot, strike, t, r, q, option_type, steps, american_style,
                                  up, down, p);
}

} // namespace qk::tlm
