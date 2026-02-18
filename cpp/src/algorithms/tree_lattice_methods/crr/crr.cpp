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
    if (vol <= detail::kEps) {
        double fwd = spot * std::exp((r - q) * t);
        return std::exp(-r * t) * detail::intrinsic_value(fwd, strike, option_type);
    }

    // Small step counts can under-resolve low-vol/high-carry cases. Use a
    // conservative minimum depth to improve robustness in production/fuzzing.
    const int32_t eff_steps = std::max<int32_t>(steps, 100);

    double dt = t / static_cast<double>(eff_steps);
    double up = std::exp(vol * std::sqrt(dt));
    double down = 1.0 / up;
    double growth = std::exp((r - q) * dt);
    double p = (growth - down) / (up - down);
    return detail::binomial_price(spot, strike, t, r, q, option_type, eff_steps, american_style,
                                  up, down, p);
}

} // namespace qk::tlm
