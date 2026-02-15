#include "algorithms/tree_lattice_methods/trinomial_tree/trinomial_tree.h"

#include "algorithms/tree_lattice_methods/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qk::tlm {

double trinomial_tree_price(double spot, double strike, double t, double vol, double r, double q,
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
    double disc = std::exp(-r * dt);

    double dx = vol * std::sqrt(2.0 * dt);
    double up = std::exp(dx);
    double edr = std::exp((r - q) * dt * 0.5);
    double ev = std::exp(vol * std::sqrt(dt * 0.5));
    double pu = std::pow((edr - 1.0 / ev) / (ev - 1.0 / ev), 2.0);
    double pd = std::pow((ev - edr) / (ev - 1.0 / ev), 2.0);
    double pm = std::max(0.0, 1.0 - pu - pd);
    double sum_p = pu + pm + pd;
    pu /= sum_p;
    pm /= sum_p;
    pd /= sum_p;

    int32_t width = 2 * steps + 1;
    std::vector<double> values(static_cast<std::size_t>(width), 0.0);
    for (int32_t j = -steps; j <= steps; ++j) {
        double node_spot = spot * std::pow(up, static_cast<double>(j));
        values[static_cast<std::size_t>(j + steps)] = detail::intrinsic_value(node_spot, strike, option_type);
    }

    for (int32_t n = steps - 1; n >= 0; --n) {
        std::vector<double> next(static_cast<std::size_t>(2 * n + 1), 0.0);
        for (int32_t j = -n; j <= n; ++j) {
            int32_t idx = j + n;
            int32_t center = j + (n + 1);
            double cont = disc * (pu * values[static_cast<std::size_t>(center + 1)] +
                                  pm * values[static_cast<std::size_t>(center)] +
                                  pd * values[static_cast<std::size_t>(center - 1)]);
            if (american_style) {
                double node_spot = spot * std::pow(up, static_cast<double>(j));
                next[static_cast<std::size_t>(idx)] =
                    std::max(cont, detail::intrinsic_value(node_spot, strike, option_type));
            } else {
                next[static_cast<std::size_t>(idx)] = cont;
            }
        }
        values.swap(next);
    }

    return values[0];
}

} // namespace qk::tlm
