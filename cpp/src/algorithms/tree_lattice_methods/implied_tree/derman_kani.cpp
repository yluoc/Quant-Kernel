#include "algorithms/tree_lattice_methods/implied_tree/derman_kani.h"

#include "algorithms/tree_lattice_methods/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qk::tlm {

double derman_kani_implied_tree_price(
    double spot, double strike, double t, double r, double q,
    int32_t option_type,
    const std::function<double(double, double)>& local_vol_surface,
    ImpliedTreeConfig config
) {
    if (!detail::valid_option_type(option_type) || !is_finite_safe(spot) || !is_finite_safe(strike) ||
        !is_finite_safe(t) || !is_finite_safe(r) || !is_finite_safe(q) ||
        spot <= 0.0 || strike <= 0.0 || t < 0.0 || config.steps <= 0) {
        return detail::nan_value();
    }
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    double dt = t / static_cast<double>(config.steps);
    double disc = std::exp(-r * dt);

    std::vector<std::vector<double>> spots(static_cast<std::size_t>(config.steps + 1));
    spots[0].push_back(spot);
    for (int32_t n = 0; n < config.steps; ++n) {
        std::vector<double> next(2 * spots[static_cast<std::size_t>(n)].size(), 0.0);
        for (std::size_t i = 0; i < spots[static_cast<std::size_t>(n)].size(); ++i) {
            double node_spot = spots[static_cast<std::size_t>(n)][i];
            double node_t = static_cast<double>(n) * dt;
            double sigma = local_vol_surface(node_spot, node_t);
            if (!is_finite_safe(sigma) || sigma < 0.0) return detail::nan_value();
            sigma = std::max(sigma, detail::kEps);
            double up = std::exp(sigma * std::sqrt(dt));
            double down = 1.0 / up;
            next[2 * i] = node_spot * down;
            next[2 * i + 1] = node_spot * up;
        }
        spots[static_cast<std::size_t>(n + 1)] = std::move(next);
    }

    std::vector<double> values(spots.back().size(), 0.0);
    for (std::size_t i = 0; i < spots.back().size(); ++i) {
        values[i] = detail::intrinsic_value(spots.back()[i], strike, option_type);
    }

    for (int32_t n = config.steps - 1; n >= 0; --n) {
        std::vector<double> next_values(spots[static_cast<std::size_t>(n)].size(), 0.0);
        for (std::size_t i = 0; i < spots[static_cast<std::size_t>(n)].size(); ++i) {
            double node_spot = spots[static_cast<std::size_t>(n)][i];
            double node_t = static_cast<double>(n) * dt;
            double sigma = std::max(local_vol_surface(node_spot, node_t), detail::kEps);
            double up = std::exp(sigma * std::sqrt(dt));
            double down = 1.0 / up;
            double p = detail::clamp_probability((std::exp((r - q) * dt) - down) / (up - down));
            double cont = disc * (p * values[2 * i + 1] + (1.0 - p) * values[2 * i]);
            if (config.american_style) {
                next_values[i] = std::max(cont, detail::intrinsic_value(node_spot, strike, option_type));
            } else {
                next_values[i] = cont;
            }
        }
        values.swap(next_values);
    }

    return values[0];
}

} // namespace qk::tlm
