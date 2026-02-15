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

    int32_t N = config.steps;
    double dt = t / static_cast<double>(N);
    double disc = std::exp(-r * dt);

    // Reference volatility sets the fixed log-space grid spacing.
    double sigma_ref = local_vol_surface(spot, 0.0);
    if (!is_finite_safe(sigma_ref) || sigma_ref < 0.0) return detail::nan_value();
    sigma_ref = std::max(sigma_ref, detail::kEps);

    // Trinomial spacing: dx = sigma_ref * sqrt(3 * dt) keeps p_m >= 0
    // when local vol is close to sigma_ref.
    double dx = sigma_ref * std::sqrt(3.0 * dt);
    double up = std::exp(dx);

    // Recombining trinomial tree: at step n there are (2n+1) nodes
    // with index j in [-n, n].  Node (n, j) has spot = S0 * exp(j * dx).

    // Terminal payoff at step N: (2N+1) nodes.
    std::vector<double> values(static_cast<std::size_t>(2 * N + 1), 0.0);
    for (int32_t j = -N; j <= N; ++j) {
        double node_spot = spot * std::pow(up, static_cast<double>(j));
        values[static_cast<std::size_t>(j + N)] =
            detail::intrinsic_value(node_spot, strike, option_type);
    }

    // Backward induction with node-specific trinomial probabilities.
    for (int32_t n = N - 1; n >= 0; --n) {
        std::vector<double> next(static_cast<std::size_t>(2 * n + 1), 0.0);
        for (int32_t j = -n; j <= n; ++j) {
            double node_spot = spot * std::pow(up, static_cast<double>(j));
            double node_t = static_cast<double>(n) * dt;
            double sigma = local_vol_surface(node_spot, node_t);
            if (!is_finite_safe(sigma) || sigma < 0.0) return detail::nan_value();
            sigma = std::max(sigma, detail::kEps);

            // Match drift and variance on the fixed grid.
            //   E[Dx]   = mu * dt,  Var[Dx] = sigma^2 * dt
            //   mu = r - q - sigma^2 / 2
            double mu = r - q - 0.5 * sigma * sigma;
            double var_dt = sigma * sigma * dt;
            double mu_dt = mu * dt;
            double dx2 = dx * dx;

            double pu = 0.5 * ((var_dt + mu_dt * mu_dt) / dx2 + mu_dt / dx);
            double pd = 0.5 * ((var_dt + mu_dt * mu_dt) / dx2 - mu_dt / dx);
            double pm = 1.0 - pu - pd;

            // Clamp and renormalise for numerical safety.
            pu = std::max(0.0, pu);
            pd = std::max(0.0, pd);
            pm = std::max(0.0, pm);
            double sum_p = pu + pm + pd;
            pu /= sum_p;
            pm /= sum_p;
            pd /= sum_p;

            // Children live in the (n+1)-level array of size 2(n+1)+1.
            // Node j at level n maps to children j-1, j, j+1 at level n+1.
            int32_t center = j + (n + 1);   // index of middle child
            double cont = disc * (pu * values[static_cast<std::size_t>(center + 1)] +
                                  pm * values[static_cast<std::size_t>(center)] +
                                  pd * values[static_cast<std::size_t>(center - 1)]);

            if (config.american_style) {
                next[static_cast<std::size_t>(j + n)] =
                    std::max(cont, detail::intrinsic_value(node_spot, strike, option_type));
            } else {
                next[static_cast<std::size_t>(j + n)] = cont;
            }
        }
        values.swap(next);
    }

    return values[0];
}

} // namespace qk::tlm
