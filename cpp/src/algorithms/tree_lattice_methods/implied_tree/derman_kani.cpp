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

    double sigma_ref = local_vol_surface(spot, 0.0);
    if (!is_finite_safe(sigma_ref) || sigma_ref < 0.0) return detail::nan_value();
    sigma_ref = std::max(sigma_ref, detail::kEps);

    double dx = sigma_ref * std::sqrt(3.0 * dt);
    double up = std::exp(dx);
    double inv_up = 1.0 / up;
    double dx2 = dx * dx;
    double inv_dx = 1.0 / dx;
    double inv_dx2 = 1.0 / dx2;
    std::size_t max_width = static_cast<std::size_t>(2 * N + 1);
    std::vector<double> buf_a(max_width);
    std::vector<double> buf_b(max_width);
    double* values = buf_a.data();
    double* next = buf_b.data();

    double node_spot = spot * std::pow(inv_up, static_cast<double>(N));
    for (int32_t j = -N; j <= N; ++j) {
        values[j + N] = detail::intrinsic_value(node_spot, strike, option_type);
        node_spot *= up;
    }

    for (int32_t n = N - 1; n >= 0; --n) {
        double node_t = static_cast<double>(n) * dt;
        double ns = spot * std::pow(inv_up, static_cast<double>(n));
        for (int32_t j = -n; j <= n; ++j) {
            double sigma = local_vol_surface(ns, node_t);
            if (!is_finite_safe(sigma) || sigma < 0.0) return detail::nan_value();
            sigma = std::max(sigma, detail::kEps);

            double sigma2 = sigma * sigma;
            double mu = r - q - 0.5 * sigma2;
            double var_dt = sigma2 * dt;
            double mu_dt = mu * dt;

            double drift_var = (var_dt + mu_dt * mu_dt) * inv_dx2;
            double drift_dir = mu_dt * inv_dx;
            double pu = 0.5 * (drift_var + drift_dir);
            double pd = 0.5 * (drift_var - drift_dir);
            double pm = 1.0 - pu - pd;

            pu = std::max(0.0, pu);
            pd = std::max(0.0, pd);
            pm = std::max(0.0, pm);
            double sum_p = pu + pm + pd;
            pu /= sum_p;
            pm /= sum_p;
            pd /= sum_p;

            int32_t center = j + (n + 1);
            double cont = disc * (pu * values[center + 1] +
                                  pm * values[center] +
                                  pd * values[center - 1]);

            if (config.american_style) {
                next[j + n] = std::max(cont, detail::intrinsic_value(ns, strike, option_type));
            } else {
                next[j + n] = cont;
            }
            ns *= up;
        }
        double* tmp = values; values = next; next = tmp;
    }

    return values[0];
}

} // namespace qk::tlm
