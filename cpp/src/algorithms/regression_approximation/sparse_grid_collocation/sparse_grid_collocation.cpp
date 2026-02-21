#include "algorithms/regression_approximation/sparse_grid_collocation/sparse_grid_collocation.h"

#include "algorithms/regression_approximation/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qk::ram {

double sparse_grid_collocation_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const SparseGridCollocationParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.level < 1 || params.nodes_per_dim < 2) {
        return detail::nan_value();
    }
    const double bsm_call = detail::call_from_bsm(spot, strike, t, vol, r, q);
    if (!is_finite_safe(bsm_call)) return detail::nan_value();

    const int level_nodes = (1 << std::min(params.level, 10)) + 1;
    const int n = std::min(257, std::max(params.nodes_per_dim, level_nodes));
    const double mu = (r - q - 0.5 * vol * vol) * t;
    const double sigma_t = vol * std::sqrt(t);
    constexpr double z_max = 6.0;

    std::vector<double> z_nodes(static_cast<std::size_t>(n));
    std::vector<double> values(static_cast<std::size_t>(n));
    std::vector<double> bary_w(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        const double x = std::cos(M_PI * static_cast<double>(j) / static_cast<double>(n - 1));
        const double z = z_max * x;
        z_nodes[static_cast<std::size_t>(j)] = z;
        const double st = spot * std::exp(mu + sigma_t * z);
        values[static_cast<std::size_t>(j)] = std::max(st - strike, 0.0);
        double bw = ((j == 0 || j == n - 1) ? 0.5 : 1.0);
        if ((j & 1) != 0) bw = -bw;
        bary_w[static_cast<std::size_t>(j)] = bw;
    }

    auto interp = [&](double z) -> double {
        double num = 0.0;
        double den = 0.0;
        for (int j = 0; j < n; ++j) {
            const double dz = z - z_nodes[static_cast<std::size_t>(j)];
            if (std::fabs(dz) <= 1e-12) return values[static_cast<std::size_t>(j)];
            const double t_j = bary_w[static_cast<std::size_t>(j)] / dz;
            num += t_j * values[static_cast<std::size_t>(j)];
            den += t_j;
        }
        return num / den;
    };

    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    const int n_quad = std::min(256, std::max(64, n * 2));
    detail::gauss_hermite_nodes(n_quad, gh_nodes, gh_weights);

    double expected_payoff = 0.0;
    for (int i = 0; i < n_quad; ++i) {
        const double z = std::sqrt(2.0) * gh_nodes[static_cast<std::size_t>(i)];
        const double w = gh_weights[static_cast<std::size_t>(i)] / std::sqrt(M_PI);
        expected_payoff += w * std::max(0.0, interp(z));
    }

    const double approx_call = std::exp(-r * t) * std::max(0.0, expected_payoff);
    const double stable_call = detail::stabilized_call_price(approx_call, bsm_call, spot, strike, t, r, q);
    return detail::call_put_from_call_parity(stable_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::ram
