#include "algorithms/regression_approximation/radial_basis_functions/radial_basis_functions.h"

#include "algorithms/regression_approximation/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qk::ram {

double radial_basis_function_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const RadialBasisFunctionParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.centers < 4 || !is_finite_safe(params.rbf_shape) || !is_finite_safe(params.ridge) ||
        params.rbf_shape <= 0.0 || params.ridge < 0.0) {
        return detail::nan_value();
    }
    const double bsm_call = detail::call_from_bsm(spot, strike, t, vol, r, q);
    if (!is_finite_safe(bsm_call)) return detail::nan_value();

    const int n = std::min(params.centers, 192);
    const double eps = params.rbf_shape;
    const double ridge = params.ridge;

    const double mu = (r - q - 0.5 * vol * vol) * t;
    const double sigma_t = vol * std::sqrt(t);
    constexpr double z_max = 6.0;

    std::vector<double> centers(static_cast<std::size_t>(n));
    const double dz = (2.0 * z_max) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        centers[static_cast<std::size_t>(i)] = -z_max + static_cast<double>(i) * dz;
    }

    std::vector<double> A(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    std::vector<double> wvec(static_cast<std::size_t>(n), 0.0);
    std::vector<double> y_center(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        const double zi = centers[static_cast<std::size_t>(i)];
        const double st = spot * std::exp(mu + sigma_t * zi);
        y_center[static_cast<std::size_t>(i)] = std::max(st - strike, 0.0);
        wvec[static_cast<std::size_t>(i)] = y_center[static_cast<std::size_t>(i)];

        for (int j = 0; j < n; ++j) {
            const double zj = centers[static_cast<std::size_t>(j)];
            const double d = zi - zj;
            double phi = std::exp(-(eps * d) * (eps * d));
            if (i == j) phi += ridge;
            A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)] = phi;
        }
    }
    bool solved_collocation = detail::solve_linear_system(A, wvec, n);

    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    const int n_quad = std::min(256, std::max(64, n * 3));
    detail::gauss_hermite_nodes(n_quad, gh_nodes, gh_weights);

    double expected_payoff = 0.0;
    for (int i = 0; i < n_quad; ++i) {
        const double z = std::sqrt(2.0) * gh_nodes[static_cast<std::size_t>(i)];
        const double q_w = gh_weights[static_cast<std::size_t>(i)] / std::sqrt(M_PI);

        double payoff_hat = 0.0;
        if (solved_collocation) {
            for (int j = 0; j < n; ++j) {
                const double d = z - centers[static_cast<std::size_t>(j)];
                payoff_hat += wvec[static_cast<std::size_t>(j)] * std::exp(-(eps * d) * (eps * d));
            }
            if (!is_finite_safe(payoff_hat)) {
                solved_collocation = false;
            }
        }
        if (!solved_collocation) {
            double num = 0.0;
            double den = 0.0;
            for (int j = 0; j < n; ++j) {
                const double d = z - centers[static_cast<std::size_t>(j)];
                const double k = std::exp(-(eps * d) * (eps * d));
                num += k * y_center[static_cast<std::size_t>(j)];
                den += k;
            }
            if (den <= 1e-14) return detail::nan_value();
            payoff_hat = num / den;
        }
        expected_payoff += q_w * std::max(0.0, payoff_hat);
    }

    const double approx_call = std::exp(-r * t) * std::max(0.0, expected_payoff);
    const double stable_call = detail::stabilized_call_price(approx_call, bsm_call, spot, strike, t, r, q);
    return detail::call_put_from_call_parity(stable_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::ram
