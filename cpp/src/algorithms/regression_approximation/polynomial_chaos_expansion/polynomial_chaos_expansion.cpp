#include "algorithms/regression_approximation/polynomial_chaos_expansion/polynomial_chaos_expansion.h"

#include "algorithms/regression_approximation/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qk::ram {

double polynomial_chaos_expansion_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const PolynomialChaosExpansionParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.polynomial_order < 1 || params.quadrature_points < 4) {
        return detail::nan_value();
    }
    const double bsm_call = detail::call_from_bsm(spot, strike, t, vol, r, q);
    if (!is_finite_safe(bsm_call)) return detail::nan_value();

    const int p = std::min(params.polynomial_order, 24);
    const int n_quad = std::min(256, std::max(params.quadrature_points, p + 4));

    std::vector<double> nodes;
    std::vector<double> weights;
    detail::gauss_hermite_nodes(n_quad, nodes, weights);
    if (static_cast<int>(nodes.size()) != n_quad || static_cast<int>(weights.size()) != n_quad) {
        return detail::nan_value();
    }

    const double mu = (r - q - 0.5 * vol * vol) * t;
    const double sigma_t = vol * std::sqrt(t);

    std::vector<double> coeff(static_cast<std::size_t>(p + 1), 0.0);
    std::vector<double> basis(static_cast<std::size_t>(p + 1), 0.0);
    std::vector<double> inv_sqrt_fact(static_cast<std::size_t>(p + 1), 1.0);
    for (int n = 2; n <= p; ++n) {
        inv_sqrt_fact[static_cast<std::size_t>(n)] =
            inv_sqrt_fact[static_cast<std::size_t>(n - 1)] / std::sqrt(static_cast<double>(n));
    }

    for (int i = 0; i < n_quad; ++i) {
        const double z = std::sqrt(2.0) * nodes[static_cast<std::size_t>(i)];
        const double st = spot * std::exp(mu + sigma_t * z);
        const double payoff = std::max(st - strike, 0.0);
        const double w = weights[static_cast<std::size_t>(i)] / std::sqrt(M_PI);

        // Probabilists Hermite via recurrence: He_{n+1}(z)=z He_n(z)-n He_{n-1}(z)
        basis[0] = 1.0;
        if (p >= 1) basis[1] = z;
        for (int n = 1; n < p; ++n) {
            basis[static_cast<std::size_t>(n + 1)] =
                z * basis[static_cast<std::size_t>(n)] - static_cast<double>(n) * basis[static_cast<std::size_t>(n - 1)];
        }
        for (int n = 0; n <= p; ++n) {
            const double psi = basis[static_cast<std::size_t>(n)] * inv_sqrt_fact[static_cast<std::size_t>(n)];
            coeff[static_cast<std::size_t>(n)] += w * payoff * psi;
        }
    }

    double expected_payoff = 0.0;
    for (int i = 0; i < n_quad; ++i) {
        const double z = std::sqrt(2.0) * nodes[static_cast<std::size_t>(i)];
        const double w = weights[static_cast<std::size_t>(i)] / std::sqrt(M_PI);

        basis[0] = 1.0;
        if (p >= 1) basis[1] = z;
        for (int n = 1; n < p; ++n) {
            basis[static_cast<std::size_t>(n + 1)] =
                z * basis[static_cast<std::size_t>(n)] - static_cast<double>(n) * basis[static_cast<std::size_t>(n - 1)];
        }
        double payoff_hat = 0.0;
        for (int n = 0; n <= p; ++n) {
            const double psi = basis[static_cast<std::size_t>(n)] * inv_sqrt_fact[static_cast<std::size_t>(n)];
            payoff_hat += coeff[static_cast<std::size_t>(n)] * psi;
        }
        expected_payoff += w * std::max(0.0, payoff_hat);
    }

    const double approx_call = std::exp(-r * t) * std::max(0.0, expected_payoff);
    const double stable_call = detail::stabilized_call_price(approx_call, bsm_call, spot, strike, t, r, q);
    return detail::call_put_from_call_parity(stable_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::ram
