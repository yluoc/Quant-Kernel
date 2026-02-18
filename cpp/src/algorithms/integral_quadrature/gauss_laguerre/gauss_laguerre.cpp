#include "algorithms/integral_quadrature/gauss_laguerre/gauss_laguerre.h"

#include "algorithms/integral_quadrature/common/internal_util.h"

namespace qk::iqm {

double gauss_laguerre_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            const GaussLaguerreParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) return detail::deterministic_price(spot, strike, t, r, q, option_type);

    int32_t n_points = params.n_points;
    if (n_points < 8 || n_points > 128) return detail::nan_value();

    std::vector<double> nodes;
    std::vector<double> weights;
    if (!detail::gauss_laguerre_rule(n_points, nodes, weights)) return detail::nan_value();

    double log_moneyness = std::log(spot / strike);
    double integral = 0.0;
    for (int32_t i = 0; i < n_points; ++i) {
        double u = nodes[static_cast<std::size_t>(i)];
        double base = detail::lewis_integrand(u, log_moneyness, t, vol, r, q);
        integral += weights[static_cast<std::size_t>(i)] * std::exp(u) * base;
    }

    double call_price = spot * std::exp(-q * t)
        - std::sqrt(spot * strike) * std::exp(-r * t) * integral / detail::kPi;
    call_price = std::max(0.0, call_price);

    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

} // namespace qk::iqm
