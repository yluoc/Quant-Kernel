#include "algorithms/integral_quadrature/adaptive_quadrature/adaptive_quadrature.h"

#include "algorithms/integral_quadrature/common/internal_util.h"

namespace qk::iqm {

double adaptive_quadrature_price(double spot, double strike, double t, double vol,
                                 double r, double q, int32_t option_type,
                                 const AdaptiveQuadratureParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(params.abs_tol) || !is_finite_safe(params.rel_tol) ||
        !is_finite_safe(params.integration_limit)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) return detail::deterministic_price(spot, strike, t, r, q, option_type);

    if (params.abs_tol <= 0.0 || params.rel_tol <= 0.0 || params.max_depth < 2 ||
        params.integration_limit <= 0.0) {
        return detail::nan_value();
    }

    double log_moneyness = std::log(spot / strike);
    auto integrand = [&](double u) {
        return detail::lewis_integrand(u, log_moneyness, t, vol, r, q);
    };

    double integral = detail::integrate_adaptive_simpson(
        integrand, 0.0, params.integration_limit,
        params.abs_tol, params.rel_tol, params.max_depth
    );

    double call_price = spot * std::exp(-q * t)
        - std::sqrt(spot * strike) * std::exp(-r * t) * integral / detail::kPi;
    call_price = std::max(0.0, call_price);

    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

} // namespace qk::iqm
