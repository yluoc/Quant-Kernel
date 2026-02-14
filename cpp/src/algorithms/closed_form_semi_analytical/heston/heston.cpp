#include "algorithms/closed_form_semi_analytical/heston/heston.h"

#include "algorithms/closed_form_semi_analytical/common/internal_util.h"

namespace qk::cfa {

double heston_price_cf(double spot, double strike, double t, double r, double q,
                       const HestonParams& params, int32_t option_type,
                       int32_t integration_steps, double integration_limit) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(params.v0) || !is_finite_safe(params.kappa) ||
        !is_finite_safe(params.theta) || !is_finite_safe(params.sigma) ||
        !is_finite_safe(params.rho)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || params.v0 < 0.0 || params.kappa <= 0.0 ||
        params.theta < 0.0 || params.sigma <= 0.0 || params.rho <= -1.0 || params.rho >= 1.0) {
        return detail::nan_value();
    }
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (integration_limit <= 1e-6) return detail::nan_value();

    double log_spot = std::log(spot);
    auto cf = [&](std::complex<double> u) -> std::complex<double> {
        std::complex<double> beta = params.kappa - params.rho * params.sigma * detail::kI * u;
        std::complex<double> d = std::sqrt(beta * beta + params.sigma * params.sigma * (detail::kI * u + u * u));
        std::complex<double> g = (beta - d) / (beta + d);
        std::complex<double> exp_neg_d_t = std::exp(-d * t);
        std::complex<double> C = detail::kI * u * (log_spot + (r - q) * t)
            + (params.kappa * params.theta / (params.sigma * params.sigma))
                * ((beta - d) * t - 2.0 * std::log((1.0 - g * exp_neg_d_t) / (1.0 - g)));
        std::complex<double> D = ((beta - d) / (params.sigma * params.sigma))
            * ((1.0 - exp_neg_d_t) / (1.0 - g * exp_neg_d_t));
        return std::exp(C + D * params.v0);
    };

    double log_strike = std::log(strike);
    double p1 = detail::probability_p1(cf, log_strike, integration_steps, integration_limit);
    double p2 = detail::probability_p2(cf, log_strike, integration_steps, integration_limit);
    if (!is_finite_safe(p1) || !is_finite_safe(p2)) return detail::nan_value();

    double call_price = spot * std::exp(-q * t) * p1 - strike * std::exp(-r * t) * p2;
    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

} // namespace qk::cfa
