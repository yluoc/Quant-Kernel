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
    if (integration_steps < 16) return detail::nan_value();
    if (integration_limit <= 1e-6) return detail::nan_value();

    double log_spot = std::log(spot);
    double sigma2 = params.sigma * params.sigma;
    double inv_sigma2 = 1.0 / sigma2;
    double kappa_theta_inv_s2 = params.kappa * params.theta * inv_sigma2;
    double rho_sigma = params.rho * params.sigma;
    double drift = log_spot + (r - q) * t;

    auto cf = [&](std::complex<double> u) -> std::complex<double> {
        std::complex<double> iu = detail::kI * u;
        std::complex<double> beta = params.kappa - rho_sigma * iu;
        std::complex<double> d = std::sqrt(beta * beta + sigma2 * (iu + u * u));
        std::complex<double> bmd = beta - d;
        std::complex<double> g = bmd / (beta + d);
        std::complex<double> exp_neg_d_t = std::exp(-d * t);
        std::complex<double> C = iu * drift
            + kappa_theta_inv_s2 * (bmd * t - 2.0 * std::log((1.0 - g * exp_neg_d_t) / (1.0 - g)));
        std::complex<double> D = (bmd * inv_sigma2) * ((1.0 - exp_neg_d_t) / (1.0 - g * exp_neg_d_t));
        return std::exp(C + D * params.v0);
    };

    double log_strike = std::log(strike);
    double p1, p2;
    detail::probability_p1p2(cf, log_strike, integration_steps, integration_limit, p1, p2);
    if (!is_finite_safe(p1) || !is_finite_safe(p2)) return detail::nan_value();

    double call_price = spot * std::exp(-q * t) * p1 - strike * std::exp(-r * t) * p2;
    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

} // namespace qk::cfa
