#include "algorithms/closed_form_semi_analytical/variance_gamma/variance_gamma.h"

#include "algorithms/closed_form_semi_analytical/common/internal_util.h"

namespace qk::cfa {

double variance_gamma_price_cf(double spot, double strike, double t, double r, double q,
                               const VarianceGammaParams& params, int32_t option_type,
                               int32_t integration_steps, double integration_limit) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(params.sigma) || !is_finite_safe(params.theta) ||
        !is_finite_safe(params.nu)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || params.sigma <= 0.0 || params.nu <= 0.0) {
        return detail::nan_value();
    }
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (integration_limit <= 1e-6) return detail::nan_value();

    double sigma2 = params.sigma * params.sigma;
    double theta_nu = params.theta * params.nu;
    double half_sigma2_nu = 0.5 * sigma2 * params.nu;
    double drift_arg = 1.0 - theta_nu - half_sigma2_nu;
    if (drift_arg <= 0.0) return detail::nan_value();
    double omega = std::log(drift_arg) / params.nu;
    double log_spot = std::log(spot);
    double neg_t_over_nu = -t / params.nu;
    double drift_inner = log_spot + (r - q + omega) * t;

    auto cf = [&](std::complex<double> u) -> std::complex<double> {
        std::complex<double> iu = detail::kI * u;
        std::complex<double> base = 1.0 - theta_nu * iu + half_sigma2_nu * u * u;
        std::complex<double> drift = std::exp(iu * drift_inner);
        std::complex<double> jump = std::pow(base, neg_t_over_nu);
        return drift * jump;
    };

    double log_strike = std::log(strike);
    double p1, p2;
    detail::probability_p1p2(cf, log_strike, integration_steps, integration_limit, p1, p2);
    if (!is_finite_safe(p1) || !is_finite_safe(p2)) return detail::nan_value();

    double call_price = spot * std::exp(-q * t) * p1 - strike * std::exp(-r * t) * p2;
    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

} // namespace qk::cfa
