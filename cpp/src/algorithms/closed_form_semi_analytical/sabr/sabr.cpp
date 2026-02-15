#include "algorithms/closed_form_semi_analytical/sabr/sabr.h"

#include "algorithms/closed_form_semi_analytical/black_1976/black_1976.h"
#include "algorithms/closed_form_semi_analytical/common/internal_util.h"

namespace qk::cfa {

double sabr_hagan_lognormal_iv(double forward, double strike, double t,
                               const SABRParams& params) {
    if (!is_finite_safe(forward) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(params.alpha) || !is_finite_safe(params.beta) ||
        !is_finite_safe(params.rho) || !is_finite_safe(params.nu)) {
        return detail::nan_value();
    }
    if (forward <= 0.0 || strike <= 0.0 || t < 0.0 || params.alpha <= 0.0 ||
        params.beta < 0.0 || params.beta > 1.0 || params.nu < 0.0 ||
        params.rho <= -1.0 || params.rho >= 1.0) {
        return detail::nan_value();
    }

    double one_minus_beta = 1.0 - params.beta;
    double log_fk = std::log(forward / strike);
    double beta2 = one_minus_beta * one_minus_beta;
    double beta4 = beta2 * beta2;

    if (std::fabs(log_fk) < 1e-10) {
        double f_pow = std::pow(forward, one_minus_beta);
        double term1 = (beta2 / 24.0) * (params.alpha * params.alpha) / (f_pow * f_pow);
        double term2 = (params.rho * params.beta * params.nu * params.alpha) / (4.0 * f_pow);
        double term3 = ((2.0 - 3.0 * params.rho * params.rho) * params.nu * params.nu) / 24.0;
        return (params.alpha / f_pow) * (1.0 + (term1 + term2 + term3) * t);
    }

    double fk_pow = std::pow(forward * strike, 0.5 * one_minus_beta);
    double fk_pow_full = fk_pow * fk_pow; // pow(F*K, 1-beta)
    double z = (params.nu / params.alpha) * fk_pow * log_fk;

    double sqrt_arg = 1.0 - 2.0 * params.rho * z + z * z;
    if (sqrt_arg <= 0.0) return detail::nan_value();
    double x_z = std::log((std::sqrt(sqrt_arg) + z - params.rho) / (1.0 - params.rho));
    double z_over_x = (std::fabs(z) < 1e-10 || std::fabs(x_z) < 1e-10) ? 1.0 : (z / x_z);

    double log_fk2 = log_fk * log_fk;
    double log_fk4 = log_fk2 * log_fk2;
    double A = params.alpha / (fk_pow * (1.0 + beta2 * log_fk2 / 24.0 + beta4 * log_fk4 / 1920.0));

    double alpha2 = params.alpha * params.alpha;
    double nu2 = params.nu * params.nu;
    double term1 = (beta2 / 24.0) * alpha2 / fk_pow_full;
    double term2 = (params.rho * params.beta * params.nu * params.alpha) / (4.0 * fk_pow);
    double term3 = ((2.0 - 3.0 * params.rho * params.rho) * nu2) / 24.0;
    double B = 1.0 + (term1 + term2 + term3) * t;

    return A * z_over_x * B;
}

double sabr_hagan_black76_price(double forward, double strike, double t, double r,
                                const SABRParams& params, int32_t option_type) {
    double iv = sabr_hagan_lognormal_iv(forward, strike, t, params);
    if (!is_finite_safe(iv)) return detail::nan_value();
    return black76_price(forward, strike, t, iv, r, option_type);
}

} // namespace qk::cfa
