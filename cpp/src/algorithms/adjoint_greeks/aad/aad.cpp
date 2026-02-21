#include "algorithms/adjoint_greeks/aad/aad.h"

#include "algorithms/adjoint_greeks/common/internal_util.h"

#include <algorithm>
#include <cmath>

namespace qk::agm {

double aad_delta(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const AadParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.tape_steps < 4 || !is_finite_safe(params.regularization) || params.regularization < 0.0) {
        return detail::nan_value();
    }

    if (t <= detail::kEps || vol <= detail::kEps) {
        return detail::deterministic_delta(spot, strike, t, r, q, option_type);
    }

    const double sqrt_t = std::sqrt(t);
    const double vol2 = vol * vol;
    const double log_m = std::log(spot / strike);
    const double num = log_m + (r - q + 0.5 * vol2) * t;
    const double denom = vol * sqrt_t;
    const double d1 = num / denom;
    const double d2 = d1 - denom;
    const double Nd1 = norm_cdf(d1);
    const double Nd2 = norm_cdf(d2);
    const double qf = std::exp(-q * t);
    const double df = std::exp(-r * t);
    const double call_price = spot * qf * Nd1 - strike * df * Nd2;

    double a_call = 1.0;
    double a_term1 = a_call;
    double a_term2 = -a_call;

    double a_spot = a_term1 * qf * Nd1;
    double a_Nd1 = a_term1 * spot * qf;
    double a_Nd2 = a_term2 * strike * df;

    double a_d1 = a_Nd1 * norm_pdf(d1);
    double a_d2 = a_Nd2 * norm_pdf(d2);

    a_d1 += a_d2;
    double a_denom = -a_d2;

    double inv_denom = 1.0 / denom;
    a_denom += a_d1 * (-num) * inv_denom * inv_denom;
    double a_num = a_d1 * inv_denom;

    double a_log_m = a_num;
    a_spot += a_log_m / spot;

    double delta_call = a_spot;
    if (!is_finite_safe(delta_call) || !is_finite_safe(call_price)) return detail::nan_value();

    double delta = (option_type == QK_CALL) ? delta_call : (delta_call - qf);

    // Tikhonov regularization toward ATM prior, weakened with larger tape depth.
    const double tape_scale = 1.0 / std::sqrt(static_cast<double>(params.tape_steps));
    const double lambda = params.regularization * tape_scale;
    const double reg_strength = lambda / (1.0 + lambda);
    const double atm_prior = (option_type == QK_CALL) ? 0.5 * qf : -0.5 * qf;
    delta = delta * (1.0 - reg_strength) + atm_prior * reg_strength;

    return detail::clamp_delta(delta, t, q);
}

} // namespace qk::agm
