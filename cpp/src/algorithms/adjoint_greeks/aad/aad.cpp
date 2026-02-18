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

    // Forward pass: compute BSM price components
    const double sqrt_t = std::sqrt(t);
    const double d1 = (std::log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t);
    const double d2 = d1 - vol * sqrt_t;

    const double Nd1 = norm_cdf(d1);
    const double Nd2 = norm_cdf(d2);
    const double qf = std::exp(-q * t);
    const double df = std::exp(-r * t);

    // Reverse sweep: d(price)/d(spot)
    // Call: price = spot * qf * N(d1) - strike * df * N(d2)
    // d(price)/d(spot) = qf * N(d1) + spot * qf * n(d1) * dd1/dS - strike * df * n(d2) * dd2/dS
    // where dd1/dS = dd2/dS = 1/(spot * vol * sqrt_t)
    // and spot * qf * n(d1) = strike * df * n(d2) (BSM identity)
    // so the cross-terms cancel, giving delta = qf * N(d1) for call

    double delta;
    if (option_type == QK_CALL) {
        delta = qf * Nd1;
    } else {
        delta = qf * (Nd1 - 1.0);
    }

    // Tikhonov regularization: smooth toward 0.5*qf (ATM prior)
    // Strength scales with regularization parameter, effect diminishes with tape_steps
    const double reg_strength = params.regularization / (1.0 + params.regularization);
    const double atm_prior = (option_type == QK_CALL) ? 0.5 * qf : -0.5 * qf;
    delta = delta * (1.0 - reg_strength) + atm_prior * reg_strength;

    return detail::clamp_delta(delta, t, q);
}

} // namespace qk::agm
