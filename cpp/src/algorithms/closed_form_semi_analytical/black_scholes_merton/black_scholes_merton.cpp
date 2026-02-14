#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"

#include "algorithms/closed_form_semi_analytical/common/internal_util.h"

namespace qk::cfa {

double black_scholes_merton_price(double spot, double strike, double t, double vol,
                                  double r, double q, int32_t option_type) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    double sqrt_t = std::sqrt(t);
    if (vol <= detail::kEps) {
        double forward = spot * std::exp((r - q) * t);
        return std::exp(-r * t) * detail::intrinsic_value(forward, strike, option_type);
    }

    double d1 = (std::log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t);
    double d2 = d1 - vol * sqrt_t;
    double df = std::exp(-r * t);
    double qf = std::exp(-q * t);

    if (option_type == QK_CALL) {
        return spot * qf * norm_cdf(d1) - strike * df * norm_cdf(d2);
    }
    return strike * df * norm_cdf(-d2) - spot * qf * norm_cdf(-d1);
}

} // namespace qk::cfa
