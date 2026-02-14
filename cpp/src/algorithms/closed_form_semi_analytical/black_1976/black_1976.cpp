#include "algorithms/closed_form_semi_analytical/black_1976/black_1976.h"

#include "algorithms/closed_form_semi_analytical/common/internal_util.h"

namespace qk::cfa {

double black76_price(double forward, double strike, double t, double vol,
                     double r, int32_t option_type) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(forward) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r)) {
        return detail::nan_value();
    }
    if (forward <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(forward, strike, option_type);

    double df = std::exp(-r * t);
    if (vol <= detail::kEps) {
        return df * detail::intrinsic_value(forward, strike, option_type);
    }

    double sqrt_t = std::sqrt(t);
    double d1 = (std::log(forward / strike) + 0.5 * vol * vol * t) / (vol * sqrt_t);
    double d2 = d1 - vol * sqrt_t;

    if (option_type == QK_CALL) {
        return df * (forward * norm_cdf(d1) - strike * norm_cdf(d2));
    }
    return df * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1));
}

} // namespace qk::cfa
