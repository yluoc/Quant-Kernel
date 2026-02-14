#include "algorithms/closed_form_semi_analytical/bachelier/bachelier.h"

#include "algorithms/closed_form_semi_analytical/common/internal_util.h"

namespace qk::cfa {

double bachelier_price(double forward, double strike, double t, double normal_vol,
                       double r, int32_t option_type) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(forward) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(normal_vol) || !is_finite_safe(r)) {
        return detail::nan_value();
    }
    if (t < 0.0 || normal_vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(forward, strike, option_type);

    double df = std::exp(-r * t);
    double stddev = normal_vol * std::sqrt(t);
    if (stddev <= detail::kEps) {
        return df * detail::intrinsic_value(forward, strike, option_type);
    }

    double d = (forward - strike) / stddev;
    if (option_type == QK_CALL) {
        return df * ((forward - strike) * norm_cdf(d) + stddev * norm_pdf(d));
    }
    return df * ((strike - forward) * norm_cdf(-d) + stddev * norm_pdf(d));
}

} // namespace qk::cfa
