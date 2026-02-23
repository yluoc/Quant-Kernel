#include "algorithms/fourier_transform_methods/cos_method/cos_method.h"

namespace qk::ftm {

double cos_method_fang_oosterlee_price(double spot, double strike, double t, double vol,
                                       double r, double q, int32_t option_type,
                                       const COSMethodParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(params.truncation_width)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) return detail::deterministic_price(spot, strike, t, r, q, option_type);

    if (params.n_terms < 8 || params.truncation_width <= 0.0) return detail::nan_value();

    double vol2 = vol * vol;
    double c1 = std::log(spot) + (r - q - 0.5 * vol2) * t;
    double c2 = vol2 * t;
    double a = c1 - params.truncation_width * std::sqrt(std::max(c2, 0.0));
    double b = c1 + params.truncation_width * std::sqrt(std::max(c2, 0.0));

    auto phi = make_bsm_log_charfn(spot, vol, r, q);
    return cos_method_fang_oosterlee_price_impl(phi, spot, strike, t, r, q, option_type, a, b, params);
}

double cos_method_fang_oosterlee_heston_price(double spot, double strike, double t,
                                              double r, double q,
                                              double v0, double kappa, double theta,
                                              double sigma, double rho,
                                              int32_t option_type,
                                              const COSMethodParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(v0) || !is_finite_safe(kappa) || !is_finite_safe(theta) ||
        !is_finite_safe(sigma) || !is_finite_safe(rho) ||
        !is_finite_safe(params.truncation_width)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || v0 < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    if (params.n_terms < 8 || params.truncation_width <= 0.0) return detail::nan_value();

    // For Heston, use long-run variance theta for the truncation interval estimate.
    double eff_var = std::max(v0, theta);
    double c1 = std::log(spot) + (r - q - 0.5 * eff_var) * t;
    double c2 = eff_var * t;
    double a = c1 - params.truncation_width * std::sqrt(std::max(c2, 0.0));
    double b = c1 + params.truncation_width * std::sqrt(std::max(c2, 0.0));

    auto phi = make_heston_log_charfn(spot, r, q, v0, kappa, theta, sigma, rho);
    return cos_method_fang_oosterlee_price_impl(phi, spot, strike, t, r, q, option_type, a, b, params);
}

} // namespace qk::ftm
