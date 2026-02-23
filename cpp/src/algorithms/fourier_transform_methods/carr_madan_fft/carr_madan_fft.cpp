#include "algorithms/fourier_transform_methods/carr_madan_fft/carr_madan_fft.h"

#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"

namespace qk::ftm {

double carr_madan_fft_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            const CarrMadanFFTParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(params.eta) || !is_finite_safe(params.alpha)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) return detail::deterministic_price(spot, strike, t, r, q, option_type);
    if (t < 2e-3 || vol * std::sqrt(t) < 1e-3) {
        return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, option_type);
    }

    auto phi = make_bsm_log_charfn(spot, vol, r, q);
    double result = carr_madan_fft_price_impl(phi, spot, strike, t, r, q, option_type, params);
    if (!is_finite_safe(result)) {
        return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, option_type);
    }
    return result;
}

double carr_madan_fft_heston_price(double spot, double strike, double t,
                                   double r, double q,
                                   double v0, double kappa, double theta,
                                   double sigma, double rho,
                                   int32_t option_type,
                                   const CarrMadanFFTParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(v0) || !is_finite_safe(kappa) || !is_finite_safe(theta) ||
        !is_finite_safe(sigma) || !is_finite_safe(rho) ||
        !is_finite_safe(params.eta) || !is_finite_safe(params.alpha)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || v0 < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    auto phi = make_heston_log_charfn(spot, r, q, v0, kappa, theta, sigma, rho);
    return carr_madan_fft_price_impl(phi, spot, strike, t, r, q, option_type, params);
}

} // namespace qk::ftm
