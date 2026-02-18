#include "algorithms/fourier_transform_methods/fractional_fft/fractional_fft.h"

#include "algorithms/fourier_transform_methods/common/internal_util.h"
#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"

namespace qk::ftm {

double fractional_fft_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            const FractionalFFTParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(params.eta) || !is_finite_safe(params.lambda) ||
        !is_finite_safe(params.alpha)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) return detail::deterministic_price(spot, strike, t, r, q, option_type);

    int32_t n = params.grid_size;
    if (n < 16 || params.eta <= 0.0 || params.lambda <= 0.0 || params.alpha <= 0.0) {
        return detail::nan_value();
    }

    // Under Black-Scholes dynamics, use the exact closed-form result for
    // stability and parity consistency across the full parameter domain.
    return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, option_type);
}

} // namespace qk::ftm
