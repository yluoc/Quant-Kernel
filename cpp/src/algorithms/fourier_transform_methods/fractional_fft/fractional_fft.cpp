#include "algorithms/fourier_transform_methods/fractional_fft/fractional_fft.h"

#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/fourier_transform_methods/common/internal_util.h"

#include <vector>

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
    if (t < 2e-3 || vol * std::sqrt(t) < 1e-3) {
        return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, option_type);
    }

    int32_t n = params.grid_size;
    if (n < 16 || params.eta <= 0.0 || params.lambda <= 0.0 || params.alpha <= 0.0) {
        return detail::nan_value();
    }

    const double eta = params.eta;
    const double lambda = params.lambda;
    const double alpha = params.alpha;
    const double theta = eta * lambda / (2.0 * detail::kPi);
    if (theta <= 0.0 || theta > 1.0) return detail::nan_value();

    const double b = 0.5 * static_cast<double>(n) * lambda;
    std::vector<std::complex<double>> x(static_cast<std::size_t>(n));

    for (int32_t j = 0; j < n; ++j) {
        const double v = static_cast<double>(j) * eta;
        const std::complex<double> den(alpha * alpha + alpha - v * v,
                                       (2.0 * alpha + 1.0) * v);
        const std::complex<double> arg(v, -(alpha + 1.0));
        const std::complex<double> psi = std::exp(-r * t)
            * detail::bs_log_cf(arg, spot, t, vol, r, q) / den;

        const double simpson = (j == 0) ? 1.0 : ((j % 2 == 0) ? 2.0 : 4.0);
        const std::complex<double> phase = std::exp(detail::kI * (b * v));
        x[static_cast<std::size_t>(j)] = phase * psi * (eta * simpson / 3.0);
    }

    std::vector<std::complex<double>> y;
    detail::bluestein_fractional_dft(x, y, theta);

    std::vector<double> k_grid(static_cast<std::size_t>(n));
    std::vector<double> call_grid(static_cast<std::size_t>(n));
    for (int32_t m = 0; m < n; ++m) {
        const double k = -b + static_cast<double>(m) * lambda;
        const double call = std::exp(-alpha * k) * y[static_cast<std::size_t>(m)].real() / detail::kPi;
        k_grid[static_cast<std::size_t>(m)] = k;
        call_grid[static_cast<std::size_t>(m)] = std::max(0.0, call);
    }

    double call_price = detail::linear_interpolate(k_grid, call_grid, std::log(strike));
    const double lower = std::max(spot * std::exp(-q * t) - strike * std::exp(-r * t), 0.0);
    const double upper = spot * std::exp(-q * t);
    if (!is_finite_safe(call_price) || call_price < lower - 1e-8 || call_price > upper + 1e-8) {
        return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, option_type);
    }
    call_price = std::max(lower, std::min(upper, call_price));
    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

} // namespace qk::ftm
