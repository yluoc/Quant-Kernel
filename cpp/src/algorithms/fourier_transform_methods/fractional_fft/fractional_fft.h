#ifndef QK_FTM_FRACTIONAL_FFT_H
#define QK_FTM_FRACTIONAL_FFT_H

#include "algorithms/fourier_transform_methods/common/internal_util.h"
#include "algorithms/fourier_transform_methods/common/params.h"
#include "common/charfn_concepts.h"

#include <vector>

namespace qk::ftm {

/// Template implementation: Fractional FFT parameterized on CharFn.
/// `phi` is expected to be a log-price characteristic function (spot baked in).
template <typename CharFn>
double fractional_fft_price_impl(CharFn&& phi, double spot, double strike, double t,
                                 double r, double q, int32_t option_type,
                                 const FractionalFFTParams& params) {
    static_assert(is_charfn_v<std::decay_t<CharFn>>, "CharFn must satisfy is_charfn_v");

    int32_t n = params.grid_size;
    if (n < 16 || params.eta <= 0.0 || params.lambda <= 0.0 || params.alpha <= 0.0) {
        return detail::nan_value();
    }

    const double eta = params.eta;
    const double lambda = params.lambda;
    const double alpha = params.alpha;
    const double theta_fft = eta * lambda / (2.0 * detail::kPi);
    if (theta_fft <= 0.0 || theta_fft > 1.0) return detail::nan_value();

    const double b = 0.5 * static_cast<double>(n) * lambda;
    std::vector<std::complex<double>> x(static_cast<std::size_t>(n));

    for (int32_t j = 0; j < n; ++j) {
        const double v = static_cast<double>(j) * eta;
        const std::complex<double> den(alpha * alpha + alpha - v * v,
                                       (2.0 * alpha + 1.0) * v);
        const std::complex<double> arg(v, -(alpha + 1.0));
        const std::complex<double> psi = std::exp(-r * t) * phi(arg, t) / den;

        const double simpson = (j == 0) ? 1.0 : ((j % 2 == 0) ? 2.0 : 4.0);
        const std::complex<double> phase = std::exp(detail::kI * (b * v));
        x[static_cast<std::size_t>(j)] = phase * psi * (eta * simpson / 3.0);
    }

    std::vector<std::complex<double>> y;
    detail::bluestein_fractional_dft(x, y, theta_fft);

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
        return detail::nan_value();
    }
    call_price = std::max(lower, std::min(upper, call_price));
    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

/// BSM Fractional FFT (original interface, unchanged).
double fractional_fft_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            const FractionalFFTParams& params = {256, 0.25, 0.05, 1.5});

/// Heston Fractional FFT.
double fractional_fft_heston_price(double spot, double strike, double t,
                                   double r, double q,
                                   double v0, double kappa, double theta,
                                   double sigma, double rho,
                                   int32_t option_type,
                                   const FractionalFFTParams& params = {256, 0.25, 0.05, 1.5});

} // namespace qk::ftm

#endif /* QK_FTM_FRACTIONAL_FFT_H */
