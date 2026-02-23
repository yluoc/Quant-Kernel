#ifndef QK_FTM_CARR_MADAN_FFT_H
#define QK_FTM_CARR_MADAN_FFT_H

#include "algorithms/fourier_transform_methods/common/internal_util.h"
#include "algorithms/fourier_transform_methods/common/params.h"
#include "common/charfn_concepts.h"

#include <cstddef>
#include <vector>

namespace qk::ftm {

/// Template implementation: Carr-Madan FFT parameterized on CharFn.
/// CharFn must satisfy is_charfn_v (callable: (complex<double> u, double t) -> complex<double>).
/// `phi` is expected to be a log-price characteristic function (spot baked in).
template <typename CharFn>
double carr_madan_fft_price_impl(CharFn&& phi, double spot, double strike, double t,
                                 double r, double q, int32_t option_type,
                                 const CarrMadanFFTParams& params) {
    static_assert(is_charfn_v<std::decay_t<CharFn>>, "CharFn must satisfy is_charfn_v");

    int32_t n = params.grid_size;
    if (n < 16 || !detail::is_power_of_two(n) || params.eta <= 0.0 || params.alpha <= 0.0) {
        return detail::nan_value();
    }

    double eta = params.eta;
    double alpha = params.alpha;
    double lambda = 2.0 * detail::kPi / (static_cast<double>(n) * eta);
    double b = 0.5 * static_cast<double>(n) * lambda;

    std::vector<std::complex<double>> input(static_cast<std::size_t>(n));

    for (int32_t j = 0; j < n; ++j) {
        double v = static_cast<double>(j) * eta;
        std::complex<double> den(alpha * alpha + alpha - v * v,
                                 (2.0 * alpha + 1.0) * v);
        std::complex<double> arg(v, -(alpha + 1.0));
        std::complex<double> psi = std::exp(-r * t) * phi(arg, t) / den;

        double simpson = (j == 0) ? 1.0 : ((j % 2 == 0) ? 2.0 : 4.0);
        std::complex<double> phase = std::exp(detail::kI * (b * v));
        input[static_cast<std::size_t>(j)] = phase * psi * (eta * simpson / 3.0);
    }

    detail::fft_inplace(input);

    std::vector<double> k_grid(static_cast<std::size_t>(n));
    std::vector<double> call_grid(static_cast<std::size_t>(n));

    for (int32_t m = 0; m < n; ++m) {
        double k = -b + static_cast<double>(m) * lambda;
        double call = std::exp(-alpha * k) * input[static_cast<std::size_t>(m)].real() / detail::kPi;
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

/// BSM Carr-Madan FFT (original interface, unchanged).
double carr_madan_fft_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            const CarrMadanFFTParams& params = {4096, 0.25, 1.5});

/// Heston Carr-Madan FFT.
double carr_madan_fft_heston_price(double spot, double strike, double t,
                                   double r, double q,
                                   double v0, double kappa, double theta,
                                   double sigma, double rho,
                                   int32_t option_type,
                                   const CarrMadanFFTParams& params = {4096, 0.25, 1.5});

} // namespace qk::ftm

#endif /* QK_FTM_CARR_MADAN_FFT_H */
