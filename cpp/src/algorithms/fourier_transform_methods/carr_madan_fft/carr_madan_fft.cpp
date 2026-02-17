#include "algorithms/fourier_transform_methods/carr_madan_fft/carr_madan_fft.h"

#include "algorithms/fourier_transform_methods/common/internal_util.h"

#include <cstddef>
#include <vector>

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
        std::complex<double> psi = std::exp(-r * t)
            * detail::bs_log_cf(arg, spot, t, vol, r, q) / den;

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
    if (!is_finite_safe(call_price)) return detail::nan_value();

    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

} // namespace qk::ftm
