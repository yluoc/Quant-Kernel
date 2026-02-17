#include "algorithms/fourier_transform_methods/hilbert_transform/hilbert_transform.h"

#include "algorithms/fourier_transform_methods/common/internal_util.h"

namespace qk::ftm {

double hilbert_transform_price(double spot, double strike, double t, double vol,
                               double r, double q, int32_t option_type,
                               const HilbertTransformParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(params.integration_limit)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) return detail::deterministic_price(spot, strike, t, r, q, option_type);

    if (params.integration_steps < 16 || params.integration_limit <= 1e-8) {
        return detail::nan_value();
    }

    std::complex<double> phi_minus_i = detail::bs_log_return_cf(std::complex<double>(0.0, -1.0),
                                                                 t, vol, r, q);
    if (std::abs(phi_minus_i) <= detail::kEps) return detail::nan_value();

    double x = std::log(strike / spot);

    auto p2_integrand = [&](double u) -> double {
        std::complex<double> cf = detail::bs_log_return_cf(std::complex<double>(u, 0.0), t, vol, r, q);
        std::complex<double> ratio = std::exp(-detail::kI * (u * x)) * cf
            / (detail::kI * u);
        return ratio.real();
    };

    auto p1_integrand = [&](double u) -> double {
        std::complex<double> cf = detail::bs_log_return_cf(std::complex<double>(u, -1.0), t, vol, r, q);
        std::complex<double> ratio = std::exp(-detail::kI * (u * x)) * cf
            / (detail::kI * u * phi_minus_i);
        return ratio.real();
    };

    double i_p2 = detail::integrate_trapezoid(p2_integrand, 1e-10, params.integration_limit,
                                              params.integration_steps);
    double i_p1 = detail::integrate_trapezoid(p1_integrand, 1e-10, params.integration_limit,
                                              params.integration_steps);

    double p1 = detail::clamp01(0.5 + i_p1 / detail::kPi);
    double p2 = detail::clamp01(0.5 + i_p2 / detail::kPi);

    double call_price = spot * std::exp(-q * t) * p1 - strike * std::exp(-r * t) * p2;
    call_price = std::max(0.0, call_price);

    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

} // namespace qk::ftm
