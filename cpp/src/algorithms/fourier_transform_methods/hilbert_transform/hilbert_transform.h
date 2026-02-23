#ifndef QK_FTM_HILBERT_TRANSFORM_H
#define QK_FTM_HILBERT_TRANSFORM_H

#include "algorithms/fourier_transform_methods/common/internal_util.h"
#include "algorithms/fourier_transform_methods/common/params.h"
#include "common/charfn_concepts.h"

namespace qk::ftm {

/// Template implementation: Hilbert transform parameterized on CharFn.
/// `phi` is expected to be a log-RETURN characteristic function (no spot baked in).
template <typename CharFn>
double hilbert_transform_price_impl(CharFn&& phi, double spot, double strike,
                                    double t, double r, double q,
                                    int32_t option_type,
                                    const HilbertTransformParams& params) {
    static_assert(is_charfn_v<std::decay_t<CharFn>>, "CharFn must satisfy is_charfn_v");

    if (params.integration_steps < 16 || params.integration_limit <= 1e-8) {
        return detail::nan_value();
    }

    std::complex<double> phi_minus_i = phi(std::complex<double>(0.0, -1.0), t);
    if (std::abs(phi_minus_i) <= detail::kEps) return detail::nan_value();

    double x = std::log(strike / spot);

    auto p2_integrand = [&](double u) -> double {
        std::complex<double> cf = phi(std::complex<double>(u, 0.0), t);
        std::complex<double> ratio = std::exp(-detail::kI * (u * x)) * cf
            / (detail::kI * u);
        return ratio.real();
    };

    auto p1_integrand = [&](double u) -> double {
        std::complex<double> cf = phi(std::complex<double>(u, -1.0), t);
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

/// BSM Hilbert transform (original interface, unchanged).
double hilbert_transform_price(double spot, double strike, double t, double vol,
                               double r, double q, int32_t option_type,
                               const HilbertTransformParams& params = {4096, 300.0});

/// Heston Hilbert transform.
double hilbert_transform_heston_price(double spot, double strike, double t,
                                      double r, double q,
                                      double v0, double kappa, double theta,
                                      double sigma, double rho,
                                      int32_t option_type,
                                      const HilbertTransformParams& params = {4096, 300.0});

} // namespace qk::ftm

#endif /* QK_FTM_HILBERT_TRANSFORM_H */
