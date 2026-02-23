#ifndef QK_FTM_LEWIS_FOURIER_INVERSION_H
#define QK_FTM_LEWIS_FOURIER_INVERSION_H

#include "algorithms/fourier_transform_methods/common/internal_util.h"
#include "algorithms/fourier_transform_methods/common/params.h"
#include "common/charfn_concepts.h"

namespace qk::ftm {

/// Template implementation: Lewis Fourier inversion parameterized on CharFn.
/// `phi` is expected to be a log-RETURN characteristic function (no spot baked in).
template <typename CharFn>
double lewis_fourier_inversion_price_impl(CharFn&& phi, double spot, double strike,
                                          double t, double r, double q,
                                          int32_t option_type,
                                          const LewisFourierInversionParams& params) {
    static_assert(is_charfn_v<std::decay_t<CharFn>>, "CharFn must satisfy is_charfn_v");

    if (params.integration_steps < 16 || params.integration_limit <= 1e-8) {
        return detail::nan_value();
    }

    double x = std::log(spot / strike);
    auto integrand = [&](double u) -> double {
        std::complex<double> arg(u, -0.5);
        std::complex<double> cf = phi(arg, t);
        std::complex<double> numerator = std::exp(detail::kI * (u * x)) * cf;
        return (numerator / (u * u + 0.25)).real();
    };

    double integral = detail::integrate_trapezoid(integrand, 1e-10, params.integration_limit,
                                                  params.integration_steps);

    double call_price = spot * std::exp(-q * t)
        - std::sqrt(spot * strike) * std::exp(-r * t) * integral / detail::kPi;
    call_price = std::max(0.0, call_price);

    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

/// BSM Lewis Fourier inversion (original interface, unchanged).
double lewis_fourier_inversion_price(double spot, double strike, double t, double vol,
                                     double r, double q, int32_t option_type,
                                     const LewisFourierInversionParams& params = {4096, 300.0});

/// Heston Lewis Fourier inversion.
double lewis_fourier_inversion_heston_price(double spot, double strike, double t,
                                            double r, double q,
                                            double v0, double kappa, double theta,
                                            double sigma, double rho,
                                            int32_t option_type,
                                            const LewisFourierInversionParams& params = {4096, 300.0});

} // namespace qk::ftm

#endif /* QK_FTM_LEWIS_FOURIER_INVERSION_H */
