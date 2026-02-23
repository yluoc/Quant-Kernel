#ifndef QK_FTM_COS_METHOD_H
#define QK_FTM_COS_METHOD_H

#include "algorithms/fourier_transform_methods/common/internal_util.h"
#include "algorithms/fourier_transform_methods/common/params.h"
#include "common/charfn_concepts.h"

namespace qk::ftm {

/// Template implementation: COS method parameterized on CharFn.
/// `phi` is expected to be a log-price characteristic function (spot baked in).
/// The caller must supply the truncation interval [a, b].
template <typename CharFn>
double cos_method_fang_oosterlee_price_impl(CharFn&& phi, double spot, double strike,
                                            double t, double r, double q,
                                            int32_t option_type,
                                            double a, double b,
                                            const COSMethodParams& params) {
    static_assert(is_charfn_v<std::decay_t<CharFn>>, "CharFn must satisfy is_charfn_v");

    int32_t n = params.n_terms;
    if (n < 8 || !(b > a + detail::kEps)) return detail::nan_value();

    double log_strike = std::log(strike);
    double c = std::min(std::max(log_strike, a), b);
    double d = b;

    double call_price = 0.0;
    if (c < d) {
        double interval = b - a;
        double inv_interval = 1.0 / interval;
        double sum = 0.0;

        for (int32_t k = 0; k < n; ++k) {
            double u = static_cast<double>(k) * detail::kPi * inv_interval;

            double chi = 0.0;
            double psi = 0.0;
            if (k == 0) {
                chi = std::exp(d) - std::exp(c);
                psi = d - c;
            } else {
                double ud = u * (d - a);
                double uc = u * (c - a);
                double exp_d = std::exp(d);
                double exp_c = std::exp(c);
                chi = ((std::cos(ud) * exp_d - std::cos(uc) * exp_c)
                    + u * (std::sin(ud) * exp_d - std::sin(uc) * exp_c)) / (1.0 + u * u);
                psi = (std::sin(ud) - std::sin(uc)) / u;
            }

            double u_k = 2.0 * inv_interval * (chi - strike * psi);
            std::complex<double> phi_val = phi(std::complex<double>(u, 0.0), t);
            double f_k = (phi_val * std::exp(-detail::kI * (u * a))).real();
            double weight = (k == 0) ? 0.5 : 1.0;
            sum += weight * f_k * u_k;
        }
        call_price = std::exp(-r * t) * sum;
    }

    call_price = std::max(0.0, call_price);
    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

/// BSM COS method (original interface, unchanged).
double cos_method_fang_oosterlee_price(double spot, double strike, double t, double vol,
                                       double r, double q, int32_t option_type,
                                       const COSMethodParams& params = {256, 10.0});

/// Heston COS method.
double cos_method_fang_oosterlee_heston_price(double spot, double strike, double t,
                                              double r, double q,
                                              double v0, double kappa, double theta,
                                              double sigma, double rho,
                                              int32_t option_type,
                                              const COSMethodParams& params = {256, 10.0});

} // namespace qk::ftm

#endif /* QK_FTM_COS_METHOD_H */
