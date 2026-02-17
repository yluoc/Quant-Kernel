#include "algorithms/fourier_transform_methods/cos_method/cos_method.h"

#include "algorithms/fourier_transform_methods/common/internal_util.h"

namespace qk::ftm {

double cos_method_fang_oosterlee_price(double spot, double strike, double t, double vol,
                                       double r, double q, int32_t option_type,
                                       const COSMethodParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(params.truncation_width)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) return detail::deterministic_price(spot, strike, t, r, q, option_type);

    int32_t n = params.n_terms;
    if (n < 8 || params.truncation_width <= 0.0) return detail::nan_value();

    double vol2 = vol * vol;
    double c1 = std::log(spot) + (r - q - 0.5 * vol2) * t;
    double c2 = vol2 * t;
    double a = c1 - params.truncation_width * std::sqrt(std::max(c2, 0.0));
    double b = c1 + params.truncation_width * std::sqrt(std::max(c2, 0.0));
    if (!(b > a + detail::kEps)) return detail::nan_value();

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
            std::complex<double> phi = detail::bs_log_cf(std::complex<double>(u, 0.0), spot, t, vol, r, q);
            double f_k = (phi * std::exp(-detail::kI * (u * a))).real();
            double weight = (k == 0) ? 0.5 : 1.0;
            sum += weight * f_k * u_k;
        }
        call_price = std::exp(-r * t) * sum;
    }

    call_price = std::max(0.0, call_price);
    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

} // namespace qk::ftm
