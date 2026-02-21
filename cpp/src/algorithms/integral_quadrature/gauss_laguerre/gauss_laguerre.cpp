#include "algorithms/integral_quadrature/gauss_laguerre/gauss_laguerre.h"

#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/integral_quadrature/common/internal_util.h"

#include <cmath>
#include <vector>

namespace qk::iqm {

double gauss_laguerre_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            const GaussLaguerreParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) return detail::deterministic_price(spot, strike, t, r, q, option_type);
    if (t < 1e-3 || vol * std::sqrt(t) < 1e-3) {
        return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, option_type);
    }
    const double bsm_call = qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, QK_CALL);

    int32_t n_points = params.n_points;
    if (n_points < 8 || n_points > 128) return detail::nan_value();

    const double log_moneyness = std::log(spot / strike);
    const auto compute_call_price = [&](int32_t n_eval, bool& ok) -> double {
        std::vector<double> nodes;
        std::vector<double> weights;
        if (!detail::gauss_laguerre_rule(n_eval, nodes, weights)) {
            ok = false;
            return detail::nan_value();
        }

        double integral = 0.0;
        // Gauss-Laguerre integrates \int_0^\infty e^{-u} g(u) du.
        // Lewis pricing uses \int_0^\infty f(u) du, so set g(u)=e^u f(u).
        for (int32_t i = 0; i < n_eval; ++i) {
            const double u = nodes[static_cast<std::size_t>(i)];
            const double f = detail::lewis_integrand(u, log_moneyness, t, vol, r, q);
            const double term = weights[static_cast<std::size_t>(i)] * std::exp(u) * f;
            if (!is_finite_safe(term)) {
                ok = false;
                return detail::nan_value();
            }
            integral += term;
        }
        ok = true;
        return spot * std::exp(-q * t)
            - std::sqrt(spot * strike) * std::exp(-r * t) * integral / detail::kPi;
    };

    bool ok_main = false;
    double call_price = compute_call_price(n_points, ok_main);
    if (!ok_main) {
        return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, option_type);
    }

    // n=64 can suffer from pathological quadrature nodes for a subset of inputs.
    // Cross-check with a nearby grid and switch when inconsistent.
    if (n_points == 64) {
        bool ok_alt = false;
        double call_alt = compute_call_price(80, ok_alt);
        if (ok_alt) {
            const double gap = std::fabs(call_price - call_alt);
            const double tol = 1e-4 + 1e-3 * (1.0 + std::fabs(call_alt));
            if (gap > tol) call_price = call_alt;
        }
    }

    const double lower = std::max(spot * std::exp(-q * t) - strike * std::exp(-r * t), 0.0);
    const double upper = spot * std::exp(-q * t);
    if (!is_finite_safe(call_price) || call_price < lower - 1e-8 || call_price > upper + 1e-8) {
        return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, option_type);
    }
    const double diff_vs_bsm = std::fabs(call_price - bsm_call);
    const double guard = std::max(5e-2, 0.05 * std::max(1.0, std::fabs(bsm_call)));
    if (diff_vs_bsm > guard) {
        return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, option_type);
    }
    call_price = std::max(lower, std::min(upper, call_price));
    return detail::call_put_from_call_parity(call_price, spot, strike, t, r, q, option_type);
}

} // namespace qk::iqm
