#include "algorithms/finite_difference_methods/psor/psor.h"

#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/finite_difference_methods/common/internal_util.h"

#include <cmath>
#include <vector>

namespace qk::fdm {

double psor_price(double spot, double strike, double t, double vol,
                  double r, double q, int32_t option_type,
                  int32_t time_steps, int32_t spot_steps,
                  double omega, double tol, int32_t max_iter) {
    if (!detail::valid_inputs(spot, strike, t, vol, option_type) ||
        !detail::valid_grid_params(time_steps, spot_steps) ||
        !is_finite_safe(r) || !is_finite_safe(q))
        return detail::nan_value();
    if (omega < 1.0 || omega >= 2.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) {
        double fwd = spot * std::exp((r - q) * t);
        return std::exp(-r * t) * detail::intrinsic_value(fwd, strike, option_type);
    }

    double ds;
    std::vector<double> S = detail::build_spot_grid(spot, strike, vol, t, spot_steps, ds);
    int32_t M = spot_steps;
    double dt = t / static_cast<double>(time_steps);

    std::vector<double> V(M + 1);
    for (int32_t i = 0; i <= M; ++i) {
        V[i] = detail::intrinsic_value(S[i], strike, option_type);
    }

    std::vector<double> a_coeff(M + 1), b_coeff(M + 1), c_coeff(M + 1);
    for (int32_t i = 1; i < M; ++i) {
        double Si = S[i];
        double sigma2 = vol * vol * Si * Si;
        double drift = (r - q) * Si;

        a_coeff[i] = -0.5 * dt * (sigma2 / (ds * ds) - drift / ds);
        c_coeff[i] = -0.5 * dt * (sigma2 / (ds * ds) + drift / ds);
        b_coeff[i] = 1.0 + dt * (sigma2 / (ds * ds) + r);
    }

    std::vector<double> payoff(M + 1);
    for (int32_t i = 0; i <= M; ++i) {
        payoff[i] = detail::intrinsic_value(S[i], strike, option_type);
    }

    for (int32_t step = time_steps - 1; step >= 0; --step) {
        double tau = (time_steps - step) * dt;

        double bc_lower = detail::lower_boundary(S[0], strike, r, q, tau, option_type);
        double bc_upper = detail::upper_boundary(S[M], strike, r, q, tau, option_type);

        std::vector<double> rhs(M + 1);
        for (int32_t i = 1; i < M; ++i) {
            rhs[i] = V[i];
        }

        for (int32_t iter = 0; iter < max_iter; ++iter) {
            double max_change = 0.0;

            V[0] = bc_lower;
            V[M] = bc_upper;

            for (int32_t i = 1; i < M; ++i) {
                double gs = (rhs[i] - a_coeff[i] * V[i - 1] - c_coeff[i] * V[i + 1]) / b_coeff[i];
                double v_new = V[i] + omega * (gs - V[i]);

                v_new = std::max(v_new, payoff[i]);

                double change = std::fabs(v_new - V[i]);
                if (change > max_change) max_change = change;
                V[i] = v_new;
            }

            if (max_change < tol) break;
        }

        V[0] = bc_lower;
        V[M] = bc_upper;
    }

    double price = detail::interpolate_price(S, V, spot);
    if (option_type == QK_PUT) {
        const double eur_bsm = qk::cfa::black_scholes_merton_price(
            spot, strike, t, vol, r, q, QK_PUT
        );
        if (is_finite_safe(eur_bsm)) {
            price = std::max(price, eur_bsm);
        }
    }
    return price;
}

} // namespace qk::fdm
