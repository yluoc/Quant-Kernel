#include "algorithms/finite_difference_methods/implicit_fd/implicit_fd.h"

#include "algorithms/finite_difference_methods/common/internal_util.h"

#include <cmath>
#include <vector>

namespace qk::fdm {

double implicit_fd_price(double spot, double strike, double t, double vol,
                         double r, double q, int32_t option_type,
                         int32_t time_steps, int32_t spot_steps,
                         bool american_style) {
    if (!detail::valid_inputs(spot, strike, t, vol, option_type) ||
        !detail::valid_grid_params(time_steps, spot_steps) ||
        !is_finite_safe(r) || !is_finite_safe(q))
        return detail::nan_value();
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

    int32_t n_interior = M - 1;
    std::vector<double> a_sub(n_interior), b_diag(n_interior), c_sup(n_interior), d_rhs(n_interior);

    for (int32_t step = time_steps - 1; step >= 0; --step) {
        double tau = (time_steps - step) * dt;

        for (int32_t j = 0; j < n_interior; ++j) {
            int32_t i = j + 1;
            double Si = S[i];
            double sigma2 = vol * vol * Si * Si;
            double drift = (r - q) * Si;

            double alpha = 0.5 * dt * (sigma2 / (ds * ds) - drift / ds);
            double beta  = 0.5 * dt * (sigma2 / (ds * ds) + drift / ds);
            double gamma = dt * (sigma2 / (ds * ds) + r);

            a_sub[j]  = -alpha;
            b_diag[j] = 1.0 + gamma;
            c_sup[j]  = -beta;
            d_rhs[j]  = V[i];
        }

        double bc_lower = detail::lower_boundary(S[0], strike, r, q, tau, option_type);
        double bc_upper = detail::upper_boundary(S[M], strike, r, q, tau, option_type);
        {
            double S1 = S[1];
            double sigma2 = vol * vol * S1 * S1;
            double drift = (r - q) * S1;
            double alpha = 0.5 * dt * (sigma2 / (ds * ds) - drift / ds);
            d_rhs[0] += alpha * bc_lower;
        }
        {
            double Sm1 = S[M - 1];
            double sigma2 = vol * vol * Sm1 * Sm1;
            double drift = (r - q) * Sm1;
            double beta = 0.5 * dt * (sigma2 / (ds * ds) + drift / ds);
            d_rhs[n_interior - 1] += beta * bc_upper;
        }

        detail::thomas_solve(a_sub.data(), b_diag.data(), c_sup.data(), d_rhs.data(), n_interior);

        V[0] = bc_lower;
        for (int32_t j = 0; j < n_interior; ++j) {
            V[j + 1] = d_rhs[j];
        }
        V[M] = bc_upper;

        if (american_style) {
            for (int32_t i = 1; i < M; ++i) {
                V[i] = std::max(V[i], detail::intrinsic_value(S[i], strike, option_type));
            }
        }
    }

    return detail::interpolate_price(S, V, spot);
}

} // namespace qk::fdm
