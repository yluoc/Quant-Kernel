#include "algorithms/finite_difference_methods/explicit_fd/explicit_fd.h"

#include "algorithms/finite_difference_methods/common/internal_util.h"

#include <cmath>
#include <vector>

namespace qk::fdm {

double explicit_fd_price(double spot, double strike, double t, double vol,
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

    // Enforce CFL stability: dt <= ds^2 / (vol^2 * S_max^2)
    double s_max = S[M];
    double dt_max = (ds * ds) / (vol * vol * s_max * s_max + detail::kEps);
    int32_t actual_time_steps = time_steps;
    double dt = t / static_cast<double>(actual_time_steps);
    if (dt > dt_max) {
        actual_time_steps = static_cast<int32_t>(std::ceil(t / dt_max)) + 1;
        dt = t / static_cast<double>(actual_time_steps);
    }

    std::vector<double> V(M + 1);
    for (int32_t i = 0; i <= M; ++i) {
        V[i] = detail::intrinsic_value(S[i], strike, option_type);
    }

    std::vector<double> V_new(M + 1);

    for (int32_t n = actual_time_steps - 1; n >= 0; --n) {
        double tau = (actual_time_steps - n) * dt;

        for (int32_t i = 1; i < M; ++i) {
            double Si = S[i];
            double sigma2 = vol * vol * Si * Si;
            double drift = (r - q) * Si;

            double a = 0.5 * dt * (sigma2 / (ds * ds) - drift / ds);
            double b = 1.0 - dt * (sigma2 / (ds * ds) + r);
            double c = 0.5 * dt * (sigma2 / (ds * ds) + drift / ds);

            V_new[i] = a * V[i - 1] + b * V[i] + c * V[i + 1];
        }

        V_new[0] = detail::lower_boundary(S[0], strike, r, q, tau, option_type);
        V_new[M] = detail::upper_boundary(S[M], strike, r, q, tau, option_type);

        if (american_style) {
            for (int32_t i = 1; i < M; ++i) {
                V_new[i] = std::max(V_new[i], detail::intrinsic_value(S[i], strike, option_type));
            }
        }

        std::swap(V, V_new);
    }

    return detail::interpolate_price(S, V, spot);
}

} // namespace qk::fdm
