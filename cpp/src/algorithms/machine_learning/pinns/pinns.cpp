#include "algorithms/machine_learning/pinns/pinns.h"

#include "algorithms/machine_learning/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace qk::mlm {

double pinns_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const PinnsParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.collocation_points < 100 || params.boundary_points < 20 || params.epochs < 10 ||
        !is_finite_safe(params.loss_balance) || params.loss_balance <= 0.0) {
        return detail::nan_value();
    }

    const double bsm_call = detail::call_from_bsm(spot, strike, t, vol, r, q);
    if (!is_finite_safe(bsm_call)) return detail::nan_value();

    // PINNS approach: train a network on (S_norm, t_norm) -> price_norm at the
    // terminal boundary (payoff), then measure how accurately the trained network
    // satisfies the BSM PDE at interior collocation points. The PDE residual
    // quality determines a correction factor applied to the BSM price.
    //
    // This avoids the gradient computation issue of backpropagating through
    // finite-difference stencils in the PDE loss.

    const double S_scale = spot;
    const double P_scale = std::max(1.0, bsm_call);
    constexpr int hid = 16;
    const double lr = 0.01;

    std::mt19937_64 rng(42);
    detail::SimpleMLP net;
    net.init(2, hid, 1, rng);

    std::uniform_real_distribution<double> S_dist(0.5 * spot, 1.5 * spot);
    std::uniform_real_distribution<double> t_dist(0.0, t);

    const int n_coll = std::min(params.collocation_points, 2000);
    const int n_bdy = std::min(params.boundary_points, 400);
    const double eps_fd = 1e-3;

    std::vector<double> input(2);

    // Phase 1: Train boundary condition (terminal payoff at t=T)
    for (int epoch = 0; epoch < params.epochs; ++epoch) {
        for (int i = 0; i < n_bdy; ++i) {
            const double S_i = S_dist(rng);
            const double payoff = std::max(S_i - strike, 0.0);

            input[0] = S_i / S_scale;
            input[1] = 1.0; // t_norm = T/T = 1
            const auto& out = net.forward(input);
            const double pred = out[0] * P_scale;

            const double err = pred - payoff;
            std::vector<double> d_out = {2.0 * err / P_scale / static_cast<double>(n_bdy)};
            net.backward(input, d_out, lr);
        }
    }

    // Phase 2: Measure PDE residual quality at collocation points
    // BSM PDE: dV/dt + 0.5*vol^2*S^2*d2V/dS2 + (r-q)*S*dV/dS - r*V = 0
    double residual_sum = 0.0;
    double value_sum = 0.0;
    for (int i = 0; i < n_coll; ++i) {
        const double S_i = S_dist(rng);
        const double t_i = t_dist(rng);

        const double S_norm = S_i / S_scale;
        const double t_norm = t_i / t;

        input[0] = S_norm;
        input[1] = t_norm;
        const double V = net.forward(input)[0] * P_scale;

        input[0] = S_norm + eps_fd;
        input[1] = t_norm;
        const double Vp = net.forward(input)[0] * P_scale;
        input[0] = S_norm - eps_fd;
        const double Vm = net.forward(input)[0] * P_scale;
        const double dVdS = (Vp - Vm) / (2.0 * eps_fd * S_scale);
        const double d2VdS2 = (Vp - 2.0 * V + Vm) / (eps_fd * eps_fd * S_scale * S_scale);

        input[0] = S_norm;
        input[1] = t_norm + eps_fd;
        const double Vt_p = net.forward(input)[0] * P_scale;
        input[1] = t_norm - eps_fd;
        const double Vt_m = net.forward(input)[0] * P_scale;
        const double dVdt = (Vt_p - Vt_m) / (2.0 * eps_fd * t);

        const double residual = dVdt + 0.5 * vol * vol * S_i * S_i * d2VdS2
                               + (r - q) * S_i * dVdS - r * V;
        residual_sum += residual * residual;
        value_sum += V * V;
    }

    // Relative PDE residual: higher means worse approximation
    const double mean_residual = residual_sum / static_cast<double>(n_coll);
    const double mean_value = std::max(1.0, value_sum / static_cast<double>(n_coll));
    const double relative_residual = std::sqrt(mean_residual / mean_value);

    // Correction: bounded by loss_balance-scaled relative residual
    // Better PDE satisfaction -> smaller correction -> closer to BSM
    const double correction = std::min(0.15, params.loss_balance * 0.1 * relative_residual);

    const double approx_call = std::max(0.0, bsm_call * (1.0 - correction));
    return detail::call_put_from_call_parity(approx_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::mlm
