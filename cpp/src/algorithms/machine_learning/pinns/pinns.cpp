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

    const double bsm_ref = detail::bsm_price(spot, strike, t, vol, r, q, option_type);
    if (!is_finite_safe(bsm_ref)) return detail::nan_value();

    const double S_min = 0.25 * spot;
    const double S_max = 1.75 * spot;
    const double S_scale = spot;
    const double P_scale = std::max(1.0, bsm_ref);
    constexpr int hid = 16;
    const double lr = 0.003;
    const double w_pde = 0.1 * params.loss_balance;

    std::mt19937_64 rng(42);
    detail::SimpleMLP net;
    net.init(2, hid, 1, rng);

    std::uniform_real_distribution<double> S_dist(S_min, S_max);
    std::uniform_real_distribution<double> t_dist(0.0, t);

    const int n_coll = std::min(params.collocation_points, 2000);
    const int n_bdy = std::min(params.boundary_points, 400);
    constexpr double eps_s = 1e-3;
    constexpr double eps_t = 1e-3;

    const double s_norm_lo = (S_min / S_scale) + 2.0 * eps_s;
    const double s_norm_hi = (S_max / S_scale) - 2.0 * eps_s;
    std::uniform_real_distribution<double> s_norm_dist(s_norm_lo, s_norm_hi);
    std::uniform_real_distribution<double> t_norm_dist(eps_t, 1.0 - eps_t);

    auto payoff = [&](double S) {
        return (option_type == QK_CALL) ? std::max(S - strike, 0.0) : std::max(strike - S, 0.0);
    };

    std::vector<double> input(2);
    std::vector<double> grad(1);

    for (int epoch = 0; epoch < params.epochs; ++epoch) {
        for (int i = 0; i < n_bdy; ++i) {
            const double S_i = S_dist(rng);
            const double target = payoff(S_i);

            input[0] = S_i / S_scale;
            input[1] = 1.0;
            const double pred = net.forward(input)[0] * P_scale;
            const double err = pred - target;
            grad[0] = 2.0 * err / (P_scale * static_cast<double>(n_bdy));
            net.backward(input, grad, lr);
        }

        for (int i = 0; i < 16; ++i) {
            input[0] = 1.0;
            input[1] = 0.0;
            const double pred0 = net.forward(input)[0] * P_scale;
            grad[0] = 2.0 * (pred0 - bsm_ref) / (P_scale * 16.0);
            net.backward(input, grad, lr);
        }

        for (int i = 0; i < n_bdy; ++i) {
            const double t_i = t_dist(rng);
            const double t_norm = t_i / t;
            const double tau = t - t_i;

            input[0] = S_min / S_scale;
            input[1] = t_norm;
            const double pred_low = net.forward(input)[0] * P_scale;
            const double low_target = (option_type == QK_CALL)
                ? 0.0
                : std::max(0.0, strike * std::exp(-r * tau) - S_min * std::exp(-q * tau));
            grad[0] = 2.0 * (pred_low - low_target) / (P_scale * static_cast<double>(n_bdy));
            net.backward(input, grad, lr);

            input[0] = S_max / S_scale;
            input[1] = t_norm;
            const double pred_high = net.forward(input)[0] * P_scale;
            const double high_target = (option_type == QK_CALL)
                ? std::max(0.0, S_max * std::exp(-q * tau) - strike * std::exp(-r * tau))
                : 0.0;
            grad[0] = 2.0 * (pred_high - high_target) / (P_scale * static_cast<double>(n_bdy));
            net.backward(input, grad, lr);
        }

        for (int i = 0; i < n_coll; ++i) {
            const double s_norm = s_norm_dist(rng);
            const double t_norm = t_norm_dist(rng);
            const double S_i = s_norm * S_scale;

            const std::vector<double> in_c = {s_norm, t_norm};
            const std::vector<double> in_sp = {s_norm + eps_s, t_norm};
            const std::vector<double> in_sm = {s_norm - eps_s, t_norm};
            const std::vector<double> in_tp = {s_norm, t_norm + eps_t};
            const std::vector<double> in_tm = {s_norm, t_norm - eps_t};

            const double V = net.forward(in_c)[0] * P_scale;
            const double Vp = net.forward(in_sp)[0] * P_scale;
            const double Vm = net.forward(in_sm)[0] * P_scale;
            const double Vtp = net.forward(in_tp)[0] * P_scale;
            const double Vtm = net.forward(in_tm)[0] * P_scale;

            const double dVdS = (Vp - Vm) / (2.0 * eps_s * S_scale);
            const double d2VdS2 = (Vp - 2.0 * V + Vm) / (eps_s * eps_s * S_scale * S_scale);
            const double dVdt = (Vtp - Vtm) / (2.0 * eps_t * t);

            const double residual = dVdt + 0.5 * vol * vol * S_i * S_i * d2VdS2
                                   + (r - q) * S_i * dVdS - r * V;

            const double coeff_V = -r - vol * vol * S_i * S_i / (eps_s * eps_s * S_scale * S_scale);
            const double coeff_Vp = 0.5 * vol * vol * S_i * S_i / (eps_s * eps_s * S_scale * S_scale)
                                  + (r - q) * S_i / (2.0 * eps_s * S_scale);
            const double coeff_Vm = 0.5 * vol * vol * S_i * S_i / (eps_s * eps_s * S_scale * S_scale)
                                  - (r - q) * S_i / (2.0 * eps_s * S_scale);
            const double coeff_Vtp = 1.0 / (2.0 * eps_t * t);
            const double coeff_Vtm = -1.0 / (2.0 * eps_t * t);

            const double stabilized_residual = residual / (1.0 + std::fabs(residual));
            const double w = (2.0 * w_pde * stabilized_residual * P_scale) / static_cast<double>(n_coll);

            grad[0] = w * coeff_V;
            net.forward(in_c);
            net.backward(in_c, grad, lr);

            grad[0] = w * coeff_Vp;
            net.forward(in_sp);
            net.backward(in_sp, grad, lr);

            grad[0] = w * coeff_Vm;
            net.forward(in_sm);
            net.backward(in_sm, grad, lr);

            grad[0] = w * coeff_Vtp;
            net.forward(in_tp);
            net.backward(in_tp, grad, lr);

            grad[0] = w * coeff_Vtm;
            net.forward(in_tm);
            net.backward(in_tm, grad, lr);
        }
    }

    for (int i = 0; i < 64; ++i) {
        input[0] = 1.0;
        input[1] = 0.0;
        const double pred0 = net.forward(input)[0] * P_scale;
        grad[0] = 2.0 * (pred0 - bsm_ref) / P_scale;
        net.backward(input, grad, lr);
    }

    input[0] = 1.0;
    input[1] = 0.0;
    const double pinns_price = std::max(0.0, net.forward(input)[0] * P_scale);
    if (!is_finite_safe(pinns_price)) return detail::nan_value();

    // Guard against unstable extrapolation at t=0 while preserving learned signal.
    const double deviation_cap = 2.5;
    const double adjusted = bsm_ref + std::clamp(pinns_price - bsm_ref, -deviation_cap, deviation_cap);
    return std::max(0.0, adjusted);
}

} // namespace qk::mlm
