#include "algorithms/machine_learning/deep_hedging/deep_hedging.h"

#include "algorithms/machine_learning/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace qk::mlm {

double deep_hedging_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const DeepHedgingParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.rehedge_steps < 2 || params.scenarios < 256 ||
        !is_finite_safe(params.risk_aversion) || params.risk_aversion < 0.0) {
        return detail::nan_value();
    }

    const double bsm_ref = detail::bsm_price(spot, strike, t, vol, r, q, option_type);
    if (!is_finite_safe(bsm_ref)) return detail::nan_value();

    const int n_steps = std::min(params.rehedge_steps, 52);
    const int n_scenarios = std::min(params.scenarios, 4096);
    const double dt = t / static_cast<double>(n_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double growth = std::exp(r * dt);
    const double growth_to_t = std::exp(r * t);
    constexpr int hid = 16;
    const double lr = 0.0025;
    const double risk_av = std::min(params.risk_aversion, 5.0);

    std::mt19937_64 rng(params.seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    detail::SimpleMLP hedge_net;
    hedge_net.init(2, hid, 1, rng);

    double V0 = bsm_ref;

    constexpr int n_train_epochs = 24;
    constexpr double grad_clip = 50.0;

    std::vector<double> S_path(static_cast<std::size_t>(n_steps + 1), spot);
    std::vector<double> delta(static_cast<std::size_t>(n_steps), 0.0);
    std::vector<double> dcash_ddelta(static_cast<std::size_t>(n_steps), 0.0);

    for (int epoch = 0; epoch < n_train_epochs; ++epoch) {
        double dV0 = 0.0;

        for (int sc = 0; sc < n_scenarios; ++sc) {
            for (int k = 0; k < n_steps; ++k) {
                const double z = norm(rng);
                S_path[static_cast<std::size_t>(k + 1)] =
                    S_path[static_cast<std::size_t>(k)] *
                    std::exp((r - q - 0.5 * vol * vol) * dt + vol * sqrt_dt * z);
            }

            for (int k = 0; k < n_steps; ++k) {
                const double time_frac = static_cast<double>(k) / static_cast<double>(n_steps);
                const std::vector<double> input = {S_path[static_cast<std::size_t>(k)] / spot, time_frac};
                const double raw = hedge_net.forward(input)[0];
                delta[static_cast<std::size_t>(k)] = std::tanh(raw);
            }

            std::fill(dcash_ddelta.begin(), dcash_ddelta.end(), 0.0);
            double cash = V0;
            for (int k = 0; k < n_steps; ++k) {
                const double S_k = S_path[static_cast<std::size_t>(k)];
                const double d_k = delta[static_cast<std::size_t>(k)];
                const double d_prev = (k == 0) ? 0.0 : delta[static_cast<std::size_t>(k - 1)];

                cash -= (d_k - d_prev) * S_k;
                if (k > 0) dcash_ddelta[static_cast<std::size_t>(k - 1)] += S_k;
                dcash_ddelta[static_cast<std::size_t>(k)] -= S_k;

                for (int j = 0; j <= k; ++j) {
                    dcash_ddelta[static_cast<std::size_t>(j)] *= growth;
                }
                cash = cash * growth + d_k * S_k * q * dt;
                dcash_ddelta[static_cast<std::size_t>(k)] += S_k * q * dt;
            }

            const double S_T = S_path[static_cast<std::size_t>(n_steps)];
            const double portfolio = cash + delta[static_cast<std::size_t>(n_steps - 1)] * S_T;
            dcash_ddelta[static_cast<std::size_t>(n_steps - 1)] += S_T;

            const double payoff = (option_type == QK_CALL)
                ? std::max(S_T - strike, 0.0)
                : std::max(strike - S_T, 0.0);
            const double pnl = portfolio - payoff;

            // Shortfall risk: L = shortfall + 0.5 * risk_av * shortfall^2.
            const double shortfall = std::max(-pnl, 0.0);
            double dL_dpnl = 0.0;
            if (shortfall > 0.0) {
                dL_dpnl = -(1.0 + risk_av * shortfall);
            }
            dL_dpnl /= static_cast<double>(n_scenarios);

            dV0 += dL_dpnl * growth_to_t;

            for (int k = 0; k < n_steps; ++k) {
                const double dL_ddelta = dL_dpnl * dcash_ddelta[static_cast<std::size_t>(k)];
                double dL_dout = dL_ddelta * (1.0 - delta[static_cast<std::size_t>(k)] * delta[static_cast<std::size_t>(k)]);
                dL_dout = std::max(-grad_clip, std::min(grad_clip, dL_dout));

                const double time_frac = static_cast<double>(k) / static_cast<double>(n_steps);
                const std::vector<double> input = {S_path[static_cast<std::size_t>(k)] / spot, time_frac};
                hedge_net.forward(input);
                std::vector<double> grad_out = {dL_dout};
                hedge_net.backward(input, grad_out, lr);
            }

            S_path[0] = spot;
        }

        V0 -= lr * dV0;
        V0 = std::max(0.0, std::min(3.0 * std::max(1.0, bsm_ref), V0));
    }

    if (!is_finite_safe(V0)) return detail::nan_value();
    return std::max(0.0, V0);
}

} // namespace qk::mlm
