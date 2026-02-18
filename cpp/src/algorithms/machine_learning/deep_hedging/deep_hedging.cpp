#include "algorithms/machine_learning/deep_hedging/deep_hedging.h"

#include "algorithms/machine_learning/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <numeric>
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

    const double bsm_call = detail::call_from_bsm(spot, strike, t, vol, r, q);
    if (!is_finite_safe(bsm_call)) return detail::nan_value();

    const int n_steps = std::min(params.rehedge_steps, 52);
    const int n_scenarios = std::min(params.scenarios, 4096);
    const double dt = t / static_cast<double>(n_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double disc = std::exp(-r * t);
    constexpr int hid = 16;
    const double lr = 0.005;
    const double risk_av = std::min(params.risk_aversion, 5.0);

    std::mt19937_64 rng(params.seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    // Hedging network: input=(S_norm, time_frac), output=hedge_ratio
    detail::SimpleMLP hedge_net;
    hedge_net.init(2, hid, 1, rng);

    // Initialize V_0 with BSM
    double V0 = bsm_call;

    // Training loop
    constexpr int n_train_epochs = 20;
    std::vector<double> input(2);

    for (int epoch = 0; epoch < n_train_epochs; ++epoch) {
        std::vector<double> pnl(n_scenarios);

        for (int sc = 0; sc < n_scenarios; ++sc) {
            double S = spot;
            double hedge_gains = 0.0;

            for (int k = 0; k < n_steps; ++k) {
                const double time_frac = static_cast<double>(k) / static_cast<double>(n_steps);
                input[0] = S / spot;
                input[1] = time_frac;

                const auto& out = hedge_net.forward(input);
                const double delta_h = std::tanh(out[0]); // bound hedge ratio to [-1, 1]

                const double dW = sqrt_dt * norm(rng);
                const double S_new = S * std::exp((r - q - 0.5 * vol * vol) * dt + vol * dW);
                hedge_gains += delta_h * (S_new - S) * disc;
                S = S_new;
            }

            // Terminal payoff (call)
            const double payoff = std::max(S - strike, 0.0) * disc;
            // P&L = initial premium + hedge gains - payoff
            pnl[sc] = V0 * disc + hedge_gains - payoff;
        }

        // CVaR loss at 95th percentile
        std::vector<double> sorted_pnl = pnl;
        std::sort(sorted_pnl.begin(), sorted_pnl.end());
        const int tail_start = static_cast<int>(0.05 * n_scenarios);
        double cvar = 0.0;
        const int tail_count = std::max(1, tail_start);
        for (int i = 0; i < tail_count; ++i) {
            cvar += sorted_pnl[i];
        }
        cvar /= static_cast<double>(tail_count);

        // Mean P&L
        const double mean_pnl = std::accumulate(pnl.begin(), pnl.end(), 0.0) / n_scenarios;

        // Gradient of V0: increase V0 if mean P&L < 0 (underpriced)
        const double dV0 = -mean_pnl + risk_av * std::min(0.0, cvar);
        V0 += lr * dV0;
        V0 = std::max(0.0, V0);

        // Update hedge network with a simplified gradient
        for (int sc = 0; sc < std::min(n_scenarios, 256); ++sc) {
            const double grad_sign = (pnl[sc] < 0.0) ? 1.0 : -0.1;
            input[0] = 1.0;
            input[1] = 0.5;
            hedge_net.forward(input);
            std::vector<double> d_out = {grad_sign * lr * 0.1};
            hedge_net.backward(input, d_out, lr);
        }
    }

    const double approx_call = std::max(0.0, V0);
    return detail::call_put_from_call_parity(approx_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::mlm
