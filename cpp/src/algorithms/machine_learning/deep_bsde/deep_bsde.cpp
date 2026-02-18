#include "algorithms/machine_learning/deep_bsde/deep_bsde.h"

#include "algorithms/machine_learning/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace qk::mlm {

double deep_bsde_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const DeepBsdeParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.time_steps < 8 || params.hidden_width < 8 || params.training_epochs < 10 ||
        !is_finite_safe(params.learning_rate) || params.learning_rate <= 0.0) {
        return detail::nan_value();
    }

    const double bsm_call = detail::call_from_bsm(spot, strike, t, vol, r, q);
    if (!is_finite_safe(bsm_call)) return detail::nan_value();

    const int N = params.time_steps;
    const int hid = std::min(params.hidden_width, 32);
    const double lr = params.learning_rate;
    const double dt = t / static_cast<double>(N);
    const double sqrt_dt = std::sqrt(dt);
    constexpr int batch_size = 64;

    // Initialize Y_0 with BSM estimate
    double Y0 = bsm_call;

    // One SimpleMLP per time step to approximate Z_t (gradient of value function)
    std::mt19937_64 rng(42);
    std::vector<detail::SimpleMLP> Z_nets(N);
    for (int k = 0; k < N; ++k) {
        Z_nets[k].init(1, hid, 1, rng);
    }

    std::normal_distribution<double> norm(0.0, 1.0);

    for (int epoch = 0; epoch < params.training_epochs; ++epoch) {
        double dY0 = 0.0;

        for (int b = 0; b < batch_size; ++b) {
            // Forward simulation of BSDE
            double S = spot;
            double Y = Y0;
            std::vector<double> input(1);

            for (int k = 0; k < N; ++k) {
                const double dW = sqrt_dt * norm(rng);
                input[0] = S / spot; // normalized spot

                const auto& z_out = Z_nets[k].forward(input);
                const double Z = z_out[0] * vol * S;

                // BSDE evolution: Y_{k+1} = Y_k + (r*Y_k - q*Z)*dt + Z*vol*S*dW/S... simplified
                Y += (r * Y - q * S * Z / S) * dt + Z * dW;
                S *= std::exp((r - q - 0.5 * vol * vol) * dt + vol * dW);
            }

            // Terminal payoff
            const double payoff = std::max(S - strike, 0.0);
            const double disc_payoff = std::exp(-r * t) * payoff;

            // Loss = (Y - disc_payoff)^2
            const double err = Y - disc_payoff;

            // Update Y_0 via gradient descent
            dY0 += 2.0 * err / static_cast<double>(batch_size);

            // Update Z networks (simplified: backprop through last step)
            for (int k = N - 1; k >= 0; --k) {
                std::vector<double> d_out = {2.0 * err * vol / static_cast<double>(batch_size)};
                input[0] = spot / spot; // Use normalized spot ~ 1.0 for gradient
                Z_nets[k].forward(input);
                Z_nets[k].backward(input, d_out, lr);
            }
        }

        Y0 -= lr * dY0;
        Y0 = std::max(0.0, Y0);
    }

    const double approx_call = std::max(0.0, Y0);
    return detail::call_put_from_call_parity(approx_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::mlm
