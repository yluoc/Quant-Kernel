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

    const double bsm_ref = detail::bsm_price(spot, strike, t, vol, r, q, option_type);
    if (!is_finite_safe(bsm_ref)) return detail::nan_value();

    const int N = params.time_steps;
    const int hid = std::min(params.hidden_width, 32);
    const double lr = params.learning_rate;
    const double dt = t / static_cast<double>(N);
    const double sqrt_dt = std::sqrt(dt);
    constexpr int batch_size = 64;
    constexpr double grad_clip = 25.0;

    double Y0 = bsm_ref;

    std::mt19937_64 rng(42);
    std::vector<detail::SimpleMLP> Z_nets(N);
    for (int k = 0; k < N; ++k) {
        Z_nets[k].init(1, hid, 1, rng);
    }

    std::normal_distribution<double> norm(0.0, 1.0);

    for (int epoch = 0; epoch < params.training_epochs; ++epoch) {
        double dY0 = 0.0;

        for (int b = 0; b < batch_size; ++b) {
            double S = spot;
            double Y = Y0;

            std::vector<double> s_norms(static_cast<std::size_t>(N), 1.0);
            std::vector<double> dWs(static_cast<std::size_t>(N), 0.0);

            for (int k = 0; k < N; ++k) {
                const double z = norm(rng);
                const double dW = sqrt_dt * z;
                const double s_norm = S / spot;

                s_norms[static_cast<std::size_t>(k)] = s_norm;
                dWs[static_cast<std::size_t>(k)] = dW;

                const std::vector<double> input = {s_norm};
                const double z_raw = Z_nets[static_cast<std::size_t>(k)].forward(input)[0];
                const double Z = z_raw * spot;

                // Discrete BSDE with linear driver f = rY.
                Y = Y + r * Y * dt + Z * dW;
                S *= std::exp((r - q - 0.5 * vol * vol) * dt + vol * dW);
            }

            const double payoff = (option_type == QK_CALL)
                ? std::max(S - strike, 0.0)
                : std::max(strike - S, 0.0);
            const double err = Y - payoff;

            double adj = 2.0 * err / static_cast<double>(batch_size); // dL / dY_N
            for (int k = N - 1; k >= 0; --k) {
                const double dL_dZ = adj * dWs[static_cast<std::size_t>(k)];
                double dL_dout = dL_dZ * spot;
                dL_dout = std::max(-grad_clip, std::min(grad_clip, dL_dout));

                const std::vector<double> input = {s_norms[static_cast<std::size_t>(k)]};
                Z_nets[static_cast<std::size_t>(k)].forward(input);
                std::vector<double> grad_out = {dL_dout};
                Z_nets[static_cast<std::size_t>(k)].backward(input, grad_out, lr);

                adj *= (1.0 + r * dt);
            }
            dY0 += adj;
        }

        Y0 -= lr * dY0;
        Y0 = std::max(0.0, std::min(3.0 * std::max(1.0, bsm_ref), Y0));
    }

    if (!is_finite_safe(Y0)) return detail::nan_value();
    return std::max(0.0, Y0);
}

} // namespace qk::mlm
