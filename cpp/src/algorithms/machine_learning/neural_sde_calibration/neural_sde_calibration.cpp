#include "algorithms/machine_learning/neural_sde_calibration/neural_sde_calibration.h"

#include "algorithms/machine_learning/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace qk::mlm {

double neural_sde_calibration_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const NeuralSdeCalibrationParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        !is_finite_safe(params.target_implied_vol) || params.target_implied_vol < 0.0 ||
        params.calibration_steps < 4 ||
        !is_finite_safe(params.regularization) || params.regularization < 0.0) {
        return detail::nan_value();
    }

    constexpr int hid = 8;
    const double lr = 0.01;

    std::mt19937_64 rng(42);
    detail::SimpleMLP net;
    net.init(3, hid, 1, rng);

    const double moneyness = std::log(spot / strike);
    const double target_correction = params.target_implied_vol - vol;

    std::vector<double> input = {moneyness, t, vol};

    for (int step = 0; step < params.calibration_steps; ++step) {
        const auto& out = net.forward(input);
        const double pred_correction = out[0];
        const double err = pred_correction - target_correction;

        const double reg_grad = 2.0 * params.regularization * pred_correction;
        std::vector<double> d_out = {2.0 * err + reg_grad};
        net.backward(input, d_out, lr);
    }

    const auto& final_out = net.forward(input);
    const double correction = final_out[0];
    const double effective_vol = std::max(0.01, vol + correction);

    const double call_price = detail::call_from_bsm(spot, strike, t, effective_vol, r, q);
    if (!is_finite_safe(call_price)) return detail::nan_value();

    const double approx_call = std::max(0.0, call_price);
    return detail::call_put_from_call_parity(approx_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::mlm
