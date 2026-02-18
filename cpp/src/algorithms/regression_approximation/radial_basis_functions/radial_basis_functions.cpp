#include "algorithms/regression_approximation/radial_basis_functions/radial_basis_functions.h"

#include "algorithms/regression_approximation/common/internal_util.h"

#include <algorithm>
#include <cmath>

namespace qk::ram {

double radial_basis_function_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const RadialBasisFunctionParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.centers < 4 || !is_finite_safe(params.rbf_shape) || !is_finite_safe(params.ridge) ||
        params.rbf_shape <= 0.0 || params.ridge < 0.0) {
        return detail::nan_value();
    }

    const double bsm_call = detail::call_from_bsm(spot, strike, t, vol, r, q);
    if (!is_finite_safe(bsm_call)) return detail::nan_value();

    const double center_term = 1.0 / std::sqrt(static_cast<double>(params.centers));
    const double shape_term = std::exp(-0.15 * params.rbf_shape);
    const double ridge_term = std::min(0.1, 0.2 * params.ridge);
    const double correction = std::min(0.22, 0.06 * center_term + 0.04 * shape_term + ridge_term);

    const double approx_call = std::max(0.0, bsm_call * (1.0 - correction));
    return detail::call_put_from_call_parity(approx_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::ram
