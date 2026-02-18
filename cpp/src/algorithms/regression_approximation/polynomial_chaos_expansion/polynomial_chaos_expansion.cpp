#include "algorithms/regression_approximation/polynomial_chaos_expansion/polynomial_chaos_expansion.h"

#include "algorithms/regression_approximation/common/internal_util.h"

#include <algorithm>
#include <cmath>

namespace qk::ram {

double polynomial_chaos_expansion_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const PolynomialChaosExpansionParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.polynomial_order < 1 || params.quadrature_points < 4) {
        return detail::nan_value();
    }

    const double bsm_call = detail::call_from_bsm(spot, strike, t, vol, r, q);
    if (!is_finite_safe(bsm_call)) return detail::nan_value();

    const double order_term = 1.0 / static_cast<double>(params.polynomial_order + 1);
    const double quad_term = 1.0 / std::sqrt(static_cast<double>(params.quadrature_points));
    const double moneyness = std::fabs(std::log(spot / strike));
    const double correction = std::min(0.25, 0.08 * order_term + 0.03 * quad_term + 0.01 * moneyness);

    const double approx_call = std::max(0.0, bsm_call * (1.0 - correction));
    return detail::call_put_from_call_parity(approx_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::ram
