#include "algorithms/regression_approximation/sparse_grid_collocation/sparse_grid_collocation.h"

#include "algorithms/regression_approximation/common/internal_util.h"

#include <algorithm>
#include <cmath>

namespace qk::ram {

double sparse_grid_collocation_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const SparseGridCollocationParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.level < 1 || params.nodes_per_dim < 2) {
        return detail::nan_value();
    }

    const double bsm_call = detail::call_from_bsm(spot, strike, t, vol, r, q);
    if (!is_finite_safe(bsm_call)) return detail::nan_value();

    const double level_term = 1.0 / static_cast<double>(params.level + 1);
    const double node_term = 1.0 / std::sqrt(static_cast<double>(params.nodes_per_dim));
    const double dimensionality_proxy = std::fabs(std::log(spot / strike));
    const double correction = std::min(0.2, 0.05 * level_term + 0.04 * node_term + 0.005 * dimensionality_proxy);

    const double approx_call = std::max(0.0, bsm_call * (1.0 - correction));
    return detail::call_put_from_call_parity(approx_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::ram
