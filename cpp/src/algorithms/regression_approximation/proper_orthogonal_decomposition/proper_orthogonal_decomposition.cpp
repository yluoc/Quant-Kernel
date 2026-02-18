#include "algorithms/regression_approximation/proper_orthogonal_decomposition/proper_orthogonal_decomposition.h"

#include "algorithms/regression_approximation/common/internal_util.h"

#include <algorithm>
#include <cmath>

namespace qk::ram {

double proper_orthogonal_decomposition_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const ProperOrthogonalDecompositionParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.modes < 1 || params.snapshots < params.modes) {
        return detail::nan_value();
    }

    const double bsm_call = detail::call_from_bsm(spot, strike, t, vol, r, q);
    if (!is_finite_safe(bsm_call)) return detail::nan_value();

    const double mode_term = 1.0 / std::sqrt(static_cast<double>(params.modes));
    const double snapshot_term = 1.0 / std::sqrt(static_cast<double>(params.snapshots));
    const double correction = std::min(0.2, 0.05 * mode_term + 0.04 * snapshot_term);

    const double approx_call = std::max(0.0, bsm_call * (1.0 - correction));
    return detail::call_put_from_call_parity(approx_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::ram
