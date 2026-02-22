#include "algorithms/adjoint_greeks/pathwise_derivative/pathwise_derivative.h"

#include "algorithms/adjoint_greeks/common/internal_util.h"
#include "common/mc_engine.h"
#include "common/model_concepts.h"

#include <cmath>

namespace qk::agm {

double pathwise_derivative_delta(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const PathwiseDerivativeParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.paths < 256) {
        return detail::nan_value();
    }

    if (t <= detail::kEps || vol <= detail::kEps) {
        return detail::deterministic_delta(spot, strike, t, r, q, option_type);
    }

    const int n_paths = params.paths;
    const double disc = std::exp(-r * t);

    auto gen = mc::make_mt19937_normal(params.seed);
    auto model = models::make_bsm_terminal(vol, r, q);
    auto dST_dS0 = models::make_bsm_pathwise_dST_dSpot();
    auto accum = [&](double S_T, double z, int) -> double {
        if (option_type == QK_CALL) {
            return (S_T > strike) ? dST_dS0(spot, S_T, t, z) : 0.0;
        } else {
            return (S_T < strike) ? -dST_dS0(spot, S_T, t, z) : 0.0;
        }
    };

    double mean = mc::estimate_terminal(spot, t, n_paths, gen, model, accum);
    return detail::clamp_delta(disc * mean, t, q);
}

} // namespace qk::agm
