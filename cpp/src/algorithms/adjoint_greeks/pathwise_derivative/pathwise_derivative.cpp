#include "algorithms/adjoint_greeks/pathwise_derivative/pathwise_derivative.h"

#include "algorithms/adjoint_greeks/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <random>

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
    const double sqrt_t = std::sqrt(t);
    const double drift = (r - q - 0.5 * vol * vol) * t;
    const double disc = std::exp(-r * t);

    std::mt19937_64 rng(params.seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    double sum = 0.0;
    for (int i = 0; i < n_paths; ++i) {
        const double z = norm(rng);
        const double S_T = spot * std::exp(drift + vol * sqrt_t * z);

        if (option_type == QK_CALL) {
            if (S_T > strike) {
                sum += S_T / spot;
            }
        } else {
            if (S_T < strike) {
                sum -= S_T / spot;
            }
        }
    }

    const double delta = disc * sum / static_cast<double>(n_paths);
    return detail::clamp_delta(delta, t, q);
}

} // namespace qk::agm
