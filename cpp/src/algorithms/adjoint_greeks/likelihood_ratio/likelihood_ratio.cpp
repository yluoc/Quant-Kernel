#include "algorithms/adjoint_greeks/likelihood_ratio/likelihood_ratio.h"

#include "algorithms/adjoint_greeks/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace qk::agm {

double likelihood_ratio_delta(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const LikelihoodRatioParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.paths < 256 || !is_finite_safe(params.weight_clip) || params.weight_clip <= 0.0) {
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

    auto sample_term = [&](double z) {
        const double S_T = spot * std::exp(drift + vol * sqrt_t * z);
        const double payoff = (option_type == QK_CALL)
            ? std::max(S_T - strike, 0.0)
            : std::max(strike - S_T, 0.0);
        double score = z / (vol * sqrt_t * spot);
        score = std::max(-params.weight_clip, std::min(params.weight_clip, score));
        return payoff * score;
    };

    // Antithetic pairing materially reduces LR variance in tail-heavy regimes.
    double sum = 0.0;
    const int n_pairs = n_paths / 2;
    for (int i = 0; i < n_pairs; ++i) {
        const double z = norm(rng);
        sum += sample_term(z);
        sum += sample_term(-z);
    }
    if ((n_paths & 1) != 0) {
        sum += sample_term(norm(rng));
    }

    const double delta = disc * sum / static_cast<double>(n_paths);
    return detail::clamp_delta(delta, t, q);
}

} // namespace qk::agm
