#include "algorithms/adjoint_greeks/likelihood_ratio/likelihood_ratio.h"

#include "algorithms/adjoint_greeks/common/internal_util.h"
#include "common/mc_engine.h"
#include "common/model_concepts.h"

#include <algorithm>
#include <cmath>

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
    const double disc = std::exp(-r * t);

    auto gen = mc::make_mt19937_normal(params.seed);
    auto model = models::make_bsm_terminal(vol, r, q);
    auto score_fn = models::make_bsm_lr_score(vol);
    auto accum = [&](double S_T, double z, int) -> double {
        const double payoff = (option_type == QK_CALL)
            ? std::max(S_T - strike, 0.0)
            : std::max(strike - S_T, 0.0);
        double score = score_fn(spot, t, z);
        score = std::max(-params.weight_clip, std::min(params.weight_clip, score));
        return payoff * score;
    };

    double mean = mc::estimate_terminal_antithetic(spot, t, n_paths, gen, model, accum);
    return detail::clamp_delta(disc * mean, t, q);
}

} // namespace qk::agm
