#include "algorithms/monte_carlo_methods/heston_monte_carlo/heston_lr_delta.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"
#include "common/mc_engine.h"
#include "common/model_concepts.h"

#include <algorithm>
#include <cmath>

namespace qk::mcm {

// ---------------------------------------------------------------------------
// Likelihood-ratio delta estimator for Heston MC.
//
// Derivation of the LR weight:
//
//   Under log-Euler discretization, the first spot step is:
//     log(S_1) = log(S_0) + (r - q - 0.5*v_0^+)*dt + sqrt(v_0^+)*sqrt_dt*z1_0
//
//   The density of log(S_1) given S_0 is Normal with:
//     mean  = log(S_0) + mu_0 * dt,   sigma_1 = sqrt(v_0^+) * sqrt_dt
//
//   The score (d/dS_0 of log-density) is:
//     w = z1_0 / (sqrt(v_0^+) * sqrt_dt * S_0)
//
//   Subsequent step transitions are conditionally independent of S_0, so
//   only the first step's z1 contributes to the total score.
//
// Antithetic variance reduction:
//   Each pair shares the same z-sequence: one path uses (z1, z2) directly,
//   the antithetic path uses (-z1, -z2).  The LR weight for the antithetic
//   path is exactly -w (negated first z1).  The RNG state is saved on the
//   stack before each pair and restored for the antithetic path, so there
//   is no per-path heap allocation.
// ---------------------------------------------------------------------------

double heston_likelihood_ratio_delta(
    double spot, double strike, double t,
    double r, double q,
    double v0, double kappa, double theta,
    double sigma, double rho,
    int32_t option_type,
    int32_t paths, int32_t steps, uint64_t seed,
    double weight_clip)
{
    if (!detail::valid_option_type(option_type) ||
        spot <= 0.0 || strike <= 0.0 || t < 0.0 ||
        v0 < 0.0 || kappa < 0.0 || theta < 0.0 || sigma < 0.0 ||
        rho < -1.0 || rho > 1.0 ||
        !detail::valid_mc_counts(paths, steps) ||
        weight_clip <= 0.0) {
        return detail::nan_value();
    }

    // Degenerate maturity — intrinsic delta.
    if (t <= detail::kEps) {
        // For near-zero maturity the option is either deep ITM or OTM.
        if (option_type == QK_CALL) return (spot > strike) ? 1.0 : 0.0;
        return (spot < strike) ? -1.0 : 0.0;
    }

    const double disc = std::exp(-r * t);
    const double dt = t / static_cast<double>(steps);
    const double sqrt_dt = std::sqrt(dt);
    const double sqrt_v0 = std::sqrt(std::max(v0, 0.0));

    auto gen = mc::make_mt19937_normal(seed);
    auto step = models::make_heston_euler_step(r, q, kappa, theta, sigma, rho);

    double sum = 0.0;
    const int32_t n_pairs = paths / 2;

    for (int32_t i = 0; i < n_pairs; ++i) {
        // Save RNG state for antithetic pair (stack copy, no heap).
        auto gen_saved = gen;

        // --- Forward path ---
        double s = spot, v = v0;
        double z1_first = 0.0;
        for (int32_t j = 0; j < steps; ++j) {
            const double z1 = gen();
            const double z2 = gen();
            if (j == 0) z1_first = z1;
            auto result = step(s, v, dt, sqrt_dt, z1, z2);
            s = result.s;
            v = result.v;
        }
        double pf = detail::payoff(s, strike, option_type);

        // LR score from first step.
        double score = (sqrt_v0 > 1e-15)
            ? z1_first / (sqrt_v0 * sqrt_dt * spot)
            : 0.0;
        score = std::max(-weight_clip, std::min(weight_clip, score));

        // --- Antithetic path (negate all z draws) ---
        s = spot; v = v0;
        for (int32_t j = 0; j < steps; ++j) {
            const double z1 = -gen_saved();
            const double z2 = -gen_saved();
            auto result = step(s, v, dt, sqrt_dt, z1, z2);
            s = result.s;
            v = result.v;
        }
        double pa = detail::payoff(s, strike, option_type);

        // Antithetic score is -score (first z1 is negated).
        sum += pf * score + pa * (-score);
    }

    // Handle odd remaining path.
    if ((paths & 1) != 0) {
        double s = spot, v = v0;
        double z1_first = 0.0;
        for (int32_t j = 0; j < steps; ++j) {
            const double z1 = gen();
            const double z2 = gen();
            if (j == 0) z1_first = z1;
            auto result = step(s, v, dt, sqrt_dt, z1, z2);
            s = result.s;
            v = result.v;
        }
        double p = detail::payoff(s, strike, option_type);
        double score = (sqrt_v0 > 1e-15)
            ? z1_first / (sqrt_v0 * sqrt_dt * spot)
            : 0.0;
        score = std::max(-weight_clip, std::min(weight_clip, score));
        sum += p * score;
    }

    return disc * sum / static_cast<double>(paths);
}

} // namespace qk::mcm
