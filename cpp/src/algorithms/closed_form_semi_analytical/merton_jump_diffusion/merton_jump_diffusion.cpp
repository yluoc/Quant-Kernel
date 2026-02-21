#include "algorithms/closed_form_semi_analytical/merton_jump_diffusion/merton_jump_diffusion.h"

#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/closed_form_semi_analytical/common/internal_util.h"

namespace qk::cfa {

double merton_jump_diffusion_price(double spot, double strike, double t, double vol,
                                   double r, double q,
                                   const MertonJumpDiffusionParams& params,
                                   int32_t option_type) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(params.jump_intensity) || !is_finite_safe(params.jump_mean) ||
        !is_finite_safe(params.jump_vol)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0 ||
        params.jump_intensity < 0.0 || params.jump_vol < 0.0) {
        return detail::nan_value();
    }
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    int32_t max_terms = params.max_terms > 0 ? params.max_terms : 50;
    double jump_vol2 = params.jump_vol * params.jump_vol;
    double mu_plus_half_jump_var = params.jump_mean + 0.5 * jump_vol2;
    double one_plus_k = std::exp(mu_plus_half_jump_var);
    double kappa_j = one_plus_k - 1.0;

    // Align with QuantLib JumpDiffusionEngine semantics:
    // Poisson mixing uses lambda' = (1 + kappa_j) * lambda.
    double lambda_t = params.jump_intensity * one_plus_k * t;

    double weight = std::exp(-lambda_t);
    double price = 0.0;

    double vol2 = vol * vol;
    double inv_t = 1.0 / t;
    double base_r = r - params.jump_intensity * kappa_j;

    for (int32_t n = 0; n <= max_terms; ++n) {
        double n_d = static_cast<double>(n);
        double var_n = vol2 + n_d * jump_vol2 * inv_t;
        double vol_n = std::sqrt(std::max(0.0, var_n));
        double r_n = base_r + n_d * mu_plus_half_jump_var * inv_t;

        price += weight * black_scholes_merton_price(spot, strike, t, vol_n, r_n, q, option_type);

        if (weight < 1e-15) break;
        weight *= lambda_t / static_cast<double>(n + 1);
    }
    return price;
}

} // namespace qk::cfa
