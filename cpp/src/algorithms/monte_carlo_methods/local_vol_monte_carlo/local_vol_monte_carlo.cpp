#include "algorithms/monte_carlo_methods/local_vol_monte_carlo/local_vol_monte_carlo.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"
#include "common/mc_engine.h"
#include "common/model_concepts.h"

#include <cmath>

namespace qk::mcm {

// ---------------------------------------------------------------------------
// Internal template pricer: works with any sigma_fn(s, t) callable.
//
// The loop mirrors estimate_stepwise() but adds explicit time tracking so
// that sigma_fn can depend on calendar time.  No new engine abstraction —
// just a local loop following the same computational pattern.
// ---------------------------------------------------------------------------
namespace {

template<typename SigmaFn>
double local_vol_price_impl(
    double spot, double strike, double t,
    double r, double q, SigmaFn&& sigma_fn,
    int32_t option_type,
    int32_t paths, int32_t steps, uint64_t seed)
{
    const double disc = std::exp(-r * t);
    const double dt = t / static_cast<double>(steps);
    const double sqrt_dt = std::sqrt(dt);

    auto gen = mc::make_mt19937_normal(seed);
    auto step = models::make_local_vol_euler_step(r, q,
                    std::forward<SigmaFn>(sigma_fn));

    double sum = 0.0;
    for (int32_t i = 0; i < paths; ++i) {
        double s = spot;
        double current_t = 0.0;
        for (int32_t j = 0; j < steps; ++j) {
            const double dw = sqrt_dt * gen();
            s = step(s, current_t, dt, dw);
            current_t += dt;
        }
        sum += detail::payoff(s, strike, option_type);
    }

    return disc * sum / static_cast<double>(paths);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public C-ABI-compatible entry point with constant local vol.
// When vol is constant, this produces results statistically identical to
// euler_maruyama_price() with the same parameters.
// ---------------------------------------------------------------------------
double local_vol_monte_carlo_price(double spot, double strike, double t,
                                   double vol, double r, double q,
                                   int32_t option_type,
                                   int32_t paths, int32_t steps, uint64_t seed) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        !detail::valid_mc_counts(paths, steps)) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    return local_vol_price_impl(
        spot, strike, t, r, q,
        models::make_local_vol_constant(vol),
        option_type, paths, steps, seed
    );
}

} // namespace qk::mcm
