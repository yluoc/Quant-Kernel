#ifndef QK_MODEL_CONCEPTS_H
#define QK_MODEL_CONCEPTS_H

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace qk::models {

// ===========================================================================
// Compile-time callable-signature checks (C++17, no concepts required).
//
// Usage:  static_assert(is_terminal_model_v<decltype(m)>,
//             "TerminalModel must be callable as (double spot, double t, double z) -> double");
//
// These detect signature mismatches at the call site with a readable message
// instead of deep template-instantiation errors inside mc_engine.h.
// ===========================================================================
namespace detail {

template<typename F, typename = void>
struct is_terminal_model : std::false_type {};

template<typename F>
struct is_terminal_model<F, std::void_t<decltype(
    static_cast<double>(std::declval<F&>()(
        std::declval<double>(), std::declval<double>(), std::declval<double>())))>>
    : std::true_type {};

template<typename F, typename = void>
struct is_step_model : std::false_type {};

template<typename F>
struct is_step_model<F, std::void_t<decltype(
    static_cast<double>(std::declval<F&>()(
        std::declval<double>(), std::declval<double>(), std::declval<double>())))>>
    : std::true_type {};

template<typename F, typename = void>
struct is_pathwise_dST_dSpot : std::false_type {};

template<typename F>
struct is_pathwise_dST_dSpot<F, std::void_t<decltype(
    static_cast<double>(std::declval<F&>()(
        std::declval<double>(), std::declval<double>(),
        std::declval<double>(), std::declval<double>())))>>
    : std::true_type {};

template<typename F, typename = void>
struct is_lr_score : std::false_type {};

template<typename F>
struct is_lr_score<F, std::void_t<decltype(
    static_cast<double>(std::declval<F&>()(
        std::declval<double>(), std::declval<double>(), std::declval<double>())))>>
    : std::true_type {};

} // namespace detail

// ---------------------------------------------------------------------------
// Public trait aliases — use these in static_assert.
// ---------------------------------------------------------------------------

// TerminalModel: (double spot, double t, double z) -> double S_T
template<typename F>
inline constexpr bool is_terminal_model_v = detail::is_terminal_model<std::decay_t<F>>::value;

// StepModel: (double s, double dt, double dw) -> double s_next
template<typename F>
inline constexpr bool is_step_model_v = detail::is_step_model<std::decay_t<F>>::value;

// PathwiseSensitivity: (double spot, double S_T, double t, double z) -> double dS_T/dSpot
template<typename F>
inline constexpr bool is_pathwise_dST_dSpot_v = detail::is_pathwise_dST_dSpot<std::decay_t<F>>::value;

// LikelihoodRatioScore: (double spot, double t, double z) -> double score
template<typename F>
inline constexpr bool is_lr_score_v = detail::is_lr_score<std::decay_t<F>>::value;

// ===========================================================================
// BSM model factories.
//
// Each returns a lightweight lambda/functor.  Under LTO these inline to
// identical codegen as hand-written loops.  No heap, no vtable, no virtual.
// ===========================================================================

// ---------------------------------------------------------------------------
// Terminal-value model: maps (spot, t, z) -> S_T via GBM log-normal formula.
//   drift     = (r - q - 0.5 * vol^2) * t
//   diffusion = vol * sqrt(t) * z
//   S_T       = spot * exp(drift + diffusion)
// ---------------------------------------------------------------------------
inline auto make_bsm_terminal(double vol, double r, double q) {
    auto f = [vol, r, q](double spot, double t, double z) -> double {
        double drift = (r - q - 0.5 * vol * vol) * t;
        double diffusion = vol * std::sqrt(std::max(t, 0.0)) * z;
        return spot * std::exp(drift + diffusion);
    };
    static_assert(is_terminal_model_v<decltype(f)>,
        "make_bsm_terminal must return a callable matching (double, double, double) -> double");
    return f;
}

// ---------------------------------------------------------------------------
// Pathwise sensitivity: maps (spot, S_T, t, z) -> dS_T/dSpot.
// For GBM, dS_T/dSpot = S_T / spot (the ratio is model-independent of z, t).
// ---------------------------------------------------------------------------
inline auto make_bsm_pathwise_dST_dSpot() {
    auto f = [](double spot, double S_T, double /*t*/, double /*z*/) -> double {
        return S_T / spot;
    };
    static_assert(is_pathwise_dST_dSpot_v<decltype(f)>,
        "make_bsm_pathwise_dST_dSpot must return a callable matching (double, double, double, double) -> double");
    return f;
}

// ---------------------------------------------------------------------------
// Likelihood-ratio score: maps (spot, t, z) -> d(log p)/d(spot).
// For GBM: score = z / (vol * sqrt(t) * spot).
// ---------------------------------------------------------------------------
inline auto make_bsm_lr_score(double vol) {
    auto f = [vol](double spot, double t, double z) -> double {
        const double sqrt_t = std::sqrt(t);
        return z / (vol * sqrt_t * spot);
    };
    static_assert(is_lr_score_v<decltype(f)>,
        "make_bsm_lr_score must return a callable matching (double, double, double) -> double");
    return f;
}

// ---------------------------------------------------------------------------
// Euler-Maruyama step: maps (s, dt, dw) -> s_next.
//   s_next = s + drift * s * dt + vol * s * dw
//   Floored at 1e-12 to prevent non-positive prices.
// ---------------------------------------------------------------------------
inline auto make_bsm_euler_step(double vol, double drift) {
    auto f = [vol, drift](double s, double dt, double dw) -> double {
        return std::max(1e-12, s + drift * s * dt + vol * s * dw);
    };
    static_assert(is_step_model_v<decltype(f)>,
        "make_bsm_euler_step must return a callable matching (double, double, double) -> double");
    return f;
}

// ---------------------------------------------------------------------------
// Milstein step: maps (s, dt, dw) -> s_next.
//   s_next = s + drift*s*dt + vol*s*dw + 0.5*vol^2*s*(dw^2 - dt)
//   Floored at 1e-12 to prevent non-positive prices.
// ---------------------------------------------------------------------------
inline auto make_bsm_milstein_step(double vol, double drift) {
    auto f = [vol, drift](double s, double dt, double dw) -> double {
        return std::max(1e-12, s + drift * s * dt + vol * s * dw
                        + 0.5 * vol * vol * s * (dw * dw - dt));
    };
    static_assert(is_step_model_v<decltype(f)>,
        "make_bsm_milstein_step must return a callable matching (double, double, double) -> double");
    return f;
}

// ===========================================================================
// Two-dimensional state for stochastic volatility models (e.g. Heston).
// ===========================================================================

struct StepResult2D {
    double s;
    double v;
};

// ---------------------------------------------------------------------------
// Compile-time trait: StepModel2D must be callable as
//   (double s, double v, double dt, double sqrt_dt, double z1, double z2)
//     -> StepResult2D
// ---------------------------------------------------------------------------
namespace detail {

template<typename F, typename = void>
struct is_step_model_2d : std::false_type {};

template<typename F>
struct is_step_model_2d<F, std::void_t<decltype(
    static_cast<StepResult2D>(std::declval<F&>()(
        std::declval<double>(), std::declval<double>(),
        std::declval<double>(), std::declval<double>(),
        std::declval<double>(), std::declval<double>())))>>
    : std::true_type {};

} // namespace detail

template<typename F>
inline constexpr bool is_step_model_2d_v = detail::is_step_model_2d<std::decay_t<F>>::value;

// ---------------------------------------------------------------------------
// Heston Euler step with full truncation (log-Euler for spot, Euler for var).
//
//   v_pos   = max(v, 0)                          // full truncation
//   sqrt_v  = sqrt(v_pos)
//   dw1     = sqrt_dt * z1
//   dw2     = sqrt_dt * (rho * z1 + sqrt(1 - rho^2) * z2)
//   s_next  = s * exp((r - q - 0.5 * v_pos) * dt + sqrt_v * dw1)
//   v_next  = max(0, v + kappa * (theta - v_pos) * dt + sigma * sqrt_v * dw2)
//
// Parameters: r (risk-free), q (dividend), kappa (mean-reversion speed),
//             theta (long-run variance), sigma (vol-of-vol), rho (correlation).
// ---------------------------------------------------------------------------
inline auto make_heston_euler_step(double r, double q,
                                   double kappa, double theta,
                                   double sigma, double rho) {
    const double rho_comp = std::sqrt(std::max(1.0 - rho * rho, 0.0));
    auto f = [r, q, kappa, theta, sigma, rho, rho_comp](
        double s, double v, double dt, double sqrt_dt,
        double z1, double z2) -> StepResult2D
    {
        const double v_pos  = std::max(v, 0.0);
        const double sqrt_v = std::sqrt(v_pos);
        const double dw1 = sqrt_dt * z1;
        const double dw2 = sqrt_dt * (rho * z1 + rho_comp * z2);
        const double s_next = s * std::exp((r - q - 0.5 * v_pos) * dt + sqrt_v * dw1);
        const double v_next = std::max(0.0, v + kappa * (theta - v_pos) * dt
                                             + sigma * sqrt_v * dw2);
        return {std::max(s_next, 1e-12), v_next};
    };
    static_assert(is_step_model_2d_v<decltype(f)>,
        "make_heston_euler_step must return a callable matching "
        "(double, double, double, double, double, double) -> StepResult2D");
    return f;
}

// ===========================================================================
// Local volatility model factories.
//
// The local vol step uses a callable sigma_fn(s, t) to determine volatility
// at each point in the (spot, time) plane.  The factory is a template so
// that the callable inlines fully under LTO — no std::function overhead.
// ===========================================================================

// ---------------------------------------------------------------------------
// Constant local vol wrapper.  When injected into the local vol pricer,
// produces results identical to BSM Euler discretization.
// ---------------------------------------------------------------------------
inline auto make_local_vol_constant(double vol) {
    return [vol](double /*s*/, double /*t*/) -> double { return vol; };
}

// ---------------------------------------------------------------------------
// Local vol Euler step factory.
//
// Returns a callable with signature:
//   (double s, double t, double dt, double dw) -> double s_next
//
// Note: this is NOT a StepModel (which takes (s, dt, dw)).  The extra `t`
// parameter lets sigma_fn depend on calendar time.  The local vol pricer
// manages the time counter and calls this step directly.
//
// Euler discretization:
//   s_next = s + (r - q) * s * dt + sigma_fn(s, t) * s * dw
// ---------------------------------------------------------------------------
template<typename SigmaFn>
auto make_local_vol_euler_step(double r, double q, SigmaFn sigma_fn) {
    auto f = [r, q, sigma_fn = std::move(sigma_fn)](
        double s, double t, double dt, double dw) -> double
    {
        const double sig = sigma_fn(s, t);
        return std::max(1e-12, s + (r - q) * s * dt + sig * s * dw);
    };
    return f;
}

} // namespace qk::models

#endif /* QK_MODEL_CONCEPTS_H */
