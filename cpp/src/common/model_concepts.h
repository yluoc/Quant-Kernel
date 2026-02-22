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

} // namespace qk::models

#endif /* QK_MODEL_CONCEPTS_H */
