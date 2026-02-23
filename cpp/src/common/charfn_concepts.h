#ifndef QK_CHARFN_CONCEPTS_H
#define QK_CHARFN_CONCEPTS_H

#include <cmath>
#include <complex>
#include <type_traits>

namespace qk::ftm {

/// Compile-time trait: F is a CharFn if it is invocable as
///   std::complex<double> F(std::complex<double> u, double t)
template <typename F>
inline constexpr bool is_charfn_v =
    std::is_invocable_r_v<std::complex<double>, F,
                          std::complex<double> /*u*/, double /*t*/>;

namespace detail {

constexpr double kCharPi = 3.1415926535897932384626433832795;
const std::complex<double> kCharI(0.0, 1.0);

} // namespace detail

// ---------------------------------------------------------------------------
// BSM characteristic function factories
// ---------------------------------------------------------------------------

/// Log-price CF for BSM (captures spot): used by Carr-Madan, COS, Fractional FFT.
///   phi(u, t) = exp(i*u*mu - 0.5*vol^2*t*u^2)
///   where mu = log(spot) + (r - q - 0.5*vol^2)*t
inline auto make_bsm_log_charfn(double spot, double vol, double r, double q) {
    return [spot, vol, r, q](std::complex<double> u, double t) -> std::complex<double> {
        double mu = std::log(spot) + (r - q - 0.5 * vol * vol) * t;
        return std::exp(detail::kCharI * u * mu - 0.5 * vol * vol * t * u * u);
    };
}

/// Log-return CF for BSM (no spot): used by Lewis, Hilbert.
///   phi(u, t) = exp(i*u*mu - 0.5*vol^2*t*u^2)
///   where mu = (r - q - 0.5*vol^2)*t
inline auto make_bsm_logreturn_charfn(double vol, double r, double q) {
    return [vol, r, q](std::complex<double> u, double t) -> std::complex<double> {
        double mu = (r - q - 0.5 * vol * vol) * t;
        return std::exp(detail::kCharI * u * mu - 0.5 * vol * vol * t * u * u);
    };
}

// ---------------------------------------------------------------------------
// Heston characteristic function factories
// ---------------------------------------------------------------------------

/// Log-price CF for Heston (captures spot): used by Carr-Madan, COS, Fractional FFT.
///   phi(u, t) = exp(i*u*log(S) + i*u*(r-q)*t + C(u,t) + D(u,t)*v0)
inline auto make_heston_log_charfn(double spot, double r, double q,
                                   double v0, double kappa, double theta,
                                   double sigma, double rho) {
    return [spot, r, q, v0, kappa, theta, sigma, rho](
               std::complex<double> u, double t) -> std::complex<double> {
        const auto i = detail::kCharI;
        const std::complex<double> iu = i * u;

        const std::complex<double> a = kappa - rho * sigma * iu;
        const std::complex<double> b = sigma * sigma * (iu + u * u);
        const std::complex<double> d = std::sqrt(a * a + b);

        const std::complex<double> g = (a - d) / (a + d);
        const std::complex<double> edt = std::exp(-d * t);

        const std::complex<double> C =
            (kappa * theta / (sigma * sigma)) *
            ((a - d) * t - 2.0 * std::log((1.0 - g * edt) / (1.0 - g)));
        const std::complex<double> D =
            ((a - d) / (sigma * sigma)) *
            (1.0 - edt) / (1.0 - g * edt);

        return std::exp(iu * std::log(spot) + iu * (r - q) * t + C + D * v0);
    };
}

/// Log-return CF for Heston (no spot): used by Lewis, Hilbert.
///   Same as log-price CF but without the i*u*log(S) term.
inline auto make_heston_logreturn_charfn(double r, double q,
                                         double v0, double kappa, double theta,
                                         double sigma, double rho) {
    return [r, q, v0, kappa, theta, sigma, rho](
               std::complex<double> u, double t) -> std::complex<double> {
        const auto i = detail::kCharI;
        const std::complex<double> iu = i * u;

        const std::complex<double> a = kappa - rho * sigma * iu;
        const std::complex<double> b = sigma * sigma * (iu + u * u);
        const std::complex<double> d = std::sqrt(a * a + b);

        const std::complex<double> g = (a - d) / (a + d);
        const std::complex<double> edt = std::exp(-d * t);

        const std::complex<double> C =
            (kappa * theta / (sigma * sigma)) *
            ((a - d) * t - 2.0 * std::log((1.0 - g * edt) / (1.0 - g)));
        const std::complex<double> D =
            ((a - d) / (sigma * sigma)) *
            (1.0 - edt) / (1.0 - g * edt);

        return std::exp(iu * (r - q) * t + C + D * v0);
    };
}

// Static assertions to verify the factories satisfy the trait.
static_assert(is_charfn_v<decltype(make_bsm_log_charfn(0, 0, 0, 0))>);
static_assert(is_charfn_v<decltype(make_bsm_logreturn_charfn(0, 0, 0))>);
static_assert(is_charfn_v<decltype(make_heston_log_charfn(0, 0, 0, 0, 0, 0, 0, 0))>);
static_assert(is_charfn_v<decltype(make_heston_logreturn_charfn(0, 0, 0, 0, 0, 0, 0))>);

} // namespace qk::ftm

#endif /* QK_CHARFN_CONCEPTS_H */
