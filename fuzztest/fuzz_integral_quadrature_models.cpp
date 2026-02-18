#include <cmath>

#include <quantkernel/qk_abi.h>
#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/integral_quadrature/integral_quadrature_models.h"

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"

void GaussHermiteMatchesBsm(double S, double K, double T, double vol, double r, double q,
                            int option_type) {
    qk::iqm::GaussHermiteParams p{};
    p.n_points = 128;

    double quad = qk::iqm::gauss_hermite_price(S, K, T, vol, r, q, option_type, p);
    if (!std::isfinite(quad)) return;

    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(quad, bsm, 0.25);
}
FUZZ_TEST(IntegralQuadrature, GaussHermiteMatchesBsm)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

void LaguerreLegendreAdaptiveAreFiniteAndNearBsm(double S, double K, double T, double vol,
                                                  double r, double q, int option_type) {
    qk::iqm::GaussLaguerreParams gl{};
    gl.n_points = 64;
    qk::iqm::GaussLegendreParams gg{};
    gg.n_points = 128;
    gg.integration_limit = 200.0;
    qk::iqm::AdaptiveQuadratureParams aq{};
    aq.abs_tol = 1e-9;
    aq.rel_tol = 1e-8;
    aq.max_depth = 14;
    aq.integration_limit = 200.0;

    double lag = qk::iqm::gauss_laguerre_price(S, K, T, vol, r, q, option_type, gl);
    double leg = qk::iqm::gauss_legendre_price(S, K, T, vol, r, q, option_type, gg);
    double adp = qk::iqm::adaptive_quadrature_price(S, K, T, vol, r, q, option_type, aq);

    EXPECT_TRUE(std::isfinite(lag));
    EXPECT_TRUE(std::isfinite(leg));
    EXPECT_TRUE(std::isfinite(adp));

    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(lag, bsm, 0.1);
    EXPECT_NEAR(leg, bsm, 0.02);
    EXPECT_NEAR(adp, bsm, 0.02);
}
FUZZ_TEST(IntegralQuadrature, LaguerreLegendreAdaptiveAreFiniteAndNearBsm)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

void IntegralQuadraturePutCallParity(double S, double K, double T, double vol,
                                     double r, double q, int method_id) {
    double call = 0.0;
    double put = 0.0;

    if (method_id == 0) {
        qk::iqm::GaussHermiteParams p{128};
        call = qk::iqm::gauss_hermite_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::iqm::gauss_hermite_price(S, K, T, vol, r, q, QK_PUT, p);
    } else if (method_id == 1) {
        qk::iqm::GaussLaguerreParams p{64};
        call = qk::iqm::gauss_laguerre_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::iqm::gauss_laguerre_price(S, K, T, vol, r, q, QK_PUT, p);
    } else if (method_id == 2) {
        qk::iqm::GaussLegendreParams p{128, 200.0};
        call = qk::iqm::gauss_legendre_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::iqm::gauss_legendre_price(S, K, T, vol, r, q, QK_PUT, p);
    } else {
        qk::iqm::AdaptiveQuadratureParams p{1e-9, 1e-8, 14, 200.0};
        call = qk::iqm::adaptive_quadrature_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::iqm::adaptive_quadrature_price(S, K, T, vol, r, q, QK_PUT, p);
    }

    if (!std::isfinite(call) || !std::isfinite(put)) return;
    double lhs = call - put;
    double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);
    EXPECT_NEAR(lhs, rhs, 2e-2);
}
FUZZ_TEST(IntegralQuadrature, IntegralQuadraturePutCallParity)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::InRange(0, 3));
