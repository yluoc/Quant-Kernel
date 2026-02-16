#include <cmath>

#include <quantkernel/qk_abi.h>
#include "algorithms/closed_form_semi_analytical/closed_form_models.h"

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
// BSM put-call parity: |C - P - (S*e^{-qT} - K*e^{-rT})| < tol
// ---------------------------------------------------------------------------
void BsmPutCallParity(double S, double K, double T, double vol, double r, double q) {
    double call = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    double put  = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_PUT);
    double lhs  = call - put;
    double rhs  = S * std::exp(-q * T) - K * std::exp(-r * T);
    EXPECT_NEAR(lhs, rhs, 1e-8);
}
FUZZ_TEST(ClosedForm, BsmPutCallParity)
    .WithDomains(fuzztest::InRange(50.0, 200.0),   // S
                 fuzztest::InRange(50.0, 200.0),   // K
                 fuzztest::InRange(0.1, 3.0),      // T
                 fuzztest::InRange(0.05, 1.0),     // vol
                 fuzztest::InRange(0.0, 0.15),     // r
                 fuzztest::InRange(0.0, 0.15));     // q

// ---------------------------------------------------------------------------
// BSM call & put are non-negative
// ---------------------------------------------------------------------------
void BsmNonNegative(double S, double K, double T, double vol, double r, double q,
                    int option_type) {
    double price = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_GE(price, -1e-10);
}
FUZZ_TEST(ClosedForm, BsmNonNegative)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

// ---------------------------------------------------------------------------
// Black76 matches BSM when F = S*e^{(r-q)T}
// ---------------------------------------------------------------------------
void Black76MatchesBsm(double S, double K, double T, double vol, double r, double q) {
    double F       = S * std::exp((r - q) * T);
    double bsm     = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    double b76     = qk::cfa::black76_price(F, K, T, vol, r, QK_CALL);
    EXPECT_NEAR(b76, bsm, 1e-10);
}
FUZZ_TEST(ClosedForm, Black76MatchesBsm)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15));

// ---------------------------------------------------------------------------
// Heston put-call parity
// ---------------------------------------------------------------------------
void HestonPutCallParity(double S, double K, double T, double r, double q,
                         double v0, double kappa, double theta, double sigma, double rho) {
    qk::cfa::HestonParams p{};
    p.v0 = v0; p.kappa = kappa; p.theta = theta; p.sigma = sigma; p.rho = rho;

    double call = qk::cfa::heston_price_cf(S, K, T, r, q, p, QK_CALL, 1536, 140.0);
    double put  = qk::cfa::heston_price_cf(S, K, T, r, q, p, QK_PUT,  1536, 140.0);
    double lhs  = call - put;
    double rhs  = S * std::exp(-q * T) - K * std::exp(-r * T);
    EXPECT_NEAR(lhs, rhs, 1e-1);
}
FUZZ_TEST(ClosedForm, HestonPutCallParity)
    .WithDomains(fuzztest::InRange(50.0, 200.0),    // S
                 fuzztest::InRange(50.0, 200.0),    // K
                 fuzztest::InRange(0.1, 3.0),       // T
                 fuzztest::InRange(0.0, 0.15),      // r
                 fuzztest::InRange(0.0, 0.15),      // q
                 fuzztest::InRange(0.01, 0.25),     // v0
                 fuzztest::InRange(0.5, 5.0),       // kappa
                 fuzztest::InRange(0.01, 0.25),     // theta
                 fuzztest::InRange(0.1, 1.0),       // sigma
                 fuzztest::InRange(-0.9, -0.1));     // rho

// ---------------------------------------------------------------------------
// Merton reduces to BSM when lambda = 0
// ---------------------------------------------------------------------------
void MertonReducesToBsm(double S, double K, double T, double vol, double r, double q,
                        int option_type) {
    qk::cfa::MertonJumpDiffusionParams p{};
    p.jump_intensity = 0.0;
    p.jump_mean = -0.1;
    p.jump_vol = 0.2;
    p.max_terms = 80;

    double mjd = qk::cfa::merton_jump_diffusion_price(S, K, T, vol, r, q, p, option_type);
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(mjd, bsm, 1e-10);
}
FUZZ_TEST(ClosedForm, MertonReducesToBsm)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

// ---------------------------------------------------------------------------
// Variance Gamma put-call parity
// ---------------------------------------------------------------------------
void VgPutCallParity(double S, double K, double T, double r, double q) {
    qk::cfa::VarianceGammaParams p{};
    p.sigma = 0.20;
    p.theta = -0.10;
    p.nu = 0.20;

    double call = qk::cfa::variance_gamma_price_cf(S, K, T, r, q, p, QK_CALL, 1536, 140.0);
    double put  = qk::cfa::variance_gamma_price_cf(S, K, T, r, q, p, QK_PUT,  1536, 140.0);
    double lhs  = call - put;
    double rhs  = S * std::exp(-q * T) - K * std::exp(-r * T);
    EXPECT_NEAR(lhs, rhs, 1e-1);
}
FUZZ_TEST(ClosedForm, VgPutCallParity)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15));

// ---------------------------------------------------------------------------
// SABR implied vol is positive
// ---------------------------------------------------------------------------
void SabrIvPositive(double F, double K, double T) {
    qk::cfa::SABRParams p{};
    p.alpha = 0.20;
    p.beta = 0.50;
    p.rho = -0.20;
    p.nu = 0.40;

    double iv = qk::cfa::sabr_hagan_lognormal_iv(F, K, T, p);
    EXPECT_GT(iv, 0.0);
}
FUZZ_TEST(ClosedForm, SabrIvPositive)
    .WithDomains(fuzztest::InRange(50.0, 200.0),    // F
                 fuzztest::InRange(50.0, 200.0),    // K
                 fuzztest::InRange(0.1, 3.0));       // T

// ---------------------------------------------------------------------------
// Dupire recovers constant BS vol (fixed-param regression, not fuzzed)
// ---------------------------------------------------------------------------
TEST(ClosedForm, DupireRecoversConstantVol) {
    const double S = 100.0;
    const double sigma = 0.25;
    const double r = 0.02;
    const double q = 0.01;

    auto call_surface = [&](double K, double T) {
        return qk::cfa::black_scholes_merton_price(S, K, T, sigma, r, q, QK_CALL);
    };

    double local = qk::cfa::dupire_local_vol_fd(call_surface, 100.0, 1.0, r, q, 0.5, 1e-3);
    EXPECT_GT(local, 0.0);
    EXPECT_NEAR(local, sigma, 0.02);
}
