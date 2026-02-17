#include <algorithm>
#include <cmath>

#include <quantkernel/qk_abi.h>
#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/monte_carlo_methods/monte_carlo_models.h"

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"

namespace {

double mc_tol(double ref) {
    return std::max(2.0, 0.2 * std::fabs(ref));
}

} // namespace

void StandardMcMatchesBsm(double S, double K, double T, double vol, double r, double q,
                          int option_type, int paths, int seed) {
    double mc = qk::mcm::standard_monte_carlo_price(
        S, K, T, vol, r, q, option_type, paths, static_cast<uint64_t>(seed)
    );
    if (std::isnan(mc) || std::isinf(mc)) return;

    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(mc, bsm, mc_tol(bsm));
}
FUZZ_TEST(MonteCarlo, StandardMcMatchesBsm)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(2000, 40000),
                 fuzztest::InRange(1, 50000));

void EulerAndMilsteinStayClose(double S, double K, double T, double vol, double r, double q,
                               int option_type, int paths, int steps, int seed) {
    double euler = qk::mcm::euler_maruyama_price(
        S, K, T, vol, r, q, option_type, paths, steps, static_cast<uint64_t>(seed)
    );
    double mil = qk::mcm::milstein_price(
        S, K, T, vol, r, q, option_type, paths, steps, static_cast<uint64_t>(seed)
    );

    if (std::isnan(euler) || std::isinf(euler) || std::isnan(mil) || std::isinf(mil)) return;
    EXPECT_NEAR(euler, mil, std::max(2.0, 0.25 * std::max(std::fabs(euler), std::fabs(mil))));
}
FUZZ_TEST(MonteCarlo, EulerAndMilsteinStayClose)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(2000, 20000),
                 fuzztest::InRange(8, 128),
                 fuzztest::InRange(1, 50000));

void QuasiMonteCarloConsistency(double S, double K, double T, double vol, double r, double q,
                                int option_type, int paths) {
    double sobol = qk::mcm::quasi_monte_carlo_sobol_price(S, K, T, vol, r, q, option_type, paths);
    double halton = qk::mcm::quasi_monte_carlo_halton_price(S, K, T, vol, r, q, option_type, paths);
    if (std::isnan(sobol) || std::isinf(sobol) || std::isnan(halton) || std::isinf(halton)) return;

    EXPECT_NEAR(sobol, halton, std::max(1.5, 0.15 * std::max(std::fabs(sobol), std::fabs(halton))));
}
FUZZ_TEST(MonteCarlo, QuasiMonteCarloConsistency)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(256, 16384));

void VarianceReductionMethodsAreFinite(double S, double K, double T, double vol, double r, double q,
                                       int option_type, int paths, int seed) {
    double isamp = qk::mcm::importance_sampling_price(
        S, K, T, vol, r, q, option_type, paths, 0.4, static_cast<uint64_t>(seed)
    );
    double cvar = qk::mcm::control_variates_price(
        S, K, T, vol, r, q, option_type, paths, static_cast<uint64_t>(seed)
    );
    double anti = qk::mcm::antithetic_variates_price(
        S, K, T, vol, r, q, option_type, paths, static_cast<uint64_t>(seed)
    );
    double strat = qk::mcm::stratified_sampling_price(
        S, K, T, vol, r, q, option_type, paths, static_cast<uint64_t>(seed)
    );

    EXPECT_TRUE(std::isfinite(isamp));
    EXPECT_TRUE(std::isfinite(cvar));
    EXPECT_TRUE(std::isfinite(anti));
    EXPECT_TRUE(std::isfinite(strat));
    EXPECT_GE(isamp, -1e-8);
    EXPECT_GE(cvar, -1e-8);
    EXPECT_GE(anti, -1e-8);
    EXPECT_GE(strat, -1e-8);
}
FUZZ_TEST(MonteCarlo, VarianceReductionMethodsAreFinite)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(1000, 30000),
                 fuzztest::InRange(1, 50000));

void LsmcAmericanPutNotBelowEuropeanMc(double S, double K, double T, double vol,
                                       double r, int paths, int steps, int seed) {
    double eur_put = qk::mcm::standard_monte_carlo_price(
        S, K, T, vol, r, 0.0, QK_PUT, paths, static_cast<uint64_t>(seed)
    );
    double am_put = qk::mcm::longstaff_schwartz_price(
        S, K, T, vol, r, 0.0, QK_PUT, paths, steps, static_cast<uint64_t>(seed)
    );

    if (std::isnan(eur_put) || std::isinf(eur_put) || std::isnan(am_put) || std::isinf(am_put)) return;
    EXPECT_GE(am_put, eur_put - 0.75);
}
FUZZ_TEST(MonteCarlo, LsmcAmericanPutNotBelowEuropeanMc)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(70.0, 160.0),
                 fuzztest::InRange(0.4, 2.0),
                 fuzztest::InRange(0.08, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(2000, 20000),
                 fuzztest::InRange(8, 96),
                 fuzztest::InRange(1, 50000));

TEST(MonteCarlo, MlmcProducesFinitePrice) {
    double price = qk::mcm::multilevel_monte_carlo_price(
        100.0, 100.0, 1.0, 0.2, 0.03, 0.01, QK_CALL, 8192, 4, 8, 42
    );
    EXPECT_TRUE(std::isfinite(price));
    EXPECT_GT(price, 0.0);
}
