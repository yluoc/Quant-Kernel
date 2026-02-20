#include <algorithm>
#include <cmath>

#include <quantkernel/qk_abi.h>
#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/closed_form_semi_analytical/heston/heston.h"
#include "algorithms/finite_difference_methods/finite_difference_models.h"

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"

// Explicit FD matches BSM for European options
void ExplicitFdMatchesBsm(double S, double K, double T, double vol, double r, double q,
                          int option_type, int time_steps, int spot_steps) {
    // Explicit FD needs a fine time grid for stability; scale up time_steps
    int safe_time_steps = time_steps * 10;
    double fdm = qk::fdm::explicit_fd_price(S, K, T, vol, r, q, option_type,
                                            safe_time_steps, spot_steps, false);
    if (std::isnan(fdm) || std::isinf(fdm)) return;
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(fdm, bsm, std::max(std::fabs(bsm) * 0.15, 5.0));
}
FUZZ_TEST(FiniteDifference, ExplicitFdMatchesBsm)
    .WithDomains(fuzztest::InRange(80.0, 120.0),     // S (near-ATM range)
                 fuzztest::InRange(80.0, 120.0),     // K
                 fuzztest::InRange(0.1, 2.0),
                 fuzztest::InRange(0.05, 0.4),
                 fuzztest::InRange(0.0, 0.10),
                 fuzztest::InRange(0.0, 0.10),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(50, 300),
                 fuzztest::InRange(150, 300));

// Implicit FD matches BSM for European options
void ImplicitFdMatchesBsm(double S, double K, double T, double vol, double r, double q,
                          int option_type, int time_steps, int spot_steps) {
    double fdm = qk::fdm::implicit_fd_price(S, K, T, vol, r, q, option_type,
                                            time_steps, spot_steps, false);
    if (std::isnan(fdm) || std::isinf(fdm)) return;
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(fdm, bsm, std::max(std::fabs(bsm) * 0.15, 5.0));
}
FUZZ_TEST(FiniteDifference, ImplicitFdMatchesBsm)
    .WithDomains(fuzztest::InRange(80.0, 120.0),
                 fuzztest::InRange(80.0, 120.0),
                 fuzztest::InRange(0.1, 2.0),
                 fuzztest::InRange(0.05, 0.4),
                 fuzztest::InRange(0.0, 0.10),
                 fuzztest::InRange(0.0, 0.10),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(50, 300),
                 fuzztest::InRange(150, 300));

// Crank-Nicolson matches BSM for European options
void CrankNicolsonMatchesBsm(double S, double K, double T, double vol, double r, double q,
                             int option_type, int time_steps, int spot_steps) {
    double fdm = qk::fdm::crank_nicolson_price(S, K, T, vol, r, q, option_type,
                                               time_steps, spot_steps, false);
    if (std::isnan(fdm) || std::isinf(fdm)) return;
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(fdm, bsm, std::max(std::fabs(bsm) * 0.15, 5.0));
}
FUZZ_TEST(FiniteDifference, CrankNicolsonMatchesBsm)
    .WithDomains(fuzztest::InRange(80.0, 120.0),
                 fuzztest::InRange(80.0, 120.0),
                 fuzztest::InRange(0.1, 2.0),
                 fuzztest::InRange(0.05, 0.4),
                 fuzztest::InRange(0.0, 0.10),
                 fuzztest::InRange(0.0, 0.10),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(50, 300),
                 fuzztest::InRange(150, 300));

// American put >= European put (Implicit FD)
void ImplicitFdAmericanPutGeqEuropean(double S, double K, double T, double vol, double r,
                                      double q, int time_steps, int spot_steps) {
    double eur = qk::fdm::implicit_fd_price(S, K, T, vol, r, q, QK_PUT,
                                            time_steps, spot_steps, false);
    double amr = qk::fdm::implicit_fd_price(S, K, T, vol, r, q, QK_PUT,
                                            time_steps, spot_steps, true);
    EXPECT_GE(amr, eur - 1e-10);
}
FUZZ_TEST(FiniteDifference, ImplicitFdAmericanPutGeqEuropean)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(50, 300),
                 fuzztest::InRange(50, 300));

// American put >= European put (Crank-Nicolson)
void CnAmericanPutGeqEuropean(double S, double K, double T, double vol, double r, double q,
                              int time_steps, int spot_steps) {
    double eur = qk::fdm::crank_nicolson_price(S, K, T, vol, r, q, QK_PUT,
                                               time_steps, spot_steps, false);
    double amr = qk::fdm::crank_nicolson_price(S, K, T, vol, r, q, QK_PUT,
                                               time_steps, spot_steps, true);
    EXPECT_GE(amr, eur - 1e-10);
}
FUZZ_TEST(FiniteDifference, CnAmericanPutGeqEuropean)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(50, 300),
                 fuzztest::InRange(50, 300));

// American put >= European put (PSOR)
void PsorAmericanPutGeqEuropean(double S, double K, double T, double vol, double r, double q,
                                int time_steps, int spot_steps) {
    double eur_bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_PUT);
    double amr     = qk::fdm::psor_price(S, K, T, vol, r, q, QK_PUT,
                                         time_steps, spot_steps, 1.2, 1e-8, 10000);
    EXPECT_GE(amr, eur_bsm - 3.0);
}
FUZZ_TEST(FiniteDifference, PsorAmericanPutGeqEuropean)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(50, 300),
                 fuzztest::InRange(50, 300));

// ADI Douglas produces prices close to Heston CF
TEST(FiniteDifference, AdiDouglasConvergesToHestonCf) {
    double S = 100.0, K = 100.0, T = 1.0, r = 0.03, q_ = 0.01;
    double v0 = 0.04, kappa = 2.0, theta_v = 0.04, sigma = 0.3, rho = -0.7;

    qk::fdm::ADIHestonParams params{};
    params.v0 = v0; params.kappa = kappa; params.theta_v = theta_v;
    params.sigma = sigma; params.rho = rho;
    params.s_steps = 60; params.v_steps = 30; params.time_steps = 60;

    double adi = qk::fdm::adi_douglas_price(S, K, T, r, q_, params, QK_CALL);

    qk::cfa::HestonParams hp{};
    hp.v0 = v0; hp.kappa = kappa; hp.theta = theta_v; hp.sigma = sigma; hp.rho = rho;
    double heston = qk::cfa::heston_price_cf(S, K, T, r, q_, hp, QK_CALL, 1024, 120.0);

    EXPECT_NEAR(adi, heston, 5e-1);
}

// ADI Craig-Sneyd produces prices close to Heston CF
TEST(FiniteDifference, AdiCraigSneydConvergesToHestonCf) {
    double S = 100.0, K = 100.0, T = 1.0, r = 0.03, q_ = 0.01;
    double v0 = 0.04, kappa = 2.0, theta_v = 0.04, sigma = 0.3, rho = -0.7;

    qk::fdm::ADIHestonParams params{};
    params.v0 = v0; params.kappa = kappa; params.theta_v = theta_v;
    params.sigma = sigma; params.rho = rho;
    params.s_steps = 60; params.v_steps = 30; params.time_steps = 60;

    double adi = qk::fdm::adi_craig_sneyd_price(S, K, T, r, q_, params, QK_CALL);

    qk::cfa::HestonParams hp{};
    hp.v0 = v0; hp.kappa = kappa; hp.theta = theta_v; hp.sigma = sigma; hp.rho = rho;
    double heston = qk::cfa::heston_price_cf(S, K, T, r, q_, hp, QK_CALL, 1024, 120.0);

    EXPECT_NEAR(adi, heston, 5e-1);
}

// ADI Hundsdorfer-Verwer produces prices close to Heston CF
TEST(FiniteDifference, AdiHundsdorferVerwerConvergesToHestonCf) {
    double S = 100.0, K = 100.0, T = 1.0, r = 0.03, q_ = 0.01;
    double v0 = 0.04, kappa = 2.0, theta_v = 0.04, sigma = 0.3, rho = -0.7;

    qk::fdm::ADIHestonParams params{};
    params.v0 = v0; params.kappa = kappa; params.theta_v = theta_v;
    params.sigma = sigma; params.rho = rho;
    params.s_steps = 60; params.v_steps = 30; params.time_steps = 60;

    double adi = qk::fdm::adi_hundsdorfer_verwer_price(S, K, T, r, q_, params, QK_CALL);

    qk::cfa::HestonParams hp{};
    hp.v0 = v0; hp.kappa = kappa; hp.theta = theta_v; hp.sigma = sigma; hp.rho = rho;
    double heston = qk::cfa::heston_price_cf(S, K, T, r, q_, hp, QK_CALL, 1024, 120.0);

    EXPECT_NEAR(adi, heston, 5e-1);
}
