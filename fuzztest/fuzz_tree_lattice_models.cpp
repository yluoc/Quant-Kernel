#include <algorithm>
#include <cmath>

#include <quantkernel/qk_abi.h>
#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/tree_lattice_methods/tree_lattice_models.h"

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"

// Relative tolerance: 5% of reference price, with a floor of 1.0
static double rel_tol(double ref) {
    return std::max(std::fabs(ref) * 0.05, 1.0);
}

// ---------------------------------------------------------------------------
// CRR matches BSM for European options
// ---------------------------------------------------------------------------
void CrrMatchesBsm(double S, double K, double T, double vol, double r, double q,
                   int option_type, int steps) {
    double crr = qk::tlm::crr_price(S, K, T, vol, r, q, option_type, steps, false);
    if (std::isnan(crr) || std::isinf(crr)) return;
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(crr, bsm, rel_tol(bsm));
}
FUZZ_TEST(TreeLattice, CrrMatchesBsm)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(50, 500));

// ---------------------------------------------------------------------------
// Jarrow-Rudd matches BSM for European options
// ---------------------------------------------------------------------------
void JrMatchesBsm(double S, double K, double T, double vol, double r, double q,
                  int option_type, int steps) {
    double jr  = qk::tlm::jarrow_rudd_price(S, K, T, vol, r, q, option_type, steps, false);
    if (std::isnan(jr) || std::isinf(jr)) return;
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(jr, bsm, rel_tol(bsm));
}
FUZZ_TEST(TreeLattice, JrMatchesBsm)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(50, 500));

// ---------------------------------------------------------------------------
// Tian matches BSM for European options
// ---------------------------------------------------------------------------
void TianMatchesBsm(double S, double K, double T, double vol, double r, double q,
                    int option_type, int steps) {
    double tian = qk::tlm::tian_price(S, K, T, vol, r, q, option_type, steps, false);
    if (std::isnan(tian) || std::isinf(tian)) return;
    double bsm  = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(tian, bsm, rel_tol(bsm));
}
FUZZ_TEST(TreeLattice, TianMatchesBsm)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(50, 500));

// ---------------------------------------------------------------------------
// Leisen-Reimer matches BSM for European options (steps must be odd)
// ---------------------------------------------------------------------------
void LrMatchesBsm(double S, double K, double T, double vol, double r, double q,
                  int option_type, int steps) {
    // LR requires odd steps
    if (steps % 2 == 0) steps += 1;
    double lr  = qk::tlm::leisen_reimer_price(S, K, T, vol, r, q, option_type, steps, false);
    if (std::isnan(lr) || std::isinf(lr)) return;  // skip degenerate inputs
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(lr, bsm, rel_tol(bsm));
}
FUZZ_TEST(TreeLattice, LrMatchesBsm)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(50, 500));

// ---------------------------------------------------------------------------
// Trinomial tree matches BSM for European options
// ---------------------------------------------------------------------------
void TrinomialMatchesBsm(double S, double K, double T, double vol, double r, double q,
                         int option_type, int steps) {
    double tri = qk::tlm::trinomial_tree_price(S, K, T, vol, r, q, option_type, steps, false);
    if (std::isnan(tri) || std::isinf(tri)) return;
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(tri, bsm, rel_tol(bsm));
}
FUZZ_TEST(TreeLattice, TrinomialMatchesBsm)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}),
                 fuzztest::InRange(50, 500));

// ---------------------------------------------------------------------------
// American put >= European put (CRR)
// ---------------------------------------------------------------------------
void AmericanPutGeqEuropeanPut(double S, double K, double T, double vol, double r, double q,
                               int steps) {
    double eur = qk::tlm::crr_price(S, K, T, vol, r, q, QK_PUT, steps, false);
    double amr = qk::tlm::crr_price(S, K, T, vol, r, q, QK_PUT, steps, true);
    EXPECT_GE(amr, eur - 1e-10);
}
FUZZ_TEST(TreeLattice, AmericanPutGeqEuropeanPut)
    .WithDomains(fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(50.0, 200.0),
                 fuzztest::InRange(0.1, 3.0),
                 fuzztest::InRange(0.05, 1.0),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(0.0, 0.15),
                 fuzztest::InRange(50, 500));

// ---------------------------------------------------------------------------
// Derman-Kani implied tree matches BSM with constant local vol
// (fixed params, not fuzzed — uses lambda)
// ---------------------------------------------------------------------------
TEST(TreeLattice, DermanKaniConstLocalVolMatchesBsm) {
    double S = 100.0, K = 100.0, T = 1.0, vol = 0.2, r = 0.03, q = 0.01;
    auto local_vol = [vol](double /*spot*/, double /*time*/) { return vol; };
    qk::tlm::ImpliedTreeConfig cfg{};
    cfg.steps = 12;
    cfg.american_style = false;
    double dk  = qk::tlm::derman_kani_implied_tree_price(S, K, T, r, q, QK_CALL, local_vol, cfg);
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    EXPECT_NEAR(dk, bsm, 2.5e-1);
}
