#include "test_harness.h"
#include <quantkernel/qk_api.h>
#include <cmath>

QK_TEST(standard_mc_positive_price) {
    double price = qk_mcm_standard_monte_carlo_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 100000, 42);
    QK_ASSERT_TRUE(price > 0.0);
}

QK_TEST(standard_mc_near_bsm) {
    double bsm = qk_cf_black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL);
    double mc = qk_mcm_standard_monte_carlo_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 500000, 42);
    QK_ASSERT_NEAR(mc, bsm, 0.5);
}

QK_TEST(euler_maruyama_positive) {
    double price = qk_mcm_euler_maruyama_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 10000, 100, 42);
    QK_ASSERT_TRUE(price > 0.0);
}

QK_TEST(milstein_positive) {
    double price = qk_mcm_milstein_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_PUT, 10000, 100, 42);
    QK_ASSERT_TRUE(price > 0.0);
}

QK_TEST(standard_mc_batch_matches_scalar) {
    const int32_t n = 4;
    double spot[] = {100, 105, 95, 110};
    double strike[] = {100, 100, 100, 100};
    double t[] = {1.0, 1.0, 1.0, 0.5};
    double vol[] = {0.2, 0.2, 0.3, 0.2};
    double r[] = {0.05, 0.05, 0.05, 0.05};
    double q[] = {0.0, 0.0, 0.0, 0.02};
    int32_t ot[] = {QK_CALL, QK_PUT, QK_CALL, QK_PUT};
    int32_t paths[] = {10000, 10000, 10000, 10000};
    uint64_t seed[] = {42, 43, 44, 45};
    double out[n];

    int32_t rc = qk_mcm_standard_monte_carlo_price_batch(spot, strike, t, vol, r, q, ot, paths, seed, n, out);
    QK_ASSERT_EQ(rc, QK_OK);

    for (int32_t i = 0; i < n; ++i) {
        double scalar = qk_mcm_standard_monte_carlo_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], ot[i], paths[i], seed[i]);
        QK_ASSERT_NEAR(out[i], scalar, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Golden-seed regression tests: pin absolute numerical outputs for fixed
// inputs and seeds.  Any change to RNG draw order, antithetic pairing, or
// accumulation logic will break these.
//
// Tolerance: 1e-12 (near machine-epsilon).  We use bit-reproducible mt19937_64
// seeded deterministically; the only source of variation would be a change in
// the code path itself, not floating-point non-determinism.
// ---------------------------------------------------------------------------
QK_TEST(golden_standard_mc_call) {
    double price = qk_mcm_standard_monte_carlo_price(
        100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 50000, 12345);
    QK_ASSERT_NEAR(price, 10.533371511490282, 1e-12);
}

QK_TEST(golden_standard_mc_put) {
    double price = qk_mcm_standard_monte_carlo_price(
        100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_PUT, 50000, 12345);
    QK_ASSERT_NEAR(price, 5.6296577522043556, 1e-12);
}

QK_TEST(golden_euler_maruyama_call) {
    double price = qk_mcm_euler_maruyama_price(
        100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 10000, 100, 12345);
    QK_ASSERT_NEAR(price, 10.218359279533905, 1e-12);
}

QK_TEST_MAIN()
