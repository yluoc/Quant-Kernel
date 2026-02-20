#include "test_harness.h"
#include <quantkernel/qk_api.h>
#include <cmath>

// Known BSM call value: S=100, K=100, T=1, vol=0.2, r=0.05, q=0.0
// Reference: ~10.4506
QK_TEST(bsm_call_known_value) {
    double price = qk_cf_black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL);
    QK_ASSERT_NEAR(price, 10.4506, 0.01);
}

QK_TEST(bsm_put_known_value) {
    double price = qk_cf_black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_PUT);
    QK_ASSERT_NEAR(price, 5.5735, 0.01);
}

// Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
QK_TEST(bsm_put_call_parity) {
    double S = 100.0, K = 105.0, T = 0.5, vol = 0.25, r = 0.05, q = 0.02;
    double call = qk_cf_black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    double put = qk_cf_black_scholes_merton_price(S, K, T, vol, r, q, QK_PUT);
    double parity = S * std::exp(-q * T) - K * std::exp(-r * T);
    QK_ASSERT_NEAR(call - put, parity, 1e-10);
}

// Deep ITM call ≈ S*exp(-qT) - K*exp(-rT)
QK_TEST(bsm_deep_itm_call) {
    double price = qk_cf_black_scholes_merton_price(200.0, 50.0, 1.0, 0.2, 0.05, 0.0, QK_CALL);
    double intrinsic = 200.0 - 50.0 * std::exp(-0.05);
    QK_ASSERT_NEAR(price, intrinsic, 0.5);
}

// Deep OTM call ≈ 0
QK_TEST(bsm_deep_otm_call) {
    double price = qk_cf_black_scholes_merton_price(50.0, 200.0, 0.1, 0.2, 0.05, 0.0, QK_CALL);
    QK_ASSERT_NEAR(price, 0.0, 1e-10);
}

// Black76 known value
QK_TEST(black76_call_known) {
    double price = qk_cf_black76_price(100.0, 100.0, 1.0, 0.2, 0.05, QK_CALL);
    QK_ASSERT_TRUE(price > 0.0 && price < 100.0);
}

// Bachelier known value
QK_TEST(bachelier_call_known) {
    double price = qk_cf_bachelier_price(100.0, 100.0, 1.0, 20.0, 0.05, QK_CALL);
    QK_ASSERT_TRUE(price > 0.0 && price < 100.0);
}

QK_TEST(bsm_batch_matches_scalar) {
    const int32_t n = 8;
    double spot[] = {100, 105, 95, 110, 90, 100, 100, 100};
    double strike[] = {100, 100, 100, 100, 100, 95, 105, 110};
    double t[] = {1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0};
    double vol[] = {0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2};
    double r[] = {0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05};
    double q[] = {0.0, 0.0, 0.0, 0.02, 0.02, 0.0, 0.0, 0.0};
    int32_t ot[] = {QK_CALL, QK_PUT, QK_CALL, QK_PUT, QK_CALL, QK_CALL, QK_PUT, QK_CALL};
    double out[n];

    int32_t rc = qk_cf_black_scholes_merton_price_batch(spot, strike, t, vol, r, q, ot, n, out);
    QK_ASSERT_EQ(rc, QK_OK);

    for (int32_t i = 0; i < n; ++i) {
        double scalar = qk_cf_black_scholes_merton_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], ot[i]);
        QK_ASSERT_NEAR(out[i], scalar, 1e-12);
    }
}

QK_TEST_MAIN()
