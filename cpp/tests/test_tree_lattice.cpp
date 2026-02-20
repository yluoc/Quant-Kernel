#include "test_harness.h"
#include <quantkernel/qk_api.h>
#include <cmath>

// CRR converges to BSM as steps increase
QK_TEST(crr_converges_to_bsm) {
    double bsm = qk_cf_black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL);
    double crr_500 = qk_tlm_crr_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 500, 0);
    QK_ASSERT_NEAR(crr_500, bsm, 0.05);
}

QK_TEST(jarrow_rudd_positive_price) {
    double price = qk_tlm_jarrow_rudd_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 200, 0);
    QK_ASSERT_TRUE(price > 0.0);
}

QK_TEST(tian_positive_price) {
    double price = qk_tlm_tian_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_PUT, 200, 0);
    QK_ASSERT_TRUE(price > 0.0);
}

QK_TEST(leisen_reimer_converges_to_bsm) {
    double bsm = qk_cf_black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL);
    double lr = qk_tlm_leisen_reimer_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 201, 0);
    QK_ASSERT_NEAR(lr, bsm, 0.01);
}

QK_TEST(crr_batch_matches_scalar) {
    const int32_t n = 4;
    double spot[] = {100, 105, 95, 110};
    double strike[] = {100, 100, 100, 100};
    double t[] = {1.0, 1.0, 1.0, 0.5};
    double vol[] = {0.2, 0.2, 0.3, 0.2};
    double r[] = {0.05, 0.05, 0.05, 0.05};
    double q[] = {0.0, 0.0, 0.0, 0.02};
    int32_t ot[] = {QK_CALL, QK_PUT, QK_CALL, QK_PUT};
    int32_t steps[] = {100, 100, 100, 100};
    int32_t american[] = {0, 0, 0, 0};
    double out[n];

    int32_t rc = qk_tlm_crr_price_batch(spot, strike, t, vol, r, q, ot, steps, american, n, out);
    QK_ASSERT_EQ(rc, QK_OK);

    for (int32_t i = 0; i < n; ++i) {
        double scalar = qk_tlm_crr_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], ot[i], steps[i], american[i]);
        QK_ASSERT_NEAR(out[i], scalar, 1e-12);
    }
}

QK_TEST_MAIN()
