#include "test_harness.h"
#include <quantkernel/qk_api.h>
#include <cmath>

QK_TEST(carr_madan_fft_near_bsm) {
    double bsm = qk_cf_black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL);
    double fft = qk_ftm_carr_madan_fft_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 4096, 0.25, 1.5);
    QK_ASSERT_NEAR(fft, bsm, 0.1);
}

QK_TEST(cos_method_near_bsm) {
    double bsm = qk_cf_black_scholes_merton_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL);
    double cos_price = qk_ftm_cos_fang_oosterlee_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 256, 10.0);
    QK_ASSERT_NEAR(cos_price, bsm, 0.1);
}

QK_TEST(lewis_fourier_positive) {
    double price = qk_ftm_lewis_fourier_inversion_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 4096, 300.0);
    QK_ASSERT_TRUE(price > 0.0);
}

QK_TEST(hilbert_transform_positive) {
    double price = qk_ftm_hilbert_transform_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 4096, 300.0);
    QK_ASSERT_TRUE(price > 0.0);
}

QK_TEST(carr_madan_batch_matches_scalar) {
    const int32_t n = 4;
    double spot[] = {100, 105, 95, 110};
    double strike[] = {100, 100, 100, 100};
    double t[] = {1.0, 1.0, 1.0, 0.5};
    double vol[] = {0.2, 0.2, 0.3, 0.2};
    double r[] = {0.05, 0.05, 0.05, 0.05};
    double q[] = {0.0, 0.0, 0.0, 0.02};
    int32_t ot[] = {QK_CALL, QK_PUT, QK_CALL, QK_PUT};
    int32_t gs[] = {4096, 4096, 4096, 4096};
    double eta[] = {0.25, 0.25, 0.25, 0.25};
    double alpha[] = {1.5, 1.5, 1.5, 1.5};
    double out[n];

    int32_t rc = qk_ftm_carr_madan_fft_price_batch(spot, strike, t, vol, r, q, ot, gs, eta, alpha, n, out);
    QK_ASSERT_EQ(rc, QK_OK);

    for (int32_t i = 0; i < n; ++i) {
        double scalar = qk_ftm_carr_madan_fft_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], ot[i], gs[i], eta[i], alpha[i]);
        QK_ASSERT_NEAR(out[i], scalar, 1e-12);
    }
}

QK_TEST_MAIN()
