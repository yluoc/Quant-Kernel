#include "test_harness.h"
#include <quantkernel/qk_api.h>
#include <cmath>
#include <cstdint>

QK_TEST(fdm_explicit_batch_matches_scalar) {
    double spot[]           = {100,105,95,110,90,100,100,100};
    double strike[]         = {100,100,100,100,100,95,105,110};
    double t[]              = {1.0,1.0,1.0,0.5,0.5,1.0,1.0,1.0};
    double vol[]            = {0.2,0.2,0.3,0.2,0.2,0.2,0.2,0.2};
    double r[]              = {0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05};
    double q_arr[]          = {0.0,0.0,0.0,0.02,0.02,0.0,0.0,0.0};
    int32_t ot[]            = {QK_CALL,QK_PUT,QK_CALL,QK_PUT,QK_CALL,QK_CALL,QK_PUT,QK_CALL};
    int32_t time_steps[]    = {50,50,50,50,50,50,50,50};
    int32_t spot_steps[]    = {50,50,50,50,50,50,50,50};
    int32_t american_style[] = {0,1,0,1,0,0,1,0};
    double out[8];

    int32_t rc = qk_fdm_explicit_fd_price_batch(
        spot, strike, t, vol, r, q_arr, ot,
        time_steps, spot_steps, american_style, 8, out);
    QK_ASSERT_EQ(rc, QK_OK);

    for (int i = 0; i < 8; ++i) {
        double scalar = qk_fdm_explicit_fd_price(
            spot[i], strike[i], t[i], vol[i], r[i], q_arr[i],
            ot[i], time_steps[i], spot_steps[i], american_style[i]);
        QK_ASSERT_NEAR(out[i], scalar, 1e-12);
    }
}

QK_TEST(fdm_adi_douglas_batch_matches_scalar) {
    double spot[]    = {100,105,95,110};
    double strike[]  = {100,100,100,100};
    double t[]       = {1.0,0.5,1.0,0.5};
    double r[]       = {0.05,0.05,0.03,0.03};
    double q_arr[]   = {0.01,0.01,0.0,0.0};
    double v0[]      = {0.04,0.06,0.04,0.06};
    double kappa[]   = {2.0,2.0,1.5,1.5};
    double theta_v[] = {0.04,0.04,0.06,0.06};
    double sigma[]   = {0.3,0.3,0.4,0.4};
    double rho_arr[] = {-0.7,-0.5,-0.7,-0.5};
    int32_t ot[]         = {QK_CALL,QK_PUT,QK_CALL,QK_PUT};
    int32_t s_steps[]    = {20,20,20,20};
    int32_t v_steps[]    = {10,10,10,10};
    int32_t time_steps[] = {20,20,20,20};
    double out[4];

    int32_t rc = qk_fdm_adi_douglas_price_batch(
        spot, strike, t, r, q_arr, v0, kappa, theta_v, sigma, rho_arr,
        ot, s_steps, v_steps, time_steps, 4, out);
    QK_ASSERT_EQ(rc, QK_OK);

    for (int i = 0; i < 4; ++i) {
        double scalar = qk_fdm_adi_douglas_price(
            spot[i], strike[i], t[i], r[i], q_arr[i],
            v0[i], kappa[i], theta_v[i], sigma[i], rho_arr[i],
            ot[i], s_steps[i], v_steps[i], time_steps[i]);
        QK_ASSERT_NEAR(out[i], scalar, 1e-12);
    }
}

QK_TEST(iqm_gauss_hermite_batch_matches_scalar) {
    double spot[]    = {100,105,95,110,90,100,100,100};
    double strike[]  = {100,100,100,100,100,95,105,110};
    double t[]       = {1.0,1.0,1.0,0.5,0.5,1.0,1.0,1.0};
    double vol[]     = {0.2,0.2,0.3,0.2,0.2,0.2,0.2,0.2};
    double r[]       = {0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05};
    double q_arr[]   = {0.0,0.0,0.0,0.02,0.02,0.0,0.0,0.0};
    int32_t ot[]     = {QK_CALL,QK_PUT,QK_CALL,QK_PUT,QK_CALL,QK_CALL,QK_PUT,QK_CALL};
    int32_t n_points[] = {64,64,64,64,64,64,64,64};
    double out[8];

    int32_t rc = qk_iqm_gauss_hermite_price_batch(
        spot, strike, t, vol, r, q_arr, ot, n_points, 8, out);
    QK_ASSERT_EQ(rc, QK_OK);

    for (int i = 0; i < 8; ++i) {
        double scalar = qk_iqm_gauss_hermite_price(
            spot[i], strike[i], t[i], vol[i], r[i], q_arr[i],
            ot[i], n_points[i]);
        QK_ASSERT_NEAR(out[i], scalar, 1e-12);
    }
}

QK_TEST(ram_pce_batch_matches_scalar) {
    double spot[]    = {100,105,95,110,90,100,100,100};
    double strike[]  = {100,100,100,100,100,95,105,110};
    double t[]       = {1.0,1.0,1.0,0.5,0.5,1.0,1.0,1.0};
    double vol[]     = {0.2,0.2,0.3,0.2,0.2,0.2,0.2,0.2};
    double r[]       = {0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05};
    double q_arr[]   = {0.0,0.0,0.0,0.02,0.02,0.0,0.0,0.0};
    int32_t ot[]     = {QK_CALL,QK_PUT,QK_CALL,QK_PUT,QK_CALL,QK_CALL,QK_PUT,QK_CALL};
    int32_t polynomial_order[]   = {4,4,4,4,4,4,4,4};
    int32_t quadrature_points[]  = {32,32,32,32,32,32,32,32};
    double out[8];

    int32_t rc = qk_ram_polynomial_chaos_expansion_price_batch(
        spot, strike, t, vol, r, q_arr, ot,
        polynomial_order, quadrature_points, 8, out);
    QK_ASSERT_EQ(rc, QK_OK);

    for (int i = 0; i < 8; ++i) {
        double scalar = qk_ram_polynomial_chaos_expansion_price(
            spot[i], strike[i], t[i], vol[i], r[i], q_arr[i],
            ot[i], polynomial_order[i], quadrature_points[i]);
        QK_ASSERT_NEAR(out[i], scalar, 1e-12);
    }
}

QK_TEST(agm_aad_batch_matches_scalar) {
    double spot[]    = {100,105,95,110,90,100,100,100};
    double strike[]  = {100,100,100,100,100,95,105,110};
    double t[]       = {1.0,1.0,1.0,0.5,0.5,1.0,1.0,1.0};
    double vol[]     = {0.2,0.2,0.3,0.2,0.2,0.2,0.2,0.2};
    double r[]       = {0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05};
    double q_arr[]   = {0.0,0.0,0.0,0.02,0.02,0.0,0.0,0.0};
    int32_t ot[]     = {QK_CALL,QK_PUT,QK_CALL,QK_PUT,QK_CALL,QK_CALL,QK_PUT,QK_CALL};
    int32_t tape_steps[]   = {64,64,64,64,64,64,64,64};
    double regularization[] = {1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6};
    double out[8];

    int32_t rc = qk_agm_aad_delta_batch(
        spot, strike, t, vol, r, q_arr, ot,
        tape_steps, regularization, 8, out);
    QK_ASSERT_EQ(rc, QK_OK);

    for (int i = 0; i < 8; ++i) {
        double scalar = qk_agm_aad_delta(
            spot[i], strike[i], t[i], vol[i], r[i], q_arr[i],
            ot[i], tape_steps[i], regularization[i]);
        QK_ASSERT_NEAR(out[i], scalar, 1e-12);
    }
}

QK_TEST(mlm_deep_bsde_batch_matches_scalar) {
    double spot[]    = {100,105,95,110};
    double strike[]  = {100,100,100,100};
    double t[]       = {1.0,1.0,1.0,0.5};
    double vol[]     = {0.2,0.2,0.3,0.2};
    double r[]       = {0.05,0.05,0.05,0.05};
    double q_arr[]   = {0.0,0.0,0.0,0.02};
    int32_t ot[]     = {QK_CALL,QK_PUT,QK_CALL,QK_PUT};
    int32_t time_steps[]      = {10,10,10,10};
    int32_t hidden_width[]    = {16,16,16,16};
    int32_t training_epochs[] = {20,20,20,20};
    double learning_rate[]    = {5e-3,5e-3,5e-3,5e-3};
    double out[4];

    int32_t rc = qk_mlm_deep_bsde_price_batch(
        spot, strike, t, vol, r, q_arr, ot,
        time_steps, hidden_width, training_epochs, learning_rate, 4, out);
    QK_ASSERT_EQ(rc, QK_OK);

    for (int i = 0; i < 4; ++i) {
        double scalar = qk_mlm_deep_bsde_price(
            spot[i], strike[i], t[i], vol[i], r[i], q_arr[i],
            ot[i], time_steps[i], hidden_width[i],
            training_epochs[i], learning_rate[i]);
        QK_ASSERT_NEAR(out[i], scalar, 1e-12);
    }
}

QK_TEST(fdm_psor_batch_matches_scalar) {
    double spot[]    = {100,105,95,110};
    double strike[]  = {100,100,100,100};
    double t[]       = {1.0,1.0,1.0,0.5};
    double vol[]     = {0.2,0.2,0.3,0.2};
    double r[]       = {0.05,0.05,0.05,0.05};
    double q_arr[]   = {0.0,0.0,0.0,0.02};
    int32_t ot[]     = {QK_CALL,QK_PUT,QK_CALL,QK_PUT};
    int32_t time_steps[] = {50,50,50,50};
    int32_t spot_steps[] = {50,50,50,50};
    double omega[]   = {1.2,1.2,1.2,1.2};
    double tol[]     = {1e-8,1e-8,1e-8,1e-8};
    int32_t max_iter[] = {10000,10000,10000,10000};
    double out[4];

    int32_t rc = qk_fdm_psor_price_batch(
        spot, strike, t, vol, r, q_arr, ot,
        time_steps, spot_steps, omega, tol, max_iter, 4, out);
    QK_ASSERT_EQ(rc, QK_OK);

    for (int i = 0; i < 4; ++i) {
        double scalar = qk_fdm_psor_price(
            spot[i], strike[i], t[i], vol[i], r[i], q_arr[i],
            ot[i], time_steps[i], spot_steps[i],
            omega[i], tol[i], max_iter[i]);
        QK_ASSERT_NEAR(out[i], scalar, 1e-12);
    }
}

QK_TEST(batch_null_ptr_returns_error) {
    int32_t rc = qk_fdm_explicit_fd_price_batch(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, 1, nullptr);
    QK_ASSERT_EQ(rc, QK_ERR_NULL_PTR);
}

// ---------------------------------------------------------------------------
// Golden-seed regression tests for adjoint Greek estimators.
// See test_monte_carlo.cpp header comment for tolerance rationale.
// ---------------------------------------------------------------------------
QK_TEST(golden_pathwise_delta_call) {
    double delta = qk_agm_pathwise_derivative_delta(
        100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 50000, 12345);
    QK_ASSERT_NEAR(delta, 0.63559188066751005, 1e-12);
}

QK_TEST(golden_likelihood_ratio_delta_call) {
    double delta = qk_agm_likelihood_ratio_delta(
        100.0, 100.0, 1.0, 0.2, 0.05, 0.0, QK_CALL, 50000, 12345, 10.0);
    QK_ASSERT_NEAR(delta, 0.64537180232018476, 1e-12);
}

QK_TEST_MAIN()
