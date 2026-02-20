#include "test_harness.h"
#include <quantkernel/qk_api.h>
#include <cstring>

QK_TEST(null_pointer_returns_error) {
    double out[1];
    int32_t ot[] = {QK_CALL};
    int32_t rc = qk_cf_black_scholes_merton_price_batch(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, ot, 1, out);
    QK_ASSERT_EQ(rc, QK_ERR_NULL_PTR);
}

QK_TEST(null_output_returns_error) {
    double spot[] = {100.0};
    double strike[] = {100.0};
    double t[] = {1.0};
    double vol[] = {0.2};
    double r[] = {0.05};
    double q[] = {0.0};
    int32_t ot[] = {QK_CALL};
    int32_t rc = qk_cf_black_scholes_merton_price_batch(spot, strike, t, vol, r, q, ot, 1, nullptr);
    QK_ASSERT_EQ(rc, QK_ERR_NULL_PTR);
}

QK_TEST(bad_size_returns_error) {
    double spot[] = {100.0};
    double strike[] = {100.0};
    double t[] = {1.0};
    double vol[] = {0.2};
    double r[] = {0.05};
    double q[] = {0.0};
    int32_t ot[] = {QK_CALL};
    double out[1];
    int32_t rc = qk_cf_black_scholes_merton_price_batch(spot, strike, t, vol, r, q, ot, -3, out);
    QK_ASSERT_EQ(rc, QK_ERR_BAD_SIZE);
}

QK_TEST(zero_size_returns_error) {
    double spot[] = {100.0};
    double strike[] = {100.0};
    double t[] = {1.0};
    double vol[] = {0.2};
    double r[] = {0.05};
    double q[] = {0.0};
    int32_t ot[] = {QK_CALL};
    double out[1];
    int32_t rc = qk_cf_black_scholes_merton_price_batch(spot, strike, t, vol, r, q, ot, 0, out);
    QK_ASSERT_EQ(rc, QK_ERR_BAD_SIZE);
}

QK_TEST(get_last_error_after_null) {
    double out[1];
    qk_clear_last_error();
    qk_cf_black_scholes_merton_price_batch(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, out);
    const char* msg = qk_get_last_error();
    QK_ASSERT_TRUE(std::strlen(msg) > 0);
}

QK_TEST(null_ptr_identifies_correct_parameter) {
    double spot[] = {100.0};
    double strike[] = {100.0};
    double t[] = {1.0};
    double vol[] = {0.2};
    double r[] = {0.05};
    double q[] = {0.0};
    int32_t ot[] = {QK_CALL};
    double out[1];

    qk_clear_last_error();
    int32_t rc = qk_cf_black_scholes_merton_price_batch(spot, strike, t, nullptr, r, q, ot, 1, out);
    QK_ASSERT_EQ(rc, QK_ERR_NULL_PTR);
    const char* msg = qk_get_last_error();
    QK_ASSERT_TRUE(std::strstr(msg, "null") != nullptr);
    QK_ASSERT_TRUE(std::strstr(msg, "spot") == nullptr);

    qk_clear_last_error();
    rc = qk_cf_black_scholes_merton_price_batch(spot, strike, t, vol, r, q, ot, 1, nullptr);
    QK_ASSERT_EQ(rc, QK_ERR_NULL_PTR);
    msg = qk_get_last_error();
    QK_ASSERT_TRUE(std::strstr(msg, "out_prices") != nullptr);
}

QK_TEST(get_last_error_after_bad_size) {
    double spot[] = {100.0};
    double strike[] = {100.0};
    double t[] = {1.0};
    double vol[] = {0.2};
    double r[] = {0.05};
    double q[] = {0.0};
    int32_t ot[] = {QK_CALL};
    double out[1];
    qk_clear_last_error();
    qk_cf_black_scholes_merton_price_batch(spot, strike, t, vol, r, q, ot, -3, out);
    const char* msg = qk_get_last_error();
    QK_ASSERT_TRUE(std::strstr(msg, "bad batch size") != nullptr);
}

QK_TEST(clear_error_works) {
    qk_clear_last_error();
    const char* msg = qk_get_last_error();
    QK_ASSERT_EQ(std::strlen(msg), (size_t)0);
}

QK_TEST(abi_version_valid) {
    int32_t major = 0, minor = 0;
    qk_abi_version(&major, &minor);
    QK_ASSERT_EQ(major, QK_ABI_MAJOR);
    QK_ASSERT_TRUE(minor >= 9);
}

QK_TEST_MAIN()
