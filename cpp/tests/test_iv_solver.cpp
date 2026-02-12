#include <quantkernel/qk_api.h>
#include <cmath>
#include <cstdio>

extern void test_registry_add(const char* name, void (*func)());

#define TEST_IV(name)                                              \
    static void test_iv_##name();                                  \
    static struct RegIV_##name {                                   \
        static void reg() { test_registry_add("iv_" #name, test_iv_##name); } \
    } regiv_##name;                                                \
    static void test_iv_##name()

#define ASSERT_NEAR(a, b, tol)                                     \
    do {                                                           \
        double _a = (a), _b = (b), _t = (tol);                    \
        if (std::fabs(_a - _b) > _t) {                            \
            std::fprintf(stderr,                                   \
                "  FAIL %s:%d: |%.10g - %.10g| = %.10g > %.10g\n",\
                __FILE__, __LINE__, _a, _b, std::fabs(_a-_b), _t);\
            throw 1;                                               \
        }                                                          \
    } while (0)

#define ASSERT_EQ(a, b)                                            \
    do {                                                           \
        auto _a = (a); auto _b = (b);                              \
        if (_a != _b) {                                            \
            std::fprintf(stderr,                                   \
                "  FAIL %s:%d: %lld != %lld\n",                    \
                __FILE__, __LINE__,                                \
                (long long)_a, (long long)_b);                     \
            throw 1;                                               \
        }                                                          \
    } while (0)

#define ASSERT_TRUE(cond)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            std::fprintf(stderr,                                   \
                "  FAIL %s:%d: condition false: %s\n",             \
                __FILE__, __LINE__, #cond);                        \
            throw 1;                                               \
        }                                                          \
    } while (0)

/* ---- Round-trip: price → IV → compare ---- */
TEST_IV(round_trip) {
    /* First, get BS price with known vol */
    double S = 100.0, K = 100.0, T = 1.0, vol = 0.25, r = 0.05, q = 0.0;
    int32_t ot = 0;

    QKBSInput bs_in{};
    bs_in.n = 1;
    bs_in.spot = &S; bs_in.strike = &K; bs_in.time_to_expiry = &T;
    bs_in.volatility = &vol; bs_in.risk_free_rate = &r;
    bs_in.dividend_yield = &q; bs_in.option_type = &ot;

    double price, delta, gamma, vega, theta, rho;
    int32_t ec;
    QKBSOutput bs_out{};
    bs_out.price = &price; bs_out.delta = &delta; bs_out.gamma = &gamma;
    bs_out.vega = &vega; bs_out.theta = &theta; bs_out.rho = &rho;
    bs_out.error_codes = &ec;

    qk_bs_price(&bs_in, &bs_out);
    ASSERT_EQ(ec, 0);

    /* Now recover vol from that price */
    QKIVInput iv_in{};
    iv_in.n = 1;
    iv_in.spot = &S; iv_in.strike = &K; iv_in.time_to_expiry = &T;
    iv_in.risk_free_rate = &r; iv_in.dividend_yield = &q;
    iv_in.option_type = &ot; iv_in.market_price = &price;
    iv_in.tol = 1e-10;
    iv_in.max_iter = 200;

    double iv;
    int32_t iters, iv_ec;
    QKIVOutput iv_out{};
    iv_out.implied_vol = &iv;
    iv_out.iterations  = &iters;
    iv_out.error_codes = &iv_ec;

    int32_t rc = qk_iv_solve(&iv_in, &iv_out);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(iv_ec, 0);
    ASSERT_NEAR(iv, 0.25, 1e-6);
    ASSERT_TRUE(iters > 0 && iters < 50);
}

/* ---- Round-trip for put ---- */
TEST_IV(round_trip_put) {
    double S = 110.0, K = 100.0, T = 0.5, vol = 0.30, r = 0.03, q = 0.02;
    int32_t ot = 1;

    QKBSInput bs_in{};
    bs_in.n = 1;
    bs_in.spot = &S; bs_in.strike = &K; bs_in.time_to_expiry = &T;
    bs_in.volatility = &vol; bs_in.risk_free_rate = &r;
    bs_in.dividend_yield = &q; bs_in.option_type = &ot;

    double price, delta, gamma, vega, theta, rho;
    int32_t ec;
    QKBSOutput bs_out{};
    bs_out.price = &price; bs_out.delta = &delta; bs_out.gamma = &gamma;
    bs_out.vega = &vega; bs_out.theta = &theta; bs_out.rho = &rho;
    bs_out.error_codes = &ec;

    qk_bs_price(&bs_in, &bs_out);

    QKIVInput iv_in{};
    iv_in.n = 1;
    iv_in.spot = &S; iv_in.strike = &K; iv_in.time_to_expiry = &T;
    iv_in.risk_free_rate = &r; iv_in.dividend_yield = &q;
    iv_in.option_type = &ot; iv_in.market_price = &price;
    iv_in.tol = 1e-10; iv_in.max_iter = 200;

    double iv;
    int32_t iters, iv_ec;
    QKIVOutput iv_out{};
    iv_out.implied_vol = &iv;
    iv_out.iterations  = &iters;
    iv_out.error_codes = &iv_ec;

    qk_iv_solve(&iv_in, &iv_out);
    ASSERT_EQ(iv_ec, 0);
    ASSERT_NEAR(iv, 0.30, 1e-6);
}

/* ---- Invalid price → error ---- */
TEST_IV(bad_price) {
    double S = 100.0, K = 100.0, T = 1.0, r = 0.05, q = 0.0;
    double mkt = -5.0;  /* negative price */
    int32_t ot = 0;

    QKIVInput iv_in{};
    iv_in.n = 1;
    iv_in.spot = &S; iv_in.strike = &K; iv_in.time_to_expiry = &T;
    iv_in.risk_free_rate = &r; iv_in.dividend_yield = &q;
    iv_in.option_type = &ot; iv_in.market_price = &mkt;
    iv_in.tol = 1e-8; iv_in.max_iter = 100;

    double iv;
    int32_t iters, iv_ec;
    QKIVOutput iv_out{};
    iv_out.implied_vol = &iv;
    iv_out.iterations  = &iters;
    iv_out.error_codes = &iv_ec;

    int32_t rc = qk_iv_solve(&iv_in, &iv_out);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(iv_ec, QK_ROW_ERR_BAD_PRICE);
    ASSERT_TRUE(std::isnan(iv));
}

/* ---- Null pointer ---- */
TEST_IV(null_ptr) {
    QKIVOutput out{};
    int32_t rc = qk_iv_solve(nullptr, &out);
    ASSERT_EQ(rc, QK_ERR_NULL_PTR);
}

void register_iv_tests() {
    RegIV_round_trip::reg();
    RegIV_round_trip_put::reg();
    RegIV_bad_price::reg();
    RegIV_null_ptr::reg();
}
