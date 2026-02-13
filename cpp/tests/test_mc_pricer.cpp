#include <quantkernel/qk_api.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

extern void test_registry_add(const char* name, void (*func)());

#define TEST_MC(name)                                              \
    static void test_mc_##name();                                  \
    static struct RegMC_##name {                                   \
        static void reg() { test_registry_add("mc_" #name, test_mc_##name); } \
    } regmc_##name;                                                \
    static void test_mc_##name()

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

TEST_MC(price_close_to_bs) {
    double S = 100.0, K = 100.0, T = 1.0, vol = 0.20, r = 0.05, q = 0.0;
    int32_t ot = QK_CALL, paths = 200000;
    uint64_t seed = UINT64_C(42);

    QKMCInput mc_in{};
    mc_in.n = 1;
    mc_in.spot = &S; mc_in.strike = &K; mc_in.time_to_expiry = &T;
    mc_in.volatility = &vol; mc_in.risk_free_rate = &r;
    mc_in.dividend_yield = &q; mc_in.option_type = &ot;
    mc_in.num_paths = &paths; mc_in.rng_seed = &seed;

    double mc_price, mc_std_error;
    int32_t paths_used, mc_ec;
    QKMCOutput mc_out{};
    mc_out.price = &mc_price; mc_out.std_error = &mc_std_error;
    mc_out.paths_used = &paths_used; mc_out.error_codes = &mc_ec;

    int32_t rc = qk_mc_price(&mc_in, &mc_out);
    ASSERT_EQ(rc, QK_OK);
    ASSERT_EQ(mc_ec, QK_ROW_OK);
    ASSERT_EQ(paths_used, paths);
    ASSERT_TRUE(mc_std_error > 0.0);

    QKBSInput bs_in{};
    bs_in.n = 1;
    bs_in.spot = &S; bs_in.strike = &K; bs_in.time_to_expiry = &T;
    bs_in.volatility = &vol; bs_in.risk_free_rate = &r;
    bs_in.dividend_yield = &q; bs_in.option_type = &ot;

    double bs_price, delta, gamma, vega, theta, rho;
    int32_t bs_ec;
    QKBSOutput bs_out{};
    bs_out.price = &bs_price; bs_out.delta = &delta; bs_out.gamma = &gamma;
    bs_out.vega = &vega; bs_out.theta = &theta; bs_out.rho = &rho;
    bs_out.error_codes = &bs_ec;
    rc = qk_bs_price(&bs_in, &bs_out);
    ASSERT_EQ(rc, QK_OK);
    ASSERT_EQ(bs_ec, QK_ROW_OK);

    ASSERT_NEAR(mc_price, bs_price, 4.0 * mc_std_error + 0.05);
}

TEST_MC(seed_reproducibility) {
    double S = 100.0, K = 100.0, T = 1.0, vol = 0.20, r = 0.05, q = 0.0;
    int32_t ot = QK_CALL, paths = 50000;
    uint64_t seed = UINT64_C(7);

    QKMCInput in{};
    in.n = 1;
    in.spot = &S; in.strike = &K; in.time_to_expiry = &T;
    in.volatility = &vol; in.risk_free_rate = &r;
    in.dividend_yield = &q; in.option_type = &ot;
    in.num_paths = &paths; in.rng_seed = &seed;

    double p1, se1, p2, se2;
    int32_t used1, ec1, used2, ec2;
    QKMCOutput out1{};
    out1.price = &p1; out1.std_error = &se1; out1.paths_used = &used1; out1.error_codes = &ec1;
    QKMCOutput out2{};
    out2.price = &p2; out2.std_error = &se2; out2.paths_used = &used2; out2.error_codes = &ec2;

    ASSERT_EQ(qk_mc_price(&in, &out1), QK_OK);
    ASSERT_EQ(qk_mc_price(&in, &out2), QK_OK);
    ASSERT_EQ(ec1, QK_ROW_OK);
    ASSERT_EQ(ec2, QK_ROW_OK);
    ASSERT_EQ(p1, p2);
    ASSERT_EQ(se1, se2);
}

TEST_MC(bad_paths_row_error) {
    double S = 100.0, K = 100.0, T = 1.0, vol = 0.20, r = 0.05, q = 0.0;
    int32_t ot = QK_CALL, paths = 0;
    uint64_t seed = UINT64_C(123);

    QKMCInput in{};
    in.n = 1;
    in.spot = &S; in.strike = &K; in.time_to_expiry = &T;
    in.volatility = &vol; in.risk_free_rate = &r;
    in.dividend_yield = &q; in.option_type = &ot;
    in.num_paths = &paths; in.rng_seed = &seed;

    double price, std_error;
    int32_t used, ec;
    QKMCOutput out{};
    out.price = &price; out.std_error = &std_error; out.paths_used = &used; out.error_codes = &ec;

    ASSERT_EQ(qk_mc_price(&in, &out), QK_OK);
    ASSERT_EQ(ec, QK_ROW_ERR_BAD_PATHS);
    ASSERT_EQ(used, 0);
    ASSERT_TRUE(std::isnan(price));
    ASSERT_TRUE(std::isnan(std_error));
}

void register_mc_tests() {
    RegMC_price_close_to_bs::reg();
    RegMC_seed_reproducibility::reg();
    RegMC_bad_paths_row_error::reg();
}
