#include <quantkernel/qk_api.h>
#include <cmath>
#include <cstdio>
#include <cstring>

/* Pull in the test macros — declared in test_main.cpp */
extern void test_registry_add(const char* name, void (*func)());

#define TEST_BS(name)                                             \
    static void test_bs_##name();                                 \
    static struct RegBS_##name {                                  \
        static void reg() { test_registry_add("bs_" #name, test_bs_##name); } \
    } regbs_##name;                                               \
    static void test_bs_##name()

/* Helper macros (duplicated to keep each TU self-contained) */
#define ASSERT_NEAR(a, b, tol)                                    \
    do {                                                          \
        double _a = (a), _b = (b), _t = (tol);                   \
        if (std::fabs(_a - _b) > _t) {                           \
            std::fprintf(stderr,                                  \
                "  FAIL %s:%d: |%.10g - %.10g| = %.10g > %.10g\n",\
                __FILE__, __LINE__, _a, _b, std::fabs(_a-_b), _t);\
            throw 1;                                              \
        }                                                         \
    } while (0)

#define ASSERT_EQ(a, b)                                           \
    do {                                                          \
        auto _a = (a); auto _b = (b);                             \
        if (_a != _b) {                                           \
            std::fprintf(stderr,                                  \
                "  FAIL %s:%d: %lld != %lld\n",                   \
                __FILE__, __LINE__,                               \
                (long long)_a, (long long)_b);                    \
            throw 1;                                              \
        }                                                         \
    } while (0)

#define ASSERT_TRUE(cond)                                         \
    do {                                                          \
        if (!(cond)) {                                            \
            std::fprintf(stderr,                                  \
                "  FAIL %s:%d: condition false: %s\n",            \
                __FILE__, __LINE__, #cond);                       \
            throw 1;                                              \
        }                                                         \
    } while (0)

/* ---- ATM call reference values ---- */
/* S=100, K=100, T=1, vol=0.20, r=0.05, q=0.0, CALL
   Ref: price≈10.4506, delta≈0.6368, gamma≈0.01876, vega≈0.3752, rho≈0.5323 */
TEST_BS(atm_call) {
    double S = 100.0, K = 100.0, T = 1.0, vol = 0.20, r = 0.05, q = 0.0;
    int32_t ot = 0; /* CALL */

    QKBSInput in{};
    in.n = 1;
    in.spot = &S; in.strike = &K; in.time_to_expiry = &T;
    in.volatility = &vol; in.risk_free_rate = &r;
    in.dividend_yield = &q; in.option_type = &ot;

    double price, delta, gamma, vega, theta, rho;
    int32_t ec;
    QKBSOutput out{};
    out.price = &price; out.delta = &delta; out.gamma = &gamma;
    out.vega = &vega; out.theta = &theta; out.rho = &rho;
    out.error_codes = &ec;

    int32_t rc = qk_bs_price(&in, &out);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(ec, 0);
    ASSERT_NEAR(price, 10.4506, 0.001);
    ASSERT_NEAR(delta, 0.6368, 0.001);
    ASSERT_NEAR(gamma, 0.01876, 0.0005);
    ASSERT_NEAR(vega, 0.3752, 0.001);
    ASSERT_NEAR(rho, 0.5323, 0.002);
}

/* ---- ATM put (same params) ---- */
TEST_BS(atm_put) {
    double S = 100.0, K = 100.0, T = 1.0, vol = 0.20, r = 0.05, q = 0.0;
    int32_t ot = 1; /* PUT */

    QKBSInput in{};
    in.n = 1;
    in.spot = &S; in.strike = &K; in.time_to_expiry = &T;
    in.volatility = &vol; in.risk_free_rate = &r;
    in.dividend_yield = &q; in.option_type = &ot;

    double price, delta, gamma, vega, theta, rho;
    int32_t ec;
    QKBSOutput out{};
    out.price = &price; out.delta = &delta; out.gamma = &gamma;
    out.vega = &vega; out.theta = &theta; out.rho = &rho;
    out.error_codes = &ec;

    int32_t rc = qk_bs_price(&in, &out);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(ec, 0);

    /* Put-call parity: P = C - S + K*e^(-rT) */
    double call_price = 10.4506;
    double parity_put = call_price - S + K * std::exp(-r * T);
    ASSERT_NEAR(price, parity_put, 0.001);
    ASSERT_TRUE(delta < 0.0); /* put delta is negative */
}

/* ---- Batch with one invalid row ---- */
TEST_BS(batch_with_invalid) {
    double spots[3]   = {100.0, -50.0, 100.0};
    double strikes[3] = {100.0, 100.0, 100.0};
    double times[3]   = {1.0, 1.0, 1.0};
    double vols[3]    = {0.20, 0.20, 0.20};
    double rates[3]   = {0.05, 0.05, 0.05};
    double divs[3]    = {0.0, 0.0, 0.0};
    int32_t types[3]  = {0, 0, 0};

    QKBSInput in{};
    in.n = 3;
    in.spot = spots; in.strike = strikes; in.time_to_expiry = times;
    in.volatility = vols; in.risk_free_rate = rates;
    in.dividend_yield = divs; in.option_type = types;

    double prices[3], deltas[3], gammas[3], vegas[3], thetas[3], rhos[3];
    int32_t ecs[3];
    QKBSOutput out{};
    out.price = prices; out.delta = deltas; out.gamma = gammas;
    out.vega = vegas; out.theta = thetas; out.rho = rhos;
    out.error_codes = ecs;

    int32_t rc = qk_bs_price(&in, &out);
    ASSERT_EQ(rc, 0);

    /* Row 0: valid */
    ASSERT_EQ(ecs[0], 0);
    ASSERT_NEAR(prices[0], 10.4506, 0.001);

    /* Row 1: negative spot → error */
    ASSERT_EQ(ecs[1], QK_ROW_ERR_NEGATIVE_S);
    ASSERT_TRUE(std::isnan(prices[1]));

    /* Row 2: valid again */
    ASSERT_EQ(ecs[2], 0);
    ASSERT_NEAR(prices[2], 10.4506, 0.001);
}

/* ---- Null pointer returns error ---- */
TEST_BS(null_ptr) {
    QKBSOutput out{};
    int32_t rc = qk_bs_price(nullptr, &out);
    ASSERT_EQ(rc, QK_ERR_NULL_PTR);
}

/* ---- ABI version check ---- */
TEST_BS(abi_version) {
    int32_t major = -1, minor = -1;
    qk_abi_version(&major, &minor);
    ASSERT_EQ(major, QK_ABI_MAJOR);
    ASSERT_EQ(minor, QK_ABI_MINOR);
}

TEST_BS(plugin_api_export) {
    const QKPluginAPI* api = nullptr;
    int32_t rc = qk_plugin_get_api(QK_ABI_MAJOR, QK_ABI_MINOR, &api);
    ASSERT_EQ(rc, QK_OK);
    ASSERT_TRUE(api != nullptr);
    ASSERT_EQ(api->abi_major, QK_ABI_MAJOR);
    ASSERT_EQ(api->abi_minor, QK_ABI_MINOR);
    ASSERT_TRUE(api->plugin_name != nullptr);
    ASSERT_TRUE(api->bs_price != nullptr);
    ASSERT_TRUE(api->iv_solve != nullptr);
    ASSERT_TRUE(api->mc_price != nullptr);

    double S = 100.0, K = 100.0, T = 1.0, vol = 0.20, r = 0.05, q = 0.0;
    int32_t ot = QK_CALL;
    QKBSInput in{};
    in.n = 1;
    in.spot = &S; in.strike = &K; in.time_to_expiry = &T;
    in.volatility = &vol; in.risk_free_rate = &r;
    in.dividend_yield = &q; in.option_type = &ot;

    double price, delta, gamma, vega, theta, rho;
    int32_t ec;
    QKBSOutput out{};
    out.price = &price; out.delta = &delta; out.gamma = &gamma;
    out.vega = &vega; out.theta = &theta; out.rho = &rho;
    out.error_codes = &ec;

    rc = api->bs_price(&in, &out);
    ASSERT_EQ(rc, QK_OK);
    ASSERT_EQ(ec, QK_ROW_OK);
    ASSERT_NEAR(price, 10.4506, 0.001);

    api = nullptr;
    rc = qk_plugin_get_api(QK_ABI_MAJOR + 1, QK_ABI_MINOR, &api);
    ASSERT_EQ(rc, QK_ERR_ABI_MISMATCH);
    ASSERT_TRUE(api == nullptr);
}

void register_bs_tests() {
    RegBS_atm_call::reg();
    RegBS_atm_put::reg();
    RegBS_batch_with_invalid::reg();
    RegBS_null_ptr::reg();
    RegBS_abi_version::reg();
    RegBS_plugin_api_export::reg();
}
