#include <quantkernel/qk_abi.h>
#include <cmath>
#include <cstdio>

#include "algorithms/closed_form_semi_analytical/closed_form_models.h"

extern void test_registry_add(const char* name, void (*func)());

#define TEST_CF(name)                                                \
    static void test_cf_##name();                                    \
    static struct RegCF_##name {                                     \
        static void reg() { test_registry_add("cf_" #name, test_cf_##name); } \
    } regcf_##name;                                                  \
    static void test_cf_##name()

#define ASSERT_NEAR(a, b, tol)                                       \
    do {                                                             \
        double _a = (a), _b = (b), _t = (tol);                      \
        if (std::fabs(_a - _b) > _t) {                              \
            std::fprintf(stderr,                                     \
                "  FAIL %s:%d: |%.10g - %.10g| = %.10g > %.10g\n",  \
                __FILE__, __LINE__, _a, _b, std::fabs(_a-_b), _t);  \
            throw 1;                                                 \
        }                                                            \
    } while (0)

#define ASSERT_TRUE(cond)                                            \
    do {                                                             \
        if (!(cond)) {                                               \
            std::fprintf(stderr,                                     \
                "  FAIL %s:%d: condition false: %s\n",               \
                __FILE__, __LINE__, #cond);                          \
            throw 1;                                                 \
        }                                                            \
    } while (0)

TEST_CF(black_scholes_merton_sanity) {
    double S = 100.0, K = 105.0, T = 0.75, vol = 0.25, r = 0.03, q = 0.01;
    double call = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    double put = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_PUT);
    ASSERT_TRUE(call > 0.0);
    ASSERT_TRUE(put > 0.0);
}

TEST_CF(black76_matches_black_scholes_merton_with_forward) {
    double S = 100.0, K = 100.0, T = 1.0, vol = 0.20, r = 0.05, q = 0.02;
    double F = S * std::exp((r - q) * T);
    double bsm_like = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    double b76 = qk::cfa::black76_price(F, K, T, vol, r, QK_CALL);
    ASSERT_NEAR(b76, bsm_like, 1e-10);
}

TEST_CF(bachelier_atm_closed_form) {
    double F = 100.0, K = 100.0, T = 1.0, sig_n = 8.0, r = 0.03;
    double call = qk::cfa::bachelier_price(F, K, T, sig_n, r, QK_CALL);
    double expected = std::exp(-r * T) * sig_n / std::sqrt(2.0 * 3.14159265358979323846);
    ASSERT_NEAR(call, expected, 1e-10);
}

TEST_CF(heston_put_call_parity) {
    qk::cfa::HestonParams p{};
    p.v0 = 0.04;
    p.kappa = 2.0;
    p.theta = 0.04;
    p.sigma = 0.5;
    p.rho = -0.5;

    double S = 100.0, K = 100.0, T = 1.0, r = 0.02, q = 0.01;
    double call = qk::cfa::heston_price_cf(S, K, T, r, q, p, QK_CALL, 1536, 140.0);
    double put = qk::cfa::heston_price_cf(S, K, T, r, q, p, QK_PUT, 1536, 140.0);

    ASSERT_TRUE(call > 0.0);
    ASSERT_TRUE(put > 0.0);
    double lhs = call - put;
    double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);
    ASSERT_NEAR(lhs, rhs, 8e-3);
}

TEST_CF(merton_reduces_to_black_scholes_merton_when_lambda_zero) {
    qk::cfa::MertonJumpDiffusionParams p{};
    p.jump_intensity = 0.0;
    p.jump_mean = -0.1;
    p.jump_vol = 0.2;
    p.max_terms = 80;

    double S = 95.0, K = 100.0, T = 1.2, vol = 0.22, r = 0.015, q = 0.005;
    double mjd = qk::cfa::merton_jump_diffusion_price(S, K, T, vol, r, q, p, QK_CALL);
    double bsm_like = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    ASSERT_NEAR(mjd, bsm_like, 1e-12);
}

TEST_CF(variance_gamma_put_call_parity) {
    qk::cfa::VarianceGammaParams p{};
    p.sigma = 0.20;
    p.theta = -0.10;
    p.nu = 0.20;

    double S = 100.0, K = 95.0, T = 1.2, r = 0.03, q = 0.01;
    double call = qk::cfa::variance_gamma_price_cf(S, K, T, r, q, p, QK_CALL, 1536, 140.0);
    double put = qk::cfa::variance_gamma_price_cf(S, K, T, r, q, p, QK_PUT, 1536, 140.0);

    ASSERT_TRUE(call > 0.0);
    ASSERT_TRUE(put > 0.0);
    double lhs = call - put;
    double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);
    ASSERT_NEAR(lhs, rhs, 8e-3);
}

TEST_CF(sabr_hagan_atm_limit) {
    qk::cfa::SABRParams p{};
    p.alpha = 0.20;
    p.beta = 0.50;
    p.rho = -0.20;
    p.nu = 0.40;

    double F = 100.0;
    double iv0 = qk::cfa::sabr_hagan_lognormal_iv(F, F, 0.0, p);
    ASSERT_NEAR(iv0, 0.02, 1e-12);

    double iv = qk::cfa::sabr_hagan_lognormal_iv(F, 105.0, 1.0, p);
    ASSERT_TRUE(iv > 0.0);

    double call = qk::cfa::sabr_hagan_black76_price(F, 105.0, 1.0, 0.03, p, QK_CALL);
    ASSERT_TRUE(call > 0.0);
}

TEST_CF(dupire_recovers_bs_constant_vol) {
    const double S = 100.0;
    const double sigma = 0.25;
    const double r = 0.02;
    const double q = 0.01;

    auto call_surface = [&](double K, double T) {
        return qk::cfa::black_scholes_merton_price(S, K, T, sigma, r, q, QK_CALL);
    };

    double local = qk::cfa::dupire_local_vol_fd(call_surface, 100.0, 1.0, r, q, 0.5, 1e-3);
    ASSERT_TRUE(local > 0.0);
    ASSERT_NEAR(local, sigma, 0.02);
}

void register_cf_tests() {
    RegCF_black_scholes_merton_sanity::reg();
    RegCF_black76_matches_black_scholes_merton_with_forward::reg();
    RegCF_bachelier_atm_closed_form::reg();
    RegCF_heston_put_call_parity::reg();
    RegCF_merton_reduces_to_black_scholes_merton_when_lambda_zero::reg();
    RegCF_variance_gamma_put_call_parity::reg();
    RegCF_sabr_hagan_atm_limit::reg();
    RegCF_dupire_recovers_bs_constant_vol::reg();
}
