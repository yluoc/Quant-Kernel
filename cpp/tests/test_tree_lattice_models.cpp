#include <quantkernel/qk_abi.h>
#include <cmath>
#include <cstdio>

#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/tree_lattice_methods/tree_lattice_models.h"

extern void test_registry_add(const char* name, void (*func)());

#define TEST_TLM(name)                                                 \
    static void test_tlm_##name();                                     \
    static struct RegTLM_##name {                                      \
        static void reg() { test_registry_add("tlm_" #name, test_tlm_##name); } \
    } regtlm_##name;                                                   \
    static void test_tlm_##name()

#define ASSERT_NEAR(a, b, tol)                                         \
    do {                                                               \
        double _a = (a), _b = (b), _t = (tol);                        \
        if (std::fabs(_a - _b) > _t) {                                \
            std::fprintf(stderr,                                       \
                "  FAIL %s:%d: |%.10g - %.10g| = %.10g > %.10g\n",    \
                __FILE__, __LINE__, _a, _b, std::fabs(_a - _b), _t);  \
            throw 1;                                                   \
        }                                                              \
    } while (0)

#define ASSERT_TRUE(cond)                                              \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::fprintf(stderr,                                       \
                "  FAIL %s:%d: condition false: %s\n",                 \
                __FILE__, __LINE__, #cond);                            \
            throw 1;                                                   \
        }                                                              \
    } while (0)

TEST_TLM(crr_matches_black_scholes_merton_for_european_call) {
    double S = 100.0, K = 100.0, T = 1.0, vol = 0.2, r = 0.03, q = 0.01;
    double crr = qk::tlm::crr_price(S, K, T, vol, r, q, QK_CALL, 400, false);
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    ASSERT_NEAR(crr, bsm, 6e-2);
}

TEST_TLM(jarrow_rudd_matches_black_scholes_merton_for_european_call) {
    double S = 100.0, K = 105.0, T = 1.3, vol = 0.25, r = 0.04, q = 0.01;
    double jr = qk::tlm::jarrow_rudd_price(S, K, T, vol, r, q, QK_CALL, 500, false);
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    ASSERT_NEAR(jr, bsm, 6e-2);
}

TEST_TLM(tian_matches_black_scholes_merton_for_european_put) {
    double S = 95.0, K = 100.0, T = 0.9, vol = 0.22, r = 0.02, q = 0.005;
    double tian = qk::tlm::tian_price(S, K, T, vol, r, q, QK_PUT, 300, false);
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_PUT);
    ASSERT_NEAR(tian, bsm, 7e-2);
}

TEST_TLM(leisen_reimer_matches_black_scholes_merton_for_european_put) {
    double S = 105.0, K = 100.0, T = 1.1, vol = 0.19, r = 0.025, q = 0.005;
    double lr = qk::tlm::leisen_reimer_price(S, K, T, vol, r, q, QK_PUT, 201, false);
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_PUT);
    ASSERT_NEAR(lr, bsm, 3e-2);
}

TEST_TLM(trinomial_matches_black_scholes_merton_for_european_call) {
    double S = 100.0, K = 98.0, T = 0.75, vol = 0.18, r = 0.03, q = 0.01;
    double tri = qk::tlm::trinomial_tree_price(S, K, T, vol, r, q, QK_CALL, 250, false);
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    ASSERT_NEAR(tri, bsm, 3e-2);
}

TEST_TLM(american_put_is_not_below_european_put_for_crr) {
    double S = 100.0, K = 110.0, T = 1.0, vol = 0.25, r = 0.04, q = 0.0;
    double eur = qk::tlm::crr_price(S, K, T, vol, r, q, QK_PUT, 300, false);
    double amr = qk::tlm::crr_price(S, K, T, vol, r, q, QK_PUT, 300, true);
    ASSERT_TRUE(amr >= eur);
}

TEST_TLM(derman_kani_implied_tree_matches_bsm_when_local_vol_is_constant) {
    double S = 100.0, K = 100.0, T = 1.0, vol = 0.2, r = 0.03, q = 0.01;
    auto local_vol = [vol](double /*spot*/, double /*time*/) { return vol; };
    qk::tlm::ImpliedTreeConfig cfg{};
    cfg.steps = 12;
    cfg.american_style = false;
    double dk = qk::tlm::derman_kani_implied_tree_price(S, K, T, r, q, QK_CALL, local_vol, cfg);
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    ASSERT_NEAR(dk, bsm, 2.5e-1);
}

void register_tlm_tests() {
    RegTLM_crr_matches_black_scholes_merton_for_european_call::reg();
    RegTLM_jarrow_rudd_matches_black_scholes_merton_for_european_call::reg();
    RegTLM_tian_matches_black_scholes_merton_for_european_put::reg();
    RegTLM_leisen_reimer_matches_black_scholes_merton_for_european_put::reg();
    RegTLM_trinomial_matches_black_scholes_merton_for_european_call::reg();
    RegTLM_american_put_is_not_below_european_put_for_crr::reg();
    RegTLM_derman_kani_implied_tree_matches_bsm_when_local_vol_is_constant::reg();
}
