#include <quantkernel/qk_api.h>

#include <cstdio>
#include <cstring>

#include "algorithms/closed_form_semi_analytical/closed_form_models.h"
#include "algorithms/tree_lattice_methods/tree_lattice_models.h"
#include "algorithms/finite_difference_methods/finite_difference_models.h"
#include "algorithms/monte_carlo_methods/monte_carlo_models.h"
#include "algorithms/fourier_transform_methods/fourier_transform_models.h"
#include "algorithms/integral_quadrature/integral_quadrature_models.h"
#include "algorithms/regression_approximation/regression_approximation_models.h"
#include "algorithms/adjoint_greeks/adjoint_greeks_models.h"
#include "algorithms/machine_learning/machine_learning_models.h"

/* --- Thread-local error message buffer --- */

static thread_local char tl_error_msg[256] = "";

static void set_error_msg(const char* msg) {
    std::strncpy(tl_error_msg, msg, sizeof(tl_error_msg) - 1);
    tl_error_msg[sizeof(tl_error_msg) - 1] = '\0';
}

static void set_error_null_ptr(const char* param_name) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "null pointer: %s", param_name);
    set_error_msg(buf);
}

static void set_error_bad_size(int32_t n) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "bad batch size: %d", n);
    set_error_msg(buf);
}

/* --- Batch validation macros --- */

#define QK_BATCH_VALIDATE_N(n) \
    do { if ((n) <= 0) { set_error_bad_size(n); return QK_ERR_BAD_SIZE; } } while(0)

#define QK_BATCH_NULL_CHECK_1(a) \
    do { if (!(a)) { set_error_null_ptr(#a); return QK_ERR_NULL_PTR; } } while(0)

#define QK_BATCH_NULL_CHECK_INNER(ptrs, n, ...) \
    do { const char* _all_names[] = {__VA_ARGS__}; \
         for (size_t _i = 0; _i < (n); ++_i) \
             if (!(ptrs)[_i]) { set_error_null_ptr(_all_names[_i]); return QK_ERR_NULL_PTR; } \
    } while(0)

/* Stringify each argument individually so _names[_i] works correctly. */
#define QK_BNCS_1(a) #a
#define QK_BNCS_2(a,b) #a,#b
#define QK_BNCS_3(a,b,c) #a,#b,#c
#define QK_BNCS_4(a,b,c,d) #a,#b,#c,#d
#define QK_BNCS_5(a,b,c,d,e) #a,#b,#c,#d,#e
#define QK_BNCS_6(a,b,c,d,e,f) #a,#b,#c,#d,#e,#f
#define QK_BNCS_7(a,b,c,d,e,f,g) #a,#b,#c,#d,#e,#f,#g
#define QK_BNCS_8(a,b,c,d,e,f,g,h) #a,#b,#c,#d,#e,#f,#g,#h
#define QK_BNCS_9(a,b,c,d,e,f,g,h,i) #a,#b,#c,#d,#e,#f,#g,#h,#i
#define QK_BNCS_10(a,b,c,d,e,f,g,h,i,j) #a,#b,#c,#d,#e,#f,#g,#h,#i,#j
#define QK_BNCS_11(a,b,c,d,e,f,g,h,i,j,k) #a,#b,#c,#d,#e,#f,#g,#h,#i,#j,#k
#define QK_BNCS_12(a,b,c,d,e,f,g,h,i,j,k,l) #a,#b,#c,#d,#e,#f,#g,#h,#i,#j,#k,#l
#define QK_BNCS_13(a,b,c,d,e,f,g,h,i,j,k,l,m) #a,#b,#c,#d,#e,#f,#g,#h,#i,#j,#k,#l,#m
#define QK_BNCS_14(a,b,c,d,e,f,g,h,i,j,k,l,m,nn) #a,#b,#c,#d,#e,#f,#g,#h,#i,#j,#k,#l,#m,#nn
#define QK_BNCS_15(a,b,c,d,e,f,g,h,i,j,k,l,m,nn,o) #a,#b,#c,#d,#e,#f,#g,#h,#i,#j,#k,#l,#m,#nn,#o
#define QK_BNCS_16(a,b,c,d,e,f,g,h,i,j,k,l,m,nn,o,p) #a,#b,#c,#d,#e,#f,#g,#h,#i,#j,#k,#l,#m,#nn,#o,#p

#define QK_BNC_COUNT(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,N,...) N
#define QK_BNC_N(...) QK_BNC_COUNT(__VA_ARGS__,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)
#define QK_BNC_CAT(a,b) a##b
#define QK_BNC_SEL(n) QK_BNC_CAT(QK_BNCS_,n)
#define QK_BNC_EXPAND(...) __VA_ARGS__
#define QK_BNC_NAMES2(n,...) QK_BNC_EXPAND(QK_BNC_SEL(n)(__VA_ARGS__))
#define QK_BNC_NAMES(...) QK_BNC_NAMES2(QK_BNC_N(__VA_ARGS__),__VA_ARGS__)

#define QK_BATCH_NULL_CHECK(...) \
    do { const void* _ptrs[] = {__VA_ARGS__}; \
         QK_BATCH_NULL_CHECK_INNER(_ptrs, sizeof(_ptrs)/sizeof(_ptrs[0]), QK_BNC_NAMES(__VA_ARGS__)); \
    } while(0)

namespace {

const QKPluginAPI k_plugin_api = {
    QK_ABI_MAJOR,
    QK_ABI_MINOR,
    "quantkernel.cpp.closed_form_trees_fdm_mcm_ftm_iqm_ram_agm_mlm.v10"
};

} /* namespace */

extern "C" {

void qk_abi_version(int32_t* major, int32_t* minor) {
    if (major) *major = QK_ABI_MAJOR;
    if (minor) *minor = QK_ABI_MINOR;
}

int32_t qk_plugin_get_api(int32_t host_abi_major,
                          int32_t host_abi_minor,
                          const QKPluginAPI** out_api) {
    if (!out_api) return QK_ERR_NULL_PTR;
    if (host_abi_major != QK_ABI_MAJOR || host_abi_minor > QK_ABI_MINOR) {
        *out_api = nullptr;
        return QK_ERR_ABI_MISMATCH;
    }
    *out_api = &k_plugin_api;
    return QK_OK;
}

const char* qk_get_last_error(void) {
    return tl_error_msg;
}

void qk_clear_last_error(void) {
    tl_error_msg[0] = '\0';
}

double qk_cf_black_scholes_merton_price(double spot, double strike, double t, double vol,
                                        double r, double q, int32_t option_type) {
    return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, option_type);
}

double qk_cf_black76_price(double forward, double strike, double t, double vol,
                           double r, int32_t option_type) {
    return qk::cfa::black76_price(forward, strike, t, vol, r, option_type);
}

double qk_cf_bachelier_price(double forward, double strike, double t, double normal_vol,
                             double r, int32_t option_type) {
    return qk::cfa::bachelier_price(forward, strike, t, normal_vol, r, option_type);
}

int32_t qk_cf_black_scholes_merton_price_batch(const double* spot,
                                               const double* strike,
                                               const double* t,
                                               const double* vol,
                                               const double* r,
                                               const double* q,
                                               const int32_t* option_type,
                                               int32_t n,
                                               double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, out_prices);
    QK_BATCH_VALIDATE_N(n);

    #pragma omp simd
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::cfa::black_scholes_merton_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i]
        );
    }
    return QK_OK;
}

int32_t qk_cf_black76_price_batch(const double* forward,
                                  const double* strike,
                                  const double* t,
                                  const double* vol,
                                  const double* r,
                                  const int32_t* option_type,
                                  int32_t n,
                                  double* out_prices) {
    QK_BATCH_NULL_CHECK(forward, strike, t, vol, r, option_type, out_prices);
    QK_BATCH_VALIDATE_N(n);

    #pragma omp simd
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::cfa::black76_price(
            forward[i], strike[i], t[i], vol[i], r[i], option_type[i]
        );
    }
    return QK_OK;
}

int32_t qk_cf_bachelier_price_batch(const double* forward,
                                    const double* strike,
                                    const double* t,
                                    const double* normal_vol,
                                    const double* r,
                                    const int32_t* option_type,
                                    int32_t n,
                                    double* out_prices) {
    QK_BATCH_NULL_CHECK(forward, strike, t, normal_vol, r, option_type, out_prices);
    QK_BATCH_VALIDATE_N(n);

    #pragma omp simd
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::cfa::bachelier_price(
            forward[i], strike[i], t[i], normal_vol[i], r[i], option_type[i]
        );
    }
    return QK_OK;
}

double qk_cf_heston_price_cf(double spot, double strike, double t, double r, double q,
                             double v0, double kappa, double theta, double sigma, double rho,
                             int32_t option_type, int32_t integration_steps, double integration_limit) {
    qk::cfa::HestonParams params{};
    params.v0 = v0;
    params.kappa = kappa;
    params.theta = theta;
    params.sigma = sigma;
    params.rho = rho;
    return qk::cfa::heston_price_cf(
        spot, strike, t, r, q, params, option_type, integration_steps, integration_limit
    );
}

double qk_cf_merton_jump_diffusion_price(double spot, double strike, double t, double vol,
                                         double r, double q, double jump_intensity,
                                         double jump_mean, double jump_vol, int32_t max_terms,
                                         int32_t option_type) {
    qk::cfa::MertonJumpDiffusionParams params{};
    params.jump_intensity = jump_intensity;
    params.jump_mean = jump_mean;
    params.jump_vol = jump_vol;
    params.max_terms = max_terms;
    return qk::cfa::merton_jump_diffusion_price(spot, strike, t, vol, r, q, params, option_type);
}

double qk_cf_variance_gamma_price_cf(double spot, double strike, double t, double r, double q,
                                     double sigma, double theta, double nu, int32_t option_type,
                                     int32_t integration_steps, double integration_limit) {
    qk::cfa::VarianceGammaParams params{};
    params.sigma = sigma;
    params.theta = theta;
    params.nu = nu;
    return qk::cfa::variance_gamma_price_cf(
        spot, strike, t, r, q, params, option_type, integration_steps, integration_limit
    );
}

double qk_cf_sabr_hagan_lognormal_iv(double forward, double strike, double t,
                                     double alpha, double beta, double rho, double nu) {
    qk::cfa::SABRParams params{};
    params.alpha = alpha;
    params.beta = beta;
    params.rho = rho;
    params.nu = nu;
    return qk::cfa::sabr_hagan_lognormal_iv(forward, strike, t, params);
}

double qk_cf_sabr_hagan_black76_price(double forward, double strike, double t, double r,
                                      double alpha, double beta, double rho, double nu,
                                      int32_t option_type) {
    qk::cfa::SABRParams params{};
    params.alpha = alpha;
    params.beta = beta;
    params.rho = rho;
    params.nu = nu;
    return qk::cfa::sabr_hagan_black76_price(forward, strike, t, r, params, option_type);
}

double qk_cf_dupire_local_vol(double strike, double t, double call_price,
                              double dC_dT, double dC_dK, double d2C_dK2,
                              double r, double q) {
    return qk::cfa::dupire_local_vol(strike, t, call_price, dC_dT, dC_dK, d2C_dK2, r, q);
}

int32_t qk_cf_heston_price_cf_batch(const double* spot, const double* strike,
                                     const double* t, const double* r, const double* q,
                                     const double* v0, const double* kappa, const double* theta,
                                     const double* sigma, const double* rho,
                                     const int32_t* option_type, const int32_t* integration_steps,
                                     const double* integration_limit,
                                     int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, r, q, v0, kappa, theta, sigma, rho,
                        option_type, integration_steps, integration_limit, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::cfa::HestonParams params{};
        params.v0 = v0[i]; params.kappa = kappa[i]; params.theta = theta[i];
        params.sigma = sigma[i]; params.rho = rho[i];
        out_prices[i] = qk::cfa::heston_price_cf(
            spot[i], strike[i], t[i], r[i], q[i], params,
            option_type[i], integration_steps[i], integration_limit[i]
        );
    }
    return QK_OK;
}

int32_t qk_cf_merton_jump_diffusion_price_batch(const double* spot, const double* strike,
                                                  const double* t, const double* vol,
                                                  const double* r, const double* q,
                                                  const double* jump_intensity, const double* jump_mean,
                                                  const double* jump_vol, const int32_t* max_terms,
                                                  const int32_t* option_type,
                                                  int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, jump_intensity, jump_mean,
                        jump_vol, max_terms, option_type, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::cfa::MertonJumpDiffusionParams params{};
        params.jump_intensity = jump_intensity[i]; params.jump_mean = jump_mean[i];
        params.jump_vol = jump_vol[i]; params.max_terms = max_terms[i];
        out_prices[i] = qk::cfa::merton_jump_diffusion_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], params, option_type[i]
        );
    }
    return QK_OK;
}

int32_t qk_cf_variance_gamma_price_cf_batch(const double* spot, const double* strike,
                                              const double* t, const double* r, const double* q,
                                              const double* sigma, const double* theta, const double* nu,
                                              const int32_t* option_type, const int32_t* integration_steps,
                                              const double* integration_limit,
                                              int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, r, q, sigma, theta, nu,
                        option_type, integration_steps, integration_limit, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::cfa::VarianceGammaParams params{};
        params.sigma = sigma[i]; params.theta = theta[i]; params.nu = nu[i];
        out_prices[i] = qk::cfa::variance_gamma_price_cf(
            spot[i], strike[i], t[i], r[i], q[i], params,
            option_type[i], integration_steps[i], integration_limit[i]
        );
    }
    return QK_OK;
}

int32_t qk_cf_sabr_hagan_lognormal_iv_batch(const double* forward, const double* strike,
                                              const double* t, const double* alpha,
                                              const double* beta, const double* rho,
                                              const double* nu,
                                              int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(forward, strike, t, alpha, beta, rho, nu, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp simd
    for (int32_t i = 0; i < n; ++i) {
        qk::cfa::SABRParams params{};
        params.alpha = alpha[i]; params.beta = beta[i];
        params.rho = rho[i]; params.nu = nu[i];
        out_prices[i] = qk::cfa::sabr_hagan_lognormal_iv(forward[i], strike[i], t[i], params);
    }
    return QK_OK;
}

int32_t qk_cf_sabr_hagan_black76_price_batch(const double* forward, const double* strike,
                                               const double* t, const double* r,
                                               const double* alpha, const double* beta,
                                               const double* rho, const double* nu,
                                               const int32_t* option_type,
                                               int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(forward, strike, t, r, alpha, beta, rho, nu, option_type, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp simd
    for (int32_t i = 0; i < n; ++i) {
        qk::cfa::SABRParams params{};
        params.alpha = alpha[i]; params.beta = beta[i];
        params.rho = rho[i]; params.nu = nu[i];
        out_prices[i] = qk::cfa::sabr_hagan_black76_price(
            forward[i], strike[i], t[i], r[i], params, option_type[i]
        );
    }
    return QK_OK;
}

int32_t qk_cf_dupire_local_vol_batch(const double* strike, const double* t,
                                      const double* call_price, const double* dC_dT,
                                      const double* dC_dK, const double* d2C_dK2,
                                      const double* r, const double* q,
                                      int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(strike, t, call_price, dC_dT, dC_dK, d2C_dK2, r, q, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp simd
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::cfa::dupire_local_vol(
            strike[i], t[i], call_price[i], dC_dT[i], dC_dK[i], d2C_dK2[i], r[i], q[i]
        );
    }
    return QK_OK;
}

double qk_tlm_crr_price(double spot, double strike, double t, double vol,
                        double r, double q, int32_t option_type,
                        int32_t steps, int32_t american_style) {
    return qk::tlm::crr_price(spot, strike, t, vol, r, q, option_type, steps, american_style != 0);
}

int32_t qk_tlm_crr_price_batch(const double* spot,
                               const double* strike,
                               const double* t,
                               const double* vol,
                               const double* r,
                               const double* q,
                               const int32_t* option_type,
                               const int32_t* steps,
                               const int32_t* american_style,
                               int32_t n,
                               double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, steps, american_style, out_prices);
    QK_BATCH_VALIDATE_N(n);

    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::tlm::crr_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            steps[i], american_style[i] != 0
        );
    }
    return QK_OK;
}

double qk_tlm_jarrow_rudd_price(double spot, double strike, double t, double vol,
                                double r, double q, int32_t option_type,
                                int32_t steps, int32_t american_style) {
    return qk::tlm::jarrow_rudd_price(
        spot, strike, t, vol, r, q, option_type, steps, american_style != 0
    );
}

double qk_tlm_tian_price(double spot, double strike, double t, double vol,
                         double r, double q, int32_t option_type,
                         int32_t steps, int32_t american_style) {
    return qk::tlm::tian_price(spot, strike, t, vol, r, q, option_type, steps, american_style != 0);
}

double qk_tlm_leisen_reimer_price(double spot, double strike, double t, double vol,
                                  double r, double q, int32_t option_type,
                                  int32_t steps, int32_t american_style) {
    return qk::tlm::leisen_reimer_price(
        spot, strike, t, vol, r, q, option_type, steps, american_style != 0
    );
}

double qk_tlm_trinomial_tree_price(double spot, double strike, double t, double vol,
                                   double r, double q, int32_t option_type,
                                   int32_t steps, int32_t american_style) {
    return qk::tlm::trinomial_tree_price(
        spot, strike, t, vol, r, q, option_type, steps, american_style != 0
    );
}

double qk_tlm_derman_kani_const_local_vol_price(double spot, double strike, double t,
                                                double local_vol, double r, double q,
                                                int32_t option_type, int32_t steps,
                                                int32_t american_style) {
    auto surface = [local_vol](double /*s*/, double /*tau*/) { return local_vol; };
    qk::tlm::ImpliedTreeConfig cfg{};
    cfg.steps = steps;
    cfg.american_style = (american_style != 0);
    return qk::tlm::derman_kani_implied_tree_price(spot, strike, t, r, q, option_type, surface, cfg);
}

int32_t qk_tlm_jarrow_rudd_price_batch(const double* spot, const double* strike,
                                         const double* t, const double* vol,
                                         const double* r, const double* q,
                                         const int32_t* option_type, const int32_t* steps,
                                         const int32_t* american_style,
                                         int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, steps, american_style, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::tlm::jarrow_rudd_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            steps[i], american_style[i] != 0
        );
    }
    return QK_OK;
}

int32_t qk_tlm_tian_price_batch(const double* spot, const double* strike,
                                  const double* t, const double* vol,
                                  const double* r, const double* q,
                                  const int32_t* option_type, const int32_t* steps,
                                  const int32_t* american_style,
                                  int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, steps, american_style, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::tlm::tian_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            steps[i], american_style[i] != 0
        );
    }
    return QK_OK;
}

int32_t qk_tlm_leisen_reimer_price_batch(const double* spot, const double* strike,
                                           const double* t, const double* vol,
                                           const double* r, const double* q,
                                           const int32_t* option_type, const int32_t* steps,
                                           const int32_t* american_style,
                                           int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, steps, american_style, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::tlm::leisen_reimer_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            steps[i], american_style[i] != 0
        );
    }
    return QK_OK;
}

int32_t qk_tlm_trinomial_tree_price_batch(const double* spot, const double* strike,
                                            const double* t, const double* vol,
                                            const double* r, const double* q,
                                            const int32_t* option_type, const int32_t* steps,
                                            const int32_t* american_style,
                                            int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, steps, american_style, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::tlm::trinomial_tree_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            steps[i], american_style[i] != 0
        );
    }
    return QK_OK;
}

int32_t qk_tlm_derman_kani_const_local_vol_price_batch(const double* spot, const double* strike,
                                                         const double* t, const double* local_vol,
                                                         const double* r, const double* q,
                                                         const int32_t* option_type, const int32_t* steps,
                                                         const int32_t* american_style,
                                                         int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, local_vol, r, q, option_type, steps, american_style, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        auto surface = [lv = local_vol[i]](double, double) { return lv; };
        qk::tlm::ImpliedTreeConfig cfg{};
        cfg.steps = steps[i];
        cfg.american_style = (american_style[i] != 0);
        out_prices[i] = qk::tlm::derman_kani_implied_tree_price(
            spot[i], strike[i], t[i], r[i], q[i], option_type[i], surface, cfg
        );
    }
    return QK_OK;
}

/* --- Finite Difference methods --- */

double qk_fdm_explicit_fd_price(double spot, double strike, double t, double vol,
                                double r, double q, int32_t option_type,
                                int32_t time_steps, int32_t spot_steps,
                                int32_t american_style) {
    return qk::fdm::explicit_fd_price(spot, strike, t, vol, r, q, option_type,
                                      time_steps, spot_steps, american_style != 0);
}

double qk_fdm_implicit_fd_price(double spot, double strike, double t, double vol,
                                double r, double q, int32_t option_type,
                                int32_t time_steps, int32_t spot_steps,
                                int32_t american_style) {
    return qk::fdm::implicit_fd_price(spot, strike, t, vol, r, q, option_type,
                                      time_steps, spot_steps, american_style != 0);
}

double qk_fdm_crank_nicolson_price(double spot, double strike, double t, double vol,
                                   double r, double q, int32_t option_type,
                                   int32_t time_steps, int32_t spot_steps,
                                   int32_t american_style) {
    return qk::fdm::crank_nicolson_price(spot, strike, t, vol, r, q, option_type,
                                         time_steps, spot_steps, american_style != 0);
}

double qk_fdm_adi_douglas_price(double spot, double strike, double t, double r, double q,
                                double v0, double kappa, double theta_v, double sigma,
                                double rho, int32_t option_type,
                                int32_t s_steps, int32_t v_steps, int32_t time_steps) {
    qk::fdm::ADIHestonParams params{};
    params.v0 = v0;
    params.kappa = kappa;
    params.theta_v = theta_v;
    params.sigma = sigma;
    params.rho = rho;
    params.s_steps = s_steps;
    params.v_steps = v_steps;
    params.time_steps = time_steps;
    return qk::fdm::adi_douglas_price(spot, strike, t, r, q, params, option_type);
}

double qk_fdm_adi_craig_sneyd_price(double spot, double strike, double t, double r, double q,
                                    double v0, double kappa, double theta_v, double sigma,
                                    double rho, int32_t option_type,
                                    int32_t s_steps, int32_t v_steps, int32_t time_steps) {
    qk::fdm::ADIHestonParams params{};
    params.v0 = v0;
    params.kappa = kappa;
    params.theta_v = theta_v;
    params.sigma = sigma;
    params.rho = rho;
    params.s_steps = s_steps;
    params.v_steps = v_steps;
    params.time_steps = time_steps;
    return qk::fdm::adi_craig_sneyd_price(spot, strike, t, r, q, params, option_type);
}

double qk_fdm_adi_hundsdorfer_verwer_price(double spot, double strike, double t, double r, double q,
                                           double v0, double kappa, double theta_v, double sigma,
                                           double rho, int32_t option_type,
                                           int32_t s_steps, int32_t v_steps, int32_t time_steps) {
    qk::fdm::ADIHestonParams params{};
    params.v0 = v0;
    params.kappa = kappa;
    params.theta_v = theta_v;
    params.sigma = sigma;
    params.rho = rho;
    params.s_steps = s_steps;
    params.v_steps = v_steps;
    params.time_steps = time_steps;
    return qk::fdm::adi_hundsdorfer_verwer_price(spot, strike, t, r, q, params, option_type);
}

double qk_fdm_psor_price(double spot, double strike, double t, double vol,
                         double r, double q, int32_t option_type,
                         int32_t time_steps, int32_t spot_steps,
                         double omega, double tol, int32_t max_iter) {
    return qk::fdm::psor_price(spot, strike, t, vol, r, q, option_type,
                               time_steps, spot_steps, omega, tol, max_iter);
}

/* --- Finite Difference batch APIs --- */

int32_t qk_fdm_explicit_fd_price_batch(const double* spot, const double* strike,
                                         const double* t, const double* vol,
                                         const double* r, const double* q,
                                         const int32_t* option_type,
                                         const int32_t* time_steps, const int32_t* spot_steps,
                                         const int32_t* american_style,
                                         int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, time_steps, spot_steps, american_style, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::fdm::explicit_fd_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            time_steps[i], spot_steps[i], american_style[i] != 0
        );
    }
    return QK_OK;
}

int32_t qk_fdm_implicit_fd_price_batch(const double* spot, const double* strike,
                                         const double* t, const double* vol,
                                         const double* r, const double* q,
                                         const int32_t* option_type,
                                         const int32_t* time_steps, const int32_t* spot_steps,
                                         const int32_t* american_style,
                                         int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, time_steps, spot_steps, american_style, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::fdm::implicit_fd_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            time_steps[i], spot_steps[i], american_style[i] != 0
        );
    }
    return QK_OK;
}

int32_t qk_fdm_crank_nicolson_price_batch(const double* spot, const double* strike,
                                            const double* t, const double* vol,
                                            const double* r, const double* q,
                                            const int32_t* option_type,
                                            const int32_t* time_steps, const int32_t* spot_steps,
                                            const int32_t* american_style,
                                            int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, time_steps, spot_steps, american_style, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::fdm::crank_nicolson_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            time_steps[i], spot_steps[i], american_style[i] != 0
        );
    }
    return QK_OK;
}

int32_t qk_fdm_adi_douglas_price_batch(const double* spot, const double* strike,
                                          const double* t, const double* r, const double* q,
                                          const double* v0, const double* kappa,
                                          const double* theta_v, const double* sigma,
                                          const double* rho, const int32_t* option_type,
                                          const int32_t* s_steps, const int32_t* v_steps,
                                          const int32_t* time_steps,
                                          int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, r, q, v0, kappa, theta_v, sigma, rho,
                        option_type, s_steps, v_steps, time_steps, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::fdm::ADIHestonParams params{};
        params.v0 = v0[i]; params.kappa = kappa[i]; params.theta_v = theta_v[i];
        params.sigma = sigma[i]; params.rho = rho[i];
        params.s_steps = s_steps[i]; params.v_steps = v_steps[i]; params.time_steps = time_steps[i];
        out_prices[i] = qk::fdm::adi_douglas_price(spot[i], strike[i], t[i], r[i], q[i], params, option_type[i]);
    }
    return QK_OK;
}

int32_t qk_fdm_adi_craig_sneyd_price_batch(const double* spot, const double* strike,
                                              const double* t, const double* r, const double* q,
                                              const double* v0, const double* kappa,
                                              const double* theta_v, const double* sigma,
                                              const double* rho, const int32_t* option_type,
                                              const int32_t* s_steps, const int32_t* v_steps,
                                              const int32_t* time_steps,
                                              int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, r, q, v0, kappa, theta_v, sigma, rho,
                        option_type, s_steps, v_steps, time_steps, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::fdm::ADIHestonParams params{};
        params.v0 = v0[i]; params.kappa = kappa[i]; params.theta_v = theta_v[i];
        params.sigma = sigma[i]; params.rho = rho[i];
        params.s_steps = s_steps[i]; params.v_steps = v_steps[i]; params.time_steps = time_steps[i];
        out_prices[i] = qk::fdm::adi_craig_sneyd_price(spot[i], strike[i], t[i], r[i], q[i], params, option_type[i]);
    }
    return QK_OK;
}

int32_t qk_fdm_adi_hundsdorfer_verwer_price_batch(const double* spot, const double* strike,
                                                     const double* t, const double* r, const double* q,
                                                     const double* v0, const double* kappa,
                                                     const double* theta_v, const double* sigma,
                                                     const double* rho, const int32_t* option_type,
                                                     const int32_t* s_steps, const int32_t* v_steps,
                                                     const int32_t* time_steps,
                                                     int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, r, q, v0, kappa, theta_v, sigma, rho,
                        option_type, s_steps, v_steps, time_steps, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::fdm::ADIHestonParams params{};
        params.v0 = v0[i]; params.kappa = kappa[i]; params.theta_v = theta_v[i];
        params.sigma = sigma[i]; params.rho = rho[i];
        params.s_steps = s_steps[i]; params.v_steps = v_steps[i]; params.time_steps = time_steps[i];
        out_prices[i] = qk::fdm::adi_hundsdorfer_verwer_price(spot[i], strike[i], t[i], r[i], q[i], params, option_type[i]);
    }
    return QK_OK;
}

int32_t qk_fdm_psor_price_batch(const double* spot, const double* strike,
                                  const double* t, const double* vol,
                                  const double* r, const double* q,
                                  const int32_t* option_type,
                                  const int32_t* time_steps, const int32_t* spot_steps,
                                  const double* omega, const double* tol,
                                  const int32_t* max_iter,
                                  int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, time_steps, spot_steps, omega, tol, max_iter, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::fdm::psor_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            time_steps[i], spot_steps[i], omega[i], tol[i], max_iter[i]
        );
    }
    return QK_OK;
}

/* --- Monte Carlo methods --- */

double qk_mcm_standard_monte_carlo_price(double spot, double strike, double t, double vol,
                                         double r, double q, int32_t option_type,
                                         int32_t paths, uint64_t seed) {
    return qk::mcm::standard_monte_carlo_price(
        spot, strike, t, vol, r, q, option_type, paths, seed
    );
}

int32_t qk_mcm_standard_monte_carlo_price_batch(const double* spot,
                                                const double* strike,
                                                const double* t,
                                                const double* vol,
                                                const double* r,
                                                const double* q,
                                                const int32_t* option_type,
                                                const int32_t* paths,
                                                const uint64_t* seed,
                                                int32_t n,
                                                double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, seed, out_prices);
    QK_BATCH_VALIDATE_N(n);

    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::mcm::standard_monte_carlo_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            paths[i], seed[i]
        );
    }
    return QK_OK;
}

double qk_mcm_euler_maruyama_price(double spot, double strike, double t, double vol,
                                   double r, double q, int32_t option_type,
                                   int32_t paths, int32_t steps, uint64_t seed) {
    return qk::mcm::euler_maruyama_price(
        spot, strike, t, vol, r, q, option_type, paths, steps, seed
    );
}

double qk_mcm_milstein_price(double spot, double strike, double t, double vol,
                             double r, double q, int32_t option_type,
                             int32_t paths, int32_t steps, uint64_t seed) {
    return qk::mcm::milstein_price(
        spot, strike, t, vol, r, q, option_type, paths, steps, seed
    );
}

double qk_mcm_longstaff_schwartz_price(double spot, double strike, double t, double vol,
                                       double r, double q, int32_t option_type,
                                       int32_t paths, int32_t steps, uint64_t seed) {
    return qk::mcm::longstaff_schwartz_price(
        spot, strike, t, vol, r, q, option_type, paths, steps, seed
    );
}

double qk_mcm_quasi_monte_carlo_sobol_price(double spot, double strike, double t, double vol,
                                            double r, double q, int32_t option_type,
                                            int32_t paths) {
    return qk::mcm::quasi_monte_carlo_sobol_price(
        spot, strike, t, vol, r, q, option_type, paths
    );
}

double qk_mcm_quasi_monte_carlo_halton_price(double spot, double strike, double t, double vol,
                                             double r, double q, int32_t option_type,
                                             int32_t paths) {
    return qk::mcm::quasi_monte_carlo_halton_price(
        spot, strike, t, vol, r, q, option_type, paths
    );
}

double qk_mcm_multilevel_monte_carlo_price(double spot, double strike, double t, double vol,
                                           double r, double q, int32_t option_type,
                                           int32_t base_paths, int32_t levels, int32_t base_steps,
                                           uint64_t seed) {
    return qk::mcm::multilevel_monte_carlo_price(
        spot, strike, t, vol, r, q, option_type, base_paths, levels, base_steps, seed
    );
}

double qk_mcm_importance_sampling_price(double spot, double strike, double t, double vol,
                                        double r, double q, int32_t option_type,
                                        int32_t paths, double shift, uint64_t seed) {
    return qk::mcm::importance_sampling_price(
        spot, strike, t, vol, r, q, option_type, paths, shift, seed
    );
}

double qk_mcm_control_variates_price(double spot, double strike, double t, double vol,
                                     double r, double q, int32_t option_type,
                                     int32_t paths, uint64_t seed) {
    return qk::mcm::control_variates_price(
        spot, strike, t, vol, r, q, option_type, paths, seed
    );
}

double qk_mcm_antithetic_variates_price(double spot, double strike, double t, double vol,
                                        double r, double q, int32_t option_type,
                                        int32_t paths, uint64_t seed) {
    return qk::mcm::antithetic_variates_price(
        spot, strike, t, vol, r, q, option_type, paths, seed
    );
}

double qk_mcm_stratified_sampling_price(double spot, double strike, double t, double vol,
                                        double r, double q, int32_t option_type,
                                        int32_t paths, uint64_t seed) {
    return qk::mcm::stratified_sampling_price(
        spot, strike, t, vol, r, q, option_type, paths, seed
    );
}

int32_t qk_mcm_euler_maruyama_price_batch(const double* spot, const double* strike,
                                            const double* t, const double* vol,
                                            const double* r, const double* q,
                                            const int32_t* option_type, const int32_t* paths,
                                            const int32_t* steps, const uint64_t* seed,
                                            int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, steps, seed, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::mcm::euler_maruyama_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            paths[i], steps[i], seed[i]
        );
    }
    return QK_OK;
}

int32_t qk_mcm_milstein_price_batch(const double* spot, const double* strike,
                                      const double* t, const double* vol,
                                      const double* r, const double* q,
                                      const int32_t* option_type, const int32_t* paths,
                                      const int32_t* steps, const uint64_t* seed,
                                      int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, steps, seed, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::mcm::milstein_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            paths[i], steps[i], seed[i]
        );
    }
    return QK_OK;
}

int32_t qk_mcm_longstaff_schwartz_price_batch(const double* spot, const double* strike,
                                                const double* t, const double* vol,
                                                const double* r, const double* q,
                                                const int32_t* option_type, const int32_t* paths,
                                                const int32_t* steps, const uint64_t* seed,
                                                int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, steps, seed, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::mcm::longstaff_schwartz_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            paths[i], steps[i], seed[i]
        );
    }
    return QK_OK;
}

int32_t qk_mcm_quasi_monte_carlo_sobol_price_batch(const double* spot, const double* strike,
                                                     const double* t, const double* vol,
                                                     const double* r, const double* q,
                                                     const int32_t* option_type, const int32_t* paths,
                                                     int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::mcm::quasi_monte_carlo_sobol_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], paths[i]
        );
    }
    return QK_OK;
}

int32_t qk_mcm_quasi_monte_carlo_halton_price_batch(const double* spot, const double* strike,
                                                      const double* t, const double* vol,
                                                      const double* r, const double* q,
                                                      const int32_t* option_type, const int32_t* paths,
                                                      int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::mcm::quasi_monte_carlo_halton_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], paths[i]
        );
    }
    return QK_OK;
}

int32_t qk_mcm_multilevel_monte_carlo_price_batch(const double* spot, const double* strike,
                                                    const double* t, const double* vol,
                                                    const double* r, const double* q,
                                                    const int32_t* option_type, const int32_t* base_paths,
                                                    const int32_t* levels, const int32_t* base_steps,
                                                    const uint64_t* seed,
                                                    int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, base_paths, levels, base_steps, seed, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::mcm::multilevel_monte_carlo_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            base_paths[i], levels[i], base_steps[i], seed[i]
        );
    }
    return QK_OK;
}

int32_t qk_mcm_importance_sampling_price_batch(const double* spot, const double* strike,
                                                const double* t, const double* vol,
                                                const double* r, const double* q,
                                                const int32_t* option_type, const int32_t* paths,
                                                const double* shift, const uint64_t* seed,
                                                int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, shift, seed, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::mcm::importance_sampling_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            paths[i], shift[i], seed[i]
        );
    }
    return QK_OK;
}

int32_t qk_mcm_control_variates_price_batch(const double* spot, const double* strike,
                                              const double* t, const double* vol,
                                              const double* r, const double* q,
                                              const int32_t* option_type, const int32_t* paths,
                                              const uint64_t* seed,
                                              int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, seed, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::mcm::control_variates_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            paths[i], seed[i]
        );
    }
    return QK_OK;
}

int32_t qk_mcm_antithetic_variates_price_batch(const double* spot, const double* strike,
                                                 const double* t, const double* vol,
                                                 const double* r, const double* q,
                                                 const int32_t* option_type, const int32_t* paths,
                                                 const uint64_t* seed,
                                                 int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, seed, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::mcm::antithetic_variates_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            paths[i], seed[i]
        );
    }
    return QK_OK;
}

int32_t qk_mcm_stratified_sampling_price_batch(const double* spot, const double* strike,
                                                 const double* t, const double* vol,
                                                 const double* r, const double* q,
                                                 const int32_t* option_type, const int32_t* paths,
                                                 const uint64_t* seed,
                                                 int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, seed, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        out_prices[i] = qk::mcm::stratified_sampling_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i],
            paths[i], seed[i]
        );
    }
    return QK_OK;
}

/* --- Fourier transform methods --- */

double qk_ftm_carr_madan_fft_price(double spot, double strike, double t, double vol,
                                   double r, double q, int32_t option_type,
                                   int32_t grid_size, double eta, double alpha) {
    qk::ftm::CarrMadanFFTParams params{};
    params.grid_size = grid_size;
    params.eta = eta;
    params.alpha = alpha;
    return qk::ftm::carr_madan_fft_price(spot, strike, t, vol, r, q, option_type, params);
}

int32_t qk_ftm_carr_madan_fft_price_batch(const double* spot,
                                          const double* strike,
                                          const double* t,
                                          const double* vol,
                                          const double* r,
                                          const double* q,
                                          const int32_t* option_type,
                                          const int32_t* grid_size,
                                          const double* eta,
                                          const double* alpha,
                                          int32_t n,
                                          double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, grid_size, eta, alpha, out_prices);
    QK_BATCH_VALIDATE_N(n);

    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::ftm::CarrMadanFFTParams params{};
        params.grid_size = grid_size[i];
        params.eta = eta[i];
        params.alpha = alpha[i];
        out_prices[i] = qk::ftm::carr_madan_fft_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params
        );
    }
    return QK_OK;
}

double qk_ftm_cos_fang_oosterlee_price(double spot, double strike, double t, double vol,
                                       double r, double q, int32_t option_type,
                                       int32_t n_terms, double truncation_width) {
    qk::ftm::COSMethodParams params{};
    params.n_terms = n_terms;
    params.truncation_width = truncation_width;
    return qk::ftm::cos_method_fang_oosterlee_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_ftm_fractional_fft_price(double spot, double strike, double t, double vol,
                                   double r, double q, int32_t option_type,
                                   int32_t grid_size, double eta, double lambda,
                                   double alpha) {
    qk::ftm::FractionalFFTParams params{};
    params.grid_size = grid_size;
    params.eta = eta;
    params.lambda = lambda;
    params.alpha = alpha;
    return qk::ftm::fractional_fft_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_ftm_lewis_fourier_inversion_price(double spot, double strike, double t, double vol,
                                            double r, double q, int32_t option_type,
                                            int32_t integration_steps,
                                            double integration_limit) {
    qk::ftm::LewisFourierInversionParams params{};
    params.integration_steps = integration_steps;
    params.integration_limit = integration_limit;
    return qk::ftm::lewis_fourier_inversion_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_ftm_hilbert_transform_price(double spot, double strike, double t, double vol,
                                      double r, double q, int32_t option_type,
                                      int32_t integration_steps,
                                      double integration_limit) {
    qk::ftm::HilbertTransformParams params{};
    params.integration_steps = integration_steps;
    params.integration_limit = integration_limit;
    return qk::ftm::hilbert_transform_price(spot, strike, t, vol, r, q, option_type, params);
}

int32_t qk_ftm_cos_fang_oosterlee_price_batch(const double* spot, const double* strike,
                                                const double* t, const double* vol,
                                                const double* r, const double* q,
                                                const int32_t* option_type, const int32_t* n_terms,
                                                const double* truncation_width,
                                                int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, n_terms, truncation_width, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::ftm::COSMethodParams params{};
        params.n_terms = n_terms[i];
        params.truncation_width = truncation_width[i];
        out_prices[i] = qk::ftm::cos_method_fang_oosterlee_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params
        );
    }
    return QK_OK;
}

int32_t qk_ftm_fractional_fft_price_batch(const double* spot, const double* strike,
                                            const double* t, const double* vol,
                                            const double* r, const double* q,
                                            const int32_t* option_type, const int32_t* grid_size,
                                            const double* eta, const double* lambda_,
                                            const double* alpha,
                                            int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, grid_size, eta, lambda_, alpha, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::ftm::FractionalFFTParams params{};
        params.grid_size = grid_size[i];
        params.eta = eta[i];
        params.lambda = lambda_[i];
        params.alpha = alpha[i];
        out_prices[i] = qk::ftm::fractional_fft_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params
        );
    }
    return QK_OK;
}

int32_t qk_ftm_lewis_fourier_inversion_price_batch(const double* spot, const double* strike,
                                                     const double* t, const double* vol,
                                                     const double* r, const double* q,
                                                     const int32_t* option_type,
                                                     const int32_t* integration_steps,
                                                     const double* integration_limit,
                                                     int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, integration_steps, integration_limit, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::ftm::LewisFourierInversionParams params{};
        params.integration_steps = integration_steps[i];
        params.integration_limit = integration_limit[i];
        out_prices[i] = qk::ftm::lewis_fourier_inversion_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params
        );
    }
    return QK_OK;
}

int32_t qk_ftm_hilbert_transform_price_batch(const double* spot, const double* strike,
                                               const double* t, const double* vol,
                                               const double* r, const double* q,
                                               const int32_t* option_type,
                                               const int32_t* integration_steps,
                                               const double* integration_limit,
                                               int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, integration_steps, integration_limit, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::ftm::HilbertTransformParams params{};
        params.integration_steps = integration_steps[i];
        params.integration_limit = integration_limit[i];
        out_prices[i] = qk::ftm::hilbert_transform_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params
        );
    }
    return QK_OK;
}

/* --- Integral quadrature methods --- */

double qk_iqm_gauss_hermite_price(double spot, double strike, double t, double vol,
                                  double r, double q, int32_t option_type,
                                  int32_t n_points) {
    qk::iqm::GaussHermiteParams params{};
    params.n_points = n_points;
    return qk::iqm::gauss_hermite_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_iqm_gauss_laguerre_price(double spot, double strike, double t, double vol,
                                   double r, double q, int32_t option_type,
                                   int32_t n_points) {
    qk::iqm::GaussLaguerreParams params{};
    params.n_points = n_points;
    return qk::iqm::gauss_laguerre_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_iqm_gauss_legendre_price(double spot, double strike, double t, double vol,
                                   double r, double q, int32_t option_type,
                                   int32_t n_points, double integration_limit) {
    qk::iqm::GaussLegendreParams params{};
    params.n_points = n_points;
    params.integration_limit = integration_limit;
    return qk::iqm::gauss_legendre_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_iqm_adaptive_quadrature_price(double spot, double strike, double t, double vol,
                                        double r, double q, int32_t option_type,
                                        double abs_tol, double rel_tol,
                                        int32_t max_depth, double integration_limit) {
    qk::iqm::AdaptiveQuadratureParams params{};
    params.abs_tol = abs_tol;
    params.rel_tol = rel_tol;
    params.max_depth = max_depth;
    params.integration_limit = integration_limit;
    return qk::iqm::adaptive_quadrature_price(spot, strike, t, vol, r, q, option_type, params);
}

/* --- Regression approximation methods --- */

double qk_ram_polynomial_chaos_expansion_price(double spot, double strike, double t, double vol,
                                               double r, double q, int32_t option_type,
                                               int32_t polynomial_order,
                                               int32_t quadrature_points) {
    qk::ram::PolynomialChaosExpansionParams params{};
    params.polynomial_order = polynomial_order;
    params.quadrature_points = quadrature_points;
    return qk::ram::polynomial_chaos_expansion_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_ram_radial_basis_function_price(double spot, double strike, double t, double vol,
                                          double r, double q, int32_t option_type,
                                          int32_t centers, double rbf_shape, double ridge) {
    qk::ram::RadialBasisFunctionParams params{};
    params.centers = centers;
    params.rbf_shape = rbf_shape;
    params.ridge = ridge;
    return qk::ram::radial_basis_function_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_ram_sparse_grid_collocation_price(double spot, double strike, double t, double vol,
                                            double r, double q, int32_t option_type,
                                            int32_t level, int32_t nodes_per_dim) {
    qk::ram::SparseGridCollocationParams params{};
    params.level = level;
    params.nodes_per_dim = nodes_per_dim;
    return qk::ram::sparse_grid_collocation_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_ram_proper_orthogonal_decomposition_price(double spot, double strike, double t, double vol,
                                                    double r, double q, int32_t option_type,
                                                    int32_t modes, int32_t snapshots) {
    qk::ram::ProperOrthogonalDecompositionParams params{};
    params.modes = modes;
    params.snapshots = snapshots;
    return qk::ram::proper_orthogonal_decomposition_price(spot, strike, t, vol, r, q, option_type, params);
}

/* --- Adjoint Greeks methods --- */

double qk_agm_pathwise_derivative_delta(double spot, double strike, double t, double vol,
                                        double r, double q, int32_t option_type,
                                        int32_t paths, uint64_t seed) {
    qk::agm::PathwiseDerivativeParams params{};
    params.paths = paths;
    params.seed = seed;
    return qk::agm::pathwise_derivative_delta(spot, strike, t, vol, r, q, option_type, params);
}

double qk_agm_likelihood_ratio_delta(double spot, double strike, double t, double vol,
                                     double r, double q, int32_t option_type,
                                     int32_t paths, uint64_t seed, double weight_clip) {
    qk::agm::LikelihoodRatioParams params{};
    params.paths = paths;
    params.seed = seed;
    params.weight_clip = weight_clip;
    return qk::agm::likelihood_ratio_delta(spot, strike, t, vol, r, q, option_type, params);
}

double qk_agm_aad_delta(double spot, double strike, double t, double vol,
                        double r, double q, int32_t option_type,
                        int32_t tape_steps, double regularization) {
    qk::agm::AadParams params{};
    params.tape_steps = tape_steps;
    params.regularization = regularization;
    return qk::agm::aad_delta(spot, strike, t, vol, r, q, option_type, params);
}

/* --- Machine learning methods --- */

double qk_mlm_deep_bsde_price(double spot, double strike, double t, double vol,
                              double r, double q, int32_t option_type,
                              int32_t time_steps, int32_t hidden_width,
                              int32_t training_epochs, double learning_rate) {
    qk::mlm::DeepBsdeParams params{};
    params.time_steps = time_steps;
    params.hidden_width = hidden_width;
    params.training_epochs = training_epochs;
    params.learning_rate = learning_rate;
    return qk::mlm::deep_bsde_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_mlm_pinns_price(double spot, double strike, double t, double vol,
                          double r, double q, int32_t option_type,
                          int32_t collocation_points, int32_t boundary_points,
                          int32_t epochs, double loss_balance) {
    qk::mlm::PinnsParams params{};
    params.collocation_points = collocation_points;
    params.boundary_points = boundary_points;
    params.epochs = epochs;
    params.loss_balance = loss_balance;
    return qk::mlm::pinns_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_mlm_deep_hedging_price(double spot, double strike, double t, double vol,
                                 double r, double q, int32_t option_type,
                                 int32_t rehedge_steps, double risk_aversion,
                                 int32_t scenarios, uint64_t seed) {
    qk::mlm::DeepHedgingParams params{};
    params.rehedge_steps = rehedge_steps;
    params.risk_aversion = risk_aversion;
    params.scenarios = scenarios;
    params.seed = seed;
    return qk::mlm::deep_hedging_price(spot, strike, t, vol, r, q, option_type, params);
}

double qk_mlm_neural_sde_calibration_price(double spot, double strike, double t, double vol,
                                           double r, double q, int32_t option_type,
                                           double target_implied_vol,
                                           int32_t calibration_steps,
                                           double regularization) {
    qk::mlm::NeuralSdeCalibrationParams params{};
    params.target_implied_vol = target_implied_vol;
    params.calibration_steps = calibration_steps;
    params.regularization = regularization;
    return qk::mlm::neural_sde_calibration_price(spot, strike, t, vol, r, q, option_type, params);
}

/* --- Integral quadrature batch APIs --- */

int32_t qk_iqm_gauss_hermite_price_batch(const double* spot, const double* strike,
                                            const double* t, const double* vol,
                                            const double* r, const double* q,
                                            const int32_t* option_type,
                                            const int32_t* n_points,
                                            int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, n_points, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::iqm::GaussHermiteParams params{};
        params.n_points = n_points[i];
        out_prices[i] = qk::iqm::gauss_hermite_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

int32_t qk_iqm_gauss_laguerre_price_batch(const double* spot, const double* strike,
                                             const double* t, const double* vol,
                                             const double* r, const double* q,
                                             const int32_t* option_type,
                                             const int32_t* n_points,
                                             int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, n_points, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::iqm::GaussLaguerreParams params{};
        params.n_points = n_points[i];
        out_prices[i] = qk::iqm::gauss_laguerre_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

int32_t qk_iqm_gauss_legendre_price_batch(const double* spot, const double* strike,
                                             const double* t, const double* vol,
                                             const double* r, const double* q,
                                             const int32_t* option_type,
                                             const int32_t* n_points,
                                             const double* integration_limit,
                                             int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, n_points, integration_limit, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::iqm::GaussLegendreParams params{};
        params.n_points = n_points[i];
        params.integration_limit = integration_limit[i];
        out_prices[i] = qk::iqm::gauss_legendre_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

int32_t qk_iqm_adaptive_quadrature_price_batch(const double* spot, const double* strike,
                                                  const double* t, const double* vol,
                                                  const double* r, const double* q,
                                                  const int32_t* option_type,
                                                  const double* abs_tol, const double* rel_tol,
                                                  const int32_t* max_depth,
                                                  const double* integration_limit,
                                                  int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, abs_tol, rel_tol, max_depth, integration_limit, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::iqm::AdaptiveQuadratureParams params{};
        params.abs_tol = abs_tol[i]; params.rel_tol = rel_tol[i];
        params.max_depth = max_depth[i]; params.integration_limit = integration_limit[i];
        out_prices[i] = qk::iqm::adaptive_quadrature_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

/* --- Regression approximation batch APIs --- */

int32_t qk_ram_polynomial_chaos_expansion_price_batch(const double* spot, const double* strike,
                                                        const double* t, const double* vol,
                                                        const double* r, const double* q,
                                                        const int32_t* option_type,
                                                        const int32_t* polynomial_order,
                                                        const int32_t* quadrature_points,
                                                        int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, polynomial_order, quadrature_points, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::ram::PolynomialChaosExpansionParams params{};
        params.polynomial_order = polynomial_order[i]; params.quadrature_points = quadrature_points[i];
        out_prices[i] = qk::ram::polynomial_chaos_expansion_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

int32_t qk_ram_radial_basis_function_price_batch(const double* spot, const double* strike,
                                                   const double* t, const double* vol,
                                                   const double* r, const double* q,
                                                   const int32_t* option_type,
                                                   const int32_t* centers, const double* rbf_shape,
                                                   const double* ridge,
                                                   int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, centers, rbf_shape, ridge, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::ram::RadialBasisFunctionParams params{};
        params.centers = centers[i]; params.rbf_shape = rbf_shape[i]; params.ridge = ridge[i];
        out_prices[i] = qk::ram::radial_basis_function_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

int32_t qk_ram_sparse_grid_collocation_price_batch(const double* spot, const double* strike,
                                                      const double* t, const double* vol,
                                                      const double* r, const double* q,
                                                      const int32_t* option_type,
                                                      const int32_t* level, const int32_t* nodes_per_dim,
                                                      int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, level, nodes_per_dim, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::ram::SparseGridCollocationParams params{};
        params.level = level[i]; params.nodes_per_dim = nodes_per_dim[i];
        out_prices[i] = qk::ram::sparse_grid_collocation_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

int32_t qk_ram_proper_orthogonal_decomposition_price_batch(const double* spot, const double* strike,
                                                             const double* t, const double* vol,
                                                             const double* r, const double* q,
                                                             const int32_t* option_type,
                                                             const int32_t* modes, const int32_t* snapshots,
                                                             int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, modes, snapshots, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::ram::ProperOrthogonalDecompositionParams params{};
        params.modes = modes[i]; params.snapshots = snapshots[i];
        out_prices[i] = qk::ram::proper_orthogonal_decomposition_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

/* --- Adjoint Greeks batch APIs --- */

int32_t qk_agm_pathwise_derivative_delta_batch(const double* spot, const double* strike,
                                                  const double* t, const double* vol,
                                                  const double* r, const double* q,
                                                  const int32_t* option_type,
                                                  const int32_t* paths, const uint64_t* seed,
                                                  int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, seed, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::agm::PathwiseDerivativeParams params{};
        params.paths = paths[i]; params.seed = seed[i];
        out_prices[i] = qk::agm::pathwise_derivative_delta(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

int32_t qk_agm_likelihood_ratio_delta_batch(const double* spot, const double* strike,
                                               const double* t, const double* vol,
                                               const double* r, const double* q,
                                               const int32_t* option_type,
                                               const int32_t* paths, const uint64_t* seed,
                                               const double* weight_clip,
                                               int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, paths, seed, weight_clip, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::agm::LikelihoodRatioParams params{};
        params.paths = paths[i]; params.seed = seed[i]; params.weight_clip = weight_clip[i];
        out_prices[i] = qk::agm::likelihood_ratio_delta(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

int32_t qk_agm_aad_delta_batch(const double* spot, const double* strike,
                                  const double* t, const double* vol,
                                  const double* r, const double* q,
                                  const int32_t* option_type,
                                  const int32_t* tape_steps, const double* regularization,
                                  int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, tape_steps, regularization, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp simd
    for (int32_t i = 0; i < n; ++i) {
        qk::agm::AadParams params{};
        params.tape_steps = tape_steps[i]; params.regularization = regularization[i];
        out_prices[i] = qk::agm::aad_delta(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

/* --- Machine learning batch APIs --- */

int32_t qk_mlm_deep_bsde_price_batch(const double* spot, const double* strike,
                                        const double* t, const double* vol,
                                        const double* r, const double* q,
                                        const int32_t* option_type,
                                        const int32_t* time_steps, const int32_t* hidden_width,
                                        const int32_t* training_epochs, const double* learning_rate,
                                        int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, time_steps, hidden_width, training_epochs, learning_rate, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::mlm::DeepBsdeParams params{};
        params.time_steps = time_steps[i]; params.hidden_width = hidden_width[i];
        params.training_epochs = training_epochs[i]; params.learning_rate = learning_rate[i];
        out_prices[i] = qk::mlm::deep_bsde_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

int32_t qk_mlm_pinns_price_batch(const double* spot, const double* strike,
                                    const double* t, const double* vol,
                                    const double* r, const double* q,
                                    const int32_t* option_type,
                                    const int32_t* collocation_points, const int32_t* boundary_points,
                                    const int32_t* epochs, const double* loss_balance,
                                    int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, collocation_points, boundary_points, epochs, loss_balance, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::mlm::PinnsParams params{};
        params.collocation_points = collocation_points[i]; params.boundary_points = boundary_points[i];
        params.epochs = epochs[i]; params.loss_balance = loss_balance[i];
        out_prices[i] = qk::mlm::pinns_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

int32_t qk_mlm_deep_hedging_price_batch(const double* spot, const double* strike,
                                           const double* t, const double* vol,
                                           const double* r, const double* q,
                                           const int32_t* option_type,
                                           const int32_t* rehedge_steps, const double* risk_aversion,
                                           const int32_t* scenarios, const uint64_t* seed,
                                           int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, rehedge_steps, risk_aversion, scenarios, seed, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::mlm::DeepHedgingParams params{};
        params.rehedge_steps = rehedge_steps[i]; params.risk_aversion = risk_aversion[i];
        params.scenarios = scenarios[i]; params.seed = seed[i];
        out_prices[i] = qk::mlm::deep_hedging_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

int32_t qk_mlm_neural_sde_calibration_price_batch(const double* spot, const double* strike,
                                                     const double* t, const double* vol,
                                                     const double* r, const double* q,
                                                     const int32_t* option_type,
                                                     const double* target_implied_vol,
                                                     const int32_t* calibration_steps,
                                                     const double* regularization,
                                                     int32_t n, double* out_prices) {
    QK_BATCH_NULL_CHECK(spot, strike, t, vol, r, q, option_type, target_implied_vol, calibration_steps, regularization, out_prices);
    QK_BATCH_VALIDATE_N(n);
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < n; ++i) {
        qk::mlm::NeuralSdeCalibrationParams params{};
        params.target_implied_vol = target_implied_vol[i]; params.calibration_steps = calibration_steps[i];
        params.regularization = regularization[i];
        out_prices[i] = qk::mlm::neural_sde_calibration_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], params);
    }
    return QK_OK;
}

} /* extern "C" */
