#include <quantkernel/qk_api.h>

#include "algorithms/closed_form_semi_analytical/closed_form_models.h"
#include "algorithms/tree_lattice_methods/tree_lattice_models.h"

namespace {

const QKPluginAPI k_plugin_api = {
    QK_ABI_MAJOR,
    QK_ABI_MINOR,
    "quantkernel.cpp.closed_form_and_trees.v4"
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
    if (host_abi_major != QK_ABI_MAJOR || host_abi_minor < QK_ABI_MINOR) {
        *out_api = nullptr;
        return QK_ERR_ABI_MISMATCH;
    }
    *out_api = &k_plugin_api;
    return QK_OK;
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

double qk_tlm_crr_price(double spot, double strike, double t, double vol,
                        double r, double q, int32_t option_type,
                        int32_t steps, int32_t american_style) {
    return qk::tlm::crr_price(spot, strike, t, vol, r, q, option_type, steps, american_style != 0);
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

} /* extern "C" */
