#include <quantkernel/qk_api.h>

#include "algorithms/closed_form_semi_analytical/closed_form_models.h"
#include "algorithms/tree_lattice_methods/tree_lattice_models.h"
#include "algorithms/finite_difference_methods/finite_difference_models.h"
#include "algorithms/monte_carlo_methods/monte_carlo_models.h"
#include "algorithms/fourier_transform_methods/fourier_transform_models.h"

namespace {

const QKPluginAPI k_plugin_api = {
    QK_ABI_MAJOR,
    QK_ABI_MINOR,
    "quantkernel.cpp.closed_form_trees_fdm_mcm_ftm.v7"
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

/* --- Monte Carlo methods --- */

double qk_mcm_standard_monte_carlo_price(double spot, double strike, double t, double vol,
                                         double r, double q, int32_t option_type,
                                         int32_t paths, uint64_t seed) {
    return qk::mcm::standard_monte_carlo_price(
        spot, strike, t, vol, r, q, option_type, paths, seed
    );
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

} /* extern "C" */
