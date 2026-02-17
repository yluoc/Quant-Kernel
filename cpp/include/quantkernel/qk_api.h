#ifndef QK_API_H
#define QK_API_H

#include "qk_abi.h"

#ifdef _WIN32
  #define QK_EXPORT __declspec(dllexport)
#else
  #define QK_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

QK_EXPORT void    qk_abi_version(int32_t* major, int32_t* minor);
QK_EXPORT int32_t qk_plugin_get_api(int32_t host_abi_major,
                                    int32_t host_abi_minor,
                                    const QKPluginAPI** out_api);
/* --- Closed-form formulas --- */
QK_EXPORT double qk_cf_black_scholes_merton_price(double spot, double strike, double t, double vol,
                                                   double r, double q, int32_t option_type);
QK_EXPORT double qk_cf_black76_price(double forward, double strike, double t, double vol,
                                     double r, int32_t option_type);
QK_EXPORT double qk_cf_bachelier_price(double forward, double strike, double t, double normal_vol,
                                       double r, int32_t option_type);
QK_EXPORT double qk_cf_heston_price_cf(double spot, double strike, double t, double r, double q,
                                       double v0, double kappa, double theta, double sigma, double rho,
                                       int32_t option_type, int32_t integration_steps, double integration_limit);
QK_EXPORT double qk_cf_merton_jump_diffusion_price(double spot, double strike, double t, double vol,
                                                   double r, double q, double jump_intensity,
                                                   double jump_mean, double jump_vol, int32_t max_terms,
                                                   int32_t option_type);
QK_EXPORT double qk_cf_variance_gamma_price_cf(double spot, double strike, double t, double r, double q,
                                               double sigma, double theta, double nu, int32_t option_type,
                                               int32_t integration_steps, double integration_limit);
QK_EXPORT double qk_cf_sabr_hagan_lognormal_iv(double forward, double strike, double t,
                                               double alpha, double beta, double rho, double nu);
QK_EXPORT double qk_cf_sabr_hagan_black76_price(double forward, double strike, double t, double r,
                                                double alpha, double beta, double rho, double nu,
                                                int32_t option_type);
QK_EXPORT double qk_cf_dupire_local_vol(double strike, double t, double call_price,
                                        double dC_dT, double dC_dK, double d2C_dK2,
                                        double r, double q);
/* --- Tree/Lattice methods --- */
QK_EXPORT double qk_tlm_crr_price(double spot, double strike, double t, double vol,
                                  double r, double q, int32_t option_type,
                                  int32_t steps, int32_t american_style);
QK_EXPORT double qk_tlm_jarrow_rudd_price(double spot, double strike, double t, double vol,
                                          double r, double q, int32_t option_type,
                                          int32_t steps, int32_t american_style);
QK_EXPORT double qk_tlm_tian_price(double spot, double strike, double t, double vol,
                                   double r, double q, int32_t option_type,
                                   int32_t steps, int32_t american_style);
QK_EXPORT double qk_tlm_leisen_reimer_price(double spot, double strike, double t, double vol,
                                            double r, double q, int32_t option_type,
                                            int32_t steps, int32_t american_style);
QK_EXPORT double qk_tlm_trinomial_tree_price(double spot, double strike, double t, double vol,
                                             double r, double q, int32_t option_type,
                                             int32_t steps, int32_t american_style);
QK_EXPORT double qk_tlm_derman_kani_const_local_vol_price(double spot, double strike, double t,
                                                          double local_vol, double r, double q,
                                                          int32_t option_type, int32_t steps,
                                                          int32_t american_style);
/* --- Finite Difference methods --- */
QK_EXPORT double qk_fdm_explicit_fd_price(double spot, double strike, double t, double vol,
                                          double r, double q, int32_t option_type,
                                          int32_t time_steps, int32_t spot_steps,
                                          int32_t american_style);
QK_EXPORT double qk_fdm_implicit_fd_price(double spot, double strike, double t, double vol,
                                          double r, double q, int32_t option_type,
                                          int32_t time_steps, int32_t spot_steps,
                                          int32_t american_style);
QK_EXPORT double qk_fdm_crank_nicolson_price(double spot, double strike, double t, double vol,
                                             double r, double q, int32_t option_type,
                                             int32_t time_steps, int32_t spot_steps,
                                             int32_t american_style);
QK_EXPORT double qk_fdm_adi_douglas_price(double spot, double strike, double t, double r, double q,
                                          double v0, double kappa, double theta_v, double sigma,
                                          double rho, int32_t option_type,
                                          int32_t s_steps, int32_t v_steps, int32_t time_steps);
QK_EXPORT double qk_fdm_adi_craig_sneyd_price(double spot, double strike, double t, double r, double q,
                                              double v0, double kappa, double theta_v, double sigma,
                                              double rho, int32_t option_type,
                                              int32_t s_steps, int32_t v_steps, int32_t time_steps);
QK_EXPORT double qk_fdm_adi_hundsdorfer_verwer_price(double spot, double strike, double t, double r, double q,
                                                     double v0, double kappa, double theta_v, double sigma,
                                                     double rho, int32_t option_type,
                                                     int32_t s_steps, int32_t v_steps, int32_t time_steps);
QK_EXPORT double qk_fdm_psor_price(double spot, double strike, double t, double vol,
                                   double r, double q, int32_t option_type,
                                   int32_t time_steps, int32_t spot_steps,
                                   double omega, double tol, int32_t max_iter);
/* --- Monte Carlo methods --- */
QK_EXPORT double qk_mcm_standard_monte_carlo_price(double spot, double strike, double t, double vol,
                                                   double r, double q, int32_t option_type,
                                                   int32_t paths, uint64_t seed);
QK_EXPORT double qk_mcm_euler_maruyama_price(double spot, double strike, double t, double vol,
                                             double r, double q, int32_t option_type,
                                             int32_t paths, int32_t steps, uint64_t seed);
QK_EXPORT double qk_mcm_milstein_price(double spot, double strike, double t, double vol,
                                       double r, double q, int32_t option_type,
                                       int32_t paths, int32_t steps, uint64_t seed);
QK_EXPORT double qk_mcm_longstaff_schwartz_price(double spot, double strike, double t, double vol,
                                                 double r, double q, int32_t option_type,
                                                 int32_t paths, int32_t steps, uint64_t seed);
QK_EXPORT double qk_mcm_quasi_monte_carlo_sobol_price(double spot, double strike, double t, double vol,
                                                      double r, double q, int32_t option_type,
                                                      int32_t paths);
QK_EXPORT double qk_mcm_quasi_monte_carlo_halton_price(double spot, double strike, double t, double vol,
                                                       double r, double q, int32_t option_type,
                                                       int32_t paths);
QK_EXPORT double qk_mcm_multilevel_monte_carlo_price(double spot, double strike, double t, double vol,
                                                     double r, double q, int32_t option_type,
                                                     int32_t base_paths, int32_t levels, int32_t base_steps,
                                                     uint64_t seed);
QK_EXPORT double qk_mcm_importance_sampling_price(double spot, double strike, double t, double vol,
                                                  double r, double q, int32_t option_type,
                                                  int32_t paths, double shift, uint64_t seed);
QK_EXPORT double qk_mcm_control_variates_price(double spot, double strike, double t, double vol,
                                               double r, double q, int32_t option_type,
                                               int32_t paths, uint64_t seed);
QK_EXPORT double qk_mcm_antithetic_variates_price(double spot, double strike, double t, double vol,
                                                  double r, double q, int32_t option_type,
                                                  int32_t paths, uint64_t seed);
QK_EXPORT double qk_mcm_stratified_sampling_price(double spot, double strike, double t, double vol,
                                                  double r, double q, int32_t option_type,
                                                  int32_t paths, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* QK_API_H */
