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
QK_EXPORT const char* qk_get_last_error(void);
QK_EXPORT void        qk_clear_last_error(void);
QK_EXPORT double qk_cf_black_scholes_merton_price(double spot, double strike, double t, double vol,
                                                   double r, double q, int32_t option_type);
QK_EXPORT double qk_cf_black76_price(double forward, double strike, double t, double vol,
                                     double r, int32_t option_type);
QK_EXPORT double qk_cf_bachelier_price(double forward, double strike, double t, double normal_vol,
                                       double r, int32_t option_type);
QK_EXPORT int32_t qk_cf_black_scholes_merton_price_batch(const double* spot,
                                                         const double* strike,
                                                         const double* t,
                                                         const double* vol,
                                                         const double* r,
                                                         const double* q,
                                                         const int32_t* option_type,
                                                         int32_t n,
                                                         double* out_prices);
QK_EXPORT int32_t qk_cf_black76_price_batch(const double* forward,
                                            const double* strike,
                                            const double* t,
                                            const double* vol,
                                            const double* r,
                                            const int32_t* option_type,
                                            int32_t n,
                                            double* out_prices);
QK_EXPORT int32_t qk_cf_bachelier_price_batch(const double* forward,
                                              const double* strike,
                                              const double* t,
                                              const double* normal_vol,
                                              const double* r,
                                              const int32_t* option_type,
                                              int32_t n,
                                              double* out_prices);
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
QK_EXPORT int32_t qk_cf_heston_price_cf_batch(const double* spot, const double* strike,
                                               const double* t, const double* r, const double* q,
                                               const double* v0, const double* kappa, const double* theta,
                                               const double* sigma, const double* rho,
                                               const int32_t* option_type, const int32_t* integration_steps,
                                               const double* integration_limit,
                                               int32_t n, double* out_prices);
QK_EXPORT int32_t qk_cf_merton_jump_diffusion_price_batch(const double* spot, const double* strike,
                                                           const double* t, const double* vol,
                                                           const double* r, const double* q,
                                                           const double* jump_intensity, const double* jump_mean,
                                                           const double* jump_vol, const int32_t* max_terms,
                                                           const int32_t* option_type,
                                                           int32_t n, double* out_prices);
QK_EXPORT int32_t qk_cf_variance_gamma_price_cf_batch(const double* spot, const double* strike,
                                                       const double* t, const double* r, const double* q,
                                                       const double* sigma, const double* theta, const double* nu,
                                                       const int32_t* option_type, const int32_t* integration_steps,
                                                       const double* integration_limit,
                                                       int32_t n, double* out_prices);
QK_EXPORT int32_t qk_cf_sabr_hagan_lognormal_iv_batch(const double* forward, const double* strike,
                                                       const double* t, const double* alpha,
                                                       const double* beta, const double* rho,
                                                       const double* nu,
                                                       int32_t n, double* out_prices);
QK_EXPORT int32_t qk_cf_sabr_hagan_black76_price_batch(const double* forward, const double* strike,
                                                        const double* t, const double* r,
                                                        const double* alpha, const double* beta,
                                                        const double* rho, const double* nu,
                                                        const int32_t* option_type,
                                                        int32_t n, double* out_prices);
QK_EXPORT int32_t qk_cf_dupire_local_vol_batch(const double* strike, const double* t,
                                                const double* call_price, const double* dC_dT,
                                                const double* dC_dK, const double* d2C_dK2,
                                                const double* r, const double* q,
                                                int32_t n, double* out_prices);
QK_EXPORT double qk_tlm_crr_price(double spot, double strike, double t, double vol,
                                  double r, double q, int32_t option_type,
                                  int32_t steps, int32_t american_style);
QK_EXPORT int32_t qk_tlm_crr_price_batch(const double* spot, const double* strike,
                                         const double* t, const double* vol,
                                         const double* r, const double* q,
                                         const int32_t* option_type,
                                         const int32_t* steps,
                                         const int32_t* american_style,
                                         int32_t n, double* out_prices);
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
QK_EXPORT double qk_tlm_derman_kani_call_surface_price(double spot, double strike, double t,
                                                        double r, double q, int32_t option_type,
                                                        const double* surface_strikes, int32_t n_strikes,
                                                        const double* surface_maturities, int32_t n_maturities,
                                                        const double* surface_call_prices,
                                                        int32_t steps, int32_t american_style);
QK_EXPORT int32_t qk_tlm_jarrow_rudd_price_batch(const double* spot, const double* strike,
                                                   const double* t, const double* vol,
                                                   const double* r, const double* q,
                                                   const int32_t* option_type, const int32_t* steps,
                                                   const int32_t* american_style,
                                                   int32_t n, double* out_prices);
QK_EXPORT int32_t qk_tlm_tian_price_batch(const double* spot, const double* strike,
                                           const double* t, const double* vol,
                                           const double* r, const double* q,
                                           const int32_t* option_type, const int32_t* steps,
                                           const int32_t* american_style,
                                           int32_t n, double* out_prices);
QK_EXPORT int32_t qk_tlm_leisen_reimer_price_batch(const double* spot, const double* strike,
                                                     const double* t, const double* vol,
                                                     const double* r, const double* q,
                                                     const int32_t* option_type, const int32_t* steps,
                                                     const int32_t* american_style,
                                                     int32_t n, double* out_prices);
QK_EXPORT int32_t qk_tlm_trinomial_tree_price_batch(const double* spot, const double* strike,
                                                      const double* t, const double* vol,
                                                      const double* r, const double* q,
                                                      const int32_t* option_type, const int32_t* steps,
                                                      const int32_t* american_style,
                                                      int32_t n, double* out_prices);
QK_EXPORT int32_t qk_tlm_derman_kani_const_local_vol_price_batch(const double* spot, const double* strike,
                                                                   const double* t, const double* local_vol,
                                                                   const double* r, const double* q,
                                                                   const int32_t* option_type, const int32_t* steps,
                                                                   const int32_t* american_style,
                                                                   int32_t n, double* out_prices);
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
QK_EXPORT int32_t qk_fdm_explicit_fd_price_batch(const double* spot, const double* strike,
                                                   const double* t, const double* vol,
                                                   const double* r, const double* q,
                                                   const int32_t* option_type,
                                                   const int32_t* time_steps, const int32_t* spot_steps,
                                                   const int32_t* american_style,
                                                   int32_t n, double* out_prices);
QK_EXPORT int32_t qk_fdm_implicit_fd_price_batch(const double* spot, const double* strike,
                                                   const double* t, const double* vol,
                                                   const double* r, const double* q,
                                                   const int32_t* option_type,
                                                   const int32_t* time_steps, const int32_t* spot_steps,
                                                   const int32_t* american_style,
                                                   int32_t n, double* out_prices);
QK_EXPORT int32_t qk_fdm_crank_nicolson_price_batch(const double* spot, const double* strike,
                                                      const double* t, const double* vol,
                                                      const double* r, const double* q,
                                                      const int32_t* option_type,
                                                      const int32_t* time_steps, const int32_t* spot_steps,
                                                      const int32_t* american_style,
                                                      int32_t n, double* out_prices);
QK_EXPORT int32_t qk_fdm_adi_douglas_price_batch(const double* spot, const double* strike,
                                                    const double* t, const double* r, const double* q,
                                                    const double* v0, const double* kappa,
                                                    const double* theta_v, const double* sigma,
                                                    const double* rho, const int32_t* option_type,
                                                    const int32_t* s_steps, const int32_t* v_steps,
                                                    const int32_t* time_steps,
                                                    int32_t n, double* out_prices);
QK_EXPORT int32_t qk_fdm_adi_craig_sneyd_price_batch(const double* spot, const double* strike,
                                                        const double* t, const double* r, const double* q,
                                                        const double* v0, const double* kappa,
                                                        const double* theta_v, const double* sigma,
                                                        const double* rho, const int32_t* option_type,
                                                        const int32_t* s_steps, const int32_t* v_steps,
                                                        const int32_t* time_steps,
                                                        int32_t n, double* out_prices);
QK_EXPORT int32_t qk_fdm_adi_hundsdorfer_verwer_price_batch(const double* spot, const double* strike,
                                                               const double* t, const double* r, const double* q,
                                                               const double* v0, const double* kappa,
                                                               const double* theta_v, const double* sigma,
                                                               const double* rho, const int32_t* option_type,
                                                               const int32_t* s_steps, const int32_t* v_steps,
                                                               const int32_t* time_steps,
                                                               int32_t n, double* out_prices);
QK_EXPORT int32_t qk_fdm_psor_price_batch(const double* spot, const double* strike,
                                            const double* t, const double* vol,
                                            const double* r, const double* q,
                                            const int32_t* option_type,
                                            const int32_t* time_steps, const int32_t* spot_steps,
                                            const double* omega, const double* tol,
                                            const int32_t* max_iter,
                                            int32_t n, double* out_prices);
QK_EXPORT double qk_mcm_standard_monte_carlo_price(double spot, double strike, double t, double vol,
                                                   double r, double q, int32_t option_type,
                                                   int32_t paths, uint64_t seed);
QK_EXPORT int32_t qk_mcm_standard_monte_carlo_price_batch(const double* spot, const double* strike,
                                                          const double* t, const double* vol,
                                                          const double* r, const double* q,
                                                          const int32_t* option_type,
                                                          const int32_t* paths,
                                                          const uint64_t* seed,
                                                          int32_t n, double* out_prices);
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
QK_EXPORT int32_t qk_mcm_euler_maruyama_price_batch(const double* spot, const double* strike,
                                                      const double* t, const double* vol,
                                                      const double* r, const double* q,
                                                      const int32_t* option_type, const int32_t* paths,
                                                      const int32_t* steps, const uint64_t* seed,
                                                      int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mcm_milstein_price_batch(const double* spot, const double* strike,
                                                const double* t, const double* vol,
                                                const double* r, const double* q,
                                                const int32_t* option_type, const int32_t* paths,
                                                const int32_t* steps, const uint64_t* seed,
                                                int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mcm_longstaff_schwartz_price_batch(const double* spot, const double* strike,
                                                          const double* t, const double* vol,
                                                          const double* r, const double* q,
                                                          const int32_t* option_type, const int32_t* paths,
                                                          const int32_t* steps, const uint64_t* seed,
                                                          int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mcm_quasi_monte_carlo_sobol_price_batch(const double* spot, const double* strike,
                                                               const double* t, const double* vol,
                                                               const double* r, const double* q,
                                                               const int32_t* option_type, const int32_t* paths,
                                                               int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mcm_quasi_monte_carlo_halton_price_batch(const double* spot, const double* strike,
                                                                const double* t, const double* vol,
                                                                const double* r, const double* q,
                                                                const int32_t* option_type, const int32_t* paths,
                                                                int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mcm_multilevel_monte_carlo_price_batch(const double* spot, const double* strike,
                                                              const double* t, const double* vol,
                                                              const double* r, const double* q,
                                                              const int32_t* option_type, const int32_t* base_paths,
                                                              const int32_t* levels, const int32_t* base_steps,
                                                              const uint64_t* seed,
                                                              int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mcm_importance_sampling_price_batch(const double* spot, const double* strike,
                                                          const double* t, const double* vol,
                                                          const double* r, const double* q,
                                                          const int32_t* option_type, const int32_t* paths,
                                                          const double* shift, const uint64_t* seed,
                                                          int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mcm_control_variates_price_batch(const double* spot, const double* strike,
                                                        const double* t, const double* vol,
                                                        const double* r, const double* q,
                                                        const int32_t* option_type, const int32_t* paths,
                                                        const uint64_t* seed,
                                                        int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mcm_antithetic_variates_price_batch(const double* spot, const double* strike,
                                                           const double* t, const double* vol,
                                                           const double* r, const double* q,
                                                           const int32_t* option_type, const int32_t* paths,
                                                           const uint64_t* seed,
                                                           int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mcm_stratified_sampling_price_batch(const double* spot, const double* strike,
                                                           const double* t, const double* vol,
                                                           const double* r, const double* q,
                                                           const int32_t* option_type, const int32_t* paths,
                                                           const uint64_t* seed,
                                                           int32_t n, double* out_prices);
QK_EXPORT double qk_ftm_carr_madan_fft_price(double spot, double strike, double t, double vol,
                                             double r, double q, int32_t option_type,
                                             int32_t grid_size, double eta, double alpha);
QK_EXPORT int32_t qk_ftm_carr_madan_fft_price_batch(const double* spot, const double* strike,
                                                    const double* t, const double* vol,
                                                    const double* r, const double* q,
                                                    const int32_t* option_type,
                                                    const int32_t* grid_size,
                                                    const double* eta,
                                                    const double* alpha,
                                                    int32_t n, double* out_prices);
QK_EXPORT double qk_ftm_cos_fang_oosterlee_price(double spot, double strike, double t, double vol,
                                                 double r, double q, int32_t option_type,
                                                 int32_t n_terms, double truncation_width);
QK_EXPORT double qk_ftm_fractional_fft_price(double spot, double strike, double t, double vol,
                                             double r, double q, int32_t option_type,
                                             int32_t grid_size, double eta, double lambda,
                                             double alpha);
QK_EXPORT double qk_ftm_lewis_fourier_inversion_price(double spot, double strike, double t, double vol,
                                                      double r, double q, int32_t option_type,
                                                      int32_t integration_steps,
                                                      double integration_limit);
QK_EXPORT double qk_ftm_hilbert_transform_price(double spot, double strike, double t, double vol,
                                                double r, double q, int32_t option_type,
                                                int32_t integration_steps,
                                                double integration_limit);
QK_EXPORT int32_t qk_ftm_cos_fang_oosterlee_price_batch(const double* spot, const double* strike,
                                                          const double* t, const double* vol,
                                                          const double* r, const double* q,
                                                          const int32_t* option_type, const int32_t* n_terms,
                                                          const double* truncation_width,
                                                          int32_t n, double* out_prices);
QK_EXPORT int32_t qk_ftm_fractional_fft_price_batch(const double* spot, const double* strike,
                                                      const double* t, const double* vol,
                                                      const double* r, const double* q,
                                                      const int32_t* option_type, const int32_t* grid_size,
                                                      const double* eta, const double* lambda_,
                                                      const double* alpha,
                                                      int32_t n, double* out_prices);
QK_EXPORT int32_t qk_ftm_lewis_fourier_inversion_price_batch(const double* spot, const double* strike,
                                                               const double* t, const double* vol,
                                                               const double* r, const double* q,
                                                               const int32_t* option_type,
                                                               const int32_t* integration_steps,
                                                               const double* integration_limit,
                                                               int32_t n, double* out_prices);
QK_EXPORT int32_t qk_ftm_hilbert_transform_price_batch(const double* spot, const double* strike,
                                                         const double* t, const double* vol,
                                                         const double* r, const double* q,
                                                         const int32_t* option_type,
                                                         const int32_t* integration_steps,
                                                         const double* integration_limit,
                                                         int32_t n, double* out_prices);
QK_EXPORT double qk_iqm_gauss_hermite_price(double spot, double strike, double t, double vol,
                                            double r, double q, int32_t option_type,
                                            int32_t n_points);
QK_EXPORT double qk_iqm_gauss_laguerre_price(double spot, double strike, double t, double vol,
                                             double r, double q, int32_t option_type,
                                             int32_t n_points);
QK_EXPORT double qk_iqm_gauss_legendre_price(double spot, double strike, double t, double vol,
                                             double r, double q, int32_t option_type,
                                             int32_t n_points, double integration_limit);
QK_EXPORT double qk_iqm_adaptive_quadrature_price(double spot, double strike, double t, double vol,
                                                  double r, double q, int32_t option_type,
                                                  double abs_tol, double rel_tol,
                                                  int32_t max_depth, double integration_limit);
QK_EXPORT int32_t qk_iqm_gauss_hermite_price_batch(const double* spot, const double* strike,
                                                      const double* t, const double* vol,
                                                      const double* r, const double* q,
                                                      const int32_t* option_type,
                                                      const int32_t* n_points,
                                                      int32_t n, double* out_prices);
QK_EXPORT int32_t qk_iqm_gauss_laguerre_price_batch(const double* spot, const double* strike,
                                                       const double* t, const double* vol,
                                                       const double* r, const double* q,
                                                       const int32_t* option_type,
                                                       const int32_t* n_points,
                                                       int32_t n, double* out_prices);
QK_EXPORT int32_t qk_iqm_gauss_legendre_price_batch(const double* spot, const double* strike,
                                                       const double* t, const double* vol,
                                                       const double* r, const double* q,
                                                       const int32_t* option_type,
                                                       const int32_t* n_points,
                                                       const double* integration_limit,
                                                       int32_t n, double* out_prices);
QK_EXPORT int32_t qk_iqm_adaptive_quadrature_price_batch(const double* spot, const double* strike,
                                                            const double* t, const double* vol,
                                                            const double* r, const double* q,
                                                            const int32_t* option_type,
                                                            const double* abs_tol, const double* rel_tol,
                                                            const int32_t* max_depth,
                                                            const double* integration_limit,
                                                            int32_t n, double* out_prices);
QK_EXPORT double qk_ram_polynomial_chaos_expansion_price(double spot, double strike, double t, double vol,
                                                         double r, double q, int32_t option_type,
                                                         int32_t polynomial_order,
                                                         int32_t quadrature_points);
QK_EXPORT double qk_ram_radial_basis_function_price(double spot, double strike, double t, double vol,
                                                    double r, double q, int32_t option_type,
                                                    int32_t centers, double rbf_shape, double ridge);
QK_EXPORT double qk_ram_sparse_grid_collocation_price(double spot, double strike, double t, double vol,
                                                      double r, double q, int32_t option_type,
                                                      int32_t level, int32_t nodes_per_dim);
QK_EXPORT double qk_ram_proper_orthogonal_decomposition_price(double spot, double strike, double t, double vol,
                                                              double r, double q, int32_t option_type,
                                                              int32_t modes, int32_t snapshots);
QK_EXPORT int32_t qk_ram_polynomial_chaos_expansion_price_batch(const double* spot, const double* strike,
                                                                  const double* t, const double* vol,
                                                                  const double* r, const double* q,
                                                                  const int32_t* option_type,
                                                                  const int32_t* polynomial_order,
                                                                  const int32_t* quadrature_points,
                                                                  int32_t n, double* out_prices);
QK_EXPORT int32_t qk_ram_radial_basis_function_price_batch(const double* spot, const double* strike,
                                                             const double* t, const double* vol,
                                                             const double* r, const double* q,
                                                             const int32_t* option_type,
                                                             const int32_t* centers, const double* rbf_shape,
                                                             const double* ridge,
                                                             int32_t n, double* out_prices);
QK_EXPORT int32_t qk_ram_sparse_grid_collocation_price_batch(const double* spot, const double* strike,
                                                                const double* t, const double* vol,
                                                                const double* r, const double* q,
                                                                const int32_t* option_type,
                                                                const int32_t* level, const int32_t* nodes_per_dim,
                                                                int32_t n, double* out_prices);
QK_EXPORT int32_t qk_ram_proper_orthogonal_decomposition_price_batch(const double* spot, const double* strike,
                                                                       const double* t, const double* vol,
                                                                       const double* r, const double* q,
                                                                       const int32_t* option_type,
                                                                       const int32_t* modes, const int32_t* snapshots,
                                                                       int32_t n, double* out_prices);
QK_EXPORT double qk_agm_pathwise_derivative_delta(double spot, double strike, double t, double vol,
                                                  double r, double q, int32_t option_type,
                                                  int32_t paths, uint64_t seed);
QK_EXPORT double qk_agm_likelihood_ratio_delta(double spot, double strike, double t, double vol,
                                               double r, double q, int32_t option_type,
                                               int32_t paths, uint64_t seed, double weight_clip);
QK_EXPORT double qk_agm_aad_delta(double spot, double strike, double t, double vol,
                                  double r, double q, int32_t option_type,
                                  int32_t tape_steps, double regularization);
QK_EXPORT int32_t qk_agm_pathwise_derivative_delta_batch(const double* spot, const double* strike,
                                                            const double* t, const double* vol,
                                                            const double* r, const double* q,
                                                            const int32_t* option_type,
                                                            const int32_t* paths, const uint64_t* seed,
                                                            int32_t n, double* out_prices);
QK_EXPORT int32_t qk_agm_likelihood_ratio_delta_batch(const double* spot, const double* strike,
                                                         const double* t, const double* vol,
                                                         const double* r, const double* q,
                                                         const int32_t* option_type,
                                                         const int32_t* paths, const uint64_t* seed,
                                                         const double* weight_clip,
                                                         int32_t n, double* out_prices);
QK_EXPORT int32_t qk_agm_aad_delta_batch(const double* spot, const double* strike,
                                            const double* t, const double* vol,
                                            const double* r, const double* q,
                                            const int32_t* option_type,
                                            const int32_t* tape_steps, const double* regularization,
                                            int32_t n, double* out_prices);
QK_EXPORT double qk_mlm_deep_bsde_price(double spot, double strike, double t, double vol,
                                        double r, double q, int32_t option_type,
                                        int32_t time_steps, int32_t hidden_width,
                                        int32_t training_epochs, double learning_rate);
QK_EXPORT double qk_mlm_pinns_price(double spot, double strike, double t, double vol,
                                    double r, double q, int32_t option_type,
                                    int32_t collocation_points, int32_t boundary_points,
                                    int32_t epochs, double loss_balance);
QK_EXPORT double qk_mlm_deep_hedging_price(double spot, double strike, double t, double vol,
                                           double r, double q, int32_t option_type,
                                           int32_t rehedge_steps, double risk_aversion,
                                           int32_t scenarios, uint64_t seed);
QK_EXPORT double qk_mlm_neural_sde_calibration_price(double spot, double strike, double t, double vol,
                                                     double r, double q, int32_t option_type,
                                                     double target_implied_vol,
                                                     int32_t calibration_steps,
                                                     double regularization);
QK_EXPORT int32_t qk_mlm_deep_bsde_price_batch(const double* spot, const double* strike,
                                                  const double* t, const double* vol,
                                                  const double* r, const double* q,
                                                  const int32_t* option_type,
                                                  const int32_t* time_steps, const int32_t* hidden_width,
                                                  const int32_t* training_epochs, const double* learning_rate,
                                                  int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mlm_pinns_price_batch(const double* spot, const double* strike,
                                              const double* t, const double* vol,
                                              const double* r, const double* q,
                                              const int32_t* option_type,
                                              const int32_t* collocation_points, const int32_t* boundary_points,
                                              const int32_t* epochs, const double* loss_balance,
                                              int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mlm_deep_hedging_price_batch(const double* spot, const double* strike,
                                                     const double* t, const double* vol,
                                                     const double* r, const double* q,
                                                     const int32_t* option_type,
                                                     const int32_t* rehedge_steps, const double* risk_aversion,
                                                     const int32_t* scenarios, const uint64_t* seed,
                                                     int32_t n, double* out_prices);
QK_EXPORT int32_t qk_mlm_neural_sde_calibration_price_batch(const double* spot, const double* strike,
                                                               const double* t, const double* vol,
                                                               const double* r, const double* q,
                                                               const int32_t* option_type,
                                                               const double* target_implied_vol,
                                                               const int32_t* calibration_steps,
                                                               const double* regularization,
                                                               int32_t n, double* out_prices);

#ifdef __cplusplus
}
#endif

#endif /* QK_API_H */
