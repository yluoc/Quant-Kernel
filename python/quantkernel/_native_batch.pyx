# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp

from libc.stdint cimport int32_t, uint64_t

cdef extern from "quantkernel/qk_api.h":
    int32_t qk_cf_black_scholes_merton_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_cf_black76_price_batch(
        const double* forward, const double* strike, const double* t, const double* vol,
        const double* r, const int32_t* option_type, int32_t n, double* out_prices
    ) nogil
    int32_t qk_cf_bachelier_price_batch(
        const double* forward, const double* strike, const double* t, const double* normal_vol,
        const double* r, const int32_t* option_type, int32_t n, double* out_prices
    ) nogil
    int32_t qk_cf_heston_price_cf_batch(
        const double* spot, const double* strike, const double* t, const double* r, const double* q,
        const double* v0, const double* kappa, const double* theta, const double* sigma, const double* rho,
        const int32_t* option_type, const int32_t* integration_steps, const double* integration_limit,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_cf_merton_jump_diffusion_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const double* jump_intensity, const double* jump_mean,
        const double* jump_vol, const int32_t* max_terms, const int32_t* option_type,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_cf_variance_gamma_price_cf_batch(
        const double* spot, const double* strike, const double* t, const double* r, const double* q,
        const double* sigma, const double* theta, const double* nu,
        const int32_t* option_type, const int32_t* integration_steps, const double* integration_limit,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_cf_sabr_hagan_lognormal_iv_batch(
        const double* forward, const double* strike, const double* t,
        const double* alpha, const double* beta, const double* rho, const double* nu,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_cf_sabr_hagan_black76_price_batch(
        const double* forward, const double* strike, const double* t, const double* r,
        const double* alpha, const double* beta, const double* rho, const double* nu,
        const int32_t* option_type, int32_t n, double* out_prices
    ) nogil
    int32_t qk_cf_dupire_local_vol_batch(
        const double* strike, const double* t, const double* call_price,
        const double* dC_dT, const double* dC_dK, const double* d2C_dK2,
        const double* r, const double* q, int32_t n, double* out_prices
    ) nogil
    int32_t qk_tlm_crr_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* steps,
        const int32_t* american_style, int32_t n, double* out_prices
    ) nogil
    int32_t qk_tlm_jarrow_rudd_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* steps,
        const int32_t* american_style, int32_t n, double* out_prices
    ) nogil
    int32_t qk_tlm_tian_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* steps,
        const int32_t* american_style, int32_t n, double* out_prices
    ) nogil
    int32_t qk_tlm_leisen_reimer_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* steps,
        const int32_t* american_style, int32_t n, double* out_prices
    ) nogil
    int32_t qk_tlm_trinomial_tree_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* steps,
        const int32_t* american_style, int32_t n, double* out_prices
    ) nogil
    int32_t qk_tlm_derman_kani_const_local_vol_price_batch(
        const double* spot, const double* strike, const double* t, const double* local_vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* steps,
        const int32_t* american_style, int32_t n, double* out_prices
    ) nogil
    int32_t qk_mcm_standard_monte_carlo_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* paths,
        const uint64_t* seed, int32_t n, double* out_prices
    ) nogil
    int32_t qk_mcm_euler_maruyama_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* paths,
        const int32_t* steps, const uint64_t* seed, int32_t n, double* out_prices
    ) nogil
    int32_t qk_mcm_milstein_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* paths,
        const int32_t* steps, const uint64_t* seed, int32_t n, double* out_prices
    ) nogil
    int32_t qk_mcm_longstaff_schwartz_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* paths,
        const int32_t* steps, const uint64_t* seed, int32_t n, double* out_prices
    ) nogil
    int32_t qk_mcm_quasi_monte_carlo_sobol_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* paths,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_mcm_quasi_monte_carlo_halton_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* paths,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_mcm_multilevel_monte_carlo_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* base_paths,
        const int32_t* levels, const int32_t* base_steps, const uint64_t* seed,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_mcm_importance_sampling_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* paths,
        const double* shift, const uint64_t* seed, int32_t n, double* out_prices
    ) nogil
    int32_t qk_mcm_control_variates_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* paths,
        const uint64_t* seed, int32_t n, double* out_prices
    ) nogil
    int32_t qk_mcm_antithetic_variates_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* paths,
        const uint64_t* seed, int32_t n, double* out_prices
    ) nogil
    int32_t qk_mcm_stratified_sampling_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* paths,
        const uint64_t* seed, int32_t n, double* out_prices
    ) nogil
    int32_t qk_ftm_carr_madan_fft_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* grid_size,
        const double* eta, const double* alpha, int32_t n, double* out_prices
    ) nogil
    int32_t qk_ftm_cos_fang_oosterlee_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* n_terms,
        const double* truncation_width, int32_t n, double* out_prices
    ) nogil
    int32_t qk_ftm_fractional_fft_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type, const int32_t* grid_size,
        const double* eta, const double* lambda_, const double* alpha,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_ftm_lewis_fourier_inversion_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* integration_steps, const double* integration_limit,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_ftm_hilbert_transform_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* integration_steps, const double* integration_limit,
        int32_t n, double* out_prices
    ) nogil
    # --- FDM batch ---
    int32_t qk_fdm_explicit_fd_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* time_steps, const int32_t* spot_steps, const int32_t* american_style,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_fdm_implicit_fd_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* time_steps, const int32_t* spot_steps, const int32_t* american_style,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_fdm_crank_nicolson_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* time_steps, const int32_t* spot_steps, const int32_t* american_style,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_fdm_adi_douglas_price_batch(
        const double* spot, const double* strike, const double* t, const double* r, const double* q,
        const double* v0, const double* kappa, const double* theta_v, const double* sigma,
        const double* rho, const int32_t* option_type,
        const int32_t* s_steps, const int32_t* v_steps, const int32_t* time_steps,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_fdm_adi_craig_sneyd_price_batch(
        const double* spot, const double* strike, const double* t, const double* r, const double* q,
        const double* v0, const double* kappa, const double* theta_v, const double* sigma,
        const double* rho, const int32_t* option_type,
        const int32_t* s_steps, const int32_t* v_steps, const int32_t* time_steps,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_fdm_adi_hundsdorfer_verwer_price_batch(
        const double* spot, const double* strike, const double* t, const double* r, const double* q,
        const double* v0, const double* kappa, const double* theta_v, const double* sigma,
        const double* rho, const int32_t* option_type,
        const int32_t* s_steps, const int32_t* v_steps, const int32_t* time_steps,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_fdm_psor_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* time_steps, const int32_t* spot_steps,
        const double* omega, const double* tol, const int32_t* max_iter,
        int32_t n, double* out_prices
    ) nogil
    # --- IQM batch ---
    int32_t qk_iqm_gauss_hermite_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* n_points,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_iqm_gauss_laguerre_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* n_points,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_iqm_gauss_legendre_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* n_points, const double* integration_limit,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_iqm_adaptive_quadrature_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const double* abs_tol, const double* rel_tol, const int32_t* max_depth,
        const double* integration_limit,
        int32_t n, double* out_prices
    ) nogil
    # --- RAM batch ---
    int32_t qk_ram_polynomial_chaos_expansion_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* polynomial_order, const int32_t* quadrature_points,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_ram_radial_basis_function_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* centers, const double* rbf_shape, const double* ridge,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_ram_sparse_grid_collocation_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* level, const int32_t* nodes_per_dim,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_ram_proper_orthogonal_decomposition_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* modes, const int32_t* snapshots,
        int32_t n, double* out_prices
    ) nogil
    # --- AGM batch ---
    int32_t qk_agm_pathwise_derivative_delta_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* paths, const uint64_t* seed,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_agm_likelihood_ratio_delta_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* paths, const uint64_t* seed, const double* weight_clip,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_agm_aad_delta_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* tape_steps, const double* regularization,
        int32_t n, double* out_prices
    ) nogil
    # --- MLM batch ---
    int32_t qk_mlm_deep_bsde_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* time_steps, const int32_t* hidden_width,
        const int32_t* training_epochs, const double* learning_rate,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_mlm_pinns_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* collocation_points, const int32_t* boundary_points,
        const int32_t* epochs, const double* loss_balance,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_mlm_deep_hedging_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const int32_t* rehedge_steps, const double* risk_aversion,
        const int32_t* scenarios, const uint64_t* seed,
        int32_t n, double* out_prices
    ) nogil
    int32_t qk_mlm_neural_sde_calibration_price_batch(
        const double* spot, const double* strike, const double* t, const double* vol,
        const double* r, const double* q, const int32_t* option_type,
        const double* target_implied_vol, const int32_t* calibration_steps,
        const double* regularization,
        int32_t n, double* out_prices
    ) nogil


def black_scholes_merton_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
):
    cdef Py_ssize_t n = spot.shape[0]
    if strike.shape[0] != n or t.shape[0] != n or vol.shape[0] != n or r.shape[0] != n or q.shape[0] != n or option_type.shape[0] != n:
        raise ValueError("All input arrays must have the same length")
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_cf_black_scholes_merton_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_cf_black_scholes_merton_price_batch failed with error code {rc}")
    return out


def black76_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] forward,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
):
    cdef Py_ssize_t n = forward.shape[0]
    if strike.shape[0] != n or t.shape[0] != n or vol.shape[0] != n or r.shape[0] != n or option_type.shape[0] != n:
        raise ValueError("All input arrays must have the same length")
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_cf_black76_price_batch(
            <const double*>forward.data, <const double*>strike.data, <const double*>t.data,
            <const double*>vol.data, <const double*>r.data, <const int32_t*>option_type.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_cf_black76_price_batch failed with error code {rc}")
    return out


def bachelier_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] forward,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] normal_vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
):
    cdef Py_ssize_t n = forward.shape[0]
    if strike.shape[0] != n or t.shape[0] != n or normal_vol.shape[0] != n or r.shape[0] != n or option_type.shape[0] != n:
        raise ValueError("All input arrays must have the same length")
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_cf_bachelier_price_batch(
            <const double*>forward.data, <const double*>strike.data, <const double*>t.data,
            <const double*>normal_vol.data, <const double*>r.data, <const int32_t*>option_type.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_cf_bachelier_price_batch failed with error code {rc}")
    return out


def heston_price_cf_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] v0,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] kappa,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sigma,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rho,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] integration_steps,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] integration_limit,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_cf_heston_price_cf_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data,
            <const double*>r.data, <const double*>q.data,
            <const double*>v0.data, <const double*>kappa.data, <const double*>theta.data,
            <const double*>sigma.data, <const double*>rho.data,
            <const int32_t*>option_type.data, <const int32_t*>integration_steps.data,
            <const double*>integration_limit.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_cf_heston_price_cf_batch failed with error code {rc}")
    return out


def merton_jump_diffusion_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] jump_intensity,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] jump_mean,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] jump_vol,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] max_terms,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_cf_merton_jump_diffusion_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data,
            <const double*>jump_intensity.data, <const double*>jump_mean.data, <const double*>jump_vol.data,
            <const int32_t*>max_terms.data, <const int32_t*>option_type.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_cf_merton_jump_diffusion_price_batch failed with error code {rc}")
    return out


def variance_gamma_price_cf_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sigma,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] nu,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] integration_steps,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] integration_limit,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_cf_variance_gamma_price_cf_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data,
            <const double*>r.data, <const double*>q.data,
            <const double*>sigma.data, <const double*>theta.data, <const double*>nu.data,
            <const int32_t*>option_type.data, <const int32_t*>integration_steps.data,
            <const double*>integration_limit.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_cf_variance_gamma_price_cf_batch failed with error code {rc}")
    return out


def sabr_hagan_lognormal_iv_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] forward,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] alpha,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] beta,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rho,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] nu,
):
    cdef Py_ssize_t n = forward.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_cf_sabr_hagan_lognormal_iv_batch(
            <const double*>forward.data, <const double*>strike.data, <const double*>t.data,
            <const double*>alpha.data, <const double*>beta.data,
            <const double*>rho.data, <const double*>nu.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_cf_sabr_hagan_lognormal_iv_batch failed with error code {rc}")
    return out


def sabr_hagan_black76_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] forward,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] alpha,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] beta,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rho,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] nu,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
):
    cdef Py_ssize_t n = forward.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_cf_sabr_hagan_black76_price_batch(
            <const double*>forward.data, <const double*>strike.data, <const double*>t.data,
            <const double*>r.data, <const double*>alpha.data, <const double*>beta.data,
            <const double*>rho.data, <const double*>nu.data, <const int32_t*>option_type.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_cf_sabr_hagan_black76_price_batch failed with error code {rc}")
    return out


def dupire_local_vol_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] call_price,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] dC_dT,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] dC_dK,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] d2C_dK2,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
):
    cdef Py_ssize_t n = strike.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_cf_dupire_local_vol_batch(
            <const double*>strike.data, <const double*>t.data, <const double*>call_price.data,
            <const double*>dC_dT.data, <const double*>dC_dK.data, <const double*>d2C_dK2.data,
            <const double*>r.data, <const double*>q.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_cf_dupire_local_vol_batch failed with error code {rc}")
    return out


def crr_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] american_style,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_tlm_crr_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>steps.data, <const int32_t*>american_style.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_tlm_crr_price_batch failed with error code {rc}")
    return out


def jarrow_rudd_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] american_style,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_tlm_jarrow_rudd_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>steps.data, <const int32_t*>american_style.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_tlm_jarrow_rudd_price_batch failed with error code {rc}")
    return out


def tian_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] american_style,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_tlm_tian_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>steps.data, <const int32_t*>american_style.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_tlm_tian_price_batch failed with error code {rc}")
    return out


def leisen_reimer_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] american_style,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_tlm_leisen_reimer_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>steps.data, <const int32_t*>american_style.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_tlm_leisen_reimer_price_batch failed with error code {rc}")
    return out


def trinomial_tree_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] american_style,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_tlm_trinomial_tree_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>steps.data, <const int32_t*>american_style.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_tlm_trinomial_tree_price_batch failed with error code {rc}")
    return out


def derman_kani_const_local_vol_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] local_vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] american_style,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_tlm_derman_kani_const_local_vol_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>local_vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>steps.data, <const int32_t*>american_style.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_tlm_derman_kani_const_local_vol_price_batch failed with error code {rc}")
    return out


def standard_monte_carlo_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mcm_standard_monte_carlo_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <const uint64_t*>seed.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mcm_standard_monte_carlo_price_batch failed with error code {rc}")
    return out


def euler_maruyama_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] steps,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mcm_euler_maruyama_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <const int32_t*>steps.data, <const uint64_t*>seed.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mcm_euler_maruyama_price_batch failed with error code {rc}")
    return out


def milstein_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] steps,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mcm_milstein_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <const int32_t*>steps.data, <const uint64_t*>seed.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mcm_milstein_price_batch failed with error code {rc}")
    return out


def longstaff_schwartz_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] steps,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mcm_longstaff_schwartz_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <const int32_t*>steps.data, <const uint64_t*>seed.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mcm_longstaff_schwartz_price_batch failed with error code {rc}")
    return out


def quasi_monte_carlo_sobol_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mcm_quasi_monte_carlo_sobol_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mcm_quasi_monte_carlo_sobol_price_batch failed with error code {rc}")
    return out


def quasi_monte_carlo_halton_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mcm_quasi_monte_carlo_halton_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mcm_quasi_monte_carlo_halton_price_batch failed with error code {rc}")
    return out


def multilevel_monte_carlo_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] base_paths,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] levels,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] base_steps,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mcm_multilevel_monte_carlo_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>base_paths.data, <const int32_t*>levels.data, <const int32_t*>base_steps.data,
            <const uint64_t*>seed.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mcm_multilevel_monte_carlo_price_batch failed with error code {rc}")
    return out


def importance_sampling_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] shift,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mcm_importance_sampling_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <const double*>shift.data, <const uint64_t*>seed.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mcm_importance_sampling_price_batch failed with error code {rc}")
    return out


def control_variates_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mcm_control_variates_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <const uint64_t*>seed.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mcm_control_variates_price_batch failed with error code {rc}")
    return out


def antithetic_variates_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mcm_antithetic_variates_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <const uint64_t*>seed.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mcm_antithetic_variates_price_batch failed with error code {rc}")
    return out


def stratified_sampling_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mcm_stratified_sampling_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <const uint64_t*>seed.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mcm_stratified_sampling_price_batch failed with error code {rc}")
    return out


def carr_madan_fft_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] grid_size,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] eta,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] alpha,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_ftm_carr_madan_fft_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>grid_size.data, <const double*>eta.data, <const double*>alpha.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_ftm_carr_madan_fft_price_batch failed with error code {rc}")
    return out


def cos_fang_oosterlee_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] n_terms,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] truncation_width,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_ftm_cos_fang_oosterlee_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>n_terms.data, <const double*>truncation_width.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_ftm_cos_fang_oosterlee_price_batch failed with error code {rc}")
    return out


def fractional_fft_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] grid_size,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] eta,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] lambda_,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] alpha,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_ftm_fractional_fft_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>grid_size.data, <const double*>eta.data, <const double*>lambda_.data,
            <const double*>alpha.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_ftm_fractional_fft_price_batch failed with error code {rc}")
    return out


def lewis_fourier_inversion_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] integration_steps,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] integration_limit,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_ftm_lewis_fourier_inversion_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>integration_steps.data, <const double*>integration_limit.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_ftm_lewis_fourier_inversion_price_batch failed with error code {rc}")
    return out


def hilbert_transform_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] integration_steps,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] integration_limit,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_ftm_hilbert_transform_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>integration_steps.data, <const double*>integration_limit.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_ftm_hilbert_transform_price_batch failed with error code {rc}")
    return out


def explicit_fd_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] time_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] spot_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] american_style,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_fdm_explicit_fd_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>time_steps.data, <const int32_t*>spot_steps.data, <const int32_t*>american_style.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_fdm_explicit_fd_price_batch failed with error code {rc}")
    return out


def implicit_fd_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] time_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] spot_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] american_style,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_fdm_implicit_fd_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>time_steps.data, <const int32_t*>spot_steps.data, <const int32_t*>american_style.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_fdm_implicit_fd_price_batch failed with error code {rc}")
    return out


def crank_nicolson_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] time_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] spot_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] american_style,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_fdm_crank_nicolson_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>time_steps.data, <const int32_t*>spot_steps.data, <const int32_t*>american_style.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_fdm_crank_nicolson_price_batch failed with error code {rc}")
    return out


def adi_douglas_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] v0,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] kappa,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta_v,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sigma,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rho,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] s_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] v_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] time_steps,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_fdm_adi_douglas_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data,
            <const double*>r.data, <const double*>q.data,
            <const double*>v0.data, <const double*>kappa.data, <const double*>theta_v.data,
            <const double*>sigma.data, <const double*>rho.data, <const int32_t*>option_type.data,
            <const int32_t*>s_steps.data, <const int32_t*>v_steps.data, <const int32_t*>time_steps.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_fdm_adi_douglas_price_batch failed with error code {rc}")
    return out


def adi_craig_sneyd_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] v0,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] kappa,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta_v,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sigma,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rho,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] s_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] v_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] time_steps,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_fdm_adi_craig_sneyd_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data,
            <const double*>r.data, <const double*>q.data,
            <const double*>v0.data, <const double*>kappa.data, <const double*>theta_v.data,
            <const double*>sigma.data, <const double*>rho.data, <const int32_t*>option_type.data,
            <const int32_t*>s_steps.data, <const int32_t*>v_steps.data, <const int32_t*>time_steps.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_fdm_adi_craig_sneyd_price_batch failed with error code {rc}")
    return out


def adi_hundsdorfer_verwer_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] v0,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] kappa,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] theta_v,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sigma,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rho,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] s_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] v_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] time_steps,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_fdm_adi_hundsdorfer_verwer_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data,
            <const double*>r.data, <const double*>q.data,
            <const double*>v0.data, <const double*>kappa.data, <const double*>theta_v.data,
            <const double*>sigma.data, <const double*>rho.data, <const int32_t*>option_type.data,
            <const int32_t*>s_steps.data, <const int32_t*>v_steps.data, <const int32_t*>time_steps.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_fdm_adi_hundsdorfer_verwer_price_batch failed with error code {rc}")
    return out


def psor_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] time_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] spot_steps,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] omega,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] tol,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] max_iter,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_fdm_psor_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>time_steps.data, <const int32_t*>spot_steps.data,
            <const double*>omega.data, <const double*>tol.data, <const int32_t*>max_iter.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_fdm_psor_price_batch failed with error code {rc}")
    return out


def gauss_hermite_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] n_points,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_iqm_gauss_hermite_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>n_points.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_iqm_gauss_hermite_price_batch failed with error code {rc}")
    return out


def gauss_laguerre_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] n_points,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_iqm_gauss_laguerre_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>n_points.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_iqm_gauss_laguerre_price_batch failed with error code {rc}")
    return out


def gauss_legendre_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] n_points,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] integration_limit,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_iqm_gauss_legendre_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>n_points.data, <const double*>integration_limit.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_iqm_gauss_legendre_price_batch failed with error code {rc}")
    return out


def adaptive_quadrature_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] abs_tol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rel_tol,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] max_depth,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] integration_limit,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_iqm_adaptive_quadrature_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const double*>abs_tol.data, <const double*>rel_tol.data, <const int32_t*>max_depth.data,
            <const double*>integration_limit.data, <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_iqm_adaptive_quadrature_price_batch failed with error code {rc}")
    return out


def polynomial_chaos_expansion_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] polynomial_order,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] quadrature_points,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_ram_polynomial_chaos_expansion_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>polynomial_order.data, <const int32_t*>quadrature_points.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_ram_polynomial_chaos_expansion_price_batch failed with error code {rc}")
    return out


def radial_basis_function_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] centers,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rbf_shape,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] ridge,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_ram_radial_basis_function_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>centers.data, <const double*>rbf_shape.data, <const double*>ridge.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_ram_radial_basis_function_price_batch failed with error code {rc}")
    return out


def sparse_grid_collocation_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] level,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] nodes_per_dim,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_ram_sparse_grid_collocation_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>level.data, <const int32_t*>nodes_per_dim.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_ram_sparse_grid_collocation_price_batch failed with error code {rc}")
    return out


def proper_orthogonal_decomposition_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] modes,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] snapshots,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_ram_proper_orthogonal_decomposition_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>modes.data, <const int32_t*>snapshots.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_ram_proper_orthogonal_decomposition_price_batch failed with error code {rc}")
    return out


def pathwise_derivative_delta_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_agm_pathwise_derivative_delta_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <const uint64_t*>seed.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_agm_pathwise_derivative_delta_batch failed with error code {rc}")
    return out


def likelihood_ratio_delta_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] paths,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] weight_clip,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_agm_likelihood_ratio_delta_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>paths.data, <const uint64_t*>seed.data, <const double*>weight_clip.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_agm_likelihood_ratio_delta_batch failed with error code {rc}")
    return out


def aad_delta_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] tape_steps,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] regularization,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_agm_aad_delta_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>tape_steps.data, <const double*>regularization.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_agm_aad_delta_batch failed with error code {rc}")
    return out


def deep_bsde_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] time_steps,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] hidden_width,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] training_epochs,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] learning_rate,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mlm_deep_bsde_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>time_steps.data, <const int32_t*>hidden_width.data,
            <const int32_t*>training_epochs.data, <const double*>learning_rate.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mlm_deep_bsde_price_batch failed with error code {rc}")
    return out


def pinns_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] collocation_points,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] boundary_points,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] epochs,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] loss_balance,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mlm_pinns_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>collocation_points.data, <const int32_t*>boundary_points.data,
            <const int32_t*>epochs.data, <const double*>loss_balance.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mlm_pinns_price_batch failed with error code {rc}")
    return out


def deep_hedging_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] rehedge_steps,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] risk_aversion,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] scenarios,
    cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] seed,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mlm_deep_hedging_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const int32_t*>rehedge_steps.data, <const double*>risk_aversion.data,
            <const int32_t*>scenarios.data, <const uint64_t*>seed.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mlm_deep_hedging_price_batch failed with error code {rc}")
    return out


def neural_sde_calibration_price_batch(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] spot,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] strike,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] t,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vol,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] option_type,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] target_implied_vol,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] calibration_steps,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] regularization,
):
    cdef Py_ssize_t n = spot.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
    cdef int32_t rc
    with nogil:
        rc = qk_mlm_neural_sde_calibration_price_batch(
            <const double*>spot.data, <const double*>strike.data, <const double*>t.data, <const double*>vol.data,
            <const double*>r.data, <const double*>q.data, <const int32_t*>option_type.data,
            <const double*>target_implied_vol.data, <const int32_t*>calibration_steps.data,
            <const double*>regularization.data,
            <int32_t>n, <double*>out.data
        )
    if rc != 0:
        raise ValueError(f"qk_mlm_neural_sde_calibration_price_batch failed with error code {rc}")
    return out
