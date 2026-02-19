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
