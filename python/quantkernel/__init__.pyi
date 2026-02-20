"""Type stubs for quantkernel package."""

import numpy as np
from numpy.typing import ArrayLike

QK_CALL: int
QK_PUT: int
QK_OK: int
QK_ERR_NULL_PTR: int
QK_ERR_BAD_SIZE: int
QK_ERR_ABI_MISMATCH: int
QK_ERR_RUNTIME_INIT: int
QK_ERR_INVALID_INPUT: int
ABI_MAJOR: int
ABI_MINOR: int

class QKError(RuntimeError): ...
class QKNullPointerError(QKError): ...
class QKBadSizeError(QKError): ...
class QKInvalidInputError(QKError): ...

class QuantKernel:
    CALL: int
    PUT: int

    def __init__(self) -> None: ...

    @property
    def native_batch_available(self) -> bool: ...

    def black_scholes_merton_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int,
    ) -> float: ...
    def black76_price(
        self, forward: float, strike: float, t: float, vol: float,
        r: float, option_type: int,
    ) -> float: ...
    def bachelier_price(
        self, forward: float, strike: float, t: float, normal_vol: float,
        r: float, option_type: int,
    ) -> float: ...
    def heston_price_cf(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta: float, sigma: float, rho: float,
        option_type: int, integration_steps: int = ..., integration_limit: float = ...,
    ) -> float: ...
    def merton_jump_diffusion_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        jump_intensity: float, jump_mean: float, jump_vol: float,
        max_terms: int, option_type: int,
    ) -> float: ...
    def variance_gamma_price_cf(
        self, spot: float, strike: float, t: float, r: float, q: float,
        sigma: float, theta: float, nu: float, option_type: int,
        integration_steps: int = ..., integration_limit: float = ...,
    ) -> float: ...
    def sabr_hagan_lognormal_iv(
        self, forward: float, strike: float, t: float,
        alpha: float, beta: float, rho: float, nu: float,
    ) -> float: ...
    def sabr_hagan_black76_price(
        self, forward: float, strike: float, t: float, r: float,
        alpha: float, beta: float, rho: float, nu: float, option_type: int,
    ) -> float: ...
    def dupire_local_vol(
        self, strike: float, t: float, call_price: float,
        dC_dT: float, dC_dK: float, d2C_dK2: float, r: float, q: float,
    ) -> float: ...

    def black_scholes_merton_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike, option_type: ArrayLike,
    ) -> np.ndarray: ...
    def black76_price_batch(
        self, forward: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, option_type: ArrayLike,
    ) -> np.ndarray: ...
    def bachelier_price_batch(
        self, forward: ArrayLike, strike: ArrayLike, t: ArrayLike,
        normal_vol: ArrayLike, r: ArrayLike, option_type: ArrayLike,
    ) -> np.ndarray: ...
    def heston_price_cf_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        r: ArrayLike, q: ArrayLike, v0: ArrayLike, kappa: ArrayLike,
        theta: ArrayLike, sigma: ArrayLike, rho: ArrayLike,
        option_type: ArrayLike, integration_steps: ArrayLike,
        integration_limit: ArrayLike,
    ) -> np.ndarray: ...
    def merton_jump_diffusion_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        jump_intensity: ArrayLike, jump_mean: ArrayLike,
        jump_vol: ArrayLike, max_terms: ArrayLike, option_type: ArrayLike,
    ) -> np.ndarray: ...
    def variance_gamma_price_cf_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        r: ArrayLike, q: ArrayLike, sigma: ArrayLike, theta: ArrayLike,
        nu: ArrayLike, option_type: ArrayLike,
        integration_steps: ArrayLike, integration_limit: ArrayLike,
    ) -> np.ndarray: ...
    def sabr_hagan_lognormal_iv_batch(
        self, forward: ArrayLike, strike: ArrayLike, t: ArrayLike,
        alpha: ArrayLike, beta: ArrayLike, rho: ArrayLike, nu: ArrayLike,
    ) -> np.ndarray: ...
    def sabr_hagan_black76_price_batch(
        self, forward: ArrayLike, strike: ArrayLike, t: ArrayLike,
        r: ArrayLike, alpha: ArrayLike, beta: ArrayLike,
        rho: ArrayLike, nu: ArrayLike, option_type: ArrayLike,
    ) -> np.ndarray: ...
    def dupire_local_vol_batch(
        self, strike: ArrayLike, t: ArrayLike, call_price: ArrayLike,
        dC_dT: ArrayLike, dC_dK: ArrayLike, d2C_dK2: ArrayLike,
        r: ArrayLike, q: ArrayLike,
    ) -> np.ndarray: ...

    def crr_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, steps: int,
        american_style: bool = ...,
    ) -> float: ...
    def crr_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, steps: ArrayLike, american_style: ArrayLike,
    ) -> np.ndarray: ...
    def jarrow_rudd_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, steps: int,
        american_style: bool = ...,
    ) -> float: ...
    def jarrow_rudd_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, steps: ArrayLike, american_style: ArrayLike,
    ) -> np.ndarray: ...
    def tian_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, steps: int,
        american_style: bool = ...,
    ) -> float: ...
    def tian_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, steps: ArrayLike, american_style: ArrayLike,
    ) -> np.ndarray: ...
    def leisen_reimer_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, steps: int,
        american_style: bool = ...,
    ) -> float: ...
    def leisen_reimer_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, steps: ArrayLike, american_style: ArrayLike,
    ) -> np.ndarray: ...
    def trinomial_tree_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, steps: int,
        american_style: bool = ...,
    ) -> float: ...
    def trinomial_tree_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, steps: ArrayLike, american_style: ArrayLike,
    ) -> np.ndarray: ...
    def derman_kani_const_local_vol_price(
        self, spot: float, strike: float, t: float, local_vol: float,
        r: float, q: float, option_type: int, steps: int,
        american_style: bool = ...,
    ) -> float: ...
    def derman_kani_const_local_vol_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        local_vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, steps: ArrayLike, american_style: ArrayLike,
    ) -> np.ndarray: ...

    def explicit_fd_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, time_steps: int,
        spot_steps: int, american_style: bool = ...,
    ) -> float: ...
    def implicit_fd_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, time_steps: int,
        spot_steps: int, american_style: bool = ...,
    ) -> float: ...
    def crank_nicolson_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, time_steps: int,
        spot_steps: int, american_style: bool = ...,
    ) -> float: ...
    def adi_douglas_price(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta_v: float, sigma: float, rho: float,
        option_type: int, s_steps: int = ..., v_steps: int = ...,
        time_steps: int = ...,
    ) -> float: ...
    def adi_craig_sneyd_price(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta_v: float, sigma: float, rho: float,
        option_type: int, s_steps: int = ..., v_steps: int = ...,
        time_steps: int = ...,
    ) -> float: ...
    def adi_hundsdorfer_verwer_price(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta_v: float, sigma: float, rho: float,
        option_type: int, s_steps: int = ..., v_steps: int = ...,
        time_steps: int = ...,
    ) -> float: ...
    def psor_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, time_steps: int,
        spot_steps: int, omega: float = ..., tol: float = ...,
        max_iter: int = ...,
    ) -> float: ...

    def standard_monte_carlo_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int, seed: int = ...,
    ) -> float: ...
    def standard_monte_carlo_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike, seed: ArrayLike,
    ) -> np.ndarray: ...
    def euler_maruyama_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int, steps: int,
        seed: int = ...,
    ) -> float: ...
    def euler_maruyama_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike, steps: ArrayLike,
        seed: ArrayLike,
    ) -> np.ndarray: ...
    def milstein_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int, steps: int,
        seed: int = ...,
    ) -> float: ...
    def milstein_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike, steps: ArrayLike,
        seed: ArrayLike,
    ) -> np.ndarray: ...
    def longstaff_schwartz_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int, steps: int,
        seed: int = ...,
    ) -> float: ...
    def longstaff_schwartz_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike, steps: ArrayLike,
        seed: ArrayLike,
    ) -> np.ndarray: ...
    def quasi_monte_carlo_sobol_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int,
    ) -> float: ...
    def quasi_monte_carlo_sobol_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike,
    ) -> np.ndarray: ...
    def quasi_monte_carlo_halton_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int,
    ) -> float: ...
    def quasi_monte_carlo_halton_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike,
    ) -> np.ndarray: ...
    def multilevel_monte_carlo_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, base_paths: int,
        levels: int, base_steps: int, seed: int = ...,
    ) -> float: ...
    def multilevel_monte_carlo_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, base_paths: ArrayLike,
        levels: ArrayLike, base_steps: ArrayLike, seed: ArrayLike,
    ) -> np.ndarray: ...
    def importance_sampling_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int,
        shift: float, seed: int = ...,
    ) -> float: ...
    def importance_sampling_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike, shift: ArrayLike,
        seed: ArrayLike,
    ) -> np.ndarray: ...
    def control_variates_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int, seed: int = ...,
    ) -> float: ...
    def control_variates_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike, seed: ArrayLike,
    ) -> np.ndarray: ...
    def antithetic_variates_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int, seed: int = ...,
    ) -> float: ...
    def antithetic_variates_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike, seed: ArrayLike,
    ) -> np.ndarray: ...
    def stratified_sampling_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int, seed: int = ...,
    ) -> float: ...
    def stratified_sampling_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike, seed: ArrayLike,
    ) -> np.ndarray: ...

    def carr_madan_fft_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, grid_size: int = ...,
        eta: float = ..., alpha: float = ...,
    ) -> float: ...
    def carr_madan_fft_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, grid_size: ArrayLike, eta: ArrayLike,
        alpha: ArrayLike,
    ) -> np.ndarray: ...
    def cos_method_fang_oosterlee_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, n_terms: int = ...,
        truncation_width: float = ...,
    ) -> float: ...
    def cos_method_fang_oosterlee_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, n_terms: ArrayLike,
        truncation_width: ArrayLike,
    ) -> np.ndarray: ...
    def fractional_fft_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, grid_size: int = ...,
        eta: float = ..., lambda_: float = ..., alpha: float = ...,
    ) -> float: ...
    def fractional_fft_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, grid_size: ArrayLike, eta: ArrayLike,
        lambda_: ArrayLike, alpha: ArrayLike,
    ) -> np.ndarray: ...
    def lewis_fourier_inversion_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int,
        integration_steps: int = ..., integration_limit: float = ...,
    ) -> float: ...
    def lewis_fourier_inversion_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, integration_steps: ArrayLike,
        integration_limit: ArrayLike,
    ) -> np.ndarray: ...
    def hilbert_transform_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int,
        integration_steps: int = ..., integration_limit: float = ...,
    ) -> float: ...
    def hilbert_transform_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, integration_steps: ArrayLike,
        integration_limit: ArrayLike,
    ) -> np.ndarray: ...

    def gauss_hermite_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, n_points: int = ...,
    ) -> float: ...
    def gauss_laguerre_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, n_points: int = ...,
    ) -> float: ...
    def gauss_legendre_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, n_points: int = ...,
        integration_limit: float = ...,
    ) -> float: ...
    def adaptive_quadrature_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, abs_tol: float = ...,
        rel_tol: float = ..., max_depth: int = ...,
        integration_limit: float = ...,
    ) -> float: ...

    def polynomial_chaos_expansion_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int,
        polynomial_order: int = ..., quadrature_points: int = ...,
    ) -> float: ...
    def radial_basis_function_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, centers: int = ...,
        rbf_shape: float = ..., ridge: float = ...,
    ) -> float: ...
    def sparse_grid_collocation_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, level: int = ...,
        nodes_per_dim: int = ...,
    ) -> float: ...
    def proper_orthogonal_decomposition_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, modes: int = ...,
        snapshots: int = ...,
    ) -> float: ...

    def pathwise_derivative_delta(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int = ...,
        seed: int = ...,
    ) -> float: ...
    def likelihood_ratio_delta(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, paths: int = ...,
        seed: int = ..., weight_clip: float = ...,
    ) -> float: ...
    def aad_delta(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, tape_steps: int = ...,
        regularization: float = ...,
    ) -> float: ...

    def deep_bsde_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, time_steps: int = ...,
        hidden_width: int = ..., training_epochs: int = ...,
        learning_rate: float = ...,
    ) -> float: ...
    def pinns_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int,
        collocation_points: int = ..., boundary_points: int = ...,
        epochs: int = ..., loss_balance: float = ...,
    ) -> float: ...
    def deep_hedging_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, rehedge_steps: int = ...,
        risk_aversion: float = ..., scenarios: int = ..., seed: int = ...,
    ) -> float: ...
    def neural_sde_calibration_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int,
        target_implied_vol: float = ..., calibration_steps: int = ...,
        regularization: float = ...,
    ) -> float: ...

    def explicit_fd_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, time_steps: ArrayLike,
        spot_steps: ArrayLike, american_style: ArrayLike,
    ) -> np.ndarray: ...
    def implicit_fd_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, time_steps: ArrayLike,
        spot_steps: ArrayLike, american_style: ArrayLike,
    ) -> np.ndarray: ...
    def crank_nicolson_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, time_steps: ArrayLike,
        spot_steps: ArrayLike, american_style: ArrayLike,
    ) -> np.ndarray: ...
    def adi_douglas_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        r: ArrayLike, q: ArrayLike, v0: ArrayLike, kappa: ArrayLike,
        theta_v: ArrayLike, sigma: ArrayLike, rho: ArrayLike,
        option_type: ArrayLike, s_steps: ArrayLike,
        v_steps: ArrayLike, time_steps: ArrayLike,
    ) -> np.ndarray: ...
    def adi_craig_sneyd_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        r: ArrayLike, q: ArrayLike, v0: ArrayLike, kappa: ArrayLike,
        theta_v: ArrayLike, sigma: ArrayLike, rho: ArrayLike,
        option_type: ArrayLike, s_steps: ArrayLike,
        v_steps: ArrayLike, time_steps: ArrayLike,
    ) -> np.ndarray: ...
    def adi_hundsdorfer_verwer_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        r: ArrayLike, q: ArrayLike, v0: ArrayLike, kappa: ArrayLike,
        theta_v: ArrayLike, sigma: ArrayLike, rho: ArrayLike,
        option_type: ArrayLike, s_steps: ArrayLike,
        v_steps: ArrayLike, time_steps: ArrayLike,
    ) -> np.ndarray: ...
    def psor_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, time_steps: ArrayLike,
        spot_steps: ArrayLike, omega: ArrayLike, tol: ArrayLike,
        max_iter: ArrayLike,
    ) -> np.ndarray: ...

    def gauss_hermite_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, n_points: ArrayLike,
    ) -> np.ndarray: ...
    def gauss_laguerre_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, n_points: ArrayLike,
    ) -> np.ndarray: ...
    def gauss_legendre_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, n_points: ArrayLike,
        integration_limit: ArrayLike,
    ) -> np.ndarray: ...
    def adaptive_quadrature_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, abs_tol: ArrayLike,
        rel_tol: ArrayLike, max_depth: ArrayLike,
        integration_limit: ArrayLike,
    ) -> np.ndarray: ...

    def polynomial_chaos_expansion_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, polynomial_order: ArrayLike,
        quadrature_points: ArrayLike,
    ) -> np.ndarray: ...
    def radial_basis_function_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, centers: ArrayLike,
        rbf_shape: ArrayLike, ridge: ArrayLike,
    ) -> np.ndarray: ...
    def sparse_grid_collocation_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, level: ArrayLike,
        nodes_per_dim: ArrayLike,
    ) -> np.ndarray: ...
    def proper_orthogonal_decomposition_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, modes: ArrayLike,
        snapshots: ArrayLike,
    ) -> np.ndarray: ...

    def pathwise_derivative_delta_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike, seed: ArrayLike,
    ) -> np.ndarray: ...
    def likelihood_ratio_delta_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, paths: ArrayLike, seed: ArrayLike,
        weight_clip: ArrayLike,
    ) -> np.ndarray: ...
    def aad_delta_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, tape_steps: ArrayLike,
        regularization: ArrayLike,
    ) -> np.ndarray: ...

    def deep_bsde_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, time_steps: ArrayLike,
        hidden_width: ArrayLike, training_epochs: ArrayLike,
        learning_rate: ArrayLike,
    ) -> np.ndarray: ...
    def pinns_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, collocation_points: ArrayLike,
        boundary_points: ArrayLike, epochs: ArrayLike,
        loss_balance: ArrayLike,
    ) -> np.ndarray: ...
    def deep_hedging_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, rehedge_steps: ArrayLike,
        risk_aversion: ArrayLike, scenarios: ArrayLike,
        seed: ArrayLike,
    ) -> np.ndarray: ...
    def neural_sde_calibration_price_batch(
        self, spot: ArrayLike, strike: ArrayLike, t: ArrayLike,
        vol: ArrayLike, r: ArrayLike, q: ArrayLike,
        option_type: ArrayLike, target_implied_vol: ArrayLike,
        calibration_steps: ArrayLike, regularization: ArrayLike,
    ) -> np.ndarray: ...

    def get_accelerator(
        self, backend: str = ..., max_workers: int | None = ...,
    ) -> "QuantAccelerator": ...
    def price_batch(
        self, method: str, jobs: object, backend: str = ...,
        max_workers: int | None = ...,
    ) -> object: ...

class QuantAccelerator:
    def __init__(
        self, qk: QuantKernel, backend: str = ...,
        max_workers: int | None = ...,
    ) -> None: ...
    def price_batch(self, method: str, jobs: object) -> object: ...
