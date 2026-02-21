"""Rule-based batch acceleration helpers for QuantKernel.

This module operationalizes practical acceleration rules:
- Use vectorized math for light closed-form formulas.
- Use optional CuPy for GPU vectorization when available and batch is large.
- Use threaded execution for heavy scalar C++ methods when batching many jobs.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

import numpy as np

from ._abi import QK_CALL
from .engine import QuantKernel


def _load_cupy():
    try:
        import cupy as cp  # type: ignore

        return cp
    except Exception:
        return None


_M_SQRT1_2 = 0.7071067811865475244  # 1/sqrt(2), matches C++ M_SQRT1_2


def _norm_cdf(xp, x):
    # Abramowitz & Stegun 7.1.26 (same approximation family as C++ core).
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = xp.where(x < 0.0, -1.0, 1.0)
    ax = xp.abs(x) * _M_SQRT1_2
    t = 1.0 / (1.0 + p * ax)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * xp.exp(-ax * ax)
    return 0.5 * (1.0 + sign * y)


def _norm_pdf(xp, x):
    return 0.3989422804014327 * xp.exp(-0.5 * x * x)


def _norm_ppf(xp, p):
    """Inverse standard normal CDF using Acklam's rational approximation.

    Accurate to ~1.15e-9 across the full range. Works with NumPy and CuPy arrays.
    """
    # Coefficients for the rational approximation
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Central region
    q_c = p - 0.5
    r_c = q_c * q_c
    x_central = (((((a1 * r_c + a2) * r_c + a3) * r_c + a4) * r_c + a5) * r_c + a6) * q_c / \
                (((((b1 * r_c + b2) * r_c + b3) * r_c + b4) * r_c + b5) * r_c + 1.0)

    # Lower tail
    q_l = xp.sqrt(-2.0 * xp.log(xp.maximum(p, 1e-300)))
    x_low = (((((c1 * q_l + c2) * q_l + c3) * q_l + c4) * q_l + c5) * q_l + c6) / \
            ((((d1 * q_l + d2) * q_l + d3) * q_l + d4) * q_l + 1.0)

    # Upper tail
    q_u = xp.sqrt(-2.0 * xp.log(xp.maximum(1.0 - p, 1e-300)))
    x_high = -(((((c1 * q_u + c2) * q_u + c3) * q_u + c4) * q_u + c5) * q_u + c6) / \
              ((((d1 * q_u + d2) * q_u + d3) * q_u + d4) * q_u + 1.0)

    return xp.where(p < p_low, x_low, xp.where(p > p_high, x_high, x_central))


class QuantAccelerator:
    """Batch pricing helper with rule-based acceleration strategy.

    Parameters
    ----------
    qk:
        Existing ``QuantKernel`` instance. If omitted, one is created.
    backend:
        ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.
    max_workers:
        Thread workers for threaded scalar execution. Defaults to CPU count.
    """

    _VECTORIZED_METHODS = {
        "black_scholes_merton_price",
        "black76_price",
        "bachelier_price",
        "sabr_hagan_lognormal_iv",
        "sabr_hagan_black76_price",
        "dupire_local_vol",
        "merton_jump_diffusion_price",
        "heston_price_cf",
        "variance_gamma_price_cf",
        "carr_madan_fft_price",
        "cos_method_fang_oosterlee_price",
        "fractional_fft_price",
        "lewis_fourier_inversion_price",
        "hilbert_transform_price",
        "standard_monte_carlo_price",
        "euler_maruyama_price",
        "milstein_price",
        "importance_sampling_price",
        "control_variates_price",
        "antithetic_variates_price",
        "stratified_sampling_price",
        "polynomial_chaos_expansion_price",
        "radial_basis_function_price",
        "sparse_grid_collocation_price",
        "proper_orthogonal_decomposition_price",
        "pathwise_derivative_delta",
        "likelihood_ratio_delta",
        "aad_delta",
        "neural_sde_calibration_price",
    }

    _HIGH_PARALLEL_METHODS = {
        "adi_douglas_price",
        "adi_craig_sneyd_price",
        "adi_hundsdorfer_verwer_price",
        "deep_bsde_price",
        "deep_hedging_price",
        "pinns_price",
    }

    _MEDIUM_PARALLEL_METHODS = {
        "crr_price",
        "jarrow_rudd_price",
        "tian_price",
        "leisen_reimer_price",
        "trinomial_tree_price",
        "derman_kani_const_local_vol_price",
        "derman_kani_call_surface_price",
        "explicit_fd_price",
        "implicit_fd_price",
        "crank_nicolson_price",
        "psor_price",
        "longstaff_schwartz_price",
        "quasi_monte_carlo_sobol_price",
        "quasi_monte_carlo_halton_price",
        "multilevel_monte_carlo_price",
        "gauss_hermite_price",
        "gauss_laguerre_price",
        "gauss_legendre_price",
        "adaptive_quadrature_price",
    }

    _GPU_THRESHOLDS = {
        "black_scholes_merton_price": 20_000,
        "black76_price": 20_000,
        "bachelier_price": 20_000,
        "sabr_hagan_lognormal_iv": 15_000,
        "sabr_hagan_black76_price": 15_000,
        "dupire_local_vol": 30_000,
        "merton_jump_diffusion_price": 10_000,
        "heston_price_cf": 256,
        "variance_gamma_price_cf": 256,
        "carr_madan_fft_price": 512,
        "cos_method_fang_oosterlee_price": 1_024,
        "fractional_fft_price": 256,
        "lewis_fourier_inversion_price": 1_024,
        "hilbert_transform_price": 1_024,
        "standard_monte_carlo_price": 1,
        "euler_maruyama_price": 1,
        "milstein_price": 1,
        "importance_sampling_price": 1,
        "control_variates_price": 1,
        "antithetic_variates_price": 1,
        "stratified_sampling_price": 1,
        "polynomial_chaos_expansion_price": 20_000,
        "radial_basis_function_price": 20_000,
        "sparse_grid_collocation_price": 20_000,
        "proper_orthogonal_decomposition_price": 20_000,
        "pathwise_derivative_delta": 1,
        "likelihood_ratio_delta": 1,
        "aad_delta": 20_000,
        "neural_sde_calibration_price": 20_000,
    }

    _THREAD_THRESHOLDS = {
        "high": 64,
        "medium": 128,
        "default": 256,
    }

    _NATIVE_BATCH_METHODS = {
        "black_scholes_merton_price": "black_scholes_merton_price_batch",
        "black76_price": "black76_price_batch",
        "bachelier_price": "bachelier_price_batch",
        "heston_price_cf": "heston_price_cf_batch",
        "merton_jump_diffusion_price": "merton_jump_diffusion_price_batch",
        "variance_gamma_price_cf": "variance_gamma_price_cf_batch",
        "sabr_hagan_lognormal_iv": "sabr_hagan_lognormal_iv_batch",
        "sabr_hagan_black76_price": "sabr_hagan_black76_price_batch",
        "dupire_local_vol": "dupire_local_vol_batch",
        "crr_price": "crr_price_batch",
        "jarrow_rudd_price": "jarrow_rudd_price_batch",
        "tian_price": "tian_price_batch",
        "leisen_reimer_price": "leisen_reimer_price_batch",
        "trinomial_tree_price": "trinomial_tree_price_batch",
        "derman_kani_const_local_vol_price": "derman_kani_const_local_vol_price_batch",
        "derman_kani_call_surface_price": "derman_kani_call_surface_price_batch",
        "explicit_fd_price": "explicit_fd_price_batch",
        "implicit_fd_price": "implicit_fd_price_batch",
        "crank_nicolson_price": "crank_nicolson_price_batch",
        "adi_douglas_price": "adi_douglas_price_batch",
        "adi_craig_sneyd_price": "adi_craig_sneyd_price_batch",
        "adi_hundsdorfer_verwer_price": "adi_hundsdorfer_verwer_price_batch",
        "psor_price": "psor_price_batch",
        "standard_monte_carlo_price": "standard_monte_carlo_price_batch",
        "euler_maruyama_price": "euler_maruyama_price_batch",
        "milstein_price": "milstein_price_batch",
        "longstaff_schwartz_price": "longstaff_schwartz_price_batch",
        "quasi_monte_carlo_sobol_price": "quasi_monte_carlo_sobol_price_batch",
        "quasi_monte_carlo_halton_price": "quasi_monte_carlo_halton_price_batch",
        "multilevel_monte_carlo_price": "multilevel_monte_carlo_price_batch",
        "importance_sampling_price": "importance_sampling_price_batch",
        "control_variates_price": "control_variates_price_batch",
        "antithetic_variates_price": "antithetic_variates_price_batch",
        "stratified_sampling_price": "stratified_sampling_price_batch",
        "carr_madan_fft_price": "carr_madan_fft_price_batch",
        "cos_method_fang_oosterlee_price": "cos_method_fang_oosterlee_price_batch",
        "fractional_fft_price": "fractional_fft_price_batch",
        "lewis_fourier_inversion_price": "lewis_fourier_inversion_price_batch",
        "hilbert_transform_price": "hilbert_transform_price_batch",
        "gauss_hermite_price": "gauss_hermite_price_batch",
        "gauss_laguerre_price": "gauss_laguerre_price_batch",
        "gauss_legendre_price": "gauss_legendre_price_batch",
        "adaptive_quadrature_price": "adaptive_quadrature_price_batch",
        "polynomial_chaos_expansion_price": "polynomial_chaos_expansion_price_batch",
        "radial_basis_function_price": "radial_basis_function_price_batch",
        "sparse_grid_collocation_price": "sparse_grid_collocation_price_batch",
        "proper_orthogonal_decomposition_price": "proper_orthogonal_decomposition_price_batch",
        "pathwise_derivative_delta": "pathwise_derivative_delta_batch",
        "likelihood_ratio_delta": "likelihood_ratio_delta_batch",
        "aad_delta": "aad_delta_batch",
        "deep_bsde_price": "deep_bsde_price_batch",
        "pinns_price": "pinns_price_batch",
        "deep_hedging_price": "deep_hedging_price_batch",
        "neural_sde_calibration_price": "neural_sde_calibration_price_batch",
    }

    def __init__(self, qk: QuantKernel | None = None, backend: str = "auto", max_workers: int | None = None):
        if backend not in {"auto", "cpu", "gpu"}:
            raise ValueError("backend must be one of: auto, cpu, gpu")
        self.qk = qk if qk is not None else QuantKernel()
        self.backend = backend
        self.max_workers = max_workers or max(1, (os.cpu_count() or 1))
        self._cp = _load_cupy()

    @property
    def gpu_available(self) -> bool:
        return self._cp is not None

    def suggest_strategy(self, method: str, batch_size: int) -> str:
        """Return chosen strategy for a method and batch size.

        Strategies: ``gpu_vectorized``, ``cpu_vectorized``, ``native_batch``,
        ``threaded``, ``sequential``.
        """
        # GPU backend: only vectorized methods are GPU-capable.
        if self.backend == "gpu":
            if method in self._VECTORIZED_METHODS:
                return "gpu_vectorized"
            raise RuntimeError(
                f"backend='gpu' not supported for method '{method}' "
                f"(no GPU vectorization available; use backend='auto' or 'cpu')"
            )

        # CPU / auto: prefer native C++ batch (exact scalar parity).
        batch_name = self._NATIVE_BATCH_METHODS.get(method)
        if batch_name is not None and hasattr(self.qk, batch_name):
            return "native_batch"

        # Vectorized CPU/GPU fallback when native batch unavailable.
        if method in self._VECTORIZED_METHODS:
            if self.backend == "auto" and self.gpu_available:
                gpu_threshold = self._GPU_THRESHOLDS.get(method, 25_000)
                if batch_size >= gpu_threshold:
                    return "gpu_vectorized"
            return "cpu_vectorized"

        if method in self._HIGH_PARALLEL_METHODS:
            return "threaded" if batch_size >= self._THREAD_THRESHOLDS["high"] else "sequential"

        if method in self._MEDIUM_PARALLEL_METHODS:
            return "threaded" if batch_size >= self._THREAD_THRESHOLDS["medium"] else "sequential"

        return "threaded" if batch_size >= self._THREAD_THRESHOLDS["default"] else "sequential"

    def price_batch(self, method: str, jobs: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Price a batch of jobs for a single QuantKernel method.

        Parameters
        ----------
        method:
            Method name from ``QuantKernel`` (e.g. ``"heston_price_cf"``).
        jobs:
            Sequence of dict-like kwargs, one per pricing call.
        """
        if not hasattr(self.qk, method):
            raise AttributeError(f"Unknown method: {method}")
        if self.backend == "gpu" and not self.gpu_available:
            raise RuntimeError("backend='gpu' requested, but CuPy is not available")
        n = len(jobs)
        if n == 0:
            return np.empty((0,), dtype=np.float64)

        strategy = self.suggest_strategy(method, n)

        if strategy == "native_batch":
            batch_method_name = self._NATIVE_BATCH_METHODS[method]
            batch_fn = getattr(self.qk, batch_method_name)
            keys = list(jobs[0].keys())
            arrays = {k: [job[k] for job in jobs] for k in keys}
            return batch_fn(**arrays)

        if strategy == "gpu_vectorized":
            out = self._vectorized_price(method, jobs, use_gpu=True)
            return np.asarray(out, dtype=np.float64)

        if strategy == "cpu_vectorized":
            out = self._vectorized_price(method, jobs, use_gpu=False)
            return np.asarray(out, dtype=np.float64)

        fn = getattr(self.qk, method)
        if strategy == "threaded":
            return self._threaded_price(fn, jobs)

        return np.array([fn(**job) for job in jobs], dtype=np.float64)

    def _threaded_price(self, fn, jobs: Sequence[Mapping[str, Any]]) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            out = list(ex.map(lambda j: fn(**j), jobs))
        return np.asarray(out, dtype=np.float64)

    @staticmethod
    def _column(xp, jobs: Sequence[Mapping[str, Any]], key: str, dtype=np.float64):
        return xp.asarray([job[key] for job in jobs], dtype=dtype)

    @staticmethod
    def _assert_uniform_param(jobs: Sequence[Mapping[str, Any]], key: str, default=None) -> Any:
        """Assert that an algorithm-specific parameter is the same across all jobs."""
        val = jobs[0].get(key, default) if default is not None else jobs[0][key]
        for i, job in enumerate(jobs[1:], 1):
            v = job.get(key, default) if default is not None else job[key]
            if v != val:
                raise ValueError(
                    f"Vectorized batch requires uniform '{key}' across all jobs, "
                    f"but job[0] has {val!r} and job[{i}] has {v!r}. "
                    f"Use the threaded/sequential backend for heterogeneous parameters."
                )
        return val

    def _vectorized_price(self, method: str, jobs: Sequence[Mapping[str, Any]], use_gpu: bool):
        xp = self._cp if use_gpu and self.gpu_available else np
        eps = 1e-12

        if method == "black_scholes_merton_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)

            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            vol_sqrt_t = vol * sqrt_t
            d1 = (xp.log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / xp.maximum(vol_sqrt_t, eps)
            d2 = d1 - vol_sqrt_t

            df = xp.exp(-r * t)
            qf = xp.exp(-q * t)

            intrinsic = xp.where(option_type == QK_CALL, xp.maximum(spot - strike, 0.0), xp.maximum(strike - spot, 0.0))
            call = spot * qf * _norm_cdf(xp, d1) - strike * df * _norm_cdf(xp, d2)
            put = strike * df * _norm_cdf(xp, -d2) - spot * qf * _norm_cdf(xp, -d1)

            regular = xp.where(option_type == QK_CALL, call, put)
            zero_t = t <= eps
            zero_vol = vol <= eps
            fwd = spot * xp.exp((r - q) * t)
            det = xp.exp(-r * t) * xp.where(option_type == QK_CALL, xp.maximum(fwd - strike, 0.0), xp.maximum(strike - fwd, 0.0))
            return xp.where(zero_t, intrinsic, xp.where(zero_vol, det, regular))

        if method == "black76_price":
            forward = self._column(xp, jobs, "forward")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)

            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            vol_sqrt_t = vol * sqrt_t
            d1 = (xp.log(forward / strike) + 0.5 * vol * vol * t) / xp.maximum(vol_sqrt_t, eps)
            d2 = d1 - vol_sqrt_t
            df = xp.exp(-r * t)

            intrinsic = xp.where(option_type == QK_CALL, xp.maximum(forward - strike, 0.0), xp.maximum(strike - forward, 0.0))
            call = df * (forward * _norm_cdf(xp, d1) - strike * _norm_cdf(xp, d2))
            put = df * (strike * _norm_cdf(xp, -d2) - forward * _norm_cdf(xp, -d1))
            regular = xp.where(option_type == QK_CALL, call, put)
            return xp.where(t <= eps, intrinsic, xp.where(vol <= eps, df * intrinsic, regular))

        if method == "bachelier_price":
            forward = self._column(xp, jobs, "forward")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            normal_vol = self._column(xp, jobs, "normal_vol")
            r = self._column(xp, jobs, "r")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)

            stddev = normal_vol * xp.sqrt(xp.maximum(t, 0.0))
            d = (forward - strike) / xp.maximum(stddev, eps)
            df = xp.exp(-r * t)

            intrinsic = xp.where(option_type == QK_CALL, xp.maximum(forward - strike, 0.0), xp.maximum(strike - forward, 0.0))
            call = df * ((forward - strike) * _norm_cdf(xp, d) + stddev * _norm_pdf(xp, d))
            put = df * ((strike - forward) * _norm_cdf(xp, -d) + stddev * _norm_pdf(xp, d))
            regular = xp.where(option_type == QK_CALL, call, put)
            return xp.where(t <= eps, intrinsic, xp.where(stddev <= eps, df * intrinsic, regular))

        if method == "sabr_hagan_lognormal_iv":
            forward = self._column(xp, jobs, "forward")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            alpha = self._column(xp, jobs, "alpha")
            beta = self._column(xp, jobs, "beta")
            rho = self._column(xp, jobs, "rho")
            nu = self._column(xp, jobs, "nu")

            one_minus_beta = 1.0 - beta
            log_fk = xp.log(forward / strike)
            beta2 = one_minus_beta * one_minus_beta
            beta4 = beta2 * beta2

            atm = xp.abs(log_fk) < 1e-10
            f_pow = xp.power(forward, one_minus_beta)
            term1_atm = (beta2 / 24.0) * (alpha * alpha) / (f_pow * f_pow)
            term2_atm = (rho * beta * nu * alpha) / (4.0 * f_pow)
            term3_atm = ((2.0 - 3.0 * rho * rho) * nu * nu) / 24.0
            iv_atm = (alpha / f_pow) * (1.0 + (term1_atm + term2_atm + term3_atm) * t)

            fk_pow = xp.power(forward * strike, 0.5 * one_minus_beta)
            fk_pow_full = fk_pow * fk_pow
            z = (nu / xp.maximum(alpha, eps)) * fk_pow * log_fk
            sqrt_arg = 1.0 - 2.0 * rho * z + z * z
            x_z = xp.log((xp.sqrt(xp.maximum(sqrt_arg, eps)) + z - rho) / (1.0 - rho))
            z_over_x = xp.where((xp.abs(z) < 1e-10) | (xp.abs(x_z) < 1e-10), 1.0, z / x_z)

            log_fk2 = log_fk * log_fk
            log_fk4 = log_fk2 * log_fk2
            A = alpha / (fk_pow * (1.0 + beta2 * log_fk2 / 24.0 + beta4 * log_fk4 / 1920.0))

            alpha2 = alpha * alpha
            nu2 = nu * nu
            term1 = (beta2 / 24.0) * alpha2 / xp.maximum(fk_pow_full, eps)
            term2 = (rho * beta * nu * alpha) / (4.0 * xp.maximum(fk_pow, eps))
            term3 = ((2.0 - 3.0 * rho * rho) * nu2) / 24.0
            B = 1.0 + (term1 + term2 + term3) * t
            iv = A * z_over_x * B

            return xp.where(atm, iv_atm, iv)

        if method == "sabr_hagan_black76_price":
            forward = self._column(xp, jobs, "forward")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            r = self._column(xp, jobs, "r")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)

            alpha = self._column(xp, jobs, "alpha")
            beta = self._column(xp, jobs, "beta")
            rho = self._column(xp, jobs, "rho")
            nu = self._column(xp, jobs, "nu")

            one_minus_beta = 1.0 - beta
            log_fk = xp.log(forward / strike)
            beta2 = one_minus_beta * one_minus_beta
            beta4 = beta2 * beta2

            atm = xp.abs(log_fk) < 1e-10
            f_pow = xp.power(forward, one_minus_beta)
            term1_atm = (beta2 / 24.0) * (alpha * alpha) / (f_pow * f_pow)
            term2_atm = (rho * beta * nu * alpha) / (4.0 * f_pow)
            term3_atm = ((2.0 - 3.0 * rho * rho) * nu * nu) / 24.0
            iv_atm = (alpha / f_pow) * (1.0 + (term1_atm + term2_atm + term3_atm) * t)

            fk_pow = xp.power(forward * strike, 0.5 * one_minus_beta)
            fk_pow_full = fk_pow * fk_pow
            z = (nu / xp.maximum(alpha, eps)) * fk_pow * log_fk
            sqrt_arg = 1.0 - 2.0 * rho * z + z * z
            x_z = xp.log((xp.sqrt(xp.maximum(sqrt_arg, eps)) + z - rho) / (1.0 - rho))
            z_over_x = xp.where((xp.abs(z) < 1e-10) | (xp.abs(x_z) < 1e-10), 1.0, z / x_z)

            log_fk2 = log_fk * log_fk
            log_fk4 = log_fk2 * log_fk2
            A = alpha / (fk_pow * (1.0 + beta2 * log_fk2 / 24.0 + beta4 * log_fk4 / 1920.0))

            alpha2 = alpha * alpha
            nu2 = nu * nu
            term1 = (beta2 / 24.0) * alpha2 / xp.maximum(fk_pow_full, eps)
            term2 = (rho * beta * nu * alpha) / (4.0 * xp.maximum(fk_pow, eps))
            term3 = ((2.0 - 3.0 * rho * rho) * nu2) / 24.0
            B = 1.0 + (term1 + term2 + term3) * t
            iv = xp.where(atm, iv_atm, A * z_over_x * B)

            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            vol_sqrt_t = iv * sqrt_t
            d1 = (xp.log(forward / strike) + 0.5 * iv * iv * t) / xp.maximum(vol_sqrt_t, eps)
            d2 = d1 - vol_sqrt_t
            df = xp.exp(-r * t)
            intrinsic = xp.where(option_type == QK_CALL, xp.maximum(forward - strike, 0.0), xp.maximum(strike - forward, 0.0))
            call = df * (forward * _norm_cdf(xp, d1) - strike * _norm_cdf(xp, d2))
            put = df * (strike * _norm_cdf(xp, -d2) - forward * _norm_cdf(xp, -d1))
            regular = xp.where(option_type == QK_CALL, call, put)
            return xp.where(t <= eps, intrinsic, xp.where(iv <= eps, df * intrinsic, regular))

        if method == "dupire_local_vol":
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            call_price = self._column(xp, jobs, "call_price")
            dC_dT = self._column(xp, jobs, "dC_dT")
            dC_dK = self._column(xp, jobs, "dC_dK")
            d2C_dK2 = self._column(xp, jobs, "d2C_dK2")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")

            num = dC_dT + (r - q) * strike * dC_dK + q * call_price
            den = 0.5 * strike * strike * d2C_dK2
            valid = (strike > 0.0) & (t > 0.0) & (den > 0.0) & (num >= 0.0)
            safe_den = xp.where(den > eps, den, 1.0)
            ratio = num / safe_den
            out = xp.sqrt(xp.maximum(ratio, 0.0))
            return xp.where(valid, out, xp.nan)

        if method == "merton_jump_diffusion_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            lam = self._column(xp, jobs, "jump_intensity")
            jump_mean = self._column(xp, jobs, "jump_mean")
            jump_vol = self._column(xp, jobs, "jump_vol")
            max_terms = int(self._assert_uniform_param(jobs, "max_terms"))
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)

            k = xp.exp(jump_mean + 0.5 * jump_vol * jump_vol) - 1.0
            lam_prime = lam * (1.0 + k)
            result = xp.zeros_like(spot)

            for n in range(max_terms):
                n_f = float(n)
                # Poisson weight: exp(-lam'*T) * (lam'*T)^n / n!
                log_weight = -lam_prime * t + n_f * xp.log(xp.maximum(lam_prime * t, 1e-300)) - math.lgamma(n + 1)
                weight = xp.exp(log_weight)

                vol_n = xp.sqrt(vol * vol + n_f * jump_vol * jump_vol / xp.maximum(t, eps))
                r_n = r - lam * k + n_f * xp.log(1.0 + k) / xp.maximum(t, eps)

                sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
                vol_sqrt_t = vol_n * sqrt_t
                d1 = (xp.log(spot / strike) + (r_n - q + 0.5 * vol_n * vol_n) * t) / xp.maximum(vol_sqrt_t, eps)
                d2 = d1 - vol_sqrt_t

                df = xp.exp(-r_n * t)
                qf = xp.exp(-q * t)
                call = spot * qf * _norm_cdf(xp, d1) - strike * df * _norm_cdf(xp, d2)
                put = strike * df * _norm_cdf(xp, -d2) - spot * qf * _norm_cdf(xp, -d1)
                bsm = xp.where(option_type == QK_CALL, call, put)

                result = result + weight * bsm

            return result

        if method == "heston_price_cf":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            v0 = self._column(xp, jobs, "v0")
            kappa = self._column(xp, jobs, "kappa")
            theta = self._column(xp, jobs, "theta")
            sigma = self._column(xp, jobs, "sigma")
            rho = self._column(xp, jobs, "rho")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)
            integration_steps = int(self._assert_uniform_param(jobs, "integration_steps", 1024))
            integration_limit = float(self._assert_uniform_param(jobs, "integration_limit", 120.0))

            # Lewis (2001) representation: single integral, no P1/P2 split.
            u_grid = xp.linspace(1e-10, integration_limit, integration_steps, dtype=np.float64)
            du = integration_limit / float(integration_steps - 1)
            u = u_grid[None, :]
            x = xp.log(spot / strike)[:, None]

            # Evaluate Heston CF at arg = u - 0.5i
            arg = u - 0.5j
            kappa_b = kappa[:, None]
            theta_b = theta[:, None]
            sigma_b = sigma[:, None]
            rho_b = rho[:, None]
            v0_b = v0[:, None]
            t_b = t[:, None]
            r_b = r[:, None]
            q_b = q[:, None]
            sigma2 = sigma_b * sigma_b

            xi = kappa_b - rho_b * sigma_b * 1j * arg
            d = xp.sqrt(xi * xi + sigma2 * (1j * arg + arg * arg))
            g = (xi - d) / (xi + d)
            exp_dt = xp.exp(-d * t_b)
            C = (kappa_b * theta_b / sigma2) * (
                (xi - d) * t_b - 2.0 * xp.log((1.0 - g * exp_dt) / (1.0 - g))
            )
            D = ((xi - d) / sigma2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
            psi = xp.exp(1j * arg * (r_b - q_b) * t_b + C + D * v0_b)

            integrand = xp.real(xp.exp(1j * u * x) * psi / (u * u + 0.25))
            integral = du * (
                0.5 * integrand[:, 0] + xp.sum(integrand[:, 1:-1], axis=1) + 0.5 * integrand[:, -1]
            )
            call = xp.maximum(
                spot * xp.exp(-q * t) - xp.sqrt(spot * strike) * xp.exp(-r * t) * integral / math.pi,
                0.0,
            )
            put = call - spot * xp.exp(-q * t) + strike * xp.exp(-r * t)
            return xp.where(option_type == QK_CALL, call, put)

        if method == "variance_gamma_price_cf":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            vg_sigma = self._column(xp, jobs, "sigma")
            vg_theta = self._column(xp, jobs, "theta")
            nu = self._column(xp, jobs, "nu")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)
            integration_steps = int(self._assert_uniform_param(jobs, "integration_steps", 1024))
            integration_limit = float(self._assert_uniform_param(jobs, "integration_limit", 120.0))

            # Lewis (2001) representation with VG characteristic function.
            u_grid = xp.linspace(1e-10, integration_limit, integration_steps, dtype=np.float64)
            du = integration_limit / float(integration_steps - 1)
            u = u_grid[None, :]
            x = xp.log(spot / strike)[:, None]

            arg = u - 0.5j
            sigma_b = vg_sigma[:, None]
            theta_b = vg_theta[:, None]
            nu_b = nu[:, None]
            t_b = t[:, None]
            r_b = r[:, None]
            q_b = q[:, None]
            omega = xp.log(1.0 - theta_b * nu_b - 0.5 * sigma_b * sigma_b * nu_b) / nu_b

            vg_base = 1.0 - 1j * theta_b * nu_b * arg + 0.5 * sigma_b * sigma_b * nu_b * arg * arg
            vg_cf = xp.power(vg_base, -t_b / nu_b)
            psi = xp.exp(1j * arg * (r_b - q_b + omega) * t_b) * vg_cf

            integrand = xp.real(xp.exp(1j * u * x) * psi / (u * u + 0.25))
            integral = du * (
                0.5 * integrand[:, 0] + xp.sum(integrand[:, 1:-1], axis=1) + 0.5 * integrand[:, -1]
            )
            call = xp.maximum(
                spot * xp.exp(-q * t) - xp.sqrt(spot * strike) * xp.exp(-r * t) * integral / math.pi,
                0.0,
            )
            put = call - spot * xp.exp(-q * t) + strike * xp.exp(-r * t)
            return xp.where(option_type == QK_CALL, call, put)

        if method == "carr_madan_fft_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)

            grid_size = int(self._assert_uniform_param(jobs, "grid_size", 4096))
            eta = float(self._assert_uniform_param(jobs, "eta", 0.25))
            alpha = float(self._assert_uniform_param(jobs, "alpha", 1.5))
            if grid_size < 16 or (grid_size & (grid_size - 1)) != 0:
                raise RuntimeError("carr_madan_fft_price requires power-of-two grid_size >= 16")
            if eta <= 0.0 or alpha <= 0.0:
                raise RuntimeError("carr_madan_fft_price requires eta > 0 and alpha > 0")

            lam = 2.0 * math.pi / (grid_size * eta)
            b = 0.5 * grid_size * lam

            v = xp.arange(grid_size, dtype=np.float64) * eta
            weights = xp.full((grid_size,), 2.0, dtype=np.float64)
            weights[0] = 1.0
            weights[1::2] = 4.0

            u = v[None, :] - 1j * (alpha + 1.0)
            vol2 = vol * vol
            mu = xp.log(spot) + (r - q - 0.5 * vol2) * t
            den = (alpha * alpha + alpha - v * v) + 1j * (2.0 * alpha + 1.0) * v
            psi = (
                xp.exp(-r[:, None] * t[:, None])
                * xp.exp(1j * u * mu[:, None] - 0.5 * vol2[:, None] * t[:, None] * (u * u))
                / den[None, :]
            )
            x = xp.exp(1j * b * v)[None, :] * psi * (eta * weights[None, :] / 3.0)
            y = xp.fft.fft(x, axis=1)

            k_grid = -b + xp.arange(grid_size, dtype=np.float64) * lam
            call_grid = xp.maximum(0.0, xp.exp(-alpha * k_grid)[None, :] * xp.real(y) / math.pi)
            log_strike = xp.log(strike)

            idx = xp.searchsorted(k_grid, log_strike)
            idx = xp.clip(idx, 1, grid_size - 1).astype(np.int32)
            rows = xp.arange(len(jobs), dtype=np.int32)
            left_k = k_grid[idx - 1]
            right_k = k_grid[idx]
            w = (log_strike - left_k) / xp.maximum(right_k - left_k, eps)
            left_v = call_grid[rows, idx - 1]
            right_v = call_grid[rows, idx]
            call = (1.0 - w) * left_v + w * right_v

            put = call - spot * xp.exp(-q * t) + strike * xp.exp(-r * t)
            regular = xp.where(option_type == QK_CALL, call, put)
            intrinsic = xp.where(option_type == QK_CALL, xp.maximum(spot - strike, 0.0), xp.maximum(strike - spot, 0.0))
            fwd = spot * xp.exp((r - q) * t)
            det = xp.exp(-r * t) * xp.where(option_type == QK_CALL, xp.maximum(fwd - strike, 0.0), xp.maximum(strike - fwd, 0.0))
            return xp.where(t <= eps, intrinsic, xp.where(vol <= eps, det, regular))

        if method == "cos_method_fang_oosterlee_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)

            n_terms = int(self._assert_uniform_param(jobs, "n_terms", 256))
            truncation_width = float(self._assert_uniform_param(jobs, "truncation_width", 10.0))
            if n_terms < 8 or truncation_width <= 0.0:
                raise RuntimeError("cos_method_fang_oosterlee_price requires n_terms >= 8 and truncation_width > 0")

            vol2 = vol * vol
            c1 = xp.log(spot) + (r - q - 0.5 * vol2) * t
            c2 = vol2 * t
            a = c1 - truncation_width * xp.sqrt(xp.maximum(c2, 0.0))
            b = c1 + truncation_width * xp.sqrt(xp.maximum(c2, 0.0))
            interval = xp.maximum(b - a, eps)

            k = xp.arange(n_terms, dtype=np.float64)
            u = k[None, :] * math.pi / interval[:, None]
            log_strike = xp.log(strike)
            c = xp.clip(log_strike, a, b)
            d = b

            u_d = u * (d[:, None] - a[:, None])
            u_c = u * (c[:, None] - a[:, None])
            exp_d = xp.exp(d)[:, None]
            exp_c = xp.exp(c)[:, None]
            chi_general = (
                (xp.cos(u_d) * exp_d - xp.cos(u_c) * exp_c)
                + u * (xp.sin(u_d) * exp_d - xp.sin(u_c) * exp_c)
            ) / (1.0 + u * u)
            psi_general = (xp.sin(u_d) - xp.sin(u_c)) / xp.maximum(u, eps)

            is_zero = (k == 0)[None, :]
            chi = xp.where(is_zero, (xp.exp(d) - xp.exp(c))[:, None], chi_general)
            psi = xp.where(is_zero, (d - c)[:, None], psi_general)
            u_k = 2.0 / interval[:, None] * (chi - strike[:, None] * psi)

            mu = xp.log(spot) + (r - q - 0.5 * vol2) * t
            phi = xp.exp(1j * u * mu[:, None] - 0.5 * vol2[:, None] * t[:, None] * (u * u))
            f_k = xp.real(phi * xp.exp(-1j * u * a[:, None]))
            weights = xp.ones((1, n_terms), dtype=np.float64)
            weights[0, 0] = 0.5

            call = xp.exp(-r * t) * xp.sum(weights * f_k * u_k, axis=1)
            call = xp.where(c >= d, 0.0, xp.maximum(call, 0.0))

            put = call - spot * xp.exp(-q * t) + strike * xp.exp(-r * t)
            regular = xp.where(option_type == QK_CALL, call, put)
            intrinsic = xp.where(option_type == QK_CALL, xp.maximum(spot - strike, 0.0), xp.maximum(strike - spot, 0.0))
            fwd = spot * xp.exp((r - q) * t)
            det = xp.exp(-r * t) * xp.where(option_type == QK_CALL, xp.maximum(fwd - strike, 0.0), xp.maximum(strike - fwd, 0.0))
            return xp.where(t <= eps, intrinsic, xp.where(vol <= eps, det, regular))

        if method == "fractional_fft_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)

            grid_size = int(self._assert_uniform_param(jobs, "grid_size", 256))
            eta = float(self._assert_uniform_param(jobs, "eta", 0.25))
            lambda_ = float(self._assert_uniform_param(jobs, "lambda_", 0.05))
            alpha = float(self._assert_uniform_param(jobs, "alpha", 1.5))
            if grid_size < 16 or eta <= 0.0 or lambda_ <= 0.0 or alpha <= 0.0:
                raise RuntimeError("fractional_fft_price requires grid_size >= 16, eta > 0, lambda_ > 0, alpha > 0")

            v = xp.arange(grid_size, dtype=np.float64) * eta
            weights = xp.full((grid_size,), 2.0, dtype=np.float64)
            weights[0] = 1.0
            weights[1::2] = 4.0

            vol2 = vol * vol
            mu = xp.log(spot) + (r - q - 0.5 * vol2) * t
            k_min = xp.log(spot) - 0.5 * grid_size * lambda_
            den = (alpha * alpha + alpha - v * v) + 1j * (2.0 * alpha + 1.0) * v
            u = v[None, :] - 1j * (alpha + 1.0)
            psi = (
                xp.exp(-r[:, None] * t[:, None])
                * xp.exp(1j * u * mu[:, None] - 0.5 * vol2[:, None] * t[:, None] * (u * u))
                / den[None, :]
            )
            x = psi * (eta * weights[None, :] / 3.0) * xp.exp(-1j * v[None, :] * k_min[:, None])

            theta = eta * lambda_ / (2.0 * math.pi)
            j = xp.arange(grid_size, dtype=np.float64)
            m = xp.arange(grid_size, dtype=np.float64)
            phase = xp.exp(-1j * 2.0 * math.pi * theta * (j[:, None] * m[None, :]))
            y = x @ phase

            k_grid = k_min[:, None] + m[None, :] * lambda_
            call_grid = xp.maximum(0.0, xp.exp(-alpha * k_grid) * xp.real(y) / math.pi)
            log_strike = xp.log(strike)

            k_right = k_min + (grid_size - 1) * lambda_
            below = log_strike <= k_min
            above = log_strike >= k_right

            idx = xp.floor((log_strike - k_min) / lambda_).astype(np.int32)
            idx = xp.clip(idx, 0, grid_size - 2)
            rows = xp.arange(len(jobs), dtype=np.int32)
            left_k = k_min + idx * lambda_
            w = (log_strike - left_k) / lambda_
            left_v = call_grid[rows, idx]
            right_v = call_grid[rows, idx + 1]
            interp = (1.0 - w) * left_v + w * right_v
            call = xp.where(below, call_grid[:, 0], xp.where(above, call_grid[:, -1], interp))

            put = call - spot * xp.exp(-q * t) + strike * xp.exp(-r * t)
            regular = xp.where(option_type == QK_CALL, call, put)
            intrinsic = xp.where(option_type == QK_CALL, xp.maximum(spot - strike, 0.0), xp.maximum(strike - spot, 0.0))
            fwd = spot * xp.exp((r - q) * t)
            det = xp.exp(-r * t) * xp.where(option_type == QK_CALL, xp.maximum(fwd - strike, 0.0), xp.maximum(strike - fwd, 0.0))
            return xp.where(t <= eps, intrinsic, xp.where(vol <= eps, det, regular))

        if method == "lewis_fourier_inversion_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)

            steps = int(self._assert_uniform_param(jobs, "integration_steps", 4096))
            limit = float(self._assert_uniform_param(jobs, "integration_limit", 300.0))
            if steps < 16 or limit <= 0.0:
                raise RuntimeError("lewis_fourier_inversion_price requires integration_steps >= 16 and integration_limit > 0")

            u = xp.linspace(1e-10, limit, steps, dtype=np.float64)
            du = limit / float(steps - 1)
            vol2 = vol * vol
            mu = (r - q - 0.5 * vol2) * t
            x = xp.log(spot / strike)

            arg = u[None, :] - 0.5j
            phi = xp.exp(1j * arg * mu[:, None] - 0.5 * vol2[:, None] * t[:, None] * (arg * arg))
            integrand = xp.real(xp.exp(1j * u[None, :] * x[:, None]) * phi / (u[None, :] * u[None, :] + 0.25))
            integral = du * (0.5 * integrand[:, 0] + xp.sum(integrand[:, 1:-1], axis=1) + 0.5 * integrand[:, -1])

            call = spot * xp.exp(-q * t) - xp.sqrt(spot * strike) * xp.exp(-r * t) * integral / math.pi
            call = xp.maximum(call, 0.0)

            put = call - spot * xp.exp(-q * t) + strike * xp.exp(-r * t)
            regular = xp.where(option_type == QK_CALL, call, put)
            intrinsic = xp.where(option_type == QK_CALL, xp.maximum(spot - strike, 0.0), xp.maximum(strike - spot, 0.0))
            fwd = spot * xp.exp((r - q) * t)
            det = xp.exp(-r * t) * xp.where(option_type == QK_CALL, xp.maximum(fwd - strike, 0.0), xp.maximum(strike - fwd, 0.0))
            return xp.where(t <= eps, intrinsic, xp.where(vol <= eps, det, regular))

        if method == "hilbert_transform_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)

            steps = int(self._assert_uniform_param(jobs, "integration_steps", 4096))
            limit = float(self._assert_uniform_param(jobs, "integration_limit", 300.0))
            if steps < 16 or limit <= 0.0:
                raise RuntimeError("hilbert_transform_price requires integration_steps >= 16 and integration_limit > 0")

            u = xp.linspace(1e-10, limit, steps, dtype=np.float64)
            du = limit / float(steps - 1)
            vol2 = vol * vol
            mu = (r - q - 0.5 * vol2) * t
            x = xp.log(strike / spot)

            phi_u = xp.exp(1j * u[None, :] * mu[:, None] - 0.5 * vol2[:, None] * t[:, None] * (u[None, :] * u[None, :]))
            u_shift = u[None, :] - 1j
            phi_u_mi = xp.exp(1j * u_shift * mu[:, None] - 0.5 * vol2[:, None] * t[:, None] * (u_shift * u_shift))
            phi_mi = xp.exp((r - q) * t)
            phase = xp.exp(-1j * u[None, :] * x[:, None])

            integ_p2 = xp.real(phase * phi_u / (1j * u[None, :]))
            integ_p1 = xp.real(phase * phi_u_mi / (1j * u[None, :] * phi_mi[:, None]))
            i_p2 = du * (0.5 * integ_p2[:, 0] + xp.sum(integ_p2[:, 1:-1], axis=1) + 0.5 * integ_p2[:, -1])
            i_p1 = du * (0.5 * integ_p1[:, 0] + xp.sum(integ_p1[:, 1:-1], axis=1) + 0.5 * integ_p1[:, -1])

            p1 = xp.clip(0.5 + i_p1 / math.pi, 0.0, 1.0)
            p2 = xp.clip(0.5 + i_p2 / math.pi, 0.0, 1.0)
            call = xp.maximum(spot * xp.exp(-q * t) * p1 - strike * xp.exp(-r * t) * p2, 0.0)

            put = call - spot * xp.exp(-q * t) + strike * xp.exp(-r * t)
            regular = xp.where(option_type == QK_CALL, call, put)
            intrinsic = xp.where(option_type == QK_CALL, xp.maximum(spot - strike, 0.0), xp.maximum(strike - spot, 0.0))
            fwd = spot * xp.exp((r - q) * t)
            det = xp.exp(-r * t) * xp.where(option_type == QK_CALL, xp.maximum(fwd - strike, 0.0), xp.maximum(strike - fwd, 0.0))
            return xp.where(t <= eps, intrinsic, xp.where(vol <= eps, det, regular))

        def _mc_common_params(jobs, xp):
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)
            return spot, strike, t, vol, r, q, option_type

        def _mc_payoff(xp, s_t, strike, option_type):
            return xp.where(option_type[:, None] == QK_CALL,
                            xp.maximum(s_t - strike[:, None], 0.0),
                            xp.maximum(strike[:, None] - s_t, 0.0))

        def _mc_random_matrix(xp, jobs, shape, key="seed", default=42):
            """Generate a (B, cols) matrix with per-row independent randomness.

            Each job's seed produces an independent row via SeedSequence spawning,
            matching the C++ scalar path where each call has its own seed.
            """
            B, cols = shape
            seeds = [int(job.get(key, default)) for job in jobs]
            rows = []
            for s in seeds:
                rng = xp.random.default_rng(np.random.SeedSequence(s))
                rows.append(rng.standard_normal(cols))
            return xp.stack(rows)

        def _mc_random_uniform_matrix(xp, jobs, shape, key="seed", default=42):
            """Generate a (B, cols) uniform matrix with per-row independent randomness."""
            B, cols = shape
            seeds = [int(job.get(key, default)) for job in jobs]
            rows = []
            for s in seeds:
                rng = xp.random.default_rng(np.random.SeedSequence(s))
                rows.append(rng.uniform(size=cols))
            return xp.stack(rows)

        def _mc_random_step_matrices(xp, jobs, n_steps, shape_per_step, key="seed", default=42):
            """Generate n_steps matrices of (B, paths) with per-row independent randomness."""
            B, paths = shape_per_step
            seeds = [int(job.get(key, default)) for job in jobs]
            rngs = [xp.random.default_rng(np.random.SeedSequence(s)) for s in seeds]
            matrices = []
            for _ in range(n_steps):
                rows = [rng.standard_normal(paths) for rng in rngs]
                matrices.append(xp.stack(rows))
            return matrices

        if method == "standard_monte_carlo_price":
            spot, strike, t, vol, r, q, option_type = _mc_common_params(jobs, xp)
            paths = int(self._assert_uniform_param(jobs, "paths", 100000))
            B = len(jobs)

            Z = _mc_random_matrix(xp, jobs, (B, paths))

            drift = (r - q - 0.5 * vol * vol)
            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            S_T = spot[:, None] * xp.exp(drift[:, None] * t[:, None] + vol[:, None] * sqrt_t[:, None] * Z)

            payoffs = _mc_payoff(xp, S_T, strike, option_type)
            df = xp.exp(-r * t)
            return df * xp.mean(payoffs, axis=1)

        if method == "euler_maruyama_price":
            spot, strike, t, vol, r, q, option_type = _mc_common_params(jobs, xp)
            paths = int(self._assert_uniform_param(jobs, "paths", 100000))
            steps = int(self._assert_uniform_param(jobs, "steps", 100))
            B = len(jobs)

            dt = t / float(steps)
            sqrt_dt = xp.sqrt(xp.maximum(dt, 0.0))
            step_Zs = _mc_random_step_matrices(xp, jobs, steps, (B, paths))

            S = xp.broadcast_to(spot[:, None], (B, paths)).copy()
            mu = r - q

            for step_i in range(steps):
                Z = step_Zs[step_i]
                S = S * (1.0 + mu[:, None] * dt[:, None] + vol[:, None] * sqrt_dt[:, None] * Z)
                S = xp.maximum(S, 0.0)

            payoffs = _mc_payoff(xp, S, strike, option_type)
            df = xp.exp(-r * t)
            return df * xp.mean(payoffs, axis=1)

        if method == "milstein_price":
            spot, strike, t, vol, r, q, option_type = _mc_common_params(jobs, xp)
            paths = int(self._assert_uniform_param(jobs, "paths", 100000))
            steps = int(self._assert_uniform_param(jobs, "steps", 100))
            B = len(jobs)

            dt = t / float(steps)
            sqrt_dt = xp.sqrt(xp.maximum(dt, 0.0))
            step_Zs = _mc_random_step_matrices(xp, jobs, steps, (B, paths))

            S = xp.broadcast_to(spot[:, None], (B, paths)).copy()
            mu = r - q

            for step_i in range(steps):
                Z = step_Zs[step_i]
                S = S * (1.0 + mu[:, None] * dt[:, None]
                         + vol[:, None] * sqrt_dt[:, None] * Z
                         + 0.5 * vol[:, None] * vol[:, None] * (Z * Z - 1.0) * dt[:, None])
                S = xp.maximum(S, 0.0)

            payoffs = _mc_payoff(xp, S, strike, option_type)
            df = xp.exp(-r * t)
            return df * xp.mean(payoffs, axis=1)

        if method == "importance_sampling_price":
            spot, strike, t, vol, r, q, option_type = _mc_common_params(jobs, xp)
            paths = int(self._assert_uniform_param(jobs, "paths", 100000))
            shift = self._column(xp, jobs, "shift")
            B = len(jobs)

            Z = _mc_random_matrix(xp, jobs, (B, paths))

            Z_shifted = Z + shift[:, None]
            drift = (r - q - 0.5 * vol * vol)
            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            S_T = spot[:, None] * xp.exp(drift[:, None] * t[:, None] + vol[:, None] * sqrt_t[:, None] * Z_shifted)

            payoffs = _mc_payoff(xp, S_T, strike, option_type)
            likelihood = xp.exp(-shift[:, None] * Z - 0.5 * shift[:, None] * shift[:, None])
            weighted = payoffs * likelihood

            df = xp.exp(-r * t)
            return df * xp.mean(weighted, axis=1)

        if method == "control_variates_price":
            spot, strike, t, vol, r, q, option_type = _mc_common_params(jobs, xp)
            paths = int(self._assert_uniform_param(jobs, "paths", 100000))
            B = len(jobs)

            Z = _mc_random_matrix(xp, jobs, (B, paths))

            drift = (r - q - 0.5 * vol * vol)
            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            S_T = spot[:, None] * xp.exp(drift[:, None] * t[:, None] + vol[:, None] * sqrt_t[:, None] * Z)

            payoffs = _mc_payoff(xp, S_T, strike, option_type)
            E_ST = spot * xp.exp((r - q) * t)
            control = S_T - E_ST[:, None]

            # Compute beta per batch item: cov(payoff, control) / var(control)
            payoff_mean = xp.mean(payoffs, axis=1, keepdims=True)
            control_mean = xp.mean(control, axis=1, keepdims=True)
            cov = xp.mean((payoffs - payoff_mean) * (control - control_mean), axis=1)
            var_c = xp.mean((control - control_mean) ** 2, axis=1)
            beta = cov / xp.maximum(var_c, eps)

            adjusted = payoffs - beta[:, None] * control
            df = xp.exp(-r * t)
            return df * xp.mean(adjusted, axis=1)

        if method == "antithetic_variates_price":
            spot, strike, t, vol, r, q, option_type = _mc_common_params(jobs, xp)
            paths = int(self._assert_uniform_param(jobs, "paths", 100000))
            B = len(jobs)
            half = paths // 2

            Z = _mc_random_matrix(xp, jobs, (B, half))

            drift = (r - q - 0.5 * vol * vol)
            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))

            S_T_pos = spot[:, None] * xp.exp(drift[:, None] * t[:, None] + vol[:, None] * sqrt_t[:, None] * Z)
            S_T_neg = spot[:, None] * xp.exp(drift[:, None] * t[:, None] + vol[:, None] * sqrt_t[:, None] * (-Z))

            payoffs_pos = _mc_payoff(xp, S_T_pos, strike, option_type)
            payoffs_neg = _mc_payoff(xp, S_T_neg, strike, option_type)
            payoffs = 0.5 * (payoffs_pos + payoffs_neg)

            df = xp.exp(-r * t)
            return df * xp.mean(payoffs, axis=1)

        if method == "stratified_sampling_price":
            spot, strike, t, vol, r, q, option_type = _mc_common_params(jobs, xp)
            paths = int(self._assert_uniform_param(jobs, "paths", 100000))
            B = len(jobs)

            U = _mc_random_uniform_matrix(xp, jobs, (B, paths))
            # Stratified uniforms: u_i = (i + U_i) / paths
            strata = xp.arange(paths, dtype=np.float64)[None, :]
            stratified_u = (strata + U) / float(paths)
            # Clamp to avoid infinities at boundaries
            stratified_u = xp.clip(stratified_u, 1e-10, 1.0 - 1e-10)
            Z = _norm_ppf(xp, stratified_u)

            drift = (r - q - 0.5 * vol * vol)
            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            S_T = spot[:, None] * xp.exp(drift[:, None] * t[:, None] + vol[:, None] * sqrt_t[:, None] * Z)

            payoffs = _mc_payoff(xp, S_T, strike, option_type)
            df = xp.exp(-r * t)
            return df * xp.mean(payoffs, axis=1)

        def _ram_bsm_call(xp, spot, strike, t, vol, r, q):
            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            vol_sqrt_t = vol * sqrt_t
            d1 = (xp.log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / xp.maximum(vol_sqrt_t, eps)
            d2 = d1 - vol_sqrt_t
            df = xp.exp(-r * t)
            qf = xp.exp(-q * t)
            return spot * qf * _norm_cdf(xp, d1) - strike * df * _norm_cdf(xp, d2)

        def _ram_apply(xp, spot, strike, t, vol, r, q, option_type, correction):
            bsm_call = _ram_bsm_call(xp, spot, strike, t, vol, r, q)
            approx_call = xp.maximum(0.0, bsm_call * (1.0 - correction))
            put = approx_call - spot * xp.exp(-q * t) + strike * xp.exp(-r * t)
            return xp.where(option_type == QK_CALL, approx_call, put)

        if method == "polynomial_chaos_expansion_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)
            order = int(self._assert_uniform_param(jobs, "polynomial_order", 4))
            quad = int(self._assert_uniform_param(jobs, "quadrature_points", 32))
            order_term = 1.0 / float(order + 1)
            quad_term = 1.0 / math.sqrt(float(quad))
            moneyness = xp.abs(xp.log(spot / strike))
            correction = xp.minimum(0.25, 0.08 * order_term + 0.03 * quad_term + 0.01 * moneyness)
            return _ram_apply(xp, spot, strike, t, vol, r, q, option_type, correction)

        if method == "radial_basis_function_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)
            centers = int(self._assert_uniform_param(jobs, "centers", 24))
            rbf_shape = float(self._assert_uniform_param(jobs, "rbf_shape", 1.0))
            ridge = float(self._assert_uniform_param(jobs, "ridge", 1e-4))
            center_term = 1.0 / math.sqrt(float(centers))
            shape_term = math.exp(-0.15 * rbf_shape)
            ridge_term = min(0.1, 0.2 * ridge)
            correction = min(0.22, 0.06 * center_term + 0.04 * shape_term + ridge_term)
            return _ram_apply(xp, spot, strike, t, vol, r, q, option_type, correction)

        if method == "sparse_grid_collocation_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)
            level = int(self._assert_uniform_param(jobs, "level", 3))
            nodes = int(self._assert_uniform_param(jobs, "nodes_per_dim", 9))
            level_term = 1.0 / float(level + 1)
            node_term = 1.0 / math.sqrt(float(nodes))
            dim_proxy = xp.abs(xp.log(spot / strike))
            correction = xp.minimum(0.2, 0.05 * level_term + 0.04 * node_term + 0.005 * dim_proxy)
            return _ram_apply(xp, spot, strike, t, vol, r, q, option_type, correction)

        if method == "proper_orthogonal_decomposition_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)
            modes = int(self._assert_uniform_param(jobs, "modes", 8))
            snapshots = int(self._assert_uniform_param(jobs, "snapshots", 64))
            mode_term = 1.0 / math.sqrt(float(modes))
            snapshot_term = 1.0 / math.sqrt(float(snapshots))
            correction = min(0.2, 0.05 * mode_term + 0.04 * snapshot_term)
            return _ram_apply(xp, spot, strike, t, vol, r, q, option_type, correction)

        if method == "pathwise_derivative_delta":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)
            paths = int(self._assert_uniform_param(jobs, "paths", 20000))
            B = len(jobs)

            Z = _mc_random_matrix(xp, jobs, (B, paths))

            drift = (r - q - 0.5 * vol * vol) * t
            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            S_T = spot[:, None] * xp.exp(drift[:, None] + vol[:, None] * sqrt_t[:, None] * Z)

            contrib = S_T / spot[:, None]
            contrib = xp.where(
                option_type[:, None] == QK_CALL,
                xp.where(S_T > strike[:, None], contrib, 0.0),
                xp.where(S_T < strike[:, None], -contrib, 0.0),
            )

            disc = xp.exp(-r * t)
            qf = xp.exp(-q * t)
            delta = disc * xp.mean(contrib, axis=1)
            return xp.clip(delta, -qf, qf)

        if method == "likelihood_ratio_delta":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)
            paths = int(self._assert_uniform_param(jobs, "paths", 20000))
            weight_clip = float(self._assert_uniform_param(jobs, "weight_clip", 6.0))
            B = len(jobs)

            Z = _mc_random_matrix(xp, jobs, (B, paths))

            drift = (r - q - 0.5 * vol * vol) * t
            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            S_T = spot[:, None] * xp.exp(drift[:, None] + vol[:, None] * sqrt_t[:, None] * Z)

            payoff = xp.where(
                option_type[:, None] == QK_CALL,
                xp.maximum(S_T - strike[:, None], 0.0),
                xp.maximum(strike[:, None] - S_T, 0.0),
            )

            score = Z / (vol[:, None] * sqrt_t[:, None] * spot[:, None])
            score = xp.clip(score, -weight_clip, weight_clip)

            disc = xp.exp(-r * t)
            qf = xp.exp(-q * t)
            delta = disc * xp.mean(payoff * score, axis=1)
            return xp.clip(delta, -qf, qf)

        if method == "aad_delta":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)
            regularization = float(self._assert_uniform_param(jobs, "regularization", 1e-6))

            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            vol_sqrt_t = vol * sqrt_t
            d1 = (xp.log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / xp.maximum(vol_sqrt_t, eps)
            Nd1 = _norm_cdf(xp, d1)
            qf = xp.exp(-q * t)

            delta = xp.where(option_type == QK_CALL, qf * Nd1, qf * (Nd1 - 1.0))

            reg_strength = regularization / (1.0 + regularization)
            atm_prior = xp.where(option_type == QK_CALL, 0.5 * qf, -0.5 * qf)
            delta = delta * (1.0 - reg_strength) + atm_prior * reg_strength
            return xp.clip(delta, -qf, qf)

        if method == "neural_sde_calibration_price":
            spot = self._column(xp, jobs, "spot")
            strike = self._column(xp, jobs, "strike")
            t = self._column(xp, jobs, "t")
            vol = self._column(xp, jobs, "vol")
            r = self._column(xp, jobs, "r")
            q = self._column(xp, jobs, "q")
            option_type = self._column(xp, jobs, "option_type", dtype=np.int32)
            target_iv = float(self._assert_uniform_param(jobs, "target_implied_vol", 0.2))

            # Use target_implied_vol as effective vol (converged state)
            eff_vol = xp.full_like(vol, target_iv)
            eff_vol = xp.maximum(eff_vol, 0.01)

            sqrt_t = xp.sqrt(xp.maximum(t, 0.0))
            vol_sqrt_t = eff_vol * sqrt_t
            d1 = (xp.log(spot / strike) + (r - q + 0.5 * eff_vol * eff_vol) * t) / xp.maximum(vol_sqrt_t, eps)
            d2 = d1 - vol_sqrt_t
            df = xp.exp(-r * t)
            qf = xp.exp(-q * t)

            call = xp.maximum(0.0, spot * qf * _norm_cdf(xp, d1) - strike * df * _norm_cdf(xp, d2))
            put = call - spot * qf + strike * df
            return xp.where(option_type == QK_CALL, call, put)

        raise RuntimeError(f"Vectorized backend not implemented for method: {method}")
