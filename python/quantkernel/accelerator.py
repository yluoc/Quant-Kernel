"""Rule-based batch acceleration helpers for QuantKernel.

This module operationalizes practical acceleration rules:
- Use vectorized math for light closed-form formulas.
- Use optional CuPy for GPU vectorization when available and batch is large.
- Use threaded execution for heavy scalar C++ methods when batching many jobs.
"""

from __future__ import annotations

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


def _norm_cdf(xp, x):
    # Abramowitz & Stegun 7.1.26 (same approximation family as C++ core).
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = xp.where(x < 0.0, -1.0, 1.0)
    ax = xp.abs(x) * (2.0 ** -0.5)
    t = 1.0 / (1.0 + p * ax)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * xp.exp(-ax * ax)
    return 0.5 * (1.0 + sign * y)


def _norm_pdf(xp, x):
    return 0.3989422804014327 * xp.exp(-0.5 * x * x)


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
    }

    _HIGH_PARALLEL_METHODS = {
        "heston_price_cf",
        "variance_gamma_price_cf",
        "adi_douglas_price",
        "adi_craig_sneyd_price",
        "adi_hundsdorfer_verwer_price",
    }

    _MEDIUM_PARALLEL_METHODS = {
        "merton_jump_diffusion_price",
        "crr_price",
        "jarrow_rudd_price",
        "tian_price",
        "leisen_reimer_price",
        "trinomial_tree_price",
        "derman_kani_const_local_vol_price",
        "explicit_fd_price",
        "implicit_fd_price",
        "crank_nicolson_price",
        "psor_price",
        "standard_monte_carlo_price",
        "euler_maruyama_price",
        "milstein_price",
        "longstaff_schwartz_price",
        "quasi_monte_carlo_sobol_price",
        "quasi_monte_carlo_halton_price",
        "multilevel_monte_carlo_price",
        "importance_sampling_price",
        "control_variates_price",
        "antithetic_variates_price",
        "stratified_sampling_price",
    }

    _GPU_THRESHOLDS = {
        "black_scholes_merton_price": 20_000,
        "black76_price": 20_000,
        "bachelier_price": 20_000,
        "sabr_hagan_lognormal_iv": 15_000,
        "sabr_hagan_black76_price": 15_000,
        "dupire_local_vol": 30_000,
    }

    _THREAD_THRESHOLDS = {
        "high": 64,
        "medium": 128,
        "default": 256,
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

        Strategies: ``gpu_vectorized``, ``cpu_vectorized``, ``threaded``, ``sequential``.
        """
        if method in self._VECTORIZED_METHODS:
            if self.backend in {"auto", "gpu"} and self.gpu_available:
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
        if self.backend == "gpu" and method in self._VECTORIZED_METHODS and not self.gpu_available:
            raise RuntimeError("backend='gpu' requested, but CuPy is not available")
        n = len(jobs)
        if n == 0:
            return np.empty((0,), dtype=np.float64)

        strategy = self.suggest_strategy(method, n)

        if strategy == "gpu_vectorized":
            if self.backend == "gpu" and not self.gpu_available:
                raise RuntimeError("backend='gpu' requested, but CuPy is not available")
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

        raise RuntimeError(f"Vectorized backend not implemented for method: {method}")
