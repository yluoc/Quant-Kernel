"""QuantKernel high-level Python API."""

import ctypes as ct
import math

import numpy as np

from ._abi import QK_CALL, QK_PUT
from ._loader import load_library


class QuantKernel:
    """Thin wrapper for the QuantKernel shared library."""

    __slots__ = ('_lib', '_fn_cache', '_accel_cache', '_native_batch')

    CALL = QK_CALL
    PUT = QK_PUT

    def __init__(self):
        self._lib = load_library()
        self._fn_cache = {}
        self._accel_cache = {}
        try:
            from . import _native_batch  # pylint: disable=import-outside-toplevel
            self._native_batch = _native_batch
        except Exception:
            self._native_batch = None

    def _get_fn(self, fn_name: str):
        fn = self._fn_cache.get(fn_name)
        if fn is None:
            fn = getattr(self._lib, fn_name)
            self._fn_cache[fn_name] = fn
        return fn

    def _call_checked(self, fn_name: str, *args) -> float:
        out = self._get_fn(fn_name)(*args)
        if math.isnan(out):
            raise ValueError(f"{fn_name} returned NaN; check inputs.")
        return out

    @staticmethod
    def _as_f64_array(x, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D array-like")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _as_i32_array(x, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D array-like")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _as_u64_array(x, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=np.uint64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D array-like")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _check_same_length(n: int, **arrays) -> None:
        for name, arr in arrays.items():
            if arr.shape[0] != n:
                raise ValueError(f"{name} length mismatch: expected {n}, got {arr.shape[0]}")

    @property
    def native_batch_available(self) -> bool:
        return self._native_batch is not None

    def _call_batch_ctypes(self, fn_name: str, *arrays) -> np.ndarray:
        n = arrays[0].shape[0]
        out = np.empty(n, dtype=np.float64)
        fn = self._get_fn(fn_name)
        c_double_p = ct.POINTER(ct.c_double)
        c_int32_p = ct.POINTER(ct.c_int32)
        c_uint64_p = ct.POINTER(ct.c_uint64)
        args = []
        for arr in arrays:
            if arr.dtype == np.float64:
                args.append(arr.ctypes.data_as(c_double_p))
            elif arr.dtype == np.int32:
                args.append(arr.ctypes.data_as(c_int32_p))
            elif arr.dtype == np.uint64:
                args.append(arr.ctypes.data_as(c_uint64_p))
            else:
                raise TypeError(f"Unsupported dtype for batch call: {arr.dtype}")
        rc = fn(*args, int(n), out.ctypes.data_as(c_double_p))
        if rc != 0:
            raise ValueError(f"{fn_name} failed with error code {rc}")
        return out

    def get_accelerator(self, backend: str = "auto", max_workers: int | None = None):
        """Return a cached QuantAccelerator bound to this QuantKernel."""
        key = (backend, max_workers)
        acc = self._accel_cache.get(key)
        if acc is not None:
            return acc

        from .accelerator import QuantAccelerator

        acc = QuantAccelerator(qk=self, backend=backend, max_workers=max_workers)
        self._accel_cache[key] = acc
        return acc

    def price_batch(
        self,
        method: str,
        jobs,
        backend: str = "auto",
        max_workers: int | None = None,
    ):
        """Batch-price using rule-based acceleration.

        Parameters
        ----------
        method:
            QuantKernel method name (e.g. ``"black_scholes_merton_price"``).
        jobs:
            Sequence of dict-like kwargs, one pricing call per element.
        backend:
            ``"auto"``, ``"cpu"``, or ``"gpu"``.
        max_workers:
            Optional thread count override for threaded strategies.
        """
        return self.get_accelerator(backend=backend, max_workers=max_workers).price_batch(method, jobs)

    def _tree_price(
        self, fn_name: str, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._get_fn(fn_name)(
            spot, strike, t, vol, r, q, option_type, steps, 1 if american_style else 0
        )

    def black_scholes_merton_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float, option_type: int
    ) -> float:
        return self._get_fn("qk_cf_black_scholes_merton_price")(
            spot, strike, t, vol, r, q, option_type
        )

    def black76_price(
        self, forward: float, strike: float, t: float, vol: float, r: float, option_type: int
    ) -> float:
        return self._get_fn("qk_cf_black76_price")(forward, strike, t, vol, r, option_type)

    def bachelier_price(
        self, forward: float, strike: float, t: float, normal_vol: float, r: float, option_type: int
    ) -> float:
        return self._get_fn("qk_cf_bachelier_price")(forward, strike, t, normal_vol, r, option_type)

    def black_scholes_merton_price_batch(
        self, spot, strike, t, vol, r, q, option_type
    ) -> np.ndarray:
        """Vectorized native batch call (C++ core) for Black-Scholes-Merton."""
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot)
        if self._native_batch is not None:
            return self._native_batch.black_scholes_merton_price_batch(s, k, tau, sigma, rr, qq, ot)
        return self._call_batch_ctypes(
            "qk_cf_black_scholes_merton_price_batch", s, k, tau, sigma, rr, qq, ot
        )

    def black76_price_batch(self, forward, strike, t, vol, r, option_type) -> np.ndarray:
        """Vectorized native batch call (C++ core) for Black-76."""
        fwd = self._as_f64_array(forward, "forward")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        ot = self._as_i32_array(option_type, "option_type")
        n = fwd.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, option_type=ot)
        if self._native_batch is not None:
            return self._native_batch.black76_price_batch(fwd, k, tau, sigma, rr, ot)
        return self._call_batch_ctypes("qk_cf_black76_price_batch", fwd, k, tau, sigma, rr, ot)

    def bachelier_price_batch(self, forward, strike, t, normal_vol, r, option_type) -> np.ndarray:
        """Vectorized native batch call (C++ core) for Bachelier."""
        fwd = self._as_f64_array(forward, "forward")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        nvol = self._as_f64_array(normal_vol, "normal_vol")
        rr = self._as_f64_array(r, "r")
        ot = self._as_i32_array(option_type, "option_type")
        n = fwd.shape[0]
        self._check_same_length(n, strike=k, t=tau, normal_vol=nvol, r=rr, option_type=ot)
        if self._native_batch is not None:
            return self._native_batch.bachelier_price_batch(fwd, k, tau, nvol, rr, ot)
        return self._call_batch_ctypes("qk_cf_bachelier_price_batch", fwd, k, tau, nvol, rr, ot)

    def heston_price_cf(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta: float, sigma: float, rho: float,
        option_type: int, integration_steps: int = 1024, integration_limit: float = 120.0
    ) -> float:
        return self._call_checked(
            "qk_cf_heston_price_cf",
            spot, strike, t, r, q,
            v0, kappa, theta, sigma, rho,
            option_type, integration_steps, integration_limit
        )

    def merton_jump_diffusion_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        jump_intensity: float, jump_mean: float, jump_vol: float, max_terms: int, option_type: int
    ) -> float:
        return self._call_checked(
            "qk_cf_merton_jump_diffusion_price",
            spot, strike, t, vol, r, q,
            jump_intensity, jump_mean, jump_vol, max_terms, option_type
        )

    def variance_gamma_price_cf(
        self, spot: float, strike: float, t: float, r: float, q: float,
        sigma: float, theta: float, nu: float, option_type: int,
        integration_steps: int = 1024, integration_limit: float = 120.0
    ) -> float:
        return self._call_checked(
            "qk_cf_variance_gamma_price_cf",
            spot, strike, t, r, q,
            sigma, theta, nu, option_type,
            integration_steps, integration_limit
        )

    def sabr_hagan_lognormal_iv(
        self, forward: float, strike: float, t: float, alpha: float, beta: float, rho: float, nu: float
    ) -> float:
        return self._call_checked(
            "qk_cf_sabr_hagan_lognormal_iv",
            forward, strike, t, alpha, beta, rho, nu
        )

    def sabr_hagan_black76_price(
        self, forward: float, strike: float, t: float, r: float,
        alpha: float, beta: float, rho: float, nu: float, option_type: int
    ) -> float:
        return self._call_checked(
            "qk_cf_sabr_hagan_black76_price",
            forward, strike, t, r, alpha, beta, rho, nu, option_type
        )

    def dupire_local_vol(
        self, strike: float, t: float, call_price: float, dC_dT: float, dC_dK: float, d2C_dK2: float,
        r: float, q: float
    ) -> float:
        return self._call_checked(
            "qk_cf_dupire_local_vol",
            strike, t, call_price, dC_dT, dC_dK, d2C_dK2, r, q
        )

    def heston_price_cf_batch(
        self, spot, strike, t, r, q, v0, kappa, theta, sigma, rho,
        option_type, integration_steps, integration_limit
    ) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        _v0 = self._as_f64_array(v0, "v0")
        _kappa = self._as_f64_array(kappa, "kappa")
        _theta = self._as_f64_array(theta, "theta")
        _sigma = self._as_f64_array(sigma, "sigma")
        _rho = self._as_f64_array(rho, "rho")
        ot = self._as_i32_array(option_type, "option_type")
        _is = self._as_i32_array(integration_steps, "integration_steps")
        _il = self._as_f64_array(integration_limit, "integration_limit")
        n = s.shape[0]
        self._check_same_length(
            n, strike=k, t=tau, r=rr, q=qq, v0=_v0, kappa=_kappa, theta=_theta,
            sigma=_sigma, rho=_rho, option_type=ot, integration_steps=_is, integration_limit=_il
        )
        if self._native_batch is not None and hasattr(self._native_batch, "heston_price_cf_batch"):
            return self._native_batch.heston_price_cf_batch(s, k, tau, rr, qq, _v0, _kappa, _theta, _sigma, _rho, ot, _is, _il)
        return self._call_batch_ctypes(
            "qk_cf_heston_price_cf_batch", s, k, tau, rr, qq, _v0, _kappa, _theta, _sigma, _rho, ot, _is, _il
        )

    def merton_jump_diffusion_price_batch(
        self, spot, strike, t, vol, r, q, jump_intensity, jump_mean, jump_vol, max_terms, option_type
    ) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        _ji = self._as_f64_array(jump_intensity, "jump_intensity")
        _jm = self._as_f64_array(jump_mean, "jump_mean")
        _jv = self._as_f64_array(jump_vol, "jump_vol")
        _mt = self._as_i32_array(max_terms, "max_terms")
        ot = self._as_i32_array(option_type, "option_type")
        n = s.shape[0]
        self._check_same_length(
            n, strike=k, t=tau, vol=sigma, r=rr, q=qq,
            jump_intensity=_ji, jump_mean=_jm, jump_vol=_jv, max_terms=_mt, option_type=ot
        )
        if self._native_batch is not None and hasattr(self._native_batch, "merton_jump_diffusion_price_batch"):
            return self._native_batch.merton_jump_diffusion_price_batch(s, k, tau, sigma, rr, qq, _ji, _jm, _jv, _mt, ot)
        return self._call_batch_ctypes(
            "qk_cf_merton_jump_diffusion_price_batch", s, k, tau, sigma, rr, qq, _ji, _jm, _jv, _mt, ot
        )

    def variance_gamma_price_cf_batch(
        self, spot, strike, t, r, q, sigma, theta, nu, option_type, integration_steps, integration_limit
    ) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        _sigma = self._as_f64_array(sigma, "sigma")
        _theta = self._as_f64_array(theta, "theta")
        _nu = self._as_f64_array(nu, "nu")
        ot = self._as_i32_array(option_type, "option_type")
        _is = self._as_i32_array(integration_steps, "integration_steps")
        _il = self._as_f64_array(integration_limit, "integration_limit")
        n = s.shape[0]
        self._check_same_length(
            n, strike=k, t=tau, r=rr, q=qq, sigma=_sigma, theta=_theta, nu=_nu,
            option_type=ot, integration_steps=_is, integration_limit=_il
        )
        if self._native_batch is not None and hasattr(self._native_batch, "variance_gamma_price_cf_batch"):
            return self._native_batch.variance_gamma_price_cf_batch(s, k, tau, rr, qq, _sigma, _theta, _nu, ot, _is, _il)
        return self._call_batch_ctypes(
            "qk_cf_variance_gamma_price_cf_batch", s, k, tau, rr, qq, _sigma, _theta, _nu, ot, _is, _il
        )

    def sabr_hagan_lognormal_iv_batch(
        self, forward, strike, t, alpha, beta, rho, nu
    ) -> np.ndarray:
        fwd = self._as_f64_array(forward, "forward")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        _alpha = self._as_f64_array(alpha, "alpha")
        _beta = self._as_f64_array(beta, "beta")
        _rho = self._as_f64_array(rho, "rho")
        _nu = self._as_f64_array(nu, "nu")
        n = fwd.shape[0]
        self._check_same_length(n, strike=k, t=tau, alpha=_alpha, beta=_beta, rho=_rho, nu=_nu)
        if self._native_batch is not None and hasattr(self._native_batch, "sabr_hagan_lognormal_iv_batch"):
            return self._native_batch.sabr_hagan_lognormal_iv_batch(fwd, k, tau, _alpha, _beta, _rho, _nu)
        return self._call_batch_ctypes(
            "qk_cf_sabr_hagan_lognormal_iv_batch", fwd, k, tau, _alpha, _beta, _rho, _nu
        )

    def sabr_hagan_black76_price_batch(
        self, forward, strike, t, r, alpha, beta, rho, nu, option_type
    ) -> np.ndarray:
        fwd = self._as_f64_array(forward, "forward")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        rr = self._as_f64_array(r, "r")
        _alpha = self._as_f64_array(alpha, "alpha")
        _beta = self._as_f64_array(beta, "beta")
        _rho = self._as_f64_array(rho, "rho")
        _nu = self._as_f64_array(nu, "nu")
        ot = self._as_i32_array(option_type, "option_type")
        n = fwd.shape[0]
        self._check_same_length(n, strike=k, t=tau, r=rr, alpha=_alpha, beta=_beta, rho=_rho, nu=_nu, option_type=ot)
        if self._native_batch is not None and hasattr(self._native_batch, "sabr_hagan_black76_price_batch"):
            return self._native_batch.sabr_hagan_black76_price_batch(fwd, k, tau, rr, _alpha, _beta, _rho, _nu, ot)
        return self._call_batch_ctypes(
            "qk_cf_sabr_hagan_black76_price_batch", fwd, k, tau, rr, _alpha, _beta, _rho, _nu, ot
        )

    def dupire_local_vol_batch(
        self, strike, t, call_price, dC_dT, dC_dK, d2C_dK2, r, q
    ) -> np.ndarray:
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        _cp = self._as_f64_array(call_price, "call_price")
        _dt = self._as_f64_array(dC_dT, "dC_dT")
        _dk = self._as_f64_array(dC_dK, "dC_dK")
        _d2k = self._as_f64_array(d2C_dK2, "d2C_dK2")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        n = k.shape[0]
        self._check_same_length(n, t=tau, call_price=_cp, dC_dT=_dt, dC_dK=_dk, d2C_dK2=_d2k, r=rr, q=qq)
        if self._native_batch is not None and hasattr(self._native_batch, "dupire_local_vol_batch"):
            return self._native_batch.dupire_local_vol_batch(k, tau, _cp, _dt, _dk, _d2k, rr, qq)
        return self._call_batch_ctypes(
            "qk_cf_dupire_local_vol_batch", k, tau, _cp, _dt, _dk, _d2k, rr, qq
        )

    # --- Fourier transform methods ---

    def carr_madan_fft_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, grid_size: int = 4096, eta: float = 0.25, alpha: float = 1.5
    ) -> float:
        return self._call_checked(
            "qk_ftm_carr_madan_fft_price",
            spot, strike, t, vol, r, q, option_type, grid_size, eta, alpha
        )

    def carr_madan_fft_price_batch(
        self, spot, strike, t, vol, r, q, option_type, grid_size, eta, alpha
    ) -> np.ndarray:
        """Vectorized native batch call (C++ core) for Carr-Madan FFT."""
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        gs = self._as_i32_array(grid_size, "grid_size")
        ee = self._as_f64_array(eta, "eta")
        aa = self._as_f64_array(alpha, "alpha")
        n = s.shape[0]
        self._check_same_length(
            n, strike=k, t=tau, vol=sigma, r=rr, q=qq,
            option_type=ot, grid_size=gs, eta=ee, alpha=aa
        )
        if self._native_batch is not None and hasattr(self._native_batch, "carr_madan_fft_price_batch"):
            return self._native_batch.carr_madan_fft_price_batch(s, k, tau, sigma, rr, qq, ot, gs, ee, aa)
        return self._call_batch_ctypes(
            "qk_ftm_carr_madan_fft_price_batch", s, k, tau, sigma, rr, qq, ot, gs, ee, aa
        )

    def cos_method_fang_oosterlee_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, n_terms: int = 256, truncation_width: float = 10.0
    ) -> float:
        return self._call_checked(
            "qk_ftm_cos_fang_oosterlee_price",
            spot, strike, t, vol, r, q, option_type, n_terms, truncation_width
        )

    def fractional_fft_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, grid_size: int = 256, eta: float = 0.25,
        lambda_: float = 0.05, alpha: float = 1.5
    ) -> float:
        return self._call_checked(
            "qk_ftm_fractional_fft_price",
            spot, strike, t, vol, r, q, option_type, grid_size, eta, lambda_, alpha
        )

    def lewis_fourier_inversion_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, integration_steps: int = 4096, integration_limit: float = 300.0
    ) -> float:
        return self._call_checked(
            "qk_ftm_lewis_fourier_inversion_price",
            spot, strike, t, vol, r, q, option_type, integration_steps, integration_limit
        )

    def hilbert_transform_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, integration_steps: int = 4096, integration_limit: float = 300.0
    ) -> float:
        return self._call_checked(
            "qk_ftm_hilbert_transform_price",
            spot, strike, t, vol, r, q, option_type, integration_steps, integration_limit
        )

    def cos_method_fang_oosterlee_price_batch(
        self, spot, strike, t, vol, r, q, option_type, n_terms, truncation_width
    ) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        nt = self._as_i32_array(n_terms, "n_terms")
        tw = self._as_f64_array(truncation_width, "truncation_width")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, n_terms=nt, truncation_width=tw)
        if self._native_batch is not None and hasattr(self._native_batch, "cos_fang_oosterlee_price_batch"):
            return self._native_batch.cos_fang_oosterlee_price_batch(s, k, tau, sigma, rr, qq, ot, nt, tw)
        return self._call_batch_ctypes("qk_ftm_cos_fang_oosterlee_price_batch", s, k, tau, sigma, rr, qq, ot, nt, tw)

    def fractional_fft_price_batch(
        self, spot, strike, t, vol, r, q, option_type, grid_size, eta, lambda_, alpha
    ) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        gs = self._as_i32_array(grid_size, "grid_size")
        ee = self._as_f64_array(eta, "eta")
        ll = self._as_f64_array(lambda_, "lambda_")
        aa = self._as_f64_array(alpha, "alpha")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, grid_size=gs, eta=ee, lambda_=ll, alpha=aa)
        if self._native_batch is not None and hasattr(self._native_batch, "fractional_fft_price_batch"):
            return self._native_batch.fractional_fft_price_batch(s, k, tau, sigma, rr, qq, ot, gs, ee, ll, aa)
        return self._call_batch_ctypes("qk_ftm_fractional_fft_price_batch", s, k, tau, sigma, rr, qq, ot, gs, ee, ll, aa)

    def lewis_fourier_inversion_price_batch(
        self, spot, strike, t, vol, r, q, option_type, integration_steps, integration_limit
    ) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        _is = self._as_i32_array(integration_steps, "integration_steps")
        _il = self._as_f64_array(integration_limit, "integration_limit")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, integration_steps=_is, integration_limit=_il)
        if self._native_batch is not None and hasattr(self._native_batch, "lewis_fourier_inversion_price_batch"):
            return self._native_batch.lewis_fourier_inversion_price_batch(s, k, tau, sigma, rr, qq, ot, _is, _il)
        return self._call_batch_ctypes("qk_ftm_lewis_fourier_inversion_price_batch", s, k, tau, sigma, rr, qq, ot, _is, _il)

    def hilbert_transform_price_batch(
        self, spot, strike, t, vol, r, q, option_type, integration_steps, integration_limit
    ) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        _is = self._as_i32_array(integration_steps, "integration_steps")
        _il = self._as_f64_array(integration_limit, "integration_limit")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, integration_steps=_is, integration_limit=_il)
        if self._native_batch is not None and hasattr(self._native_batch, "hilbert_transform_price_batch"):
            return self._native_batch.hilbert_transform_price_batch(s, k, tau, sigma, rr, qq, ot, _is, _il)
        return self._call_batch_ctypes("qk_ftm_hilbert_transform_price_batch", s, k, tau, sigma, rr, qq, ot, _is, _il)

    # --- Integral quadrature methods ---

    def gauss_hermite_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, n_points: int = 128
    ) -> float:
        return self._call_checked(
            "qk_iqm_gauss_hermite_price",
            spot, strike, t, vol, r, q, option_type, n_points
        )

    def gauss_laguerre_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, n_points: int = 64
    ) -> float:
        return self._call_checked(
            "qk_iqm_gauss_laguerre_price",
            spot, strike, t, vol, r, q, option_type, n_points
        )

    def gauss_legendre_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, n_points: int = 128, integration_limit: float = 200.0
    ) -> float:
        return self._call_checked(
            "qk_iqm_gauss_legendre_price",
            spot, strike, t, vol, r, q, option_type, n_points, integration_limit
        )

    def adaptive_quadrature_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, abs_tol: float = 1e-9, rel_tol: float = 1e-8,
        max_depth: int = 14, integration_limit: float = 200.0
    ) -> float:
        return self._call_checked(
            "qk_iqm_adaptive_quadrature_price",
            spot, strike, t, vol, r, q, option_type,
            abs_tol, rel_tol, max_depth, integration_limit
        )

    # --- Regression approximation methods ---

    def polynomial_chaos_expansion_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, polynomial_order: int = 4, quadrature_points: int = 32
    ) -> float:
        return self._call_checked(
            "qk_ram_polynomial_chaos_expansion_price",
            spot, strike, t, vol, r, q, option_type, polynomial_order, quadrature_points
        )

    def radial_basis_function_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, centers: int = 24, rbf_shape: float = 1.0, ridge: float = 1e-4
    ) -> float:
        return self._call_checked(
            "qk_ram_radial_basis_function_price",
            spot, strike, t, vol, r, q, option_type, centers, rbf_shape, ridge
        )

    def sparse_grid_collocation_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, level: int = 3, nodes_per_dim: int = 9
    ) -> float:
        return self._call_checked(
            "qk_ram_sparse_grid_collocation_price",
            spot, strike, t, vol, r, q, option_type, level, nodes_per_dim
        )

    def proper_orthogonal_decomposition_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, modes: int = 8, snapshots: int = 64
    ) -> float:
        return self._call_checked(
            "qk_ram_proper_orthogonal_decomposition_price",
            spot, strike, t, vol, r, q, option_type, modes, snapshots
        )

    # --- Adjoint Greeks methods ---

    def pathwise_derivative_delta(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int = 20000, seed: int = 42
    ) -> float:
        return self._call_checked(
            "qk_agm_pathwise_derivative_delta",
            spot, strike, t, vol, r, q, option_type, paths, seed
        )

    def likelihood_ratio_delta(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int = 20000, seed: int = 42, weight_clip: float = 6.0
    ) -> float:
        return self._call_checked(
            "qk_agm_likelihood_ratio_delta",
            spot, strike, t, vol, r, q, option_type, paths, seed, weight_clip
        )

    def aad_delta(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, tape_steps: int = 64, regularization: float = 1e-6
    ) -> float:
        return self._call_checked(
            "qk_agm_aad_delta",
            spot, strike, t, vol, r, q, option_type, tape_steps, regularization
        )

    # --- Machine learning methods ---

    def deep_bsde_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, time_steps: int = 50, hidden_width: int = 64,
        training_epochs: int = 400, learning_rate: float = 5e-3
    ) -> float:
        return self._call_checked(
            "qk_mlm_deep_bsde_price",
            spot, strike, t, vol, r, q, option_type,
            time_steps, hidden_width, training_epochs, learning_rate
        )

    def pinns_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, collocation_points: int = 5000, boundary_points: int = 400,
        epochs: int = 300, loss_balance: float = 1.0
    ) -> float:
        return self._call_checked(
            "qk_mlm_pinns_price",
            spot, strike, t, vol, r, q, option_type,
            collocation_points, boundary_points, epochs, loss_balance
        )

    def deep_hedging_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, rehedge_steps: int = 26, risk_aversion: float = 0.5,
        scenarios: int = 20000, seed: int = 42
    ) -> float:
        return self._call_checked(
            "qk_mlm_deep_hedging_price",
            spot, strike, t, vol, r, q, option_type,
            rehedge_steps, risk_aversion, scenarios, seed
        )

    def neural_sde_calibration_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, target_implied_vol: float = 0.2,
        calibration_steps: int = 200, regularization: float = 1e-3
    ) -> float:
        return self._call_checked(
            "qk_mlm_neural_sde_calibration_price",
            spot, strike, t, vol, r, q, option_type,
            target_implied_vol, calibration_steps, regularization
        )

    def crr_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_crr_price", spot, strike, t, vol, r, q, option_type, steps, american_style
        )

    def crr_price_batch(
        self, spot, strike, t, vol, r, q, option_type, steps, american_style
    ) -> np.ndarray:
        return self._tree_price_batch("qk_tlm_crr_price_batch", "crr_price_batch",
                                      spot, strike, t, vol, r, q, option_type, steps, american_style)

    def jarrow_rudd_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_jarrow_rudd_price", spot, strike, t, vol, r, q, option_type, steps, american_style
        )

    def tian_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_tian_price", spot, strike, t, vol, r, q, option_type, steps, american_style
        )

    def leisen_reimer_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_leisen_reimer_price",
            spot, strike, t, vol, r, q, option_type, steps, american_style
        )

    def trinomial_tree_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_trinomial_tree_price",
            spot, strike, t, vol, r, q, option_type, steps, american_style
        )

    def derman_kani_const_local_vol_price(
        self, spot: float, strike: float, t: float, local_vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_derman_kani_const_local_vol_price",
            spot, strike, t, local_vol, r, q, option_type, steps, american_style
        )

    def _tree_price_batch(self, c_fn_name: str, native_fn_name: str,
                          spot, strike, t, vol, r, q, option_type, steps, american_style) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        st = self._as_i32_array(steps, "steps")
        am = self._as_i32_array(american_style, "american_style")
        n = s.shape[0]
        self._check_same_length(
            n, strike=k, t=tau, vol=sigma, r=rr, q=qq,
            option_type=ot, steps=st, american_style=am
        )
        if self._native_batch is not None and hasattr(self._native_batch, native_fn_name):
            return getattr(self._native_batch, native_fn_name)(s, k, tau, sigma, rr, qq, ot, st, am)
        return self._call_batch_ctypes(c_fn_name, s, k, tau, sigma, rr, qq, ot, st, am)

    def jarrow_rudd_price_batch(self, spot, strike, t, vol, r, q, option_type, steps, american_style) -> np.ndarray:
        return self._tree_price_batch("qk_tlm_jarrow_rudd_price_batch", "jarrow_rudd_price_batch",
                                      spot, strike, t, vol, r, q, option_type, steps, american_style)

    def tian_price_batch(self, spot, strike, t, vol, r, q, option_type, steps, american_style) -> np.ndarray:
        return self._tree_price_batch("qk_tlm_tian_price_batch", "tian_price_batch",
                                      spot, strike, t, vol, r, q, option_type, steps, american_style)

    def leisen_reimer_price_batch(self, spot, strike, t, vol, r, q, option_type, steps, american_style) -> np.ndarray:
        return self._tree_price_batch("qk_tlm_leisen_reimer_price_batch", "leisen_reimer_price_batch",
                                      spot, strike, t, vol, r, q, option_type, steps, american_style)

    def trinomial_tree_price_batch(self, spot, strike, t, vol, r, q, option_type, steps, american_style) -> np.ndarray:
        return self._tree_price_batch("qk_tlm_trinomial_tree_price_batch", "trinomial_tree_price_batch",
                                      spot, strike, t, vol, r, q, option_type, steps, american_style)

    def derman_kani_const_local_vol_price_batch(self, spot, strike, t, local_vol, r, q, option_type, steps, american_style) -> np.ndarray:
        return self._tree_price_batch("qk_tlm_derman_kani_const_local_vol_price_batch", "derman_kani_const_local_vol_price_batch",
                                      spot, strike, t, local_vol, r, q, option_type, steps, american_style)

    # --- Finite Difference methods ---

    def _fdm_price(
        self, fn_name: str, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, time_steps: int, spot_steps: int,
        american_style: bool = False
    ) -> float:
        return self._get_fn(fn_name)(
            spot, strike, t, vol, r, q, option_type,
            time_steps, spot_steps, 1 if american_style else 0
        )

    def explicit_fd_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, time_steps: int, spot_steps: int, american_style: bool = False
    ) -> float:
        return self._fdm_price(
            "qk_fdm_explicit_fd_price", spot, strike, t, vol, r, q,
            option_type, time_steps, spot_steps, american_style
        )

    def implicit_fd_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, time_steps: int, spot_steps: int, american_style: bool = False
    ) -> float:
        return self._fdm_price(
            "qk_fdm_implicit_fd_price", spot, strike, t, vol, r, q,
            option_type, time_steps, spot_steps, american_style
        )

    def crank_nicolson_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, time_steps: int, spot_steps: int, american_style: bool = False
    ) -> float:
        return self._fdm_price(
            "qk_fdm_crank_nicolson_price", spot, strike, t, vol, r, q,
            option_type, time_steps, spot_steps, american_style
        )

    def adi_douglas_price(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta_v: float, sigma: float, rho: float,
        option_type: int, s_steps: int = 40, v_steps: int = 20, time_steps: int = 40
    ) -> float:
        return self._call_checked(
            "qk_fdm_adi_douglas_price",
            spot, strike, t, r, q,
            v0, kappa, theta_v, sigma, rho,
            option_type, s_steps, v_steps, time_steps
        )

    def adi_craig_sneyd_price(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta_v: float, sigma: float, rho: float,
        option_type: int, s_steps: int = 40, v_steps: int = 20, time_steps: int = 40
    ) -> float:
        return self._call_checked(
            "qk_fdm_adi_craig_sneyd_price",
            spot, strike, t, r, q,
            v0, kappa, theta_v, sigma, rho,
            option_type, s_steps, v_steps, time_steps
        )

    def adi_hundsdorfer_verwer_price(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta_v: float, sigma: float, rho: float,
        option_type: int, s_steps: int = 40, v_steps: int = 20, time_steps: int = 40
    ) -> float:
        return self._call_checked(
            "qk_fdm_adi_hundsdorfer_verwer_price",
            spot, strike, t, r, q,
            v0, kappa, theta_v, sigma, rho,
            option_type, s_steps, v_steps, time_steps
        )

    def psor_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, time_steps: int, spot_steps: int,
        omega: float = 1.2, tol: float = 1e-8, max_iter: int = 10000
    ) -> float:
        return self._get_fn("qk_fdm_psor_price")(
            spot, strike, t, vol, r, q, option_type,
            time_steps, spot_steps, omega, tol, max_iter
        )

    # --- Monte Carlo methods ---

    def _mc_price(
        self, fn_name: str, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, *extra
    ) -> float:
        return self._call_checked(fn_name, spot, strike, t, vol, r, q, option_type, *extra)

    def standard_monte_carlo_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int, seed: int = 42
    ) -> float:
        return self._mc_price(
            "qk_mcm_standard_monte_carlo_price",
            spot, strike, t, vol, r, q, option_type, paths, seed
        )

    def standard_monte_carlo_price_batch(
        self, spot, strike, t, vol, r, q, option_type, paths, seed
    ) -> np.ndarray:
        """Vectorized native batch call (C++ core) for standard Monte Carlo."""
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        sd = self._as_u64_array(seed, "seed")
        n = s.shape[0]
        self._check_same_length(
            n, strike=k, t=tau, vol=sigma, r=rr, q=qq,
            option_type=ot, paths=pp, seed=sd
        )
        if self._native_batch is not None and hasattr(self._native_batch, "standard_monte_carlo_price_batch"):
            return self._native_batch.standard_monte_carlo_price_batch(s, k, tau, sigma, rr, qq, ot, pp, sd)
        return self._call_batch_ctypes(
            "qk_mcm_standard_monte_carlo_price_batch", s, k, tau, sigma, rr, qq, ot, pp, sd
        )

    def euler_maruyama_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int, steps: int, seed: int = 42
    ) -> float:
        return self._mc_price(
            "qk_mcm_euler_maruyama_price",
            spot, strike, t, vol, r, q, option_type, paths, steps, seed
        )

    def milstein_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int, steps: int, seed: int = 42
    ) -> float:
        return self._mc_price(
            "qk_mcm_milstein_price",
            spot, strike, t, vol, r, q, option_type, paths, steps, seed
        )

    def longstaff_schwartz_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int, steps: int, seed: int = 42
    ) -> float:
        return self._mc_price(
            "qk_mcm_longstaff_schwartz_price",
            spot, strike, t, vol, r, q, option_type, paths, steps, seed
        )

    def quasi_monte_carlo_sobol_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int
    ) -> float:
        return self._mc_price(
            "qk_mcm_quasi_monte_carlo_sobol_price",
            spot, strike, t, vol, r, q, option_type, paths
        )

    def quasi_monte_carlo_halton_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int
    ) -> float:
        return self._mc_price(
            "qk_mcm_quasi_monte_carlo_halton_price",
            spot, strike, t, vol, r, q, option_type, paths
        )

    def multilevel_monte_carlo_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, base_paths: int, levels: int, base_steps: int, seed: int = 42
    ) -> float:
        return self._mc_price(
            "qk_mcm_multilevel_monte_carlo_price",
            spot, strike, t, vol, r, q, option_type, base_paths, levels, base_steps, seed
        )

    def importance_sampling_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int, shift: float, seed: int = 42
    ) -> float:
        return self._mc_price(
            "qk_mcm_importance_sampling_price",
            spot, strike, t, vol, r, q, option_type, paths, shift, seed
        )

    def control_variates_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int, seed: int = 42
    ) -> float:
        return self._mc_price(
            "qk_mcm_control_variates_price",
            spot, strike, t, vol, r, q, option_type, paths, seed
        )

    def antithetic_variates_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int, seed: int = 42
    ) -> float:
        return self._mc_price(
            "qk_mcm_antithetic_variates_price",
            spot, strike, t, vol, r, q, option_type, paths, seed
        )

    def stratified_sampling_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int, seed: int = 42
    ) -> float:
        return self._mc_price(
            "qk_mcm_stratified_sampling_price",
            spot, strike, t, vol, r, q, option_type, paths, seed
        )

    def euler_maruyama_price_batch(self, spot, strike, t, vol, r, q, option_type, paths, steps, seed) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        st = self._as_i32_array(steps, "steps")
        sd = self._as_u64_array(seed, "seed")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, paths=pp, steps=st, seed=sd)
        if self._native_batch is not None and hasattr(self._native_batch, "euler_maruyama_price_batch"):
            return self._native_batch.euler_maruyama_price_batch(s, k, tau, sigma, rr, qq, ot, pp, st, sd)
        return self._call_batch_ctypes("qk_mcm_euler_maruyama_price_batch", s, k, tau, sigma, rr, qq, ot, pp, st, sd)

    def milstein_price_batch(self, spot, strike, t, vol, r, q, option_type, paths, steps, seed) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        st = self._as_i32_array(steps, "steps")
        sd = self._as_u64_array(seed, "seed")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, paths=pp, steps=st, seed=sd)
        if self._native_batch is not None and hasattr(self._native_batch, "milstein_price_batch"):
            return self._native_batch.milstein_price_batch(s, k, tau, sigma, rr, qq, ot, pp, st, sd)
        return self._call_batch_ctypes("qk_mcm_milstein_price_batch", s, k, tau, sigma, rr, qq, ot, pp, st, sd)

    def longstaff_schwartz_price_batch(self, spot, strike, t, vol, r, q, option_type, paths, steps, seed) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        st = self._as_i32_array(steps, "steps")
        sd = self._as_u64_array(seed, "seed")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, paths=pp, steps=st, seed=sd)
        if self._native_batch is not None and hasattr(self._native_batch, "longstaff_schwartz_price_batch"):
            return self._native_batch.longstaff_schwartz_price_batch(s, k, tau, sigma, rr, qq, ot, pp, st, sd)
        return self._call_batch_ctypes("qk_mcm_longstaff_schwartz_price_batch", s, k, tau, sigma, rr, qq, ot, pp, st, sd)

    def quasi_monte_carlo_sobol_price_batch(self, spot, strike, t, vol, r, q, option_type, paths) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, paths=pp)
        if self._native_batch is not None and hasattr(self._native_batch, "quasi_monte_carlo_sobol_price_batch"):
            return self._native_batch.quasi_monte_carlo_sobol_price_batch(s, k, tau, sigma, rr, qq, ot, pp)
        return self._call_batch_ctypes("qk_mcm_quasi_monte_carlo_sobol_price_batch", s, k, tau, sigma, rr, qq, ot, pp)

    def quasi_monte_carlo_halton_price_batch(self, spot, strike, t, vol, r, q, option_type, paths) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, paths=pp)
        if self._native_batch is not None and hasattr(self._native_batch, "quasi_monte_carlo_halton_price_batch"):
            return self._native_batch.quasi_monte_carlo_halton_price_batch(s, k, tau, sigma, rr, qq, ot, pp)
        return self._call_batch_ctypes("qk_mcm_quasi_monte_carlo_halton_price_batch", s, k, tau, sigma, rr, qq, ot, pp)

    def multilevel_monte_carlo_price_batch(self, spot, strike, t, vol, r, q, option_type, base_paths, levels, base_steps, seed) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        bp = self._as_i32_array(base_paths, "base_paths")
        lv = self._as_i32_array(levels, "levels")
        bs = self._as_i32_array(base_steps, "base_steps")
        sd = self._as_u64_array(seed, "seed")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, base_paths=bp, levels=lv, base_steps=bs, seed=sd)
        if self._native_batch is not None and hasattr(self._native_batch, "multilevel_monte_carlo_price_batch"):
            return self._native_batch.multilevel_monte_carlo_price_batch(s, k, tau, sigma, rr, qq, ot, bp, lv, bs, sd)
        return self._call_batch_ctypes("qk_mcm_multilevel_monte_carlo_price_batch", s, k, tau, sigma, rr, qq, ot, bp, lv, bs, sd)

    def importance_sampling_price_batch(self, spot, strike, t, vol, r, q, option_type, paths, shift, seed) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        sh = self._as_f64_array(shift, "shift")
        sd = self._as_u64_array(seed, "seed")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, paths=pp, shift=sh, seed=sd)
        if self._native_batch is not None and hasattr(self._native_batch, "importance_sampling_price_batch"):
            return self._native_batch.importance_sampling_price_batch(s, k, tau, sigma, rr, qq, ot, pp, sh, sd)
        return self._call_batch_ctypes("qk_mcm_importance_sampling_price_batch", s, k, tau, sigma, rr, qq, ot, pp, sh, sd)

    def control_variates_price_batch(self, spot, strike, t, vol, r, q, option_type, paths, seed) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        sd = self._as_u64_array(seed, "seed")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, paths=pp, seed=sd)
        if self._native_batch is not None and hasattr(self._native_batch, "control_variates_price_batch"):
            return self._native_batch.control_variates_price_batch(s, k, tau, sigma, rr, qq, ot, pp, sd)
        return self._call_batch_ctypes("qk_mcm_control_variates_price_batch", s, k, tau, sigma, rr, qq, ot, pp, sd)

    def antithetic_variates_price_batch(self, spot, strike, t, vol, r, q, option_type, paths, seed) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        sd = self._as_u64_array(seed, "seed")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, paths=pp, seed=sd)
        if self._native_batch is not None and hasattr(self._native_batch, "antithetic_variates_price_batch"):
            return self._native_batch.antithetic_variates_price_batch(s, k, tau, sigma, rr, qq, ot, pp, sd)
        return self._call_batch_ctypes("qk_mcm_antithetic_variates_price_batch", s, k, tau, sigma, rr, qq, ot, pp, sd)

    def stratified_sampling_price_batch(self, spot, strike, t, vol, r, q, option_type, paths, seed) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        sd = self._as_u64_array(seed, "seed")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, paths=pp, seed=sd)
        if self._native_batch is not None and hasattr(self._native_batch, "stratified_sampling_price_batch"):
            return self._native_batch.stratified_sampling_price_batch(s, k, tau, sigma, rr, qq, ot, pp, sd)
        return self._call_batch_ctypes("qk_mcm_stratified_sampling_price_batch", s, k, tau, sigma, rr, qq, ot, pp, sd)
