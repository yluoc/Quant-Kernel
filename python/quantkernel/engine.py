"""QuantKernel high-level Python API."""

import ctypes as ct
import math

import numpy as np

from ._abi import QK_CALL, QK_PUT, QK_ERR_NULL_PTR, QK_ERR_BAD_SIZE, QK_ERR_INVALID_INPUT
from ._loader import load_library


class QKError(RuntimeError):
    """Base exception for QuantKernel errors."""


class QKNullPointerError(QKError):
    """Raised when a null pointer is passed to a batch API."""


class QKBadSizeError(QKError):
    """Raised when an invalid batch size is passed."""


class QKInvalidInputError(QKError):
    """Raised when invalid input parameters are detected."""


_ERROR_CODE_MAP = {
    QK_ERR_NULL_PTR: QKNullPointerError,
    QK_ERR_BAD_SIZE: QKBadSizeError,
    QK_ERR_INVALID_INPUT: QKInvalidInputError,
}


class QuantKernel:
    """High-performance derivative pricing engine.

    Wraps the QuantKernel C++ shared library via ctypes, providing
    40+ pricing algorithms including closed-form, tree/lattice,
    finite difference, Monte Carlo, Fourier, and machine learning methods.

    Each method is available in scalar (single-option) and batch
    (array-in/array-out) variants. Batch methods leverage SIMD and
    OpenMP parallelism in the C++ core for maximum throughput.

    Example::

        qk = QuantKernel()
        price = qk.black_scholes_merton_price(
            spot=100.0, strike=100.0, t=1.0, vol=0.2,
            r=0.05, q=0.0, option_type=qk.CALL
        )

        # Batch pricing
        prices = qk.black_scholes_merton_price_batch(
            spot=[100]*1000, strike=[100]*1000, t=[1.0]*1000,
            vol=[0.2]*1000, r=[0.05]*1000, q=[0.0]*1000,
            option_type=[qk.CALL]*1000
        )
    """

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
            detail = ""
            try:
                raw = self._lib.qk_get_last_error()
                if raw:
                    detail = raw.decode("utf-8", errors="replace")
            except Exception:
                pass
            exc_cls = _ERROR_CODE_MAP.get(rc, QKError)
            msg = f"{fn_name} failed (rc={rc})"
            if detail:
                msg += f": {detail}"
            raise exc_cls(msg)
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
        out = self._get_fn(fn_name)(
            spot, strike, t, vol, r, q, option_type, steps, 1 if american_style else 0
        )
        if math.isnan(out):
            raise ValueError(f"{fn_name} returned NaN; check inputs.")
        return out

    def black_scholes_merton_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float, option_type: int
    ) -> float:
        """Price a European option using the Black-Scholes-Merton formula.

        Parameters
        ----------
        spot : float
            Current underlying asset price.
        strike : float
            Option strike price.
        t : float
            Time to expiration in years.
        vol : float
            Annualized volatility (e.g. 0.2 for 20%).
        r : float
            Risk-free interest rate (continuously compounded).
        q : float
            Continuous dividend yield.
        option_type : int
            ``QK_CALL`` (0) or ``QK_PUT`` (1).

        Returns
        -------
        float
            The BSM option price.
        """
        return self._call_checked(
            "qk_cf_black_scholes_merton_price",
            spot, strike, t, vol, r, q, option_type
        )

    def black76_price(
        self, forward: float, strike: float, t: float, vol: float, r: float, option_type: int
    ) -> float:
        """Price a European option on a forward using the Black-76 model.

        Parameters
        ----------
        forward : float
            Forward price of the underlying.
        strike, t, vol, r : float
            Strike, time to expiry, volatility, risk-free rate.
        option_type : int
            ``QK_CALL`` or ``QK_PUT``.
        """
        return self._call_checked("qk_cf_black76_price", forward, strike, t, vol, r, option_type)

    def bachelier_price(
        self, forward: float, strike: float, t: float, normal_vol: float, r: float, option_type: int
    ) -> float:
        """Price a European option using the Bachelier (normal) model.

        Parameters
        ----------
        forward : float
            Forward price.
        strike, t : float
            Strike price and time to expiry.
        normal_vol : float
            Normal (absolute) volatility.
        r : float
            Risk-free rate.
        option_type : int
            ``QK_CALL`` or ``QK_PUT``.
        """
        return self._call_checked("qk_cf_bachelier_price", forward, strike, t, normal_vol, r, option_type)

    def black_scholes_merton_price_batch(
        self, spot, strike, t, vol, r, q, option_type
    ) -> np.ndarray:
        """Vectorized batch Black-Scholes-Merton pricing (SIMD-accelerated).

        All parameters are 1-D array-like of equal length. Returns an
        ndarray of option prices. Raises ``QKNullPointerError`` or
        ``QKBadSizeError`` on invalid inputs.
        """
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
        """Price using Cox-Ross-Rubinstein binomial tree.

        Parameters
        ----------
        steps : int
            Number of time steps in the tree.
        american_style : bool
            If True, price an American-exercise option.
        """
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

    def derman_kani_call_surface_price(
        self,
        spot: float,
        strike: float,
        t: float,
        r: float,
        q: float,
        option_type: int,
        surface_strikes,
        surface_maturities,
        surface_call_prices,
        steps: int,
        american_style: bool = False,
    ) -> float:
        ks = self._as_f64_array(surface_strikes, "surface_strikes")
        ts = self._as_f64_array(surface_maturities, "surface_maturities")
        if ks.size < 3:
            raise ValueError("surface_strikes must have at least 3 points")
        if ts.size < 2:
            raise ValueError("surface_maturities must have at least 2 points")

        cp = np.asarray(surface_call_prices, dtype=np.float64)
        if cp.ndim == 2:
            if cp.shape != (ts.size, ks.size):
                raise ValueError(
                    f"surface_call_prices shape mismatch: expected {(ts.size, ks.size)}, got {cp.shape}"
                )
            cp_flat = np.ascontiguousarray(cp.reshape(-1))
        elif cp.ndim == 1:
            expected = ts.size * ks.size
            if cp.size != expected:
                raise ValueError(
                    f"surface_call_prices size mismatch: expected {expected}, got {cp.size}"
                )
            cp_flat = np.ascontiguousarray(cp)
        else:
            raise ValueError("surface_call_prices must be 1D or 2D array-like")

        c_double_p = ct.POINTER(ct.c_double)
        out = self._get_fn("qk_tlm_derman_kani_call_surface_price")(
            spot, strike, t, r, q, option_type,
            ks.ctypes.data_as(c_double_p), int(ks.size),
            ts.ctypes.data_as(c_double_p), int(ts.size),
            cp_flat.ctypes.data_as(c_double_p),
            int(steps), 1 if american_style else 0,
        )
        if math.isnan(out):
            raise ValueError("qk_tlm_derman_kani_call_surface_price returned NaN; check inputs.")
        return out

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

    def derman_kani_call_surface_price_batch(
        self,
        spot, strike, t, r, q, option_type,
        surface_strikes, surface_maturities, surface_call_prices,
        steps, american_style,
    ) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        st = self._as_i32_array(steps, "steps")
        am = self._as_i32_array(american_style, "american_style")
        n = s.shape[0]
        self._check_same_length(
            n, strike=k, t=tau, r=rr, q=qq,
            option_type=ot, steps=st, american_style=am
        )

        ks = self._as_f64_array(surface_strikes, "surface_strikes")
        ts = self._as_f64_array(surface_maturities, "surface_maturities")
        cp = np.asarray(surface_call_prices, dtype=np.float64)
        if cp.ndim == 2:
            cp = np.ascontiguousarray(cp.reshape(-1))
        elif cp.ndim == 1:
            cp = np.ascontiguousarray(cp)
        else:
            raise ValueError("surface_call_prices must be 1D or 2D array-like")

        c_double_p = ct.POINTER(ct.c_double)
        out = np.empty(n, dtype=np.float64)
        rc = self._get_fn("qk_tlm_derman_kani_call_surface_price_batch")(
            s.ctypes.data_as(c_double_p),
            k.ctypes.data_as(c_double_p),
            tau.ctypes.data_as(c_double_p),
            rr.ctypes.data_as(c_double_p),
            qq.ctypes.data_as(c_double_p),
            ot.ctypes.data_as(ct.POINTER(ct.c_int32)),
            ks.ctypes.data_as(c_double_p), int(ks.size),
            ts.ctypes.data_as(c_double_p), int(ts.size),
            cp.ctypes.data_as(c_double_p),
            st.ctypes.data_as(ct.POINTER(ct.c_int32)),
            am.ctypes.data_as(ct.POINTER(ct.c_int32)),
            n,
            out.ctypes.data_as(c_double_p),
        )
        if rc != 0:
            detail = ""
            try:
                raw = self._lib.qk_get_last_error()
                if raw:
                    detail = raw.decode("utf-8", errors="replace")
            except Exception:
                pass
            exc_cls = _ERROR_CODE_MAP.get(rc, QKError)
            msg = f"qk_tlm_derman_kani_call_surface_price_batch failed (rc={rc})"
            if detail:
                msg += f": {detail}"
            raise exc_cls(msg)
        return out


    def _fdm_price(
        self, fn_name: str, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, time_steps: int, spot_steps: int,
        american_style: bool = False
    ) -> float:
        out = self._get_fn(fn_name)(
            spot, strike, t, vol, r, q, option_type,
            time_steps, spot_steps, 1 if american_style else 0
        )
        if math.isnan(out):
            raise ValueError(f"{fn_name} returned NaN; check inputs.")
        return out

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
        out = self._get_fn("qk_fdm_psor_price")(
            spot, strike, t, vol, r, q, option_type,
            time_steps, spot_steps, omega, tol, max_iter
        )
        if math.isnan(out):
            raise ValueError("qk_fdm_psor_price returned NaN; check inputs.")
        return out


    def _mc_price(
        self, fn_name: str, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, *extra
    ) -> float:
        return self._call_checked(fn_name, spot, strike, t, vol, r, q, option_type, *extra)

    def standard_monte_carlo_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, paths: int, seed: int = 42
    ) -> float:
        """Price using standard Monte Carlo simulation.

        Parameters
        ----------
        paths : int
            Number of simulated price paths.
        seed : int
            Random number generator seed for reproducibility.
        """
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

    def heston_monte_carlo_price(
        self, spot: float, strike: float, t: float,
        r: float, q: float,
        v0: float, kappa: float, theta: float,
        sigma: float, rho: float,
        option_type: int, paths: int, steps: int, seed: int = 42
    ) -> float:
        """Price a European option under Heston stochastic volatility via Monte Carlo.

        Uses Euler discretization with full truncation for the variance process
        and log-Euler for the spot process.

        Parameters
        ----------
        v0 : float
            Initial variance.
        kappa : float
            Mean-reversion speed.
        theta : float
            Long-run variance.
        sigma : float
            Vol-of-vol.
        rho : float
            Spot-vol correlation (must be in [-1, 1]).
        paths : int
            Number of simulated price paths.
        steps : int
            Number of time steps per path.
        seed : int
            Random number generator seed for reproducibility.
        """
        return self._call_checked(
            "qk_mcm_heston_monte_carlo_price",
            spot, strike, t, r, q, v0, kappa, theta, sigma, rho,
            option_type, paths, steps, seed
        )

    def heston_monte_carlo_price_batch(
        self, spot, strike, t, r, q,
        v0, kappa, theta, sigma, rho,
        option_type, paths, steps, seed
    ) -> np.ndarray:
        """Vectorized native batch call (C++ core) for Heston Monte Carlo."""
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        v0a = self._as_f64_array(v0, "v0")
        ka = self._as_f64_array(kappa, "kappa")
        th = self._as_f64_array(theta, "theta")
        si = self._as_f64_array(sigma, "sigma")
        rh = self._as_f64_array(rho, "rho")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        st = self._as_i32_array(steps, "steps")
        sd = self._as_u64_array(seed, "seed")
        n = s.shape[0]
        self._check_same_length(
            n, strike=k, t=tau, r=rr, q=qq, v0=v0a, kappa=ka,
            theta=th, sigma=si, rho=rh, option_type=ot,
            paths=pp, steps=st, seed=sd
        )
        if self._native_batch is not None and hasattr(self._native_batch, "heston_monte_carlo_price_batch"):
            return self._native_batch.heston_monte_carlo_price_batch(
                s, k, tau, rr, qq, v0a, ka, th, si, rh, ot, pp, st, sd
            )
        return self._call_batch_ctypes(
            "qk_mcm_heston_monte_carlo_price_batch",
            s, k, tau, rr, qq, v0a, ka, th, si, rh, ot, pp, st, sd
        )

    def heston_lr_delta(
        self, spot: float, strike: float, t: float,
        r: float, q: float,
        v0: float, kappa: float, theta: float,
        sigma: float, rho: float,
        option_type: int, paths: int, steps: int, seed: int = 42,
        weight_clip: float = 6.0
    ) -> float:
        """Likelihood-ratio delta estimator for Heston stochastic volatility.

        Uses the score function d/dS_0(log p) = z1_0 / (sqrt(v0) * sqrt(dt) * S_0)
        where z1_0 is the first spot-driving normal draw.  Antithetic pairing is
        used for variance reduction.

        Parameters
        ----------
        weight_clip : float
            Symmetric clipping bound for the LR weight (default 6.0).
        """
        return self._call_checked(
            "qk_mcm_heston_lr_delta",
            spot, strike, t, r, q, v0, kappa, theta, sigma, rho,
            option_type, paths, steps, seed, weight_clip
        )

    def heston_lr_delta_batch(
        self, spot, strike, t, r, q,
        v0, kappa, theta, sigma, rho,
        option_type, paths, steps, seed, weight_clip
    ) -> np.ndarray:
        """Vectorized native batch call (C++ core) for Heston LR delta."""
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        v0a = self._as_f64_array(v0, "v0")
        ka = self._as_f64_array(kappa, "kappa")
        th = self._as_f64_array(theta, "theta")
        si = self._as_f64_array(sigma, "sigma")
        rh = self._as_f64_array(rho, "rho")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        st = self._as_i32_array(steps, "steps")
        sd = self._as_u64_array(seed, "seed")
        wc = self._as_f64_array(weight_clip, "weight_clip")
        n = s.shape[0]
        self._check_same_length(
            n, strike=k, t=tau, r=rr, q=qq, v0=v0a, kappa=ka,
            theta=th, sigma=si, rho=rh, option_type=ot,
            paths=pp, steps=st, seed=sd, weight_clip=wc
        )
        return self._call_batch_ctypes(
            "qk_mcm_heston_lr_delta_batch",
            s, k, tau, rr, qq, v0a, ka, th, si, rh, ot, pp, st, sd, wc
        )

    def local_vol_monte_carlo_price(
        self, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int,
        paths: int, steps: int, seed: int = 42
    ) -> float:
        """Price a European option under local volatility via Monte Carlo.

        Uses Euler discretization with a constant local volatility surface.
        When vol is constant, results match euler_maruyama_price() statistically.

        Parameters
        ----------
        vol : float
            Constant local volatility (sigma_fn(S,t) = vol for all S, t).
        paths : int
            Number of simulated price paths.
        steps : int
            Number of time steps per path.
        seed : int
            Random number generator seed for reproducibility.
        """
        return self._call_checked(
            "qk_mcm_local_vol_monte_carlo_price",
            spot, strike, t, vol, r, q, option_type, paths, steps, seed
        )

    def local_vol_monte_carlo_price_batch(
        self, spot, strike, t, vol, r, q, option_type, paths, steps, seed
    ) -> np.ndarray:
        """Vectorized native batch call (C++ core) for local vol Monte Carlo."""
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
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq,
                                option_type=ot, paths=pp, steps=st, seed=sd)
        return self._call_batch_ctypes(
            "qk_mcm_local_vol_monte_carlo_price_batch",
            s, k, tau, sigma, rr, qq, ot, pp, st, sd
        )

    def explicit_fd_price_batch(self, spot, strike, t, vol, r, q, option_type, time_steps, spot_steps, american_style) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        ts = self._as_i32_array(time_steps, "time_steps")
        ss = self._as_i32_array(spot_steps, "spot_steps")
        am = self._as_i32_array(american_style, "american_style")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, time_steps=ts, spot_steps=ss, american_style=am)
        if self._native_batch is not None and hasattr(self._native_batch, "explicit_fd_price_batch"):
            return self._native_batch.explicit_fd_price_batch(s, k, tau, sigma, rr, qq, ot, ts, ss, am)
        return self._call_batch_ctypes("qk_fdm_explicit_fd_price_batch", s, k, tau, sigma, rr, qq, ot, ts, ss, am)

    def implicit_fd_price_batch(self, spot, strike, t, vol, r, q, option_type, time_steps, spot_steps, american_style) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        ts = self._as_i32_array(time_steps, "time_steps")
        ss = self._as_i32_array(spot_steps, "spot_steps")
        am = self._as_i32_array(american_style, "american_style")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, time_steps=ts, spot_steps=ss, american_style=am)
        if self._native_batch is not None and hasattr(self._native_batch, "implicit_fd_price_batch"):
            return self._native_batch.implicit_fd_price_batch(s, k, tau, sigma, rr, qq, ot, ts, ss, am)
        return self._call_batch_ctypes("qk_fdm_implicit_fd_price_batch", s, k, tau, sigma, rr, qq, ot, ts, ss, am)

    def crank_nicolson_price_batch(self, spot, strike, t, vol, r, q, option_type, time_steps, spot_steps, american_style) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        ts = self._as_i32_array(time_steps, "time_steps")
        ss = self._as_i32_array(spot_steps, "spot_steps")
        am = self._as_i32_array(american_style, "american_style")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, time_steps=ts, spot_steps=ss, american_style=am)
        if self._native_batch is not None and hasattr(self._native_batch, "crank_nicolson_price_batch"):
            return self._native_batch.crank_nicolson_price_batch(s, k, tau, sigma, rr, qq, ot, ts, ss, am)
        return self._call_batch_ctypes("qk_fdm_crank_nicolson_price_batch", s, k, tau, sigma, rr, qq, ot, ts, ss, am)

    def adi_douglas_price_batch(self, spot, strike, t, r, q, v0, kappa, theta_v, sigma, rho, option_type, s_steps, v_steps, time_steps) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        _v0 = self._as_f64_array(v0, "v0")
        _kappa = self._as_f64_array(kappa, "kappa")
        _theta = self._as_f64_array(theta_v, "theta_v")
        _sigma = self._as_f64_array(sigma, "sigma")
        _rho = self._as_f64_array(rho, "rho")
        ot = self._as_i32_array(option_type, "option_type")
        _ss = self._as_i32_array(s_steps, "s_steps")
        _vs = self._as_i32_array(v_steps, "v_steps")
        _ts = self._as_i32_array(time_steps, "time_steps")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, r=rr, q=qq, v0=_v0, kappa=_kappa, theta_v=_theta, sigma=_sigma, rho=_rho, option_type=ot, s_steps=_ss, v_steps=_vs, time_steps=_ts)
        if self._native_batch is not None and hasattr(self._native_batch, "adi_douglas_price_batch"):
            return self._native_batch.adi_douglas_price_batch(s, k, tau, rr, qq, _v0, _kappa, _theta, _sigma, _rho, ot, _ss, _vs, _ts)
        return self._call_batch_ctypes("qk_fdm_adi_douglas_price_batch", s, k, tau, rr, qq, _v0, _kappa, _theta, _sigma, _rho, ot, _ss, _vs, _ts)

    def adi_craig_sneyd_price_batch(self, spot, strike, t, r, q, v0, kappa, theta_v, sigma, rho, option_type, s_steps, v_steps, time_steps) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        _v0 = self._as_f64_array(v0, "v0")
        _kappa = self._as_f64_array(kappa, "kappa")
        _theta = self._as_f64_array(theta_v, "theta_v")
        _sigma = self._as_f64_array(sigma, "sigma")
        _rho = self._as_f64_array(rho, "rho")
        ot = self._as_i32_array(option_type, "option_type")
        _ss = self._as_i32_array(s_steps, "s_steps")
        _vs = self._as_i32_array(v_steps, "v_steps")
        _ts = self._as_i32_array(time_steps, "time_steps")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, r=rr, q=qq, v0=_v0, kappa=_kappa, theta_v=_theta, sigma=_sigma, rho=_rho, option_type=ot, s_steps=_ss, v_steps=_vs, time_steps=_ts)
        if self._native_batch is not None and hasattr(self._native_batch, "adi_craig_sneyd_price_batch"):
            return self._native_batch.adi_craig_sneyd_price_batch(s, k, tau, rr, qq, _v0, _kappa, _theta, _sigma, _rho, ot, _ss, _vs, _ts)
        return self._call_batch_ctypes("qk_fdm_adi_craig_sneyd_price_batch", s, k, tau, rr, qq, _v0, _kappa, _theta, _sigma, _rho, ot, _ss, _vs, _ts)

    def adi_hundsdorfer_verwer_price_batch(self, spot, strike, t, r, q, v0, kappa, theta_v, sigma, rho, option_type, s_steps, v_steps, time_steps) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        _v0 = self._as_f64_array(v0, "v0")
        _kappa = self._as_f64_array(kappa, "kappa")
        _theta = self._as_f64_array(theta_v, "theta_v")
        _sigma = self._as_f64_array(sigma, "sigma")
        _rho = self._as_f64_array(rho, "rho")
        ot = self._as_i32_array(option_type, "option_type")
        _ss = self._as_i32_array(s_steps, "s_steps")
        _vs = self._as_i32_array(v_steps, "v_steps")
        _ts = self._as_i32_array(time_steps, "time_steps")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, r=rr, q=qq, v0=_v0, kappa=_kappa, theta_v=_theta, sigma=_sigma, rho=_rho, option_type=ot, s_steps=_ss, v_steps=_vs, time_steps=_ts)
        if self._native_batch is not None and hasattr(self._native_batch, "adi_hundsdorfer_verwer_price_batch"):
            return self._native_batch.adi_hundsdorfer_verwer_price_batch(s, k, tau, rr, qq, _v0, _kappa, _theta, _sigma, _rho, ot, _ss, _vs, _ts)
        return self._call_batch_ctypes("qk_fdm_adi_hundsdorfer_verwer_price_batch", s, k, tau, rr, qq, _v0, _kappa, _theta, _sigma, _rho, ot, _ss, _vs, _ts)

    def psor_price_batch(self, spot, strike, t, vol, r, q, option_type, time_steps, spot_steps, omega, tol, max_iter) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        ts = self._as_i32_array(time_steps, "time_steps")
        ss = self._as_i32_array(spot_steps, "spot_steps")
        om = self._as_f64_array(omega, "omega")
        tl = self._as_f64_array(tol, "tol")
        mi = self._as_i32_array(max_iter, "max_iter")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, time_steps=ts, spot_steps=ss, omega=om, tol=tl, max_iter=mi)
        if self._native_batch is not None and hasattr(self._native_batch, "psor_price_batch"):
            return self._native_batch.psor_price_batch(s, k, tau, sigma, rr, qq, ot, ts, ss, om, tl, mi)
        return self._call_batch_ctypes("qk_fdm_psor_price_batch", s, k, tau, sigma, rr, qq, ot, ts, ss, om, tl, mi)


    def gauss_hermite_price_batch(self, spot, strike, t, vol, r, q, option_type, n_points) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        np_ = self._as_i32_array(n_points, "n_points")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, n_points=np_)
        if self._native_batch is not None and hasattr(self._native_batch, "gauss_hermite_price_batch"):
            return self._native_batch.gauss_hermite_price_batch(s, k, tau, sigma, rr, qq, ot, np_)
        return self._call_batch_ctypes("qk_iqm_gauss_hermite_price_batch", s, k, tau, sigma, rr, qq, ot, np_)

    def gauss_laguerre_price_batch(self, spot, strike, t, vol, r, q, option_type, n_points) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        np_ = self._as_i32_array(n_points, "n_points")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, n_points=np_)
        if self._native_batch is not None and hasattr(self._native_batch, "gauss_laguerre_price_batch"):
            return self._native_batch.gauss_laguerre_price_batch(s, k, tau, sigma, rr, qq, ot, np_)
        return self._call_batch_ctypes("qk_iqm_gauss_laguerre_price_batch", s, k, tau, sigma, rr, qq, ot, np_)

    def gauss_legendre_price_batch(self, spot, strike, t, vol, r, q, option_type, n_points, integration_limit) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        np_ = self._as_i32_array(n_points, "n_points")
        il = self._as_f64_array(integration_limit, "integration_limit")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, n_points=np_, integration_limit=il)
        if self._native_batch is not None and hasattr(self._native_batch, "gauss_legendre_price_batch"):
            return self._native_batch.gauss_legendre_price_batch(s, k, tau, sigma, rr, qq, ot, np_, il)
        return self._call_batch_ctypes("qk_iqm_gauss_legendre_price_batch", s, k, tau, sigma, rr, qq, ot, np_, il)

    def adaptive_quadrature_price_batch(self, spot, strike, t, vol, r, q, option_type, abs_tol, rel_tol, max_depth, integration_limit) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        at = self._as_f64_array(abs_tol, "abs_tol")
        rt = self._as_f64_array(rel_tol, "rel_tol")
        md = self._as_i32_array(max_depth, "max_depth")
        il = self._as_f64_array(integration_limit, "integration_limit")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, abs_tol=at, rel_tol=rt, max_depth=md, integration_limit=il)
        if self._native_batch is not None and hasattr(self._native_batch, "adaptive_quadrature_price_batch"):
            return self._native_batch.adaptive_quadrature_price_batch(s, k, tau, sigma, rr, qq, ot, at, rt, md, il)
        return self._call_batch_ctypes("qk_iqm_adaptive_quadrature_price_batch", s, k, tau, sigma, rr, qq, ot, at, rt, md, il)


    def polynomial_chaos_expansion_price_batch(self, spot, strike, t, vol, r, q, option_type, polynomial_order, quadrature_points) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        po = self._as_i32_array(polynomial_order, "polynomial_order")
        qp = self._as_i32_array(quadrature_points, "quadrature_points")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, polynomial_order=po, quadrature_points=qp)
        if self._native_batch is not None and hasattr(self._native_batch, "polynomial_chaos_expansion_price_batch"):
            return self._native_batch.polynomial_chaos_expansion_price_batch(s, k, tau, sigma, rr, qq, ot, po, qp)
        return self._call_batch_ctypes("qk_ram_polynomial_chaos_expansion_price_batch", s, k, tau, sigma, rr, qq, ot, po, qp)

    def radial_basis_function_price_batch(self, spot, strike, t, vol, r, q, option_type, centers, rbf_shape, ridge) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        ct_ = self._as_i32_array(centers, "centers")
        rs = self._as_f64_array(rbf_shape, "rbf_shape")
        rg = self._as_f64_array(ridge, "ridge")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, centers=ct_, rbf_shape=rs, ridge=rg)
        if self._native_batch is not None and hasattr(self._native_batch, "radial_basis_function_price_batch"):
            return self._native_batch.radial_basis_function_price_batch(s, k, tau, sigma, rr, qq, ot, ct_, rs, rg)
        return self._call_batch_ctypes("qk_ram_radial_basis_function_price_batch", s, k, tau, sigma, rr, qq, ot, ct_, rs, rg)

    def sparse_grid_collocation_price_batch(self, spot, strike, t, vol, r, q, option_type, level, nodes_per_dim) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        lv = self._as_i32_array(level, "level")
        nd = self._as_i32_array(nodes_per_dim, "nodes_per_dim")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, level=lv, nodes_per_dim=nd)
        if self._native_batch is not None and hasattr(self._native_batch, "sparse_grid_collocation_price_batch"):
            return self._native_batch.sparse_grid_collocation_price_batch(s, k, tau, sigma, rr, qq, ot, lv, nd)
        return self._call_batch_ctypes("qk_ram_sparse_grid_collocation_price_batch", s, k, tau, sigma, rr, qq, ot, lv, nd)

    def proper_orthogonal_decomposition_price_batch(self, spot, strike, t, vol, r, q, option_type, modes, snapshots) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        md = self._as_i32_array(modes, "modes")
        sn = self._as_i32_array(snapshots, "snapshots")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, modes=md, snapshots=sn)
        if self._native_batch is not None and hasattr(self._native_batch, "proper_orthogonal_decomposition_price_batch"):
            return self._native_batch.proper_orthogonal_decomposition_price_batch(s, k, tau, sigma, rr, qq, ot, md, sn)
        return self._call_batch_ctypes("qk_ram_proper_orthogonal_decomposition_price_batch", s, k, tau, sigma, rr, qq, ot, md, sn)


    def pathwise_derivative_delta_batch(self, spot, strike, t, vol, r, q, option_type, paths, seed) -> np.ndarray:
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
        if self._native_batch is not None and hasattr(self._native_batch, "pathwise_derivative_delta_batch"):
            return self._native_batch.pathwise_derivative_delta_batch(s, k, tau, sigma, rr, qq, ot, pp, sd)
        return self._call_batch_ctypes("qk_agm_pathwise_derivative_delta_batch", s, k, tau, sigma, rr, qq, ot, pp, sd)

    def likelihood_ratio_delta_batch(self, spot, strike, t, vol, r, q, option_type, paths, seed, weight_clip) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        pp = self._as_i32_array(paths, "paths")
        sd = self._as_u64_array(seed, "seed")
        wc = self._as_f64_array(weight_clip, "weight_clip")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, paths=pp, seed=sd, weight_clip=wc)
        if self._native_batch is not None and hasattr(self._native_batch, "likelihood_ratio_delta_batch"):
            return self._native_batch.likelihood_ratio_delta_batch(s, k, tau, sigma, rr, qq, ot, pp, sd, wc)
        return self._call_batch_ctypes("qk_agm_likelihood_ratio_delta_batch", s, k, tau, sigma, rr, qq, ot, pp, sd, wc)

    def aad_delta_batch(self, spot, strike, t, vol, r, q, option_type, tape_steps, regularization) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        ts = self._as_i32_array(tape_steps, "tape_steps")
        rg = self._as_f64_array(regularization, "regularization")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, tape_steps=ts, regularization=rg)
        if self._native_batch is not None and hasattr(self._native_batch, "aad_delta_batch"):
            return self._native_batch.aad_delta_batch(s, k, tau, sigma, rr, qq, ot, ts, rg)
        return self._call_batch_ctypes("qk_agm_aad_delta_batch", s, k, tau, sigma, rr, qq, ot, ts, rg)


    def deep_bsde_price_batch(self, spot, strike, t, vol, r, q, option_type, time_steps, hidden_width, training_epochs, learning_rate) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        ts = self._as_i32_array(time_steps, "time_steps")
        hw = self._as_i32_array(hidden_width, "hidden_width")
        te = self._as_i32_array(training_epochs, "training_epochs")
        lr = self._as_f64_array(learning_rate, "learning_rate")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, time_steps=ts, hidden_width=hw, training_epochs=te, learning_rate=lr)
        if self._native_batch is not None and hasattr(self._native_batch, "deep_bsde_price_batch"):
            return self._native_batch.deep_bsde_price_batch(s, k, tau, sigma, rr, qq, ot, ts, hw, te, lr)
        return self._call_batch_ctypes("qk_mlm_deep_bsde_price_batch", s, k, tau, sigma, rr, qq, ot, ts, hw, te, lr)

    def pinns_price_batch(self, spot, strike, t, vol, r, q, option_type, collocation_points, boundary_points, epochs, loss_balance) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        cp = self._as_i32_array(collocation_points, "collocation_points")
        bp = self._as_i32_array(boundary_points, "boundary_points")
        ep = self._as_i32_array(epochs, "epochs")
        lb = self._as_f64_array(loss_balance, "loss_balance")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, collocation_points=cp, boundary_points=bp, epochs=ep, loss_balance=lb)
        if self._native_batch is not None and hasattr(self._native_batch, "pinns_price_batch"):
            return self._native_batch.pinns_price_batch(s, k, tau, sigma, rr, qq, ot, cp, bp, ep, lb)
        return self._call_batch_ctypes("qk_mlm_pinns_price_batch", s, k, tau, sigma, rr, qq, ot, cp, bp, ep, lb)

    def deep_hedging_price_batch(self, spot, strike, t, vol, r, q, option_type, rehedge_steps, risk_aversion, scenarios, seed) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        rs = self._as_i32_array(rehedge_steps, "rehedge_steps")
        ra = self._as_f64_array(risk_aversion, "risk_aversion")
        sc = self._as_i32_array(scenarios, "scenarios")
        sd = self._as_u64_array(seed, "seed")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, rehedge_steps=rs, risk_aversion=ra, scenarios=sc, seed=sd)
        if self._native_batch is not None and hasattr(self._native_batch, "deep_hedging_price_batch"):
            return self._native_batch.deep_hedging_price_batch(s, k, tau, sigma, rr, qq, ot, rs, ra, sc, sd)
        return self._call_batch_ctypes("qk_mlm_deep_hedging_price_batch", s, k, tau, sigma, rr, qq, ot, rs, ra, sc, sd)

    def neural_sde_calibration_price_batch(self, spot, strike, t, vol, r, q, option_type, target_implied_vol, calibration_steps, regularization) -> np.ndarray:
        s = self._as_f64_array(spot, "spot")
        k = self._as_f64_array(strike, "strike")
        tau = self._as_f64_array(t, "t")
        sigma = self._as_f64_array(vol, "vol")
        rr = self._as_f64_array(r, "r")
        qq = self._as_f64_array(q, "q")
        ot = self._as_i32_array(option_type, "option_type")
        ti = self._as_f64_array(target_implied_vol, "target_implied_vol")
        cs = self._as_i32_array(calibration_steps, "calibration_steps")
        rg = self._as_f64_array(regularization, "regularization")
        n = s.shape[0]
        self._check_same_length(n, strike=k, t=tau, vol=sigma, r=rr, q=qq, option_type=ot, target_implied_vol=ti, calibration_steps=cs, regularization=rg)
        if self._native_batch is not None and hasattr(self._native_batch, "neural_sde_calibration_price_batch"):
            return self._native_batch.neural_sde_calibration_price_batch(s, k, tau, sigma, rr, qq, ot, ti, cs, rg)
        return self._call_batch_ctypes("qk_mlm_neural_sde_calibration_price_batch", s, k, tau, sigma, rr, qq, ot, ti, cs, rg)
