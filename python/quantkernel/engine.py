"""QuantKernel high-level Python API."""

import ctypes as ct
from typing import Dict

import numpy as np

from ._abi import QKBSInput, QKBSOutput, QKIVInput, QKIVOutput, QK_OK, QK_CALL, QK_PUT
from ._loader import load_library


def _as_double_ptr(arr: np.ndarray) -> ct.POINTER(ct.c_double):
    """Get a ctypes double pointer from a contiguous float64 array."""
    return arr.ctypes.data_as(ct.POINTER(ct.c_double))


def _as_int32_ptr(arr: np.ndarray) -> ct.POINTER(ct.c_int32):
    """Get a ctypes int32 pointer from a contiguous int32 array."""
    return arr.ctypes.data_as(ct.POINTER(ct.c_int32))


def _as_1d_contiguous(arr: np.ndarray, dtype: np.dtype, name: str) -> np.ndarray:
    """Convert to contiguous array and enforce 1-D shape."""
    out = np.ascontiguousarray(arr, dtype=dtype)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape={out.shape}")
    return out


def _validate_same_length(n: int, arrays: tuple[tuple[str, np.ndarray], ...]) -> None:
    """Ensure all input arrays have the same batch length."""
    if n <= 0:
        raise ValueError("Input arrays must contain at least one row")
    for name, arr in arrays:
        if arr.shape[0] != n:
            raise ValueError(
                f"All inputs must have identical length; "
                f"expected {n} rows but {name} has {arr.shape[0]}"
            )


class QuantKernel:
    """High-performance derivative pricing engine.

    Wraps the C++ QuantKernel shared library via ctypes.
    All computation is vectorized — pass NumPy arrays for batch pricing.
    """

    CALL = QK_CALL
    PUT = QK_PUT

    def __init__(self):
        self._lib = load_library()

    def _require_runtime_symbol(self, symbol_name: str):
        fn = getattr(self._lib, symbol_name, None)
        if fn is None:
            raise RuntimeError(
                "Runtime control API unavailable on this library. "
                "Set QK_USE_RUNTIME=1 to load the Rust runtime shell."
            )
        return fn

    def runtime_load_plugin(self, plugin_path: str) -> None:
        """Load or replace the active runtime plugin by filesystem path."""
        fn = self._require_runtime_symbol("qk_runtime_load_plugin")
        encoded = str(plugin_path).encode("utf-8")
        rc = fn(encoded)
        if rc != QK_OK:
            raise RuntimeError(f"qk_runtime_load_plugin failed with return code {rc}")

    def runtime_unload_plugin(self) -> None:
        """Unload the active runtime plugin."""
        fn = self._require_runtime_symbol("qk_runtime_unload_plugin")
        rc = fn()
        if rc != QK_OK:
            raise RuntimeError(f"qk_runtime_unload_plugin failed with return code {rc}")

    def bs_price(
        self,
        spot: np.ndarray,
        strike: np.ndarray,
        time_to_expiry: np.ndarray,
        volatility: np.ndarray,
        risk_free_rate: np.ndarray,
        dividend_yield: np.ndarray,
        option_type: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Batch Black-Scholes pricing with full greeks.

        All inputs must be 1-D NumPy arrays of the same length.
        option_type: 0 = call, 1 = put.

        Returns dict with keys: price, delta, gamma, vega, theta, rho, error_codes.
        """
        # Ensure contiguous arrays with correct dtypes
        spot = _as_1d_contiguous(spot, np.float64, "spot")
        strike = _as_1d_contiguous(strike, np.float64, "strike")
        time_to_expiry = _as_1d_contiguous(time_to_expiry, np.float64, "time_to_expiry")
        volatility = _as_1d_contiguous(volatility, np.float64, "volatility")
        risk_free_rate = _as_1d_contiguous(risk_free_rate, np.float64, "risk_free_rate")
        dividend_yield = _as_1d_contiguous(dividend_yield, np.float64, "dividend_yield")
        option_type = _as_1d_contiguous(option_type, np.int32, "option_type")

        n = spot.shape[0]
        _validate_same_length(
            n,
            (
                ("strike", strike),
                ("time_to_expiry", time_to_expiry),
                ("volatility", volatility),
                ("risk_free_rate", risk_free_rate),
                ("dividend_yield", dividend_yield),
                ("option_type", option_type),
            ),
        )

        # Allocate output buffers
        price = np.empty(n, dtype=np.float64)
        delta = np.empty(n, dtype=np.float64)
        gamma = np.empty(n, dtype=np.float64)
        vega = np.empty(n, dtype=np.float64)
        theta = np.empty(n, dtype=np.float64)
        rho = np.empty(n, dtype=np.float64)
        error_codes = np.empty(n, dtype=np.int32)

        # Build input/output structs
        bs_in = QKBSInput(
            n=n,
            spot=_as_double_ptr(spot),
            strike=_as_double_ptr(strike),
            time_to_expiry=_as_double_ptr(time_to_expiry),
            volatility=_as_double_ptr(volatility),
            risk_free_rate=_as_double_ptr(risk_free_rate),
            dividend_yield=_as_double_ptr(dividend_yield),
            option_type=_as_int32_ptr(option_type),
        )

        bs_out = QKBSOutput(
            price=_as_double_ptr(price),
            delta=_as_double_ptr(delta),
            gamma=_as_double_ptr(gamma),
            vega=_as_double_ptr(vega),
            theta=_as_double_ptr(theta),
            rho=_as_double_ptr(rho),
            error_codes=_as_int32_ptr(error_codes),
        )

        rc = self._lib.qk_bs_price(ct.byref(bs_in), ct.byref(bs_out))
        if rc != QK_OK:
            raise RuntimeError(f"qk_bs_price failed with return code {rc}")

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
            "error_codes": error_codes,
        }

    def iv_solve(
        self,
        spot: np.ndarray,
        strike: np.ndarray,
        time_to_expiry: np.ndarray,
        risk_free_rate: np.ndarray,
        dividend_yield: np.ndarray,
        option_type: np.ndarray,
        market_price: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Batch implied volatility solver (Newton-Raphson).

        Returns dict with keys: implied_vol, iterations, error_codes.
        """
        spot = _as_1d_contiguous(spot, np.float64, "spot")
        strike = _as_1d_contiguous(strike, np.float64, "strike")
        time_to_expiry = _as_1d_contiguous(time_to_expiry, np.float64, "time_to_expiry")
        risk_free_rate = _as_1d_contiguous(risk_free_rate, np.float64, "risk_free_rate")
        dividend_yield = _as_1d_contiguous(dividend_yield, np.float64, "dividend_yield")
        option_type = _as_1d_contiguous(option_type, np.int32, "option_type")
        market_price = _as_1d_contiguous(market_price, np.float64, "market_price")

        n = spot.shape[0]
        _validate_same_length(
            n,
            (
                ("strike", strike),
                ("time_to_expiry", time_to_expiry),
                ("risk_free_rate", risk_free_rate),
                ("dividend_yield", dividend_yield),
                ("option_type", option_type),
                ("market_price", market_price),
            ),
        )

        # Allocate output buffers
        implied_vol = np.empty(n, dtype=np.float64)
        iterations = np.empty(n, dtype=np.int32)
        error_codes = np.empty(n, dtype=np.int32)

        iv_in = QKIVInput(
            n=n,
            spot=_as_double_ptr(spot),
            strike=_as_double_ptr(strike),
            time_to_expiry=_as_double_ptr(time_to_expiry),
            risk_free_rate=_as_double_ptr(risk_free_rate),
            dividend_yield=_as_double_ptr(dividend_yield),
            option_type=_as_int32_ptr(option_type),
            market_price=_as_double_ptr(market_price),
            tol=tol,
            max_iter=max_iter,
        )

        iv_out = QKIVOutput(
            implied_vol=_as_double_ptr(implied_vol),
            iterations=_as_int32_ptr(iterations),
            error_codes=_as_int32_ptr(error_codes),
        )

        rc = self._lib.qk_iv_solve(ct.byref(iv_in), ct.byref(iv_out))
        if rc != QK_OK:
            raise RuntimeError(f"qk_iv_solve failed with return code {rc}")

        return {
            "implied_vol": implied_vol,
            "iterations": iterations,
            "error_codes": error_codes,
        }
