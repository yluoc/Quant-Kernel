#!/usr/bin/env python3
"""Synthetic-grid validation for model-driven pricers vs QuantLib references.

This suite is the third validation track:
1) Historical broad-method validation (sample market sheet vs QuantLib BSM).
2) Historical American-method validation (sample market sheet vs QuantLib American FD).
3) This synthetic model validation for model-specific parameterized pricers:
   Heston, Merton jump-diffusion, Variance Gamma, and SABR.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

os.environ.setdefault("QK_LIB_PATH", str(ROOT / "build" / "cpp"))

from quantkernel import QK_CALL, QK_PUT, QuantKernel  # noqa: E402


def _load_batch_module():
    path = ROOT / "examples" / "validate_historical_batch.py"
    spec = importlib.util.spec_from_file_location("qk_hist_batch", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class HestonCase:
    label: str
    spot: float
    strike: float
    t: float
    r: float
    q: float
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    option_type: int
    integration_steps: int
    integration_limit: float
    feller_lhs: float


@dataclass(frozen=True)
class MertonCase:
    label: str
    spot: float
    strike: float
    t: float
    vol: float
    r: float
    q: float
    jump_intensity: float
    jump_mean: float
    jump_vol: float
    max_terms: int
    option_type: int


@dataclass(frozen=True)
class VarianceGammaCase:
    label: str
    spot: float
    strike: float
    t: float
    r: float
    q: float
    sigma: float
    theta: float
    nu: float
    option_type: int
    integration_steps: int
    integration_limit: float
    martingale_arg: float


@dataclass(frozen=True)
class SabrCase:
    label: str
    forward: float
    strike: float
    t: float
    r: float
    alpha: float
    beta: float
    rho: float
    nu: float
    option_type: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    abs_tol: float
    rel_tol: float
    notes: str
    build_cases: Callable[[], list]
    validate: Callable[["QuantKernel", "QuantLibSyntheticRef", list], dict]


class QuantLibSyntheticRef:
    def __init__(self, lib_path: Path):
        self._lib = ct.CDLL(str(lib_path))
        dptr = ct.POINTER(ct.c_double)
        i32ptr = ct.POINTER(ct.c_int32)

        self._heston = self._lib.ql_ref_heston_price_batch
        self._heston.argtypes = [dptr, dptr, dptr, dptr, dptr, dptr, dptr, dptr, dptr, dptr, i32ptr, i32ptr, ct.c_int32, dptr, ct.POINTER(ct.c_char), ct.c_int32]
        self._heston.restype = ct.c_int32

        self._merton = self._lib.ql_ref_merton_jump_diffusion_price_batch
        self._merton.argtypes = [dptr, dptr, dptr, dptr, dptr, dptr, dptr, dptr, dptr, i32ptr, i32ptr, ct.c_int32, dptr, ct.POINTER(ct.c_char), ct.c_int32]
        self._merton.restype = ct.c_int32

        self._vg = self._lib.ql_ref_variance_gamma_price_batch
        self._vg.argtypes = [dptr, dptr, dptr, dptr, dptr, dptr, dptr, dptr, i32ptr, ct.c_int32, dptr, ct.POINTER(ct.c_char), ct.c_int32]
        self._vg.restype = ct.c_int32

        self._sabr_iv = self._lib.ql_ref_sabr_lognormal_iv_batch
        self._sabr_iv.argtypes = [dptr, dptr, dptr, dptr, dptr, dptr, dptr, ct.c_int32, dptr, ct.POINTER(ct.c_char), ct.c_int32]
        self._sabr_iv.restype = ct.c_int32

        self._sabr_price = self._lib.ql_ref_sabr_black76_price_batch
        self._sabr_price.argtypes = [dptr, dptr, dptr, dptr, dptr, dptr, dptr, dptr, i32ptr, ct.c_int32, dptr, ct.POINTER(ct.c_char), ct.c_int32]
        self._sabr_price.restype = ct.c_int32

    @staticmethod
    def _call(fn, out: np.ndarray, *args) -> np.ndarray:
        err = ct.create_string_buffer(1024)
        rc = fn(*args, out.ctypes.data_as(ct.POINTER(ct.c_double)), err, len(err))
        if rc != 0:
            msg = err.value.decode("utf-8", errors="replace").strip() or f"rc={rc}"
            raise RuntimeError(msg)
        return out

    def heston_price_batch(self, *, spot, strike, t, r, q, v0, kappa, theta, sigma, rho, option_type, integration_steps) -> np.ndarray:
        n = int(spot.shape[0])
        out = np.empty(n, dtype=np.float64)
        return self._call(
            self._heston,
            out,
            spot.ctypes.data_as(ct.POINTER(ct.c_double)),
            strike.ctypes.data_as(ct.POINTER(ct.c_double)),
            t.ctypes.data_as(ct.POINTER(ct.c_double)),
            r.ctypes.data_as(ct.POINTER(ct.c_double)),
            q.ctypes.data_as(ct.POINTER(ct.c_double)),
            v0.ctypes.data_as(ct.POINTER(ct.c_double)),
            kappa.ctypes.data_as(ct.POINTER(ct.c_double)),
            theta.ctypes.data_as(ct.POINTER(ct.c_double)),
            sigma.ctypes.data_as(ct.POINTER(ct.c_double)),
            rho.ctypes.data_as(ct.POINTER(ct.c_double)),
            option_type.ctypes.data_as(ct.POINTER(ct.c_int32)),
            integration_steps.ctypes.data_as(ct.POINTER(ct.c_int32)),
            n,
        )

    def merton_price_batch(self, *, spot, strike, t, vol, r, q, jump_intensity, jump_mean, jump_vol, max_terms, option_type) -> np.ndarray:
        n = int(spot.shape[0])
        out = np.empty(n, dtype=np.float64)
        return self._call(
            self._merton,
            out,
            spot.ctypes.data_as(ct.POINTER(ct.c_double)),
            strike.ctypes.data_as(ct.POINTER(ct.c_double)),
            t.ctypes.data_as(ct.POINTER(ct.c_double)),
            vol.ctypes.data_as(ct.POINTER(ct.c_double)),
            r.ctypes.data_as(ct.POINTER(ct.c_double)),
            q.ctypes.data_as(ct.POINTER(ct.c_double)),
            jump_intensity.ctypes.data_as(ct.POINTER(ct.c_double)),
            jump_mean.ctypes.data_as(ct.POINTER(ct.c_double)),
            jump_vol.ctypes.data_as(ct.POINTER(ct.c_double)),
            max_terms.ctypes.data_as(ct.POINTER(ct.c_int32)),
            option_type.ctypes.data_as(ct.POINTER(ct.c_int32)),
            n,
        )

    def vg_price_batch(self, *, spot, strike, t, r, q, sigma, theta, nu, option_type) -> np.ndarray:
        n = int(spot.shape[0])
        out = np.empty(n, dtype=np.float64)
        return self._call(
            self._vg,
            out,
            spot.ctypes.data_as(ct.POINTER(ct.c_double)),
            strike.ctypes.data_as(ct.POINTER(ct.c_double)),
            t.ctypes.data_as(ct.POINTER(ct.c_double)),
            r.ctypes.data_as(ct.POINTER(ct.c_double)),
            q.ctypes.data_as(ct.POINTER(ct.c_double)),
            sigma.ctypes.data_as(ct.POINTER(ct.c_double)),
            theta.ctypes.data_as(ct.POINTER(ct.c_double)),
            nu.ctypes.data_as(ct.POINTER(ct.c_double)),
            option_type.ctypes.data_as(ct.POINTER(ct.c_int32)),
            n,
        )

    def sabr_iv_batch(self, *, forward, strike, t, alpha, beta, rho, nu) -> np.ndarray:
        n = int(forward.shape[0])
        out = np.empty(n, dtype=np.float64)
        return self._call(
            self._sabr_iv,
            out,
            forward.ctypes.data_as(ct.POINTER(ct.c_double)),
            strike.ctypes.data_as(ct.POINTER(ct.c_double)),
            t.ctypes.data_as(ct.POINTER(ct.c_double)),
            alpha.ctypes.data_as(ct.POINTER(ct.c_double)),
            beta.ctypes.data_as(ct.POINTER(ct.c_double)),
            rho.ctypes.data_as(ct.POINTER(ct.c_double)),
            nu.ctypes.data_as(ct.POINTER(ct.c_double)),
            n,
        )

    def sabr_price_batch(self, *, forward, strike, t, r, alpha, beta, rho, nu, option_type) -> np.ndarray:
        n = int(forward.shape[0])
        out = np.empty(n, dtype=np.float64)
        return self._call(
            self._sabr_price,
            out,
            forward.ctypes.data_as(ct.POINTER(ct.c_double)),
            strike.ctypes.data_as(ct.POINTER(ct.c_double)),
            t.ctypes.data_as(ct.POINTER(ct.c_double)),
            r.ctypes.data_as(ct.POINTER(ct.c_double)),
            alpha.ctypes.data_as(ct.POINTER(ct.c_double)),
            beta.ctypes.data_as(ct.POINTER(ct.c_double)),
            rho.ctypes.data_as(ct.POINTER(ct.c_double)),
            nu.ctypes.data_as(ct.POINTER(ct.c_double)),
            option_type.ctypes.data_as(ct.POINTER(ct.c_int32)),
            n,
        )


def _common_metrics(model: np.ndarray, ref: np.ndarray, abs_tol: float, rel_tol: float, rel_floor: float) -> dict[str, object]:
    finite = np.isfinite(model) & np.isfinite(ref)
    if not np.any(finite):
        return {
            "n": int(model.size),
            "n_finite": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "max_abs": float("nan"),
            "max_rel": float("nan"),
            "p95_abs": float("nan"),
            "p95_rel": float("nan"),
            "n_fail": int(model.size),
            "pass": False,
            "fail_mask": np.ones(model.size, dtype=bool),
        }
    diff = model - ref
    abs_err = np.abs(diff)
    rel_err = abs_err / np.maximum(np.abs(ref), rel_floor)
    fail_mask = (~finite) | ((abs_err > abs_tol) & (rel_err > rel_tol))
    return {
        "n": int(model.size),
        "n_finite": int(np.sum(finite)),
        "mae": float(np.mean(abs_err[finite])),
        "rmse": float(np.sqrt(np.mean((diff[finite]) ** 2))),
        "max_abs": float(np.max(abs_err[finite])),
        "max_rel": float(np.max(rel_err[finite])),
        "p95_abs": float(np.percentile(abs_err[finite], 95)),
        "p95_rel": float(np.percentile(rel_err[finite], 95)),
        "n_fail": int(np.sum(fail_mask)),
        "pass": bool(np.sum(fail_mask) == 0),
        "fail_mask": fail_mask,
        "abs_err": abs_err,
        "rel_err": rel_err,
    }


def _extract_failures(
    *,
    cases: list,
    model: np.ndarray,
    ref: np.ndarray,
    abs_err: np.ndarray,
    rel_err: np.ndarray,
    fail_mask: np.ndarray,
    max_failures: int,
) -> list[dict[str, object]]:
    idx = np.flatnonzero(fail_mask)[:max_failures]
    out: list[dict[str, object]] = []
    for i in idx:
        out.append(
            {
                "index": int(i),
                "label": getattr(cases[i], "label", f"case_{i}"),
                "model": float(model[i]) if np.isfinite(model[i]) else "nan",
                "reference": float(ref[i]) if np.isfinite(ref[i]) else "nan",
                "abs_err": float(abs_err[i]) if np.isfinite(abs_err[i]) else "nan",
                "rel_err": float(rel_err[i]) if np.isfinite(rel_err[i]) else "nan",
                "params": asdict(cases[i]),
            }
        )
    return out


def _build_heston_cases() -> list[HestonCase]:
    # Market-style dimensions used across all Heston regimes.
    market_grid = [
        {"label": "atm_1y", "spot": 100.0, "strike": 100.0, "t": 1.0, "r": 0.03, "q": 0.01},
        {"label": "otm_call_2y", "spot": 100.0, "strike": 120.0, "t": 2.0, "r": 0.04, "q": 0.015},
        {"label": "itm_call_6m", "spot": 100.0, "strike": 80.0, "t": 0.5, "r": 0.02, "q": 0.00},
    ]

    # Regimes include above/below/near Feller boundary and stress volatility-of-variance.
    # Feller expression: 2*kappa*theta - sigma^2.
    near_sigma = (2.0 * 1.5 * 0.04) ** 0.5 * 0.995
    heston_regimes = [
        {"label": "feller_above", "v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.30, "rho": -0.50},
        {"label": "feller_near", "v0": 0.04, "kappa": 1.5, "theta": 0.04, "sigma": near_sigma, "rho": -0.70},
        {"label": "feller_below", "v0": 0.04, "kappa": 1.0, "theta": 0.04, "sigma": 0.40, "rho": -0.60},
        {"label": "stress_high_volvol", "v0": 0.09, "kappa": 0.6, "theta": 0.09, "sigma": 1.20, "rho": -0.85},
    ]

    cases: list[HestonCase] = []
    for mkt in market_grid:
        for reg in heston_regimes:
            for ot, ot_label in ((QK_CALL, "call"), (QK_PUT, "put")):
                feller_lhs = 2.0 * reg["kappa"] * reg["theta"] - reg["sigma"] * reg["sigma"]
                cases.append(
                    HestonCase(
                        label=f"{mkt['label']}|{reg['label']}|{ot_label}",
                        spot=mkt["spot"],
                        strike=mkt["strike"],
                        t=mkt["t"],
                        r=mkt["r"],
                        q=mkt["q"],
                        v0=reg["v0"],
                        kappa=reg["kappa"],
                        theta=reg["theta"],
                        sigma=reg["sigma"],
                        rho=reg["rho"],
                        option_type=ot,
                        integration_steps=160,
                        integration_limit=120.0,
                        feller_lhs=feller_lhs,
                    )
                )
    return cases


def _build_merton_cases() -> list[MertonCase]:
    market_grid = [
        {"label": "atm_1y", "spot": 100.0, "strike": 100.0, "t": 1.0, "vol": 0.20, "r": 0.03, "q": 0.01},
        {"label": "otm_call_2y", "spot": 100.0, "strike": 120.0, "t": 2.0, "vol": 0.25, "r": 0.04, "q": 0.015},
        {"label": "itm_put_6m", "spot": 100.0, "strike": 90.0, "t": 0.5, "vol": 0.18, "r": 0.02, "q": 0.00},
    ]

    # Includes no-jump limit and high-intensity discontinuity stress regimes.
    jump_regimes = [
        {"label": "no_jump_limit", "jump_intensity": 0.0, "jump_mean": 0.0, "jump_vol": 0.0, "max_terms": 80},
        {"label": "normal_equity_jumps", "jump_intensity": 0.2, "jump_mean": -0.05, "jump_vol": 0.20, "max_terms": 100},
        {"label": "stress_down_jumps", "jump_intensity": 1.5, "jump_mean": -0.25, "jump_vol": 0.50, "max_terms": 150},
        {"label": "stress_high_freq_small", "jump_intensity": 3.0, "jump_mean": -0.02, "jump_vol": 0.15, "max_terms": 180},
        {"label": "stress_two_sided", "jump_intensity": 0.8, "jump_mean": 0.10, "jump_vol": 0.40, "max_terms": 140},
    ]

    cases: list[MertonCase] = []
    for mkt in market_grid:
        for reg in jump_regimes:
            for ot, ot_label in ((QK_CALL, "call"), (QK_PUT, "put")):
                cases.append(
                    MertonCase(
                        label=f"{mkt['label']}|{reg['label']}|{ot_label}",
                        spot=mkt["spot"],
                        strike=mkt["strike"],
                        t=mkt["t"],
                        vol=mkt["vol"],
                        r=mkt["r"],
                        q=mkt["q"],
                        jump_intensity=reg["jump_intensity"],
                        jump_mean=reg["jump_mean"],
                        jump_vol=reg["jump_vol"],
                        max_terms=reg["max_terms"],
                        option_type=ot,
                    )
                )
    return cases


def _build_vg_cases() -> list[VarianceGammaCase]:
    market_grid = [
        {"label": "atm_1y", "spot": 100.0, "strike": 100.0, "t": 1.0, "r": 0.03, "q": 0.01},
        {"label": "otm_call_2y", "spot": 100.0, "strike": 120.0, "t": 2.0, "r": 0.04, "q": 0.015},
        {"label": "itm_call_9m", "spot": 100.0, "strike": 85.0, "t": 0.75, "r": 0.02, "q": 0.0},
    ]

    vg_regimes = [
        {"label": "normal", "sigma": 0.20, "theta": -0.10, "nu": 0.20},
        {"label": "small_nu", "sigma": 0.25, "theta": -0.05, "nu": 0.05},
        {"label": "stress_kurtosis", "sigma": 0.45, "theta": -0.25, "nu": 0.60},
        {"label": "stress_near_admissibility", "sigma": 0.60, "theta": 0.35, "nu": 0.90},
    ]

    cases: list[VarianceGammaCase] = []
    for mkt in market_grid:
        for reg in vg_regimes:
            martingale_arg = 1.0 - reg["theta"] * reg["nu"] - 0.5 * reg["sigma"] * reg["sigma"] * reg["nu"]
            for ot, ot_label in ((QK_CALL, "call"), (QK_PUT, "put")):
                cases.append(
                    VarianceGammaCase(
                        label=f"{mkt['label']}|{reg['label']}|{ot_label}",
                        spot=mkt["spot"],
                        strike=mkt["strike"],
                        t=mkt["t"],
                        r=mkt["r"],
                        q=mkt["q"],
                        sigma=reg["sigma"],
                        theta=reg["theta"],
                        nu=reg["nu"],
                        option_type=ot,
                        integration_steps=1024,
                        integration_limit=120.0,
                        martingale_arg=martingale_arg,
                    )
                )
    return cases


def _build_sabr_cases() -> list[SabrCase]:
    forward_strike_grid = [
        {"label": "atm_1y", "forward": 100.0, "strike": 100.0, "t": 1.0, "r": 0.03},
        {"label": "otm_call_1y", "forward": 100.0, "strike": 120.0, "t": 1.0, "r": 0.03},
        {"label": "itm_call_1y", "forward": 100.0, "strike": 85.0, "t": 1.0, "r": 0.03},
        {"label": "otm_call_3y", "forward": 105.0, "strike": 125.0, "t": 3.0, "r": 0.035},
    ]

    sabr_regimes = [
        {"label": "normal", "alpha": 0.20, "beta": 0.50, "rho": -0.20, "nu": 0.40},
        {"label": "beta_near_one", "alpha": 0.18, "beta": 0.95, "rho": -0.10, "nu": 0.35},
        {"label": "beta_near_zero", "alpha": 0.30, "beta": 0.05, "rho": 0.10, "nu": 0.55},
        {"label": "stress_high_nu_rho", "alpha": 0.25, "beta": 0.60, "rho": 0.90, "nu": 1.20},
    ]

    cases: list[SabrCase] = []
    for fs in forward_strike_grid:
        for reg in sabr_regimes:
            for ot, ot_label in ((QK_CALL, "call"), (QK_PUT, "put")):
                cases.append(
                    SabrCase(
                        label=f"{fs['label']}|{reg['label']}|{ot_label}",
                        forward=fs["forward"],
                        strike=fs["strike"],
                        t=fs["t"],
                        r=fs["r"],
                        alpha=reg["alpha"],
                        beta=reg["beta"],
                        rho=reg["rho"],
                        nu=reg["nu"],
                        option_type=ot,
                    )
                )
    return cases


def _to_arr(cases: list, name: str, dtype) -> np.ndarray:
    return np.ascontiguousarray(np.array([getattr(c, name) for c in cases], dtype=dtype))


def _validate_heston(qk: QuantKernel, ql: QuantLibSyntheticRef, cases: list[HestonCase]) -> dict:
    s = _to_arr(cases, "spot", np.float64)
    k = _to_arr(cases, "strike", np.float64)
    tau = _to_arr(cases, "t", np.float64)
    rr = _to_arr(cases, "r", np.float64)
    qq = _to_arr(cases, "q", np.float64)
    v0 = _to_arr(cases, "v0", np.float64)
    kap = _to_arr(cases, "kappa", np.float64)
    the = _to_arr(cases, "theta", np.float64)
    sig = _to_arr(cases, "sigma", np.float64)
    rho = _to_arr(cases, "rho", np.float64)
    ot = _to_arr(cases, "option_type", np.int32)
    is_ = _to_arr(cases, "integration_steps", np.int32)
    il = _to_arr(cases, "integration_limit", np.float64)

    model = qk.heston_price_cf_batch(s, k, tau, rr, qq, v0, kap, the, sig, rho, ot, is_, il)
    ref = ql.heston_price_batch(
        spot=s, strike=k, t=tau, r=rr, q=qq, v0=v0, kappa=kap, theta=the, sigma=sig, rho=rho,
        option_type=ot, integration_steps=is_
    )
    return {"model": model, "ref": ref}


def _validate_merton(qk: QuantKernel, ql: QuantLibSyntheticRef, cases: list[MertonCase]) -> dict:
    s = _to_arr(cases, "spot", np.float64)
    k = _to_arr(cases, "strike", np.float64)
    tau = _to_arr(cases, "t", np.float64)
    vol = _to_arr(cases, "vol", np.float64)
    rr = _to_arr(cases, "r", np.float64)
    qq = _to_arr(cases, "q", np.float64)
    ji = _to_arr(cases, "jump_intensity", np.float64)
    jm = _to_arr(cases, "jump_mean", np.float64)
    jv = _to_arr(cases, "jump_vol", np.float64)
    mt = _to_arr(cases, "max_terms", np.int32)
    ot = _to_arr(cases, "option_type", np.int32)

    model = qk.merton_jump_diffusion_price_batch(s, k, tau, vol, rr, qq, ji, jm, jv, mt, ot)
    ref = ql.merton_price_batch(
        spot=s, strike=k, t=tau, vol=vol, r=rr, q=qq, jump_intensity=ji, jump_mean=jm, jump_vol=jv, max_terms=mt, option_type=ot
    )
    return {"model": model, "ref": ref}


def _validate_vg(qk: QuantKernel, ql: QuantLibSyntheticRef, cases: list[VarianceGammaCase]) -> dict:
    s = _to_arr(cases, "spot", np.float64)
    k = _to_arr(cases, "strike", np.float64)
    tau = _to_arr(cases, "t", np.float64)
    rr = _to_arr(cases, "r", np.float64)
    qq = _to_arr(cases, "q", np.float64)
    sigma = _to_arr(cases, "sigma", np.float64)
    theta = _to_arr(cases, "theta", np.float64)
    nu = _to_arr(cases, "nu", np.float64)
    ot = _to_arr(cases, "option_type", np.int32)
    is_ = _to_arr(cases, "integration_steps", np.int32)
    il = _to_arr(cases, "integration_limit", np.float64)

    model = qk.variance_gamma_price_cf_batch(s, k, tau, rr, qq, sigma, theta, nu, ot, is_, il)
    ref = ql.vg_price_batch(spot=s, strike=k, t=tau, r=rr, q=qq, sigma=sigma, theta=theta, nu=nu, option_type=ot)
    return {"model": model, "ref": ref}


def _validate_sabr(qk: QuantKernel, ql: QuantLibSyntheticRef, cases: list[SabrCase]) -> dict:
    fwd = _to_arr(cases, "forward", np.float64)
    k = _to_arr(cases, "strike", np.float64)
    tau = _to_arr(cases, "t", np.float64)
    rr = _to_arr(cases, "r", np.float64)
    alpha = _to_arr(cases, "alpha", np.float64)
    beta = _to_arr(cases, "beta", np.float64)
    rho = _to_arr(cases, "rho", np.float64)
    nu = _to_arr(cases, "nu", np.float64)
    ot = _to_arr(cases, "option_type", np.int32)

    model_iv = qk.sabr_hagan_lognormal_iv_batch(fwd, k, tau, alpha, beta, rho, nu)
    ref_iv = ql.sabr_iv_batch(forward=fwd, strike=k, t=tau, alpha=alpha, beta=beta, rho=rho, nu=nu)
    model_price = qk.sabr_hagan_black76_price_batch(fwd, k, tau, rr, alpha, beta, rho, nu, ot)
    ref_price = ql.sabr_price_batch(forward=fwd, strike=k, t=tau, r=rr, alpha=alpha, beta=beta, rho=rho, nu=nu, option_type=ot)
    return {"model": model_price, "ref": ref_price, "model_iv": model_iv, "ref_iv": ref_iv}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=str,
        default="heston,merton,variance_gamma,sabr",
        help="Comma-separated list in preferred order: heston,merton,variance_gamma,sabr",
    )
    parser.add_argument("--failures-limit", type=int, default=12)
    parser.add_argument("--rel-floor", type=float, default=1e-8)
    parser.add_argument("--quantlib-source-root", type=Path, default=ROOT / "third_party_source" / "quantlib_copy" / "QuantLib")
    parser.add_argument("--quantlib-build-dir", type=Path, default=ROOT / "third_party_source" / "quantlib_copy" / "build")
    parser.add_argument("--quantlib-wrapper-src", type=Path, default=ROOT / "examples" / "quantlib_synthetic_model_ref.cpp")
    parser.add_argument("--quantlib-wrapper-lib", type=Path, default=ROOT / "third_party_source" / "quantlib_copy" / "build" / "libql_ref_synthetic.so")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    qk = QuantKernel()
    batch_mod = _load_batch_module()
    batch_mod._compile_quantlib_wrapper(
        wrapper_src=args.quantlib_wrapper_src,
        wrapper_lib=args.quantlib_wrapper_lib,
        quantlib_source_root=args.quantlib_source_root,
        quantlib_build_dir=args.quantlib_build_dir,
    )
    ql = QuantLibSyntheticRef(args.quantlib_wrapper_lib)

    specs = [
        ModelSpec(
            name="heston",
            abs_tol=0.25,
            rel_tol=0.03,
            notes=(
                "Heston parameterization is aligned (v0,kappa,theta,sigma,rho). "
                "QuantKernel uses explicit integration_steps/integration_limit, while QuantLib "
                "reference uses AnalyticHestonEngine Gauss-Laguerre order (capped at 192)."
            ),
            build_cases=_build_heston_cases,
            validate=_validate_heston,
        ),
        ModelSpec(
            name="merton",
            abs_tol=0.12,
            rel_tol=0.02,
            notes=(
                "Merton jump parameters are aligned with QuantLib JumpDiffusionEngine convention: "
                "jump_intensity (lambda), log-mean jump, and log-jump-vol. Stress cases include "
                "high jump intensity and large discontinuities."
            ),
            build_cases=_build_merton_cases,
            validate=_validate_merton,
        ),
        ModelSpec(
            name="variance_gamma",
            abs_tol=0.18,
            rel_tol=0.03,
            notes=(
                "Parameter convention warning: QuantKernel uses (sigma, theta, nu), "
                "QuantLib VarianceGammaProcess constructor order is (sigma, nu, theta). "
                "Mapping is handled explicitly in wrapper."
            ),
            build_cases=_build_vg_cases,
            validate=_validate_vg,
        ),
        ModelSpec(
            name="sabr",
            abs_tol=5e-4,
            rel_tol=1e-3,
            notes=(
                "Parameter convention warning: QuantKernel SABR API order is (alpha,beta,rho,nu), "
                "while QuantLib sabrVolatility expects (alpha,beta,nu,rho). "
                "Wrapper remaps this order for reference calls."
            ),
            build_cases=_build_sabr_cases,
            validate=_validate_sabr,
        ),
    ]
    spec_map = {s.name: s for s in specs}
    requested = [x.strip() for x in args.models.split(",") if x.strip()]

    report: dict[str, object] = {
        "config": {
            "models": requested,
            "failures_limit": int(args.failures_limit),
            "rel_floor": float(args.rel_floor),
            "quantlib_wrapper_lib": str(args.quantlib_wrapper_lib),
        },
        "conventions": {
            "heston": "Same core parameters. Integration control differs between implementations.",
            "merton": "Same convention as QuantLib JumpDiffusionEngine for lambda, log-jump mean, and log-jump vol.",
            "variance_gamma": "QuantKernel: (sigma, theta, nu) vs QuantLib process ctor: (sigma, nu, theta).",
            "sabr": "QuantKernel API: (alpha,beta,rho,nu) vs QuantLib sabrVolatility: (alpha,beta,nu,rho).",
        },
        "results": {},
    }

    for name in requested:
        if name not in spec_map:
            report["results"][name] = {"error": f"Unknown model '{name}'"}
            continue
        spec = spec_map[name]
        cases = spec.build_cases()
        ts = time.perf_counter()
        try:
            outputs = spec.validate(qk, ql, cases)
        except Exception as exc:  # pragma: no cover
            report["results"][name] = {"error": str(exc)}
            continue
        te = time.perf_counter()

        base = _common_metrics(outputs["model"], outputs["ref"], spec.abs_tol, spec.rel_tol, args.rel_floor)
        result = {
            "notes": spec.notes,
            "runtime_seconds": float(te - ts),
            "n_cases": int(len(cases)),
            "abs_tol": float(spec.abs_tol),
            "rel_tol": float(spec.rel_tol),
            "summary": {
                "pass": bool(base["pass"]),
                "n_fail": int(base["n_fail"]),
                "mae": float(base["mae"]),
                "rmse": float(base["rmse"]),
                "max_abs": float(base["max_abs"]),
                "max_rel": float(base["max_rel"]),
                "p95_abs": float(base["p95_abs"]),
                "p95_rel": float(base["p95_rel"]),
            },
            "failures": _extract_failures(
                cases=cases,
                model=outputs["model"],
                ref=outputs["ref"],
                abs_err=base["abs_err"],
                rel_err=base["rel_err"],
                fail_mask=base["fail_mask"],
                max_failures=args.failures_limit,
            ),
        }

        if name == "sabr":
            iv_metrics = _common_metrics(outputs["model_iv"], outputs["ref_iv"], 5e-5, 5e-4, args.rel_floor)
            result["iv_summary"] = {
                "pass": bool(iv_metrics["pass"]),
                "n_fail": int(iv_metrics["n_fail"]),
                "mae": float(iv_metrics["mae"]),
                "max_abs": float(iv_metrics["max_abs"]),
                "max_rel": float(iv_metrics["max_rel"]),
            }
            result["iv_failures"] = _extract_failures(
                cases=cases,
                model=outputs["model_iv"],
                ref=outputs["ref_iv"],
                abs_err=iv_metrics["abs_err"],
                rel_err=iv_metrics["rel_err"],
                fail_mask=iv_metrics["fail_mask"],
                max_failures=args.failures_limit,
            )
            # SABR overall pass requires both price and IV sections to pass.
            result["summary"]["pass"] = bool(result["summary"]["pass"] and result["iv_summary"]["pass"])

        report["results"][name] = result

    report["overall_pass"] = bool(
        all((isinstance(v, dict) and v.get("summary", {}).get("pass", False)) for v in report["results"].values() if "summary" in v)
    )
    report["total_runtime_seconds"] = float(time.perf_counter() - t0)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
