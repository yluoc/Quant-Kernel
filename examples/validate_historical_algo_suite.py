#!/usr/bin/env python3
"""Historical batch validation suite for pricing and Greeks algorithms."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

os.environ.setdefault("QK_LIB_PATH", str(ROOT / "build" / "cpp"))

from quantkernel import QuantKernel  # noqa: E402


def _load_batch_module():
    path = ROOT / "examples" / "validate_historical_batch.py"
    spec = importlib.util.spec_from_file_location("qk_hist_batch", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class AlgoSpec:
    name: str
    kind: str = "price"  # "price" or "delta"
    notes: str = ""
    max_per_side: int | None = None


ALL_ALGOS = [
    AlgoSpec("bsm"),
    AlgoSpec("crr"),
    AlgoSpec("jarrow_rudd"),
    AlgoSpec("tian"),
    AlgoSpec("leisen_reimer"),
    AlgoSpec("trinomial"),
    AlgoSpec("explicit_fd"),
    AlgoSpec("implicit_fd"),
    AlgoSpec("crank_nicolson"),
    AlgoSpec("psor", notes="PSOR implementation is American-style; Euro BSM reference is indicative only."),
    AlgoSpec("carr_madan_fft"),
    AlgoSpec("cos"),
    AlgoSpec("fractional_fft"),
    AlgoSpec("lewis"),
    AlgoSpec("hilbert"),
    AlgoSpec("gauss_hermite"),
    AlgoSpec("gauss_laguerre"),
    AlgoSpec("gauss_legendre"),
    AlgoSpec("adaptive_quadrature"),
    AlgoSpec("standard_monte_carlo", max_per_side=200),
    AlgoSpec("euler_maruyama", max_per_side=160),
    AlgoSpec("milstein", max_per_side=160),
    AlgoSpec(
        "longstaff_schwartz",
        notes="Longstaff-Schwartz is American-style; Euro BSM reference is indicative only.",
        max_per_side=120,
    ),
    AlgoSpec("quasi_monte_carlo_sobol", max_per_side=200),
    AlgoSpec("quasi_monte_carlo_halton", max_per_side=200),
    AlgoSpec("multilevel_monte_carlo", max_per_side=120),
    AlgoSpec("importance_sampling", max_per_side=200),
    AlgoSpec("control_variates", max_per_side=200),
    AlgoSpec("antithetic_variates", max_per_side=200),
    AlgoSpec("stratified_sampling", max_per_side=200),
    AlgoSpec("polynomial_chaos_expansion"),
    AlgoSpec("radial_basis_function"),
    AlgoSpec("sparse_grid_collocation"),
    AlgoSpec("proper_orthogonal_decomposition"),
    AlgoSpec("deep_bsde", max_per_side=64),
    AlgoSpec("pinns", max_per_side=48),
    AlgoSpec("deep_hedging", max_per_side=48),
    AlgoSpec("neural_sde_calibration", max_per_side=128),
    AlgoSpec("pathwise_derivative_delta", kind="delta", max_per_side=200),
    AlgoSpec("likelihood_ratio_delta", kind="delta", max_per_side=200),
    AlgoSpec("aad_delta", kind="delta"),
]


def _pair_metrics(model: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(model)
    invalid_count = int(np.sum(~finite))
    if not np.any(finite):
        return {
            "n_valid": 0.0,
            "invalid_count": float(invalid_count),
            "mae": float("nan"),
            "rmse": float("nan"),
            "p95_abs": float("nan"),
            "p99_abs": float("nan"),
            "max_abs": float("nan"),
        }
    diff = model[finite] - ref[finite]
    abs_diff = np.abs(diff)
    return {
        "n_valid": float(abs_diff.size),
        "invalid_count": float(invalid_count),
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "p95_abs": float(np.percentile(abs_diff, 95)),
        "p99_abs": float(np.percentile(abs_diff, 99)),
        "max_abs": float(np.max(abs_diff)),
    }


def _market_metrics(model: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(model)
    invalid_count = int(np.sum(~finite))
    if not np.any(finite):
        return {
            "n_valid": 0.0,
            "invalid_count": float(invalid_count),
            "mae_to_mid": float("nan"),
            "rmse_to_mid": float("nan"),
            "inside_bid_ask_ratio": float("nan"),
            "mean_outside_bid_ask": float("nan"),
            "p95_outside_bid_ask": float("nan"),
        }
    m = model[finite]
    b = bid[finite]
    a = ask[finite]
    mid = 0.5 * (b + a)
    err = m - mid
    outside = np.maximum(b - m, 0.0) + np.maximum(m - a, 0.0)
    return {
        "n_valid": float(m.size),
        "invalid_count": float(invalid_count),
        "mae_to_mid": float(np.mean(np.abs(err))),
        "rmse_to_mid": float(np.sqrt(np.mean(err * err))),
        "inside_bid_ask_ratio": float(np.mean(outside == 0.0)),
        "mean_outside_bid_ask": float(np.mean(outside)),
        "p95_outside_bid_ask": float(np.percentile(outside, 95)),
    }


def _slice_batch(batch, max_n: int | None):
    if max_n is None or batch.size <= max_n:
        return batch
    return type(batch)(
        spot=np.ascontiguousarray(batch.spot[:max_n]),
        strike=np.ascontiguousarray(batch.strike[:max_n]),
        tau=np.ascontiguousarray(batch.tau[:max_n]),
        vol=np.ascontiguousarray(batch.vol[:max_n]),
        bid=np.ascontiguousarray(batch.bid[:max_n]),
        ask=np.ascontiguousarray(batch.ask[:max_n]),
        option_type=np.ascontiguousarray(batch.option_type[:max_n]),
        r=np.ascontiguousarray(batch.r[:max_n]),
        q=np.ascontiguousarray(batch.q[:max_n]),
    )


def _quantlib_delta_fd(
    ql_ref,
    batch,
    rel_bump: float,
    abs_bump: float,
) -> np.ndarray:
    bump = np.maximum(abs_bump, rel_bump * np.abs(batch.spot))
    s_up = np.ascontiguousarray(batch.spot + bump)
    s_dn = np.ascontiguousarray(np.maximum(1e-12, batch.spot - bump))
    p_up = ql_ref.price_batch(s_up, batch.strike, batch.tau, batch.vol, batch.r, batch.q, batch.option_type)
    p_dn = ql_ref.price_batch(s_dn, batch.strike, batch.tau, batch.vol, batch.r, batch.q, batch.option_type)
    denom = s_up - s_dn
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (p_up - p_dn) / denom
    return delta


def _invoke_batch(qk: QuantKernel, algo: str, batch) -> np.ndarray:
    n = batch.size
    ot = batch.option_type
    zeros_i = np.zeros(n, dtype=np.int32)
    seed = np.arange(n, dtype=np.uint64) + np.uint64(42)

    if algo == "bsm":
        return qk.black_scholes_merton_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot)
    if algo == "crr":
        return qk.crr_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                  np.full(n, 200, dtype=np.int32), zeros_i)
    if algo == "jarrow_rudd":
        return qk.jarrow_rudd_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                          np.full(n, 400, dtype=np.int32), zeros_i)
    if algo == "tian":
        return qk.tian_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                   np.full(n, 400, dtype=np.int32), zeros_i)
    if algo == "leisen_reimer":
        return qk.leisen_reimer_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                            np.full(n, 401, dtype=np.int32), zeros_i)
    if algo == "trinomial":
        return qk.trinomial_tree_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                             np.full(n, 120, dtype=np.int32), zeros_i)
    if algo == "explicit_fd":
        return qk.explicit_fd_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                          np.full(n, 140, dtype=np.int32), np.full(n, 140, dtype=np.int32), zeros_i)
    if algo == "implicit_fd":
        return qk.implicit_fd_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                          np.full(n, 140, dtype=np.int32), np.full(n, 140, dtype=np.int32), zeros_i)
    if algo == "crank_nicolson":
        return qk.crank_nicolson_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                             np.full(n, 140, dtype=np.int32), np.full(n, 140, dtype=np.int32), zeros_i)
    if algo == "psor":
        return qk.psor_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                   np.full(n, 140, dtype=np.int32), np.full(n, 140, dtype=np.int32),
                                   np.full(n, 1.2), np.full(n, 1e-8), np.full(n, 10000, dtype=np.int32))
    if algo == "carr_madan_fft":
        return qk.carr_madan_fft_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                             np.full(n, 4096, dtype=np.int32), np.full(n, 0.25), np.full(n, 1.5))
    if algo == "cos":
        return qk.cos_method_fang_oosterlee_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                        np.full(n, 256, dtype=np.int32), np.full(n, 10.0))
    if algo == "fractional_fft":
        return qk.fractional_fft_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                             np.full(n, 2048, dtype=np.int32), np.full(n, 0.25),
                                             np.full(n, 0.01), np.full(n, 1.5))
    if algo == "lewis":
        return qk.lewis_fourier_inversion_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                      np.full(n, 4096, dtype=np.int32), np.full(n, 300.0))
    if algo == "hilbert":
        return qk.hilbert_transform_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                np.full(n, 4096, dtype=np.int32), np.full(n, 300.0))
    if algo == "gauss_hermite":
        return qk.gauss_hermite_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                            np.full(n, 128, dtype=np.int32))
    if algo == "gauss_laguerre":
        return qk.gauss_laguerre_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                             np.full(n, 64, dtype=np.int32))
    if algo == "gauss_legendre":
        return qk.gauss_legendre_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                             np.full(n, 128, dtype=np.int32), np.full(n, 200.0))
    if algo == "adaptive_quadrature":
        return qk.adaptive_quadrature_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                  np.full(n, 1e-9), np.full(n, 1e-8),
                                                  np.full(n, 14, dtype=np.int32), np.full(n, 200.0))
    if algo == "standard_monte_carlo":
        return qk.standard_monte_carlo_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                   np.full(n, 12000, dtype=np.int32), seed)
    if algo == "euler_maruyama":
        return qk.euler_maruyama_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                             np.full(n, 6000, dtype=np.int32), np.full(n, 48, dtype=np.int32), seed)
    if algo == "milstein":
        return qk.milstein_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                       np.full(n, 6000, dtype=np.int32), np.full(n, 48, dtype=np.int32), seed)
    if algo == "longstaff_schwartz":
        return qk.longstaff_schwartz_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                 np.full(n, 6000, dtype=np.int32), np.full(n, 50, dtype=np.int32), seed)
    if algo == "quasi_monte_carlo_sobol":
        return qk.quasi_monte_carlo_sobol_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                      np.full(n, 8192, dtype=np.int32))
    if algo == "quasi_monte_carlo_halton":
        return qk.quasi_monte_carlo_halton_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                       np.full(n, 8192, dtype=np.int32))
    if algo == "multilevel_monte_carlo":
        return qk.multilevel_monte_carlo_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                     np.full(n, 2048, dtype=np.int32), np.full(n, 3, dtype=np.int32),
                                                     np.full(n, 16, dtype=np.int32), seed)
    if algo == "importance_sampling":
        return qk.importance_sampling_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                  np.full(n, 8000, dtype=np.int32), np.full(n, 0.4), seed)
    if algo == "control_variates":
        return qk.control_variates_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                               np.full(n, 8000, dtype=np.int32), seed)
    if algo == "antithetic_variates":
        return qk.antithetic_variates_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                  np.full(n, 8000, dtype=np.int32), seed)
    if algo == "stratified_sampling":
        return qk.stratified_sampling_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                  np.full(n, 8000, dtype=np.int32), seed)
    if algo == "polynomial_chaos_expansion":
        return qk.polynomial_chaos_expansion_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                         np.full(n, 12, dtype=np.int32), np.full(n, 128, dtype=np.int32))
    if algo == "radial_basis_function":
        return qk.radial_basis_function_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                    np.full(n, 96, dtype=np.int32), np.full(n, 2.0), np.full(n, 1e-6))
    if algo == "sparse_grid_collocation":
        return qk.sparse_grid_collocation_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                      np.full(n, 5, dtype=np.int32), np.full(n, 33, dtype=np.int32))
    if algo == "proper_orthogonal_decomposition":
        return qk.proper_orthogonal_decomposition_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                              np.full(n, 8, dtype=np.int32), np.full(n, 64, dtype=np.int32))
    if algo == "deep_bsde":
        return qk.deep_bsde_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                        np.full(n, 10, dtype=np.int32), np.full(n, 16, dtype=np.int32),
                                        np.full(n, 20, dtype=np.int32), np.full(n, 5e-3))
    if algo == "pinns":
        return qk.pinns_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                    np.full(n, 300, dtype=np.int32), np.full(n, 40, dtype=np.int32),
                                    np.full(n, 20, dtype=np.int32), np.full(n, 1.0))
    if algo == "deep_hedging":
        return qk.deep_hedging_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                           np.full(n, 8, dtype=np.int32), np.full(n, 0.5),
                                           np.full(n, 256, dtype=np.int32), seed)
    if algo == "neural_sde_calibration":
        return qk.neural_sde_calibration_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                     np.ascontiguousarray(batch.vol), np.full(n, 50, dtype=np.int32), np.full(n, 1e-3))
    if algo == "pathwise_derivative_delta":
        return qk.pathwise_derivative_delta_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                  np.full(n, 20000, dtype=np.int32), seed)
    if algo == "likelihood_ratio_delta":
        return qk.likelihood_ratio_delta_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                               np.full(n, 20000, dtype=np.int32), seed, np.full(n, 40.0))
    if algo == "aad_delta":
        return qk.aad_delta_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                  np.full(n, 96, dtype=np.int32), np.full(n, 1e-6))
    raise ValueError(f"Unknown algorithm: {algo}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=ROOT / "third_party_source" / "option_price_sample.xlsx")
    parser.add_argument("--sheet", type=str, default=None)
    parser.add_argument("--dte-basis", type=float, default=365.0)
    parser.add_argument("--carry-mode", type=str, choices=["flat", "pcp"], default="flat")
    parser.add_argument("--risk-free-rate", type=float, default=0.05)
    parser.add_argument("--dividend-yield", type=float, default=0.015)
    parser.add_argument("--pcp-q-abs-max", type=float, default=5.0)
    parser.add_argument("--sample-per-side", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(spec.name for spec in ALL_ALGOS),
        help="Comma-separated list. Available: " + ",".join(spec.name for spec in ALL_ALGOS),
    )
    parser.add_argument("--quantlib-source-root", type=Path, default=ROOT / "third_party_source" / "quantlib_copy" / "QuantLib")
    parser.add_argument("--quantlib-build-dir", type=Path, default=ROOT / "third_party_source" / "quantlib_copy" / "build")
    parser.add_argument("--quantlib-wrapper-src", type=Path, default=ROOT / "examples" / "quantlib_bsm_ref.cpp")
    parser.add_argument("--quantlib-wrapper-lib", type=Path, default=ROOT / "third_party_source" / "quantlib_copy" / "build" / "libql_ref_bsm.so")
    parser.add_argument("--delta-fd-rel-bump", type=float, default=1e-4)
    parser.add_argument("--delta-fd-abs-bump", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()

    batch_mod = _load_batch_module()

    call_batch, put_batch, ingest_stats = batch_mod._collect_batches(
        xlsx_path=args.xlsx,
        sheet_name=args.sheet,
        dte_basis=args.dte_basis,
        carry_mode=args.carry_mode,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
        pcp_q_abs_max=args.pcp_q_abs_max,
        sample_per_side=args.sample_per_side,
        seed=args.seed,
    )

    batch_mod._compile_quantlib_wrapper(
        wrapper_src=args.quantlib_wrapper_src,
        wrapper_lib=args.quantlib_wrapper_lib,
        quantlib_source_root=args.quantlib_source_root,
        quantlib_build_dir=args.quantlib_build_dir,
    )
    ql_ref = batch_mod.QuantLibBSMRef(args.quantlib_wrapper_lib)
    qk = QuantKernel()

    algo_map = {spec.name: spec for spec in ALL_ALGOS}
    requested = [x.strip() for x in args.algorithms.split(",") if x.strip()]

    report = {
        "config": {
            "xlsx": str(args.xlsx),
            "sheet": args.sheet if args.sheet else "first_sheet",
            "dte_basis": float(args.dte_basis),
            "carry_mode": args.carry_mode,
            "risk_free_rate": float(args.risk_free_rate),
            "dividend_yield": float(args.dividend_yield),
            "pcp_q_abs_max": float(args.pcp_q_abs_max),
            "sample_per_side": int(args.sample_per_side),
            "seed": int(args.seed),
            "algorithms": requested,
            "delta_fd_rel_bump": float(args.delta_fd_rel_bump),
            "delta_fd_abs_bump": float(args.delta_fd_abs_bump),
        },
        "ingest": ingest_stats,
        "results": {},
    }

    for algo in requested:
        if algo not in algo_map:
            report["results"][algo] = {"error": f"Unknown algorithm '{algo}'"}
            continue

        algo_out: dict[str, object] = {"notes": algo_map[algo].notes}
        spec = algo_map[algo]
        for side_name, batch in (("call", call_batch), ("put", put_batch)):
            if batch.size == 0:
                algo_out[side_name] = {"n": 0}
                continue
            eval_batch = _slice_batch(batch, spec.max_per_side)
            if eval_batch.size == 0:
                algo_out[side_name] = {"n": 0}
                continue

            ref = ql_ref.price_batch(
                eval_batch.spot, eval_batch.strike, eval_batch.tau, eval_batch.vol,
                eval_batch.r, eval_batch.q, eval_batch.option_type,
            )
            if spec.kind == "delta":
                ref = _quantlib_delta_fd(
                    ql_ref, eval_batch, rel_bump=args.delta_fd_rel_bump, abs_bump=args.delta_fd_abs_bump
                )
            ts = time.perf_counter()
            try:
                model = _invoke_batch(qk, algo, eval_batch)
            except Exception as exc:  # pragma: no cover - diagnostic path
                algo_out[side_name] = {"n": eval_batch.size, "source_n": batch.size, "error": str(exc)}
                continue
            te = time.perf_counter()
            side_out = {
                "n": eval_batch.size,
                "source_n": batch.size,
                "runtime_seconds": float(te - ts),
                "reference_kind": spec.kind,
                "quantkernel_vs_quantlib": _pair_metrics(model, ref),
            }
            if spec.kind == "price":
                side_out["quantkernel_vs_market"] = _market_metrics(model, eval_batch.bid, eval_batch.ask)
            algo_out[side_name] = side_out
        report["results"][algo] = algo_out

    report["total_runtime_seconds"] = float(time.perf_counter() - t0)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
