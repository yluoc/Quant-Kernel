#!/usr/bin/env python3
"""Historical batch validation suite for American-style option algorithms."""

from __future__ import annotations

import argparse
import ctypes as ct
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
    notes: str = ""
    max_per_side: int | None = None


ALL_ALGOS = [
    AlgoSpec("crr"),
    AlgoSpec("jarrow_rudd"),
    AlgoSpec("tian"),
    AlgoSpec("leisen_reimer"),
    AlgoSpec("trinomial"),
    AlgoSpec("explicit_fd"),
    AlgoSpec("implicit_fd"),
    AlgoSpec("crank_nicolson"),
    AlgoSpec("psor", notes="PSOR is American-style by construction."),
    AlgoSpec("longstaff_schwartz", notes="LSMC Monte Carlo estimator (American-style).", max_per_side=120),
]


class QuantLibAmericanRef:
    def __init__(self, lib_path: Path, t_grid: int, x_grid: int):
        self._t_grid = int(t_grid)
        self._x_grid = int(x_grid)
        self._lib = ct.CDLL(str(lib_path))
        self._fn = self._lib.ql_ref_american_price_batch
        dptr = ct.POINTER(ct.c_double)
        i32ptr = ct.POINTER(ct.c_int32)
        self._fn.argtypes = [
            dptr, dptr, dptr, dptr, dptr, dptr, i32ptr,
            ct.c_int32, ct.c_int32, ct.c_int32,
            dptr, ct.POINTER(ct.c_char), ct.c_int32,
        ]
        self._fn.restype = ct.c_int32

    def price_batch(
        self,
        spot: np.ndarray,
        strike: np.ndarray,
        tau: np.ndarray,
        vol: np.ndarray,
        r: np.ndarray,
        q: np.ndarray,
        option_type: np.ndarray,
    ) -> np.ndarray:
        n = int(spot.shape[0])
        out = np.empty(n, dtype=np.float64)
        err = ct.create_string_buffer(1024)
        rc = self._fn(
            spot.ctypes.data_as(ct.POINTER(ct.c_double)),
            strike.ctypes.data_as(ct.POINTER(ct.c_double)),
            tau.ctypes.data_as(ct.POINTER(ct.c_double)),
            vol.ctypes.data_as(ct.POINTER(ct.c_double)),
            r.ctypes.data_as(ct.POINTER(ct.c_double)),
            q.ctypes.data_as(ct.POINTER(ct.c_double)),
            option_type.ctypes.data_as(ct.POINTER(ct.c_int32)),
            n,
            self._t_grid,
            self._x_grid,
            out.ctypes.data_as(ct.POINTER(ct.c_double)),
            err,
            len(err),
        )
        if rc != 0:
            msg = err.value.decode("utf-8", errors="replace").strip() or f"rc={rc}"
            raise RuntimeError(f"QuantLib American wrapper failed: {msg}")
        return out


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


def _invoke_batch(qk: QuantKernel, algo: str, batch) -> np.ndarray:
    n = batch.size
    ot = batch.option_type
    am = np.ones(n, dtype=np.int32)
    seed = np.arange(n, dtype=np.uint64) + np.uint64(42)

    if algo == "crr":
        return qk.crr_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                  np.full(n, 300, dtype=np.int32), am)
    if algo == "jarrow_rudd":
        return qk.jarrow_rudd_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                          np.full(n, 400, dtype=np.int32), am)
    if algo == "tian":
        return qk.tian_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                   np.full(n, 400, dtype=np.int32), am)
    if algo == "leisen_reimer":
        return qk.leisen_reimer_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                            np.full(n, 401, dtype=np.int32), am)
    if algo == "trinomial":
        return qk.trinomial_tree_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                             np.full(n, 180, dtype=np.int32), am)
    if algo == "explicit_fd":
        return qk.explicit_fd_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                          np.full(n, 180, dtype=np.int32), np.full(n, 180, dtype=np.int32), am)
    if algo == "implicit_fd":
        return qk.implicit_fd_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                          np.full(n, 180, dtype=np.int32), np.full(n, 180, dtype=np.int32), am)
    if algo == "crank_nicolson":
        return qk.crank_nicolson_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                             np.full(n, 180, dtype=np.int32), np.full(n, 180, dtype=np.int32), am)
    if algo == "psor":
        return qk.psor_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                   np.full(n, 180, dtype=np.int32), np.full(n, 180, dtype=np.int32),
                                   np.full(n, 1.2), np.full(n, 1e-8), np.full(n, 10000, dtype=np.int32))
    if algo == "longstaff_schwartz":
        return qk.longstaff_schwartz_price_batch(batch.spot, batch.strike, batch.tau, batch.vol, batch.r, batch.q, ot,
                                                 np.full(n, 15000, dtype=np.int32), np.full(n, 50, dtype=np.int32), seed)
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
    parser.add_argument("--quantlib-wrapper-src", type=Path, default=ROOT / "examples" / "quantlib_american_ref.cpp")
    parser.add_argument("--quantlib-wrapper-lib", type=Path, default=ROOT / "third_party_source" / "quantlib_copy" / "build" / "libql_ref_american.so")
    parser.add_argument("--quantlib-t-grid", type=int, default=300)
    parser.add_argument("--quantlib-x-grid", type=int, default=200)
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

    qk = QuantKernel()
    ql_ref = QuantLibAmericanRef(args.quantlib_wrapper_lib, args.quantlib_t_grid, args.quantlib_x_grid)

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
            "quantlib_t_grid": int(args.quantlib_t_grid),
            "quantlib_x_grid": int(args.quantlib_x_grid),
            "reference": "quantlib_american_fd",
        },
        "ingest": ingest_stats,
        "results": {},
    }

    ref_cache: dict[tuple[str, int], np.ndarray] = {}

    for algo in requested:
        if algo not in algo_map:
            report["results"][algo] = {"error": f"Unknown algorithm '{algo}'"}
            continue

        spec = algo_map[algo]
        algo_out: dict[str, object] = {"notes": spec.notes}
        for side_name, batch in (("call", call_batch), ("put", put_batch)):
            if batch.size == 0:
                algo_out[side_name] = {"n": 0}
                continue
            eval_batch = _slice_batch(batch, spec.max_per_side)
            if eval_batch.size == 0:
                algo_out[side_name] = {"n": 0}
                continue

            cache_key = (side_name, int(eval_batch.size))
            if cache_key not in ref_cache:
                ref_cache[cache_key] = ql_ref.price_batch(
                    eval_batch.spot, eval_batch.strike, eval_batch.tau, eval_batch.vol,
                    eval_batch.r, eval_batch.q, eval_batch.option_type,
                )
            ref = ref_cache[cache_key]

            ts = time.perf_counter()
            try:
                model = _invoke_batch(qk, algo, eval_batch)
            except Exception as exc:  # pragma: no cover - diagnostic path
                algo_out[side_name] = {"n": eval_batch.size, "source_n": batch.size, "error": str(exc)}
                continue
            te = time.perf_counter()

            algo_out[side_name] = {
                "n": eval_batch.size,
                "source_n": batch.size,
                "runtime_seconds": float(te - ts),
                "quantkernel_vs_quantlib_american": _pair_metrics(model, ref),
                "quantkernel_vs_market": _market_metrics(model, eval_batch.bid, eval_batch.ask),
            }
        report["results"][algo] = algo_out

    report["total_runtime_seconds"] = float(time.perf_counter() - t0)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

