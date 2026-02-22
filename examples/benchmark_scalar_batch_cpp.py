#!/usr/bin/env python3
"""Benchmark scalar vs batch paths for Black-Scholes-Merton pricing."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from quantkernel import QK_CALL, QK_PUT, QuantKernel


def _make_inputs(n: int):
    rng = np.random.default_rng(42)
    spot = rng.uniform(80.0, 120.0, n).astype(np.float64)
    strike = rng.uniform(80.0, 120.0, n).astype(np.float64)
    tau = rng.uniform(0.25, 2.0, n).astype(np.float64)
    vol = rng.uniform(0.1, 0.6, n).astype(np.float64)
    r = rng.uniform(0.0, 0.08, n).astype(np.float64)
    q = rng.uniform(0.0, 0.04, n).astype(np.float64)
    option_type = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)
    return spot, strike, tau, vol, r, q, option_type


def _median_ms(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return statistics.median(samples)


def _find_cpp_bench() -> Path | None:
    candidates = [
        Path("build") / "cpp" / "qk_bench_bsm_batch",
        Path("build") / "qk_bench_bsm_batch",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    qk = QuantKernel()
    n = args.n
    repeats = args.repeats
    spot, strike, tau, vol, r, q, option_type = _make_inputs(n)

    jobs = [
        {
            "spot": float(spot[i]),
            "strike": float(strike[i]),
            "t": float(tau[i]),
            "vol": float(vol[i]),
            "r": float(r[i]),
            "q": float(q[i]),
            "option_type": int(option_type[i]),
        }
        for i in range(n)
    ]

    _ = qk.black_scholes_merton_price_batch(spot, strike, tau, vol, r, q, option_type)
    _ = qk.price_batch("black_scholes_merton_price", jobs[:2048], backend="cpu")

    scalar_ms = _median_ms(
        lambda: [qk.black_scholes_merton_price(
            float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
            float(r[i]), float(q[i]), int(option_type[i])
        ) for i in range(n)],
        repeats,
    )

    py_batch_ms = _median_ms(
        lambda: qk.price_batch("black_scholes_merton_price", jobs, backend="cpu"),
        repeats,
    )

    native_batch_ms = _median_ms(
        lambda: qk.black_scholes_merton_price_batch(spot, strike, tau, vol, r, q, option_type),
        repeats,
    )

    cpp_metrics = None
    cpp_bench = _find_cpp_bench()
    if cpp_bench is not None:
        proc = subprocess.run(
            [str(cpp_bench), str(n), str(repeats)],
            check=True,
            capture_output=True,
            text=True,
        )
        cpp_metrics = json.loads(proc.stdout.strip().splitlines()[-1])

    rows = [
        ("Python scalar", scalar_ms),
        ("Python batch (price_batch)", py_batch_ms),
        ("C++ batch API via Python", native_batch_ms),
    ]
    if cpp_metrics is not None:
        rows.append(("C++ direct executable (scalar)", float(cpp_metrics["cpp_scalar_ms"])))
        rows.append(("C++ direct executable (batch)", float(cpp_metrics["cpp_batch_ms"])))

    print(f"\nBenchmark (n={n}, repeats={repeats})")
    print("| Mode | Median ms | Throughput (prices/s) | Speedup vs Python scalar |")
    print("|---|---:|---:|---:|")
    for name, ms in rows:
        throughput = n / (ms / 1000.0)
        speedup = scalar_ms / max(ms, 1e-12)
        print(f"| {name} | {ms:.3f} | {throughput:,.0f} | {speedup:.2f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
