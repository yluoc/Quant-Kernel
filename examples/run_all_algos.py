#!/usr/bin/env python3
"""Benchmark QuantKernel runtime with and without GPU acceleration.

This is a Python-side benchmark harness (no CUDA/C++ changes required).
It compares:
- CPU path: ``qk.price_batch(..., backend='cpu')``
- GPU path: ``qk.price_batch(..., backend='gpu')`` (uses CuPy when available)

Run:
    PYTHONPATH=python python3 examples/run_all_algos.py

Optional:
    PYTHONPATH=python python3 examples/run_all_algos.py --repeats 5 --profile full
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np

from quantkernel import QK_CALL, QK_PUT, QuantKernel

GPU_VECTOR_METHODS = {
    "black_scholes_merton_price",
    "black76_price",
    "bachelier_price",
    "sabr_hagan_lognormal_iv",
    "sabr_hagan_black76_price",
    "dupire_local_vol",
    "carr_madan_fft_price",
    "cos_method_fang_oosterlee_price",
    "fractional_fft_price",
    "lewis_fourier_inversion_price",
    "hilbert_transform_price",
}


@dataclass(frozen=True)
class AlgoCase:
    method: str
    base_job: Mapping[str, float | int | bool]
    batch_quick: int
    batch_full: int


def _perturb(base: Mapping[str, float | int | bool], i: int) -> Dict[str, float | int | bool]:
    """Create a slightly varied job to avoid identical batches."""
    out: Dict[str, float | int | bool] = dict(base)
    tweak = (i % 11) - 5

    if "spot" in out:
        out["spot"] = float(out["spot"]) * (1.0 + 0.001 * tweak)
    if "strike" in out:
        out["strike"] = float(out["strike"]) * (1.0 + 0.0007 * tweak)
    if "forward" in out:
        out["forward"] = float(out["forward"]) * (1.0 + 0.001 * tweak)
    if "t" in out:
        out["t"] = max(1e-4, float(out["t"]) * (1.0 + 0.0005 * tweak))

    return out


def build_jobs(base: Mapping[str, float | int | bool], n: int) -> List[Dict[str, float | int | bool]]:
    return [_perturb(base, i) for i in range(n)]


def bench_once(fn: Callable[[], np.ndarray]) -> float:
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def bench_backend(
    qk: QuantKernel,
    method: str,
    jobs: Sequence[Mapping[str, float | int | bool]],
    backend: str,
    repeats: int,
):
    acc = qk.get_accelerator(backend=backend)
    strategy = acc.suggest_strategy(method, len(jobs))

    out0 = qk.price_batch(method, jobs, backend=backend)
    if np.any(~np.isfinite(out0)):
        raise RuntimeError(f"{method} on backend={backend} returned non-finite values")

    samples = [bench_once(lambda: qk.price_batch(method, jobs, backend=backend)) for _ in range(repeats)]
    ms = statistics.median(samples)
    return ms, strategy, np.asarray(out0, dtype=np.float64)


def algo_cases() -> List[AlgoCase]:
    spot = 100.0
    strike = 105.0
    t = 1.0
    vol = 0.2
    r = 0.05
    q = 0.02
    fwd = spot * math.exp((r - q) * t)

    return [
        AlgoCase("black_scholes_merton_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL}, 60_000, 200_000),
        AlgoCase("black76_price", {"forward": fwd, "strike": strike, "t": t, "vol": vol, "r": r, "option_type": QK_CALL}, 60_000, 200_000),
        AlgoCase("bachelier_price", {"forward": fwd, "strike": strike, "t": t, "normal_vol": spot * vol, "r": r, "option_type": QK_CALL}, 60_000, 200_000),
        AlgoCase("heston_price_cf", {"spot": spot, "strike": strike, "t": t, "r": r, "q": q, "v0": vol * vol, "kappa": 2.0, "theta": vol * vol, "sigma": 0.3, "rho": -0.7, "option_type": QK_CALL, "integration_steps": 768, "integration_limit": 120.0}, 256, 1024),
        AlgoCase("merton_jump_diffusion_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "jump_intensity": 0.1, "jump_mean": -0.05, "jump_vol": 0.10, "max_terms": 50, "option_type": QK_CALL}, 3_000, 15_000),
        AlgoCase("variance_gamma_price_cf", {"spot": spot, "strike": strike, "t": t, "r": r, "q": q, "sigma": 0.20, "theta": -0.10, "nu": 0.20, "option_type": QK_CALL, "integration_steps": 768, "integration_limit": 120.0}, 256, 1024),
        AlgoCase("sabr_hagan_lognormal_iv", {"forward": fwd, "strike": strike, "t": t, "alpha": 0.20, "beta": 0.50, "rho": -0.25, "nu": 0.40}, 60_000, 200_000),
        AlgoCase("sabr_hagan_black76_price", {"forward": fwd, "strike": strike, "t": t, "r": r, "alpha": 0.20, "beta": 0.50, "rho": -0.25, "nu": 0.40, "option_type": QK_CALL}, 60_000, 200_000),
        AlgoCase("dupire_local_vol", {"strike": strike, "t": t, "call_price": 8.0, "dC_dT": 2.0, "dC_dK": -0.5, "d2C_dK2": 0.03, "r": r, "q": q}, 80_000, 250_000),
        AlgoCase("carr_madan_fft_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "grid_size": 4096, "eta": 0.25, "alpha": 1.5}, 512, 2_048),
        AlgoCase("cos_method_fang_oosterlee_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "n_terms": 256, "truncation_width": 10.0}, 1_024, 4_096),
        AlgoCase("fractional_fft_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "grid_size": 256, "eta": 0.25, "lambda_": 0.05, "alpha": 1.5}, 512, 2_048),
        AlgoCase("lewis_fourier_inversion_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "integration_steps": 4096, "integration_limit": 300.0}, 1_024, 4_096),
        AlgoCase("hilbert_transform_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "integration_steps": 4096, "integration_limit": 300.0}, 1_024, 4_096),
        AlgoCase("gauss_hermite_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "n_points": 128}, 512, 2_048),
        AlgoCase("gauss_laguerre_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "n_points": 64}, 512, 2_048),
        AlgoCase("gauss_legendre_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "n_points": 128, "integration_limit": 200.0}, 512, 2_048),
        AlgoCase("adaptive_quadrature_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "abs_tol": 1e-9, "rel_tol": 1e-8, "max_depth": 14, "integration_limit": 200.0}, 512, 2_048),
        AlgoCase("crr_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "steps": 180, "american_style": False}, 256, 1_000),
        AlgoCase("jarrow_rudd_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "steps": 180, "american_style": False}, 256, 1_000),
        AlgoCase("tian_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "steps": 180, "american_style": False}, 256, 1_000),
        AlgoCase("leisen_reimer_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "steps": 180, "american_style": False}, 256, 1_000),
        AlgoCase("trinomial_tree_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "steps": 120, "american_style": False}, 256, 1_000),
        AlgoCase("derman_kani_const_local_vol_price", {"spot": spot, "strike": strike, "t": t, "local_vol": vol, "r": r, "q": q, "option_type": QK_CALL, "steps": 120, "american_style": False}, 256, 1_000),
        AlgoCase("explicit_fd_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "time_steps": 180, "spot_steps": 180, "american_style": False}, 64, 256),
        AlgoCase("implicit_fd_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "time_steps": 180, "spot_steps": 180, "american_style": False}, 64, 256),
        AlgoCase("crank_nicolson_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "time_steps": 180, "spot_steps": 180, "american_style": False}, 64, 256),
        AlgoCase("adi_douglas_price", {"spot": spot, "strike": strike, "t": t, "r": r, "q": q, "v0": vol * vol, "kappa": 2.0, "theta_v": vol * vol, "sigma": 0.3, "rho": -0.7, "option_type": QK_CALL, "s_steps": 40, "v_steps": 20, "time_steps": 40}, 64, 256),
        AlgoCase("adi_craig_sneyd_price", {"spot": spot, "strike": strike, "t": t, "r": r, "q": q, "v0": vol * vol, "kappa": 2.0, "theta_v": vol * vol, "sigma": 0.3, "rho": -0.7, "option_type": QK_CALL, "s_steps": 40, "v_steps": 20, "time_steps": 40}, 64, 256),
        AlgoCase("adi_hundsdorfer_verwer_price", {"spot": spot, "strike": strike, "t": t, "r": r, "q": q, "v0": vol * vol, "kappa": 2.0, "theta_v": vol * vol, "sigma": 0.3, "rho": -0.7, "option_type": QK_CALL, "s_steps": 40, "v_steps": 20, "time_steps": 40}, 64, 256),
        AlgoCase("psor_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "time_steps": 140, "spot_steps": 140, "omega": 1.2, "tol": 1e-8, "max_iter": 8_000}, 64, 256),
        AlgoCase("standard_monte_carlo_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "paths": 25_000, "seed": 42}, 128, 512),
        AlgoCase("euler_maruyama_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "paths": 15_000, "steps": 64, "seed": 42}, 64, 256),
        AlgoCase("milstein_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "paths": 15_000, "steps": 64, "seed": 42}, 64, 256),
        AlgoCase("longstaff_schwartz_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_PUT, "paths": 8_000, "steps": 50, "seed": 42}, 32, 128),
        AlgoCase("quasi_monte_carlo_sobol_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "paths": 16_384}, 128, 512),
        AlgoCase("quasi_monte_carlo_halton_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "paths": 16_384}, 128, 512),
        AlgoCase("multilevel_monte_carlo_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "base_paths": 8_192, "levels": 4, "base_steps": 8, "seed": 42}, 32, 128),
        AlgoCase("importance_sampling_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "paths": 20_000, "shift": 0.5, "seed": 42}, 128, 512),
        AlgoCase("control_variates_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "paths": 20_000, "seed": 42}, 128, 512),
        AlgoCase("antithetic_variates_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "paths": 20_000, "seed": 42}, 128, 512),
        AlgoCase("stratified_sampling_price", {"spot": spot, "strike": strike, "t": t, "vol": vol, "r": r, "q": q, "option_type": QK_CALL, "paths": 20_000, "seed": 42}, 128, 512),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantKernel CPU vs GPU runtime simulation")
    p.add_argument("--repeats", type=int, default=3, help="timing repeats per backend (median reported)")
    p.add_argument("--profile", choices=["quick", "full"], default="quick", help="batch-size profile")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    qk = QuantKernel()

    auto_acc = qk.get_accelerator("auto")
    gpu_ready = auto_acc.gpu_available

    print("=" * 108)
    print(" QuantKernel Runtime Simulation (CPU vs GPU)")
    print("=" * 108)
    print(f"GPU available (CuPy): {gpu_ready}")
    print(f"Profile: {args.profile} | Repeats: {args.repeats}")
    print()
    print(f"{'Method':<34} {'Batch':>8} {'CPU(ms)':>11} {'GPU(ms)':>11} {'Speedup':>10} {'CPU strategy':>14} {'GPU strategy':>14}")
    print("-" * 108)

    rows = []
    for case in algo_cases():
        batch = case.batch_full if args.profile == "full" else case.batch_quick
        jobs = build_jobs(case.base_job, batch)

        cpu_ms, cpu_strategy, cpu_out = bench_backend(qk, case.method, jobs, "cpu", args.repeats)

        gpu_ms = None
        gpu_strategy = "cpu_only"
        speedup = "N/A"

        if case.method in GPU_VECTOR_METHODS:
            if gpu_ready:
                try:
                    gpu_ms, gpu_strategy, gpu_out = bench_backend(qk, case.method, jobs, "gpu", args.repeats)
                    if gpu_ms > 0.0:
                        speedup = f"{(cpu_ms / gpu_ms):.2f}x"

                    if not np.allclose(cpu_out, gpu_out, rtol=1e-6, atol=1e-6):
                        gpu_strategy = f"{gpu_strategy}*"
                except Exception:
                    gpu_strategy = "gpu_error"
            else:
                gpu_strategy = "gpu_unavail"

        rows.append((case.method, batch, cpu_ms, gpu_ms, speedup, cpu_strategy, gpu_strategy))

    for method, batch, cpu_ms, gpu_ms, speedup, cpu_strategy, gpu_strategy in rows:
        gpu_str = f"{gpu_ms:,.3f}" if gpu_ms is not None else "N/A"
        print(f"{method:<34} {batch:>8d} {cpu_ms:>11.3f} {gpu_str:>11} {speedup:>10} {cpu_strategy:>14} {gpu_strategy:>14}")

    print("-" * 108)
    print("Legend: strategy with '*' means CPU/GPU outputs differed beyond tolerance (1e-6).")
    print("If CuPy is unavailable, GPU column may show N/A for GPU-vectorized methods.")
    print("=" * 108)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
