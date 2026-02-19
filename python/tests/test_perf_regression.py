"""Deterministic performance regression checks for batch paths."""

from __future__ import annotations

import statistics
import time

import numpy as np

from quantkernel import QK_CALL, QK_PUT


def _median_ms(fn, repeats: int = 3) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return statistics.median(samples)


def test_bsm_batch_speed_and_accuracy_regression(qk):
    rng = np.random.default_rng(7)
    n = 40000
    spot = rng.uniform(80.0, 120.0, n).astype(np.float64)
    strike = rng.uniform(80.0, 120.0, n).astype(np.float64)
    tau = rng.uniform(0.25, 2.0, n).astype(np.float64)
    vol = rng.uniform(0.1, 0.6, n).astype(np.float64)
    r = rng.uniform(0.0, 0.08, n).astype(np.float64)
    q = rng.uniform(0.0, 0.04, n).astype(np.float64)
    option_type = np.where((np.arange(n) & 1) == 0, QK_CALL, QK_PUT).astype(np.int32)

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

    # Warmup JIT/caches
    qk.black_scholes_merton_price_batch(spot, strike, tau, vol, r, q, option_type)
    qk.price_batch("black_scholes_merton_price", jobs[:2000], backend="cpu")

    scalar_ms = _median_ms(
        lambda: [qk.black_scholes_merton_price(
            float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
            float(r[i]), float(q[i]), int(option_type[i])
        ) for i in range(n)],
    )
    py_batch_ms = _median_ms(
        lambda: qk.price_batch("black_scholes_merton_price", jobs, backend="cpu"),
    )
    native_batch_ms = _median_ms(
        lambda: qk.black_scholes_merton_price_batch(spot, strike, tau, vol, r, q, option_type),
    )

    bsm_batch = qk.black_scholes_merton_price_batch(spot, strike, tau, vol, r, q, option_type)
    bsm_scalar = np.array(
        [qk.black_scholes_merton_price(float(spot[i]), float(strike[i]), float(tau[i]), float(vol[i]),
                                       float(r[i]), float(q[i]), int(option_type[i])) for i in range(n)],
        dtype=np.float64,
    )
    assert np.allclose(bsm_batch, bsm_scalar, atol=1e-12, rtol=1e-12)

    # Deterministic perf guardrails: batch modes should stay faster than scalar.
    assert py_batch_ms < scalar_ms * 0.95
    assert native_batch_ms < scalar_ms * 0.90
